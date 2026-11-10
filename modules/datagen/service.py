"""
modules/datagen/service.py
──────────────────────────
Resumable driver around ``simulation.TradingAgent``.

Why this exists
---------------
The original ``flywheel.run_simulation_round`` ran N episodes in one shot with
no persistence of *where it stopped*. If the process died at episode 37/50 you
had to rerun all 50 and regenerate identical data.

This driver makes generation **resumable**:

* Raw episodes already persist to SQLite (``trading_agent.db``) — the heavy data
  is never thrown away.
* After every episode we checkpoint ``{symbol, start_date, episodes_done,
  date, capital}`` into ``output/state/datagen.json`` (atomic write).
* On resume we reconstruct a ``TradingAgent`` at the saved trading day + capital
  and only run the *remaining* episodes, so no episode is ever generated twice.

The heavy ``simulation`` module is imported lazily inside the worker so the
FastAPI process (and the test-suite) can import this file without pandas /
yfinance / a network connection.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from common.paths import DB_PATH, ensure_dirs
from common.state import StateStore

STAGE = "datagen"


# ── read-only helpers (safe to call from the API without heavy deps) ─────────
def episodes_in_db(symbol: str, db_path: Optional[str] = None) -> int:
    """How many episodes for ``symbol`` already exist in the trade log."""
    p = str(db_path or DB_PATH)
    try:
        with sqlite3.connect(p) as con:
            row = con.execute(
                "SELECT COUNT(DISTINCT episode) FROM trade_log WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def recent_trades(limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    """Latest trade-log rows — powers the 'view generated data' panel."""
    p = str(db_path or DB_PATH)
    try:
        with sqlite3.connect(p) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT episode, symbol, decision_date, action, entry_price, "
                "quantity, exit_price, pnl, exit_reason, pre_trade_reasoning, "
                "lesson_learned FROM trade_log "
                "ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


# ── the resumable worker ─────────────────────────────────────────────────────
def run_datagen(
    symbol: str = "CANBK",
    start_date: str = "2024-01-10",
    episodes: int = 50,
    backend: str = "fallback",
    adapter_path: Optional[str] = None,
    resume: bool = True,
    db_path: Optional[str] = None,
    state_stage: Optional[str] = None,
) -> dict:
    """Generate ``episodes`` episodes, resuming from the last checkpoint.

    ``state_stage`` lets a parallel worker write to its own per-symbol state file
    (e.g. ``datagen:CANBK``); it defaults to the shared ``datagen`` stage.
    """
    ensure_dirs()
    store = StateStore(state_stage or STAGE)
    db = str(db_path or DB_PATH)

    # Lazy heavy import — keeps API/test import light.
    import simulation
    from simulation import TradingAgent, init_db, set_llm_backend, next_trading_day

    simulation.DB_PATH = db
    init_db()

    prev = store.load()
    ck = prev.checkpoint if resume else {}
    same_job = (
        resume
        and ck.get("symbol") == symbol
        and ck.get("start_date") == start_date
    )

    already = int(ck.get("episodes_done", 0)) if same_job else 0
    resumed = same_job and already > 0

    # Nothing to do — target already met (no regeneration).
    if already >= episodes:
        store.finish(message=f"{symbol}: {already} episodes already generated (target {episodes})")
        return {"episodes": already, "resumed": True, "generated": 0}

    store.start(total=episodes, message=f"Generating {symbol} episodes "
                f"({already} already done)" if resumed else f"Generating {symbol}")
    store.checkpoint(symbol=symbol, start_date=start_date)

    # Choose the decision backend (student adapter / Ollama teacher / rules).
    if backend == "adapter" and adapter_path:
        set_llm_backend("adapter", adapter_path)
    elif backend == "ollama":
        set_llm_backend("ollama")
    else:
        set_llm_backend("fallback")

    # Reconstruct the agent, fast-forwarding to the resume point if needed.
    capital = float(ck.get("capital", 10_000)) if resumed else 10_000.0
    agent = TradingAgent(symbol=symbol, start_date=start_date, capital=capital)
    if resumed and ck.get("date"):
        agent.date = ck["date"]
        agent.episode = already

    generated = 0
    for _ in range(already, episodes):
        if not agent.alive():
            store.update(message=f"Agent died at episode {agent.episode} "
                         f"(capital {agent.capital:,.0f})")
            break
        try:
            agent.run_episode()
            generated += 1
        except Exception as exc:  # skip a bad trading day, keep going
            agent.date = next_trading_day(agent.date, 1)
            store.log({"event": "episode_error", "error": str(exc)})
            continue

        # Checkpoint AFTER each episode so a crash loses at most one episode.
        store.update(current=agent.episode,
                     message=f"{symbol}: episode {agent.episode}/{episodes}")
        store.checkpoint(episodes_done=agent.episode, date=agent.date,
                         capital=agent.capital)

    wins = sum(1 for t in agent.history if t.get("pnl", 0) > 0)
    total_pnl = sum(t.get("pnl", 0) for t in agent.history)
    summary = {
        "symbol": symbol,
        "episodes": agent.episode,
        "generated_this_run": generated,
        "wins": wins,
        "total_pnl": round(total_pnl, 2),
        "final_capital": round(agent.capital, 2),
        "resumed": resumed,
    }
    store.finish(message=f"Done: {agent.episode} episodes "
                 f"(+{generated} this run)", summary=summary)
    return summary


# ════════════════════════════════════════════════════════════════════════════
#  PARALLEL GENERATION (across symbols)
# ════════════════════════════════════════════════════════════════════════════
# A single TradingAgent's episodes form a *sequential* chain — episode N+1 reads
# the capital/date that episode N produced — so they cannot be parallelised
# without changing the semantics. What IS embarrassingly parallel is running
# **one independent agent per symbol** (and/or per date-window). Each worker:
#   * runs in its own process (true parallelism past the GIL for the pandas /
#     zone-detection compute),
#   * writes to its own SQLite *shard* so there is zero write-contention,
#   * keeps its own resumable state file (``datagen:<symbol>``).
# After the workers finish we merge the shards into the main trade_log (a
# per-symbol replace, so it is idempotent and safe to resume).

from pathlib import Path


def _shard_db_path(symbol: str, db_path: Optional[str] = None) -> Path:
    main = Path(db_path or DB_PATH)
    shards = main.parent / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    return shards / f"{main.stem}__{symbol}.db"


def _datagen_worker(symbol: str, start_date: str, episodes: int, backend: str,
                    adapter_path: Optional[str], resume: bool,
                    main_db: str) -> dict:
    """Top-level (picklable) entry point for one symbol in its own process."""
    import common  # re-wire sys.path in the spawned process  # noqa: F401
    from modules.datagen.service import run_datagen as _run
    return _run(symbol=symbol, start_date=start_date, episodes=episodes,
                backend=backend, adapter_path=adapter_path, resume=resume,
                db_path=str(_shard_db_path(symbol, main_db)),
                state_stage=f"{STAGE}:{symbol}")


_TRADE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS fundamental_cache (
    symbol TEXT, as_of_quarter TEXT, data_json TEXT, fetched_at TEXT,
    PRIMARY KEY (symbol, as_of_quarter));
CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode INTEGER, symbol TEXT, decision_date TEXT, action TEXT,
    entry_price REAL, quantity INTEGER, stop_loss REAL, take_profit REAL,
    exit_price REAL, exit_date TEXT, exit_reason TEXT, pnl REAL,
    capital_before REAL, capital_after REAL,
    pre_trade_reasoning TEXT, post_trade_analysis TEXT,
    lesson_learned TEXT, what_went_wrong TEXT,
    technicals_json TEXT, created_at TEXT);
"""


def _merge_shard_into_main(symbol: str, main_db: str) -> int:
    """Replace the main DB's rows for ``symbol`` with the shard's (idempotent).

    Kept free of the heavy ``simulation`` import so it runs fast in the API
    process and is unit-testable without pandas/yfinance.
    """
    import sqlite3
    shard = _shard_db_path(symbol, main_db)
    if not shard.exists():
        return 0
    with sqlite3.connect(main_db, timeout=30) as con:
        con.execute("PRAGMA busy_timeout=30000")
        con.executescript(_TRADE_LOG_DDL)            # ensure main schema exists
        con.execute(f"ATTACH DATABASE '{shard}' AS shard")
        con.execute("DELETE FROM trade_log WHERE symbol = ?", (symbol,))
        con.execute("INSERT INTO trade_log (episode, symbol, decision_date, action, "
                    "entry_price, quantity, stop_loss, take_profit, exit_price, "
                    "exit_date, exit_reason, pnl, capital_before, capital_after, "
                    "pre_trade_reasoning, post_trade_analysis, lesson_learned, "
                    "what_went_wrong, technicals_json, created_at) "
                    "SELECT episode, symbol, decision_date, action, entry_price, "
                    "quantity, stop_loss, take_profit, exit_price, exit_date, "
                    "exit_reason, pnl, capital_before, capital_after, "
                    "pre_trade_reasoning, post_trade_analysis, lesson_learned, "
                    "what_went_wrong, technicals_json, created_at FROM shard.trade_log")
        con.execute("INSERT OR REPLACE INTO fundamental_cache "
                    "SELECT * FROM shard.fundamental_cache")
        n = con.execute("SELECT COUNT(*) FROM trade_log WHERE symbol = ?",
                        (symbol,)).fetchone()[0]
        con.commit()                 # release the write lock before detaching
        con.execute("DETACH DATABASE shard")
    return int(n)


def run_datagen_parallel(
    symbols: list[str],
    start_date: str = "2024-01-10",
    episodes: int = 50,
    backend: str = "fallback",
    adapter_path: Optional[str] = None,
    resume: bool = True,
    max_workers: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Generate data for several symbols concurrently, then merge the shards.

    Returns a per-symbol summary. The aggregate ``datagen`` state tracks overall
    progress; each symbol also has its own ``datagen:<symbol>`` state for the UI.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ensure_dirs()
    main_db = str(db_path or DB_PATH)
    symbols = [s.strip() for s in symbols if s.strip()]
    if not symbols:
        return {"error": "no symbols given"}

    workers = max_workers or min(len(symbols), os.cpu_count() or 2)
    agg = StateStore(STAGE)
    agg.start(total=len(symbols) * episodes,
              message=f"Parallel datagen: {len(symbols)} symbols × {episodes} "
                      f"episodes on {workers} workers")

    results: dict[str, dict] = {}
    done_symbols = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_datagen_worker, s, start_date, episodes, backend,
                          adapter_path, resume, main_db): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # one symbol failing must not kill the rest
                res = {"symbol": sym, "error": str(exc)}
            # Merge this symbol's shard into the main DB as soon as it's ready.
            try:
                res["rows_in_main"] = _merge_shard_into_main(sym, main_db)
            except Exception as exc:
                res["merge_error"] = str(exc)
            results[sym] = res
            done_symbols += 1
            agg.update(current=sum((r.get("episodes", 0) for r in results.values())),
                       message=f"{done_symbols}/{len(symbols)} symbols done "
                               f"(last: {sym})")

    total_eps = sum(r.get("episodes", 0) for r in results.values())
    agg.finish(message=f"Parallel datagen complete — {len(symbols)} symbols, "
               f"{total_eps} episodes total", results=results)
    return {"symbols": symbols, "workers": workers, "results": results,
            "total_episodes": total_eps}


# ════════════════════════════════════════════════════════════════════════════
#  PARALLEL EPISODES FOR A SINGLE SYMBOL (date-window sharding)
# ════════════════════════════════════════════════════════════════════════════
# One agent's episodes are sequential, but we can split the CALENDAR into W
# non-overlapping windows and run one independent agent per window in parallel.
# Window i starts further along the timeline, so "episode 123" of the whole
# dataset is produced by whichever window owns that stretch of dates — the
# windows run concurrently. Each window writes to its own shard + state; all
# shards for the symbol are then unioned into the main trade_log (the splitter's
# exact-hash de-dup removes any boundary overlap).

def _stagger_start(start_date: str, window_index: int, episodes_per_window: int) -> str:
    """Start date for window i: advance by ~episodes_per_window trading stretches."""
    from datetime import datetime, timedelta
    span = max(7, episodes_per_window * 2)          # calendar days per window
    dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=window_index * span)
    while dt.weekday() >= 5:                          # land on a weekday
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _datagen_window_worker(symbol, start_date, episodes, backend, adapter_path,
                           resume, main_db, widx) -> dict:
    """Picklable worker: one date-window of a single symbol, in its own process."""
    import common  # re-wire sys.path in the spawned process  # noqa: F401
    from modules.datagen.service import run_datagen as _run
    shard_key = f"{symbol}__w{widx}"
    res = _run(symbol=symbol, start_date=start_date, episodes=episodes,
               backend=backend, adapter_path=adapter_path, resume=resume,
               db_path=str(_shard_db_path(shard_key, main_db)),
               state_stage=f"{STAGE}:{symbol}:w{widx}")
    res["window"] = widx
    return res


def _append_shard_into_main(shard_key: str, main_db: str) -> int:
    """Append a shard's rows into main (no delete) — used for window union."""
    import sqlite3
    shard = _shard_db_path(shard_key, main_db)
    if not shard.exists():
        return 0
    with sqlite3.connect(main_db, timeout=30) as con:
        con.execute("PRAGMA busy_timeout=30000")
        con.executescript(_TRADE_LOG_DDL)
        con.execute(f"ATTACH DATABASE '{shard}' AS shard")
        con.execute("INSERT INTO trade_log (episode, symbol, decision_date, action, "
                    "entry_price, quantity, stop_loss, take_profit, exit_price, "
                    "exit_date, exit_reason, pnl, capital_before, capital_after, "
                    "pre_trade_reasoning, post_trade_analysis, lesson_learned, "
                    "what_went_wrong, technicals_json, created_at) "
                    "SELECT episode, symbol, decision_date, action, entry_price, "
                    "quantity, stop_loss, take_profit, exit_price, exit_date, "
                    "exit_reason, pnl, capital_before, capital_after, "
                    "pre_trade_reasoning, post_trade_analysis, lesson_learned, "
                    "what_went_wrong, technicals_json, created_at FROM shard.trade_log")
        con.execute("INSERT OR REPLACE INTO fundamental_cache "
                    "SELECT * FROM shard.fundamental_cache")
        con.commit()
        con.execute("DETACH DATABASE shard")
    return 1


def run_datagen_windows(
    symbol: str = "CANBK",
    start_date: str = "2024-01-10",
    episodes: int = 100,
    workers: int = 4,
    backend: str = "fallback",
    adapter_path: Optional[str] = None,
    resume: bool = True,
    db_path: Optional[str] = None,
) -> dict:
    """Generate ``episodes`` for ONE symbol across ``workers`` parallel windows."""
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ensure_dirs()
    main_db = str(db_path or DB_PATH)
    workers = max(1, workers)
    per = max(1, episodes // workers)
    agg = StateStore(STAGE)
    agg.start(total=workers * per,
              message=f"Parallel episodes: {symbol} on {workers} windows × {per}")

    starts = [_stagger_start(start_date, i, per) for i in range(workers)]
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_datagen_window_worker, symbol, starts[i], per, backend,
                          adapter_path, resume, main_db, i): i for i in range(workers)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[f"w{i}"] = fut.result()
            except Exception as exc:
                results[f"w{i}"] = {"window": i, "error": str(exc)}
            agg.update(current=sum(r.get("episodes", 0) for r in results.values()),
                       message=f"{len(results)}/{workers} windows done")

    # Union all window shards into the main trade_log for this symbol.
    with sqlite3.connect(main_db, timeout=30) as con:
        con.execute("PRAGMA busy_timeout=30000")
        con.executescript(_TRADE_LOG_DDL)
        con.execute("DELETE FROM trade_log WHERE symbol = ?", (symbol,))
        con.commit()
    for i in range(workers):
        _append_shard_into_main(f"{symbol}__w{i}", main_db)

    with sqlite3.connect(main_db, timeout=30) as con:
        rows = con.execute("SELECT COUNT(*) FROM trade_log WHERE symbol=?",
                           (symbol,)).fetchone()[0]
    total_eps = sum(r.get("episodes", 0) for r in results.values())
    agg.finish(message=f"Parallel episodes complete — {symbol}: {total_eps} episodes "
               f"across {workers} windows → {rows} rows", results=results)
    return {"symbol": symbol, "workers": workers, "episodes_per_window": per,
            "windows": results, "rows_in_main": rows}
