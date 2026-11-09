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
                "exit_price, pnl, exit_reason FROM trade_log "
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
) -> dict:
    """Generate ``episodes`` episodes, resuming from the last checkpoint."""
    ensure_dirs()
    store = StateStore(STAGE)
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
