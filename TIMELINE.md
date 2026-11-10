# TIMELINE — change log

Reverse-chronological record of what changed and why. Dates are relative
("Day N") since this is an iterative hand-off.

---

## Day 3 — Data-gen realism, parallel episodes, train tracking

Requested items 0–6 (single symbol: CANBK).

| # | Request | What shipped |
|---|---------|--------------|
| 0 | Run episodes in parallel (per-process, not per-stock) | **Date-window sharding**: one symbol's calendar is split into `W` windows, one independent agent per window runs in its own process, shards union into the main DB (exact-hash dedup removes overlap). `run.py datagen --symbol CANBK --episodes 200 --windows 4` · `POST /api/datagen/run_windows`. (Episodes inside one window stay sequential — capital/date chain — but windows run concurrently.) |
| 1 | Stop today, resume tomorrow, train on day 3 with all data | Confirmed + guaranteed: episodes persist in `trading_agent.db`, checkpoints in `output/state/`; the splitter builds from the **whole** DB and training merges **all** rounds, so later days use everything generated earlier. |
| 2 | See the reasoning | `/api/datagen/trades` now returns `pre_trade_reasoning`, `lesson_learned`, `quantity`; UI Data-Generation table shows a **Reasoning** column (full text on hover). |
| 3 | Train button / select data / know what's already trained / unique rows | Train button + round selector already present; added **exact content-hash dedup** (16-hex SHA1 of user+assistant) for unique rows; a tiny **trained-hash registry** (~16 B/row) records what fed each adapter; `GET /api/training/coverage` → `{total,new,already_trained}`; Dataset Viewer badges each row **trained**/**new**. |
| 4 | Fees & taxes in P&L | Realised P&L is now **net** of brokerage (capped), STT, exchange txn, SEBI, GST, stamp duty (Indian defaults, `.env`-configurable; `INCLUDE_COSTS=0` = gross). Gross + cost stored in the trade snapshot. |
| 5 | VIX-based quantity/points | Volatility sizing: **low VIX → more quantity + nearer targets**, **high VIX → less quantity + farther targets + wider stop** (regime factors configurable). |
| 6 | INDIA VIX CSV format | Loader handles `Ticker,Date/Time,…,Close,…` minute format (`DD-MM-YYYY HH:MM:SS`), drops ticker columns, resamples to daily; VIX auto-detected from `data/prices/INDIAVIX.csv` and **on by default**. |

Tests: **47 passing** (added cost model, VIX sizing, content-hash dedup + coverage, parallel-window merge, new API contracts). Bug fixed: SQLite `DETACH` needed a commit first.

---

## Day 2 — Robustness fixes (offline imports)

- `simulation.py` made importable **offline**: `yfinance` and `requests` are now
  lazy/optional (only needed for the online price source / Ollama teacher). A
  clear error is raised only if those backends are actually used.
- Added a subprocess regression test that blocks both packages.

---

## Day 1 — Restructure into modules (non-breaking)

- Split the flat script collection into one folder per stage under `modules/`:
  **datagen → splitter → training → evaluation**, plus **orchestrator** and
  **monitoring**; shared infra in `common/`.
- **Resumable everywhere** via an atomic JSON checkpoint store (`common/state.py`):
  datagen (per-episode), splitter (DB fingerprint), training (adapter-exists),
  evaluation (cached), orchestrator (per-round).
- **FastAPI endpoint per module** (`POST /run` + `GET /state`) under one app
  (`app/main.py`) + a dependency-free **tracking UI** at `/ui`.
- **CI** (`.github/workflows/ci.yml`) builds + runs `pytest` on every commit.
- Docs: `docs/MASTER.md` + per-module deep-dives; README rewritten.
- Parallel **multi-symbol** datagen (one process/symbol) + shard merge.
- Removed duplicate/junk files.
- Bug fixed: splitter stripped `_meta`, silently emptying walk-forward trading
  metrics → now writes a `_meta` sidecar the evaluator reads.
