# Module: Data Generation (`modules/datagen`) — a.k.a. "training data"

Generates the **raw training data**: a stream of realistic trading decisions and
their outcomes, persisted to SQLite. This is the first and most expensive stage,
so it is the one that benefits most from being **resumable**.

## What it produces

For every simulated trading day ("episode") the agent:
1. Builds a market-context snapshot (technicals + fundamentals + recent candles).
2. Asks a decision backend for a JSON action (`BUY`/`SELL`/`HOLD`, SL, TP,
   reasoning, confidence, key risk).
3. Executes the trade in a simulated book and records entry/exit/P&L.
4. Writes one row per trade to `trading_agent.db` → table `trade_log`.

Downstream, each trade becomes **two** supervised examples (see
[`splitter.md`](splitter.md)): a *pre-trade decision* and a *post-trade lesson*.

## Files

| File | Role |
|---|---|
| `simulation.py` | `TradingAgent`, DB schema (`init_db`), candle/fundamental loading, technicals, decision backends (`set_llm_backend`) |
| `zones.py` | Supply/demand zone detection, swing structure (HH/HL/LH/LL) |
| `compact_prompt.py` | Renders a token-efficient market-context prompt |
| `fundamentals_xbrl.py` | Parses official SEBI XBRL filings (PAT, GNPA, NII) |
| `candle_store.py` | Fast candle persistence/query (Polars/DuckDB optional) |
| `inference*.py` | Student-model generation backends (MLX / vLLM / HF) |
| `service.py` | **Resumable** driver + read helpers (`episodes_in_db`, `recent_trades`) |
| `api.py` | FastAPI router |

## Tech & libraries — and why

| Library | Used for | Benefit |
|---|---|---|
| **pandas / numpy** | EMA/VWAP, zones, candle math | Vectorised, battle-tested numerics |
| **yfinance** (or local CSV) | Price history | Free data; CSV mode runs fully offline/deterministically |
| **requests** | Talk to the Ollama teacher | Simple, synchronous, dependency-light |
| **SQLite** (`sqlite3`, stdlib) | Durable trade log | Append-only store that survives crashes → core of resume |
| **xml.etree** (stdlib) | XBRL fundamentals | No external XML dependency |
| **MLX / vLLM / Transformers** | Student inference backend | Lets the *fine-tuned* model drive later rounds |

Three decision backends (selectable per run):
- `fallback` — rule-based; fast smoke tests, no model needed.
- `ollama` — a local teacher LLM generates high-quality decisions.
- `adapter` — the fine-tuned student from a previous round (flywheel self-play).

## Resumability (the headline feature)

The raw episodes already persist to SQLite, so **no generated episode is ever
thrown away**. On top of that, `service.run_datagen`:

- Checkpoints `{symbol, start_date, episodes_done, date, capital}` to
  `output/state/datagen.json` **after every episode** (atomic write).
- On resume, if the request matches the checkpoint it reconstructs the
  `TradingAgent` at the saved trading day + capital and runs **only the
  remaining** episodes.
- If the target episode count is already met, it returns immediately — **zero
  regeneration**.

So a crash at episode 37/50 resumes at 38, not 1.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/datagen/run` | Launch/resume a generation job (non-blocking) |
| GET | `/api/datagen/state` | Live progress + checkpoint |
| GET | `/api/datagen/summary?symbol=CANBK` | Episodes already in the DB |
| GET | `/api/datagen/trades?limit=50` | Recent rows (dataset preview) |

## CLI

```bash
python run.py datagen --symbol CANBK --start_date 2024-01-10 --episodes 50
python run.py datagen --symbol CANBK --episodes 100            # resumes → runs 50 more
python run.py datagen --symbol CANBK --episodes 50 --no-resume # force fresh
```

## Key tunables (`.env`)

`SYMBOL`, `START_DATE`, `EPISODES`, `USE_OLLAMA`, `OLLAMA_MODEL`, `DATA_SOURCE`
(`csv`/`yfinance`), `FUNDAMENTALS_SOURCE` (`xbrl`/`yfinance`), `INITIAL_CAP`,
`RISK_PER_TRADE`, `MAX_HOLD_DAYS`, `LOOKBACK_DAYS`.
