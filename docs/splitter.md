# Module: Splitter (`modules/splitter`)

Turns the raw SQLite trade log into **clean, quality-filtered, balanced, and
split** JSONL ready for supervised fine-tuning.

## What it does (pipeline inside `data_pipeline.build_dataset`)

1. **Load** non-HOLD trades with reasoning + lessons from `trade_log`.
2. **Build two examples per trade**
   - *pre-trade*: market context → decision JSON (action, SL, TP, R:R, reasoning…)
   - *post-trade*: trade outcome → lesson JSON (what went right/wrong, adjustment)
   Both in HuggingFace **chat format** (`messages: [system, user, assistant]`).
3. **Quality filter** — drop malformed JSON, HOLDs, low-confidence, poor R:R, and
   (optionally) trades without fresh fundamentals.
4. **De-dup** — collapse near-identical setups by a coarse decision signature.
5. **Balance** — cap BUY vs SELL so one side cannot dominate (prevents the
   "always-HOLD / loss→0" memorisation collapse).
6. **Split** — `SPLIT_MODE=time` gives a **walk-forward** split (earliest dates
   train, latest dates test = genuine future), or `random`.
7. **Write** `trading_{train,val,test}.jsonl` **plus a `_meta` sidecar** per split.

## Why the `_meta` sidecar matters (a bug we fixed)

The training files must **not** contain `_meta` (SFT only wants `messages`). But
the evaluator needs each test example's recorded P&L to compute trading metrics.
Originally `_meta` was stripped on write, so `evaluate.pnls_from_test_meta`
always saw nothing → **walk-forward trading metrics were silently empty**.

Fix: the splitter now writes `trading_<split>_meta.jsonl` alongside each split,
and `evaluate.pnls_from_test_meta` reads inline `_meta` if present, otherwise
falls back to the sidecar. Locked by `tests/test_eval_sidecar.py`.

## Resumability / idempotency

Building is deterministic given (DB contents, filter params). The service
fingerprints the DB (size + row count) together with the split parameters and
stores it in `output/state/splitter.json`. A repeat request with an unchanged
fingerprint **reuses the existing split** instead of rebuilding. `force=true`
overrides.

## Tech & libraries

| Library | Used for | Benefit |
|---|---|---|
| **sqlite3** (stdlib) | Read trades + fingerprint the DB | No dependency; cheap change-detection |
| **hashlib** (stdlib) | Deterministic fingerprint | Reliable skip/resume signal |
| **json / random** (stdlib) | Serialisation + reproducible shuffling (`seed=42`) | Deterministic datasets |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/splitter/run` | Build + split for a round (idempotent unless `force`) |
| GET | `/api/splitter/state` | Progress + counts |
| GET | `/api/splitter/rounds` | All round folders + per-split counts |
| GET | `/api/splitter/dataset?round_num=1&split=train&limit=50&offset=0` | Paginated example viewer |

## CLI

```bash
python run.py split --round 1
python run.py split --round 1 --force     # rebuild even if unchanged
```

## Key tunables (`.env`)

`SPLIT_MODE` (`time`/`random`), `BALANCE_DATA`, `DEDUP_DATA`,
`REQUIRE_FUNDAMENTALS`, and (per request) `min_confidence`, `min_rr`,
`val_ratio`, `test_ratio`.
