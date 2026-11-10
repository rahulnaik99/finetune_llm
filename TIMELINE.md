# TIMELINE — change log
## Day 7 — v5 deep review (parallelism + tooling coherence)

Fixes from a review focused on parallelization and industry-tooling adoption:

- **P1 Windowed parallelism no longer overlaps/wastes compute** — single-symbol
  windows now use explicit CONTIGUOUS, NON-OVERLAPPING date ranges
  (`_window_ranges`) and the agent STOPS at its window end (`run_datagen(end_date=…)`).
  Progress is reported by windows completed, not summed episodes.
- **I3 Evidently pinned** `>=0.4,<0.5` + the import error now distinguishes
  "not installed" from "installed-but-wrong-API".
- **I4 params.yaml is authoritative** — `TrainRequest` hyper-params default to
  `None` (sentinel) and the training service resolves iters/epochs/lora_r/
  save_every/base_model from `params.yaml` (explicit values still override). Fixes
  the bug where the service's `epochs=2` silently beat `params.yaml`.
- **I5 Model ids reconciled** — `params.yaml mlx_model` now matches the canonical
  `.env`/schemas MLX id; `train.py` maps `mlx_model` on Apple Silicon.
- **I2 MLflow tracking deepened + linked** — training logs the per-iteration
  train/val loss curve + params; evaluation attaches its metrics (and the eval
  JSON artifact) to the SAME run. No-op without MLflow.
- **I7 Version bounds tightened** (Prefect `<3`, mlflow `<3`, transformers `<5`,
  polars/duckdb `<2`) + a lock-file recommendation.
- **B1** trained-hash registry moved out of `STATE_DIR` (+ denylist) so it no
  longer appears as a bogus "stage" on the status board.
- **B2** `inference_mlx` guards the version-fragile `temp`/sampler kwarg and now
  honours the per-call temperature.
- **B3** the dead `compact_prompt.SYSTEM_PRE` now matches the active
  `features_used` contract.
- **I1/I6 documented** — `candle_store.py` + `scripts/fetch_candles.py` labelled
  REFERENCE (not wired in); `dvc.yaml` labelled a single-round template.

Tests: **77 passing** (+7 regressions: window-range non-overlap, registry not a
stage, params sentinels, id reconciliation, SYSTEM_PRE, end_date param).

**Still deferred:** P2 (route parallel adapter generation through a shared vLLM
server instead of N in-process model copies) and I1 (wire CandleStore into
`get_candles`) — both need a model/runtime I can't exercise here; round-dataset
leakage redesign (incremental, globally-stable splits).

---

## Day 6 — Second audit (look-ahead & data integrity)

A DB-verified review found deeper data-integrity issues; fixed + tested:

- **Look-ahead leak in the pre-trade history** (#1): history is now built from
  **same-symbol trades strictly before the decision date** and sorted
  chronologically (`load_trades` now `ORDER BY decision_date`). Verified: 0
  look-ahead violations in the rebuilt dataset (previously confirmed present).
- **History symbol-scoped** (#2): no cross-symbol trades leak into a prompt.
- **DB-level dedup** (#3): `run_episode` skips inserting a row that already
  exists for `(symbol, decision_date)`; the shipped DB was de-duplicated
  (21 → 9 unique rows).
- **features_used contract** (#4): now persisted by the simulator, emitted in
  the training example, and part of `PRE_TRADE_SYSTEM` — so `features_used_ok`
  is meaningful instead of structurally 0.
- **Resume history hydration** (#5): on resume the agent re-loads its history
  from the DB (post-resume episodes keep real context); summary is cumulative.
- **Eval cache fingerprinted** (#6): a cache from a different test set is no
  longer reused.
- **reasoning_ok null-safe** (#7): `reasoning: null` no longer aborts the pass.
- Low: state lock keyed by the sanitised name; `.env` values containing `#`
  preserved; `volume >= 0` data check; Prefect flow honours round-level resume.

Tests: **70 passing** (+6 regressions). Rebuilt `round_01` from the cleaned DB
(leak-free): train/val/test = 11/1/1.

---

## Day 5 — Audit & fixes

An external code audit found real bugs (several inherited from the original
flat project). Fixed and regression-tested:

**Critical data correctness**
- SELL risk:reward is now computed (was hardcoded 0.0 → every SELL example was
  dropped by the quality filter). Proven: `round_01` went 0→24 train rows.
- Post-trade rows no longer collapse to one dedup key (content-hash key).
- Rule-based fallback now has a symmetric SELL rule.
- Model fields are coerced safely (`'N/A'`/None stop_loss no longer crashes the
  episode). Confidence + strategy_adjustment are stored (min_confidence works).
- Datagen state reconciles with the DB; stale checkpoints reset on symbol/date
  change.

**Resume / state**
- StateStore load-modify-save is now atomic (RLock); filenames are
  filesystem-safe (no `:`); shard/window merges are **non-destructive**
  (additive + signature dedup, param-bound ATTACH); window remainder kept.

**Training / orchestration**
- Never pass a bare LoRA adapter as `--model` (use the prev fused model or base).
- Training fails on a non-zero subprocess exit (no half-written adapter accepted).
- A failed round is **retried** on resume (not skipped forever).

**Evaluation / API**
- `profit_factor` is JSON-safe (`None`, not `inf`) → `/evaluation/metrics` can't
  500. Evaluation reads its cache and honours `force`. Model-quality splits
  pre-trade vs post-trade rows. Monitoring path allow-listing. FastAPI `lifespan`.

**UI** — `escapeHtml` now escapes quotes/backticks; messages/errors escaped.

**Hygiene** — `requirements-gpu.txt` added; splitter no longer mutates
`os.environ`; tiny-dataset slice guard; model card honours `round_num`.

Tests: **64 passing** (+6 audit regressions: SELL R:R, post-trade dedup, inf
JSON, FS-safe state, non-destructive merge, model-quality row split).

**Deferred (need design work or an environment I can't exercise here):** round
dataset leakage redesign (incremental, globally-stable splits); MLX/CUDA
training, Prefect & Evidently runtime paths (optional deps); regenerating a
large dataset from the CSV prices (the shipped DB has only 21 rows).

---


Reverse-chronological record of what changed and why.

## Day 4 — Industry-standard alignment

| Area | What shipped |
|---|---|
| Evaluation | Added **Sortino**, **t-stat / 95% significance**, and **multi-fold walk-forward** robustness; documented why **ROUGE is not used** for structured trading output. |
| Orchestration | Added **Prefect** flows (`--engine prefect`) as a standard workflow engine with retries/caching; native loop kept as fallback. FastAPI reduced to its correct API/UI role. |
| Reproducibility | Re-added a **DVC** pipeline (`dvc.yaml` + `params.yaml`) for versioned, change-aware stage runs. |
| Governance | Auto-generated **Model Card** + **Dataset Datasheet** (`run.py card`, `/api/governance/*`). |
| Data validation | **pandera-optional** CSV quality gate (`run.py validate`) with a dependency-free fallback. |
| Deps | `requirements-optional.txt` lists the framework extras (prefect, dvc, pandera, evidently, mlflow, bert-score) — all optional/lazy. |

Tests: **58 passing** (added advanced-metrics, governance+validation, and
orchestration-flows suites).

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
