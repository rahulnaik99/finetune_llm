# Master Architecture — Trading-Agent LLM Fine-tuning Pipeline

This document explains the **whole system** end to end: what each module does,
the technology and libraries it uses, why those choices are beneficial, how the
modules talk to each other, and how the pipeline resumes after a break.

Per-module deep-dives live alongside this file:
[`trainingdata.md`](trainingdata.md) ·
[`splitter.md`](splitter.md) ·
[`training.md`](training.md) ·
[`evaluation.md`](evaluation.md) ·
[`orchestrator.md`](orchestrator.md) ·
[`monitoring.md`](monitoring.md) · [`CODEWALK.md`](CODEWALK.md) ·
[`resume.md`](resume.md) · [`api.md`](api.md)

---

## 1. What the system does

It fine-tunes a small instruction LLM (NVIDIA Nemotron-Nano-4B class) to act as a
disciplined NSE equity trading agent, using a **self-improving flywheel**:

```
          ┌──────────────┐   episodes    ┌──────────────┐   JSONL     ┌──────────────┐
          │  DATAGEN     │ ────────────► │  SPLITTER    │ ──────────► │  TRAINING    │
          │ (simulation) │   SQLite      │ (dataset +   │  train/val/ │ (LoRA/QLoRA/ │
          │              │               │  split)      │  test       │  MLX)        │
          └──────┬───────┘               └──────────────┘             └──────┬───────┘
                 ▲                                                            │ adapter
                 │ better student adapter                                     ▼
          ┌──────┴───────┐                                           ┌──────────────┐
          │ ORCHESTRATOR │ ◄──────────────── metrics ─────────────── │  EVALUATION  │
          │  (flywheel)  │                                           │ (walk-fwd)   │
          └──────────────┘                                           └──────────────┘
```

Each round: the current best model generates trading decisions → decisions become
training data → the model is fine-tuned on the growing buffer → it is evaluated on
a **walk-forward** (future-dated) test set → the improved adapter drives the next
round of data generation.

---

## 2. Why it was restructured

The original project was ~25 Python files in a single flat folder with one
monolithic `flywheel.py` loop. Problems it solved by restructuring:

| Problem (before) | Solution (after) |
|---|---|
| One flat folder, hard to reason about | One **folder per stage** under `modules/` |
| A crash re-ran the whole loop and **regenerated identical data** | Every stage is **resumable** via an atomic checkpoint store |
| No way to drive a single stage programmatically | Each stage has a thin **`service.py`** + a **FastAPI `api.py`** |
| No visibility into progress / data / metrics | A **lightweight web UI** tracks every stage live |
| No automated gate on commits | **GitHub Actions CI** builds + tests on every push |
| Eval trading-metrics silently broken (`_meta` stripped) | Splitter writes a **`_meta` sidecar**; evaluator reads it |

Crucially, the restructure is **non-breaking**: the heavy domain modules still
import each other by their original bare names (`import simulation`,
`from data_pipeline import build_dataset`, …). A single bootstrap
(`common/_bootstrap.py`) adds every stage folder to `sys.path`, so no
cross-module import had to change.

---

## 3. Repository layout

```
finetune_llm/
├── run.py                     # one CLI entry point for every stage + `serve`
├── conftest.py                # pytest path bootstrap
├── pyproject.toml             # project + pytest config
├── requirements.txt           # full runtime deps (training backends)
├── requirements-dev.txt       # slim deps CI needs to build + test
├── params.yaml                # central training hyper-parameters
├── .env.example               # all tunables (copy to .env)
├── .github/workflows/ci.yml   # build + test on every commit
│
├── common/                    # shared infrastructure (no business logic)
│   ├── _bootstrap.py          # sys.path wiring → keeps bare imports working
│   ├── config.py              # .env loader (env_str/int/float/bool)
│   ├── paths.py               # one place that resolves every path
│   ├── state.py               # atomic, resumable JobState / StateStore
│   ├── jobs.py                # 1-thread-per-stage background runner
│   └── schemas.py             # pydantic request/response models
│
├── modules/                   # one folder per pipeline stage
│   ├── datagen/               # simulation, zones, prompts, fundamentals, inference
│   ├── splitter/              # data_pipeline (build + quality filter + split)
│   ├── training/              # train.py (+ mlx/qlora backends)
│   ├── evaluation/            # evaluate.py (trading + model-quality metrics)
│   ├── orchestrator/          # the flywheel, rebuilt on the 4 stages
│   └── monitoring/            # Evidently drift report + Ollama deploy
│       └── (each has service.py + api.py)
│
├── app/
│   ├── main.py                # FastAPI app: mounts all routers + the UI
│   └── ui/                    # dependency-free dashboard (HTML/CSS/JS)
│
├── docs/                      # this folder
├── tests/                     # build / unit / state / API / regression tests
├── data/                      # prices, fundamentals, per-round datasets
├── output/                    # adapters, state/, metrics/, logs/ (regenerable)
└── trading_agent.db           # SQLite trade log (the raw generated data)
```

---

## 4. The three layers

### a) Heavy domain modules (unchanged logic)
`simulation.py`, `data_pipeline.py`, `train.py`, `evaluate.py`, `zones.py`,
`compact_prompt.py`, `fundamentals_xbrl.py`, `candle_store.py`, `inference*.py`,
`monitoring.py`. These contain the actual trading / ML logic and were moved —
not rewritten — into their stage folders.

### b) Orchestration layer (new, thin)
Every stage folder adds two small files:
- **`service.py`** — a resumable function that drives the heavy module and
  reports progress to the `StateStore`.
- **`api.py`** — a FastAPI `APIRouter` exposing `POST /run` + `GET /state` (and
  stage-specific read endpoints).

### c) App + UI (new)
`app/main.py` mounts all six routers under `/api` and serves the dashboard at
`/ui`. The UI is plain HTML/CSS/JS (no build step, no CDN) and polls the APIs.

---

## 5. Technology choices & why they help

| Concern | Technology | Benefit |
|---|---|---|
| Config | `.env` + tiny parser (`common/config.py`) | No code edits to retune; works even without `python-dotenv` |
| Raw data store | **SQLite** (`trading_agent.db`) | Zero-setup, durable, append-only → never regenerated on resume |
| State / checkpoints | JSON + **atomic `os.replace`** | Crash-safe; the UI can read while a job writes |
| Background jobs | `threading` (1 worker/stage) | Non-blocking API without Celery/Redis weight |
| API | **FastAPI + pydantic + uvicorn** | Typed contracts, auto OpenAPI docs, async server |
| UI | Vanilla HTML/CSS/Canvas | Lightweight, CSP-safe, no dependencies to maintain |
| Training | **MLX-LoRA** (Apple Silicon) / **QLoRA via PEFT+TRL+bitsandbytes** (NVIDIA) | `train.py` auto-detects hardware; runs on a Mac or a free Colab T4 |
| Numerics / features | **pandas / numpy** | Vectorised technical-indicator computation |
| Eval | pure-python math + optional MLX generation | Trading metrics always available; model-quality optional |
| Drift | **Evidently** (optional) | Detects when retraining is needed |
| CI | **GitHub Actions** + **pytest** | Build + test gate on every commit across Py 3.11–3.13 |

---

## 6. Data & control flow (one round)

1. **datagen** — `simulation.TradingAgent` replays trading days, asks the current
   backend (rules / Ollama teacher / fine-tuned student) for a decision, executes
   it, and logs every trade to SQLite. Checkpoints after each episode.
2. **splitter** — `data_pipeline.build_dataset` reads the trade log, builds two
   examples per trade (pre-trade decision + post-trade lesson), quality-filters,
   de-dups, balances BUY/SELL, and writes a **walk-forward** train/val/test split
   (+ a `_meta` sidecar). Skips if the DB + params fingerprint is unchanged.
3. **training** — `train.py` fine-tunes on the **accumulated** train buffer,
   starting from the previous adapter (continual learning). Streams loss to the
   state history. Skips if the round's adapter already exists.
4. **evaluation** — `evaluate.py` computes trading metrics on the future-dated
   test split (win rate, expectancy, profit factor, Sharpe/trade, max drawdown)
   and optional model-quality metrics. Caches results per round.
5. **orchestrator** — repeats 1–4, remembering only the last finished round so a
   resumed run jumps straight to the first unfinished round.

---

## 7. Running it

```bash
pip install -r requirements.txt            # full (adds a training backend)
# or, just to build/test/serve the control plane:
pip install -r requirements-dev.txt

python run.py serve                        # UI at http://127.0.0.1:8000/ui
python run.py flywheel --rounds 5 --episodes 50
python run.py status                       # dump every stage's state
pytest -q                                  # run the test suite
```

See the root [`README.md`](../README.md) for the full quick-start, `.env`
reference, and hardware notes.
