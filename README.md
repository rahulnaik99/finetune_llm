# Trading-Agent — LLM Fine-tuning Pipeline

A modular, **resumable** pipeline that fine-tunes a small instruction LLM
(NVIDIA Nemotron-Nano-4B class) into a disciplined NSE equity trading agent,
using a self-improving **flywheel**: generate data → split → fine-tune →
evaluate → repeat with the improved model.

> **Apple Silicon (M1–M5):** trains natively via MLX — no CUDA/NVIDIA GPU.
> **NVIDIA GPU:** QLoRA 4-bit runs on a free Colab T4 (8 GB VRAM).
> `train.py` auto-detects your hardware and picks the right path.

This repository was **restructured from a flat script collection into
independent, resumable modules** — each with its own folder, a FastAPI endpoint,
and a shared tracking UI. See [`docs/MASTER.md`](docs/MASTER.md) for the full
architecture.

---

## Why this structure

| Goal | How it is met |
|---|---|
| Separate modules (datagen / splitter / training / evaluation) | One folder per stage under `modules/` |
| A break anywhere resumes **without regenerating data** | Atomic per-stage checkpoint store (`common/state.py`) — see [`docs/resume.md`](docs/resume.md) |
| Interact with each module | A FastAPI `POST /run` + `GET /state` per stage (`app/main.py`) |
| Track progress, view data, watch training/eval | Lightweight web UI at `/ui` |
| Guard every commit | GitHub Actions CI builds + runs `pytest` |
| Non-breaking | Heavy modules still import by bare name via `common/_bootstrap.py` |

---

## Layout

```
run.py                 # one CLI for every stage + `serve`
app/                   # FastAPI app (app/main.py) + dashboard (app/ui/)
common/                # config, paths, state (resume), jobs, schemas
modules/
  datagen/             # trading simulation → SQLite trade log
  splitter/            # build + quality-filter + walk-forward split (+ _meta sidecar)
  training/            # LoRA / QLoRA / MLX fine-tuning (auto-detect hardware)
  evaluation/          # walk-forward trading metrics + model quality
  orchestrator/        # the flywheel, built on the 4 stages
  monitoring/          # Evidently drift report + Ollama deploy
docs/                  # MASTER + per-module deep-dives
tests/                 # build / unit / state / API / regression
data/, output/         # datasets, adapters, state/, metrics/
trading_agent.db       # raw generated data (SQLite)
```

Full module docs: [datagen](docs/trainingdata.md) ·
[splitter](docs/splitter.md) · [training](docs/training.md) ·
[evaluation](docs/evaluation.md) · [orchestrator](docs/orchestrator.md) ·
[monitoring](docs/monitoring.md) · [resume](docs/resume.md) · [api](docs/api.md)

---

## Quick start

```bash
# 1. Environment
python3.12 -m venv venv && source venv/bin/activate
cp .env.example .env                       # edit values as needed

# 2a. Control plane only (serve UI + API, run tests)
pip install -r requirements-dev.txt

# 2b. Full runtime (adds a training backend)
pip install -r requirements.txt
pip install mlx-lm                         # Mac; or: pip install torch peft trl bitsandbytes accelerate  (NVIDIA)

# 3. Launch the UI + API
python run.py serve                        # → http://127.0.0.1:8000/ui
```

### Drive it from the CLI

```bash
python run.py datagen  --symbol CANBK --episodes 50     # resumable
python run.py split    --round 1
python run.py train    --round 1 --iters 600            # skips if adapter exists
python run.py eval     --round 1
python run.py flywheel --rounds 5 --episodes 50         # the whole loop
python run.py status                                    # dump every stage's state
```

Resume is automatic — re-running continues from the last checkpoint. Force a
fresh run with `--no-resume` (datagen/training/flywheel) or `--force` (split).

### Drive it from the API

```bash
curl -X POST localhost:8000/api/orchestrator/run \
  -H 'content-type: application/json' \
  -d '{"symbol":"CANBK","rounds":5,"episodes":50,"resume":true}'
curl localhost:8000/api/overview
```

Interactive docs at `/docs`. Full endpoint list in [`docs/api.md`](docs/api.md).

---

## The UI

A dependency-free dashboard (SAP *Clean Minimal*, light theme) served at `/ui`:

- **Overview** — status board for all six stages + flywheel launcher.
- **Data Generation** — run/resume, live progress, recent generated trades.
- **Dataset Viewer** — browse per-round train/val/test examples.
- **Training** — run + a live train/val **loss curve**.
- **Evaluation** — trading-metric cards + a **win-rate-by-round** chart.

---

## Testing & CI

```bash
pytest -q
```

`tests/` covers: every file builds (AST parse), all modules import under the new
layout, the FastAPI wiring, the resumable state store, trading-metric math, the
data balancer, zone detection, prompt rendering, XBRL parsing, and the `_meta`
sidecar regression. `.github/workflows/ci.yml` runs this on every push/PR across
Python 3.11–3.13.

---

## Configuration (`.env`)

Everything is configured in `.env` — no code edits. Highlights:

| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL` | `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit` | Student model to fine-tune |
| `USE_OLLAMA` | `1` | `1` = Ollama teacher generates data; `0` = rule-based fallback |
| `OLLAMA_MODEL` / `OLLAMA_URL` | `nemotron-bolt-e4b-16k` / `localhost:11434` | Teacher backend |
| `SYMBOL` / `START_DATE` | `CANBK` / `2024-01-10` | What + when to simulate |
| `ROUNDS` / `EPISODES` | `5` / `50` | Flywheel size |
| `MLX_ITERS` / `LORA_R` / `MLX_NUM_LAYERS` | `600` / `16` / `8` | MLX training knobs |
| `EPOCHS` / `SAVE_EVERY` | `2` / `100` | Training checkpoints |
| `DATA_SOURCE` / `FUNDAMENTALS_SOURCE` | `yfinance` / `yfinance` | `csv`/`xbrl` for offline/official data |
| `SPLIT_MODE` | `time` | `time` = walk-forward, `random` otherwise |
| `DB_PATH` / `ADAPTER_DIR` / `DATA_DIR` | `trading_agent.db` / `output/adapters` / `data/rounds` | Locations |

(See [`docs/trainingdata.md`](docs/trainingdata.md) and
[`docs/training.md`](docs/training.md) for the rest.)

---

## What LoRA / QLoRA do

The base model has ~4B parameters; full fine-tuning needs ~60 GB VRAM. LoRA
freezes those weights and trains tiny low-rank adapters (~0.8% of params); QLoRA
additionally 4-bit-quantises the frozen base. The result fine-tunes on a Mac or a
free Colab T4 and ships adapters that are only a few MB. `train.py` picks MLX or
QLoRA automatically. Details in [`docs/training.md`](docs/training.md).

---

## Hardware notes

| Platform | Backend | Install |
|---|---|---|
| Apple Silicon | MLX-LoRA | `pip install mlx-lm` |
| NVIDIA GPU | QLoRA 4-bit | `pip install torch peft trl bitsandbytes accelerate` |

The control plane (UI, API, splitter, evaluation math, tests) needs **neither** —
only `requirements-dev.txt`.
