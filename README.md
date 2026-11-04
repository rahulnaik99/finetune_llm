# Trading Agent — LLM Fine-tuning Pipeline

End-to-end pipeline for fine-tuning **nvidia/Nemotron-Nano-4B-Instruct** on
trading simulation data. Covers data generation, LoRA/QLoRA training,
production serving, orchestration, versioning, and monitoring.

> **Apple Silicon (M1–M5):** runs natively via MLX — no CUDA or NVIDIA GPU needed.
> **NVIDIA GPU:** QLoRA 4-bit runs on free Colab T4 (8 GB VRAM).
> `train.py` auto-detects your hardware and picks the right path automatically.

---

## What LoRA and QLoRA actually do

The model has 4 billion parameters. Retraining all of them needs ~60 GB VRAM and
costs hundreds of dollars. LoRA and QLoRA make fine-tuning affordable:

```
Standard fine-tune  →  update all 4B params  →  needs 60 GB VRAM
LoRA                →  freeze all 4B, train tiny A×B adapters (0.8% of params)
QLoRA               →  same as LoRA, but also compress the frozen base to 4-bit
                        (8× memory saving vs float32 at ~2% quality cost)
```

**LoRA diagram (rank r=16):**
```
Original layer:   output = W × input      W = 4096×4096 = 16M params  [FROZEN]
LoRA adds:        output = W×input + B×A×input
                           A = 4096×16 = 65K params  [TRAINABLE]
                           B =   16×4096 = 65K params  [TRAINABLE]
                  Total trainable: 131K  (0.78% of this layer)
```

QLoRA = LoRA + compress the frozen W from 16-bit to 4-bit NF4 (saves 70% VRAM).

---

## The Flywheel

```
Round 1  FALLBACK rules   →  50 episodes  →  fine-tune Nano-4B  →  V1
Round 2  V1 model         →  50 episodes  →  fine-tune Nano-4B  →  V2
Round 3  V2 model         →  50 episodes  →  fine-tune Nano-4B  →  V3
         ↑ better reasoning = better training data = better model ↑
```

Outcome labels (SL hit / TP hit / P&L) come from **real historical prices** —
the model cannot hallucinate its way to a profit.

---

## Full Pipeline

```
trading_agent_sim.ipynb
  ↓  run episodes (FALLBACK=True first round, fine-tuned model after)
trading_agent.db                      SQLite: trade_log + fundamental_cache
  ↓
python data_pipeline.py               extract → compact_prompt → quality-filter → split
  ↓
data/trading_train.jsonl  (~80%)
data/trading_val.jsonl    (~10%)
data/trading_test.jsonl   (~10%)
  ↓
python train.py                       auto-detects hardware → picks LoRA or QLoRA
  ├── Apple Silicon  →  MLX LoRA      (mlx-lm, native, no CUDA)
  ├── NVIDIA ≥16GB   →  HF LoRA       (bfloat16 frozen + adapters)
  ├── NVIDIA 8-16GB  →  HF QLoRA 8bit (8-bit base + adapters)
  └── NVIDIA <8GB    →  HF QLoRA 4bit (NF4 base + adapters, Colab T4)
  ↓
output/nemotron-trading/final_adapter/    (~60 MB LoRA weights)
  ↓
python evaluate.py                    JSON validity + schema accuracy + reasoning quality
  ↓  (pass)
inference_mlx.py / inference_vllm.py  drop-in for Ollama in simulation
  ↓
Back to simulation with better model → repeat
```

For the **automated loop**:
```bash
python flywheel.py --symbol CANBK --rounds 5 --episodes 50
# or scheduled via Prefect:
python prefect_flows.py
```

---

## Quick start

### 1. Install

```bash
# Core (all platforms)
pip install yfinance pandas numpy mlflow dvc prefect evidently polars duckdb

# Apple Silicon
pip install mlx-lm

# NVIDIA GPU
pip install transformers datasets peft trl bitsandbytes accelerate
pip install unsloth        # optional: 2× faster LoRA on CUDA
pip install vllm           # optional: production serving
```

### 2. Generate training data

```bash
# Run simulation (generates trading_agent.db)
jupyter notebook trading_agent_sim.ipynb
# Set FALLBACK_MODE = True for first round (no Ollama needed)
# Set MAX_EPISODES = 50 per run, run across 10 symbols, 3+ date ranges

# Convert to JSONL training format
python data_pipeline.py --db trading_agent.db --out data/trading.jsonl
```

### 3. Train (one command — auto-detects hardware)

```bash
python train.py \
  --train data/trading_train.jsonl \
  --val   data/trading_val.jsonl

# Force a specific method:
python train.py --train data/trading_train.jsonl --force_method qlora_4bit
python train.py --train data/trading_train.jsonl --force_method mlx_lora
```

### 4. Resume interrupted training

```bash
# MLX (Apple Silicon) — resumes from last checkpoint automatically
python train.py --train data/trading_train.jsonl  # finds latest *.safetensors

# HuggingFace — pass checkpoint dir
python train.py --train data/trading_train.jsonl \
  --output_dir output/nemotron-trading  # trainer auto-resumes from checkpoint
```

### 5. Track with MLflow

```bash
# Start MLflow server (keep open in separate terminal)
mlflow ui           # open http://localhost:5000

# All train.py runs log automatically when MLFLOW_TRACKING_URI is set
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py --train data/trading_train.jsonl
# → see loss curves, params, adapter artifacts in the UI
```

---

## Training method comparison

| Method | VRAM | Time (500 ex) | Quality vs full FT | Use when |
|---|---|---|---|---|
| **MLX LoRA** | 4-6 GB (unified) | ~6h on M5 Air | ~98% | Apple Silicon |
| **HF LoRA** | 12-14 GB | ~2h on A100 | ~99% | NVIDIA ≥ 16 GB |
| **HF QLoRA 8-bit** | 8-10 GB | ~2.5h on A10G | ~98% | NVIDIA 8-16 GB |
| **HF QLoRA 4-bit** | 4-6 GB | ~3h on T4 | ~97% | Free Colab T4 |
| Full fine-tune | 60 GB | ~12h on A100 | 100% | Not needed at this scale |

At 500 training examples the quality difference between QLoRA 4-bit and full fine-tuning
is **negligible** — you would need 100k+ examples before it matters.

---

## Hardware guide

### Apple M5 Air 16 GB (your setup)

```bash
# train.py auto-selects MLX LoRA
python train.py --train data/trading_train.jsonl

# Run overnight in sessions (external cooling recommended)
# Session 1 (30-40 min): --iters 150
# Session 2: --iters 150  (auto-resumes from checkpoint)
# Session 3: --iters 150  (auto-resumes)
# Session 4: --iters 150  (done)
```

**Important:** `bitsandbytes` is CUDA-only — it will not install on Apple Silicon.
`train.py` never calls it on MPS — it routes to the MLX path automatically.

### Model sizes in 16 GB unified memory

| Model | Ollama size | Inference | Train on Mac |
|---|---|---|---|
| nemotron-nano-4b | ~2.5 GB | ✅ fast | ✅ |
| nemotron-bolt-e4b | ~3.0 GB | ✅ fast | ✅ |
| qwen3:8b | ~5.0 GB | ✅ | ✅ |
| qwen3:14b | ~9.0 GB | ✅ | ⚠️ tight |
| qwen3:30b | ~18 GB | ❌ | ❌ |

### Google Colab A100 (fastest)

```python
# Upload trading_agent.db then:
!pip install -r requirements.txt
!pip install peft trl bitsandbytes accelerate unsloth
!python data_pipeline.py --db trading_agent.db --out data/trading.jsonl
!python train.py --train data/trading_train.jsonl --val data/trading_val.jsonl
# Download output/nemotron-trading/final_adapter/ (~60 MB)
# Total: ~2 hours. Cost: ~$3 on A100 Pro
```

---

## Hyperparameter reference (params.yaml)

All hyperparameters live in `params.yaml`. DVC tracks changes:

```yaml
lora_r:        16     # rank — higher = more capacity, more VRAM
                       # r=8  for <200 examples
                       # r=16 for 200-2000 examples   ← our case
                       # r=32 for >2000 examples
lora_alpha:    32     # always 2× lora_r (standard rule of thumb)
lora_dropout:  0.05
max_seq_length: 1024  # compact prompts are ~188 tokens; 1024 is safe
learning_rate: 2e-4   # standard starting point for LoRA
batch_size:    2
grad_accum:    8      # effective batch = 2 × 8 = 16
num_epochs:    3
```

Change any value, then `dvc repro` — it re-runs only the affected stages.

---

## Token optimisation (compact_prompt.py)

Standard prompts for trading data are ~990 tokens. `compact_prompt.py` reduces
this to **~188 tokens (82% reduction)** with zero information loss:

| Section | Before | After |
|---|---|---|
| System prompt | ~250 tok | ~55 tok |
| 5 raw OHLCV candles | ~180 tok | ~50 tok (derived metrics) |
| S/R zones as JSON | ~120 tok | ~35 tok (compact notation) |
| EMA/VWAP labels | ~60 tok | ~25 tok |
| Trade history (5 ep) | ~250 tok | ~90 tok (table format) |
| Fundamentals | ~20 tok | ~30 tok (real 1-line summary) |
| **Total** | **~990** | **~188** |

Training on M5 Air: 990 tokens/sample → ~8h. 188 tokens/sample → **~1.5h**.

---

## Teacher / student strategy

Do **not** use the same 4B model to generate its own training data (model collapse):

| Role | Model | Job |
|---|---|---|
| **Teacher** (generate data) | `nemotron-bolt-e4b-16k` or `qwen3:8b` | Runs in Ollama, generates training data |
| **Student** (fine-tune) | `Nemotron-Nano-4B-Instruct` | Fine-tuned on teacher's data, deployed for inference |

For the very first round, use `FALLBACK_MODE = True` (rule-based, zero hallucination).

---

## Production framework stack

| Layer | Framework | Install | Docs |
|---|---|---|---|
| Fast DataFrames | **Polars** | `pip install polars` | candle_store.py |
| Candle queries | **DuckDB** | `pip install duckdb` | candle_store.py |
| Experiment tracking | **MLflow** | `pip install mlflow` | auto in train.py |
| Fine-tuning (GPU) | **Axolotl** | `pip install axolotl` | axolotl_config.yaml |
| Fine-tuning (Mac) | **MLX-LM** | `pip install mlx-lm` | train.py PATH A |
| Production serving | **vLLM** | `pip install vllm` | inference_vllm.py |
| Orchestration | **Prefect** | `pip install prefect` | prefect_flows.py |
| Data versioning | **DVC** | `pip install dvc` | dvc.yaml |
| Drift monitoring | **Evidently** | `pip install evidently` | monitoring.py |

---

## File reference

| File | What it does |
|---|---|
| `trading_agent_sim.ipynb` | Simulation: runs episodes, stores to SQLite |
| `data_pipeline.py` | SQLite → quality-filtered JSONL (Polars-powered) |
| `compact_prompt.py` | 82% token reduction — replaces verbose prompt builders |
| `candle_store.py` | 4-year 1-min OHLCV storage via Polars + DuckDB |
| **`train.py`** | **Unified training: auto-detects hardware → LoRA or QLoRA** |
| `axolotl_config.yaml` | Axolotl alternative — one YAML replaces all training code |
| `inference.py` | Drop-in for Ollama (PyTorch path) |
| `inference_mlx.py` | Drop-in for Ollama (Apple Silicon MLX path) |
| `inference_vllm.py` | Production batch inference via vLLM |
| `flywheel.py` | Integrated data → train → deploy loop |
| `prefect_flows.py` | Scheduled daily scoring + weekly fine-tuning |
| `evaluate.py` | JSON validity, schema accuracy, reasoning quality |
| `monitoring.py` | Evidently AI drift detection — when to retrain |
| `dvc.yaml` | DVC pipeline: fetch → simulate → train → evaluate |
| `params.yaml` | Single source of truth for all hyperparameters |
| `requirements.txt` | All dependencies with platform notes |
| `generate_viz.py` | Interactive HTML dashboard (candles, decisions, P&L) |
| `show_training_data.py` | Browse training samples in browser |

---

## Pros and cons

| | |
|---|---|
| ✅ One command trains on any hardware | `python train.py` — auto routes to MLX, LoRA, or QLoRA |
| ✅ 82% token reduction | compact_prompt.py — 1.5h training vs 8h |
| ✅ Checkpointable | Resume any interrupted session |
| ✅ Full observability | MLflow tracks every run, DVC versions every dataset |
| ✅ Production-ready serving | vLLM replaces Ollama for 10-50× throughput |
| ✅ Drift detection | Evidently alerts when market regime shifts |
| ❌ Minimum 500 episodes needed | 15 episodes is not enough data |
| ❌ Training slow on Mac | ~1.5h with compact prompts; run overnight |
| ❌ Periodic retraining required | Market regimes shift; monitor with monitoring.py |
| ❌ bitsandbytes = CUDA only | QLoRA 4-bit needs NVIDIA; Mac uses MLX instead |

---

## Interview cheat sheet

Questions you can now answer with real code:

| Question | Answer | File |
|---|---|---|
| "What is LoRA?" | Train A×B adapters (0.8% of params) instead of full model | `train.py` `_print_lora_diagram()` |
| "What is QLoRA?" | LoRA + 4-bit NF4 quantisation of the frozen base | `train.py` `train_hf_qlora()` |
| "How do you track experiments?" | MLflow — params, metrics, artifacts per run | `train.py` + `runner.py` |
| "How do you version data?" | DVC — git for code, DVC for data+models | `dvc.yaml` |
| "How do you serve an LLM in production?" | vLLM — PagedAttention, continuous batching | `inference_vllm.py` |
| "How do you know when to retrain?" | Evidently AI — monitors action/confidence drift | `monitoring.py` |
| "How do you query 4 years of 1-min data?" | DuckDB predicate pushdown on Parquet | `candle_store.py` |
| "How do you orchestrate ML pipelines?" | Prefect flows with retries and scheduling | `prefect_flows.py` |
