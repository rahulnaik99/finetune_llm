# Trading Agent — LLM Fine-tuning Pipeline

End-to-end pipeline for generating trading simulation data, fine-tuning
**nvidia/Nemotron-Nano-4B-Instruct** with LoRA/QLoRA, and deploying the
fine-tuned model back into the simulation in place of Ollama.

> **Works on Apple Silicon (M1/M2/M3/M4/M5)** via MLX — no CUDA or NVIDIA GPU required.
> For faster training, export the dataset to Google Colab A100 (see section below).

---

## The Flywheel

```
Round 1: FALLBACK (rule-based) ──→ 50 episodes ──→ fine-tune Nano-4B (V1)
Round 2: Nano-4B V1           ──→ 50 episodes ──→ fine-tune Nano-4B (V2)
Round 3: Nano-4B V2           ──→ 50 episodes ──→ fine-tune Nano-4B (V3)
...                                                keeps improving
```

Each round the model produces better reasoning → better quality training data →
better next model. The outcome labels (SL hit / TP hit / actual P&L) come from
**real historical prices**, so the model cannot hallucinate its way to a profit.

---

## Full Pipeline

```
trading_agent_sim.ipynb
  ↓  run N episodes (FALLBACK_MODE=True for first round, fine-tuned model after)
trading_agent.db                       ← SQLite: trade_log + fundamental_cache
  ↓
python data_pipeline.py                ← extract, format, quality-filter, split
  ↓
data/trading_train.jsonl  (~80%)
data/trading_val.jsonl    (~10%)
data/trading_test.jsonl   (~10%)
  ↓
[Apple Silicon] python train_mlx.py    ← MLX LoRA, native M-series, no CUDA
[NVIDIA GPU]    python train_qlora.py  ← QLoRA via Unsloth + SFTTrainer
  ↓
output/.../final_adapter/              ← LoRA weights ~60 MB
  ↓
python evaluate.py                     ← JSON validity, schema accuracy, reasoning quality
  ↓
[Pass] inference_mlx.py / inference.py ← drop-in for Ollama in simulation
  ↓
Back to simulation with better model → repeat
```

For the **automated loop**, use:

```bash
python flywheel.py --symbol CANBK --start_date 2024-01-10 \
    --rounds 5 --episodes 50
```

---

## Training data format

Each episode generates **two training samples**:

### Pre-trade (what should I do?)
```
system  → You are a disciplined NSE equity trader. Protect capital first...
user    → [CANBK 2024-10-10]
           C:119.78 (+1.9%) | 5d:116.8-123.5 C@44pct
           ATR:3.26 | Vol:4.8M(1.0×avg) | 3d:-0.35%
           EMA: 50@118.9 < 100@120.1 < 200@122.5 [BEARISH]
           VWAP:119.2 C+0.49% | C vs EMA200:-2.22%
           S:116.5-117.2(-2.5%)  R:122.0-123.5(+2.5%)
           Fund(2024-Q2): PAT ₹5381cr | P/E 9.2
           HISTORY:
           01  BUY  SL_HIT   -52  9948  wait for EMA alignment
           CAP:9948 (death<2000)
assistant → {"action":"HOLD","quantity":0,"stop_loss":0,"take_profit":0,
              "reasoning":"EMA50<EMA100<EMA200 — bearish stack...","confidence":0.7}
```

### Post-trade (what happened and what did I learn?)
```
system  → Review the completed trade and extract one concrete lesson...
user    → [CANBK Ep01 2024-10-10]
           BUY 16@119.78 SL:116.50 TP:124.00
           EXIT:116.50 [SL_HIT] on 2024-10-11
           P&L:-52 | 10000->9948
           REASONING: Price at demand zone, expecting bounce...
assistant → {"outcome":"loss","lesson_learned":"Do not enter BUY when EMA stack bearish",
              "strategy_adjustment":"Require 50>100>200 before any long entry"}
```

**Token count (compact format):** ~188 tokens vs ~990 (verbose) — **82% reduction, zero quality loss.**
See `compact_prompt.py` for the full optimisation breakdown.

---

## Data storage design

```
data/
├── candles/
│   ├── CANBK_1m_2021-01.parquet   ← 1-min OHLCV per stock per month
│   ├── CANBK_1h_2021-Q1.parquet   ← 1-hour OHLCV per stock per quarter
│   └── CANBK_1d.parquet           ← full daily history (one file per stock)
└── trading_agent.db               ← SQLite: fundamentals cache + trade log
```

**Why Parquet over CSV:** 8× smaller, 40× faster date-range reads, native pandas support.
SQLite for fundamentals: quarterly results change 4×/year — tiny volume, easy to query.

### Lookback windows given to the model

| Timeframe | Bars stored | What LLM receives | Purpose |
|---|---|---|---|
| Daily (1D) | 4 years | Last 60 bars (compressed) | Trend + EMA context |
| 1-hour | 4 years | Last 48 bars summary | Swing structure |
| 15-min | 4 years | Last 32 bars summary | Entry zone timing |
| 1-min | 4 years | Last 60 bars summary | Precise entry |
| Quarterly | 4 years | Last 4 quarters (1 line) | Fundamental health |

The 4-year raw history is on disk for computing indicators and backtesting SL/TP.
The model prompt gets a **compressed multi-timeframe summary** — never raw bars.

---

## Hardware guide

### Apple M5 Air 16GB (your setup)

```bash
# Install MLX (Apple's native ML framework — faster than PyTorch MPS)
pip install mlx-lm

# Generate data (Ollama uses MLX under the hood — already fast)
# Configure in trading_agent_sim.ipynb:
OLLAMA_MODEL = "nemotron-bolt-e4b-16k"   # fits in ~3 GB
FALLBACK_MODE = False

# Fine-tune (run overnight with external cooling)
python train_mlx.py \
    --train data/trading_train.jsonl \
    --model mlx-community/Nemotron-Nano-4B-Instruct-4bit \
    --iters 600 --batch_size 4 --lora_layers 8

# Estimated time on M5 Air: 6-10 hours for 500 examples, 600 iters
```

**Important:** `bitsandbytes` (used by `train_qlora.py`) is CUDA-only and will not run on
Apple Silicon. Always use `train_mlx.py` on M-series Macs.

### Model fit in 16 GB unified memory

| Model | Size (4-bit) | Inference | Train on Mac |
|---|---|---|---|
| nemotron-nano-4b | ~2.5 GB | ✅ | ✅ |
| nemotron-bolt-e4b | ~3.0 GB | ✅ | ✅ |
| qwen3:8b | ~5.0 GB | ✅ | ✅ |
| qwen3:14b | ~9.0 GB | ✅ | ⚠️ tight |
| qwen3:30b | ~18 GB | ❌ | ❌ |

### Checkpointing (train in intervals)

MLX saves checkpoints every `--save-every N` steps. Resume with:

```bash
# Session 1 (run for 30-40 min)
python train_mlx.py --iters 150 --save-every 50

# Session 2 (next day — resume from checkpoint)
python train_mlx.py --iters 150 --save-every 50 \
    --resume-adapter-file output/nemotron-mlx/adapters/0000150_adapters.safetensors

# 4-6 sessions × ~40 min = full training spread across days
```

### Google Colab A100 Pro (fastest option)

```python
# Upload trading_agent.db then run in Colab:
!pip install unsloth
!python data_pipeline.py --db trading_agent.db --out data/trading.jsonl
!python train_qlora.py \
    --train data/trading_train.jsonl \
    --val   data/trading_val.jsonl \
    --epochs 3 --lora_r 16
# Download output/nemotron-trading/final_adapter/ (~60 MB)
# Total time: ~2 hours. Cost: ~$3 on A100 Pro
```

---

## Teacher / student strategy (data quality)

Do **not** use the same 4B model to generate training data for itself — this causes
*model collapse* (each round's slightly biased outputs amplify the bias).

| Role | Model | Job |
|---|---|---|
| **Teacher** (generate data) | `nemotron-bolt-e4b-16k` or `qwen3:8b` | Rich reasoning, runs in ~3-9 GB |
| **Student** (to fine-tune) | `Nemotron-Nano-4B-Instruct` | Small, fast local inference |

For the very first round, use `FALLBACK_MODE = True` (rule-based, zero hallucination,
perfectly consistent format). Switch to the teacher model once you have 50+ episodes.

---

## Minimum data requirements

| Requirement | Target | Why |
|---|---|---|
| Total episodes | ≥ 500 | Luck averages out; genuine patterns survive |
| Symbols | ≥ 10 NSE stocks | Different risk profiles, sectors, volatility |
| Date ranges | ≥ 3 market regimes | Bull, bear, sideways — model must generalise |
| Quality-filtered samples | ≥ 400 train + 50 val | Rule of thumb for instruction fine-tuning |
| Win / Loss balance | 40-60% each | Prevents model learning "always HOLD" |

---

## Token optimisation summary

| Section | Before | After | Note |
|---|---|---|---|
| System prompt | ~250 tok | ~55 tok | Schema known after 50 examples |
| 5 raw OHLCV candles | ~180 tok | ~50 tok | Derived metrics (ATR, momentum, %range pos) carry more signal |
| S/R zones as JSON | ~120 tok | ~35 tok | `116.5-117.2(-2.5%)` vs full object |
| EMA/VWAP labels | ~60 tok | ~25 tok | Single-line `50@118.9 < 100@120.1 [BEARISH]` |
| Trade history (5 ep) | ~250 tok | ~90 tok | Table format; lesson capped at 60 chars |
| Fundamentals | ~20 tok | ~30 tok | Zero-value placeholder → real 1-line summary |
| **Total** | **~990 tok** | **~188 tok** | **82% reduction** |

Use `compact_prompt.py` as a drop-in for the verbose prompt builders in the notebook.

---

## Visualisation tools

```bash
# Interactive trading session dashboard (candles + VWAP + decisions + capital curve)
python generate_viz.py --db trading_agent.db --out viz/dashboard.html
open viz/dashboard.html

# Training data viewer (see exactly what each sample looks like before training)
python show_training_data.py --db trading_agent.db --out viz/training_data.html
open viz/training_data.html
```

Both generate **self-contained HTML files** — no server needed, open directly in a browser.

---

## File reference

| File | Purpose |
|---|---|
| `data_pipeline.py` | SQLite → quality-filtered JSONL (pre-trade + post-trade pairs) |
| `compact_prompt.py` | Optimised prompt builder — 82% fewer tokens, same information |
| `train_mlx.py` | **Apple Silicon** LoRA training via MLX (M1/M2/M3/M4/M5) |
| `train_qlora.py` | **NVIDIA GPU** QLoRA training via Unsloth + SFTTrainer |
| `flywheel.py` | Integrated loop: simulate → build dataset → fine-tune → repeat |
| `evaluate.py` | JSON validity, schema accuracy, reasoning quality metrics |
| `inference.py` | Drop-in for Ollama (CUDA / PyTorch) |
| `inference_mlx.py` | Drop-in for Ollama (Apple Silicon / MLX) |
| `generate_viz.py` | Interactive HTML dashboard: candles, VWAP, decisions, capital curve |
| `show_training_data.py` | Interactive HTML viewer: browse every training sample with syntax highlight |

---

## Pros and cons

| | |
|---|---|
| ✅ Domain-specific | Learns Indian market vocabulary, NSE patterns, GNPA/NII/NIM, S/R zones |
| ✅ Learns from failure | Post-trade analysis pairs teach "what I got wrong" — rare in LLM training |
| ✅ Structured output | Model learns your exact JSON schema reliably after ~50 examples |
| ✅ Small footprint | 4B model + LoRA adapters ≈ 3 GB — runs on 16 GB Mac with room to spare |
| ✅ 82% token reduction | `compact_prompt.py` cuts training cost 5× with no quality loss |
| ✅ Checkpointable | Resume training across sessions; no 8-hour continuous run required |
| ❌ Data starvation | 15 episodes is nowhere near enough. Run 500+ across 10+ symbols |
| ❌ No ground truth labels | We label by outcome, not by whether reasoning was logically correct |
| ❌ Distribution shift | Fine-tune on 2024, deploy in 2026 — periodic retraining required |
| ❌ Training slow on Mac | ~7-10 hours for 500 examples on M5 Air (run overnight in sessions) |
# finetune_llm
