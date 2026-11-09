# Trading Agent LLM Fine-Tuning Pipeline — Complete Reference

> **Purpose:** This document is a comprehensive reference for the Trading Agent LLM Fine-Tuning
> project. It covers the simulation, data pipeline, LoRA/QLoRA training, production deployment,
> and all frameworks used. Designed to be fed as a source into NotebookLM or any LLM for Q&A.

---

## 1. Project Overview

### What we built

A two-part system:

**Part A — Trading Agent Simulation** (`trading_agent_sim.ipynb`)
A reinforcement-style simulation where an LLM agent:
- Receives 30 days of historical Indian NSE stock data up to a fixed date (`2024-10-10`)
- Sees computed technical indicators (EMA, VWAP, S/R zones, HH/HL/LH/LL, ATR)
- Reads quarterly fundamentals (PAT, GNPA, NII, P/E) from yfinance filtered to `as_of` date
- Decides: BUY, SELL, or HOLD with exact entry, stop-loss, and take-profit levels
- Gets evaluated against real forward prices (SL hit = loss, TP hit = profit)
- Writes a post-mortem in the same turn explaining what happened
- Carries full trade history as context for future decisions

**Part B — LLM Fine-Tuning Pipeline** (`llm_finetune/`)
Takes the simulation data and fine-tunes `nvidia/Nemotron-Nano-4B-Instruct` to be a
domain-specific trading specialist. After fine-tuning, the model replaces the base Ollama call
in the simulation, improving decision quality each round.

### The Flywheel

```
Round 1: Rule-based fallback    → 50 episodes → fine-tune → Nano-4B V1
Round 2: Nano-4B V1             → 50 episodes → fine-tune → Nano-4B V2
Round 3: Nano-4B V2             → 50 episodes → fine-tune → Nano-4B V3
```
Each round generates better data (smarter decisions), which trains a better model.
The outcome labels (SL hit / TP hit) come from real historical prices — the model
cannot hallucinate its way to a profit.

---

## 2. Data Sources and Storage

### Data sources used

| Source | Data | How accessed | Strict cutoff |
|---|---|---|---|
| yfinance | OHLCV candles (all timeframes) | `yf.download(end=as_of_date)` | Yes — `end=` param |
| yfinance | Quarterly P&L, balance sheet | `tk.quarterly_financials` filtered | Yes — filter `<= as_of` |
| SQLite | Trade log + fundamentals cache | `trading_agent.db` | N/A |
| Polars + DuckDB | 4-year 1-min candle archive | Parquet shards | Yes — SQL `WHERE ts <= ?` |

### Storage design

**SQLite (`trading_agent.db`):**
- `trade_log` — every episode: entry, SL, TP, exit reason, P&L, LLM reasoning, post-mortem
- `fundamental_cache` — quarterly P&L cached by `(symbol, quarter_key)`. Never fetched twice for the same quarter.

**Parquet (`data/candles/SYMBOL/TIMEFRAME/YYYY-MM.parquet`):**
- Monthly shards per symbol per timeframe
- DuckDB for date-range queries — scans only the months needed, not full files
- 8× smaller than CSV, 40× faster date-range reads

### Lookback windows given to the model

The model never sees raw multi-year data. It receives compressed summaries:
- Daily (1D): last 60 bars — trend + EMA context
- 1-hour: last 48 bars — swing structure
- 15-min: last 32 bars — entry zone timing
- 1-min: last 60 bars — precise entry
- Quarterly: last 4 quarters (one line summary)
- Trade history: last 5 episodes with lessons

---

## 3. The Simulation in Detail

### Episode structure

Each episode (one trading day simulation):

1. **Fetch candles** — yfinance daily OHLCV strictly up to `as_of_date`
2. **Compute technicals** — all in pure Python/NumPy (not LLM):
   - EMA 50, 100, 200 (exponential moving averages)
   - VWAP (volume-weighted average price, cumulative from first bar)
   - Swing pivots — detect local highs/lows with `order=2` window
   - HH/HL/LH/LL — classify the last 3 swing pairs as trend structure
   - Support/resistance zones — cluster nearby pivot levels within 0.8% bandwidth
3. **Get fundamentals** — yfinance quarterly P&L filtered to `as_of_date`, cached in SQLite
4. **Pre-trade LLM call** — agent receives compact market summary + history, returns JSON
5. **Check SL/TP** — scan real forward candles (not shown to agent):
   - BUY: Low ≤ SL → SL_HIT | High ≥ TP → TP_HIT
   - Same-bar conflict: SL wins (conservative)
   - No hit after 5 bars: TIME_EXIT at close
6. **Post-trade LLM call** (same episode) — agent analyses outcome, extracts lesson
7. **Persist** — save to SQLite with all reasoning
8. **Advance date** — skip weekends, move to next trading day

### Capital rules

- Start: ₹10,000
- Death threshold: capital < ₹2,000 (20% of initial) → simulation stops
- Break-even: capital ≥ ₹10,000 → logged as milestone
- Risk per trade: 2% of capital maximum (enforced in prompt)

### Compact prompt format (82% token reduction)

**Before (verbose): ~990 tokens**
```
=== MARKET DATA: CANBK as of 2024-10-10 ===
Last 5 candles (Date | O H L C V):
  2024-10-06 | O:121.20 H:123.50 L:120.10 C:121.80 V:4,200,000
  ...5 rows...
```

**After (compact): ~188 tokens**
```
[CANBK 2024-10-10]
C:119.78 (+1.9%) | 5d:116.8-123.5 C@44pct
ATR:3.26 | Vol:4.8M(1.0×avg) | 3d:-0.35%
EMA: 50@118.9 < 100@120.1 < 200@122.5 [BEARISH]
VWAP:119.2 C+0.49%
S:116.5-117.2(-2.5%)  R:122.0-123.5(+2.5%)
Fund(2024-Q2): PAT ₹5381cr | P/E 9.2
HISTORY:
01  BUY  SL_HIT   -52   9948  wait for EMA alignment
CAP:9948 (death<2000)
```

**Why same quality:** Derived metrics (`C@44pct` = close at 44th percentile of 5-day range) carry
more decision signal than raw OHLCV. `S:116.5-117.2(-2.5%)` gives zone + distance in one token.
The 82% reduction means 1.5h training time instead of 8h on Apple M5 Air.

---

## 4. Training Data Format

### Each episode generates two training samples

**Sample 1 — Pre-trade decision:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a disciplined NSE equity trader..."},
    {"role": "user",   "content": "[CANBK 2024-10-10]\nC:119.78 ..."},
    {"role": "assistant", "content": "{\"action\":\"HOLD\",\"quantity\":0,...}"}
  ]
}
```

**Sample 2 — Post-trade analysis (same episode):**
```json
{
  "messages": [
    {"role": "system", "content": "Review the completed trade, extract one lesson..."},
    {"role": "user",   "content": "[CANBK Ep01]\nBUY 16@119.78 SL:116.5 TP:124.0\nEXIT:116.5 [SL_HIT]..."},
    {"role": "assistant", "content": "{\"outcome\":\"loss\",\"lesson_learned\":\"wait for EMA alignment\"}"}
  ]
}
```

### Quality filters applied in `data_pipeline.py`

- Action must be BUY or SELL (HOLD samples dropped — no entry learning signal)
- Confidence ≥ 0.5
- Risk/Reward ≥ 1.3
- Lesson must not be "—" or the fallback string

### Dataset split

- 80% train, 10% validation, 10% test
- Random shuffle with fixed seed for reproducibility
- DVC tracks every dataset version

### Minimum data before training

- ≥ 500 episodes total across multiple symbols and date ranges
- ≥ 10 NSE symbols (different risk profiles)
- ≥ 3 market regimes (bull, bear, sideways)
- ≥ 400 quality-filtered train examples + 50 val
- 40–60% win rate balance (prevents "always HOLD" collapse)

---

## 5. The Transformer Architecture (What We Modify)

### Nemotron-Nano-4B architecture

Nemotron-Nano-4B is built on the **Mistral architecture** with:
- 4,096 hidden dimension (d_model)
- 32 transformer blocks
- 32 attention heads (head dimension = 128)
- FFN intermediate dimension = 14,336
- Vocabulary size = 32,000 tokens
- Context window = 32,768 tokens
- Total parameters ≈ 4 billion

### Each transformer block contains

```
RMSNorm
Multi-Head Self-Attention
  ├── q_proj  (4096 × 4096)  "What am I looking for?"
  ├── k_proj  (4096 × 4096)  "What do I have to offer?"
  ├── v_proj  (4096 × 4096)  "What is my actual content?"
  └── o_proj  (4096 × 4096)  "Combine the attended values"
RMSNorm
Feed-Forward Network (MLP)
  ├── gate_proj  (4096 × 14336)  gating mechanism (SwiGLU)
  ├── up_proj    (4096 × 14336)  expand to FFN dimension
  └── down_proj  (14336 × 4096)  compress back
```

**LoRA adds adapter matrices to all 7 projection layers × 32 blocks = 224 LoRA pairs total.**

### Why Q, K, V projections matter for domain specialization

- **Q_proj** (Query): controls what the model "looks for" in the context. Fine-tuning Q teaches the model to search for "GNPA", "EMA stack", "support zone" patterns in trading contexts.
- **K_proj** (Key): controls what information "presents itself" as relevant. Fine-tuning K teaches it to flag trading-specific signals.
- **V_proj** (Value): what content flows forward when attention is paid. Fine-tuning V encodes trading knowledge.
- **O_proj** (Output): combines the attended values. Fine-tuning O shapes how signals combine into decisions.
- **gate/up/down** (FFN): stores factual knowledge. Fine-tuning here teaches "GNPA 1.33% = clean book", "EMA50 < EMA200 = bearish".

---

## 6. LoRA — Low-Rank Adaptation

### The core mathematical insight

When fine-tuning a pre-trained model on a specific task, the **weight change matrix ΔW has
low intrinsic rank**. The task-specific information can be captured by a much smaller matrix.

**Standard fine-tuning:**
```
output = (W + ΔW) × input
```
where ΔW is the same shape as W (e.g., 4096 × 4096 = 16.7M parameters per layer).

**LoRA decomposition:**
```
output = W × input + (B × A) × input × (alpha/r)

where:
  W  is frozen (not trained)
  A  ∈ R^(r × d_in)   — small matrix, random Gaussian init
  B  ∈ R^(d_out × r)  — small matrix, zero init (so ΔW = 0 at start)
  r  = rank (e.g., 16)
  alpha = scaling factor (typically 2×r = 32)
```

For one 4096×4096 layer with r=16:
- Original W: 4096 × 4096 = **16,777,216 parameters**
- LoRA A: 16 × 4096 = 65,536 parameters
- LoRA B: 4096 × 16 = 65,536 parameters
- **Total LoRA: 131,072 parameters = 0.78% of the original layer**

Across all 7 projection types × 32 blocks: ~40M trainable parameters vs 4B total = **1% of the model**.

### PEFT library implementation

`peft` (HuggingFace) wraps each target layer with a `LoraLinear` module:

```python
class LoraLinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: float):
        self.W = original          # frozen, requires_grad=False
        self.A = nn.Linear(d_in, r, bias=False)   # trainable
        self.B = nn.Linear(r, d_out, bias=False)  # trainable
        self.scale = alpha / r

        nn.init.kaiming_uniform_(self.A.weight)  # random init for A
        nn.init.zeros_(self.B.weight)            # zero init for B

    def forward(self, x):
        # Original path + adapter path
        return self.W(x) + self.B(self.A(x)) * self.scale
```

**Key PEFT function:** `get_peft_model(model, lora_config)` walks the entire model graph,
replaces every matching `nn.Linear` with `LoraLinear`, and freezes everything else.
`model.print_trainable_parameters()` shows the before/after.

### Why alpha = 2 × r

`alpha/r` is the effective learning rate multiplier for the adapter. At `alpha = 2r`, this
multiplier = 2.0. Empirically this is the sweet spot across most fine-tuning tasks — the
adapters learn fast enough to converge but not so fast they diverge. You'll see `alpha = 2r`
in almost every production LoRA config.

### LoRA rank selection guide

| Rank r | Use when | Why |
|---|---|---|
| 4–8 | < 200 training examples | Small data = simple task = low-rank update |
| 16 | 200–2,000 examples | Standard choice, good balance |
| 32 | 2,000–10,000 examples | More capacity needed |
| 64+ | > 10,000 examples | Approaches full fine-tune quality |

Our dataset (~500 quality examples): **r=16 is correct**.

### Merging the adapter back into the model

After training, you can merge A and B back into W:
```
W_merged = W_original + (B × A) × (alpha/r)
```
This is computed once and stored as a single matrix. At inference:
- Adapter-only: two extra matrix multiplies per layer (slight overhead)
- Merged model: single matrix multiply, same as original (zero overhead)

Merging is done with `model.merge_and_unload()` in PEFT.

---

## 7. QLoRA — Quantized LoRA

### Why LoRA alone isn't enough on limited hardware

Even with LoRA (only 1% of params trainable), the frozen base model must still be loaded
into memory to do forward passes. At bfloat16, that's:
- 4B parameters × 2 bytes = **8 GB just for the frozen model**
- Plus optimizer states for adapters (~320 MB)
- Plus activations during forward pass (~3-4 GB)
- Total: ~12-13 GB — exceeds many consumer GPUs and Apple M5 Air's usable ML memory

QLoRA solves this by **compressing the frozen base to 4-bit** before training.

### NF4 (Normal Float 4) quantization

Standard 4-bit quantization stores weights as integers 0–15 with uniform spacing.
NF4 is better because **LLM weights follow a roughly Gaussian (normal) distribution**.

NF4's 16 quantization levels are spaced non-uniformly to minimize reconstruction error
for Gaussian-distributed values:
```
NF4 levels: [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
              0.080,  0.161,  0.246,  0.338,  0.441,  0.563,  0.723, 1.0]
```
Notice more levels near 0 (where most weights cluster) and fewer at the extremes.

**Compression ratio:** 32-bit float → 4-bit index = **8× memory reduction**.
**Quality cost:** RMSE ≈ 0.001 for typical LLM weights ≈ **~2% downstream quality loss**.

### Double quantization

The quantization constant (one float32 per 64 weights = 0.5 bytes/weight extra overhead)
is itself quantized to 8-bit. This saves an additional ~0.37 bytes per parameter.

### The QLoRA forward pass

```
Storage (NF4, 0.5 bytes/param):
  W = [4-bit integer indices]     ← on GPU memory

Forward pass:
  W_dequant = lookup(W_indices) × scale    ← dequantize to bfloat16
  output = W_dequant × input               ← matrix multiply in bfloat16
         + B × A × input × scale_lora      ← LoRA adapter in bfloat16

Backward pass:
  ∂loss/∂B and ∂loss/∂A computed          ← only adapters accumulate gradients
  W_dequant used for gradient chain rule   ← passes through but not updated
```

The base weights **never get updated** — they're just read and dequantized during forward.

### bitsandbytes library

`bitsandbytes` provides:
- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`
- `prepare_model_for_kbit_training(model)` — sets up layernorms in fp32, enables gradient checkpointing
- `adamw_bnb_8bit` — 8-bit Adam optimizer (halves optimizer memory: 320 MB → 160 MB)

**Important:** bitsandbytes is **CUDA-only** — it will not work on Apple Silicon MPS.
On Mac, use MLX-LM instead (see Section 9).

### Memory comparison

| Method | Model | Optimizer | Grads | Activations | Total |
|---|---|---|---|---|---|
| Full fine-tune (fp32) | 16 GB | 32 GB | 16 GB | 6 GB | ~70 GB |
| LoRA (bfloat16 base) | 8 GB | 0.32 GB | 0.16 GB | 4 GB | ~13 GB |
| QLoRA 8-bit | 4 GB | 0.32 GB | 0.16 GB | 3 GB | ~8 GB |
| QLoRA 4-bit NF4 | 2 GB | 0.32 GB | 0.16 GB | 2 GB | ~5 GB |

At 500 training examples, the quality difference between QLoRA 4-bit and full fine-tuning
is negligible — you need 100,000+ examples before it matters.

---

## 8. The Training Loop (Step by Step)

### Libraries used

- `transformers` (HuggingFace) — model + tokenizer loading
- `peft` — LoRA wrapper
- `trl` — SFTTrainer (Supervised Fine-Tuning Trainer)
- `bitsandbytes` — 4-bit quantization + 8-bit Adam (NVIDIA only)
- `datasets` (HuggingFace) — JSONL loading
- `mlx-lm` (Apple) — alternative full stack for Apple Silicon

### Step 1: Tokenization

The tokenizer converts text to integer token IDs using Byte-Pair Encoding (BPE):
```
"EMA50 < EMA100" → [EMA, 50, <, EMA, 100] → [11120, 2135, 529, 27100, 2753]
```

Nemotron uses the **ChatML template**:
```
<|im_start|>system
You are a disciplined NSE equity trader...<|im_end|>
<|im_start|>user
[CANBK 2024-10-10]...<|im_end|>
<|im_start|>assistant
{"action":"HOLD",...}<|im_end|>
```

Special tokens: `<|im_start|>` = 32002, `<|im_end|>` = 32003.

### Step 2: The loss mask (critical detail)

`DataCollatorForCompletionOnlyLM` (from TRL) sets loss = 0 for all system and user tokens.
Only the **assistant turn tokens** contribute to the cross-entropy loss.

```
Input:  [system tokens] [user tokens] [assistant tokens]
Loss:   [     0      ] [     0     ] [        1       ]
```

This teaches the model **what to decide**, not to memorise the input market data.
This is also why compact prompts are better: 65% of a 188-token example is assistant
output (the learning signal), vs only 7% for a 990-token verbose prompt.

### Step 3: Forward pass

The tokenized batch flows through all 32 transformer blocks:

```
For each block:
  x = RMSNorm(x)
  q = q_proj(x)  →  actually: q = W_q × x + (B_q × A_q × x) × scale
  k = k_proj(x)  →  actually: k = W_k × x + (B_k × A_k × x) × scale
  v = v_proj(x)  →  actually: v = W_v × x + (B_v × A_v × x) × scale
  attn = softmax(q × k^T / sqrt(d_head)) × v
  x = x + o_proj(attn)
  x = x + FFN(RMSNorm(x))
```

At the end, the LM head projects to vocabulary logits (probability of each next token).

### Step 4: Loss computation

For each assistant token position:
```
loss_i = -log P(correct_token_i | all_previous_tokens)
```

The cross-entropy loss penalizes the model for getting the JSON field values wrong.
If the model predicts `"BUY"` when the label is `"HOLD"`, the loss for that token is high.

Total batch loss = mean over all unmasked (assistant) positions.

### Step 5: Backward pass (backpropagation)

PyTorch's autograd computes gradients for all parameters with `requires_grad=True`.
Since only LoRA A and B have `requires_grad=True`, only those 40M parameters accumulate gradients.

Gradients still **pass through** the frozen base weights during the chain rule computation —
they just don't accumulate into those weights. The frozen weights participate in the gradient
computation graph but are never updated.

### Step 6: Adam optimizer step

For each trainable parameter θ (LoRA A and B only):
```
g  = ∂loss/∂θ                           gradient from backward pass
m  = β₁ × m  + (1-β₁) × g             update first moment (β₁=0.9)
v  = β₂ × v  + (1-β₂) × g²            update second moment (β₂=0.999)
m̂  = m / (1-β₁^t)                      bias correction
v̂  = v / (1-β₂^t)
θ  = θ - lr × m̂ / (sqrt(v̂) + ε)        update parameter (lr=2e-4)
```

`adamw_bnb_8bit` stores m and v in 8-bit (halves optimizer memory) with negligible quality loss.

### Step 7: Gradient accumulation

With `batch_size=2` and `grad_accum=8`:
1. Run forward+backward on 2 examples, accumulate (don't zero) gradients
2. Repeat 8 times (16 examples total processed)
3. Do one optimizer step
4. Zero gradients

**Effective batch = 16 examples** but only 2×-worth of activations in memory at once.
This is how we simulate large batches on limited hardware.

### Gradient checkpointing

Normally all intermediate activations are stored during forward pass for use in backpropagation.
With gradient checkpointing (`gradient_checkpointing=True`):
- Activations are **not stored** during forward pass
- During backward pass, activations are **recomputed** on-the-fly
- Memory cost: ~4 GB → ~2 GB
- Speed cost: ~15-20% slower (worth it on limited hardware)

---

## 9. Apple Silicon Path (MLX-LM)

### Why not PyTorch + bitsandbytes on Mac

`bitsandbytes` relies on CUDA-specific operations for 4-bit quantization. It does not
support Apple Metal Performance Shaders (MPS). `train.py` never calls bitsandbytes on
Apple Silicon — it routes to the MLX path automatically.

### MLX-LM training

Apple built MLX specifically for M-series chips. Key differences:
- Uses the Neural Engine + GPU cores natively
- Unified memory pool (CPU and GPU share the same physical memory)
- MLX handles its own quantization format (different from NF4 but similar concept)
- 2-4× faster than PyTorch MPS for LLM operations

**Command:**
```bash
python -m mlx_lm.lora \
  --model mlx-community/Nemotron-Nano-4B-Instruct-4bit \
  --train \
  --data data/mlx_data/ \
  --adapter-path output/adapters \
  --num-layers 8 \
  --iters 600 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --grad-checkpoint
```

**Speed on M5 Air 16GB:**
- 500 examples × 600 iterations ≈ 5-8 hours
- With compact prompts (188 tokens): ~1.5-2 hours
- Run overnight in 4 sessions of 150 iterations each

**Checkpointing:** saves `0000150_adapters.safetensors`, `0000300_adapters.safetensors`, etc.
Resume with `--resume-adapter-file output/adapters/0000150_adapters.safetensors`.

---

## 10. Hardware Decision Matrix

### Auto-detection in `train.py`

```
detect_hardware() →
  Apple Silicon (arm64 + Darwin)?  →  MLX LoRA
  NVIDIA CUDA available?
    VRAM >= 16 GB?  →  HF LoRA (bfloat16, no quantization)
    VRAM 8-16 GB?   →  HF QLoRA 8-bit
    VRAM < 8 GB?    →  HF QLoRA 4-bit NF4
  CPU only?         →  HF LoRA (slow, emergency fallback)
```

### Method comparison

| Method | VRAM needed | Training time (500 ex) | Quality vs full FT |
|---|---|---|---|
| MLX LoRA | 4-6 GB unified | 5-8h M5 Air (1.5h compact) | ~98% |
| HF LoRA bfloat16 | 12-14 GB | ~2h A100 | ~99% |
| HF QLoRA 8-bit | 8-10 GB | ~2.5h A10G | ~98% |
| HF QLoRA 4-bit | 4-6 GB | ~3h Colab T4 | ~97% |
| Full fine-tune | 60+ GB | 12+h A100 | 100% |

---

## 11. Axolotl — Production Fine-Tuning

Axolotl is the industry standard fine-tuning framework. Used at Mistral, Together.ai, and
most LLM fine-tuning teams. It replaces `train_qlora.py` with a single YAML config:

```yaml
base_model: nvidia/Nemotron-Nano-4B-Instruct
adapter: qlora
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
datasets:
  - path: data/trading_train.jsonl
    type: chat_template
    chat_template: chatml
train_on_inputs: false   # critical: only train on assistant turns
sequence_len: 1024
micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4
num_epochs: 3
report_to: [mlflow]
```

Run with: `axolotl train axolotl_config.yaml`

---

## 12. MLflow — Experiment Tracking

### What MLflow tracks

Every `train.py` run logs to MLflow automatically when `MLFLOW_TRACKING_URI` is set:

**Parameters (logged once per run):**
- `method`, `model`, `lora_r`, `epochs`, `learning_rate`
- `n_train` (number of training examples)
- `hardware` type, `vram_gb`
- `token_strategy` (compact vs verbose)

**Metrics (logged every N steps):**
- `train_loss`, `val_loss`
- `json_validity` (does model output parseable JSON?)
- `schema_accuracy` (are all required fields present?)
- `training_time_h`

**Artifacts:**
- The trained adapter weights (`adapter_model.safetensors`)
- Per-stock scores CSV from the stock scoring pipeline

### Model registry

```
nemotron-trading
  Version 1 → Round 1 (val_loss=0.68, json_validity=0.72)
  Version 2 → Round 2 (val_loss=0.54, json_validity=0.85)
  Version 3 → Round 3 (val_loss=0.44, json_validity=0.95) ← Production
```

### MLflow integration in the stock scoring project

`app/graph/runner.py` logs every scoring run to MLflow after completion:
- Parameters: n_stocks, as_of date, model config, weights version
- Metrics: n_valid, completeness_pct, avg_score, wall_time_s
- Artifacts: per-stock scores CSV

Graceful — if MLflow is not installed or server is down, it silently skips and the scoring
run completes normally.

---

## 13. vLLM — Production Inference

### Why vLLM instead of Ollama

Ollama is designed for development (single-user, serialized requests). vLLM is designed for
production (concurrent requests, optimal GPU utilization).

| | Ollama | vLLM |
|---|---|---|
| Concurrency | Serialized (queue) | Continuous batching |
| Memory management | Simple | PagedAttention (near-zero waste) |
| Throughput | 10-20 tokens/sec | 50-200 tokens/sec |
| API | Ollama format | OpenAI-compatible |
| Use case | Local dev | Production serving |

**PagedAttention:** vLLM manages the KV cache like a memory pager — allocates pages on
demand instead of pre-allocating worst-case for every request. Result: 2-4× higher throughput
than naive serving.

### Integration in the stock scoring project

`app/service/models/llm_factory.py` supports `provider: "vllm"` alongside ollama/openai/olmx.
Since vLLM exposes an OpenAI-compatible API, the switch is just a URL change:

```json
{
  "company":   {"provider": "vllm", "model": "nemotron-trading", "temperature": 0.2},
  "technical": {"provider": "vllm", "model": "nemotron-trading", "temperature": 0.15}
}
```

Set `VLLM_BASE_URL=http://localhost:8001` in `.env`.

---

## 14. Prefect — Pipeline Orchestration

### Why Prefect over cron

Cron runs a command on a schedule. Prefect:
- Retries failed tasks with backoff
- Logs every run with status, duration, parameters, errors
- Provides a web UI at http://localhost:4200
- Supports conditional logic (skip training if < 10 examples)
- Deployable to cloud for team visibility

### Scheduled flows in `prefect_flows.py`

**`daily_scoring_run`** — every weekday at 9:30 AM IST:
- Fetches the configured universe (98 NSE stocks)
- Runs the full scoring pipeline
- Logs to MLflow

**`weekly_flywheel`** — every Sunday at 2 AM:
- Runs N simulation episodes (data generation)
- Builds quality-filtered JSONL dataset
- Fine-tunes the model (Axolotl or MLX-LM)
- Evaluates the new adapter
- Logs all metrics to MLflow

---

## 15. DVC — Data Version Control

### Why DVC alongside git

Git tracks code changes. DVC tracks:
- The 4-year Parquet candle files (too large for git)
- The SQLite `trading_agent.db` (binary, changes every run)
- The trained adapter weights
- The JSONL training datasets

Every git commit has an associated DVC commit — giving you:
- Exact code that generated each dataset
- Exact dataset that trained each model
- Full reproduction of any historical run

### `dvc.yaml` pipeline stages

```
fetch_candles → simulate → build_dataset → finetune → evaluate
```

Running `dvc repro`:
1. Checks which stages need to re-run (based on changed inputs/params)
2. Runs only the necessary stages
3. Change `lora_r` in `params.yaml` → only `finetune` and `evaluate` re-run

### `params.yaml` as central config

All hyperparameters live in `params.yaml`. `train.py` reads this file at startup.
`dvc repro` tracks changes to `params.yaml` as experiment triggers.

---

## 16. Evidently AI — Model Monitoring

### The problem: model drift

After fine-tuning on Oct 2024 data, the model is deployed. Over time:
- Market regime changes (rate hike cycle → rate cut cycle)
- The model's decision distribution shifts ("always BUY" in bull market, then losses in bear)
- Without monitoring, you don't know when to retrain

### What `monitoring.py` tracks

Compares reference dataset (training distribution) vs current live decisions:
- **Action distribution** — has BUY/SELL/HOLD ratio shifted?
- **Confidence** — is the model less confident than during training?
- **Risk/Reward** — is it accepting worse R:R setups?

`report.run(reference_data=training_df, current_data=live_df)` produces:
- HTML drift report
- `dataset_drift` flag (True/False)
- Number of drifted columns

Alerts when `action_drifted=True` — this is the signal to trigger a new flywheel round.

---

## 17. Polars + DuckDB — Fast Data Processing

### Why Polars over pandas

`pandas` is single-threaded and reads entire files into memory.
`polars` is multi-threaded with lazy evaluation — processes only what's needed.

For 4-year 1-min candle data (393,000 rows × 10 symbols = 3.9M rows):
- pandas: 3-5 seconds to load, 8 GB memory
- polars lazy: 0.08 seconds to query, ~200 MB memory

Key polars API used:
```python
df = pl.scan_parquet("data/candles/CANBK/1m/*.parquet")  # lazy — nothing read yet
result = df.filter(pl.col("ts") > "2024-01-01").tail(60).collect()  # only read what's needed
```

### Why DuckDB for analytical queries

DuckDB reads only the Parquet row groups (blocks) that match the WHERE clause.
No need to load the entire file:

```sql
SELECT avg(close), count(*) 
FROM read_parquet('data/candles/CANBK/1m/*.parquet')
WHERE ts >= '2024-01-01' AND ts <= '2024-10-10'
```

This scans ~30 days of 1-min data instead of 4 years — 50× faster than loading all files.

---

## 18. Teacher/Student Strategy

### The model collapse problem

If we use Nemotron-Nano-4B to generate its own training data and then fine-tune it on that data:
- Generation 1: Model makes slightly biased decisions
- Fine-tune on biased data: model becomes more biased
- Generation 2: Model generates more biased data
- Result: "model collapse" — model converges to a narrow set of outputs

This is documented in research as a known problem with recursive self-distillation.

### Our solution

| Role | Model | Purpose |
|---|---|---|
| **Teacher** | nemotron-bolt-e4b-16k or qwen3:8b | Generates training data. Runs in Ollama. Better reasoning quality. |
| **Student** | Nemotron-Nano-4B-Instruct | Gets fine-tuned on teacher's data. Runs locally, fast inference. |

For the very first round: use `FALLBACK_MODE=True` (rule-based, zero hallucination, consistent format).

### Why this works even though outcome labels come from the market

The model collapse concern is primarily about **labels** being generated by the model.
In our case:
- The LLM generates: reasoning text, decision, confidence
- The **outcome labels** (SL hit / TP hit / P&L) come from real historical prices
- Real prices act as an external anchor — the model cannot hallucinate a winning trade

The post-trade analysis is generated by the teacher model, so even if it's slightly biased,
the signal that the trade actually lost money (from real prices) overrides that bias.

---

## 19. Complete File Reference

| File | Library/Tool | Purpose |
|---|---|---|
| `trading_agent_sim.ipynb` | yfinance, numpy, pandas | Simulation: runs episodes, stores to SQLite |
| `data_pipeline.py` | Polars, sqlite3 | SQLite → quality-filtered JSONL |
| `compact_prompt.py` | Python | 82% token reduction — replaces verbose prompts |
| `candle_store.py` | Polars, DuckDB | 4-year 1-min OHLCV storage and query |
| `train.py` | PEFT, TRL, bitsandbytes, MLX-LM | **Auto-detect hardware → LoRA or QLoRA** |
| `axolotl_config.yaml` | Axolotl | Production fine-tuning (one YAML) |
| `inference.py` | transformers, peft | Drop-in for Ollama (PyTorch) |
| `inference_mlx.py` | mlx-lm | Drop-in for Ollama (Apple Silicon) |
| `inference_vllm.py` | vllm, openai | Production batch inference |
| `flywheel.py` | papermill | Automated data → train → deploy loop |
| `prefect_flows.py` | Prefect | Scheduled daily scoring + weekly fine-tuning |
| `evaluate.py` | transformers, peft | JSON validity, schema accuracy, reasoning quality |
| `monitoring.py` | Evidently AI | Model drift detection |
| `dvc.yaml` | DVC | Data + model pipeline with reproducibility |
| `params.yaml` | PyYAML | Central hyperparameter store |
| `requirements.txt` | pip | All dependencies with platform notes |
| `scripts/fetch_candles.py` | yfinance, Polars | Fetch and store historical candles as Parquet |
| `generate_viz.py` | Plotly.js | Interactive HTML trading dashboard |
| `show_training_data.py` | — | Browse training samples in browser |
| `finetune_walkthrough.ipynb` | All of above | 11-chapter educational notebook |

---

## 20. Interview Answer Bank

### "What is LoRA?"

LoRA (Low-Rank Adaptation) freezes the 4 billion pre-trained parameters and adds two small
matrices A (rank×d_in) and B (d_out×rank) to each target layer. The weight update ΔW = B×A
has far fewer parameters than W itself (0.78% per layer). Only A and B are trained.
The theoretical justification: task-specific weight updates are empirically low-rank.

### "What is QLoRA?"

QLoRA combines LoRA with 4-bit NF4 quantization of the frozen base model.
The base model weights are stored as 4-bit integers (0-15) mapped to 16 non-uniform
quantization levels optimized for Gaussian-distributed weights. During forward pass,
weights are dequantized to bfloat16 on the fly. LoRA adapters remain in full bfloat16.
Net memory: ~5.5 GB vs 70 GB for full fine-tuning — enabling training on consumer hardware.

### "Which transformer components do you modify?"

Q, K, V, O projections (attention mechanism) and gate, up, down projections (FFN).
7 layers × 32 blocks = 224 LoRA pairs. Q/K/V teach the model what to attend to in
trading contexts. The FFN stores factual knowledge (e.g., "GNPA 1.33% = clean book").

### "How does the Adam optimizer work?"

Adam maintains two running averages per trainable parameter:
1. First moment (m): EMA of gradients — like momentum, averages the gradient direction
2. Second moment (v): EMA of squared gradients — estimates per-parameter learning rate scale

Update rule: θ = θ - lr × m̂ / (√v̂ + ε)

This gives each parameter its own effective learning rate scaled by its gradient variance —
parameters that get consistent gradients learn faster, noisy ones learn slower.

With `adamw_bnb_8bit`: m and v are stored in 8-bit integers, halving optimizer memory.

### "What is gradient checkpointing?"

During the normal forward pass, all intermediate activations are stored for use during
backpropagation (memory intensive). With gradient checkpointing, activations are discarded
during forward pass and recomputed from saved checkpoints during backpropagation.
Saves ~50% activation memory at the cost of ~15-20% slower training.

### "How do you know when to retrain the model?"

Evidently AI compares the training distribution vs live production decisions.
If the action distribution shifts (e.g., was 50% BUY during training, now 90% HOLD),
it signals a market regime change. We alert when `action_drifted=True` and trigger
a new flywheel round to generate fresh data in the new regime.

### "How does DVC fit into your ML pipeline?"

DVC (Data Version Control) tracks large binary files (Parquet candles, SQLite databases,
model adapters) alongside git. Every git commit has a DVC reference pointing to the exact
data state. `dvc repro` re-runs only the pipeline stages affected by changed inputs or params.
Change `lora_r` in `params.yaml` → only finetune and evaluate stages re-run.
This ensures every model is fully reproducible from a git commit + dvc pull.

### "Why only train on assistant tokens?"

The cross-entropy loss is masked to zero for system and user tokens.
Only the assistant's JSON output tokens contribute to the loss.
This teaches the model to produce correct decisions, not to memorise the input market data.
Additionally, it's more computationally efficient — 65% of a compact 188-token example
is assistant content (pure learning signal), vs only 7% of a verbose 990-token example.

### "What is vLLM's PagedAttention?"

In standard LLM inference, the KV cache (key-value pairs for each token in context)
must be pre-allocated for the maximum possible sequence length. Most of this allocation
is wasted (most sequences are shorter). PagedAttention borrows the concept from OS virtual
memory paging — allocates KV cache in fixed-size blocks, assigns them dynamically as
sequences grow. Result: near-zero memory waste, 2-4× higher throughput than naive serving.
