# The Trading-Agent Fine-Tuning Story
### A plain-English, in-depth storybook of *what* we built, *how* it works, and *why* every choice was made

> Read this top to bottom like a story. Part I sets the scene. Parts II–VI explain
> every moving piece (flywheel, MLX, LoRA, loss, data splits, model types...).
> Part VII is pros/cons/alternatives/industry-standards. Part VIII is a long
> Q&A that answers every specific question with a worked example.

---

# PART I — The Scene: what problem are we solving?

Imagine you hire a brilliant but generic intern. They know *everything on the internet* —
history, poetry, Python, cooking — but they have **never** traded an Indian bank stock.
Ask them "should I buy Canara Bank today?" and they'll waffle, invent numbers, or format
their answer differently every time.

That intern is a **base Large Language Model** (LLM) — in our case NVIDIA's
**Nemotron** family. Powerful, but not a *trading specialist*.

**Our goal:** turn that generic intern into a disciplined NSE equity trader that, given
today's chart + fundamentals, outputs a clean decision every single time:

```json
{"action":"BUY","stop_loss":116.5,"take_profit":123.0,"risk_reward":1.8,
 "reasoning":"Bullish EMA stack, price reclaimed VWAP, demand zone at 116.5...",
 "confidence":0.72,"key_risk":"Below-average volume on the breakout"}
```

**Why not just prompt the base model every time?** Three reasons:
1. It wastes tokens re-explaining what "GNPA" or "VWAP" means on every call.
2. It frequently hallucinates JSON field names or invents prices.
3. It has no consistent "house style" or risk discipline.

**The fix:** *fine-tuning* — show the model hundreds of worked examples of good trading
decisions until the behaviour is baked into its weights. After that, the model **is** the
trader; no elaborate prompt needed.

---

# PART II — The Flywheel: the heartbeat of the project

## What is a "flywheel"?

A flywheel is a heavy wheel that, once you push it, keeps spinning and gets easier to
push each time. In software/ML, a **data flywheel** is a self-reinforcing loop where each
turn produces better data, which produces a better model, which produces better data...

Our flywheel (`flywheel.py`):

```
         ┌─────────────────────────────────────────────┐
         │                                             │
         ▼                                             │
  1. SIMULATE   →  2. BUILD DATASET  →  3. FINE-TUNE  ─┘
  (trade a stock   (turn the trades    (teach the model
   day by day)      into JSONL)         the good decisions)
```

- **Round 1:** the model doesn't exist yet, so a *teacher* (a stronger model, or a simple
  rule) makes the decisions. Those decisions + their real-world outcomes become data.
- **Round 2:** we fine-tune the *student* on Round 1's data. Now the student plays.
- **Round 3+:** the smarter student generates even better data, which trains an even
  smarter student. The wheel accelerates.

## Why a flywheel instead of a one-shot dataset?

Because good trading data is *expensive and rare*. Rather than hand-label thousands of
charts, we **manufacture** data by simulating trades and letting **real historical prices**
grade each decision. Every turn of the wheel cheaply produces more, better-graded data.

## The crucial honesty mechanism

The model proposes a trade (BUY at 119, stop 116.5, target 123). Then we **replay the real
future prices** it never saw. Did price hit 116.5 first (loss) or 123 first (win)? The
market — not the model — decides the label. **The model cannot hallucinate a profit.**
This is what keeps the flywheel from spinning into fantasy.

---

# PART III — The Simulation & the Data

## What is an "episode"?

An **episode** = one simulated trading day. In each episode the agent:

1. Looks at the last ~30 days of candles up to a fixed "as-of" date.
2. Computes technicals (EMA, VWAP, support/resistance) — in plain Python, not the LLM.
3. Reads the latest quarterly fundamentals (PAT, GNPA, P/E).
4. **Decides** BUY / SELL / HOLD with exact stop-loss and take-profit.
5. The simulator replays the *next* few days of real prices to see if stop or target hit.
6. Writes everything (decision, reasoning, outcome, P&L, a "lesson learned") to SQLite.
7. Advances to the next trading day.

`EPISODES=50` means 50 such days per round. `ROUNDS=5` means the whole simulate→train loop
happens 5 times. So `5 × 50 = 250` episodes of experience for one stock.

## Three kinds of data (and why they must be separate)

When you fine-tune, you split your examples into **three buckets**:

| Split | Typical % | Purpose | Analogy |
|---|---|---|---|
| **Training** | 80% | The examples the model actually learns from. | Homework problems you study. |
| **Validation (val)** | 10% | Checked *during* training to see if it's generalising, not memorising. | Weekly quizzes while studying. |
| **Test** | 10% | Touched **only once at the very end** to get an honest final score. | The final exam. |

**Why separate them?** If you quiz yourself on the exact homework answers, a 100% score
proves nothing — you just memorised. The val and test sets are questions the model never
trained on, so a good score there means it actually *learned the skill*.

`data_pipeline.py` makes this split (80/10/10) after a **quality filter** (drop HOLD-only
samples, require confidence ≥ 0.5 and risk:reward ≥ 1.3) so we train on decisive, sensible
trades.

## Input data vs training data — the difference

- **Input data** = raw market facts: OHLCV candles, fundamentals, computed indicators.
  This is the *question* side.
- **Training data** = the *question + the correct answer* packaged together, in the
  model's chat format:

```json
{"messages":[
  {"role":"system","content":"You are a disciplined NSE equity trader..."},
  {"role":"user","content":"[CANBK 2024-01-10] C:119.78 EMA:50>100>200 ..."},
  {"role":"assistant","content":"{\"action\":\"BUY\",\"stop_loss\":116.5,...}"}
]}
```

The model learns to produce the `assistant` part when shown the `system`+`user` part.

## How is test data "computed"? (grading the exam)

Two layers:
1. **Format/skill grading** (`evaluate.py`): take each test example's question, ask the
   fine-tuned model, and check — is the output valid JSON? Are all required fields present?
   Is the reasoning non-empty and on-topic? This yields metrics like *JSON-validity 97%*.
2. **Outcome grading** (the simulator): because every stored trade also has its *real*
   price outcome (SL/TP hit), we can measure win-rate and P&L — the ground truth the market
   supplied, not the model's opinion.

---

# PART IV — The Models: types, and why some can train and some can't

## The teacher / student pattern

- **Teacher** = a strong model that *generates* the training data by making good decisions.
  We use `nemotron-bolt-e4b-16k` running in **Ollama** (your extended-context model).
- **Student** = the model we *fine-tune* to imitate (and eventually surpass) the teacher.
  We use `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit`.

**Why not let the model teach itself (self-distillation)?** If a 4B model generates its own
training data and trains on it, its small biases amplify each round until it collapses into
repetitive nonsense ("model collapse"). A stronger/independent teacher + **real price
labels** prevents that.

## Why can some models be fine-tuned and others not? (a key subtlety you hit)

Not every model file can be LoRA-trained on your Mac. Three gates:

1. **Format.** Ollama models are **GGUF** (a format optimised for *inference*, not training).
   You cannot LoRA-train a GGUF directly. You need the original **safetensors** weights
   (what `mlx-community` publishes). That's why we train the `mlx-community` repo, not the
   Ollama tag.
2. **Architecture.** LoRA attaches to standard attention layers (`q_proj`, `k_proj`...).
   Plain **Llama/Mistral/Nemotron-dense** architectures expose those cleanly → trainable.
   Exotic ones (large **Mixture-of-Experts**, **Mamba hybrids**) may not expose them the
   same way → the trainer may error or skip layers. (That's the risk we flagged with the
   30B MoE and OptiQ variants.)
3. **Size vs memory.** A 30B model needs ~15 GB *just to load* at 4-bit — no room left on a
   16 GB Mac for training. A ~4B model fits with room for adapters + optimizer. That's why
   you picked the **4B Bolt E4B**, not the 30B Lightning.

## Which model is better for *inference* vs *training*?

| Need | Prefer | Why |
|---|---|---|
| **Inference** (just running it) | GGUF in Ollama / quantised MLX | Compact, fast, low memory; no gradients needed. Your `-16k` Ollama tag is ideal here. |
| **Training** (teaching it) | Full/‑quantised **safetensors** in MLX or PyTorch | Training needs the weight structure + gradients; GGUF can't do that. |

So the *same* Bolt E4B plays both roles in different formats: **safetensors to train**,
then convert to **GGUF to serve** — re-applying your 16k context in the Ollama Modelfile.

---

# PART V — Training Internals (the engine room)

## PyTorch vs MLX — two engines, same idea

- **PyTorch** = the industry-standard deep-learning framework. Runs great on **NVIDIA GPUs**
  (CUDA). On a Mac it uses "MPS", which is slower and some libraries (like `bitsandbytes`
  for 4-bit) **don't support it at all**.
- **MLX** = Apple's own framework, built for **M-series chips**. Uses unified memory (CPU
  and GPU share the same RAM) and the Neural Engine. On your M5 Air it's the native, fast
  path — which is exactly why `train.py` auto-selects MLX and avoids PyTorch+CUDA.

Same maths underneath (matrix multiplies, gradients); different hardware backend.

## LoRA — teaching without rewriting the whole brain

The model has **billions** of weights. Rewriting all of them (full fine-tuning) needs
~60 GB of memory and massive compute. **LoRA (Low-Rank Adaptation)** freezes the giant
original and adds two tiny matrices (A and B) beside each attention layer. You train only
those — about **0.05–1%** of the parameters.

```
output = W·x        +   (B · A · x)·scale
         ↑ frozen giant     ↑ tiny trainable adapter (what learns)
```

In your run the log said `Trainable parameters: 0.047% (3.473M / 7463M)` — only 3.5 million
of 7.46 **billion** weights were being trained. That's LoRA.

## QLoRA — LoRA on a compressed model

Even frozen, the giant model takes memory to hold. **QLoRA** stores that frozen base in
**4-bit** (each weight uses 4 bits instead of 16), cutting memory ~4×, while the LoRA
adapters stay full precision. Your `...OptiQ-4bit` model is already 4-bit, so MLX is doing
exactly this: 4-bit frozen base + full-precision adapters. Result: a 7.46B model trains in
**~5.5 GB** — fits 16 GB comfortably.

## Adapter vs Fused model

- **Adapter** = just the trained A/B matrices — a tiny file (a few MB). Useless alone; it's
  a "patch" that must sit on top of the base model. Saved to `output/adapters/round_05/adapters`.
- **Fused model** = base model **+** adapter mathematically merged into one standalone model
  (`W_new = W + B·A`). Bigger, but self-contained and faster to run (no patch step at
  inference). Saved to `output/adapters/round_05/fused_model`. This is what you deploy.

## mlx_iters — how long training runs

`MLX_ITERS=600` means 600 training steps. Each step: show a small batch, measure error,
nudge the adapters. More iters = more learning (until it overfits). For a **smoke test** we
use 50 (fast, rough); for a **real run** 600 (good). It's a dial in `.env`, not code.

## max-tokens — a *generation* setting (not training)

When you *run* the model (`mlx_lm generate --max-tokens 200`), `max-tokens` caps how many
tokens it may produce in its answer. 200 is plenty for one JSON decision. It has nothing to
do with training; it just stops the model from rambling forever at inference time.

## What is "loss"? What is "val" loss? How are they calculated?

**Loss** = a single number measuring how *wrong* the model is. We use **cross-entropy**:
for each token the model should produce, it outputs a probability; loss is high if it gave
low probability to the correct token, near-zero if it was confident and right.

```
loss for one token = −log( probability the model gave to the CORRECT token )
```

Example: the correct next token is `BUY`.
- Model says P(BUY)=0.9 → loss = −log(0.9) ≈ **0.10** (good).
- Model says P(BUY)=0.1 → loss = −log(0.1) ≈ **2.30** (bad).

The batch loss is the average over all the assistant-answer tokens. **Only the assistant
tokens count** — the model isn't graded on repeating the question back.

- **train_loss** = measured on the homework (training batch). Falls as it memorises/learns.
- **val loss** = the same formula but on the **validation** set the model never trained on.
  It's the honest "is it actually generalising?" number, checked every ~50 steps.

**Reading your run:** `val 2.934 → 0.011`. A val loss near **0.01** is *suspiciously* low —
it means the model memorised a trivially repetitive target. That happened because the data
was the **rule-based fallback** (mostly "HOLD" with a fixed template). On real, varied LLM
data you'd expect val to settle around **0.3–0.8** — lower is good, but *too* low signals
the data was too easy/repetitive.

## One full training step, start to finish (worked example)

```
Batch: 2 examples (each = system+user+assistant JSON)
 1. FORWARD : feed tokens through the 4-bit frozen model + LoRA adapters
              → model predicts a probability for each next token
 2. LOSS    : cross-entropy on the assistant tokens only  (e.g. 0.42)
 3. BACKWARD: compute how to tweak A,B to lower that 0.42  (gradients)
 4. UPDATE  : Adam optimizer nudges A,B (base stays frozen)
 5. every 50 steps → compute val loss on held-out data
 6. every 100 steps → save a checkpoint
repeat 600 times → save adapter → fuse → done
```

---

# PART VI — End-to-end example (one trade's whole life)

```
Day: 2024-01-10, CANBK
INPUT  →  C:119.78, EMA50>EMA100>EMA200 (bullish), price above VWAP,
          demand zone 116.5–117.2, PAT ₹5,381cr, P/E 9.2
TEACHER(LLM) decides →  BUY 16 @119.78, SL 116.5, TP 123.0, R:R 1.8,
          "reasoning: bullish stack + VWAP reclaim + demand-zone support"
MARKET replays next 5 days →  high reached 123.1 before low hit 116.5
OUTCOME →  TP_HIT, +₹52, capital 10,052
STORED →  this becomes 2 training rows (the decision + the post-trade lesson)
LATER  →  data_pipeline packs it into JSONL (80% train / 10% val / 10% test)
TRAIN  →  student model learns to produce that BUY JSON from that input
EXAM   →  on unseen test days, does it still produce valid, sensible JSON?
```

---

# PART VII — Pros, Cons, Alternatives, Industry Standards

## Pros of our approach
- **Cheap, self-generating data** via simulation + real-price grading (no manual labelling).
- **Runs on a laptop** (16 GB M5 Air) thanks to QLoRA + MLX — no cloud bill.
- **Honest labels** — the market grades every trade; the model can't fake wins.
- **Config-driven** (`.env`) — reproducible, no code edits.
- **Full MLOps stack** — MLflow (tracking), DVC (data versioning), Prefect (scheduling),
  Evidently (drift) — resume-grade engineering.

## Cons / limitations
- **Backtest ≠ live trading.** Historical replay ignores slippage, liquidity, fees, and the
  fact that your own orders move price. A great backtest can still lose live.
- **Survivorship & look-ahead risks.** Easy to accidentally leak future info; we guard with
  strict as-of cutoffs, but it needs vigilance.
- **Small data → memorisation.** As your 0.01 loss showed, too-easy/too-few examples make
  the model memorise rather than reason.
- **Teacher ceiling.** The student can only be as wise as the teacher's data (plus outcome
  signal). A weak teacher caps quality.
- **Not financial advice / not production-trading-safe** without risk controls, compliance,
  and live paper-trading validation.

## What could be improved
- **More & more diverse data**: 10+ symbols, multiple market regimes (bull/bear/sideways),
  2,000+ examples → push rank to 32.
- **Outcome-weighted training / RLHF / DPO**: train harder on trades that actually won,
  down-weight losers — moving from "imitate the teacher" to "optimise for profit".
- **Walk-forward validation**: test on strictly *later* dates than training (true out-of-sample).
- **Richer evaluation**: Sharpe ratio, max drawdown, not just JSON validity.
- **Guardrails**: position sizing, portfolio limits, circuit-breakers before any live use.

## Alternatives we could have chosen (and why we didn't)

| Decision | We chose | Alternatives | Why our choice |
|---|---|---|---|
| Fine-tune method | **QLoRA (LoRA + 4-bit)** | Full fine-tune; prompt-only; RAG | Full FT needs 60 GB; prompt-only is inconsistent; QLoRA fits a laptop. |
| Framework (Mac) | **MLX** | PyTorch+MPS; llama.cpp finetune | MLX is native/fast on M-series; PyTorch MPS lacks 4-bit support. |
| Framework (GPU) | **PyTorch + PEFT/TRL + bitsandbytes** | JAX; raw CUDA | Industry standard, huge ecosystem. |
| Training config | **Axolotl / mlx-lm CLI** | Hand-written training loop; HF Trainer raw | Axolotl is the de-facto standard; less boilerplate, fewer bugs. |
| Serving | **vLLM (GPU) / Ollama (local)** | TGI; raw transformers; TorchServe | vLLM = best throughput (PagedAttention); Ollama = simplest local. |
| Data labels | **Real price outcomes** | Human labels; synthetic self-labels | Cheapest *and* most honest; avoids model collapse. |

## What is "industry standard" here?
- **LoRA/QLoRA** is the default way teams fine-tune open models in 2024-2025.
- **PyTorch + Hugging Face (PEFT, TRL, Transformers, Datasets)** is the dominant stack.
- **Axolotl** is the most common turnkey fine-tuning wrapper.
- **vLLM** is the standard high-throughput inference server.
- **MLflow / DVC / Prefect / Evidently** are mainstream MLOps tools — all present here.

---

# PART VIII — Deep Q&A (with examples)

**Q1. Why fine-tune at all instead of a clever prompt?**
A prompt is temporary and inconsistent; fine-tuning bakes the behaviour into the weights so
the model is reliably a trader with no prompt gymnastics. Example: after fine-tuning, the
input can be a terse `[CANBK ...]` line and you still get perfect JSON — the "you are a
disciplined trader, output these 8 fields..." instructions are now *internalised*.

**Q2. What exactly is the flywheel optimising?**
Each turn: better model → better decisions → better-graded data → better model. The target
over rounds is higher JSON validity, better reasoning, and (ideally) higher win-rate/P&L on
unseen days.

**Q3. What is an episode vs an iteration?**
An **episode** is a *simulated trading day* (data generation). An **iteration (mlx_iters)**
is a *training step* (learning). 50 episodes might create ~80 training examples; 600
iterations then train on them. Totally different phases.

**Q4. Why is my loss 0.01 — isn't lower always better?**
Lower is better *only up to a point*. 0.01 means near-perfect memorisation, which only
happens when targets are trivially predictable (your rule-based "HOLD" template). With real,
varied LLM data, ~0.3–0.8 is the healthy zone. Think of a student scoring 100% because the
"exam" was identical to the homework — impressive number, meaningless skill.

**Q5. How is val loss different from train loss, concretely?**
Same cross-entropy formula, different data. Suppose after training:
- train_loss = 0.05 (it memorised homework), val_loss = 0.9 (struggles on new questions)
  → **overfitting**. 
- train_loss = 0.4, val_loss = 0.45 → **healthy generalisation**.
The *gap* between them is the tell.

**Q6. Why can't I fine-tune the Ollama model directly?**
Ollama serves **GGUF** (inference-only format). Training needs **safetensors** (weights +
structure for gradients). So we train `mlx-community/...safetensors`, then convert the
result to GGUF to put it back in Ollama.

**Q7. Why did MLX say rank 8 when `.env` says LORA_R=16?**
The MLX CLI defaults rank to 8 and reads rank from its own config, not our `--lora_r`. It's
cosmetic — rank 8 is fine for this data size. We expose `MLX_NUM_LAYERS`/`LORA_R` in `.env`;
raising rank only helps once you have thousands of varied examples.

**Q8. What's the difference between adapter and fused model, in practice?**
Adapter = 60 MB patch you ship to someone who already has the base model. Fused = the whole
model with the patch baked in — larger, standalone, faster to serve. Deploy the fused one.

**Q9. What does max-tokens control?**
Only generation length at inference. `--max-tokens 200` = "produce at most 200 tokens."
Enough for one JSON decision. Unrelated to training.

**Q10. What are "input / training / testing" data again, with sizes?**
Input = raw facts (candles/fundamentals). Training (80%) = input+answer the model learns
from. Val (10%) = held-out, checked during training. Test (10%) = held-out, scored once at
the end. Example: 100 quality examples → 80 train / 10 val / 10 test.

**Q11. How is test data scored?**
`evaluate.py` asks the fine-tuned model each test question and checks JSON validity, schema
completeness, and reasoning quality; separately, stored trades carry real price outcomes so
we can also report win-rate/P&L.

**Q12. PyTorch or MLX — which, and why?**
On NVIDIA → PyTorch (CUDA ecosystem, 4-bit via bitsandbytes). On Apple Silicon → MLX
(native, fast, unified memory; bitsandbytes doesn't work on Mac). Your M5 Air → MLX.

**Q13. Which model type is best for training vs inference?**
Training: dense, standard-architecture, safetensors, small enough to fit (our 4B Bolt E4B).
Inference: quantised GGUF/MLX for speed and low memory (your Ollama -16k tag). MoE/Mamba
hybrids and 30B+ are great to *run* but awkward/too big to *train* on 16 GB.

**Q14. What is QLoRA actually saving?**
Memory. 4-bit frozen base (~2.3 GB for 7.46B) + tiny bf16 adapters + small optimizer state
≈ 5.5 GB total, vs ~60 GB for full fine-tuning. That's the difference between "runs on a
MacBook Air" and "needs a data-center GPU."

**Q15. If we had unlimited hardware, what would we do differently?**
Possibly full fine-tuning or higher-rank LoRA on a larger model, with RLHF/DPO using
real P&L as reward, walk-forward validation across years and many symbols, and an ensemble
of specialists (one per sector). On a laptop, QLoRA on a 4B is the pragmatic sweet spot.

**Q16. Is this ready to trade real money?**
No. It's a strong *learning/portfolio* project. Live trading needs slippage/fee modelling,
paper-trading validation, risk limits, monitoring, and compliance. Treat outputs as
research, not advice.

---

## One-paragraph summary (tell it to a friend)

We take a generic AI, let a stronger "teacher" AI play a stock-trading simulator day by day,
and let the **real market** grade every trade. Those graded decisions become lessons. Using
**QLoRA on Apple's MLX**, we cheaply teach a small 4-billion-parameter model to copy the
good decisions — training only ~0.05% of its weights so it fits on a laptop. We watch a
**loss** number fall to know it's learning, and check a separate **validation** number to be
sure it's *actually* learning and not memorising. Repeat the loop (the **flywheel**) and the
model gets better each turn. The output is a specialist that reliably emits clean, disciplined
trade decisions — a complete, resume-grade demonstration of modern LLM fine-tuning and MLOps.
