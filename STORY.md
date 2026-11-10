# STORY — the trading-agent fine-tuning flywheel

A plain-English narrative of what this project is, how it works, and how it
grew. For the architecture reference see [`docs/MASTER.md`](docs/MASTER.md); for
the dated change log see [`TIMELINE.md`](TIMELINE.md).

---

## The idea

Teach a small LLM to trade one NSE stock (CANBK) with discipline, by letting it
**learn from its own simulated trades** in a loop:

> generate decisions → turn them into training data → fine-tune → evaluate on
> future dates → let the improved model generate the next batch.

That loop is the **flywheel**. Each turn the student gets a little better, so the
data it produces gets a little better, so the next student is better still.

---

## The cast (modules)

- **datagen** — a simulated trader replays market days. For each day it reads the
  context (EMAs, VWAP, support/resistance zones, India VIX regime, quarterly
  fundamentals), decides BUY/SELL/HOLD **with a written reason**, lives through
  the outcome, and writes the trade — reasoning and all — to SQLite.
- **splitter** — turns those raw trades into clean, unique, balanced training
  examples and a **walk-forward** train/val/test split (earliest dates train,
  latest dates test = the real future).
- **training** — LoRA/QLoRA/MLX fine-tuning on the growing buffer, starting from
  the previous adapter (continual learning).
- **evaluation** — scores each round on the future-dated test set: win rate,
  expectancy, profit factor, Sharpe, drawdown — plus model-quality checks.
- **orchestrator** — runs the whole flywheel; **monitoring** watches for drift.

Every module has its own folder, a FastAPI endpoint, and a tile on the live UI.

---

## The plot twist everyone hits: *"this takes forever / I had to stop"*

Real generation and training are slow, and laptops get closed. So the whole
system is built to **never lose work**:

- Raw episodes live in SQLite and are never regenerated.
- Every stage checkpoints its progress atomically, so a crash at episode 37/50
  resumes at 38 — and if you **stop today and come back on day 3, training still
  uses every episode generated on days 1 and 2.**
- Faster generation for a single symbol: split the calendar into windows and run
  them in **parallel processes** (`--windows N`).

---

## The details that make the numbers honest

- **Fees & taxes**: P&L is net of brokerage, STT, exchange charges, SEBI fee,
  GST and stamp duty — not a fantasy gross number.
- **Volatility-aware sizing**: when India VIX is low the agent takes **more
  quantity for fewer points**; when VIX is high it takes **less quantity for
  more points** with a wider stop — sizing to the regime, from your VIX CSV.
- **Know what's been learned**: every training row has a 16-char content hash;
  a tiny registry records which rows already trained the model, so the UI shows
  **new vs already-trained** coverage and the dataset stays **unique**.

---

## How it grew

1. **Day 1** — broke a flat pile of scripts into resumable modules, gave each a
   FastAPI endpoint + a tracking UI, added CI and docs, fixed a silent
   evaluation bug (stripped `_meta`).
2. **Day 2** — made the simulator import **offline** (no `yfinance`/`requests`
   needed for rule-based runs).
3. **Days 5–7** — external audits hardened the pipeline: fixed look-ahead
   leakage in the prompt history, made data generation non-overlapping and
   `params.yaml`-authoritative, deepened MLflow tracking, and tightened
   dependency pins. (See TIMELINE.md.)
3. **Day 3** — parallel episodes for one symbol, multi-day resume guarantees,
   reasoning visible in the UI, train-coverage/unique-row tracking, fees & taxes
   in P&L, and VIX-based position sizing with your VIX CSV format.

The result: a small, honest, resumable loop you can watch, stop, resume, and
trust — on a laptop.
