# The Trading-Agent Fine-Tuning Story
### A plain-English, in-depth storybook of *what* we built, *how* it works (matched to the code), and *why* every choice was made

> **Who this is for:** anyone — even if you have never traded a stock, never trained an AI model,
> and never read this code. Read it top to bottom like a story. Every new word is explained the
> first time it appears, and there is a glossary right after this page.
>
> **How it is organised**
>
> | Part | What you learn |
> |---|---|
> | 0 | 60-second summary + glossary (read this first) |
> | I | The problem and the rules of the game (long-only swing trading) |
> | II | A map of every file in the project |
> | III | The flywheel — the loop that runs everything |
> | IV | The simulation in depth — how one trading day is played |
> | V | The prompt — exactly what the AI is shown |
> | VI | Turning trades into training data |
> | VII | The models — teacher, student, and why some can train |
> | VIII | Training internals — LoRA, QLoRA, loss |
> | IX | Evaluation — how we grade the model |
> | X | One trade's whole life, with real arithmetic |
> | XI | Every setting in `.env` |
> | XII | How to run it, step by step, and what to do when it breaks |
> | XIII | The tests |
> | XIV | Bug history — what was wrong and how it was fixed |
> | XV | Honest limitations of the current code |
> | XVI | Pros, cons, alternatives, industry standards |
> | XVII | Deep Q&A |
>
> **A note on accuracy:** this storybook was written by reading the actual code
> (`simulation.py`, `data_pipeline.py`, `compact_prompt.py`, `flywheel.py`, `evaluate.py`, the tests).
> A few files that the project *uses* (`train.py`, `config.py`, `zones.py`, `fundamentals_xbrl.py`,
> `inference_mlx.py`) were **not in the uploaded zip**. Where this story talks about them it says
> only what the other files reveal, and marks them **(not in this upload)**.

---

# PART 0 — The 60-second version, and a glossary

## The whole project in one paragraph

We want a small AI model (about 4 billion parameters) that can look at a stock's chart and
fundamentals and answer with a clean, disciplined trading decision in JSON: **BUY** (with a stop-loss
and a take-profit) or **HOLD** (stay in cash). We have no big labelled dataset, so we *manufacture*
one. A "teacher" AI plays a trading simulator one historical day at a time. After each decision, the
simulator replays the *real* prices that followed and records whether the trade hit its stop-loss
or its target. Those graded decisions are packaged as lessons, and a small "student" model is
fine-tuned on them using a memory-saving technique (LoRA/QLoRA) that fits on a laptop. Then the
loop repeats with the improved student. That repeating loop is called the **flywheel**.

## The strategy rule that governs everything

> **Long-only swing trading.** The only two actions are **BUY** and **HOLD**. There is no SELL and no
> shorting, anywhere in the pipeline. A trade is held for days up to about two weeks
> (`MAX_HOLD_DAYS = 10` trading days).

## Glossary — trading words

| Word | Plain meaning |
|---|---|
| **NSE** | National Stock Exchange of India, where these stocks trade. |
| **Symbol / ticker** | A short code for a company, e.g. `CANBK` = Canara Bank, `RBLBANK`. |
| **Candle (OHLCV)** | One day (or hour, or minute) of price summarised as **O**pen, **H**igh, **L**ow, **C**lose and **V**olume (shares traded). |
| **Long** | Buying a stock hoping the price rises. |
| **Short / SELL-to-open** | Betting the price falls. **Not allowed here.** |
| **Swing trading** | Holding a position for several days to a couple of weeks, not minutes. |
| **HOLD** | In this project: do nothing, stay in cash, don't open a trade today. |
| **Stop-loss (SL)** | A price *below* your entry where you give up and sell to limit the loss. |
| **Take-profit (TP)** | A price *above* your entry where you sell to lock in the win. |
| **Risk : Reward (R:R)** | Reward ÷ risk. Entry 100, SL 98, TP 104 → risk 2, reward 4 → R:R = 2.0. We require **at least 1.5**. |
| **Position size / quantity** | How many shares you buy. Limited so a stop-out loses at most **2%** of capital. |
| **EMA** | *Exponential Moving Average* — a smoothed average price that weights recent days more. EMA-50 / 100 / 200 = averages over roughly 50 / 100 / 200 days. |
| **EMA stack** | The order of the three EMAs. **Bullish** = 50 > 100 > 200 (short-term above long-term, i.e. an uptrend). **Bearish** = 50 < 100 < 200. Otherwise **mixed**. |
| **VWAP** | *Volume-Weighted Average Price* — the average price people actually paid, weighted by volume. Price above VWAP = buyers are in control. |
| **ATR** | *Average True Range* — how much a stock typically moves per day. Used to measure "how far is that level, in normal daily moves?". (Here approximated as the average of the last 5 daily high–low ranges.) |
| **Support / Resistance** | A price zone where falling prices tend to stop (support) or rising prices tend to stall (resistance). |
| **Swing high / swing low** | A local peak / local valley in the price chart. |
| **HH, HL, LH, LL** | Higher-High, Higher-Low, Lower-High, Lower-Low — the pattern of peaks and valleys that defines an uptrend (HH+HL) or downtrend (LH+LL). |
| **PDH / PDL** | *Previous Day's High / Low* — a popular reference level. |
| **Demand / Supply zone** | Areas on a short-timeframe chart where buyers (demand) or sellers (supply) previously overwhelmed the other side. |
| **Confluence** | Two signals agreeing — here, a 15-minute zone sitting *inside* a 1-hour zone. |
| **India VIX** | The market's "fear gauge" — expected volatility. Low = calm, high = nervous. |
| **PAT** | *Profit After Tax* — a company's quarterly net profit. |
| **P/E** | Price-to-Earnings ratio — how many rupees investors pay per rupee of yearly profit. |
| **Fundamentals** | Company financials (profit, revenue, P/E) as opposed to the price chart ("technicals"). |
| **XBRL** | The official machine-readable format companies use to file results with Indian regulators. |
| **P&L** | Profit and Loss in rupees. |
| **Drawdown** | How far your account falls from its highest point. |
| **Backtest / simulation** | Replaying history to see how a strategy *would* have done. |

## Glossary — AI / machine-learning words

| Word | Plain meaning |
|---|---|
| **LLM** | *Large Language Model* — an AI that reads and writes text (ChatGPT-like). |
| **Parameters / weights** | The numbers inside a model that store what it learned. "4B" = 4 billion of them. |
| **Fine-tuning** | Continuing a model's training on your own examples so it specialises. |
| **LoRA** | *Low-Rank Adaptation* — fine-tune by training a tiny add-on patch instead of all the weights. |
| **QLoRA** | LoRA on top of a model compressed to 4-bit numbers, to save memory. |
| **Adapter** | The small trained patch LoRA produces. |
| **Fused model** | Base model + adapter merged into one standalone model. |
| **Teacher / Student** | Teacher = the model that *makes* training decisions. Student = the model we *train* to copy (and later improve on) them. |
| **Prompt** | The text you give a model. Has a **system** part (the rules) and a **user** part (today's situation). |
| **Token** | A chunk of text (~4 characters). Models read and bill in tokens. |
| **JSONL** | A text file with one JSON object per line — the standard format for training data. |
| **Epoch / iteration** | One pass through all data / one small training step. |
| **Loss** | A number for "how wrong is the model?" Lower is better (to a point). |
| **Overfitting** | Memorising the homework instead of learning the skill. |
| **Inference** | *Using* a trained model (as opposed to training it). |
| **Quantisation (4-bit)** | Storing each weight with fewer bits to shrink the model. |
| **GGUF / safetensors** | Two file formats for model weights. GGUF = for running (Ollama). Safetensors = for training. |
| **Ollama** | A program that runs models locally on your computer. |
| **MLX** | Apple's machine-learning framework for M-series Macs. |
| **Flywheel** | A self-reinforcing loop: better model → better data → better model. |
| **Episode** | One simulated trading day (data generation phase). |
| **Walk-forward** | Train on the past, test on the *future* — never the reverse. |
| **Look-ahead bias** | Accidentally letting the model see information from the future. Fatal for trading backtests. |

---

# PART I — The Scene: what problem are we solving?

Imagine you hire a brilliant but generic intern. They know *everything on the internet* —
history, poetry, Python, cooking — but they have **never** traded an Indian bank stock.
Ask them "should I buy Canara Bank today?" and they'll waffle, invent numbers, or format
their answer differently every time.

That intern is a **base Large Language Model** — in our case NVIDIA's **Nemotron** family.
Powerful, but not a *trading specialist*.

**Our goal:** turn that generic intern into a disciplined NSE swing trader that, given today's
chart and fundamentals, outputs a clean decision every single time:

```json
{"action":"BUY","quantity":16,"stop_loss":116.5,"take_profit":125.0,"risk_reward":1.59,
 "reasoning":"Bullish EMA stack, price above VWAP, demand-zone support at 116.5",
 "confidence":0.72,"key_risk":"Below-average volume on the breakout",
 "features_used":["ema_stack","vwap","nearest_support"]}
```

Notice the fields. The system prompt (Part V) forces exactly these keys:
`action | quantity | stop_loss | take_profit | risk_reward | reasoning | confidence | key_risk | features_used`.

**Why not just prompt the base model every time?**
1. It wastes tokens re-explaining what "VWAP" means on every call.
2. It frequently hallucinates JSON field names or invents prices.
3. It has no consistent "house style" or risk discipline.

**The fix:** *fine-tuning* — show the model hundreds of worked examples of good decisions until
the behaviour is baked into its weights.

## Why "long-only swing"? (a design decision with consequences)

Earlier versions of this project allowed SELL (shorting) and held trades only about 5 days.
Two problems appeared:

1. **A real bug:** SELL trades had `risk_reward = 0.0` in the data, so the quality filter threw
   every SELL away — the training set ended up with *no* short examples, silently.
2. **Realism:** many retail Indian traders cannot freely short delivery stocks.

So the rules were simplified and enforced **everywhere** (details in Part XIV):

| Rule | Value | Where enforced |
|---|---|---|
| Only BUY or HOLD | SELL is converted to HOLD | `sanitize_decision()` in `simulation.py` |
| Stop-loss below entry, target above | `0 < SL < entry < TP` | `sanitize_decision()` |
| Minimum reward:risk | `MIN_RR = 1.5` | `sanitize_decision()` + `data_pipeline.quality_filter()` |
| Max risk per trade | `RISK_PER_TRADE = 2%` of capital | `sanitize_decision()` |
| Max holding period | `MAX_HOLD_DAYS = 10` trading days | `check_sl_tp()` |
| Account "death" | capital below 20% of start | `TradingAgent.alive()` |

---

# PART II — A map of the project (every file, in plain words)

## Files in this upload

| File | Role | One-line explanation |
|---|---|---|
| `flywheel.py` | **Orchestrator** | Runs the whole loop: simulate → build dataset → fine-tune → evaluate → repeat. |
| `simulation.py` | **The trading world** | Loads prices, computes indicators, asks the AI for a decision, replays real prices to grade it, logs to SQLite. Contains the `TradingAgent` class. |
| `compact_prompt.py` | **The prompt writer** | The *single source of truth* for the system prompts and the compact user prompt. Used by both the simulator and the dataset builder. |
| `data_pipeline.py` | **Data factory** | Reads the SQLite log and writes train / val / test JSONL files with quality filtering, de-duplication and balancing. |
| `evaluate.py` | **The exam** | Grades the model (JSON validity, schema, reasoning…) and computes trading metrics (win rate, Sharpe…). |
| `inference_vllm.py` | **Fast serving client** | An async client for a vLLM server — for production-speed inference on a GPU. |
| `show_training_data.py` | **Viewer #1** | Builds an HTML page showing every training sample exactly as the model sees it. |
| `build_data_explorer.py` | **Viewer #2** | Builds a sortable, searchable, offline HTML table of the training data. |
| `tests/test_build.py` | **Safety net** | 11 tests, including a full offline end-to-end run on synthetic prices. |
| `.env.example` | **Settings template** | Copy to `.env` and edit — no code changes needed. |
| `README.md` | Quick-start & commands | Run instructions. |
| `flow.md` | Diagrams | Mermaid architecture diagrams. |
| `TIMELINE.md` | History | What was built, what broke, how it was fixed — in order. |
| `CHANGES.md` | Latest change-log | The long-only / bug-fix release notes. |
| `STORYBOOK.md` | **This file** | The beginner-friendly deep explanation. |

## Files the code uses that are **not in this upload**

| File | What the other code says it does |
|---|---|
| `config.py` | Provides `load_env()` (reads `.env`) and the helpers `env_str`, `env_int`, `env_bool`. |
| `train.py` | One training command that auto-detects hardware: MLX LoRA on Apple Silicon, HF LoRA or QLoRA on NVIDIA. `flywheel.py` calls it as a subprocess with `--train --val --model --output_dir --epochs --lora_r --iters --save_every`. |
| `zones.py` | Detects demand/supply zones from 15-minute and 1-hour candles (`detect_zones`, `nearest_zones`, `confluence`, `fmt_zone`). |
| `fundamentals_xbrl.py` | Parses official XBRL result filings into fundamentals (`get_fundamentals_xbrl`). |
| `inference_mlx.py` | Loads the fine-tuned MLX model for the "adapter" backend (`generate_mlx`). |
| Others named in the docs | `runner.py`, `llm_factory.py`, `candle_store.py`, `prefect_flows.py`, `deploy_to_ollama.py`, `dpo_sketch.py` (MLOps extras: tracking, serving factory, data layer, scheduling, GGUF deploy, preference-learning sketch). |

> **Practical consequence:** with only this zip you can read everything, run the data-pipeline /
> prompt / sanitiser / evaluation logic and the offline end-to-end test — but you cannot run the full
> flywheel until the missing files are alongside it.

## Folders and files the pipeline creates when it runs

```
trading_agent.db                  SQLite database: every simulated day (the "trade_log")
data/rounds/round_01/             trading_train.jsonl  trading_val.jsonl  trading_test.jsonl
data/rounds/round_02/             ...one folder per round
data/rounds/merged_train.jsonl    the exact file handed to training this round
output/adapters/round_01/         adapters/ and fused_model/   (the trained model)
output/adapters/flywheel_log.json a JSON summary of every round
metrics/eval_results.json         written by `python evaluate.py`
data/prices/CANBK.csv             YOUR input: price history (you supply this)
data/prices/INDIAVIX.csv          YOUR input (optional): India VIX history
data/fundamentals/CANBK/*.xml     YOUR input (optional): XBRL result filings
```

---

# PART III — The Flywheel: the heartbeat of the project

## What is a "flywheel"?

A flywheel is a heavy wheel that, once pushed, keeps spinning and gets easier to push each
time. In ML, a **data flywheel** is a loop where each turn makes better data, which makes a
better model, which makes better data…

```
        ┌────────────────────────────────────────────────────────────┐
        │                                                            │
        ▼                                                            │
 1. SIMULATE ──► 2. BUILD DATASET ──► 3. FINE-TUNE ──► 4. EVALUATE ──┘
 (play N trading   (turn the trades    (teach the        (grade it on dates
  days, grade       into JSONL         student the       it never trained on)
  with real         lessons)           good decisions)
  prices)
```

## Why a flywheel instead of one big dataset?

Good trading data is *expensive and rare*. Instead of hand-labelling thousands of charts, we
**manufacture** data by simulating trades and letting **real historical prices** grade each
decision. Every turn of the wheel cheaply produces more graded data.

## The crucial honesty mechanism

The model proposes a trade (BUY at 119.78, stop 116.5, target 125). Then the simulator replays
the **real future prices** the model never saw. Did the price fall to 116.5 first (loss) or
rise to 125 first (win)? The market — not the model — decides the label. **The model cannot
hallucinate a profit.** That is what keeps the flywheel from spinning into fantasy.

## What `flywheel.py` actually does, function by function

### `run_simulation_round(...)` — Step 1

1. Points the simulator at the database (`simulation.DB_PATH = db_path`) and calls `init_db()`.
2. **Chooses who makes the decisions** (the "backend"), in this priority order:

   | Condition | Backend | Who decides |
   |---|---|---|
   | An adapter path exists on disk | `"adapter"` | The **fine-tuned student** from the previous round |
   | else `USE_OLLAMA=1` | `"ollama"` | The **teacher** (Nemotron via Ollama) |
   | else | `"fallback"` | A **hand-written rule** (cheap smoke test, low-quality data) |

3. Creates `TradingAgent(symbol, start_date, capital=10_000)` and calls `run_episode()` once per
   trading day, `n_episodes` times. If the agent's capital falls below 20% of the start, it
   stops ("died"). If a day throws an error, that day is skipped and the clock moves on.
4. Returns a summary: episodes, wins, losses, total P&L, final capital, win rate.

### `build_dataset_for_round(...)` — Step 2

Calls `data_pipeline.build_dataset()` and writes
`data/rounds/round_NN/trading_{train,val,test}.jsonl`. Uses `min_confidence=0.5` and the same
`MIN_RR` the simulator enforces. Returns the number of **training** examples.
If fewer than **10** training examples exist, the round **skips training** and moves on.

### `finetune_round(...)` — Step 3

1. Reads **only this round's** `trading_train.jsonl` and copies it to `data/rounds/merged_train.jsonl`.
   *(Why only one round's file? `build_dataset()` already reads the **entire** database every
   round, so each round's file already contains everything so far. Merging rounds 1..N would copy every
   example N times and leak test dates into training — a bug that was fixed, see Part XIV.)*
2. Decides where to start: from the **previous round's model** if there is one (this is
   *continual learning*), else from `BASE_MODEL`.
3. Runs `train.py` as a subprocess with epochs, LoRA rank, iterations and checkpoint interval
   taken from `.env`.
4. Finds the result, preferring `fused_model` (a standalone model the next round can load and
   continue from), then `final_adapter`, then the raw `adapters` folder.
5. If nothing was produced, raises an error — and the flywheel catches it and **keeps the
   previous model** rather than crashing.

### `evaluate_round(...)` — Step 4 (never fatal)

On the walk-forward **test** file it computes trading metrics from recorded P&L, and — if a
trained model and MLX are present — also model-quality metrics. Results are printed and logged
to MLflow if installed. Any failure here is caught and printed; it never stops the loop.

### The main loop

```
for round in 1..ROUNDS:
    sim_stats  = run_simulation_round(...)         # who plays? adapter > ollama > fallback
    n_examples = build_dataset_for_round(...)
    if n_examples < 10: log "not trained"; continue
    new_model  = finetune_round(..., prev_adapter=current_adapter)
    current_adapter = new_model
    evaluate_round(...)
save output/adapters/flywheel_log.json
```

## A round-by-round story (default: 5 rounds × 50 episodes)

- **Round 1:** no student exists. The **teacher** (Ollama) — or the rule-based fallback if
  `USE_OLLAMA=0` — plays 50 days. The database now has 50 rows. A dataset is built and the student
  is fine-tuned from `BASE_MODEL`. Evaluate.
- **Round 2:** the Round-1 student now **plays** 50 days (starting again from `START_DATE`).
  Its decisions are added to the database. The dataset is rebuilt from *all* rows. The student
  is fine-tuned again, starting from the Round-1 model.
- **Rounds 3–5:** repeat. The loop keeps growing the database and refining the model.

> ⚠ **Important subtlety:** every round restarts at `START_DATE`, so later rounds replay the
> *same calendar days* as earlier rounds. De-duplication (Part VI) keeps only the **latest**
> decision for a given symbol-and-day, so a newer round's decision *replaces* an older round's.
> To get genuinely **new** data, run several different symbols or different `START_DATE`s
> (the README suggests this for reaching 500+ examples).

---

# PART IV — The Simulation in depth (`simulation.py`)

## What is an "episode"?

An **episode = one simulated trading day.** The `TradingAgent` class holds the state: the
symbol, current date, capital, history of previous trades, and episode counter. `run_episode()`
does the following, in this exact order:

| # | Step | Code |
|---|---|---|
| 1 | Increment the episode counter and print status | `run_episode` |
| 2 | Load the **last 250 daily candles** up to *today* | `get_candles(symbol, date, TECH_LOOKBACK_DAYS)` |
| 3 | Compute **technical indicators** from those candles | `compute_technicals()` |
| 4 | Add **VIX regime** context (if a VIX file is configured) | `vix_context()` |
| 5 | Add **intraday demand/supply zones** (if minute data exists) | `compute_zones()` |
| 6 | Fetch the latest **fundamentals** known *as of today* and tag them fresh/stale/none | `get_fundamentals()`, `_fund_completeness()` |
| 7 | Build the prompt and ask the **AI backend** for a decision; enforce the rules | `pre_trade_decision()` → `sanitize_decision()` |
| 8 | Load the **next ≤10 days** of real prices the AI never saw | `get_forward_candles()` |
| 9 | Walk through them to see whether **SL, TP, or the time limit** hits first | `check_sl_tp()` |
| 10 | Compute **P&L**, update capital | `compute_pnl()` |
| 11 | If it was a BUY, ask the AI for a **post-trade lesson** | `post_trade_analysis()` |
| 12 | **Save everything** to SQLite, add to history, move to the next trading day | `run_episode`, `next_trading_day()` |

`EPISODES=50` means 50 such days per round; `ROUNDS=5` means the whole loop runs 5 times.

## Where the price data comes from

`DATA_SOURCE` picks the source:

- **`csv`** (the default in `.env.example`) — reads `data/prices/{SYMBOL}.csv`. The CSV loader is
  forgiving: it finds the date column by name, accepts `o/h/l/c/v` short names, treats
  volume as optional, and works with **either daily bars or 1-minute bars**. Minute data is
  resampled to daily bars (and to 15-minute and 1-hour bars for the zone detector).
- **`yfinance`** — downloads history from Yahoo Finance **once**, with retry/back-off, then caches it
  (downloading hundreds of times triggers rate limits and "possibly delisted" errors).

Either way the whole history is loaded **once** and cached in memory; each episode just *slices* it.

## The golden rule: no peeking at the future

Every data function uses a strict **as-of cutoff**: `get_candles` keeps only rows
`<= as_of_date`; `get_intraday` keeps only data up to the end of the as-of day;
`vix_context` keeps only VIX values up to the as-of day; `get_fundamentals` only returns filings
published on or before the as-of date. The *only* function allowed to look forward is
`get_forward_candles`, which is used **after** the decision, to grade it. Leaking the future
(look-ahead bias) would make a backtest look wonderful and be worthless.

## The indicators, explained one by one (`compute_technicals`)

**1. EMA-50, EMA-100, EMA-200.** Computed with `close.ewm(span=N, adjust=False)`. For EMA-200 to
mean what it says, the function needs **at least 200 bars** — that is why the simulator now feeds
it **250 bars** (`TECH_LOOKBACK_DAYS`). *(An earlier version used only 30 bars, so "EMA-200" was a
fiction. Fixed.)*

**2. EMA stack.** `bullish (50>100>200)`, `bearish (50<100<200)`, or `mixed`.

**3. VWAP.** `Σ(typical_price × volume) / Σ(volume)` where typical price = (high+low+close)/3,
computed over the last `LOOKBACK_DAYS` (30) bars only. A VWAP over all 250 bars would be
meaningless for a swing decision, so it has its own, shorter window.

**4. Swing pivots.** A day is a **swing high** if its high equals the maximum of a 5-bar window
(2 bars each side); a **swing low** likewise. Because a pivot needs 2 bars *after* it to be
confirmed, the most recent 2 bars can never be pivots — another quiet guard against look-ahead.

**5. Trend structure.** Look at the last three swing highs and three swing lows:

| Highs | Lows | Label | Tags |
|---|---|---|---|
| rising | rising | `uptrend` | HH, HL |
| falling | falling | `downtrend` | LH, LL |
| rising | falling | `expanding` | HH, LL |
| falling | rising | `contracting` | LH, HL |
| anything else / too few | | `ranging` | — |

**6. Support & resistance zones.** Swing-low prices that sit within 0.8% of each other are
clustered into a **support zone** (low, high, mid); swing-high prices cluster into **resistance
zones**. Support = zones *below* the current price; resistance = zones *above*. The **nearest** of
each is picked, and the 3 closest of each kind are kept.

**7. Level-break features (`_level_breaks`).** Reference levels and what *today's* candle did to them:

| Level | States |
|---|---|
| Previous-day high (PDH) | `broke` (close above), `rejected` (touched but closed below), `below` |
| Previous-day low (PDL) | `lost` (close below), `held` (dipped below but closed above), `above` |
| 2-day high/low | broken or intact |
| Last completed week's high/low | same states as PDH/PDL |

**8. India VIX context (`vix_context`).** Level, **percentile versus the last 252 days**, a **regime**
(`<13 LOW`, `<18 NORMAL`, `<25 ELEVATED`, else `EXTREME` — thresholds configurable), 5-day
direction (`rising`/`falling`/`flat` using a ±0.3 move), and where the level sits in its 20-day range
(`top`/`mid`/`bottom`). Optional: no VIX file → this block is simply omitted.

**9. Intraday zones (`compute_zones`).** From the raw **minute** data it builds 15-minute and
1-hour candles and asks `zones.py` for demand/supply zones near the current price. If a 15-minute
zone sits *inside* a 1-hour zone, that is flagged **confluence**. A daily-only data source has no
minute data, so this block is skipped.

**10. Fundamentals.** `FUNDAMENTALS_SOURCE=xbrl` reads official filings from
`data/fundamentals/{SYMBOL}/*.xml`. Fundamentals are *optional* (the project trains over ~15
years, and old filings are often missing) — no filing ⇒ an empty dict and the tag `none`. The tag is
`fresh` if the latest filing is ≤ ~140 days old, `stale` if older. With `yfinance` as the source, the
data is cached per quarter in the `fundamental_cache` table.

## The decision: who answers, and how the answer is policed

`pre_trade_decision()` builds the prompts (Part V), calls `_llm_generate()` (which routes to the
chosen backend), and parses the JSON — tolerating answers wrapped in markdown code fences or
surrounded by chatter. Then the rulebook is applied by **`sanitize_decision()`**. The AI is *never*
trusted blindly:

| The AI said… | What the code does | Why |
|---|---|---|
| `SELL` | Convert to **HOLD** ("Long-only: SELL not allowed") | No shorting |
| Anything not `BUY` | HOLD | Only BUY/HOLD exist |
| BUY with SL/TP/qty not numbers | HOLD ("Invalid BUY levels") | Unusable |
| BUY where it is *not* true that `0 < SL < entry < TP` | HOLD | Nonsense levels |
| BUY whose **recomputed** R:R `(TP−entry)/(entry−SL)` is below `MIN_RR` | HOLD | Models routinely state R:R wrongly, so it is recomputed from the real levels |
| BUY with a large quantity | Capped to `min(model qty, int(capital × 2% ÷ (entry−SL)), int(capital ÷ entry))` | Enforce the 2% risk rule and what cash can actually buy |
| BUY with no quantity | Use the maximum allowed by the same formula | |
| Quantity < 1 after capping | HOLD | Can't buy a fraction of nothing |
| `confidence` missing or outside 0–1 | Set to 0.5 | |

**Worked example.** Capital 10,000. Entry (today's close) 119.78. The model says SL 116.5, TP 125, qty 16.
- Check levels: 0 < 116.5 < 119.78 < 125 ✔
- R:R = (125 − 119.78) ÷ (119.78 − 116.5) = 5.22 ÷ 3.28 = **1.59** ≥ 1.5 ✔
- Risk cap: int(10,000 × 0.02 ÷ 3.28) = int(60.97) = **60**; affordable: int(10,000 ÷ 119.78) = **83**
- Quantity = min(16, 60, 83) = **16** ✔ → a valid BUY.

If the model had said TP = 123: R:R = 3.22 ÷ 3.28 = **0.98** → *rejected*, becomes
`HOLD` with the reason "R:R 0.98 below minimum 1.5".

### The rule-based fallback

If the AI backend returns nothing usable (or the backend is `"fallback"`), a simple rule fires:
BUY only if the EMA stack is **bullish**, price is **above VWAP**, and a support zone exists.
Then SL = support-low × 0.998; TP = the next resistance **but never more than 3× the risk away**
(an older version produced absurd R:R ≈ 47 targets that could never be reached in 10 days); BUY only
if the resulting R:R ≥ 1.5 — otherwise HOLD. It is deliberately simple: it exists so Round 1 can run
without an LLM, but its data is repetitive, mostly HOLD, and low quality.

## Grading the trade with real prices (`check_sl_tp`)

For a BUY, walk day by day through the next up-to-10 real candles:

1. If that day's **low ≤ stop-loss** → `SL_HIT`, exit at the stop-loss price.
2. Else if that day's **high ≥ take-profit** → `TP_HIT`, exit at the target.
3. If neither happens in 10 days → `TIME_EXIT`, exit at the 10th day's **close**.

> **Conservative tie-break:** SL is checked *before* TP each day. Daily candles can't tell which
> was touched first inside one day, so when both are possible the simulator assumes the **worse**
> outcome. This avoids flattering the model.

For a HOLD there is nothing to grade: no trade, `exit_reason = "HOLD"`, P&L 0.
**P&L for a BUY** = `(exit_price − entry_price) × quantity`. Capital is updated immediately.

## The post-trade lesson

After every BUY, a second prompt shows the AI the finished trade (entry, SL, TP, exit price, exit
reason, P&L, and its own earlier reasoning) and asks for JSON:
`outcome | exit_reason_analysis | what_went_right | what_went_wrong | lesson_learned | strategy_adjustment`.
If the model gives nothing usable, a template answer is used (e.g. "Cut losses at SL"). These lessons are
(a) shown to the agent as **history** on future days, and (b) used as **training examples**.

## Death threshold

`alive()` is true while capital ≥ `INITIAL_CAP × DEATH_THRESHOLD` (default 20% → 2,000 on a
10,000 start). The prompt shows the agent its own "death" line (`CAP:9948 (death<2000)`) so it
learns capital preservation. `mdd()` computes the maximum drawdown percentage of the run.

## The database: `trade_log`

Every episode becomes one row in SQLite (`trading_agent.db`):

| Column | Meaning |
|---|---|
| `id`, `episode`, `symbol`, `decision_date` | Which row / which day of the run / which stock / which date |
| `action` | `BUY` or `HOLD` |
| `entry_price`, `quantity`, `stop_loss`, `take_profit` | The trade plan |
| `exit_price`, `exit_date`, `exit_reason` | What really happened: `SL_HIT`, `TP_HIT`, `TIME_EXIT`, or `HOLD` |
| `pnl`, `capital_before`, `capital_after` | Money result |
| `pre_trade_reasoning` | The model's stated reasons |
| `post_trade_analysis`, `lesson_learned`, `what_went_wrong` | Short post-trade fields |
| `technicals_json` | A snapshot of **all the indicators the model saw**, plus `_fund` (the fundamentals) — so the *exact* prompt can be rebuilt later |
| `decision_json` | The model's **complete, sanitised decision** (real confidence, key_risk, features_used) — the training target |
| `post_json` | The model's complete post-trade analysis |
| `created_at` | Timestamp |

Two extra columns (`decision_json`, `post_json`) were added in the latest release; `init_db()`
auto-migrates older databases by adding them. A second table, `fundamental_cache`, stores
yfinance fundamentals per symbol/quarter.

---

# PART V — The Prompt: exactly what the AI is shown (`compact_prompt.py`)

A prompt has two halves. The **system prompt** is the standing rulebook; the **user prompt** is
today's situation.

## The system prompt (the rulebook)

`SYSTEM_PRE` (shortened here to its meaning):

> You are a disciplined NSE **swing** trader. **LONG-ONLY** — the only actions are BUY or HOLD, never SELL.
> Trades are held several days to a couple of weeks. Protect capital first. For BUY: `stop_loss < entry < take_profit`,
> **R:R ≥ 1.5**, **max risk 2%** of capital per trade. HOLD if the setup is unclear. Ground **every** decision
> in the *specific features given* (EMA stack, VWAP, PDH/PDL state, zones, VIX regime, fundamentals) — **never invent data**.
> Output JSON with keys `action | quantity | stop_loss | take_profit | risk_reward | reasoning | confidence | key_risk | features_used`.
> `reasoning` = 2–3 short clauses each citing a named feature. `features_used` = the exact feature names relied on,
> e.g. `["ema_stack","vwap","pdh_state","zone.1h.demand","vix.regime"]`.

`SYSTEM_POST` is the second rulebook, for after the trade: *review the completed long swing trade and
extract one concrete lesson*, output `outcome | exit_reason_analysis | what_went_right | what_went_wrong |
lesson_learned | strategy_adjustment`.

### Why this file is "the single source of truth"

Both `simulation.py` (when it *generates* data and when it *runs the model*) and `data_pipeline.py`
(when it *builds training files*) import these same strings and the same prompt-building
functions. That guarantees **the model is trained on exactly the prompt it will see in use.**
Previously the two files each wrote their own prompts — the model was trained on one style of
question and then asked another, which quietly hurts quality ("train/inference skew"). That bug
is fixed, and a test checks that every training example starts with one of the two shared system prompts.

## The user prompt: one real example, annotated

This is real output of `python compact_prompt.py` (made-up numbers for a Canara Bank day):

```
[CANBK 2024-10-10]                                  ← symbol and date
HEADLINE: bias:neutral | trend:downtrend            ← one-line verdict to anchor on
C:119.78 (+1.94%) | 5d:116.8-123.5 C@44pct          ← close, day change, 5-day range, where in range
ATR:3.26 | Vol:4.8M(1.0x avg) | 3d:-0.35% | body:0.39
REL: vsEMA200:-2.2% | vsEMA50:+0.7% | vsVWAP:+0.5% | ATR%:2.72% | toSup:0.9ATR | toRes:-0.91ATR | 5dPos:44pct

EMA: 50@118.9 < 100@120.1 < 200@122.5 [MIXED]       ← the EMA stack
VWAP:119.2 C+0.49% | C vs EMA200:-2.22%
Trend:downtrend | Struct:LH,LL

S:116.5-117.2(-2.5%)  (nearest -2.5%)               ← support zone, % from price
R:122.0-123.5(+2.5%)  (nearest +2.48%)              ← resistance zone
SH:123.5(10-07) | SL:116.8(10-09)                   ← last swing high / low

Fund(2024-Q2): PAT ₹5381cr | Rev ₹9200cr | P/E 9.2  ← fundamentals (if available)

HISTORY:                                            ← the agent's own last ≤5 trades
EP  ACT   RESULT      P&L    CAP     LESSON
01  BUY   SL_HIT         -52     9948  wait for EMA alignment

CAP:9948 (death<2000)                               ← capital, and the "death" line
```

Additional blocks appear automatically when the data exists: a **LEVELS** line
(PDH/PDL/2-day/weekly states), a **VIX** line, and a **ZONES** block (15-minute and 1-hour demand/supply
zones plus `CONFLUENCE`). The whole prompt is about **350–420 tokens** — far smaller than the earlier
verbose format, which matters because shorter prompts are cheaper, faster and leave more room in memory.

## "REL" — scale-free features (a clever trick)

A price of "119.78" means nothing across stocks: ₹120 for a bank might be ₹4,000 for another company.
The **REL** line re-expresses the situation as *percentages and ATR multiples*:

- `vsEMA200:-2.2%` — price is 2.2% below its 200-day average.
- `toSup:0.9ATR` — price is 0.9 typical-daily-moves above support.
- `toRes:-0.91ATR` — the nearest resistance is ~0.9 daily moves *above* (negative sign = the level is above price).

So the model can learn "near support with a bullish stack tends to work" instead of memorising
"119.78". Raw prices are still included because the model needs them to write exact SL and TP
numbers. The REL line can be switched off with `RELATIVE_ENCODING=0`.

## The HEADLINE

`bias` is **bullish** if the EMA stack is bullish *and* price is above VWAP; **bearish** if the reverse;
otherwise **neutral**. It also lists the trend, PDH `broke`/`rejected`, the VIX regime, and
`confluent-demand`/`confluent-supply`. It acts like the headline of a news article: the model gets
the gist first, then the details.

---

# PART VI — Turning trades into training data (`data_pipeline.py`)

## Input data vs training data

- **Input data** = raw market facts (candles, fundamentals, indicators) — the *question* side.
- **Training data** = the *question plus the correct answer* packaged in the model's chat format:

```json
{"messages":[
  {"role":"system",    "content":"You are a disciplined NSE swing trader. LONG-ONLY ..."},
  {"role":"user",      "content":"[CANBK 2024-10-10]\nHEADLINE: bias:neutral | ..."},
  {"role":"assistant", "content":"{\"action\":\"HOLD\",\"quantity\":0, ... }"}
]}
```

The model learns to produce the `assistant` part when shown the `system` + `user` part.

## Two kinds of example per day

| Type | Made for | Input → Output |
|---|---|---|
| **pre_trade** | *every* episode (BUY and HOLD) | market situation + history → decision JSON |
| **post_trade** | BUY episodes with a real lesson | finished trade → lesson JSON |

## The pipeline, step by step

```
trade_log (SQLite)
   │ 1. load_all_rows()          keep BUY & HOLD only (legacy SELL rows dropped)
   ▼
rows
   │ 2. build_examples()         walk in order; rebuild prompts; keep per-run history
   ▼
raw examples (pre + post)
   │ 3. quality_filter()         drop weak / malformed / non-long examples
   ▼
   │ 4. balance_and_dedup()      de-dup by (type, symbol, date); cap HOLDs
   ▼
   │ 5. split (time or random)   train / val / test
   ▼
trading_train.jsonl   trading_val.jsonl   trading_test.jsonl  (test keeps `_meta`)
```

**Step 2 — building examples.** For each row the prompt is rebuilt from the stored
`technicals_json` and `_fund`, with the same `build_compact_user()` the simulator used. The
**history** shown is the previous episodes *of the same run* (HOLDs included) — exactly what the
agent saw. A new run is detected when `episode == 1`, which resets the history. The **target**
(`decision_target`) is the teacher's *real* stored `decision_json`, with its real confidence,
key_risk and features_used; only `risk_reward` is recomputed from the actual levels. *(Older code
hard-coded confidence and dropped `features_used`. Rows from old databases without `decision_json`
are reconstructed as a fallback, using confidence 0.5.)*

**Step 3 — the quality filter**

| Example | Kept if… |
|---|---|
| Malformed assistant JSON | never |
| *(optional)* `REQUIRE_FUNDAMENTALS=1` | the fundamentals tag is `fresh` |
| pre_trade **HOLD** | always (volume is limited in step 4) |
| pre_trade **BUY** | `confidence ≥ 0.5`, `risk_reward ≥ MIN_RR`, and `0 < SL < TP` |
| pre_trade anything else (e.g. SELL) | never |
| post_trade | the lesson is not a placeholder (`—`, empty, `N/A`, "Analysis unavailable…") |

**Step 4 — de-duplicate and balance**
- *De-dup key:* `(type, symbol, date)`. The **latest** row wins — so a newer model's decision for
  a given day replaces an older round's. (An older version keyed on fields that were all `None`, which
  collapsed *every* post-trade example into one.)
- *HOLD cap:* pre-trade HOLDs are limited to `HOLD_RATIO × (number of BUYs)` (default 0.5×). Why?
  Most days are HOLD. Without a cap, a model learns that "HOLD" is the safest answer to everything.
  *But* some HOLDs are kept so it also learns when to **stay out**. Post-trade examples are untouched.

**Step 5 — the split.** The three buckets:

| Split | Typical share | Purpose | Analogy |
|---|---|---|---|
| **Train** | 80% | What the model learns from | Homework |
| **Validation (val)** | 10% | Checked *during* training to spot memorising | Weekly quizzes |
| **Test** | 10% | Scored **once at the end** | The final exam |

`SPLIT_MODE=time` (the default) is **walk-forward**: examples are sorted by date, the **earliest
80% train, the next 10% validate, the *latest* 10% test**. The model is always examined on days
*after* everything it studied — just like real trading. `SPLIT_MODE=random` shuffles instead
(easier, but it lets the model "peek" at neighbouring days, which flatters it).

The **test file keeps a `_meta` field** (action, P&L, date, exit reason…) so `evaluate.py` can
compute trading metrics. Train and val are written *without* `_meta`.

## A real run of this pipeline (the offline end-to-end test)

The test generates 700 days of synthetic prices, plays 150 episodes with the rule-based fallback, and builds a dataset:

```
Loaded 150 long-only episodes (13 BUY, 137 HOLD)
Built 163 raw examples                (150 pre-trade + 13 post-trade)
After quality filter: 163
After de-dup: 163  (0 duplicates)
After balance: 32  (BUY 13 / HOLD 6 [cap 0.5x BUY] / post-trade 13)
Split=time:  train 26 | val 3 | test 3
⚠ WARNING: < 50 examples is very few for fine-tuning.
```

Notice how 137 HOLDs shrank to 6: that is the HOLD cap at work. Notice the warning — **real
training needs hundreds of examples** (the README aims for 500+).

## Looking at the data with your own eyes

- `python show_training_data.py --db trading_agent.db --out viz/training_data.html` — an HTML page of every
  sample *exactly* as the model sees it (it calls the real pipeline functions, so it can't drift).
- `python build_data_explorer.py --db trading_agent.db` (or `--jsonl ...` or `--demo`) — a single offline
  HTML table with sortable columns, a search box, and click-to-expand rows.

---

# PART VII — The Models: types, and why some can train and some can't

## The teacher / student pattern

- **Teacher** = a strong model that *generates* the decisions. Here: `nemotron-bolt-e4b-16k` running
  in **Ollama** (a custom 16k-context version).
- **Student** = the model we *fine-tune*. Here: `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit`.
- **Rule-based fallback** = a third "decider", for smoke tests only.

**Why not let the model teach itself?** If a small model generates its own training data and
trains on it, its biases amplify each round until it collapses into repetitive nonsense ("model
collapse"). Two things prevent this here: an *independent* teacher in Round 1, and **real price
outcomes** that record whether a decision actually worked.

> **A honest wrinkle:** the training target is the *decision* the model made — whether it won
> or lost. The system keeps losers, too. The market's verdict is recorded (P&L, exit reason, lesson) and used
> for evaluation and post-trade lessons, but training currently just imitates the decisions
> that pass the quality filter. Turning the outcome into a *training signal* is the job of
> preference-learning (DPO) — a sketch of which exists in the wider project (`dpo_sketch.py`, not in this upload).

## Why can some models be fine-tuned and others not?

1. **Format.** Ollama models are **GGUF** (built for *running*). You cannot LoRA-train a GGUF. You need the
   **safetensors** weights — which is what `mlx-community` publishes.
2. **Architecture.** LoRA attaches to standard attention layers (`q_proj`, `k_proj`…). Plain dense
   Llama/Mistral/Nemotron-style models expose them cleanly. Exotic ones (big **Mixture-of-Experts**,
   **Mamba hybrids**) may not.
3. **Size vs memory.** A 30B model needs ~15 GB just to *load* at 4-bit — no room to train on a 16 GB Mac.
   A ~4B model fits with room for adapters and optimiser state.

## Inference vs training: which model format?

| Need | Prefer | Why |
|---|---|---|
| **Inference** | GGUF in Ollama, quantised MLX, or vLLM on a GPU | Compact and fast; no gradients needed. |
| **Training** | safetensors in MLX or PyTorch | Training needs the weight structure and gradients. |

So the *same* model plays both roles in different formats: **safetensors to train**, then convert to **GGUF to serve**.

## The three "decider" backends in code

`set_llm_backend(mode, adapter_path)` switches `_llm_generate()`:

| Mode | What happens | Needs |
|---|---|---|
| `fallback` | Returns nothing → rule-based decision is used | Nothing |
| `ollama` | POSTs to `OLLAMA_URL` with the model `OLLAMA_MODEL`, 8192-token context, 180 s timeout | Ollama running |
| `adapter` | Lazily loads the fine-tuned model once (MLX first, else HF) | `inference_mlx.py` *(not in this upload)* |

On *any* error a backend prints a message and returns nothing, so the rule-based fallback takes over —
**the simulation never crashes because a model hiccuped.** (`inference_vllm.py` is a separate async client
for a vLLM server, used for production serving rather than the simulation loop.)

---

# PART VIII — Training Internals (the engine room)

> The training script itself, `train.py`, is **not in this upload**; `flywheel.py` calls it. This part
> explains the ideas behind it.

## PyTorch vs MLX — two engines, same idea

- **PyTorch** = the industry-standard framework, best on **NVIDIA GPUs** (CUDA). On a Mac it uses "MPS", which
  is slower and where libraries like `bitsandbytes` (4-bit) **don't work**.
- **MLX** = Apple's framework for **M-series chips**, using unified memory. On a MacBook it's the native, fast path.

Same maths underneath; different hardware backend. `train.py` auto-selects.

## LoRA — teaching without rewriting the whole brain

The model has **billions** of weights. Full fine-tuning would need ~60 GB. **LoRA** freezes the original and
adds two tiny matrices (A and B) beside each attention layer. Only those train — roughly **0.05–1%** of the parameters.

```
output = W·x        +   (B · A · x)·scale
         ↑ frozen giant     ↑ tiny trainable adapter (what learns)
```

An example log line: `Trainable parameters: 0.047% (3.473M / 7463M)` — only 3.5 million of 7.46 **billion** weights were trained.

**Rank (`LORA_R`)** = the "width" of the patch: bigger = more capacity but more memory. **`MLX_NUM_LAYERS`** =
how many of the model's layers get a patch.

## QLoRA — LoRA on a compressed model

Even a frozen giant needs memory. **QLoRA** keeps the frozen base in **4-bit** (4 bits per weight instead of 16),
cutting memory about 4×, while adapters stay in full precision. The `...OptiQ-4bit` model is already 4-bit, so
MLX is effectively doing QLoRA: a 7.46B model trains in about **5.5 GB**.

## Adapter vs Fused model

- **Adapter** = just the trained A/B matrices — a small file (tens of MB). Useless alone; a "patch" that needs its base model.
- **Fused model** = base **+** adapter merged (`W_new = W + B·A`) into one standalone model. Bigger but self-contained.
  **`flywheel.py` prefers the fused model**, because the next round can both *load it for inference* and
  *continue training from it*. (An older version picked the raw adapter folder, which can't be loaded without its base.)

## `MLX_ITERS` — how long training runs

`MLX_ITERS=600` = 600 training steps. Each step shows a small batch, measures error, nudges the adapter.
More steps = more learning (until overfitting). Use ~50 for a smoke test, ~600 for a real run. `SAVE_EVERY=100` writes a checkpoint every 100 steps.

## `max-tokens` — a *generation* setting, not a training one

When *running* the model, `max-tokens` caps how long its answer may be. One JSON decision fits in a few
hundred tokens (`evaluate.py` uses 256). It has nothing to do with training.

## What is "loss"? What is "val" loss?

**Loss** = one number for "how wrong is the model?". We use **cross-entropy**: for each token the model
should produce, loss is high if it gave that token a low probability.

```
loss for one token = −log( probability the model gave to the CORRECT token )
```

Example: the correct next token is `BUY`.
- Model says P(BUY)=0.9 → loss = −log(0.9) ≈ **0.10** (good).
- Model says P(BUY)=0.1 → loss = −log(0.1) ≈ **2.30** (bad).

The batch loss is averaged over the **assistant tokens only** — the model isn't graded on repeating the question.

- **train loss** = on the batch it's studying. Falls as it learns *or memorises*.
- **val loss** = same formula on the held-out validation set. The honest "is it generalising?" number.

**Reading the numbers:**

| Train | Val | Meaning |
|---|---|---|
| 0.40 | 0.45 | Healthy — learning and generalising |
| 0.05 | 0.90 | **Overfitting** — memorised homework, struggles on new questions |
| 0.01 | 0.01 | *Suspicious* — usually means the targets were trivially repetitive (e.g. rule-based, mostly-HOLD data) |

On real, varied LLM-generated data, val loss tends to settle around 0.3–0.8. Too low can mean "too easy", not "too good".

## One full training step

```
Batch: 2 examples (each = system + user + assistant JSON)
 1. FORWARD : feed tokens through the 4-bit frozen model + LoRA adapters → probabilities
 2. LOSS    : cross-entropy on assistant tokens only  (e.g. 0.42)
 3. BACKWARD: work out how to tweak A and B to lower that number (gradients)
 4. UPDATE  : the Adam optimiser nudges A, B (the base stays frozen)
 5. every ~50 steps  → compute val loss on held-out data
 6. every SAVE_EVERY → save a checkpoint
repeat MLX_ITERS times → save adapter → fuse → done
```

---

# PART IX — Evaluation: how we grade the model (`evaluate.py`)

There are two very different questions: *"Does the model speak the format correctly?"* and
*"Would its trades have made money?"* `evaluate.py` answers both, on the **test split** — dates
the model never trained on.

## A) Model-quality metrics (no market replay)

For each test example, the model is given the system + user prompt and its answer is compared with the stored one:

| Metric | What it measures |
|---|---|
| `json_validity` | % of answers that parse as JSON |
| `schema_ok` | % that contain all required keys: `action, stop_loss, take_profit, risk_reward, reasoning, confidence` |
| `reasoning_ok` | % with a non-empty reasoning |
| `features_used_ok` | % with a non-empty `features_used` list |
| `action_match` | % where the model's BUY/HOLD agrees with the reference decision |
| `long_only_ok` | % of answers that are BUY or HOLD (no SELL) |

*Run with no `--model` and the script does a **harness self-check**: it echoes the stored answers as if
they were the model, so you should see ~100% — proof that the grading code itself works.*

## B) Trading metrics (pure maths on recorded P&L)

These use the **BUY** trades in the test set (HOLDs are *excluded* — they aren't trades, and counting their
zero P&L would flatter the win-rate and distort the Sharpe).

**Worked example.** Five trades: **+84, −66, +120, −40, +60**

| Metric | Formula | Result |
|---|---|---|
| `n` | number of trades | 5 |
| `win_rate` | wins ÷ trades | 3/5 = **60%** |
| `avg_win` | total wins ÷ #wins | 264 ÷ 3 = **88** |
| `avg_loss` | −(total losses ÷ #losses) | **−53** |
| `expectancy` | average P&L per trade | 158 ÷ 5 = **31.6** |
| `profit_factor` | gross wins ÷ gross losses | 264 ÷ 106 = **2.49** |
| `sharpe_per_trade` | mean ÷ standard deviation | 31.6 ÷ 72.1 = **0.44** |
| `total_pnl` | sum | **158** |
| `max_drawdown` | worst fall from a peak of the running total | running total 84, 18, 138, 98, 158 → worst drop **−66** |

How to read them: **expectancy > 0** means the strategy makes money per trade on average. **Profit factor > 1** means wins outweigh losses.
A 40% win rate can still be profitable if wins are much bigger than losses (that's why R:R ≥ 1.5 matters).
The Sharpe here is *per trade*, not annualised.

> ⚠ With only a handful of test trades these numbers are **noisy**. Ten trades can't prove a strategy works.

---

# PART X — One trade's whole life (with real arithmetic)

```
DAY:      2024-10-10, CANBK   (episode 12, capital 10,000)

1. DATA   250 daily candles up to today → EMA-50/100/200, VWAP, swings, zones, PDH/PDL,
          VIX regime, fundamentals (PAT ₹5,381cr, P/E 9.2) ; all strictly "as of" today.

2. PROMPT compact_prompt builds ~400 tokens, with HEADLINE, REL, EMA, zones, last 5 trades.

3. MODEL  (teacher) answers:
          BUY, qty 16, SL 116.50, TP 125.00, confidence 0.72,
          reasoning "bullish stack, above VWAP, demand-zone support",
          features_used ["ema_stack","vwap","nearest_support"]

4. POLICE sanitize_decision():
          0 < 116.5 < 119.78 < 125            ✔
          R:R = (125−119.78)/(119.78−116.5) = 1.59 ≥ 1.5 ✔
          qty = min(16, risk-cap 60, affordable 83) = 16 ✔     → keep BUY

5. REPLAY the real next ≤10 trading days (the model never saw these):
          day 1  low 118.1 high 121.0   neither level touched
          day 2  low 118.6 high 122.4   neither
          day 3  low 119.9 high 125.3   high ≥ 125 → TP_HIT at 125.00

6. P&L    (125.00 − 119.78) × 16 = +83.52      capital 10,000 → 10,083.52

7. LESSON model reviews the finished trade → lesson_learned: "Demand-zone longs in a
          bullish stack worked; wait for VWAP confirmation."

8. STORE  one row in trade_log (+ decision_json, post_json, technicals_json)

9. LATER  data_pipeline turns it into 2 examples (decision + lesson), filtered (conf 0.72 ≥ 0.5,
          R:R 1.59 ≥ 1.5), de-duplicated, and placed in train/val/test by DATE.

10. TRAIN the student learns to emit that BUY JSON from that prompt.

11. EXAM  on later, unseen dates: valid JSON? sensible action? positive expectancy?
```

*(The three replay days in step 5 are illustrative.)* What if price had instead fallen to 116.5 first? →
`SL_HIT`, exit at 116.50, P&L = (116.50 − 119.78) × 16 = **−52.48**. If neither level had been touched in 10 days →
`TIME_EXIT` at the 10th day's close.

---

# PART XI — Every setting (`.env`)

Copy `.env.example` to `.env`. **No code editing needed.** (CLI flags such as `--symbol` override `.env`.)

## Model & teacher
| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL` | `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit` | The student to fine-tune (use an ungated `mlx-community` 4-bit repo on Mac). |
| `USE_OLLAMA` | `1` | `1` = LLM teacher via Ollama generates data; `0` = rule-based fallback (smoke test only). |
| `OLLAMA_MODEL` | `nemotron-bolt-e4b-16k` | Which Ollama model is the teacher. |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama endpoint. |

## What to simulate
| Variable | Default | Meaning |
|---|---|---|
| `SYMBOL` | `CANBK` | NSE symbol for this run (re-run with others to accumulate data). |
| `START_DATE` | `2024-01-10` | First simulated day. **Every round restarts here.** |
| `ROUNDS` | `5` | Times around the flywheel. |
| `EPISODES` | `50` | Simulated days per round. |

## Data sources
| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `csv` in `.env.example` (`yfinance` if unset in code) | Where prices come from. |
| `CSV_DIR` | `data/prices` | Looks for `{CSV_DIR}/{SYMBOL}.csv`. |
| `CSV_PATH` | *(empty)* | Explicit single CSV; overrides `CSV_DIR`. |
| `FUNDAMENTALS_SOURCE` | `xbrl` in `.env.example` (`yfinance` if unset) | Official filings vs Yahoo. |
| `XBRL_DIR` | `data/fundamentals` | Looks for `{XBRL_DIR}/{SYMBOL}/*.xml`. |
| `VIX_CSV` | blank in code; `data/prices/INDIAVIX.csv` in `.env.example` | India VIX history (optional; blank = no VIX block). |
| `VIX_LOW / VIX_NORMAL / VIX_ELEVATED` | `13 / 18 / 25` | VIX regime thresholds. |

## Swing-trading rules
| Variable | Default | Meaning |
|---|---|---|
| `MAX_HOLD_DAYS` | `10` | A BUY exits after at most this many trading days. |
| `TECH_LOOKBACK_DAYS` | `250` | Daily bars fed to the indicators (EMA-200 needs ≥ 200). |
| `LOOKBACK_DAYS` | `30` | Window for VWAP. |
| `RISK_PER_TRADE` | `0.02` | Max fraction of capital to risk on one trade. |
| `MIN_RR` | `1.5` | Minimum reward:risk for a BUY. |
| `INITIAL_CAP` | `10000` | Starting capital. |
| `DEATH_THRESHOLD` | `0.20` *(code default)* | Agent stops when capital < this × `INITIAL_CAP`. |
| `RELATIVE_ENCODING` | `1` *(code default)* | Include the scale-free `REL:` prompt line. |

## Dataset
| Variable | Default | Meaning |
|---|---|---|
| `SPLIT_MODE` | `time` | `time` = walk-forward (train earliest, test latest); `random` = shuffled. |
| `HOLD_RATIO` | `0.5` | Keep at most this × (#BUY) pre-trade HOLD examples. |
| `DEDUP_DATA` | `1` | De-duplicate by (type, symbol, date); latest wins. |
| `BALANCE_DATA` | `1` | Apply the HOLD cap. |
| `REQUIRE_FUNDAMENTALS` | `0` *(code default)* | If `1`, keep only examples whose fundamentals are `fresh`. |

## Training
| Variable | Default | Meaning |
|---|---|---|
| `MLX_ITERS` | `600` | Training steps. |
| `LORA_R` | `16` | LoRA rank. |
| `MLX_NUM_LAYERS` | `8` | Layers that get adapters. |
| `EPOCHS` | `2` | Passes over the data (passed to `train.py`). |
| `SAVE_EVERY` | `100` | Checkpoint interval. |

## Paths
| Variable | Default | Meaning |
|---|---|---|
| `DB_PATH` | `trading_agent.db` | SQLite file. |
| `ADAPTER_DIR` | `output/adapters` | Where trained models go. |
| `DATA_DIR` | `data/rounds` | Where JSONL files go. |

---

# PART XII — How to run it, step by step

## What you need first
1. **A Mac with Apple Silicon** (MLX path) — or an NVIDIA GPU for the QLoRA path.
2. **Python 3.12** and a virtual environment.
3. **Price data:** a CSV at `data/prices/{SYMBOL}.csv` (daily or 1-minute OHLCV). Optionally a VIX CSV and XBRL filings.
4. **Ollama** with the teacher model (`ollama pull nemotron-bolt-e4b-16k`) if you want real teacher data.
5. **The complete project folder** — this upload is missing `config.py`, `train.py`, `zones.py` and others (see Part II).

## Step 0 — one-time setup
```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install mlx-lm                 # Mac training backend
ollama pull nemotron-bolt-e4b-16k  # the teacher
cp .env.example .env               # then edit values
```

## Step 1 — smoke test (~10 minutes): prove the plumbing
Set in `.env`: `USE_OLLAMA=1  ROUNDS=1  EPISODES=20  MLX_ITERS=50`, then:
```bash
rm -f trading_agent.db && rm -rf output/adapters data/rounds      # clean slate
PYTHONWARNINGS=ignore python flywheel.py
ls -la output/adapters/round_01/fused_model                       # model produced?
```
With 50 iterations quality will be rough — you're testing *plumbing*, not performance.

## Step 2 — the real run
Set `ROUNDS=5  EPISODES=50  MLX_ITERS=600`, clean slate, run `python flywheel.py`.
**Healthy signs:** a *mix* of BUY and HOLD (not all HOLD), validation loss settling roughly 0.3–0.8 (not ~0.01).

## Step 3 — get enough data
Each episode yields about 1–2 examples before filtering. Run several symbols (they all append to the same database):
```bash
for SYM in CANBK SBIN HDFCBANK ICICIBANK AXISBANK PNB BANKBARODA; do
  SYMBOL=$SYM START_DATE=2023-06-01 ROUNDS=1 EPISODES=80 PYTHONWARNINGS=ignore python flywheel.py
done
```

## Step 4 — look at and grade the results
```bash
python show_training_data.py --db trading_agent.db --out viz/training_data.html
python build_data_explorer.py --db trading_agent.db
python evaluate.py --test data/rounds/round_01/trading_test.jsonl --model output/adapters/round_01/fused_model --pnl-from-meta
```

## Step 5 — deploy back to Ollama
Convert the fused MLX model to GGUF (needs llama.cpp tools), write a `Modelfile` with `num_ctx 16384`,
then `ollama create`. The 16k context is a *runtime setting* re-applied here — fine-tuning never changes it.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **0 training examples** or "Too few examples" after upgrading | Your **old `trading_agent.db` has only short (SELL) trades and no `decision_json`** — it yields no long-only examples | Delete `trading_agent.db`, `data/rounds`, `output/adapters` and re-run |
| Everything is HOLD | Ollama isn't running, or `USE_OLLAMA=0` | `ollama ps`; `ollama pull ...`; set `USE_OLLAMA=1` |
| Val loss ≈ 0.01 | Trained on rule-based data | Use the LLM teacher |
| `Training failed — no adapter found` | `train.py` didn't produce output | Mac: `pip install mlx-lm`. NVIDIA: `pip install torch peft trl bitsandbytes` |
| `CSV not found: data/prices/XYZ.csv` | Missing price file | Add the CSV or set `CSV_DIR`/`CSV_PATH` / `DATA_SOURCE=yfinance` |
| `No candle data for XYZ on/before DATE` | `START_DATE` is earlier than your CSV begins | Pick a later start date |
| `No module named 'config'` / `'zones'` / `'fundamentals_xbrl'` | Missing files (not in this upload) | Use the full project folder |
| `No module named 'torch'` on a Mac | No CUDA stack | Normal — use MLX |
| `401 Unauthorized` on model download | Gated repo | Use an `mlx-community` ungated repo, or log in to Hugging Face |
| Pandas/yfinance warning spam | yfinance internals | Prefix commands with `PYTHONWARNINGS=ignore` |
| Fewer than 50 examples warning | Too little data | More episodes, more symbols |

---

# PART XIII — The tests (`tests/test_build.py`)

Run with `python tests/test_build.py` (or `pytest tests/test_build.py -q`). Heavy libraries (`yfinance`, `requests`) are *stubbed*, so tests run anywhere.

| Test | What it proves |
|---|---|
| `test_all_files_compile` | Every `.py` file parses — no syntax errors. |
| `test_core_imports` | The core modules import cleanly. |
| `test_trading_metrics` | The P&L maths (win rate, profit factor, drawdown…) is right. |
| `test_balance_and_dedup` | De-dup keeps the latest and the HOLD cap works. |
| `test_quality_filter_long_only` | SELL and low-R:R / low-confidence examples are rejected. |
| `test_sanitize_decision_long_only` | SELL→HOLD; bad levels→HOLD; R:R recomputed; quantity capped. |
| `test_zones` | Demand/supply zone detection. |
| `test_config_env` | `.env` loading. |
| `test_prompt_renders` | The compact prompt renders. |
| `test_xbrl_parse` | XBRL filing parsing. |
| `test_end_to_end_offline` | **The big one:** synthetic prices → 150 simulated days → dataset → evaluation. Checks: EMA-200 uses ≥ 200 bars; no SELL ever stored; every row has `decision_json`; training prompts use the shared system prompts; no `_meta` in train; `_meta` present in test; metric count matches BUY count. |

**Status of this upload:** the release notes record **11/11** passing in the full project. In *this* zip, **7 of 11 pass**; the other four
(`test_core_imports`, `test_config_env`, `test_zones`, `test_xbrl_parse`) fail only with `ModuleNotFoundError` because
`config.py`, `zones.py` and `fundamentals_xbrl.py` are not included. The logic-bearing tests — sanitiser, filter, de-dup, metrics,
prompt, and the full offline end-to-end run — all pass.

---

# PART XIV — Bug history: what was wrong, how it was found, how it was fixed

This is the story of the latest release (`CHANGES.md`), in plain language. Reading it teaches *how ML pipelines silently go wrong*.

| # | The bug | Why it mattered | The fix |
|---|---|---|---|
| 1 | **SELL trades were never usable.** `risk_reward` was 0.0 for SELL, so the quality filter dropped every one → datasets contained only a fraction of what was generated, and empty datasets were possible. | Silent data loss. | SELL removed *by design*; BUY R:R computed from real levels. |
| 2 | **Train/inference prompt mismatch.** The data builder and the simulator wrote *different* prompts. | The model practiced on one question style and was tested on another. | Both now use `compact_prompt.py`; fundamentals are stored in the DB so the exact prompt can be rebuilt. |
| 3 | **Hard-coded confidence; missing `features_used`.** Targets were reconstructed with a fake confidence. | The model learned fake confidence and never learned to cite features. | The teacher's full decision is stored as `decision_json` and used as the target. |
| 4 | **"EMA-200" on 30 bars.** | An indicator named EMA-200 that wasn't. | `TECH_LOOKBACK_DAYS=250`; VWAP has its own 30-bar window. |
| 5 | **Stale docs.** | Documentation described old behaviour. | README/flow/STORYBOOK/TIMELINE updated; `.env.example` added. |
| 6 | **Broken de-dup.** The key was all-`None`, collapsing *all* post-trade examples into one; it also used price-scale-dependent bins. | Throwing away nearly all lessons. | Keyed by `(type, symbol, date)`, latest wins. |
| 7 | **Test split had no `_meta`.** | `evaluate.py` trading metrics were always empty. | Test split keeps `_meta`; HOLDs excluded from P&L metrics. |
| 8 | **Flywheel merged all rounds' train files.** | Each round already contains everything → every example duplicated N times, and earlier "train" splits contained dates later used as "test" (**data leakage**). | Use only the latest round's file. |
| 9 | **Flywheel picked the raw `adapters` folder as the model.** | Can't be loaded without its base model. | Prefers `fused_model`. |
| 10 | **Viewers showed stale prompts.** | You'd inspect data the model never saw. | They now call the real pipeline functions. |
| 11 | **Unrounded `close` in prompts.** | Noisy, token-wasting numbers. | Rounded to 2 decimals. |
| 12 | **Rule-based fallback produced R:R ≈ 47.** | Unreachable targets. | Target capped at 3× the risk. |
| 13 | **Hold period 5 days.** | Too short for "swing". | `MAX_HOLD_DAYS = 10`. |

**Lesson:** several of these (2, 6, 7, 8) were *silent* — nothing crashed; the results just quietly got worse or meaningless. That's why this project has tests and why evaluation matters.

---

# PART XV — Honest limitations of the current code

A good engineer states what *isn't* solved. These come from reading the code, not guesswork.

**Known by the maintainers (`CHANGES.md`)**
1. **`prefect_flows.py` passes `--adapter` to `flywheel.py`, which has no such flag.** The scheduled-run wrapper would fail until one side is fixed.
2. **Every round restarts at `START_DATE`.** Same-day decisions from later rounds *replace* earlier ones; to truly add data, vary symbols/dates.
3. **Your old database must be deleted** after upgrading (it holds only short trades and has no `decision_json`).

**Found while reading the code**
4. **Overlapping trades are not modelled.** One decision is made every trading day, and each BUY is graded independently, with capital updated immediately — even though a trade can last up to 10 days. In reality, you'd have fewer free funds while a trade is open. Treat simulated P&L as a *quality signal for each decision*, not an exact account curve.
5. **A BUY with no forward data is recorded as `HOLD`-exit with 0 P&L.** If the simulation reaches the end of your price file, trades near the end quietly become no-ops.
6. **No trading costs.** No brokerage, taxes, slippage, or the market impact of your own orders.
7. **Daily-candle ambiguity.** When a day's range includes both SL and TP, SL is assumed first (conservative, but still an assumption).
8. **The "HOLD cap" can erase all HOLDs.** The cap is `HOLD_RATIO × #BUY`; if a round has **zero BUYs**, the cap is 0 and *every* HOLD is dropped, leaving no pre-trade examples.
9. **Training imitates decisions, not outcomes.** Losing trades that pass the confidence/R:R filter are still training targets. Outcome-based learning (e.g. DPO) isn't wired in here.
10. **Small data.** The smoke-test dataset above has 32 examples. Fine-tuning on a few dozen examples teaches *format*, not *skill*.
11. **`DEATH_THRESHOLD` is configurable, but the prompt text always shows "death<20% of capital".** Change one without the other and the prompt line will disagree with the rule.
12. **Single symbol per run, one market (NSE), mostly banks.** Generalisation to other sectors/regimes is untested.
13. **Missing pieces in this upload** (Part II) mean the full loop can't run from this zip alone.

---

# PART XVI — Pros, Cons, Alternatives, Industry Standards

## Pros of our approach
- **Cheap, self-generating data** from simulation + real-price grading — no manual labelling.
- **Runs on a laptop** (16 GB Apple Silicon) thanks to QLoRA + MLX.
- **Honest labels** — the market grades every trade.
- **Rules enforced in code, not trusted to the AI** (`sanitize_decision`) — the model can't output illegal trades.
- **Train/inference parity** — the same prompt functions on both sides.
- **No look-ahead** by construction — strict as-of cutoffs and a **walk-forward** split.
- **Config-driven** (`.env`) — reproducible, no code edits.
- **Resilient** — any model or training failure falls back gracefully instead of crashing.
- **Tested** — including a full offline end-to-end run.
- **MLOps-ready** — MLflow tracking, DVC, Prefect, Evidently hooks exist in the wider project.

## Cons / limitations
See Part XV. In short: backtest ≠ live; no costs; small data; teacher ceiling; imitation rather than outcome-optimisation; not safe for real money.

## What could be improved
- **More and more diverse data:** 10+ symbols, multiple regimes (bull/bear/sideways), 2,000+ examples; then raise LoRA rank.
- **Outcome-aware training:** DPO/RLHF-style learning using real P&L (win vs loss) as the preference signal.
- **A proper portfolio simulator:** overlapping positions, cash tracking, costs, slippage.
- **Walk-forward across years** with rolling re-training and out-of-sample reporting per regime.
- **Richer reporting:** Sharpe, drawdown and win-rate *with confidence intervals*, so tiny samples aren't over-interpreted.
- **Guardrails** for any live use: position/portfolio limits, circuit-breakers, paper trading first.

> *Already done since the earlier storybook:* walk-forward splitting (`SPLIT_MODE=time`) and Sharpe/drawdown metrics are now implemented.

## Alternatives we could have chosen

| Decision | We chose | Alternatives | Why our choice |
|---|---|---|---|
| Fine-tune method | **QLoRA (LoRA + 4-bit)** | Full fine-tune; prompt-only; RAG | Full FT needs ~60 GB; prompting is inconsistent; QLoRA fits a laptop. |
| Framework (Mac) | **MLX** | PyTorch+MPS; llama.cpp finetune | Native and fast on M-series; MPS lacks 4-bit support. |
| Framework (GPU) | **PyTorch + PEFT/TRL + bitsandbytes** | JAX; raw CUDA | Industry standard. |
| Serving | **Ollama (local) / vLLM (GPU)** | TGI; raw transformers | Ollama = simplest; vLLM = best throughput. |
| Data labels | **Real price outcomes** | Human labels; self-labels | Cheapest and most honest; avoids collapse. |
| Strategy | **Long-only swing** | Intraday; long/short | Simpler, realistic for retail, clearer labels. |
| Output | **Structured JSON** | Free text | Machine-checkable, enforceable rules. |

## What is "industry standard" here?
- **LoRA/QLoRA** is the default way teams fine-tune open models.
- **PyTorch + Hugging Face (PEFT, TRL, Transformers, Datasets)** is the dominant stack.
- **vLLM** is a standard high-throughput inference server.
- **Walk-forward validation** is the standard for time-series/trading models.
- **MLflow / DVC / Prefect / Evidently** are mainstream MLOps tools.

---

# PART XVII — Deep Q&A (with examples)

**Q1. Why fine-tune at all instead of a clever prompt?**
A prompt is temporary and inconsistent; fine-tuning bakes behaviour into the weights so the model is reliably a trader with no prompt gymnastics.

**Q2. What is the flywheel optimising?**
Each turn: better model → better decisions → fresh graded data → better model. Over rounds we hope for higher JSON validity, better reasoning, and positive expectancy on unseen days.

**Q3. Episode vs iteration?**
An **episode** is a *simulated trading day* (data generation). An **iteration** (`MLX_ITERS`) is a *training step* (learning). Totally different phases.

**Q4. Why is the model never allowed to SELL?**
Long-only is the project's rule: every layer enforces it — the prompt forbids it, `sanitize_decision` converts SELL to HOLD, the quality filter rejects it, and legacy SELL rows are ignored.

**Q5. Why does the code recompute risk:reward instead of trusting the model?**
LLMs are bad at arithmetic. The model might say "R:R 2.5" with levels that really give 0.98. The real number is computed from entry, SL and TP.

**Q6. Why are HOLD examples kept but limited?**
Without HOLDs, a model never learns to stay out and would buy everything. With too many, it learns "HOLD" is always safe. The cap (`HOLD_RATIO × #BUY`) strikes a balance.

**Q7. Why is the split by *time* rather than random?**
In real trading you only ever predict the *future*. A random split lets the model train on Tuesday and be "tested" on Monday — a leak. Time-ordering tests it honestly.

**Q8. What does `features_used` do?**
It forces the model to name the evidence it relied on (`["ema_stack","vwap",...]`). That makes reasoning checkable, discourages invented facts, and is itself evaluated (`features_used_ok`).

**Q9. What is the `REL:` line for?**
Scale-free features (percent and ATR distances) so the model learns patterns that transfer across stocks rather than memorising particular prices.

**Q10. Why 250 bars for the indicators but only 30 for VWAP?**
EMA-200 needs ≥200 bars to be real. A volume-weighted average across ~a year would be meaningless for swing decisions, so VWAP gets a short window.

**Q11. Why store `technicals_json` and `_fund` in every row?**
So the *exact* prompt can be rebuilt later for training. If you only stored the decision, you could not recreate what the model saw.

**Q12. Why did loss hit 0.01 and is lower always better?**
No. 0.01 means near-perfect memorisation, which happens when targets are trivially predictable (rule-based mostly-HOLD data). Healthy: val loss in the 0.3–0.8 area, close to train loss.

**Q13. Why can't I fine-tune the Ollama model directly?**
Ollama serves **GGUF** (inference-only). Training needs **safetensors**. Train the safetensors, then convert to GGUF to serve.

**Q14. What's the difference between adapter and fused model?**
Adapter = a small patch that needs the base model. Fused = patch merged in, standalone. The flywheel uses fused so the next round can load it and keep training.

**Q15. How is "death" used?**
If capital falls below 20% of the start, the agent stops. It's shown the death line in every prompt, so it learns capital preservation.

**Q16. Why must the old database be deleted?**
It contains short trades and no `decision_json`. It migrates fine but yields **zero** long-only training examples.

**Q17. Can I use this with real money?**
**No.** It's a learning/portfolio project. Live trading needs costs/slippage modelling, paper-trading validation, risk limits, monitoring and compliance. Outputs are research, not financial advice.

**Q18. If I only read three files to understand the system, which?**
`compact_prompt.py` (what the AI sees), `simulation.py` — especially `sanitize_decision` and `TradingAgent.run_episode` (how a day is played), and `data_pipeline.py` — `build_dataset` (how trades become lessons).

---

## One-paragraph summary (tell it to a friend)

We take a generic AI and let a stronger "teacher" AI play a stock-trading simulator one historical day at a time, **buying or
staying in cash — never shorting**. A strict rulebook in code throws out any illegal or low-quality trade, and the **real
market prices that followed** grade every decision. Those graded decisions become lessons; using **QLoRA on Apple's MLX** we
cheaply teach a small 4-billion-parameter model to copy the good ones, training only ~0.05% of its weights so it fits on a
laptop. We watch **loss** fall to know it's learning and a separate **validation** number to be sure it isn't merely
memorising, then test it on **later dates it has never seen**. Repeat the loop (the **flywheel**) and the model improves each turn.
The result is a specialist that reliably emits clean, rule-following trade decisions — a complete demonstration of modern LLM
fine-tuning, data engineering, and honest evaluation.
