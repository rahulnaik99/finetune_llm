# CODEWALK — the whole codebase explained, step by step

This is a guided tour of the project written so that **anyone** — even someone
new to machine learning or trading — can follow it. For every module you get:

- **what it is** and **why it exists** (its purpose),
- a **real-world analogy** so the idea sticks,
- the **exact file and line numbers** of each function, with a plain-English
  explanation of what the code does.

Line numbers are accurate to within a line or two of the shipped source and will
drift as files change — the **function names are the stable anchors**. Open the
file next to this page and read along.

---

## 0. The big picture (read this first)

Imagine a **trainee stock trader using a flight simulator**. Every "day" they
look at the charts, decide to buy/sell/hold, write down *why*, then watch what
happens and jot a lesson. After many days you collect their decision notes and
use them to **coach** a small AI so it trades more sensibly. Then the improved
AI takes over the simulator and the loop repeats. That loop is the whole project:

```
   DATAGEN              SPLITTER            TRAINING            EVALUATION
 (play the sim,   →   (turn notes into  →  (coach the AI  →   (grade it on the
  write notes)         flashcards)          on flashcards)     unseen future)
        ▲                                                           │
        └──────────────── the better AI drives the next round ◄─────┘
                         (ORCHESTRATOR runs this loop)
```

Supporting cast: **`common/`** is the shared toolbox (settings, file paths, a
"save game" system, background jobs). **`app/`** is the website/dashboard and the
command-line remote. **`governance/`** writes the "nutrition label" for the model
and data. **`monitoring/`** watches for the market changing under the model.

Every stage follows the **same three-layer shape**, so once you understand one,
you understand them all:

1. a **heavy module** with the real logic (e.g. `simulation.py`),
2. a thin **`service.py`** that runs it and saves progress,
3. a thin **`api.py`** that exposes it as web buttons.

---

# PART 1 — `common/` (the shared toolbox)

Purpose: things every stage needs. No trading logic lives here.

## 1.1 `common/_bootstrap.py` — "make the old imports still work"

- **L21–L35** — a list of every folder in the project.
- **L38–L43 `setup_paths()`** — adds those folders to Python's search path so a
  file can say `import simulation` even though it now lives in
  `modules/datagen/`.
- **L46** — runs it automatically on import.

*Real world:* when you reorganise a kitchen into labelled drawers, this is the
sticky note on the fridge telling everyone which drawer each tool moved to — so
old recipes ("grab the whisk") still work without being rewritten.

## 1.2 `common/config.py` — read settings from a `.env` file

- **L21–L44 `load_env()`** — reads `KEY=value` lines (e.g. `SYMBOL=CANBK`) into
  the program's environment. Note **L39–41**: it only treats a `#` as a comment
  when it follows a space, so a value like `TOKEN=ab#cd` is kept intact.
- **L47–L70 `env_str / env_int / env_float / env_bool`** — fetch a setting with
  a safe default and the right type (a missing or mistyped value never crashes).

*Real world:* the settings menu of a video game — you change the difficulty in
one place instead of editing the game's code.

## 1.3 `common/paths.py` — one place that knows where everything lives

- **L19–L21 `_p()`** — "use the env override if set, otherwise this default."
- **L24–L37** — the actual locations: the database, datasets, adapters, the
  `state/` and `metrics/` output folders, etc.
- **L40–L43 `ensure_dirs()`** — creates the output folders if missing.
- **L46–L47 `round_dir(3)`** — returns `data/rounds/round_03`.

*Real world:* the building directory in a lobby — every department has one agreed
address, so nobody wanders into the wrong room.

## 1.4 `common/params.py` — the single source of training settings

- **L17–L25 `load_params()`** — reads `params.yaml` (learning rate, LoRA rank,
  iterations…) once and caches it.
- **L28–L29 `param(key, default)`** — fetch one value.

*Real world:* the master recipe card. Change "bake 30 min → 25 min" here and
every cook follows the new number, instead of each cook guessing.

## 1.5 `common/state.py` — the "save game" system (the heart of resumability)

Purpose: every stage writes a tiny progress file so work survives a crash and the
dashboard can watch it live.

- **L47–L51 `_lock_for()`** — hands out one lock per stage so two threads can't
  scribble on the same file at once.
- **L55–L98 `class JobState`** — the progress card itself: `status`, `progress`
  (0–1), `current`/`total`, `message`, `error`, a `checkpoint` (resume pointers),
  and a `history` timeline for charts. **L74/L91** convert it to/from JSON.
- **L101–L196 `class StateStore`** — load/save one stage's card:
  - **L104–L113 `__init__`** — picks the file, e.g. `output/state/datagen.json`,
    and **sanitises the name** (a `:` from `datagen:CANBK` becomes `_` so it's
    legal on Windows).
  - **L115–L123 `load()`** — reads it; a half-written/corrupt file returns a
    blank card instead of crashing (the dashboard polls this constantly).
  - **L125–L136 `save()`** — the key trick (**L132**): write to a temp file then
    `os.replace()` onto the real name. That swap is atomic, so a reader always
    sees the old or the new file, never a torn half-file.
  - **L138–L195 `start / update / log / checkpoint / finish / fail`** — the
    verbs a stage calls; each wraps the read-modify-write in the lock so
    concurrent updates can't clobber each other.
- **L199–L213 `all_states()`** — reads every stage's card for the dashboard
  overview (and **skips** the `trained_examples` registry so it isn't shown as a
  fake stage).

*Real world:* a kitchen order ticket on a rail — anyone can glance to see "soup:
done, steak: 60%", and if the chef faints the next chef resumes **from the
ticket** instead of re-cooking everything.

## 1.6 `common/jobs.py` — run work in the background, one job per stage

- **L22–L24 `is_running(stage)`** — is this stage busy?
- **L27–L47 `launch(stage, target)`** — start the work on a background thread;
  refuse if that stage is already running (the API turns that into "409 busy").
  **L36–L41** makes sure any crash is recorded as `failed` rather than vanishing.

*Real world:* one cook per station — you can't start a second omelette on a pan
already in use, and if the cook drops it the ticket is marked "burnt", not left
blank forever.

## 1.7 `common/schemas.py` — the shapes of API requests/replies

- **L15–L83** — typed request forms (`DatagenRequest`, `TrainRequest`, …) that
  validate input (e.g. "episodes must be ≥ 1"). **L56–L67 `TrainRequest`** uses
  `None` defaults so "the caller didn't set this" is distinguishable from a real
  value (that's what lets `params.yaml` win when nothing was passed).
- **L87–L106** — reply shapes (`JobAccepted`, `StateResponse`).

*Real world:* a fill-in form with boxes that reject nonsense — you can't type
"-5 episodes"; the clerk only accepts a correctly filled form.

## 1.8 `common/hashing.py` — "is this row unique / already used to train?"

- **L27–L35 `example_hash(ex)`** — a 16-character fingerprint of a training
  example's text. Same content → same fingerprint.
- **L38–L55 `load_trained` / `add_trained`** — a tiny set of fingerprints that
  have already trained the model (lives under `output/`, ~16 bytes/row).
- **L58–L68 `dedup_exact()`** — drop byte-identical duplicates.

*Real world:* a fingerprint — the same person always leaves the same print, so
you can spot duplicates and see who you've already met.

## 1.9 `common/validation.py` — a quality gate for the input CSVs

- **L28–L48 `_manual_checks()`** — columns exist, prices are numeric & positive,
  `high ≥ low`, `volume ≥ 0`.
- **L51–L72 `validate_price_csv()`** — uses the `pandera` library if installed,
  otherwise the manual checks; returns `{ok, rows, errors}` instead of crashing.

*Real world:* a bouncer checking IDs at the door — bad rows are turned away with
a reason, the party never crashes. Run it with `python run.py validate`.

---

# PART 2 — `modules/datagen/` (generate the training data)

Purpose: a simulated trader relives market days, decides with a written reason,
lives through the outcome (fees and taxes included), sizes positions by market
volatility, and records every trade to a database.

## 2.1 `simulation.py` — the simulator (the biggest file; read in this order)

**Optional dependencies (so it runs offline)**
- **L57–L70 `_require_requests()`** / **L83–L96 `_require_yf()`** — `requests`
  (for the Ollama teacher) and `yfinance` (for online prices) are fetched only
  when actually used, with a clear error if missing. *Real world:* you don't pack
  snow chains for a summer city drive — grab them only when you head to the
  mountains.

**Settings (read from `.env`)** — **L113–L172**: starting capital, risk per
trade, max days to hold, fee rates (brokerage/STT/GST…), and VIX-sizing factors.

**Who makes the decision**
- **L176–L186 `set_llm_backend()`** — choose the decider: `"fallback"` (simple
  rules), `"ollama"` (a teacher AI), or `"adapter"` (the fine-tuned student).
  *Real world:* swapping who's in the driver's seat — autopilot, instructor, or
  the learner.
- **L189–L235 `_llm_generate()`** — send the prompt to whichever decider is
  active and return its text (passing the right temperature through, **L231**).
- **L238–L253 `_parse_json()`** — pull the clean JSON decision out of the AI's
  chatter. *Real world:* fishing the one signed form out of a rambling email.

**The database**
- **L259–L288 `init_db()`** — create the `trade_log` table if absent, turn on
  **WAL mode + a busy timeout** (**L265–271**) so several processes can write
  without "database is locked", and migrate older DBs to add new columns.

**Loading price history (no cheating allowed)**
- **L329–L356 `_load_csv_raw()`** — read a price/VIX CSV; auto-detects the date
  column and the `DD-MM-YYYY HH:MM:SS` format of the India VIX file.
- **L359–L384** — turn minute bars into daily bars and cache them.
- **L420–L431 `get_candles` / `get_forward_candles`** — slice history **up to**
  a date (never peeking at the future) and the few days **after** (to see how the
  trade played out). *This no-look-ahead cut is what keeps the simulation honest.*

**Market context**
- **L434–L451 `compute_zones`** — price "floors and ceilings".
- **L478–L538 `get_fundamentals`** — quarterly company numbers (profit, bad
  loans), cached.
- **L588–L640 `vix_context`** — the day's "volatility weather report": calm,
  breezy, stormy — computed with no look-ahead.
- **L643–L686 `compute_technicals`** — the dashboard gauges (moving averages,
  VWAP, trend). **L749–L761 `_build_user_prompt`** bundles it into the question
  the decider answers.

**Making + reviewing the decision**
- **L764–L805 `pre_trade_decision()`** — ask the backend for a JSON decision; if
  it fails, **rule-based fallbacks** (bullish → BUY at L783, bearish → **SELL** at
  L797) keep generation going. *Real world:* if the GPS loses signal you still
  follow the road signs.
- **L808–L825 `post_trade_analysis()`** — after the trade, produce the "lesson"
  (what went right/wrong) — this becomes a second training example.

**Money, costs, sizing, exit**
- **L831–L844 `check_sl_tp()`** — walk the next few days and exit at stop-loss,
  take-profit, or a time limit. *Real world:* the referee ending the round when a
  bell rings.
- **L847–L853 `_safe_num()`** — turn a shaky model value (`"N/A"`, `null`) into a
  number safely so one bad field can't crash the whole day.
- **L856–L859 `compute_pnl()`** — raw profit = points × quantity.
- **L862–L881 `trade_costs()`** — subtract **real Indian fees & taxes**
  (brokerage, STT, exchange, SEBI, GST, stamp duty). *Real world:* the gap
  between the sticker price and what actually leaves your wallet after tax.
- **L884–L908 `vix_sizing()`** — size to the volatility: **low VIX → bigger
  quantity, nearer target**; **high VIX → smaller quantity, farther target +
  wider stop**. *Real world:* on a calm road you drive fast and tailgate; in a
  storm you slow down and leave a big gap.

**The agent (one simulated trader)**
- **L921–L1049 `class TradingAgent`**:
  - **L922 `__init__`**, **L930 `alive()`** (has capital survived?), **L933
    `mdd()`** (worst dip).
  - **L940 `run_episode()` — one trading day, the core loop:** gather context →
    get a decision → apply VIX sizing and re-cap to what the capital can afford →
    simulate the exit → compute **net** P&L (gross − costs) → write the trade
    (reason, lesson, fees, confidence) to the DB **only if that (symbol, date)
    isn't already logged** (dedup) → advance to the day **after the exit** (so the
    next day can't include a trade that hasn't closed).

## 2.2 `zones.py` — supply/demand zones
- **L21–L71 `detect_zones()`** finds price shelves; **L74–L108** give the nearest
  floor/ceiling, overlap tests, and pretty-printing. *Real world:* the floors and
  ceilings a bouncing ball keeps hitting in a room.

## 2.3 `compact_prompt.py` — a token-efficient prompt
- **L48–L235 `build_compact_user()`** packs all the context into a short prompt;
  **L237–L254** build the post-trade prompt and estimate token count. *Real
  world:* a telegram — every word costs money, so you abbreviate without losing
  meaning.

## 2.4 `fundamentals_xbrl.py` — read official company filings
- **L43–L126 `parse_xbrl_file()`** pulls profit/bad-loans/interest-income out of
  an official SEBI XBRL filing; **L191–L302** combine quarters into
  model-friendly text. *Real world:* reading numbers off a company's official tax
  return instead of a rumour.

## 2.5 `service.py` — the resumable driver + parallelism
- **L37–L48 `episodes_in_db`** — how many episodes already exist.
- **L51–L64 `history_from_db`** — rebuild the agent's past trades from the DB (so
  a **resumed** run keeps real context, not an empty slate).
- **L67–L82 `recent_trades`** — latest rows **including the reasoning** (for the
  UI table).
- **L86–L204 `run_datagen()`** — generate N episodes, **resumably**: it reconciles
  the saved checkpoint with the real DB, continues from where it stopped, and
  checkpoints after every episode. *Real world:* a bookmark in a long novel —
  reopen and continue, don't restart chapter 1.
- **L231–L306** — one process per symbol writing to its own DB **shard**, then a
  **non-destructive, dedup-by-signature merge** (`_additive_insert` L269,
  `_merge_shard_into_main` L284) into the main DB.
- **L379–L395 `_window_ranges()`** — split one symbol's calendar into
  **contiguous, non-overlapping** date windows.
- **L434–L499 `run_datagen_windows()`** — run those windows in parallel
  processes, each stopping at its window end (via `run_datagen(end_date=…)`), then
  merge. *Real world:* four interns each read a different, non-overlapping
  three-month slice of the year at the same time, then you staple their logbooks
  together — no wasted double-reading.

## 2.6 `api.py` — the web buttons
- **L28 `/datagen/run`**, **L52 `/run_parallel`** (many symbols),
  **L73 `/run_windows`** (one symbol, parallel windows), **L89 `/state`**,
  **L94 `/summary`**, **L99 `/trades`**.

---

# PART 3 — `modules/splitter/` (notes → flashcards)

Purpose: turn raw trades into clean, unique, balanced training examples and a
**walk-forward** train/validation/test split.

## 3.1 `data_pipeline.py`
- **L55–L66 `load_trades()`** — read usable trades, **ordered by decision date**
  (so "history" is chronological, not insertion order).
- **L98–L179 `build_pre_trade_example()`** — the first flashcard: *context →
  decision*. Critically (**L107–L115**) the "last 4 trades" shown in the prompt
  are filtered to the **same symbol and strictly earlier dates** — no future and
  no other-stock leaking in. It computes risk:reward for **both** BUY and SELL
  (**L128–L135**), uses the real confidence, and emits `features_used`.
- **L182–L220 `build_post_trade_example()`** — the second flashcard: *outcome →
  lesson*.
- **L223–L254 `quality_filter()`** — bin junk (malformed JSON, HOLDs,
  low-confidence, poor risk:reward). *Real world:* deleting blurry photos before
  the album.
- **L268–L298 `_dedup_key()`** — a near-duplicate signature; lesson rows (no
  action) get a **content-hash** key so distinct lessons aren't merged into one.
- **L301–L340 `balance_and_dedup()`** — drop near-duplicates and balance BUY vs
  SELL so the model can't just learn "always sell." *Real world:* a balanced
  diet, not 90% dessert.
- **L343–L445 `build_dataset()`** — the whole pipeline: build examples →
  exact-hash de-dup → balance → **walk-forward split** (oldest dates train,
  newest test = the real "future") → write the JSONL files **plus a `_meta`
  sidecar** so evaluation can still read each trade's P&L.

## 3.2 `service.py`
- **L31–L42 `_db_fingerprint()`** — a cheap hash of (DB + settings) so a rebuild
  is **skipped** when nothing changed.
- **L45–L58 `list_rounds`**, **L61–L90 `view_dataset`** (each row tagged with its
  hash and a **trained / new** badge), **L93–L140 `run_split`**.

## 3.3 `api.py` — `/splitter/run`, `/state`, `/rounds`, `/dataset` (the viewer).

---

# PART 4 — `modules/training/` (coach the AI)

Purpose: fine-tune the base model on the flashcards using LoRA/QLoRA/MLX, on
whatever hardware is present, resumably.

## 4.1 `service.py`
- **L30–L36 `_mlflow()`** — grab the MLflow experiment-tracker if installed,
  else do nothing.
- **L45–L51 `find_adapter()`** — does a trained model for this round already
  exist? **L54–L63 `_merge_train_sets()`** — study all rounds' flashcards together
  (so the model keeps what it learned).
- **L66–L220 `run_training()`** — the driver: **resolves hyper-parameters from
  `params.yaml`** when not given (so the recipe card wins); **skips** if the
  adapter exists; launches `train.py`; **streams the loss and logs the curve to
  MLflow**; **fails loudly on a non-zero exit code** (never accepts a
  half-written model); records which rows trained the model.
- **L240–L249 `training_coverage()`** — "of this round's 100 flashcards, 70 are
  new, 30 already studied."

*Why LoRA/QLoRA (the training method):* the base model has ~4 billion settings;
retraining them all needs a huge GPU. LoRA freezes them and trains tiny add-on
"adapter" layers (~1%); QLoRA also compresses the frozen model to 4-bit. *Real
world:* instead of rewriting a textbook, you add a thin booklet of margin notes.

## 4.2 `api.py` — `/training/run`, `/state` (loss curve), `/adapter`, `/coverage`.

---

# PART 5 — `modules/evaluation/` (grade it on the unseen future)

Purpose: score each round on the walk-forward test set — real trading metrics
(always) and model-quality metrics (optional).

## 5.1 `evaluate.py`
- **L66–L94 `evaluate_model_quality()`** — run the model on each test prompt and
  score it: valid JSON? all keys? non-empty reasoning? action matches the teacher?
  Only **decision** rows count toward schema/action (lesson rows are excluded).
  *Real world:* grading both the *answer* and whether the student *showed their
  work*.
- **L98–L145 `trading_metrics()`** — the backtest scorecard: win rate,
  expectancy, profit factor (**JSON-safe `None` when there are no losses, not
  `inf`**), Sharpe, **Sortino** (penalises only downside), and a **t-stat +
  95%-significance** flag ("is this edge real or luck?").
- **L148–L174 `multi_fold_metrics()`** — split the trades into time folds and
  check the edge holds in **every** period. *Real world:* a student who aces every
  monthly test, not just one lucky week.
- **L177–L209 `pnls_from_test_meta()`** — read each test trade's realised P&L
  from the `_meta` sidecar (the fix that keeps trading metrics alive after the
  training/meta split).
- *Note:* ROUGE/BLEU are deliberately **not** used — they score word-overlap for
  summaries, which is meaningless for a structured JSON trading decision.

## 5.2 `service.py`
- **L30–L39 `_test_fingerprint()`** — tie the cache to the exact test set.
- **L42–L53 `load_cached()`** — reuse a previous result only if the fingerprint
  matches.
- **L56–L134 `run_eval()`** — compute metrics + the walk-forward robustness block,
  cache them, and attach them to the training run in MLflow (linked experiments).

## 5.3 `api.py` — `/evaluation/run`, `/state`, `/metrics`.

---

# PART 6 — `modules/orchestrator/` (run the whole loop)

Purpose: run datagen → split → train → evaluate each round, remembering only the
last finished round so a restart resumes mid-loop.

- **`service.py` L28–L99 `run_flywheel()`** — the native loop: resumes from
  `last_round_done`; a **failed round is retried on the next resume**, not skipped
  forever; the newly trained model drives the next round's datagen. *Real world:*
  a relay race where the baton (the latest model) is handed to the next leg, and a
  scoreboard remembers which legs are done if the race is paused.
- **`flows.py` L83–L95 `run_prefect_flywheel()`** — the same loop expressed with
  **Prefect**, the industry workflow engine (retries, caching, run history);
  **falls back to the native loop** if Prefect isn't installed, and it honours
  round-level resume too.
- **`api.py`** — `/orchestrator/run`, `/state`.

---

# PART 7 — `modules/governance/` (the model's "nutrition label")

Purpose: auto-generate the documents the industry expects to ship with a model.

- **`service.py` L68–L132 `generate_model_card()`** — writes
  `docs/generated/MODEL_CARD.md` from the live eval metrics + config: what the
  model is, intended use, headline metrics, walk-forward robustness, limitations.
- **L135–L178 `generate_datasheet()`** — writes `DATASHEET.md`: what the data is,
  where it came from, how it was cleaned, known biases.
- **`api.py`** — `/governance/generate`, `/model_card`, `/datasheet`.

*Real world:* the nutrition label + allergy warnings on a food product — you know
what's inside and how it was made before you consume it.

---

# PART 8 — `modules/monitoring/` (watch for the market changing)

Purpose: detect when live decisions **drift** away from the training data — the
signal to retrain.

- **`service.py` L20–L27 `_safe_under_project()`** — security: refuse file paths
  outside the project (no reading `/etc/passwd`).
- **`monitoring.py` L49–L133 `run_drift_report()`** — builds an **Evidently**
  drift report (with a clear message if Evidently's version is incompatible).
- **`api.py`** — `/monitoring/drift`, `/state`.

*Real world:* a smoke detector — it doesn't cook, it just warns you when the
conditions change and it's time to act.

---

# PART 9 — `app/` (dashboard + command line)

## 9.1 `app/main.py` — the one web application
- **L40–L43 `lifespan()`** — modern startup hook (creates output dirs).
- **L52–L58 `health` / `overview`** — `/api/health` and `/api/overview` (one call
  the dashboard polls for the whole status board).
- **L61–L78** — mount every stage's buttons under `/api`, serve the dashboard at
  `/ui`, and redirect `/` → `/ui`.

## 9.2 `run.py` — the command-line remote
- **L35–L128 `main()`** — subcommands: `serve`, `datagen` (`--symbols` for
  multi-symbol parallel, `--windows` for single-symbol parallel), `split`,
  `train`, `eval`, `flywheel` (`--engine native|prefect`), `status`, `card`
  (governance), `validate` (data gate).

## 9.3 `app/ui/` — the browser dashboard (no build step, no libraries)
Tabs for Overview, Data Generation (progress + the **reasoning** column),
Dataset Viewer (**trained/new** badges), Training (live **loss curve**),
Evaluation (metric cards + chart). *Real world:* the wall of screens in mission
control — glance and see everything.

---

# How one click flows through the system (tying it together)

You press **Generate** in the dashboard →
1. the browser POSTs to `/api/datagen/run` → `modules/datagen/api.py:start` (L28).
2. that calls `common/jobs.py:launch` (L27) → a background thread runs
   `modules/datagen/service.py:run_datagen` (L86).
3. which loops `TradingAgent.run_episode` (`simulation.py` L940), VIX-sizing and
   fee-adjusting each trade, writing to the DB and checkpointing after each one.
4. the dashboard polls `/api/datagen/state` (api L89) → reads
   `output/state/datagen.json` via `StateStore.load` (state.py L115) → moves the
   progress bar.
5. later, **splitter** makes flashcards, **training** studies them, **evaluation**
   grades the model — each the *same shape*: **api → jobs → service → state**.

Read any one vertical slice (`api.py → service.py → the heavy module`) and you
understand the whole system; the other stages are the same pattern.

---

# Reference-only modules (not wired into the live pipeline)

`modules/datagen/candle_store.py` and `scripts/fetch_candles.py` are a
production-grade Polars/DuckDB "SQL-on-Parquet" data store, kept as a **reference
implementation** (see the banner at the top of each). The live simulator reads
candles via pandas/CSV today; adopting the store is the upgrade path for
very large (1-minute, multi-year) datasets.
