# CODEWALK — code-by-code explanation of every module

A guided tour of the whole codebase. For each module you get:

1. **Overall view** — what the module is for, in one breath.
2. **File-by-file, function-by-function** — the file, the **line numbers**, what
   it is, and a plain-English explanation.
3. **Real-life analogies** so the *why* sticks.

Line numbers are accurate to within a few lines of the shipped source and will
drift as the file evolves — the **function/class names are the stable anchors**.
When in doubt, grep the name. Open the file beside
this doc and read top-to-bottom.

> Mental model for the whole system (keep this picture in your head):
> **datagen** (play trades) → **splitter** (make flashcards) →
> **training** (study) → **evaluation** (mock exam) → repeat (**orchestrator**).
> **common/** is the shared toolbox; **app/** is the dashboard.

---

# 1. `common/` — the shared toolbox

**Overall view:** utilities every stage reuses — config, file paths, the
resume/checkpoint store, the background-job runner, API data shapes, and content
hashing. No trading logic lives here.

## `common/_bootstrap.py` (47 lines) — make the old imports keep working
- **L21–L35 `ROOT` + `_STAGE_DIRS`** — a list of every folder that must be on
  Python's import path. *What it is:* the project map.
- **L38–43 `setup_paths()`** — adds each of those folders to `sys.path` so a file
  can still say `import simulation` even though it now lives in
  `modules/datagen/`. *Real life:* after you reorganise your kitchen into
  labelled drawers, this is the sticky note on the fridge telling everyone which
  drawer each tool moved to — so old recipes ("grab the whisk") still work.
- **L46 `setup_paths()` call** — runs on import, so just importing `common` wires
  everything up.

## `common/config.py` (68 lines) — read settings from `.env`
- **L20–41 `load_env(path='.env')`** — loads `KEY=value` lines into the
  environment. Tries `python-dotenv`; if absent, a tiny hand-written parser
  (L32–41) does it. *Real life:* the settings menu of a game — change difficulty
  in one place instead of editing the game's code.
- **L44–46 `env_str`**, **L49–53 `env_int`**, **L56–60 `env_float`**,
  **L63–67 `env_bool`** — typed getters with defaults. `env_bool` treats
  `1/true/yes/on/y` as true. *Why:* environment values are always strings;
  these convert safely and never crash on a typo (they fall back to the default).

## `common/paths.py` (48 lines) — one place that knows where everything lives
- **L19–21 `_p(env_key, default)`** — "use the env override if set, else this
  default path."
- **L24–37** — the actual locations: `DATA_DIR`, `ROUNDS_DIR`, `PRICES_DIR`,
  `ADAPTER_DIR`, `STATE_DIR`, `METRICS_DIR`, `DB_PATH`, etc.
- **L40–43 `ensure_dirs()`** — creates the output folders if missing.
- **L46–47 `round_dir(n)`** — returns `data/rounds/round_03` for `n=3`.
- *Real life:* the building directory in a lobby — every department has one
  agreed address, so nobody wanders into the wrong room.

## `common/state.py` (202 lines) — the resume/checkpoint engine (the heart)
**Overall:** every stage writes a small JSON "progress card" so work survives a
crash and the UI can watch live.
- **L47–51 `_lock_for(name)`** — one lock per stage so two threads don't write
  the same file at once.
- **L55–98 `class JobState`** — the progress card itself: `status`, `progress`
  (0–1), `current`/`total`, `message`, `error`, `checkpoint` (resume pointers),
  `history` (timeline for charts), timestamps.
  - **L74–88 `to_dict`** / **L91–98 `from_dict`** — convert to/from JSON.
- **L101–187 `class StateStore`** — load/save one stage's card.
  - **L104–109 `__init__`** — picks the file, e.g. `output/state/datagen.json`.
  - **L112–119 `load()`** — reads it; a corrupt/half-written file returns a
    fresh blank card instead of crashing (L117–119). *Why:* the UI polls this
    file constantly; it must never explode on a torn read.
  - **L122–132 `save()`** — the critical trick: write to a temp file, then
    `os.replace()` onto the real name (**L129**). `os.replace` is atomic, so a
    reader always sees either the old or the new file, never a half-file.
    *Real life:* saving a document to a new copy and only then renaming it over
    the original — if the power dies mid-save, your old copy is still intact.
  - **L135–144 `start`**, **L146–153 `update`**, **L155–161 `log`**,
    **L163–167 `checkpoint`**, **L169–179 `finish`**, **L181–187 `fail`** —
    the verbs a stage calls: begin, nudge the %, append a timeline point, store
    resume pointers, mark done, mark failed.
- **L190–201 `all_states()`** — reads every stage's card at once (powers the
  dashboard overview).
- *Real life:* a kitchen order ticket on a rail — anyone can glance at it to see
  "soup: done, steak: 60%", and if the chef faints the next chef resumes from
  the ticket instead of re-cooking everything.

## `common/jobs.py` (48 lines) — run work in the background, one per stage
- **L14–19 module globals** — a registry of running threads + a guard lock.
- **L22–24 `is_running(stage)`** — is this stage busy?
- **L27–47 `launch(stage, target)`** — start `target()` on a daemon thread;
  refuse if that stage is already running (returns `False` → the API answers
  HTTP 409). **L36–41** wraps the work so any crash is written to the state card
  as `failed` instead of vanishing. *Real life:* a single cook per station — you
  can't start a second omelette on a pan that's already in use, and if the cook
  drops it the ticket is marked "burnt" rather than left blank forever.

## `common/schemas.py` (103 lines) — the API's request/response shapes
- **L15–21 `DatagenRequest`**, **L24–33 `DatagenParallelRequest`**,
  **L36–43 `DatagenWindowsRequest`**, **L46–53 `SplitRequest`**,
  **L56–64 `TrainRequest`**, **L67–70 `EvalRequest`**, **L73–79 `FlywheelRequest`**
  — typed inputs with validation (e.g. `episodes` must be ≥1). **L83–87
  `JobAccepted`** / **L90–102 `StateResponse`** — typed outputs.
- *Real life:* a fill-in form with boxes that reject nonsense (you can't type
  "-5 episodes"); the clerk (API) only accepts a correctly filled form.

## `common/hashing.py` (67 lines) — "unique rows" + "already trained?" cheaply
- **L25–33 `example_hash(ex)`** — a 16-character fingerprint of the user+assistant
  text of one training example (SHA1, first 16 hex). Identical content → identical
  id. *Real life:* a fingerprint — same person always leaves the same print.
- **L36–42 `load_trained()`** / **L45–53 `add_trained(hashes)`** — a tiny JSON set
  of fingerprints that have already fed the model. ~16 bytes per row → light.
- **L56–66 `dedup_exact(examples)`** — drop byte-identical duplicates (keep first)
  using a set of fingerprints. *Real life:* a guest list where you cross off a
  name the second time it appears, so nobody gets two seats.

## `common/validation.py` (77 lines) — data-quality gate for input CSVs
- **L22–25 `_load`** — read a CSV, lower-case the headers.
- **L28–44 `_manual_checks(df)`** — dependency-free checks: OHLC columns exist,
  values numeric & positive, and `high >= low`.
- **L47–68 `validate_price_csv(path)`** — use **pandera** if installed (declarative
  schema), else the manual checks; returns `{ok, rows, errors, engine}` instead
  of raising. **L71–76 `validate_inputs(prices, vix)`** — validate both files.
- *Real life:* a bouncer at the door checking IDs before letting data into the
  club — bad rows are turned away with a reason, the party never crashes. Run it
  with `python run.py validate`.

---

# 2. `modules/datagen/` — Data Generation (the "training data" factory)

**Overall view:** a simulated trader relives market days one by one, decides
BUY/SELL/HOLD **with a written reason**, lives through the result (fees and taxes
included), sizes the position by market volatility (VIX), and records every trade
to SQLite. This is the raw material everything else learns from.

> Real-life analogy for the whole module: a trainee trader with a flight
> simulator. Every "day" they look at the charts, place a trade, see what
> happened, and write a note about why. The logbook of all those trades is the
> dataset.

## `modules/datagen/simulation.py` (999 lines) — the simulator
This is the biggest file. Read it in the order below.

**Imports & optional dependencies**
- **L44–54** — imports `numpy`/`pandas`; wraps `requests` in try/except so it's
  optional.
- **L57–70 `_require_requests()`** — only fetch `requests` when the Ollama
  teacher is actually used; otherwise raise a clear message. *Why:* you shouldn't
  need a network library to run offline.
- **L72–96** — same pattern for `yfinance` (**L83–96 `_require_yf()`**). *Real
  life:* you don't pack snow chains for a summer city drive — only grab them
  when you actually head to the mountains.

**Configuration constants (read from `.env`)**
- **L113–121** — core economics: `INITIAL_CAP`, `RISK_PER_TRADE`,
  `MAX_HOLD_DAYS`, `LOOKBACK_DAYS`, `DEATH_THRESHOLD`, Ollama URL/model, DB path.
- **L123–129** — price data source (`csv` offline vs `yfinance` online).
- **L131–143** — fundamentals source + India VIX thresholds (`VIX_LOW=13`, etc.).
- **L139–143** — auto-use `data/prices/INDIAVIX.csv` if present.
- **L146–156 cost constants** — brokerage, STT, exchange, SEBI, GST, stamp duty.
- **L159–172 VIX-sizing constants** — quantity factors + target-distance
  multipliers per regime.

**The LLM backends**
- **L176–186 `set_llm_backend(mode, adapter_path)`** — choose who decides:
  `"fallback"` (rules), `"ollama"` (teacher), `"adapter"` (the fine-tuned
  student). *Real life:* swapping who sits in the driver's seat — autopilot,
  instructor, or the student.
- **L189–235 `_llm_generate(system, user)`** — send the prompt to whichever
  backend is active and return raw text.
- **L238–253 `_parse_json(raw)`** — pull a clean JSON decision out of the model's
  text (models love to add chatter around the JSON). *Real life:* fishing the
  one signed form out of a rambling email.

**Database**
- **L259–283 `init_db()`** — create the `trade_log` + `fundamental_cache` tables
  if absent. **L265–271** turn on WAL mode + a 30s busy-timeout so several
  processes (parallel windows) can write without "database is locked". *Real
  life:* a ledger that lets several clerks jot entries without fighting over the
  pen.

**Loading candles (price bars)**
- **L297–302 `_flatten_cols`** — tidy column names.
- **L305–321 `_download_with_retry`** — yfinance download with exponential
  backoff (handles rate limits).
- **L324–351 `_load_csv_raw`** — read a CSV into OHLCV. This is the function that
  understands **your INDIA VIX file**: it auto-detects the `Date/Time` column,
  parses `DD-MM-YYYY HH:MM:SS` (dayfirst), ignores the ticker text columns, and
  fills missing volume with 0 (**L340** `format="mixed"`, dayfirst).
- **L354–358 `_resample_ohlcv`** — turn minute bars into daily bars.
- **L364–379** — cache the full history and derive daily (`_get_raw_history`,
  `_load_csv_history`). *Real life:* photocopying the whole year's charts once,
  then flipping to any page instantly instead of re-fetching.
- **L382–426 `get_intraday`, `_get_full_history`, `get_candles`,
  `get_forward_candles`** — slice history *up to* a date (no peeking at the
  future) and the few days *after* (to see how the trade played out). **This
  no-look-ahead cut is what keeps the simulation honest.**

**Zones, structure, fundamentals, VIX**
- **L429–446 `compute_zones`** — supply/demand zones for the symbol/date.
- **L449–470** — quarter key + "how fresh are the fundamentals?".
- **L473–533 `get_fundamentals`** — load quarterly PAT/GNPA/NII (XBRL or
  yfinance), cached in the DB.
- **L539–577** — swing pivots, HH/HL/LH/LL trend structure, zone clustering.
- **L583–608** — load the VIX series (`_get_vix_history`, `_load_vix_csv`).
- **L611–635 `vix_context(as_of)`** — VIX level, 1-year percentile, regime
  (LOW/NORMAL/ELEVATED/EXTREME), 5-day direction, 20-day range position — all
  computed with no look-ahead (**L620** cuts the series at `as_of`). *Real life:*
  the day's "weather report" for volatility: calm, breezy, stormy.

**Technical indicators + prompt**
- **L638–681 `compute_technicals`** — EMA50/100/200 stack, VWAP, support/
  resistance, trend. *Real life:* the dashboard gauges a driver glances at.
- **L684–724 `_level_breaks`** — previous-day-high/low break state.
- **L744–756 `_build_user_prompt`** — assemble everything into the question the
  decider answers.

**Decisions**
- **L759–786 `pre_trade_decision(...)`** — ask the backend for a JSON decision;
  if the model fails, a **rule-based fallback** (L768–785) makes a sane choice so
  generation never stalls. *Real life:* if the GPS loses signal, you still follow
  the road signs.
- **L789–806 `post_trade_analysis(...)`** — after the trade, produce the
  "lesson" (what went right/wrong). This becomes the second training example.

**Exit, P&L, costs, sizing**
- **L812–825 `check_sl_tp(...)`** — walk the forward candles day by day; exit at
  stop-loss, take-profit, or time limit. *Real life:* the referee that ends the
  round when a bell (SL/TP) rings.
- **L828–831 `compute_pnl(action, entry, exit, qty)`** — raw gross profit =
  points × quantity.
- **L834–853 `trade_costs(entry, exit, qty, action)`** — **fees & taxes** for a
  round trip: brokerage (capped), STT, exchange charge, SEBI fee, GST, stamp
  duty. Returns 0 for HOLD or when `INCLUDE_COSTS=0`. *Real life:* the gap
  between the sticker price and what actually leaves your wallet after tax and
  card fees.
- **L856–864 `_vix_regime(vc)`** — bucket the VIX reading.
- **L867–880 `vix_sizing(vc)`** — **the position-sizing rule you asked for**:
  returns a `qty_factor` and a `dist_mult`. Low VIX → `qty_factor>1`,
  `dist_mult<1` (**more quantity, nearer target, fewer points**); high VIX →
  `qty_factor<1`, `dist_mult>1` (**less quantity, farther target, more points**).
  *Real life:* on a calm road you drive faster with a short following distance;
  in a storm you slow down and leave a big gap.
- **L883–887 `next_trading_day`** — skip weekends.

**The agent**
- **L893–998 `class TradingAgent`** — one simulated trader.
  - **L894–900 `__init__`** — symbol, start date, starting capital, empty history.
  - **L902–903 `alive()`** — has capital stayed above the death threshold?
  - **L905–910 `mdd()`** — running max drawdown.
  - **L912–998 `run_episode()`** — **one trading day, the core loop**:
    1. **L913–914** advance the episode counter + print status.
    2. **L917–926** gather candles, technicals, VIX context, zones, fundamentals.
    3. **L930–936** get the decision (action, qty, SL, TP, reason).
    4. **L938–952** apply **VIX sizing**: scale quantity by `qty_factor`; widen/
       tighten SL & TP by `dist_mult` (for BUY/SELL).
    5. **L955–959** simulate the exit, compute **gross P&L**, subtract
       **costs** → **net P&L**, update capital.
    6. **L961–974** build the trade record (reason, lesson, prices…).
    7. **L975–977** attach `gross_pnl`, `costs`, `vix_sizing` to the snapshot for
       transparency.
    8. **L983–997** INSERT the trade into SQLite and advance to the next day.
  - *Real life:* one full "day in the simulator": look, decide, size to the
    weather, trade, pay your fees, write the logbook entry.

## `modules/datagen/zones.py` (109 lines) — supply/demand zones
- **L21–71 `detect_zones(df)`** — find price shelves where it stalled then ran
  (supply above, demand below). *Real life:* floors and ceilings a ball bounces
  between in a room.
- **L74–87 `nearest_zones`**, **L90–93 `_overlap`**, **L96–101 `confluence`**,
  **L104–108 `fmt_zone`** — nearest floor/ceiling, overlap test, "do several
  signals agree?", and pretty-printing.

## `modules/datagen/compact_prompt.py` (297 lines) — token-efficient prompt
- **L22–44** — helpers: percent change, percentile position, candle body ratio.
- **L47–234 `build_compact_user(...)`** — pack all the context (price, EMAs,
  VWAP, zones, VIX, fundamentals, recent candles, a HEADLINE, a REL line) into a
  compact prompt. *Why:* fewer tokens = cheaper/faster and fits the context
  window. *Real life:* a telegram — every word costs money, so you abbreviate
  without losing meaning.
- **L236–247 `build_compact_post_user`** — the compact "what did we learn?"
  prompt. **L251–253 `count_tokens`** — rough token meter.

## `modules/datagen/fundamentals_xbrl.py` (363 lines) — read official filings
- **L43–126 `parse_xbrl_file(path)`** — pull PAT, GNPA, NII, etc. out of a SEBI
  XBRL XML filing. *Real life:* reading the numbers off a company's official tax
  return instead of a rumour.
- **L156–188 `load_and_dedup`**, **L191–302 `build_fundamentals_json`**,
  **L330–346 `get_fundamentals_xbrl`**, **L349–350 `compact_fundamentals`** —
  combine quarters, dedupe, and summarise into model-friendly text.

## `modules/datagen/service.py` (427 lines) — resumable driver + parallelism
- **L37–48 `episodes_in_db(symbol)`** — how many episodes already exist.
- **L51–66 `recent_trades(limit)`** — latest rows **including the reasoning** (the
  column you asked to see). *Real life:* the last page of the logbook.
- **L70–164 `run_datagen(...)`** — generate N episodes for one symbol,
  **resumably**: it reads the last checkpoint (**L100–110**), and if episodes are
  already done it **continues from there instead of redoing them** (**L112–118**
  short-circuit when the target is already met). After *each* episode it saves
  `episodes_done/date/capital` (**L150–156**) so a crash loses at most one
  episode. *Real life:* a bookmark in a long novel — reopen and continue, don't
  restart chapter 1.
- **L184–188 `_shard_db_path`** — per-worker DB file path.
- **L191–200 `_datagen_worker`** / **L253–309 `run_datagen_parallel`** — run
  **one process per symbol** in parallel, each to its own DB shard, then merge.
- **L219–250 `_merge_shard_into_main`** — fold a shard into the main DB
  (per-symbol replace; idempotent). **L246** commits before `DETACH` (a bug we
  fixed — otherwise SQLite reports "database is locked").
- **L323–330 `_stagger_start`** — compute a start date per window.
- **L333–344 `_datagen_window_worker`** + **L347–371 `_append_shard_into_main`**
  + **L374–426 `run_datagen_windows`** — **the parallel-episodes-for-one-symbol
  feature**: split the calendar into W windows, run one agent per window in
  parallel, union the shards. *Real life:* four interns each read a different
  three-month slice of the year's charts at the same time, then you staple their
  logbooks together.

## `modules/datagen/api.py` (102 lines) — the HTTP buttons
- **L28–48 `start`** (`POST /datagen/run`), **L52–69 `start_parallel`**
  (`/run_parallel`), **L73–85 `start_windows`** (`/run_windows`),
  **L89–90 `state`**, **L94–95 `summary`**, **L99–101 `trades`**. Each launches a
  background job or returns progress/data. *Real life:* the buttons on the
  factory control panel.

---

# 3. `modules/splitter/` — Dataset build + split (the "flashcard maker")

**Overall view:** turn raw trades into clean, **unique**, balanced training
examples and a **walk-forward** train/val/test split. *Real life:* turning a
messy logbook into tidy flashcards, then setting aside the newest ones as a
surprise exam.

## `modules/splitter/data_pipeline.py` (416 lines)
- **L54–65 `load_trades(db)`** — read only *usable* trades (non-HOLD, with
  reasoning + lesson).
- **L68–95** — format the stored technicals into a readable block.
- **L97–149 `build_pre_trade_example`** — make the first flashcard: *context →
  decision* (BUY/SELL, SL, TP, R:R, reasoning). 
- **L152–190 `build_post_trade_example`** — the second flashcard: *outcome →
  lesson*.
- **L193–224 `quality_filter`** — throw away junk (malformed JSON, HOLDs,
  low-confidence, poor risk:reward). *Real life:* binning blurry photos before
  the album.
- **L227–255** — `_assistant_json`, `_dedup_key` (coarse "these two setups are
  basically the same" signature).
- **L258–297 `balance_and_dedup`** — remove near-duplicates and balance BUY vs
  SELL so the model can't just learn "always sell." *Real life:* a balanced diet
  — not 90% dessert.
- **L300–396 `build_dataset(...)`** — the whole pipeline end to end:
  - **L316–318** build pre/post examples for each trade.
  - **L336–340** **exact content-hash de-dup** (your "unique rows" request)
    via `common.hashing.dedup_exact`.
  - **L343** coarse balance/de-dup (`balance_and_dedup`).
  - **L350–372** the **walk-forward split**: sort by date, oldest → train,
    newest → test (so the "exam" is genuinely the future).
  - **L374–388** write `trading_{train,val,test}.jsonl` **plus a `_meta`
    sidecar** (the fix that lets evaluation compute trading metrics — the
    training files must stay metadata-free).

## `modules/splitter/service.py` (140 lines)
- **L31–42 `_db_fingerprint`** — a cheap hash of (DB size + row count + params).
  *Why:* lets a rebuild be **skipped** when nothing changed.
- **L45–58 `list_rounds`** — all round folders + per-split counts.
- **L61–90 `view_dataset(round, split)`** — paged rows for the viewer, each
  annotated with its **hash** and a **`_trained`** flag (so the UI can badge
  "trained" vs "new"). 
- **L93–139 `run_split(...)`** — build + split; **L106–113** skip if the
  fingerprint is unchanged (idempotent/resumable). *Real life:* "the ingredients
  haven't changed, so don't re-bake the cake."

## `modules/splitter/api.py` (60 lines)
- **L25–40 `start`** (`POST /splitter/run`), **L44–45 `state`**,
  **L49–51 `rounds`**, **L55–59 `dataset`** (the viewer feed).

---

# 4. `modules/training/` — Fine-tuning (the "study session")

**Overall view:** fine-tune the base model on the accumulated flashcards using
LoRA/QLoRA/MLX, resumably, and record which rows were used.

## `modules/training/service.py` (184 lines)
- **L36–42 `find_adapter(round)`** — does a trained adapter already exist?
- **L45–54 `_merge_train_sets(round)`** — concatenate rounds 1..N's train files
  (continual learning — study old + new flashcards together).
- **L57–154 `run_training(...)`** — the driver:
  - **L70–76** if the adapter exists and `resume=True`, **skip** (don't waste GPU
    hours).
  - **L96–118** launch `train.py` as a subprocess (it auto-detects MLX vs QLoRA).
  - **L104–123** stream its output and parse `Iter/train loss/val loss`, pushing
    points to the state `history` so the UI draws the **loss curve**.
  - **L126–131** fail loudly if no adapter was produced.
  - **L135–150** on success, **register the row hashes** as "trained"
    (`add_trained`) so coverage tracking works.
  - *Real life:* a tutor who (a) won't re-teach a lesson you've mastered, (b)
    plots your test scores each week, and (c) ticks off which flashcards you've
    already studied.
- **L157–171 `_round_train_examples`** — load a round's train rows.
- **L174–183 `training_coverage(round)`** — count **new vs already-trained** rows
  by hash. *Real life:* "of tonight's 100 flashcards, 70 are new, 30 you've seen."

## `modules/training/api.py` (55 lines)
- **L22–38 `start`** (`POST /training/run`), **L42–43 `state`** (progress + loss),
  **L47–48 `adapter`** (resolved path), **L52–54 `coverage`** (new vs trained).

> `train.py`, `train_mlx.py`, `train_qlora.py` are the actual training backends
> (hardware auto-detect). They're long and library-specific; the service drives
> them, so you rarely touch them directly.

---

# 5. `modules/evaluation/` — Evaluation (the "mock exam")

**Overall view:** score each round on the future-dated test set — real trading
metrics (always) and model-quality metrics (optional).

## `modules/evaluation/evaluate.py` (262 lines)
- **L39–55 `_parse_json_block`** / **L58–62 `_messages`** — pull JSON + roles out
  of an example.
- **L66–87 `evaluate_model_quality(test, generate_fn)`** — run the model on each
  test prompt and score: valid JSON? all keys? non-empty reasoning? action
  matches the teacher? *Real life:* grading both the *answer* and whether the
  student *showed their work*.
- **L91–136 `trading_metrics(pnls)`** — the full backtest scorecard: win rate,
  avg win/loss, expectancy, profit factor, Sharpe/trade, **Sortino/trade**
  (downside-only risk, L107–110), max drawdown, and a **t-stat + `significant_95`**
  (L114, L133) that asks "is this edge real or luck?" (|t|≥2 ≈ 95% confident).
  *Real life:* Sharpe judges a bumpy road by all bumps; Sortino only by the
  potholes (downside). The t-stat is "did we beat the market enough times that
  it's probably skill, not a lucky streak?"
- **L139–165 `multi_fold_metrics(pnls, k=3)`** — split the trade stream into K
  time-ordered folds and report each fold + whether the edge holds in **every**
  period (`consistent`). *Real life:* a student who aces every monthly test, not
  just one — proof it's not a fluke. This is the standard overfitting guard.
- **L168–200 `pnls_from_test_meta(test)`** — read realised P&L from the `_meta`
  **sidecar** (falls back gracefully). *This is the bug-fix path* — without the
  sidecar these metrics were silently empty.
- **L209–217 `mlx_generate_factory`** — build a generator from an MLX model.
- **L220–257 `main`** — CLI self-check.

> Why no ROUGE/BLEU here: those score *word overlap* for summaries/translation.
> This model emits a JSON decision — correctness and P&L matter, not wording — so
> ROUGE would be misleading. A free-text `reasoning` score (BERTScore / LLM-as-
> judge) can be bolted on later as an optional extra.

## `modules/evaluation/service.py` (82 lines)
- **L28–35 `load_cached(round)`** — reuse a previous result.
- **L38–85 `run_eval(round, ...)`** — compute trading metrics **plus the
  multi-fold `walk_forward` robustness block**, optional model-quality, cache to
  `output/metrics/eval_round_NN.json`, and log a per-round summary (now including
  Sortino + t-stat) to the state `history` for the chart.

## `modules/evaluation/api.py` (42 lines)
- **L22–30 `start`** (`POST /evaluation/run`), **L34–35 `state`**,
  **L39–41 `metrics`** (cached numbers for the cards).

---

# 6. `modules/orchestrator/` — the Flywheel (the "conductor")

**Overall view:** run datagen → splitter → training → evaluation each round,
remembering only the last finished round so a restart resumes mid-loop.

## `modules/orchestrator/service.py` (97 lines)
- **L28–96 `run_flywheel(...)`** — the loop:
  - **L45–48** read `last_round_done` from the checkpoint and resume there.
  - **L51–96** for each remaining round: generate (using the current best
    adapter), split, train, evaluate; after each round it saves
    `last_round_done`/`current_adapter`. *Real life:* a relay race where the
    baton (the latest model) is handed to the next leg, and a scoreboard
    remembers which legs are done if the race is paused.

## `modules/orchestrator/flows.py` (89 lines) — standard workflow engine (Prefect)
- **L30–74** (when Prefect is installed) declare each stage as a `@task` with
  **retries + caching**, and the loop as a `@flow`. *Real life:* a factory line
  with an automatic "retry the jammed station, skip the ones already finished"
  controller and a run-history screen.
- **L76–88 `run_prefect_flywheel(...)`** — run via Prefect if available, else
  **fall back** to the native in-process orchestrator. *Why:* Prefect is the
  industry-standard runtime engine, but it stays optional so CI/offline never
  breaks. FastAPI is only the API/UI — this is the actual orchestrator.

## `modules/orchestrator/api.py` (38 lines)
- **L21–32 `start`** (`POST /orchestrator/run`), **L36–37 `state`**.

---

# 7. `modules/governance/` + frameworks — governance, DVC, Prefect

**Overall view:** the artifacts and standard tooling that make the project
"industry grade": auto-generated Model Card & Datasheet, a reproducible DVC
pipeline, and the Prefect orchestration engine (covered in §6).

## `modules/governance/service.py` (177 lines)
- **L30–39 `_latest_eval`** — read the newest `eval_round_*.json`.
- **L42–58 `_db_stats`** — total trades, symbols, date range, action mix from the
  trade-log DB.
- **L61–125 `generate_model_card(round)`** — write `docs/generated/MODEL_CARD.md`
  from the live eval metrics + config: what the model is, intended use, headline
  metrics, **walk-forward robustness**, and limitations/risks. *Real life:* the
  nutrition label + warnings on a product — you know what's inside before you use
  it. (Mitchell et al. "Model Cards".)
- **L128–171 `generate_datasheet()`** — write `docs/generated/DATASHEET.md`: what
  the data is, where it came from, how it was cleaned, known biases. *Real life:*
  the "ingredients & sourcing" sheet for the dataset. (Gebru et al. "Datasheets".)
- **L174–176 `generate_all`** — do both. CLI: `python run.py card`.

## `modules/governance/api.py` (33 lines)
- **L19–20 `generate`** (`POST /governance/generate`), **L24–26 `model_card`**,
  **L30–32 `datasheet`** — regenerate or fetch the markdown.

## `dvc.yaml` — reproducible, versioned pipeline (not Python)
- Declares five stages (datagen → split → train → evaluate → governance) with
  explicit `deps`/`outs`, reading hyper-parameters from `params.yaml`. `dvc repro`
  re-runs **only** the stages whose inputs changed; `dvc metrics show` compares
  eval metrics across commits. *Real life:* a recipe that remembers which steps
  you already did — change the icing and it only re-ices, it doesn't re-bake.

# 8. `app/` — the dashboard + launcher

## `app/main.py` (76 lines) — the one FastAPI app
- **L23–43** create the app and import every stage's router.
- **L46–47 `_startup`** — make output dirs on boot.
- **L52–53 `health`** (`/api/health`), **L57–59 `overview`** (`/api/overview` —
  one call returning every stage's state for the status board).
- **L62–64** mount all stage routers under `/api`.
- **L68–75** serve the UI at `/ui` and redirect `/` → `/ui`. *Real life:* the
  building's front desk that routes you to the right department.

## `run.py` (108 lines) — the command-line remote control
- **L29–103 `main()`** — subcommands: `serve`, `datagen` (with `--symbols` for
  multi-symbol parallel and `--windows` for single-symbol parallel), `split`,
  `train`, `eval`, `flywheel` (`--engine native|prefect`), `status`, **`card`**
  (governance artifacts) and **`validate`** (data-quality gate). *Real life:* one remote with a button
  for every action, so you don't memorise long commands.

## `app/ui/` — the browser dashboard (no build step, no libraries)
- `index.html` — tabs: Overview, Data Generation (progress + **reasoning
  column**), Dataset Viewer (**trained/new badges**), Training (loss curve +
  **coverage**), Evaluation (metric cards + chart).
- `app.js` — polls the `/api/...` endpoints every few seconds and draws the small
  canvas charts.
- `styles.css` — SAP "Clean Minimal" light theme.
- *Real life:* the mission-control wall of screens — glance and see everything.

---

# How a single request flows (tying it together)

You click **Generate** in the UI →
1. `app.js` POSTs to `/api/datagen/run` → `modules/datagen/api.py:start` (L28).
2. It calls `common.jobs.launch` (L27) → a background thread runs
   `modules/datagen/service.py:run_datagen` (L70).
3. That loops `TradingAgent.run_episode` (simulation.py L912), VIX-sizing and
   fee-adjusting each trade, writing to SQLite and checkpointing after each one.
4. The UI polls `/api/datagen/state` (api.py L89) → reads
   `output/state/datagen.json` via `StateStore.load` (state.py L112) and moves
   the progress bar.
5. Later, `splitter` makes flashcards, `training` studies them, `evaluation`
   grades the model — each the same pattern: **api → jobs → service → state**.

Read any one vertical slice (`api.py` → `service.py` → the heavy module) and
you've understood the whole system; the other stages are the same shape.


---

# Appendix — Audit deltas (Days 5–7)

New/changed functions from the three review passes (names are the stable
anchors; line numbers approximate):

**Data correctness (splitter — `modules/splitter/data_pipeline.py`)**
- `build_pre_trade_example` (L98) — SELL risk:reward now computed; history is
  **same-symbol + strictly-before the decision date** (no look-ahead / no
  cross-symbol leak); emits `features_used`; real confidence.
- `_dedup_key` (L268) — content-hash key for post-trade rows (no more collapse).
- `build_dataset` (L343) — exact content-hash de-dup; tiny-dataset slice guard;
  `split_mode` passed as an arg (no `os.environ` mutation).

**Datagen (`modules/datagen/service.py` + `simulation.py`)**
- `_safe_num` (sim L847) — safe numeric coercion (`"N/A"` can't crash an episode).
- `trade_costs` (sim L862) / `vix_sizing` (sim L895) — fees/taxes + VIX sizing.
- DB-level dedup guard in `run_episode` ((symbol, decision_date)).
- `history_from_db` (L51) — re-hydrates agent history on resume.
- `_window_ranges` (L379) + `run_datagen(end_date=…)` (L86) +
  `run_datagen_windows` (L434) — **non-overlapping** windowed parallelism.
- `_additive_insert` (L269) / `_merge_shard_into_main` (L284) — non-destructive,
  signature-dedup, param-bound ATTACH shard merge.

**Evaluation (`modules/evaluation`)**
- `trading_metrics` — Sortino, t-stat/significance; `multi_fold_metrics` (walk-
  forward robustness); `profit_factor` JSON-safe (`None`, not `inf`); reasoning
  null-safe; model-quality splits pre/post-trade rows.
- `_test_fingerprint` (service L30) — eval cache keyed to the test set; `force` honoured.

**Shared / tooling**
- `common/params.py:load_params` (L17) — `params.yaml` is the single source of
  truth (training service resolves hyper-params from it; sentinel `None` args).
- `common/hashing.py` — trained-hash registry stored under `output/` (not
  `STATE_DIR`), so it is not shown as a bogus UI "stage".
- `common/validation.py` — `volume >= 0` added to the data-quality gate.
- `modules/training/service.py:_mlflow` (L30) + `run_training` (L66) — logs the
  loss curve; `modules/evaluation/service.py:run_eval` (L56) attaches eval
  metrics to the SAME MLflow run (no-op without MLflow).
- `modules/orchestrator/flows.py` — Prefect flow honours round-level resume.

**Reference-only (NOT wired into the live pipeline):** `candle_store.py`,
`scripts/fetch_candles.py` (Polars/DuckDB Parquet store) — see their banners.
