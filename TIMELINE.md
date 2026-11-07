# Project Timeline — Trading-Agent LLM Fine-Tuning

A chronological log of the whole build. Each block is:
**what we developed → what issue we saw → what fix we added.**

---

## 1. Production-framework upgrade
**What we developed:** Added a production MLOps stack to the pipeline — MLflow (experiment tracking in `runner.py`), vLLM provider in `llm_factory.py`, Polars + DuckDB data layer (`candle_store.py`), Axolotl config, Prefect flows, DVC pipeline, Evidently monitoring.
**What we saw issue:** The project was research-grade scripts with no tracking, versioning, orchestration, or serving story — hard to reproduce or show as resume-level work.
**What fix we added:** Wired each framework in as an optional, graceful layer (never blocks a run if the tool is missing) and documented the stack in the README.

---

## 2. Unified LoRA / QLoRA training entry point
**What we developed:** One `train.py` that auto-detects hardware and picks the method — MLX LoRA (Apple Silicon), HF LoRA (≥16 GB NVIDIA), QLoRA 8-bit / 4-bit (smaller NVIDIA), CPU fallback.
**What we saw issue:** Training was scattered across `train_qlora.py` (CUDA) and `train_mlx.py` (Mac); users had to know which to run, and QLoRA wouldn't work on a Mac at all.
**What fix we added:** Hardware detection + a single command; LoRA diagram, memory-budget calculator, and an educational walkthrough notebook + NotebookLM source doc + storybook.

---

## 3. Flywheel could not import the simulation
**What we developed:** The automated `flywheel.py` loop (simulate → build dataset → fine-tune → repeat).
**What we saw issue:** `NameError: name 'TradingAgent' is not defined` — the simulation lived only inside the notebook cells, so the flywheel could not import it.
**What fix we added:** Extracted all simulation logic into an importable `simulation.py` module (data, technicals, LLM decisions, `TradingAgent`); the flywheel and notebook both import from it. Replaced fragile monkey-patching with a clean `set_llm_backend("fallback"|"ollama"|"adapter")`.

---

## 4. Training step needed torch on a Mac
**What we developed:** The flywheel's fine-tune step.
**What we saw issue:** `ModuleNotFoundError: No module named 'torch'` — the flywheel called `train_qlora.py`, which needs CUDA/torch that isn't on an M-series Mac.
**What fix we added:** Pointed the flywheel at `train.py` (auto-selects MLX on Apple Silicon); broadened adapter-path detection (`fused_model` / `adapters` / `final_adapter`); suppressed the yfinance pandas-deprecation warning flood.

---

## 5. Base model download failed (gated / wrong name)
**What we developed:** Student-model loading for MLX training.
**What we saw issue:** `401 Unauthorized` for `nvidia/Nemotron-Nano-4B-Instruct` and the guessed `mlx-community/...` name — the repo was gated or misnamed.
**What fix we added:** Discovered the real public, 4-bit, ungated repo `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit` (same Bolt-E4B family as the Ollama teacher) and set it as `BASE_MODEL`.

---

## 6. Training ran, but adapter wasn't picked up + loss collapsed
**What we developed:** First successful MLX QLoRA run (loss fell 2.9 → 0.01, adapter + fused model saved).
**What we saw issue:** (a) Flywheel printed "Training failed / Final adapter: None" because it only looked for `final_adapter`; (b) val loss of ~0.01 was *too* low — a memorisation signal because the data was rule-based fallback, not LLM.
**What fix we added:** (a) Multi-path adapter detection; (b) identified the root cause as backend=`fallback` and guided switching to `USE_OLLAMA=1` so the Nemotron teacher generates real decisions.

---

## 7. Everything hard-coded in the scripts
**What we developed:** Full configurability.
**What we saw issue:** Every change (model, episodes, rounds, iterations, backend) required editing Python.
**What fix we added:** A `.env` file + `config.py` loader (works even without python-dotenv). `flywheel.py` and `train.py` read all settings from `.env`; CLI flags still override. Smoke-test vs real-run is now just a value change.

---

## 8. yfinance rate-limiting / no offline data
**What we developed:** Market-data fetching for the simulation.
**What we saw issue:** After a few episodes yfinance returned "possibly delisted / no price data" (throttling) → 0 trades → 0 training examples. Also a `FileNotFoundError` writing `flywheel_log.json` when no training ran.
**What fix we added:** Download each symbol's full history **once** + cache + retry/backoff; then added a **local CSV source** (`DATA_SOURCE=csv`) that reads a minute/daily CSV once and slices per episode (fully offline, reproducible). Created the log directory before writing.

---

## 9. Official fundamentals instead of yfinance
**What we developed:** Fundamentals input for the model.
**What we saw issue:** yfinance fundamentals are flaky and imprecise; the user had official SEBI XBRL quarterly filings (many files: standalone/consolidated, original/revision).
**What fix we added:** `fundamentals_xbrl.py` — parses the XBRL, de-dups Original vs Revision, filters by nature, and **pre-computes QoQ / YoY / TTM / CD-ratio / NII / asset-quality deltas** into one LLM-ready JSON (`FUNDAMENTALS_SOURCE=xbrl`), gated by result date (no look-ahead).

---

## 10. Richer market context (VIX, levels, zones)
**What we developed:** More decision features fed as training data.
**What we saw issue:** The model only saw EMA/VWAP/S-R/structure; missing volatility regime and level-break / zone context that drive real entries.
**What fix we added:**
- **India VIX** context (`VIX_CSV`) → level, 1-yr percentile, regime, 5-day direction, 20-day range position (volume made optional so the index CSV loads).
- **Level-break** features — PDH/PDL (broke/rejected/held/lost), 2-day H/L, weekly H/L states.
- **Demand/Supply zones** (`zones.py`) on **15-min (short)** and **1-hour (big)** timeframes, impulse ×3 (fewer/cleaner), one retest allowed, with a 15m-inside-1h **confluence** flag. The minute CSV is read once and daily/15m/1h all derive from it.

---

## 11. Smoother training & higher ROI via better data representation
**What we developed:** Techniques to summarise the data **without removing detail or depth** — encode it so the model learns faster and generalises, rather than memorising.

**What we saw issue:**
- *Absolute prices force the model to memorise numbers that never repeat.* Relative values carry the same information but transfer across stocks and price levels.
- Repetitive / imbalanced data caused the memorisation collapse (val loss → ~0.01, mostly-HOLD).
- The teacher's reasoning could drift from the actual inputs (hallucination risk).

**What fix we added:**
- **Relative encoding** in `compact_prompt.py` (keep raw for SL/TP math, add a `REL:` line — % vs EMA50/EMA200/VWAP, ATR%, distance-to-support/resistance in ATR-multiples, 5-day range position — plus a one-line `HEADLINE:` verdict the model anchors to). Toggle: `RELATIVE_ENCODING`.
- **Dataset balancer + de-dup** in `data_pipeline.py` (`balance_and_dedup()`): removes near-duplicate setups by a coarse decision signature and caps the majority BUY/SELL side at 1.5× the minority — fixing the memorisation collapse and adding diversity. Toggles: `DEDUP_DATA`, `BALANCE_DATA`.
- **Grounded structured reasoning + `features_used`** in the teacher's output schema (`_PRE_SYS`): reasoning must cite named features, and a `features_used` list records exactly which inputs drove the decision (faithfulness; the rule-based fallback now emits the same schema).

---

## Current capabilities (end state)
- Fully **offline, reproducible** data: local price CSV (minute or daily) + India VIX CSV + official XBRL fundamentals.
- Rich, **pre-computed** context fed to the LLM (relative features, headline, EMA/VWAP, S/R, PDH/PDL/2D/weekly breaks, 15m+1h demand/supply zones + confluence, VIX regime, QoQ/YoY fundamentals).
- **Config-driven** via `.env`; MLX QLoRA training proven on Apple Silicon; teacher/student flywheel with balanced, de-duplicated, faithfully-reasoned training data.

---

## 12. Evaluation harness + walk-forward split (B), with DPO (C) & deploy-loop (D) scaffolds
**What we developed:** An honest way to know if the model actually works, and the next two levers.
**What we saw issue:** The only signal was training loss — which lied once (0.01 collapse). No held-out, time-honest trading metrics; the student never re-entered the loop.
**What fix we added:**
- **`evaluate.py`** — model-quality metrics (json_validity, schema_ok, reasoning_ok, features_used_ok, action_match) + trading metrics (win_rate, expectancy, profit_factor, per-trade Sharpe, max_drawdown) on the held-out test set. Self-check verified 100%, metrics math verified.
- **Chronological (walk-forward) split** in `data_pipeline.py` (`SPLIT_MODE=time`): train = earliest dates, test = latest (future) dates — no leakage. Examples now carry their decision date in `_meta`.
- **`dpo_sketch.py` (C):** builds preference pairs from real SL/TP outcomes (prefer the decision that made money; prefer HOLD over a losing trade) → `mlx_lm.dpo` command. Moves the model from "imitate" to "optimize for profit".
- **`deploy_to_ollama.py` (D):** fuse adapter → GGUF → Modelfile with `num_ctx 16384` → `ollama create`, closing the loop so the fine-tuned student serves with your 16k context.

---

## 13. Auto-evaluation in the flywheel + build/test suite
**What we developed:** Make every round self-report its quality, and guard the whole build.
**What we saw issue:** Eval existed (`evaluate.py`) but had to be run by hand; nothing verified that all files still build/import after edits (the sandbox had repeatedly evicted/placeholdered files).
**What fix we added:**
- **`evaluate_round()` wired into `flywheel.py`** — after each round it prints walk-forward **trading metrics** (win rate, expectancy, profit factor, Sharpe, max drawdown) from the test split, optionally runs **model-quality** metrics if a fine-tuned adapter + MLX are present, logs to **MLflow** when configured, and is wrapped so it never breaks a run. Stored under `eval` in `flywheel_log.json`.
- **`tests/test_build.py`** — 8 tests: every `.py` parses (build check), core modules import (heavy deps stubbed), plus unit checks for `trading_metrics`, `balance_and_dedup`, `zones.detect_zones`, `config` `.env` loader, compact-prompt rendering (REL + HEADLINE), and XBRL parsing. Runs under `pytest` or standalone `python tests/test_build.py`. Verified **8/8 passing**.

---

## Future / planned (not yet implemented)
- **A — real teacher data at volume (your action):** one full `USE_OLLAMA=1` run across several symbols and date ranges to reach >=500 quality examples. Everything below only pays off after this.
- **C — DPO training run:** generate pairs with `dpo_sketch.py`, train with `mlx_lm.dpo`, then confirm expectancy / profit-factor **rise** on the same walk-forward test (that is the success test).
- **D — deploy-back loop:** run `deploy_to_ollama.py` (fuse -> GGUF -> Modelfile num_ctx 16384 -> ollama create), then a flywheel round where the fine-tuned student beats the previous round's metrics.
- **Matching 2024 XBRL quarters:** the sample filing is Q2FY27; load the quarters that overlap the backtest window, or the fundamentals gate them out.
- **Risk realism:** slippage, fees, liquidity, and VIX-scaled position sizing in the P&L before anyone treats this as live-ready (it is not).
- **True demand/supply (item 4 depth):** base-detection tuning per timeframe and feeding the nearest confluent zone into the fallback SL/TP.
- **Reproducibility:** DVC-track data + adapters; MLflow run comparison dashboards across rounds.
