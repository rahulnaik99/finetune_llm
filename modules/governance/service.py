"""
modules/governance/service.py
─────────────────────────────
Auto-generates the two governance artifacts the industry expects to ship with a
model:

* **Model Card** (Mitchell et al., 2019) — what the model is, how it was trained,
  how it performs, and its limitations/intended use.
* **Dataset Datasheet** (Gebru et al., 2021) — what the training data is, where it
  came from, how it was processed, and its known biases.

Both are generated from facts the pipeline already records — the eval metrics
cache, the trade-log DB, and the config — so they never go stale by hand. Output
lands in ``docs/generated/``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from common.paths import DB_PATH, METRICS_DIR, PROJECT_ROOT

GEN_DIR = PROJECT_ROOT / "docs" / "generated"


def _latest_eval(round_num: Optional[int] = None) -> Optional[dict]:
    if not METRICS_DIR.exists():
        return None
    if round_num is not None:
        f = METRICS_DIR / f"eval_round_{round_num:02d}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    cards = sorted(METRICS_DIR.glob("eval_round_*.json"))
    if not cards:
        return None
    try:
        return json.loads(cards[-1].read_text())
    except Exception:
        return None


def _db_stats(db_path: Optional[str] = None) -> dict:
    p = str(db_path or DB_PATH)
    out = {"total_trades": 0, "symbols": [], "date_min": None, "date_max": None,
           "actions": {}}
    try:
        with sqlite3.connect(p) as con:
            out["total_trades"] = con.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
            out["symbols"] = [r[0] for r in con.execute(
                "SELECT DISTINCT symbol FROM trade_log").fetchall()]
            mn, mx = con.execute(
                "SELECT MIN(decision_date), MAX(decision_date) FROM trade_log").fetchone()
            out["date_min"], out["date_max"] = mn, mx
            out["actions"] = {a: c for a, c in con.execute(
                "SELECT action, COUNT(*) FROM trade_log GROUP BY action").fetchall()}
    except Exception:
        pass
    return out


def generate_model_card(round_num: Optional[int] = None) -> str:
    """Write docs/generated/MODEL_CARD.md from the latest eval + config."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    ev = _latest_eval(round_num) or {}
    tm = ev.get("trading", {}) or {}
    wf = ev.get("walk_forward", {}) or {}
    base_model = os.environ.get("BASE_MODEL", "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit")
    rnd = ev.get("round", round_num)

    md = f"""# Model Card — Trading-Agent (fine-tuned)

_Auto-generated on {date.today().isoformat()} from the evaluation cache + config._

## Model details
- **Base model:** `{base_model}`
- **Method:** LoRA / QLoRA (PEFT) parameter-efficient fine-tuning.
- **Task:** produce a structured NSE intraday/positional trading decision
  (JSON: action, stop_loss, take_profit, risk_reward, reasoning).
- **Evaluated round:** {rnd}

## Intended use
- **Primary:** research / education on LLM fine-tuning + a disciplined,
  rule-citing trading assistant over Indian equities.
- **Out of scope:** live capital allocation without human oversight; non-NSE
  instruments; any use that ignores the limitations below.

## Training & evaluation
- **Split:** walk-forward (time-ordered) — the test set is strictly *after* the
  training dates, so metrics reflect unseen future conditions.
- **Costs:** P&L is net of brokerage, STT, exchange, SEBI, GST and stamp duty.
- **Position sizing:** scaled by the India VIX regime.

### Headline metrics (held-out test)
| Metric | Value |
|---|---|
| Trades (n) | {tm.get('n', '—')} |
| Win rate % | {tm.get('win_rate', '—')} |
| Expectancy (₹/trade) | {tm.get('expectancy', '—')} |
| Profit factor | {tm.get('profit_factor', '—')} |
| Sharpe / trade | {tm.get('sharpe_per_trade', '—')} |
| Sortino / trade | {tm.get('sortino_per_trade', '—')} |
| t-stat (edge significance) | {tm.get('t_stat', '—')} (95%: {tm.get('significant_95', '—')}) |
| Max drawdown (₹) | {tm.get('max_drawdown', '—')} |

### Robustness (walk-forward folds)
- Folds: {wf.get('folds', '—')} · profitable folds: {wf.get('profitable_folds', '—')}
  · consistent across all folds: **{wf.get('consistent', '—')}**
- Expectancy across folds: mean {wf.get('expectancy_mean', '—')} ± {wf.get('expectancy_std', '—')}

## Limitations & risks
- Trained on a **single symbol / limited history** → not representative of all
  regimes; past performance does not imply future results.
- Simulated fills (SL/TP on daily bars) ignore slippage, gaps and liquidity.
- A high t-stat on a short series can still overfit one period — see the
  walk-forward folds, and widen the data before trusting the edge.
- The model can hallucinate; its JSON is schema-checked but not a guarantee of
  sound reasoning.

## Ethical / responsible use
Financial decisions carry real risk. This artifact is a research tool; any live
use requires independent validation, risk controls, and human sign-off.
"""
    out = GEN_DIR / "MODEL_CARD.md"
    out.write_text(md)
    return str(out)


def generate_datasheet(db_path: Optional[str] = None) -> str:
    """Write docs/generated/DATASHEET.md describing the training dataset."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    st = _db_stats(db_path)
    split_mode = os.environ.get("SPLIT_MODE", "time")

    md = f"""# Datasheet — Trading-Agent training data

_Auto-generated on {date.today().isoformat()} from the trade-log DB._

## Motivation
Supervised fine-tuning data teaching an LLM to make disciplined, rule-citing
trading decisions and to review their outcomes.

## Composition
- **Total trades logged:** {st['total_trades']}
- **Symbols:** {', '.join(st['symbols']) or '—'}
- **Date range:** {st['date_min']} → {st['date_max']}
- **Action distribution:** {st['actions']}
- Each usable trade yields **two** examples: a *pre-trade decision* and a
  *post-trade lesson* (HuggingFace chat format).

## Collection process
- **Prices:** local CSV (offline) or yfinance. **Fundamentals:** official SEBI
  XBRL filings or yfinance. **Volatility:** India VIX CSV.
- Generated by a simulated agent (rule-based, Ollama-teacher, or the fine-tuned
  student) with strict **no-look-ahead** date cutoffs.

## Preprocessing / cleaning
- Quality filter (drop malformed JSON, HOLDs, low-confidence, poor risk:reward).
- **Exact content-hash de-duplication** (unique rows) + BUY/SELL balancing.
- **Split:** `{split_mode}` (walk-forward = earliest dates train, latest test).

## Known biases & limitations
- Single-symbol / limited-history skew; survivorship and regime bias.
- Labels derive from a *simulation*, not live executed trades.
- Fundamentals cadence is quarterly; intraday context may lead/lag reality.

## Maintenance
Regenerated each round from the DB; versioned via DVC (`dvc.yaml`).
"""
    out = GEN_DIR / "DATASHEET.md"
    out.write_text(md)
    return str(out)


def generate_all(round_num: Optional[int] = None) -> dict:
    return {"model_card": generate_model_card(round_num),
            "datasheet": generate_datasheet()}
