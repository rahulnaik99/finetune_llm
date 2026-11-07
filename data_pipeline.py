"""
data_pipeline.py
────────────────
Extracts trading agent episodes from SQLite → quality-filtered JSONL
in HuggingFace chat format, ready for SFTTrainer.

Each episode creates TWO training examples:
  1. pre_trade  : (market context + history) → (BUY/SELL/HOLD + reasoning + SL/TP)
  2. post_trade : (trade outcome + candles)  → (lesson + strategy adjustment)

Usage:
    python data_pipeline.py \
        --db trading_agent.db \
        --out data/training.jsonl \
        --min_confidence 0.5 \
        --min_rr 1.5
"""
import argparse, json, sqlite3, random, re
from pathlib import Path
from datetime import datetime


# ── Nemotron / ChatML system prompt ──────────────────────────────────────────
PRE_TRADE_SYSTEM = """You are a disciplined Indian equity trader. You analyse NSE stocks using \
technical indicators (EMA, VWAP, support/resistance zones, HH/HL/LH/LL trend structure) \
and fundamental data (quarterly PAT, GNPA, NII). Your primary rule: protect capital first.

Output ONLY valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "quantity": <int>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "risk_reward": <float>,
  "reasoning": "<2-3 sentences citing specific prices and indicator values>",
  "confidence": <0.0-1.0>,
  "key_risk": "<one-line failure scenario>"
}"""

POST_TRADE_SYSTEM = """You are reviewing a completed stock trade to extract a concrete lesson \
for future improvement. Be specific, cite actual prices.

Output ONLY valid JSON:
{
  "outcome": "profit" | "loss" | "breakeven",
  "exit_reason_analysis": "<what caused the exit — cite candle data>",
  "what_went_right": "<if profit: correct setup element>",
  "what_went_wrong": "<if loss: missed/ignored signal>",
  "lesson_learned": "<one concrete trading rule>",
  "strategy_adjustment": "<how to change next entry>"
}"""


def load_trades(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT * FROM trade_log
            WHERE action != 'HOLD'
              AND pre_trade_reasoning IS NOT NULL
              AND lesson_learned IS NOT NULL
              AND lesson_learned != '—'
            ORDER BY created_at
        """).fetchall()
    return [dict(r) for r in rows]


def _parse_technicals(tech_json: str) -> str:
    """Convert stored technicals JSON to a readable market-data block."""
    try:
        t = json.loads(tech_json or "{}")
    except Exception:
        return "(technical data unavailable)"

    lines = [
        f"Close: {t.get('close','—')}  VWAP: {t.get('vwap','—')} "
        f"({'above' if t.get('above_vwap') else 'below'} VWAP)",
        f"EMA50: {t.get('ema_50','—')}  EMA100: {t.get('ema_100','—')}  EMA200: {t.get('ema_200','—')}",
        f"Stack: {t.get('ema_stack','—')}   Trend: {t.get('trend','—')}  "
        f"Structure: {', '.join(t.get('structure',[]) or ['—'])}",
        f"Nearest Support:    {t.get('nearest_support','—')}",
        f"Nearest Resistance: {t.get('nearest_resistance','—')}",
        f"Support Zones:    {t.get('support_zones','—')}",
        f"Resistance Zones: {t.get('resistance_zones','—')}",
    ]
    return "\n".join(lines)


def build_pre_trade_example(trade: dict, prev_trades: list[dict]) -> dict | None:
    """Build a (user_prompt → assistant_json) pair for the pre-trade decision."""
    tech_block = _parse_technicals(trade.get("technicals_json", "{}"))

    # History block (last 4 previous trades)
    hist_lines = []
    for pt in prev_trades[-4:]:
        pnl = pt.get("pnl", 0)
        hist_lines.append(
            f"  {pt['decision_date']} | {pt['action']} @ {pt['entry_price']:.1f}"
            f" -> {pt['exit_price']:.1f} [{pt['exit_reason']}]"
            f" P&L {'+' if pnl>=0 else ''}{pnl:.0f} | Lesson: {pt.get('lesson_learned','—')}"
        )
    hist_text = "\n".join(hist_lines) if hist_lines else "  (no previous trades)"

    user_prompt = (
        f"=== MARKET DATA: {trade['symbol']} as of {trade['decision_date']} ===\n"
        f"{tech_block}\n\n"
        f"=== TRADE HISTORY ===\n{hist_text}\n\n"
        f"=== CAPITAL: {trade['capital_before']:.0f} ===\n"
        f"Decide. Output JSON only."
    )

    # Reconstruct the assistant's answer from stored fields
    assistant_json = json.dumps({
        "action":      trade["action"],
        "quantity":    trade["quantity"],
        "stop_loss":   trade["stop_loss"],
        "take_profit": trade["take_profit"],
        "risk_reward": round((trade["take_profit"] - trade["entry_price"]) /
                             max(trade["entry_price"] - trade["stop_loss"], 0.01), 2)
                       if trade["action"] == "BUY" else 0.0,
        "reasoning":   trade["pre_trade_reasoning"],
        "confidence":  0.7,
        "key_risk":    f"SL at {trade['stop_loss']:.2f} if trend reverses",
    }, indent=2)

    return {
        "messages": [
            {"role": "system",    "content": PRE_TRADE_SYSTEM},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_json},
        ],
        "_meta": {
            "type": "pre_trade",
            "episode": trade["episode"],
            "symbol": trade["symbol"],
            "date": trade["decision_date"],
            "pnl": trade["pnl"],
            "exit_reason": trade["exit_reason"],
        },
    }


def build_post_trade_example(trade: dict) -> dict | None:
    """Build a (trade outcome → lesson) pair for the post-trade analysis."""
    outcome = "profit" if trade["pnl"] > 0 else "loss" if trade["pnl"] < 0 else "breakeven"

    user_prompt = (
        f"=== COMPLETED TRADE ===\n"
        f"Symbol: {trade['symbol']}  Date: {trade['decision_date']}\n"
        f"Action: {trade['action']}  Qty: {trade['quantity']}\n"
        f"Entry: {trade['entry_price']:.2f}  SL: {trade['stop_loss']:.2f}  TP: {trade['take_profit']:.2f}\n"
        f"Exit:  {trade['exit_price']:.2f} on {trade['exit_date'] or '—'} [{trade['exit_reason']}]\n"
        f"P&L:   {trade['pnl']:+.2f}  |  Capital: {trade['capital_before']:.0f} -> {trade['capital_after']:.0f}\n\n"
        f"Original reasoning: {trade['pre_trade_reasoning']}\n\n"
        f"Analyse. Output JSON only."
    )

    assistant_json = json.dumps({
        "outcome":            outcome,
        "exit_reason_analysis": trade.get("post_trade_analysis", "—"),
        "what_went_right":    "Setup conditions were met" if outcome == "profit" else "—",
        "what_went_wrong":    trade.get("what_went_wrong", "—"),
        "lesson_learned":     trade.get("lesson_learned", "—"),
        "strategy_adjustment": trade.get("strategy_adjustment", "—"),
    }, indent=2)

    return {
        "messages": [
            {"role": "system",    "content": POST_TRADE_SYSTEM},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_json},
        ],
        "_meta": {
            "type": "post_trade",
            "episode": trade["episode"],
            "symbol": trade["symbol"],
            "date": trade["decision_date"],
            "pnl": trade["pnl"],
        },
    }


def quality_filter(example: dict, min_confidence: float, min_rr: float) -> bool:
    """Drop low-quality examples before training."""
    meta = example.get("_meta", {})
    try:
        assistant_text = example["messages"][-1]["content"]
        parsed = json.loads(assistant_text)
    except Exception:
        return False   # malformed JSON → drop

    if meta.get("type") == "pre_trade":
        # Drop HOLDs (no learning signal for entry strategy)
        if parsed.get("action") == "HOLD":
            return False
        # Drop low-confidence decisions
        if parsed.get("confidence", 1.0) < min_confidence:
            return False
        # Drop poor R:R
        if parsed.get("risk_reward", 99) < min_rr:
            return False
    elif meta.get("type") == "post_trade":
        # Drop trivial lessons
        lesson = parsed.get("lesson_learned", "—")
        if lesson in ("—", "", "N/A", "Analysis unavailable (Ollama not running)"):
            return False

    return True


def _assistant_json(example: dict) -> dict:
    """Parse the assistant message JSON from a training example (or {})."""
    try:
        for m in example.get("messages", []):
            if m.get("role") == "assistant":
                return json.loads(m["content"])
    except Exception:
        pass
    return {}


def _dedup_key(example: dict) -> tuple:
    """
    Near-duplicate signature: action + coarse-binned decision fields. Two setups
    that round to the same bucket are treated as duplicates (keep one).
    """
    d = _assistant_json(example)
    def b(x, step):
        try:
            return round(float(x) / step) * step
        except Exception:
            return None
    return (
        d.get("action"),
        b(d.get("stop_loss"), 1.0),
        b(d.get("take_profit"), 1.0),
        b(d.get("risk_reward"), 0.25),
        b(d.get("confidence"), 0.1),
    )


def balance_and_dedup(examples: list[dict],
                      dedup: bool = True,
                      balance: bool = True,
                      max_ratio: float = 1.5,
                      seed: int = 42) -> list[dict]:
    """
    De-duplicate near-identical examples and balance BUY vs SELL so one side
    cannot dominate (the cause of the 'always-HOLD / loss→0' memorisation).
    Nothing is summarised away — only redundant/over-represented rows are trimmed.
    """
    import random as _r
    _r.seed(seed)

    # 1. De-dup by coarse decision signature (keep first occurrence)
    if dedup:
        seen, unique = set(), []
        for e in examples:
            k = _dedup_key(e)
            if k in seen:
                continue
            seen.add(k)
            unique.append(e)
        print(f"After de-dup: {len(unique)} examples (removed {len(examples)-len(unique)} near-dupes)")
        examples = unique

    # 2. Balance BUY vs SELL (cap the majority at max_ratio × minority)
    if balance:
        buys  = [e for e in examples if _assistant_json(e).get("action") == "BUY"]
        sells = [e for e in examples if _assistant_json(e).get("action") == "SELL"]
        other = [e for e in examples if _assistant_json(e).get("action") not in ("BUY", "SELL")]
        if buys and sells:
            minority = min(len(buys), len(sells))
            cap = int(minority * max_ratio)
            _r.shuffle(buys); _r.shuffle(sells)
            buys, sells = buys[:cap], sells[:cap]
            balanced = buys + sells + other
            _r.shuffle(balanced)
            print(f"After balance: {len(balanced)} (BUY {len(buys)} / SELL {len(sells)} / other {len(other)})")
            return balanced
    return examples


def build_dataset(
    db_path: str,
    out_path: str,
    min_confidence: float = 0.5,
    min_rr: float = 1.5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    do_balance: bool | None = None,
    do_dedup: bool | None = None,
) -> dict[str, int]:
    import os
    if do_balance is None:
        do_balance = os.environ.get("BALANCE_DATA", "1").strip().lower() in ("1", "true", "yes", "on")
    if do_dedup is None:
        do_dedup = os.environ.get("DEDUP_DATA", "1").strip().lower() in ("1", "true", "yes", "on")

    trades = load_trades(db_path)
    print(f"Loaded {len(trades)} trades from {db_path}")

    examples = []
    for i, trade in enumerate(trades):
        prev = trades[:i]  # everything before this episode
        pre  = build_pre_trade_example(trade, prev)
        post = build_post_trade_example(trade)
        if pre:  examples.append(pre)
        if post: examples.append(post)

    print(f"Built {len(examples)} raw examples")

    # Quality filter
    filtered = [e for e in examples if quality_filter(e, min_confidence, min_rr)]
    print(f"After quality filter: {len(filtered)} examples")

    # Balance + de-dup (fixes memorisation collapse; adds diversity)
    filtered = balance_and_dedup(filtered, dedup=do_dedup, balance=do_balance, seed=seed)

    if len(filtered) < 50:
        print("⚠  WARNING: < 50 examples is very few for fine-tuning.")
        print("   Run more simulation episodes across more symbols/dates.")

    # ── Split: 'time' = walk-forward (train=earliest, test=latest) or 'random' ──
    split_mode = os.environ.get("SPLIT_MODE", "time").strip().lower()
    n = len(filtered)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    if split_mode == "time":
        # Chronological: earliest dates train, latest dates are the (future) test.
        filtered.sort(key=lambda e: (e.get("_meta", {}).get("date") or "", e.get("_meta", {}).get("episode", 0)))
        splits = {
            "train": filtered[: n - n_val - n_test],
            "val":   filtered[n - n_val - n_test: n - n_test],
            "test":  filtered[n - n_test:],
        }
        print(f"Split=time (walk-forward): train<=val<=test by date")
    else:
        random.seed(seed)
        random.shuffle(filtered)
        splits = {
            "train": filtered[n_test + n_val:],
            "val":   filtered[n_test:n_test + n_val],
            "test":  filtered[:n_test],
        }
        print("Split=random")

    # Write JSONL files
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(out_path).stem

    counts = {}
    for split_name, split_data in splits.items():
        path = out_dir / f"{stem}_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                # Strip internal _meta before writing
                ex_clean = {k: v for k, v in ex.items() if not k.startswith("_")}
                f.write(json.dumps(ex_clean) + "\n")
        counts[split_name] = len(split_data)
        print(f"  {split_name:5s}: {len(split_data):4d} examples → {path}")

    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db",             default="trading_agent.db")
    p.add_argument("--out",            default="data/trading.jsonl")
    p.add_argument("--min_confidence", type=float, default=0.5)
    p.add_argument("--min_rr",         type=float, default=1.5)
    p.add_argument("--val_ratio",      type=float, default=0.1)
    p.add_argument("--test_ratio",     type=float, default=0.1)
    args = p.parse_args()

    counts = build_dataset(
        db_path=args.db, out_path=args.out,
        min_confidence=args.min_confidence, min_rr=args.min_rr,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
    )
    print(f"\nDataset ready: {sum(counts.values())} total examples")
    print("Next step: python train_qlora.py --data data/trading_train.jsonl")
