"""
evaluate.py
───────────
Evaluation harness for the fine-tuned trading model. Two layers:

  A) MODEL-QUALITY (no market replay) — on the held-out TEST jsonl:
       - json_validity   : fraction that parse as JSON
       - schema_ok       : fraction with all required keys
       - reasoning_ok    : fraction with non-empty reasoning
       - features_used_ok: fraction that include a non-empty features_used list
       - action_match    : agreement with the reference (teacher) action

  B) TRADING METRICS (pure math) — from realized P&L of a set of trades:
       - win_rate, avg_win, avg_loss, expectancy, profit_factor,
         sharpe (per-trade), max_drawdown_pct

The TEST set is produced by the walk-forward split (SPLIT_MODE=time), so these
numbers reflect performance on dates the model never trained on.

Usage:
  # quick self-check of the harness (uses the stored answers as the "model")
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl

  # evaluate a fine-tuned MLX adapter/fused model
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl \
      --model output/adapters/round_01/fused_model

  # trading metrics from the test set's recorded outcomes
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl --pnl-from-meta
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

REQUIRED_KEYS = ["action", "stop_loss", "take_profit", "risk_reward", "reasoning", "confidence"]


# ── parsing helpers ──────────────────────────────────────────────────────────
def _parse_json_block(text: str) -> dict:
    if not text:
        return {}
    import re
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def _messages(ex):
    sys_c = next((m["content"] for m in ex["messages"] if m["role"] == "system"), "")
    usr_c = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    asst  = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
    return sys_c, usr_c, asst


# ── A) model-quality ─────────────────────────────────────────────────────────
def evaluate_model_quality(test_path: str, generate_fn) -> dict:
    """
    generate_fn(system, user) -> model's raw text answer.
    For a self-check, pass a function that returns the stored assistant text.
    """
    rows = [json.loads(l) for l in open(test_path) if l.strip()]
    n = len(rows)
    jv = sch = rz = fu = am = 0
    for ex in rows:
        sys_c, usr_c, ref_txt = _messages(ex)
        pred = generate_fn(sys_c, usr_c)
        d = _parse_json_block(pred)
        ref = _parse_json_block(ref_txt)
        if d: jv += 1
        if d and all(k in d for k in REQUIRED_KEYS): sch += 1
        if d.get("reasoning", "").strip() not in ("", "—"): rz += 1
        if isinstance(d.get("features_used"), list) and d["features_used"]: fu += 1
        if d.get("action") and d.get("action") == ref.get("action"): am += 1
    pct = lambda x: round(100 * x / n, 1) if n else 0.0
    return {"n": n, "json_validity": pct(jv), "schema_ok": pct(sch),
            "reasoning_ok": pct(rz), "features_used_ok": pct(fu),
            "action_match": pct(am)}


# ── B) trading metrics ───────────────────────────────────────────────────────
def trading_metrics(pnls: list[float]) -> dict:
    pnls = [float(p) for p in pnls if p is not None]
    n = len(pnls)
    if n == 0:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var)
    sharpe = round(mean / std, 2) if std > 0 else 0.0   # per-trade Sharpe

    # max drawdown on the cumulative P&L curve
    cum, peak, mdd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)

    return {
        "n": n,
        "win_rate": round(100 * len(wins) / n, 1),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "expectancy": round(mean, 2),                      # avg P&L per trade
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "sharpe_per_trade": sharpe,
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(mdd, 2),
    }


def pnls_from_test_meta(test_path: str) -> list[float]:
    """Realized P&L recorded in each test example's _meta (pre_trade rows only)."""
    out = []
    for l in open(test_path):
        if not l.strip():
            continue
        ex = json.loads(l)
        meta = ex.get("_meta", {})
        if meta.get("type") == "pre_trade" and meta.get("pnl") is not None:
            out.append(meta["pnl"])
    return out


# ── generate functions ───────────────────────────────────────────────────────
def stored_answer_generate(system, user):
    """Self-check: the 'model' just echoes the reference — harness sanity test."""
    return None   # replaced per-row below; see _selfcheck


def mlx_generate_factory(model_path: str):
    """Return a generate_fn backed by an MLX fused model / adapter."""
    from mlx_lm import load, generate
    model, tok = load(model_path)
    def _gen(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return generate(model, tok, prompt=prompt, max_tokens=256, verbose=False)
    return _gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="held-out test JSONL")
    p.add_argument("--model", default=None, help="MLX fused model / adapter path")
    p.add_argument("--pnl-from-meta", action="store_true",
                   help="also report trading metrics from the test set's recorded P&L")
    args = p.parse_args()

    if args.model:
        gen = mlx_generate_factory(args.model)
        print(f"Evaluating model: {args.model}")
    else:
        # self-check: echo the stored assistant so the harness reports ~100%
        rows = [json.loads(l) for l in open(args.test) if l.strip()]
        answers = {}
        for ex in rows:
            _, usr, asst = _messages(ex)
            answers[usr] = asst
        gen = lambda s, u: answers.get(u, "")
        print("No --model given → harness self-check (echoes stored answers)")

    q = evaluate_model_quality(args.test, gen)
    print("\nMODEL QUALITY (held-out test):")
    for k, v in q.items():
        print(f"  {k:18s}: {v}")

    if args.pnl_from_meta:
        tm = trading_metrics(pnls_from_test_meta(args.test))
        print("\nTRADING METRICS (recorded outcomes on test dates):")
        for k, v in tm.items():
            print(f"  {k:18s}: {v}")

    out = {"model_quality": q}
    if args.pnl_from_meta:
        out["trading_metrics"] = trading_metrics(pnls_from_test_meta(args.test))
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/eval_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote metrics/eval_results.json")


if __name__ == "__main__":
    main()
