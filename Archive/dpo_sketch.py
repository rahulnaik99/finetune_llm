"""
dpo_sketch.py  — Direct Preference Optimization (the C in the roadmap)
─────────────────────────────────────────────────────────────────────
Goal: move the model from "imitate the teacher" to "prefer what actually MADE
MONEY", using the real SL/TP outcomes already stored in trading_agent.db.

DPO needs preference triples:  (prompt, chosen, rejected)
We build them straight from realized outcomes:

  • Trade WON  → chosen = the decision taken; rejected = a HOLD ("missing this
                 winner was worse") OR the opposite action.
  • Trade LOST → chosen = a disciplined HOLD ("should not have taken it");
                 rejected = the losing decision that was actually taken.

So the model is taught to prefer decisions whose real forward-price outcome was
better. This is the single biggest quality lever once you have teacher data.

STATUS: pair-building below is functional (reads the DB, writes JSONL).
        The training step is a documented command (MLX supports DPO).

Usage:
    python dpo_sketch.py --db trading_agent.db --out data/dpo/pairs.jsonl
    # then (Apple Silicon):
    #   python -m mlx_lm.dpo --model <fused_or_base> --data data/dpo \
    #       --beta 0.1 --iters 300 --adapter-path output/adapters/dpo
"""
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path


PRE_SYS = "You are a disciplined NSE equity trader. Output JSON only."


def _decision_json(trade, action, reasoning):
    return json.dumps({
        "action": action,
        "quantity": trade["quantity"] if action == trade["action"] else 0,
        "stop_loss": trade["stop_loss"] if action != "HOLD" else 0,
        "take_profit": trade["take_profit"] if action != "HOLD" else 0,
        "risk_reward": 0.0,
        "reasoning": reasoning,
        "confidence": 0.7 if action != "HOLD" else 0.5,
        "key_risk": "—",
        "features_used": ["ema_stack", "vwap", "nearest_support"],
    })


def build_dpo_pairs(db_path: str, out_path: str) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM trade_log WHERE action != 'HOLD' ORDER BY id"
    ).fetchall()
    con.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for r in rows:
            t = dict(r)
            # Reconstruct the prompt the model would have seen (compact if available)
            prompt = (f"=== {t['symbol']} as of {t['decision_date']} ===\n"
                      f"{t.get('technicals_json','')}\nDecide. Output JSON only.")

            took = _decision_json(t, t["action"], t.get("pre_trade_reasoning", ""))
            hold = _decision_json(t, "HOLD", "No clear edge — preserve capital.")

            if t["pnl"] > 0:
                chosen, rejected = took, hold          # prefer the winning trade over doing nothing
            elif t["pnl"] < 0:
                chosen, rejected = hold, took          # prefer NOT taking the losing trade
            else:
                continue                                # skip break-even

            f.write(json.dumps({
                "prompt": [{"role": "system", "content": PRE_SYS},
                           {"role": "user", "content": prompt}],
                "chosen": chosen,
                "rejected": rejected,
                "_pnl": t["pnl"],
            }) + "\n")
            n += 1
    print(f"Wrote {n} preference pairs → {out_path}")
    print("Next (Apple Silicon):")
    print("  python -m mlx_lm.dpo --model output/adapters/round_XX/fused_model \\")
    print("      --data data/dpo --beta 0.1 --iters 300 --adapter-path output/adapters/dpo")
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="trading_agent.db")
    p.add_argument("--out", default="data/dpo/pairs.jsonl")
    a = p.parse_args()
    build_dpo_pairs(a.db, a.out)
