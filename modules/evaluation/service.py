"""
modules/evaluation/service.py
─────────────────────────────
Evaluate a round on its walk-forward TEST split.

Two layers (mirrors evaluate.py):
  * trading metrics  — pure math from recorded P&L (always available via the
    _meta sidecar the splitter now writes).
  * model-quality    — JSON validity / schema / action-match (needs an MLX
    model; skipped gracefully when unavailable).

Results are cached to ``output/metrics/eval_round_NN.json`` and the summary is
pushed into the state history so the evaluation dashboard can chart metrics
across rounds. Re-running with the same inputs reuses the cache (resumable).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from common.paths import METRICS_DIR, ensure_dirs, round_dir
from common.state import StateStore

STAGE = "evaluation"


def load_cached(round_num: int) -> Optional[dict]:
    f = METRICS_DIR / f"eval_round_{round_num:02d}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def run_eval(
    round_num: int = 1,
    adapter_path: Optional[str] = None,
    model_quality: bool = False,
    force: bool = False,
) -> dict:
    ensure_dirs()
    store = StateStore(STAGE)
    store.start(total=1, message=f"Evaluating round {round_num}")

    test_path = round_dir(round_num) / "trading_test.jsonl"
    if not test_path.exists():
        store.fail(f"No test split for round {round_num} ({test_path})")
        return {"round": round_num, "error": "no test split"}

    from evaluate import (trading_metrics, pnls_from_test_meta,
                          evaluate_model_quality, multi_fold_metrics)

    out: dict = {"round": round_num}
    pnls = pnls_from_test_meta(str(test_path))
    out["trading"] = trading_metrics(pnls)
    out["walk_forward"] = multi_fold_metrics(pnls, k=3)   # robustness across periods

    if model_quality and adapter_path:
        cand = [Path(adapter_path), Path(adapter_path).parent / "fused_model"]
        fused = next((str(c) for c in cand if c.exists()), None)
        if fused:
            try:
                from evaluate import mlx_generate_factory
                gen = mlx_generate_factory(fused)
                out["quality"] = evaluate_model_quality(str(test_path), gen)
            except Exception as exc:
                out["quality_error"] = str(exc)

    # Cache + record.
    (METRICS_DIR / f"eval_round_{round_num:02d}.json").write_text(json.dumps(out, indent=2))
    tm = out.get("trading", {})
    store.log({"event": "eval", "round": round_num,
               "win_rate": tm.get("win_rate"), "expectancy": tm.get("expectancy"),
               "profit_factor": tm.get("profit_factor"),
               "sharpe": tm.get("sharpe_per_trade"),
               "sortino": tm.get("sortino_per_trade"),
               "t_stat": tm.get("t_stat"),
               "max_drawdown": tm.get("max_drawdown")})
    store.update(current=1)
    store.finish(message=f"round_{round_num:02d}: win {tm.get('win_rate','?')}% "
                 f"expectancy {tm.get('expectancy','?')}", round=round_num)
    return out
