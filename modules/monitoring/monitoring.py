"""
monitoring.py
─────────────
Production model monitoring using Evidently AI.
Detects when the fine-tuned model starts drifting from its training distribution
— a signal that market regime has changed and retraining is needed.

Install: pip install evidently

Run:     python monitoring.py --reference data/trading_train.jsonl
                               --current   data/new_decisions.jsonl
         open reports/drift_report.html
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def load_decisions(jsonl_path: str) -> pd.DataFrame:
    """Load model decisions from JSONL into a DataFrame for Evidently."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            # Extract the assistant's JSON decision
            assistant = next(
                (m["content"] for m in ex.get("messages", []) if m["role"] == "assistant"),
                "{}"
            )
            try:
                decision = json.loads(assistant)
            except Exception:
                continue
            if decision.get("action") in ("BUY", "SELL", "HOLD"):
                rows.append({
                    "action":      decision.get("action"),
                    "confidence":  float(decision.get("confidence", 0)),
                    "risk_reward": float(decision.get("risk_reward", 0)),
                    "stop_loss":   float(decision.get("stop_loss", 0)),
                    "take_profit": float(decision.get("take_profit", 0)),
                    "reasoning_len": len(decision.get("reasoning", "")),
                    # Encode action as numeric for drift detection
                    "action_num": {"BUY": 1, "SELL": -1, "HOLD": 0}.get(decision.get("action"), 0),
                })
    return pd.DataFrame(rows)


def run_drift_report(
    reference_path: str,
    current_path: str,
    output_dir: str = "reports",
) -> dict:
    """
    Compare reference (training) distribution vs current (live) distribution.
    Returns a summary dict; also saves an HTML report.
    """
    try:
        import evidently as _ev
    except ImportError:
        raise ImportError("Evidently is not installed. `pip install \"evidently>=0.4,<0.5\"`.")
    try:
        # 0.4.x API (pinned). Newer 0.6/0.7 moved these modules, so give a
        # precise message rather than a misleading "pip install evidently".
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import ColumnDriftMetric
    except ImportError as exc:
        raise ImportError(
            f"Installed Evidently {getattr(_ev, '__version__', '?')} is incompatible "
            "with this code (expects the 0.4.x Report API). Pin "
            "`evidently>=0.4,<0.5` or migrate to the new evidently.Report/presets API."
        ) from exc

    ref = load_decisions(reference_path)
    cur = load_decisions(current_path)

    if ref.empty or cur.empty:
        print("Not enough data for drift analysis")
        return {"status": "insufficient_data"}

    print(f"Reference: {len(ref)} decisions | Current: {len(cur)} decisions")

    # Run Evidently report
    report = Report(metrics=[
        DataDriftPreset(),         # are distributions different?
        DataQualityPreset(),       # are there nulls, outliers?
        ColumnDriftMetric("action_num"),      # has the action distribution shifted?
        ColumnDriftMetric("confidence"),      # is the model less confident?
        ColumnDriftMetric("risk_reward"),     # is it taking worse R:R setups?
    ])
    report.run(reference_data=ref, current_data=cur)

    # Save HTML
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))
    print(f"Report saved: {html_path}")

    # Extract key metrics — version-safe approach
    try:
        result = report.as_dict()
        # Evidently >=0.4: metrics is a list of dicts with 'result' key
        metrics_list = result.get("metrics", [])
        dataset_drift_result = next(
            (m.get("result", {}) for m in metrics_list
             if "dataset_drift" in str(m.get("result", {}))), {})
        action_drifted = any(
            m.get("result", {}).get("drift_detected")
            for m in metrics_list
            if "column_name" in str(m.get("result", {}))
            and "action" in str(m.get("result", {}))
        )
    except Exception:
        dataset_drift_result, action_drifted = {}, False

    summary = {
        "dataset_drift":   dataset_drift_result.get("dataset_drift"),
        "drifted_columns": dataset_drift_result.get("number_of_drifted_columns", 0),
        "action_drifted":  action_drifted,
        "html_report": str(html_path),
    }

    # Alert if action distribution has shifted (market regime change)
    if summary.get("action_drifted"):
        print("⚠  ACTION DISTRIBUTION DRIFTED — consider retraining!")
        print("   Training set: BUY% =", round((ref["action"]=="BUY").mean()*100,1), "%")
        print("   Current set:  BUY% =", round((cur["action"]=="BUY").mean()*100,1), "%")
    else:
        print("✓  No significant drift detected")

    return summary


def log_drift_to_mlflow(summary: dict, run_name: str = "drift-check") -> None:
    """Log drift metrics to MLflow for tracking over time."""
    try:
        import mlflow
        mlflow.set_experiment("model-monitoring")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "dataset_drift":   int(summary.get("dataset_drift") or 0),
                "drifted_columns": summary.get("drifted_columns") or 0,
                "action_drifted":  int(summary.get("action_drifted") or 0),
            })
            if summary.get("html_report"):
                mlflow.log_artifact(summary["html_report"])
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Training data JSONL (baseline)")
    p.add_argument("--current",   required=True, help="Recent decisions JSONL (live)")
    p.add_argument("--out",       default="reports")
    args = p.parse_args()

    summary = run_drift_report(args.reference, args.current, args.out)
    log_drift_to_mlflow(summary)
    print(json.dumps(summary, indent=2))
