"""
tests/test_eval_sidecar.py
──────────────────────────
Regression test for the bug fixed during the restructure: the split step strips
``_meta`` from the training JSONL, which previously left ``evaluate`` with no P&L
to compute walk-forward trading metrics. The splitter now writes a ``*_meta.jsonl``
sidecar and the evaluator reads it. This test locks that contract.
"""
import json

import common  # noqa: F401 — path bootstrap


def test_pnls_read_from_sidecar(tmp_path):
    import evaluate as ev

    test_jsonl = tmp_path / "trading_test.jsonl"
    meta_jsonl = tmp_path / "trading_test_meta.jsonl"

    # Training file (no _meta — exactly what SFT consumes).
    test_jsonl.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "{}"}]}) + "\n" +
        json.dumps({"messages": [{"role": "assistant", "content": "{}"}]}) + "\n"
    )
    # Sidecar carries the P&L used for trading metrics.
    meta_jsonl.write_text(
        json.dumps({"type": "pre_trade", "pnl": 120.0}) + "\n" +
        json.dumps({"type": "post_trade", "pnl": -40.0}) + "\n"   # ignored (not pre_trade)
    )

    pnls = ev.pnls_from_test_meta(str(test_jsonl))
    assert pnls == [120.0]               # only the pre_trade row counts
    m = ev.trading_metrics(pnls)
    assert m["n"] == 1 and m["total_pnl"] == 120.0


def test_inline_meta_still_supported(tmp_path):
    import evaluate as ev
    f = tmp_path / "trading_test.jsonl"
    f.write_text(json.dumps({"messages": [], "_meta": {"type": "pre_trade", "pnl": 55.0}}) + "\n")
    assert ev.pnls_from_test_meta(str(f)) == [55.0]
