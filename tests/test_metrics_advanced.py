"""Advanced evaluation metrics: Sortino, significance t-stat, multi-fold walk-forward."""
import common  # noqa: F401 — path bootstrap
import evaluate as ev


def test_sortino_and_tstat_present():
    m = ev.trading_metrics([52, -80, 237, 313, -26, -149, 52, -83])
    for k in ("sortino_per_trade", "t_stat", "significant_95"):
        assert k in m
    assert isinstance(m["significant_95"], bool)


def test_sortino_only_penalises_downside():
    # all-positive stream → no downside → Sortino falls back to 0 (no downside dev)
    up = ev.trading_metrics([10, 20, 30, 40])
    assert up["sortino_per_trade"] == 0.0            # zero downside deviation
    mixed = ev.trading_metrics([10, -5, 20, -8, 30])
    assert mixed["sortino_per_trade"] != 0.0


def test_multi_fold_structure_and_consistency():
    # strongly positive, stable stream → profitable in every fold
    good = ev.multi_fold_metrics([30, 25, 28, 32, 27, 31, 29, 26, 33], k=3)
    assert good["folds"] == 3 and len(good["per_fold"]) == 3
    assert good["profitable_folds"] == 3 and good["consistent"] is True
    assert "expectancy_mean" in good and "expectancy_std" in good


def test_multi_fold_detects_inconsistency():
    # great early, terrible late → not consistent across folds
    mixed = ev.multi_fold_metrics([50, 60, 55, -40, -55, -48], k=2)
    assert mixed["consistent"] is False


def test_multi_fold_handles_too_few():
    assert ev.multi_fold_metrics([1.0], k=3)["per_fold"] == []
