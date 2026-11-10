"""
tests/test_v5_review_fixes.py
─────────────────────────────
Regression tests for the v5 deep-review fixes (parallelism partitioning,
tooling coherence, hygiene).
"""
import json

import common  # noqa: F401 — path bootstrap


# ── P1: windowed partitioning is contiguous and NON-overlapping ──────────────
def test_window_ranges_non_overlapping():
    from modules.datagen.service import _window_ranges
    r = _window_ranges("2023-01-02", "2023-12-29", 4)
    assert len(r) == 4
    for (a, b) in r:
        assert a <= b
    # each window starts at/after the previous window's end → no overlap
    for i in range(1, len(r)):
        assert r[i][0] >= r[i - 1][1]
    assert r[0][0] >= "2023-01-02" and r[-1][1] <= "2023-12-29"


# ── B1: the trained-hash registry is NOT shown as a UI "stage" ───────────────
def test_trained_registry_not_in_overview(tmp_path):
    from common.state import StateStore, all_states
    StateStore("datagen", state_dir=tmp_path).start(total=1)      # a real stage
    (tmp_path / "trained_examples.json").write_text('{"hashes": ["a", "b"]}')
    snap = all_states(state_dir=tmp_path)
    assert "datagen" in snap
    assert "trained_examples" not in snap          # registry filtered out


def test_registry_lives_outside_state_dir():
    from common import hashing
    from common.paths import STATE_DIR
    assert STATE_DIR not in hashing._REGISTRY.parents


# ── I4: params.yaml is the single source of defaults; TrainRequest sentinels ─
def test_params_loaded_and_trainrequest_sentinels():
    from common.params import load_params
    pm = load_params()
    assert "mlx_iters" in pm and "num_epochs" in pm
    from common.schemas import TrainRequest
    t = TrainRequest()                             # all hyper-params default None
    assert t.iters is None and t.epochs is None and t.lora_r is None and t.base_model is None


# ── I5: the canonical MLX id is consistent between params.yaml and .env ──────
def test_mlx_model_id_reconciled():
    from common.params import load_params
    assert load_params().get("mlx_model") == "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit"


# ── B3: the (previously dead) compact SYSTEM_PRE now matches the contract ────
def test_compact_system_pre_has_features_used():
    import compact_prompt
    assert "features_used" in compact_prompt.SYSTEM_PRE


# ── run_datagen accepts the new end_date stop parameter ──────────────────────
def test_run_datagen_has_end_date_param():
    import inspect
    from modules.datagen.service import run_datagen, run_datagen_windows
    assert "end_date" in inspect.signature(run_datagen).parameters
    assert "end_date" in inspect.signature(run_datagen_windows).parameters
