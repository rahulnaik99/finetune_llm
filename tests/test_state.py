"""
tests/test_state.py
───────────────────
Unit tests for the resumable state/checkpoint store — the mechanism that lets a
broken pipeline resume without regenerating data.
"""
import common  # noqa: F401 — path bootstrap

from common.state import StateStore, JobState, RUNNING, DONE, FAILED, all_states


def test_roundtrip(tmp_path):
    s = StateStore("unit_datagen", state_dir=tmp_path)
    s.start(total=50, message="go")
    st = s.load()
    assert st.status == RUNNING and st.total == 50

    s.update(current=10)
    assert abs(s.load().progress - 0.2) < 1e-9

    s.checkpoint(symbol="CANBK", episodes_done=10, date="2024-02-01")
    assert s.load().checkpoint["episodes_done"] == 10


def test_resume_semantics(tmp_path):
    """A second StateStore sees the persisted checkpoint (cross-instance resume)."""
    s1 = StateStore("unit_job", state_dir=tmp_path)
    s1.start(total=100)
    s1.update(current=37)
    s1.checkpoint(symbol="CANBK", episodes_done=37, capital=10250.0)

    s2 = StateStore("unit_job", state_dir=tmp_path)   # fresh instance == restart
    ck = s2.load().checkpoint
    assert ck["episodes_done"] == 37
    assert ck["capital"] == 10250.0           # resume continues from here, not 0


def test_history_and_finish(tmp_path):
    s = StateStore("unit_train", state_dir=tmp_path)
    s.start(total=3)
    s.log({"event": "loss", "iter": 1, "train_loss": 0.9})
    s.log({"event": "loss", "iter": 2, "train_loss": 0.6})
    assert len(s.load().history) == 2
    s.finish(message="done", adapter="x/y")
    st = s.load()
    assert st.status == DONE and st.progress == 1.0 and st.checkpoint["adapter"] == "x/y"


def test_fail_and_overview(tmp_path):
    StateStore("unit_a", state_dir=tmp_path).start(total=1)
    StateStore("unit_b", state_dir=tmp_path).fail("boom")
    snap = all_states(state_dir=tmp_path)
    assert "unit_a" in snap and "unit_b" in snap
    assert snap["unit_b"]["status"] == FAILED


def test_corrupt_file_is_safe(tmp_path):
    s = StateStore("unit_corrupt", state_dir=tmp_path)
    s.path.write_text("{not json")
    st = s.load()                 # must not raise
    assert isinstance(st, JobState) and st.stage == "unit_corrupt"
