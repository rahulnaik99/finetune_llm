"""Content-hash dedup + 'already trained' registry + training coverage."""
import common  # noqa: F401 — path bootstrap
from common import hashing


def _ex(user, asst):
    return {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


def test_example_hash_stable_and_content_sensitive():
    a = _ex("buy canbk", '{"action":"BUY"}')
    b = _ex("buy canbk", '{"action":"BUY"}')
    c = _ex("buy sbin", '{"action":"SELL"}')
    assert hashing.example_hash(a) == hashing.example_hash(b)
    assert hashing.example_hash(a) != hashing.example_hash(c)
    assert len(hashing.example_hash(a)) == 16


def test_dedup_exact():
    rows = [_ex("x", "1"), _ex("x", "1"), _ex("y", "2")]
    assert len(hashing.dedup_exact(rows)) == 2


def test_trained_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(hashing, "_REGISTRY", tmp_path / "trained_examples.json")
    monkeypatch.setattr(hashing, "STATE_DIR", tmp_path)
    assert hashing.load_trained() == set()
    hashing.add_trained(["aaa", "bbb"])
    assert hashing.load_trained() == {"aaa", "bbb"}
    hashing.add_trained(["bbb", "ccc"])
    assert hashing.load_trained() == {"aaa", "bbb", "ccc"}


def test_training_coverage_counts_new_vs_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(hashing, "_REGISTRY", tmp_path / "reg.json")
    monkeypatch.setattr(hashing, "STATE_DIR", tmp_path)
    from modules.training import service as tsvc

    rows = [_ex("a", "1"), _ex("b", "2"), _ex("c", "3")]
    monkeypatch.setattr(tsvc, "_round_train_examples", lambda rn: rows)

    hashing.add_trained([hashing.example_hash(rows[0])])
    cov = tsvc.training_coverage(7)
    assert cov["total"] == 3 and cov["already_trained"] == 1 and cov["new"] == 2
