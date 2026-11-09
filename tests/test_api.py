"""
tests/test_api.py
─────────────────
API-contract tests using FastAPI's TestClient. These are the checks CI runs on
every commit to guarantee the service wiring (routers, schemas, UI mount) stays
intact. They exercise only read endpoints + input validation — no heavy jobs are
launched, so the suite stays fast and hermetic.
"""
import common  # noqa: F401 — path bootstrap

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_overview_shape():
    r = client.get("/api/overview")
    assert r.status_code == 200
    assert "stages" in r.json()


def test_root_redirects_to_ui():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert "/ui" in r.headers["location"]


@pytest.mark.parametrize("stage", ["datagen", "splitter", "training",
                                   "evaluation", "orchestrator", "monitoring"])
def test_each_stage_has_state(stage):
    r = client.get(f"/api/{stage}/state")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == stage
    assert "status" in body and "progress" in body


def test_datagen_summary_reads_db():
    r = client.get("/api/datagen/summary?symbol=CANBK")
    assert r.status_code == 200
    assert "episodes_in_db" in r.json()


def test_datagen_trades_preview():
    r = client.get("/api/datagen/trades?limit=5")
    assert r.status_code == 200
    assert "trades" in r.json()


def test_splitter_rounds_and_dataset_view():
    r = client.get("/api/splitter/rounds")
    assert r.status_code == 200
    rounds = r.json()["rounds"]
    # round folders ship with the project
    assert any(x["round"].startswith("round_") for x in rounds)

    r2 = client.get("/api/splitter/dataset?round_num=1&split=test&limit=5")
    assert r2.status_code == 200
    body = r2.json()
    assert body["split"] == "test" and "rows" in body


def test_training_adapter_lookup():
    r = client.get("/api/training/adapter?round_num=1")
    assert r.status_code == 200
    assert "adapter" in r.json()


def test_validation_rejects_bad_payload():
    # episodes must be >= 1 → 422
    r = client.post("/api/datagen/run", json={"symbol": "X", "episodes": 0})
    assert r.status_code == 422


def test_openapi_lists_all_stage_routes():
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in ("/api/datagen/run", "/api/splitter/run", "/api/training/run",
              "/api/evaluation/run", "/api/orchestrator/run"):
        assert p in paths
