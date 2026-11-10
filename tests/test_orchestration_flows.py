"""Prefect orchestration wrapper (optional dep) + governance API surface."""
import common  # noqa: F401 — path bootstrap
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_flows_module_imports_without_prefect():
    # Prefect is optional; the module must import and expose a runnable wrapper
    # that falls back to the native orchestrator when Prefect is absent.
    from modules.orchestrator import flows
    assert hasattr(flows, "run_prefect_flywheel")
    assert isinstance(flows.HAS_PREFECT, bool)


def test_governance_endpoints_registered():
    spec = client.get("/openapi.json").json()["paths"]
    for p in ("/api/governance/generate", "/api/governance/model_card",
              "/api/governance/datasheet"):
        assert p in spec


def test_governance_model_card_endpoint_returns_markdown():
    r = client.get("/api/governance/model_card")
    assert r.status_code == 200
    assert "Model Card" in r.text
