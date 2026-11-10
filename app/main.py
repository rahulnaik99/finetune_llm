"""
app/main.py
───────────
Single FastAPI application that mounts every stage router and serves the
lightweight tracking UI.

Run:
    uvicorn app.main:app --reload --port 8000
    # or: python run.py serve

Each stage also exposes its own router, so a stage can be served standalone if
you ever want to scale one out (``uvicorn modules.datagen.api:router`` pattern).
"""
from __future__ import annotations

# Wire sys.path for the bare-name imports used by the heavy domain modules.
import common  # noqa: F401  (import side-effect: setup_paths)

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common.paths import ensure_dirs
from common.state import all_states

from modules.datagen.api import router as datagen_router
from modules.splitter.api import router as splitter_router
from modules.training.api import router as training_router
from modules.evaluation.api import router as evaluation_router
from modules.orchestrator.api import router as orchestrator_router
from modules.monitoring.api import router as monitoring_router
from modules.governance.api import router as governance_router

UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI(
    title="Trading-Agent LLM Fine-tuning Pipeline",
    description="Modular, resumable pipeline: datagen → splitter → training → "
                "evaluation, orchestrated by the flywheel.",
    version="2.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()


# ── cross-stage endpoints ─────────────────────────────────────────────────────
@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/overview", tags=["meta"])
def overview() -> dict:
    """One call the UI polls to render the whole pipeline status board."""
    return {"stages": all_states()}


# ── stage routers (each under /api) ───────────────────────────────────────────
for r in (datagen_router, splitter_router, training_router,
          evaluation_router, orchestrator_router, monitoring_router,
          governance_router):
    app.include_router(r, prefix="/api")


# ── UI (mounted last so /api wins) ────────────────────────────────────────────
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")
