"""
modules/datagen/api.py
──────────────────────
FastAPI router for the test-data-generation stage.

Endpoints
    POST /datagen/run      → launch (or resume) a generation job (non-blocking)
    GET  /datagen/state    → live progress + checkpoint
    GET  /datagen/summary  → episodes already in the DB for a symbol
    GET  /datagen/trades   → recent generated rows (dataset preview)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from common import jobs
from common.schemas import DatagenRequest, JobAccepted, StateResponse
from common.state import StateStore

from .service import STAGE, episodes_in_db, recent_trades, run_datagen

router = APIRouter(prefix="/datagen", tags=["datagen"])


@router.post("/run", response_model=JobAccepted)
def start(req: DatagenRequest) -> JobAccepted:
    if jobs.is_running(STAGE):
        raise HTTPException(409, "A data-generation job is already running")

    def _job() -> None:
        run_datagen(
            symbol=req.symbol,
            start_date=req.start_date,
            episodes=req.episodes,
            backend=req.backend,
            adapter_path=req.adapter_path,
            resume=req.resume,
        )

    launched = jobs.launch(STAGE, _job)
    if not launched:
        raise HTTPException(409, "A data-generation job is already running")
    resumed = bool(req.resume and StateStore(STAGE).load().checkpoint.get("episodes_done"))
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Generating {req.episodes} episodes of {req.symbol}",
                       resumed=resumed)


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())


@router.get("/summary")
def summary(symbol: str = Query("CANBK")) -> dict:
    return {"symbol": symbol, "episodes_in_db": episodes_in_db(symbol)}


@router.get("/trades")
def trades(limit: int = Query(50, ge=1, le=500)) -> dict:
    rows = recent_trades(limit=limit)
    return {"count": len(rows), "trades": rows}
