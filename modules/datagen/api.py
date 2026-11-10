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
from common.schemas import (DatagenParallelRequest, DatagenRequest,
                            DatagenWindowsRequest, JobAccepted, StateResponse)
from common.state import StateStore

from .service import (STAGE, episodes_in_db, recent_trades, run_datagen,
                      run_datagen_parallel, run_datagen_windows)

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


@router.post("/run_parallel", response_model=JobAccepted)
def start_parallel(req: DatagenParallelRequest) -> JobAccepted:
    """Generate data for several symbols concurrently (one process per symbol).

    A single agent's episodes are a sequential chain and cannot be parallelised,
    but symbols are independent — this is where the real speedup comes from.
    """
    def _job() -> None:
        run_datagen_parallel(
            symbols=req.symbols, start_date=req.start_date, episodes=req.episodes,
            backend=req.backend, adapter_path=req.adapter_path,
            resume=req.resume, max_workers=req.max_workers,
        )

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A data-generation job is already running")
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Parallel datagen: {len(req.symbols)} symbols × "
                               f"{req.episodes} episodes")


@router.post("/run_windows", response_model=JobAccepted)
def start_windows(req: DatagenWindowsRequest) -> JobAccepted:
    """Parallelise ONE symbol's episodes across W date-windows (processes)."""
    def _job() -> None:
        run_datagen_windows(
            symbol=req.symbol, start_date=req.start_date, episodes=req.episodes,
            workers=req.workers, backend=req.backend,
            adapter_path=req.adapter_path, resume=req.resume,
        )

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A data-generation job is already running")
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Parallel episodes: {req.symbol} on {req.workers} windows")


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
