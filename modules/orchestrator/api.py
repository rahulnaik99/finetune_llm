"""
modules/orchestrator/api.py
───────────────────────────
    POST /orchestrator/run   → launch (or resume) the full flywheel
    GET  /orchestrator/state → progress across rounds + per-round results
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from common import jobs
from common.schemas import FlywheelRequest, JobAccepted, StateResponse
from common.state import StateStore

from .service import STAGE, run_flywheel

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


@router.post("/run", response_model=JobAccepted)
def start(req: FlywheelRequest) -> JobAccepted:
    def _job() -> None:
        run_flywheel(symbol=req.symbol, start_date=req.start_date,
                     rounds=req.rounds, episodes=req.episodes,
                     base_model=req.base_model, resume=req.resume)

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A flywheel run is already in progress")
    resumed = bool(req.resume and StateStore(STAGE).load().checkpoint.get("last_round_done"))
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Flywheel: {req.rounds} rounds of {req.symbol}",
                       resumed=resumed)


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())
