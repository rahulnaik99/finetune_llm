"""
modules/evaluation/api.py
─────────────────────────
    POST /evaluation/run     → evaluate a round's walk-forward test split
    GET  /evaluation/state   → progress + per-round metric history
    GET  /evaluation/metrics → cached metrics for a round
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from common import jobs
from common.schemas import EvalRequest, JobAccepted, StateResponse
from common.state import StateStore

from .service import STAGE, load_cached, run_eval

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=JobAccepted)
def start(req: EvalRequest) -> JobAccepted:
    def _job() -> None:
        run_eval(round_num=req.round_num, adapter_path=req.adapter_path,
                 model_quality=req.model_quality)

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "An evaluation job is already running")
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Evaluating round {req.round_num}")


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())


@router.get("/metrics")
def metrics(round_num: int = Query(1, ge=1)) -> dict:
    cached = load_cached(round_num)
    return cached or {"round": round_num, "trading": None}
