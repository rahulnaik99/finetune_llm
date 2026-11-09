"""
modules/monitoring/api.py
─────────────────────────
    POST /monitoring/drift   → reference vs current distribution drift report
    GET  /monitoring/state   → progress / last summary
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from common import jobs
from common.schemas import JobAccepted, StateResponse
from common.state import StateStore

from .service import STAGE, run_monitoring

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class DriftRequest(BaseModel):
    reference_path: str
    current_path: str


@router.post("/drift", response_model=JobAccepted)
def drift(req: DriftRequest) -> JobAccepted:
    def _job() -> None:
        run_monitoring(req.reference_path, req.current_path)

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A monitoring job is already running")
    return JobAccepted(stage=STAGE, status="running", message="Drift report started")


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())
