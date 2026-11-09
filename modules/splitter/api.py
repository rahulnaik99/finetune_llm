"""
modules/splitter/api.py
───────────────────────
FastAPI router for dataset build + split.

    POST /splitter/run       → build + split (idempotent unless force=True)
    GET  /splitter/state     → progress
    GET  /splitter/rounds    → all rounds + per-split counts
    GET  /splitter/dataset   → paginated view of a split's examples
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from common import jobs
from common.schemas import JobAccepted, SplitRequest, StateResponse
from common.state import StateStore

from .service import STAGE, list_rounds, run_split, view_dataset

router = APIRouter(prefix="/splitter", tags=["splitter"])


@router.post("/run", response_model=JobAccepted)
def start(req: SplitRequest) -> JobAccepted:
    def _job() -> None:
        run_split(
            round_num=req.round_num,
            min_confidence=req.min_confidence,
            min_rr=req.min_rr,
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
            split_mode=req.split_mode,
            force=req.force,
        )

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A split job is already running")
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Building dataset for round {req.round_num}")


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())


@router.get("/rounds")
def rounds() -> dict:
    data = list_rounds()
    return {"count": len(data), "rounds": data}


@router.get("/dataset")
def dataset(round_num: int = Query(1, ge=1),
            split: str = Query("train"),
            limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0)) -> dict:
    return view_dataset(round_num=round_num, split=split, limit=limit, offset=offset)
