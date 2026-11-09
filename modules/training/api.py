"""
modules/training/api.py
───────────────────────
    POST /training/run     → fine-tune a round (skips if adapter exists)
    GET  /training/state   → progress + loss history (training dashboard feed)
    GET  /training/adapter → resolved adapter path for a round
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from common import jobs
from common.schemas import JobAccepted, StateResponse, TrainRequest
from common.state import StateStore

from .service import STAGE, find_adapter, run_training

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/run", response_model=JobAccepted)
def start(req: TrainRequest) -> JobAccepted:
    def _job() -> None:
        run_training(
            round_num=req.round_num,
            base_model=req.base_model,
            prev_adapter=req.prev_adapter,
            iters=req.iters,
            epochs=req.epochs,
            lora_r=req.lora_r,
            save_every=req.save_every,
            resume=req.resume,
        )

    if not jobs.launch(STAGE, _job):
        raise HTTPException(409, "A training job is already running")
    return JobAccepted(stage=STAGE, status="running",
                       message=f"Fine-tuning round {req.round_num}")


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**StateStore(STAGE).load().to_dict())


@router.get("/adapter")
def adapter(round_num: int = Query(1, ge=1)) -> dict:
    return {"round": round_num, "adapter": find_adapter(round_num)}
