"""
modules/governance/api.py
─────────────────────────
    POST /governance/generate   → (re)generate MODEL_CARD.md + DATASHEET.md
    GET  /governance/model_card  → the current model card (markdown)
    GET  /governance/datasheet   → the current datasheet (markdown)
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from .service import GEN_DIR, generate_all, generate_datasheet, generate_model_card

router = APIRouter(prefix="/governance", tags=["governance"])


@router.post("/generate")
def generate(round_num: int | None = Query(None)) -> dict:
    return generate_all(round_num)


@router.get("/model_card", response_class=PlainTextResponse)
def model_card() -> str:
    f = GEN_DIR / "MODEL_CARD.md"
    return f.read_text() if f.exists() else generate_model_card() and (GEN_DIR / "MODEL_CARD.md").read_text()


@router.get("/datasheet", response_class=PlainTextResponse)
def datasheet() -> str:
    f = GEN_DIR / "DATASHEET.md"
    return f.read_text() if f.exists() else generate_datasheet() and (GEN_DIR / "DATASHEET.md").read_text()
