"""
common/schemas.py
─────────────────
Pydantic request/response models shared by the stage APIs. Keeping them in one
place gives the UI a single, stable contract and keeps each ``api.py`` thin.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── requests ─────────────────────────────────────────────────────────────────
class DatagenRequest(BaseModel):
    symbol: str = Field("CANBK", description="NSE symbol to simulate")
    start_date: str = Field("2024-01-10", description="First simulation day (YYYY-MM-DD)")
    episodes: int = Field(50, ge=1, le=5000, description="Target number of episodes")
    backend: str = Field("fallback", description="fallback | ollama | adapter")
    adapter_path: Optional[str] = Field(None, description="Adapter path when backend='adapter'")
    resume: bool = Field(True, description="Continue from the last checkpoint instead of restarting")


class SplitRequest(BaseModel):
    round_num: int = Field(1, ge=1, description="Round bucket to write the dataset into")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_rr: float = Field(1.3, ge=0.0)
    val_ratio: float = Field(0.1, ge=0.0, le=0.9)
    test_ratio: float = Field(0.1, ge=0.0, le=0.9)
    split_mode: str = Field("time", description="time (walk-forward) | random")
    force: bool = Field(False, description="Rebuild even if the DB is unchanged")


class TrainRequest(BaseModel):
    round_num: int = Field(1, ge=1)
    base_model: str = Field("mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit")
    prev_adapter: Optional[str] = None
    iters: int = Field(600, ge=1)
    epochs: int = Field(2, ge=1)
    lora_r: int = Field(16, ge=1)
    save_every: int = Field(100, ge=1)
    resume: bool = Field(True, description="Skip if this round's adapter already exists")


class EvalRequest(BaseModel):
    round_num: int = Field(1, ge=1)
    adapter_path: Optional[str] = None
    model_quality: bool = Field(False, description="Also run model-quality eval (needs MLX model)")


class FlywheelRequest(BaseModel):
    symbol: str = "CANBK"
    start_date: str = "2024-01-10"
    rounds: int = Field(5, ge=1, le=100)
    episodes: int = Field(50, ge=1)
    base_model: str = "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit"
    resume: bool = Field(True, description="Continue from the last completed round")


# ── responses ────────────────────────────────────────────────────────────────
class JobAccepted(BaseModel):
    stage: str
    status: str
    message: str
    resumed: bool = False


class StateResponse(BaseModel):
    stage: str
    status: str
    progress: float
    current: int
    total: int
    message: str = ""
    error: str = ""
    checkpoint: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    finished_at: Optional[float] = None
