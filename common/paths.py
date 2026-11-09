"""
common/paths.py
───────────────
Central, import-safe resolution of every path the pipeline reads or writes.

Everything is anchored to the project root so stages behave identically whether
they are launched from the CLI, from a FastAPI worker thread, or from a test.
Paths can still be overridden from the environment / .env (keeping backwards
compatibility with the original flat layout).
"""
from __future__ import annotations

import os
from pathlib import Path

from ._bootstrap import ROOT


def _p(env_key: str, default: Path) -> Path:
    v = os.environ.get(env_key)
    return Path(v) if v else default


# ── primary locations ────────────────────────────────────────────────────────
PROJECT_ROOT: Path = ROOT
DATA_DIR: Path = _p("DATA_ROOT", ROOT / "data")
ROUNDS_DIR: Path = _p("DATA_DIR", DATA_DIR / "rounds")
PRICES_DIR: Path = _p("CSV_DIR", DATA_DIR / "prices")
FUNDAMENTALS_DIR: Path = _p("XBRL_DIR", DATA_DIR / "fundamentals")

OUTPUT_DIR: Path = _p("OUTPUT_ROOT", ROOT / "output")
ADAPTER_DIR: Path = _p("ADAPTER_DIR", OUTPUT_DIR / "adapters")
STATE_DIR: Path = _p("STATE_DIR", OUTPUT_DIR / "state")
METRICS_DIR: Path = _p("METRICS_DIR", OUTPUT_DIR / "metrics")
LOG_DIR: Path = _p("LOG_DIR", OUTPUT_DIR / "logs")

DB_PATH: Path = _p("DB_PATH", ROOT / "trading_agent.db")


def ensure_dirs() -> None:
    """Create the writable output directories if they do not yet exist."""
    for d in (OUTPUT_DIR, ADAPTER_DIR, STATE_DIR, METRICS_DIR, LOG_DIR, ROUNDS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def round_dir(round_num: int) -> Path:
    return ROUNDS_DIR / f"round_{round_num:02d}"
