"""
modules/monitoring/service.py
─────────────────────────────
Thin, import-safe wrapper over ``monitoring.py`` (Evidently drift report). The
heavy Evidently import stays inside the worker; if it is not installed the
service degrades gracefully instead of crashing the API.
"""
from __future__ import annotations

from typing import Optional

from pathlib import Path

from common.paths import LOG_DIR, PROJECT_ROOT, ensure_dirs
from common.state import StateStore

STAGE = "monitoring"


def _safe_under_project(path: str) -> bool:
    """Reject paths outside the project root (prevents arbitrary local-file reads
    via the monitoring API)."""
    try:
        return PROJECT_ROOT.resolve() in Path(path).resolve().parents or \
               Path(path).resolve() == PROJECT_ROOT.resolve()
    except Exception:
        return False


def run_monitoring(reference_path: str, current_path: str,
                   output_dir: Optional[str] = None) -> dict:
    ensure_dirs()
    store = StateStore(STAGE)
    store.start(total=1, message="Computing drift report")
    # Path allow-listing: both files must live inside the project (no /etc/passwd).
    if not (_safe_under_project(reference_path) and _safe_under_project(current_path)):
        msg = "reference_path/current_path must be inside the project directory"
        store.fail(msg)
        return {"error": msg}
    try:
        from monitoring import run_drift_report
        out = run_drift_report(reference_path, current_path,
                               output_dir=output_dir or str(LOG_DIR / "drift"))
        store.update(current=1)
        store.finish(message="Drift report ready", summary=out)
        return out
    except Exception as exc:
        store.fail(str(exc))
        return {"error": str(exc)}
