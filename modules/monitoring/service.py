"""
modules/monitoring/service.py
─────────────────────────────
Thin, import-safe wrapper over ``monitoring.py`` (Evidently drift report). The
heavy Evidently import stays inside the worker; if it is not installed the
service degrades gracefully instead of crashing the API.
"""
from __future__ import annotations

from typing import Optional

from common.paths import LOG_DIR, ensure_dirs
from common.state import StateStore

STAGE = "monitoring"


def run_monitoring(reference_path: str, current_path: str,
                   output_dir: Optional[str] = None) -> dict:
    ensure_dirs()
    store = StateStore(STAGE)
    store.start(total=1, message="Computing drift report")
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
