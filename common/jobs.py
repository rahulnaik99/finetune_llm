"""
common/jobs.py
──────────────
Minimal in-process background-job runner. Each stage runs at most one job at a
time; the job body updates its :class:`~common.state.StateStore` so the UI can
poll progress. We deliberately avoid Celery/RQ — a single worker thread keeps
the footprint tiny and matches the "lightweight" requirement while still being
non-blocking for the API.
"""
from __future__ import annotations

import threading
import traceback
from typing import Callable

from .state import StateStore

_THREADS: dict[str, threading.Thread] = {}
_GUARD = threading.Lock()


def is_running(stage: str) -> bool:
    t = _THREADS.get(stage)
    return bool(t and t.is_alive())


def launch(stage: str, target: Callable[[], None]) -> bool:
    """Start ``target`` on a daemon thread for ``stage``.

    Returns ``False`` if a job for that stage is already running (callers map
    this to HTTP 409). The target is responsible for marking the StateStore as
    done/failed; we add a safety-net ``fail`` on uncaught exceptions.
    """
    with _GUARD:
        if is_running(stage):
            return False

        def _wrapped() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 — never let the thread die silently
                StateStore(stage).fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

        t = threading.Thread(target=_wrapped, name=f"job-{stage}", daemon=True)
        _THREADS[stage] = t
        t.start()
        return True
