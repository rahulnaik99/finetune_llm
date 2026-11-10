"""
common/state.py
───────────────
Durable, resumable job-state store — the backbone of "a break anywhere can be
resumed without regenerating the same data again".

Design
------
* Every long-running stage (datagen / splitter / training / evaluation /
  orchestrator) owns exactly one JSON state file under ``output/state/``.
* State is written **atomically** (temp file + ``os.replace``) so a crash mid
  write never corrupts the file.
* A stage records granular *checkpoints* (e.g. "episodes 0..37 of symbol CANBK
  already generated", "round_03 adapter already trained").  On restart the
  service inspects the checkpoint and only does the *remaining* work.
* The store is process-safe for the single-worker model used here: a thin
  ``threading.Lock`` guards in-process writers, and each write is atomic on
  disk for cross-process readers (the UI polls the same files).

The heavy data itself never lives here — raw episodes stay in SQLite, datasets
stay as JSONL, adapters stay on disk.  This file only tracks *progress* and
*pointers*, which is what makes resuming cheap and safe.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import STATE_DIR

# Lifecycle states shared by every stage.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_LOCKS: dict[str, 'threading.RLock'] = {}   # RLock so mutators can hold across load+save
_LOCKS_GUARD = threading.Lock()


def _lock_for(name: str) -> 'threading.RLock':
    with _LOCKS_GUARD:
        if name not in _LOCKS:
            _LOCKS[name] = threading.RLock()
        return _LOCKS[name]


@dataclass
class JobState:
    """Serializable progress record for a single stage."""

    stage: str
    status: str = PENDING
    # how much of the target work is done (for progress bars in the UI)
    progress: float = 0.0          # 0.0 – 1.0
    current: int = 0               # units completed
    total: int = 0                 # units requested
    message: str = ""
    error: str = ""
    # free-form, stage-specific checkpoint payload (symbols done, round done…)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    # rolling metrics / timeline the dashboard visualises
    history: list[dict[str, Any]] = field(default_factory=list)
    started_at: float | None = None
    updated_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "progress": round(self.progress, 4),
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "checkpoint": self.checkpoint,
            "history": self.history,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobState":
        known = {f: d.get(f) for f in (
            "stage", "status", "progress", "current", "total",
            "message", "error", "checkpoint", "history",
            "started_at", "updated_at", "finished_at")}
        known["checkpoint"] = known.get("checkpoint") or {}
        known["history"] = known.get("history") or []
        return cls(**known)


class StateStore:
    """Load/save a single stage's :class:`JobState` atomically."""

    def __init__(self, stage: str, state_dir: Path | None = None):
        self.stage = stage
        self.dir = Path(state_dir) if state_dir else STATE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        # Sanitise the filename — ':' (used in per-symbol stages like
        # "datagen:CANBK") is illegal on Windows. The logical stage name is kept.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stage)
        self.path = self.dir / f"{safe}.json"
        self._lock = _lock_for(safe)

    # ── read ─────────────────────────────────────────────────────────────────
    def load(self) -> JobState:
        if not self.path.exists():
            return JobState(stage=self.stage)
        try:
            return JobState.from_dict(json.loads(self.path.read_text()))
        except Exception:
            # Corrupt/partial file → start clean rather than crash the UI.
            return JobState(stage=self.stage)

    # ── write ────────────────────────────────────────────────────────────────
    def save(self, state: JobState) -> None:
        state.updated_at = time.time()
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(state.to_dict(), fh, indent=2, default=str)
                os.replace(tmp, self.path)   # atomic on POSIX + Windows
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # ── convenience mutators ─────────────────────────────────────────────────
    def start(self, total: int = 0, message: str = "", reset: bool = False) -> JobState:
        with self._lock:
            st = JobState(stage=self.stage) if reset else self.load()
            st.status = RUNNING
            st.total = total or st.total
            st.message = message
            st.error = ""
            st.started_at = st.started_at or time.time()
            st.finished_at = None
            self.save(st)
            return st

    def update(self, **fields: Any) -> JobState:
        with self._lock:                       # atomic load-modify-save
            st = self.load()
            for k, v in fields.items():
                setattr(st, k, v)
            if st.total:
                st.progress = min(1.0, st.current / st.total)
            self.save(st)
            return st

    def log(self, entry: dict[str, Any]) -> JobState:
        """Append a timeline entry (used for training-loss / eval-metric charts)."""
        with self._lock:
            st = self.load()
            entry = {"t": time.time(), **entry}
            st.history.append(entry)
            self.save(st)
            return st

    def checkpoint(self, **kv: Any) -> JobState:
        with self._lock:
            st = self.load()
            st.checkpoint.update(kv)
            self.save(st)
            return st

    def finish(self, message: str = "", **checkpoint: Any) -> JobState:
        with self._lock:
            st = self.load()
            st.status = DONE
            st.progress = 1.0
            if st.total:
                st.current = st.total
            st.message = message or st.message
            st.checkpoint.update(checkpoint)
            st.finished_at = time.time()
            self.save(st)
            return st

    def fail(self, error: str) -> JobState:
        with self._lock:
            st = self.load()
            st.status = FAILED
            st.error = str(error)
            st.finished_at = time.time()
            self.save(st)
            return st


def all_states(state_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Snapshot of every stage's state — powers the UI overview."""
    d = Path(state_dir) if state_dir else STATE_DIR
    out: dict[str, dict[str, Any]] = {}
    if not d.exists():
        return out
    _DENY = {"trained_examples"}   # not a stage — a hash registry
    for f in sorted(d.glob("*.json")):
        if f.stem in _DENY:
            continue
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            continue
    return out
