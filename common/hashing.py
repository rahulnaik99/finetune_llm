"""
common/hashing.py
─────────────────
Lightweight content hashing for training examples — the cheap way to answer
"is this row unique?" and "has this row already been used to train the model?".

We hash only the *learning signal* (the user prompt + the assistant answer), not
volatile metadata, so the same decision always maps to the same 16-hex id. A
flat JSON registry of trained hashes lives under ``output/state/`` — set
membership is O(1) and the file is tiny (16 bytes/example), so this stays light
even for large datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .paths import STATE_DIR

_REGISTRY = STATE_DIR / "trained_examples.json"


def example_hash(ex: dict) -> str:
    """Stable 16-hex id for a chat example (user + assistant content)."""
    msgs = ex.get("messages", []) if isinstance(ex, dict) else []
    parts = []
    for m in msgs:
        if m.get("role") in ("user", "assistant"):
            parts.append(f"{m['role']}:{m.get('content', '')}")
    blob = "\n".join(parts).encode("utf-8", "ignore")
    return hashlib.sha1(blob).hexdigest()[:16]


def load_trained() -> set[str]:
    if not _REGISTRY.exists():
        return set()
    try:
        return set(json.loads(_REGISTRY.read_text()).get("hashes", []))
    except Exception:
        return set()


def add_trained(hashes: Iterable[str]) -> int:
    """Register hashes as 'used for training'. Returns the new registry size."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    current = load_trained()
    current.update(h for h in hashes if h)
    tmp = _REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps({"hashes": sorted(current)}))
    tmp.replace(_REGISTRY)
    return len(current)


def dedup_exact(examples: list[dict]) -> list[dict]:
    """Drop byte-for-byte duplicate examples (keep first). Cheap set of hashes."""
    seen: set[str] = set()
    out = []
    for ex in examples:
        h = example_hash(ex)
        if h in seen:
            continue
        seen.add(h)
        out.append(ex)
    return out
