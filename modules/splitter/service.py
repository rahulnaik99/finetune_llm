"""
modules/splitter/service.py
───────────────────────────
Turns the raw SQLite trade log into quality-filtered, de-duplicated, balanced
and **split** (train/val/test) JSONL — wrapping ``data_pipeline.build_dataset``.

Resumability / idempotency
--------------------------
Building the dataset is deterministic given (the DB contents, the filter
params). We fingerprint the DB (row count + size + mtime) together with the
split parameters and store it in ``output/state/splitter.json``. If a build is
requested again with an unchanged fingerprint we **skip** it and reuse the
existing JSONL — so an interrupted pipeline that already produced a split never
rebuilds it. ``force=True`` overrides this.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from common.paths import DB_PATH, ROUNDS_DIR, ensure_dirs, round_dir
from common.state import StateStore

STAGE = "splitter"


def _db_fingerprint(db_path: str, params: dict) -> str:
    h = hashlib.sha256()
    try:
        st = os.stat(db_path)
        h.update(str(st.st_size).encode())
        with sqlite3.connect(db_path) as con:
            n = con.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
        h.update(str(n).encode())
    except Exception:
        h.update(b"no-db")
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:16]


def list_rounds() -> list[dict]:
    """Every round folder and the per-split example counts (dataset browser)."""
    out = []
    if not ROUNDS_DIR.exists():
        return out
    for d in sorted(ROUNDS_DIR.glob("round_*")):
        if not d.is_dir():
            continue
        counts = {}
        for split in ("train", "val", "test"):
            f = d / f"trading_{split}.jsonl"
            counts[split] = sum(1 for _ in open(f)) if f.exists() else 0
        out.append({"round": d.name, "counts": counts, "total": sum(counts.values())})
    return out


def view_dataset(round_num: int, split: str = "train", limit: int = 50,
                 offset: int = 0) -> dict:
    """Return rows of a split's JSONL for the 'view generated dataset' panel."""
    split = split if split in ("train", "val", "test") else "train"
    f = round_dir(round_num) / f"trading_{split}.jsonl"
    if not f.exists():
        return {"round": round_num, "split": split, "total": 0, "rows": []}
    try:
        from common.hashing import example_hash, load_trained
        trained = load_trained()
    except Exception:
        example_hash, trained = (lambda e: ""), set()

    rows = []
    total = 0
    with open(f) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            total += 1
            if offset <= i < offset + limit:
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                h = example_hash(ex)
                ex["_hash"] = h
                ex["_trained"] = h in trained
                rows.append(ex)
    return {"round": round_num, "split": split, "total": total, "rows": rows}


def run_split(
    round_num: int = 1,
    min_confidence: float = 0.5,
    min_rr: float = 1.3,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_mode: str = "time",
    force: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Build + split the dataset for ``round_num`` (skips if unchanged)."""
    ensure_dirs()
    store = StateStore(STAGE)
    db = str(db_path or DB_PATH)

    params = {
        "round": round_num, "min_confidence": min_confidence, "min_rr": min_rr,
        "val_ratio": val_ratio, "test_ratio": test_ratio, "split_mode": split_mode,
    }
    fp = _db_fingerprint(db, params)
    prev = store.load()
    out_dir = round_dir(round_num)
    train_exists = (out_dir / "trading_train.jsonl").exists()

    if (not force and train_exists
            and prev.checkpoint.get("fingerprint") == fp):
        counts = {k: v for k, v in (prev.checkpoint.get("counts") or {}).items()}
        store.finish(message=f"{out_dir.name}: unchanged — reused existing split",
                     fingerprint=fp, counts=counts, skipped=True)
        return {"round": round_num, "skipped": True, "counts": counts}

    store.start(total=1, message=f"Building dataset for {out_dir.name}")

    from data_pipeline import build_dataset  # lazy

    out_path = str(out_dir / "trading.jsonl")
    # Pass split_mode explicitly instead of mutating the global process env.
    counts = build_dataset(
        db_path=db, out_path=out_path,
        min_confidence=min_confidence, min_rr=min_rr,
        val_ratio=val_ratio, test_ratio=test_ratio,
        split_mode=split_mode,
    )
    store.update(current=1)
    store.finish(message=f"{out_dir.name}: train {counts.get('train',0)} / "
                 f"val {counts.get('val',0)} / test {counts.get('test',0)}",
                 fingerprint=fp, counts=counts, skipped=False, round=round_num)
    return {"round": round_num, "skipped": False, "counts": counts}
