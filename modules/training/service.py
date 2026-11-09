"""
modules/training/service.py
───────────────────────────
Resumable fine-tuning driver. Shells out to ``train.py`` (which auto-detects the
hardware: MLX-LoRA on Apple Silicon, QLoRA on NVIDIA) exactly as the original
flywheel did — but adds:

* **Resume**: if ``round_NN`` already has a trained adapter on disk and
  ``resume=True`` we skip retraining (no wasted GPU hours).
* **Live progress**: the child's stdout is streamed; lines that look like
  ``Iter 120: ... val loss 0.43`` are parsed and appended to the state history
  so the UI can draw the training-loss curve.
* **Accumulating buffer**: every round trains on the union of all rounds'
  ``trading_train.jsonl`` (continual learning), matching the flywheel semantics.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from common.paths import ADAPTER_DIR, ROUNDS_DIR, ensure_dirs, round_dir
from common.state import StateStore

STAGE = "training"

# Matches the numbers MLX-LoRA and HF trainers print, e.g.
#   "Iter 100: Train loss 0.812, ... Val loss 0.456"
_ITER_RE = re.compile(r"[Ii]ter\s+(\d+)")
_TRAIN_LOSS_RE = re.compile(r"[Tt]rain(?:ing)?\s+loss[:\s]+([0-9.]+)")
_VAL_LOSS_RE = re.compile(r"[Vv]al(?:idation)?\s+loss[:\s]+([0-9.]+)")


def find_adapter(round_num: int) -> Optional[str]:
    """Return the trained adapter path for a round, if present."""
    base = ADAPTER_DIR / f"round_{round_num:02d}"
    for name in ("fused_model", "adapters", "final_adapter"):
        if (base / name).exists():
            return str(base / name)
    return None


def _merge_train_sets(round_num: int) -> Path:
    """Concatenate rounds 1..N train splits → merged_train.jsonl."""
    lines: list[str] = []
    for r in range(1, round_num + 1):
        f = round_dir(r) / "trading_train.jsonl"
        if f.exists():
            lines.extend(f.read_text().splitlines(keepends=True))
    merged = ROUNDS_DIR / "merged_train.jsonl"
    merged.write_text("".join(lines))
    return merged


def run_training(
    round_num: int = 1,
    base_model: str = "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit",
    prev_adapter: Optional[str] = None,
    iters: int = 600,
    epochs: int = 2,
    lora_r: int = 16,
    save_every: int = 100,
    resume: bool = True,
) -> dict:
    """Fine-tune on the accumulated buffer for ``round_num``."""
    ensure_dirs()
    store = StateStore(STAGE)

    # ── resume: adapter already trained → skip ────────────────────────────────
    existing = find_adapter(round_num)
    if resume and existing:
        store.finish(message=f"round_{round_num:02d} adapter already exists — skipped",
                     adapter=existing, round=round_num, skipped=True)
        return {"round": round_num, "skipped": True, "adapter": existing}

    merged = _merge_train_sets(round_num)
    n_examples = sum(1 for _ in open(merged)) if merged.exists() else 0
    val_path = round_dir(round_num) / "trading_val.jsonl"
    out_dir = ADAPTER_DIR / f"round_{round_num:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_from = prev_adapter if (prev_adapter and Path(prev_adapter).exists()) else base_model

    store.start(total=iters, message=f"Training round {round_num} on {n_examples} examples")
    store.checkpoint(round=round_num, base_model=base_model, start_from=start_from,
                     n_examples=n_examples)

    train_py = Path(__file__).resolve().parent / "train.py"
    cmd = [
        sys.executable, str(train_py),
        "--train", str(merged),
        "--val", str(val_path) if val_path.exists() else str(merged),
        "--model", start_from,
        "--output_dir", str(out_dir),
        "--epochs", str(epochs),
        "--lora_r", str(lora_r),
        "--iters", str(iters),
        "--save_every", str(save_every),
    ]

    # Stream stdout → parse loss → update progress.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(ADAPTER_DIR.parent.parent))
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        m_iter = _ITER_RE.search(line)
        m_tl = _TRAIN_LOSS_RE.search(line)
        m_vl = _VAL_LOSS_RE.search(line)
        if m_iter or m_tl or m_vl:
            entry = {"event": "loss"}
            if m_iter:
                entry["iter"] = int(m_iter.group(1))
            if m_tl:
                entry["train_loss"] = float(m_tl.group(1))
            if m_vl:
                entry["val_loss"] = float(m_vl.group(1))
            store.log(entry)
            if m_iter:
                store.update(current=int(m_iter.group(1)),
                             message=f"round {round_num}: iter {m_iter.group(1)}/{iters}")
    proc.wait()

    adapter = find_adapter(round_num)
    if adapter is None:
        msg = (f"Training produced no adapter in {out_dir}. "
               "On Mac: pip install mlx-lm. On NVIDIA: pip install torch peft trl bitsandbytes.")
        store.fail(msg)
        raise RuntimeError(msg)

    store.finish(message=f"round_{round_num:02d} trained → {adapter}",
                 adapter=adapter, round=round_num, skipped=False)
    return {"round": round_num, "skipped": False, "adapter": adapter,
            "n_examples": n_examples}
