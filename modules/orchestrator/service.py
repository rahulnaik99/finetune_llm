"""
modules/orchestrator/service.py
───────────────────────────────
The flywheel, rebuilt on top of the four independent, resumable stages.

For each round it calls, in order:
    datagen.run_datagen → splitter.run_split → training.run_training
    → evaluation.run_eval

Because every stage is individually resumable (SQLite trade log, idempotent
split fingerprint, adapter-exists skip, cached metrics), the orchestrator only
needs to remember **which round it finished**. On resume it jumps straight to
the first unfinished round — no data is regenerated and no adapter retrained.

This replaces the monolithic ``flywheel.py`` loop; the old script is kept as a
thin reference entry point but delegates here.
"""
from __future__ import annotations

from typing import Optional

from common.paths import ADAPTER_DIR, ensure_dirs
from common.state import StateStore

STAGE = "orchestrator"


def run_flywheel(
    symbol: str = "CANBK",
    start_date: str = "2024-01-10",
    rounds: int = 5,
    episodes: int = 50,
    base_model: str = "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit",
    resume: bool = True,
) -> dict:
    ensure_dirs()
    store = StateStore(STAGE)

    # Lazy imports so the FastAPI process / tests stay light.
    from modules.datagen.service import run_datagen
    from modules.splitter.service import run_split
    from modules.training.service import run_training, find_adapter
    from modules.evaluation.service import run_eval

    prev = store.load()
    last_done = int(prev.checkpoint.get("last_round_done", 0)) if resume else 0
    current_adapter = prev.checkpoint.get("current_adapter") if resume else None

    store.start(total=rounds, message=f"Flywheel {symbol}: {rounds} rounds × {episodes} episodes")
    results = list(prev.checkpoint.get("rounds", [])) if resume else []

    for rnd in range(last_done + 1, rounds + 1):
        store.update(current=rnd - 1, message=f"Round {rnd}/{rounds}: data generation")

        # 1) generate (uses current best adapter if we have one)
        backend = "adapter" if current_adapter else "fallback"
        run_datagen(symbol=symbol, start_date=start_date,
                    episodes=episodes * rnd,  # accumulating target across rounds
                    backend=backend, adapter_path=current_adapter, resume=True)

        # 2) build + split this round's dataset
        store.update(message=f"Round {rnd}/{rounds}: dataset split")
        split = run_split(round_num=rnd, min_rr=1.3)

        if (split.get("counts", {}).get("train", 0) or 0) < 10:
            results.append({"round": rnd, "trained": False, "reason": "too few examples"})
            store.checkpoint(last_round_done=rnd, rounds=results,
                             current_adapter=current_adapter)
            continue

        # 3) fine-tune on the accumulated buffer
        store.update(message=f"Round {rnd}/{rounds}: fine-tuning")
        try:
            tr = run_training(round_num=rnd, base_model=base_model,
                              prev_adapter=current_adapter, resume=True)
            current_adapter = tr.get("adapter") or current_adapter or find_adapter(rnd)
        except Exception as exc:
            # Do NOT advance last_round_done — a failed round must be RETRIED on
            # the next resume, not skipped forever. Record and stop here.
            results.append({"round": rnd, "trained": False, "error": str(exc)})
            store.checkpoint(rounds=results, current_adapter=current_adapter)
            store.fail(f"Round {rnd} training failed: {exc}. Resume to retry.")
            return {"rounds": results, "final_adapter": current_adapter,
                    "failed_round": rnd}

        # 4) evaluate on the walk-forward test split
        store.update(message=f"Round {rnd}/{rounds}: evaluation")
        ev = run_eval(round_num=rnd, adapter_path=current_adapter)

        results.append({"round": rnd, "trained": True,
                        "adapter": current_adapter,
                        "trading": ev.get("trading")})
        store.checkpoint(last_round_done=rnd, rounds=results,
                         current_adapter=current_adapter)

    store.finish(message=f"Flywheel complete — {rounds} rounds. "
                 f"Final adapter: {current_adapter}",
                 current_adapter=current_adapter, rounds=results)
    return {"rounds": results, "final_adapter": current_adapter}
