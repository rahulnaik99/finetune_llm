"""
flywheel.py  (compatibility shim)
─────────────────────────────────
The flywheel loop was rebuilt on top of the four independent, resumable stages
in ``modules/orchestrator/service.py``. This file is kept so existing muscle
memory / scripts that call ``flywheel`` keep working — it simply delegates to the
new resumable orchestrator (which jumps straight to the first unfinished round
on a restart instead of regenerating data).

Prefer the new entry points:
    python run.py flywheel --rounds 5 --episodes 50
    POST /api/orchestrator/run
"""
from __future__ import annotations

import argparse

# Path bootstrap so the bare-name imports used across stages resolve even when
# this file is run directly.
import common  # noqa: F401
from common.config import load_env, env_str, env_int, env_bool

from modules.orchestrator.service import run_flywheel


def flywheel(symbol, start_date, n_rounds, episodes_per_round, adapter_dir=None,
             data_dir=None, db_path=None, base_model=None, resume=True):
    """Backwards-compatible wrapper around the resumable orchestrator."""
    return run_flywheel(
        symbol=symbol,
        start_date=start_date,
        rounds=n_rounds,
        episodes=episodes_per_round,
        base_model=base_model or "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit",
        resume=resume,
    )


if __name__ == "__main__":
    load_env()
    p = argparse.ArgumentParser(description="Resumable trading-agent flywheel")
    p.add_argument("--symbol", default=env_str("SYMBOL", "CANBK"))
    p.add_argument("--start_date", default=env_str("START_DATE", "2024-01-10"))
    p.add_argument("--rounds", type=int, default=env_int("ROUNDS", 5))
    p.add_argument("--episodes", type=int, default=env_int("EPISODES", 50))
    p.add_argument("--base_model", default=env_str(
        "BASE_MODEL", "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit"))
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print(f"  FLYWHEEL  symbol={args.symbol}  start={args.start_date}  "
          f"rounds={args.rounds}  episodes={args.episodes}")
    print(f"    base_model={args.base_model}  USE_OLLAMA={env_bool('USE_OLLAMA', False)}")
    print("=" * 70)

    out = run_flywheel(symbol=args.symbol, start_date=args.start_date,
                       rounds=args.rounds, episodes=args.episodes,
                       base_model=args.base_model, resume=not args.no_resume)
    print("Final adapter:", out.get("final_adapter"))
