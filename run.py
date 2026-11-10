#!/usr/bin/env python3
"""
run.py — single entry point for the whole pipeline.

    python run.py serve [--port 8000]       # start the API + UI
    python run.py datagen  --symbol CANBK --episodes 50
    python run.py split    --round 1
    python run.py train    --round 1 --iters 600
    python run.py eval     --round 1
    python run.py flywheel --rounds 5 --episodes 50
    python run.py status                     # print every stage's state

Every stage is resumable: re-running a command continues from the last
checkpoint instead of regenerating data / retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `python run.py ...` work from anywhere / regardless of PYTHONSAFEPATH:
# ensure the project root (this file's dir) is importable before touching common.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: F401,E402 — path bootstrap
from common.config import load_env


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    load_env()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")

    d = sub.add_parser("datagen")
    d.add_argument("--symbol", default="CANBK"); d.add_argument("--start_date", default="2024-01-10")
    d.add_argument("--episodes", type=int, default=50); d.add_argument("--backend", default="fallback")
    d.add_argument("--no-resume", action="store_true")
    d.add_argument("--symbols", default=None,
                   help="Comma-separated symbols → run them in PARALLEL (e.g. CANBK,SBIN,HDFCBANK)")
    d.add_argument("--workers", type=int, default=None, help="Parallel worker processes")
    d.add_argument("--windows", type=int, default=1,
                   help="Split ONE symbol into N parallel date-windows (parallel episodes)")

    sp = sub.add_parser("split"); sp.add_argument("--round", type=int, default=1)
    sp.add_argument("--force", action="store_true")

    tr = sub.add_parser("train"); tr.add_argument("--round", type=int, default=1)
    tr.add_argument("--iters", type=int, default=600); tr.add_argument("--no-resume", action="store_true")

    ev = sub.add_parser("eval"); ev.add_argument("--round", type=int, default=1)
    ev.add_argument("--quality", action="store_true")

    cd = sub.add_parser("card"); cd.add_argument("--round", type=int, default=None)
    vd = sub.add_parser("validate")
    vd.add_argument("--prices", default="data/prices/CANBK.csv")
    vd.add_argument("--vix", default="data/prices/INDIAVIX.csv")

    fw = sub.add_parser("flywheel")
    fw.add_argument("--symbol", default="CANBK"); fw.add_argument("--start_date", default="2024-01-10")
    fw.add_argument("--rounds", type=int, default=5); fw.add_argument("--episodes", type=int, default=50)
    fw.add_argument("--no-resume", action="store_true")
    fw.add_argument("--engine", choices=["native", "prefect"], default="native",
                    help="native in-process loop, or the standard Prefect workflow engine")

    sub.add_parser("status")
    args = p.parse_args()

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
    elif args.cmd == "datagen":
        if args.windows and args.windows > 1 and not args.symbols:
            from modules.datagen.service import run_datagen_windows
            _print(run_datagen_windows(
                symbol=args.symbol, start_date=args.start_date, episodes=args.episodes,
                workers=args.windows, backend=args.backend, resume=not args.no_resume))
        elif args.symbols:
            from modules.datagen.service import run_datagen_parallel
            _print(run_datagen_parallel(
                symbols=[s for s in args.symbols.split(",") if s.strip()],
                start_date=args.start_date, episodes=args.episodes,
                backend=args.backend, resume=not args.no_resume,
                max_workers=args.workers))
        else:
            from modules.datagen.service import run_datagen
            _print(run_datagen(symbol=args.symbol, start_date=args.start_date,
                               episodes=args.episodes, backend=args.backend,
                               resume=not args.no_resume))
    elif args.cmd == "split":
        from modules.splitter.service import run_split
        _print(run_split(round_num=args.round, force=args.force))
    elif args.cmd == "train":
        from modules.training.service import run_training
        _print(run_training(round_num=args.round, iters=args.iters,
                            resume=not args.no_resume))
    elif args.cmd == "eval":
        from modules.evaluation.service import run_eval
        _print(run_eval(round_num=args.round, model_quality=args.quality))
    elif args.cmd == "flywheel":
        if args.engine == "prefect":
            from modules.orchestrator.flows import run_prefect_flywheel
            _print(run_prefect_flywheel(symbol=args.symbol, start_date=args.start_date,
                                        rounds=args.rounds, episodes=args.episodes,
                                        resume=not args.no_resume))
        else:
            from modules.orchestrator.service import run_flywheel
            _print(run_flywheel(symbol=args.symbol, start_date=args.start_date,
                               rounds=args.rounds, episodes=args.episodes,
                               resume=not args.no_resume))
    elif args.cmd == "card":
        from modules.governance.service import generate_all
        _print(generate_all(args.round))
    elif args.cmd == "validate":
        from common.validation import validate_inputs
        _print(validate_inputs(args.prices, args.vix))
    elif args.cmd == "status":
        from common.state import all_states
        _print(all_states())


if __name__ == "__main__":
    main()
