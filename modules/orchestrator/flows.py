"""
modules/orchestrator/flows.py
─────────────────────────────
Standard workflow orchestration with **Prefect** — the industry pattern of
expressing a pipeline as a DAG of ``@task``s inside a ``@flow``, with built-in
retries, caching, structured logging, scheduling and an observability UI.

Why Prefect (not FastAPI): FastAPI is an *API/UI* layer, not an orchestrator.
Orchestration = sequencing stages, retrying flaky steps, caching completed ones,
and giving a run history. Prefect does that, is pip-installable, and runs locally
with no server. (Airflow/Kubeflow are the heavier, cluster-grade alternatives;
DVC — see ``dvc.yaml`` — is the complementary *reproducible* pipeline DAG.)

Prefect is an **optional** dependency: if it is not installed, callers transparently
fall back to the native in-process orchestrator so nothing breaks in CI.
"""
from __future__ import annotations

from typing import Optional

try:
    from prefect import flow, task
    from prefect.tasks import task_input_hash
    HAS_PREFECT = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PREFECT = False


if HAS_PREFECT:

    @task(retries=2, retry_delay_seconds=10, cache_key_fn=task_input_hash,
          log_prints=True)
    def datagen_task(symbol: str, start_date: str, episodes: int,
                     backend: str, adapter_path: Optional[str]) -> dict:
        from modules.datagen.service import run_datagen
        return run_datagen(symbol=symbol, start_date=start_date, episodes=episodes,
                           backend=backend, adapter_path=adapter_path, resume=True)

    @task(retries=1, log_prints=True)
    def split_task(round_num: int) -> dict:
        from modules.splitter.service import run_split
        return run_split(round_num=round_num, min_rr=1.3)

    @task(retries=2, retry_delay_seconds=30, log_prints=True)
    def train_task(round_num: int, base_model: str, prev_adapter: Optional[str]) -> dict:
        from modules.training.service import run_training
        return run_training(round_num=round_num, base_model=base_model,
                            prev_adapter=prev_adapter, resume=True)

    @task(retries=1, log_prints=True)
    def eval_task(round_num: int, adapter_path: Optional[str]) -> dict:
        from modules.evaluation.service import run_eval
        return run_eval(round_num=round_num, adapter_path=adapter_path)

    @flow(name="trading-flywheel", log_prints=True)
    def trading_flywheel_flow(symbol: str = "CANBK", start_date: str = "2024-01-10",
                              rounds: int = 5, episodes: int = 50,
                              base_model: str = "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit") -> dict:
        """The flywheel as a Prefect flow: datagen → split → train → eval per round."""
        from modules.training.service import find_adapter
        current_adapter = None
        results = []
        for rnd in range(1, rounds + 1):
            backend = "adapter" if current_adapter else "fallback"
            datagen_task(symbol, start_date, episodes * rnd, backend, current_adapter)
            split = split_task(rnd)
            if (split.get("counts", {}).get("train", 0) or 0) < 10:
                results.append({"round": rnd, "trained": False}); continue
            tr = train_task(rnd, base_model, current_adapter)
            current_adapter = tr.get("adapter") or current_adapter or find_adapter(rnd)
            ev = eval_task(rnd, current_adapter)
            results.append({"round": rnd, "trained": True, "trading": ev.get("trading")})
        return {"rounds": results, "final_adapter": current_adapter}


def run_prefect_flywheel(symbol: str = "CANBK", start_date: str = "2024-01-10",
                         rounds: int = 5, episodes: int = 50,
                         base_model: str = "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit",
                         resume: bool = True) -> dict:
    """Run the flywheel via Prefect if available, else the native orchestrator."""
    if HAS_PREFECT:
        return trading_flywheel_flow(symbol=symbol, start_date=start_date,
                                     rounds=rounds, episodes=episodes, base_model=base_model)
    print("[orchestrator] Prefect not installed — falling back to the native "
          "resumable orchestrator. `pip install prefect` to use the standard engine.")
    from modules.orchestrator.service import run_flywheel
    return run_flywheel(symbol=symbol, start_date=start_date, rounds=rounds,
                        episodes=episodes, base_model=base_model, resume=resume)
