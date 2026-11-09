"""
prefect_flows.py — production orchestration for scoring + fine-tuning flywheel.
Start server: prefect server start
Deploy:       prefect deploy prefect_flows.py:weekly_flywheel
"""
from prefect import flow, task
from prefect.schedules import CronSchedule
from datetime import timedelta


@task(name="fetch-universe", retries=2, retry_delay_seconds=30)
def fetch_universe(path: str = "app/config/universe.csv") -> list[dict]:
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


@task(name="run-simulation", retries=1, retry_delay_seconds=120, timeout_seconds=7200)
def run_simulation_task(symbol: str, start_date: str, n_episodes: int, adapter: str | None) -> dict:
    import subprocess, sys
    cmd = [sys.executable, "flywheel.py", "--symbol", symbol,
           "--start_date", start_date, "--rounds", "1", "--episodes", str(n_episodes)]
    if adapter:
        cmd += ["--adapter", adapter]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return {"symbol": symbol, "episodes": n_episodes}


@task(name="build-dataset", retries=1)
def build_dataset_task(db: str, out: str, rnd: int) -> int:
    from data_pipeline import build_dataset
    c = build_dataset(db, f"{out}/round_{rnd:02d}/trading.jsonl")
    return c.get("train", 0)


@task(name="finetune", timeout_seconds=28800)
def finetune_task(data_dir: str, rnd: int, adapter_dir: str, base_model: str, prev: str | None) -> str:
    import subprocess, sys
    from pathlib import Path
    new = f"{adapter_dir}/round_{rnd:02d}"
    if Path("axolotl_config.yaml").exists():
        cmd = ["axolotl", "train", "axolotl_config.yaml", f"--output_dir={new}"]
    else:
        cmd = [sys.executable, "train_mlx.py",
               "--train", f"{data_dir}/round_{rnd:02d}/trading_train.jsonl",
               "--output_dir", new]
    subprocess.run(cmd, check=True, timeout=28800)
    return f"{new}/final_adapter"


@task(name="evaluate")
def evaluate_task(adapter: str, test: str) -> dict:
    from evaluate import evaluate
    return evaluate(adapter, test, "nvidia/Nemotron-Nano-4B-Instruct")


@flow(name="daily-scoring", schedule=CronSchedule(cron="30 9 * * 1-5"),
      description="9:30 AM IST daily universe scan")
def daily_scoring_run(symbols: list[str] | None = None,
                      model_config: str = "app/service/models/model_config.json"):
    import mlflow, httpx
    if symbols is None:
        symbols = [r["symbol"] for r in fetch_universe()]
    print(f"Scoring {len(symbols)} stocks")
    with mlflow.start_run(run_name=f"daily-{len(symbols)}"):
        mlflow.log_params({"n_symbols": len(symbols), "model_config": model_config})
        scored = 0
        for sym in symbols:
            try:
                httpx.post("http://localhost:8000/api/runs",
                           json={"symbols": [sym], "mode": "inprocess"}, timeout=900)
                scored += 1
            except Exception as e:
                print(f"  {sym} failed: {e}")
        mlflow.log_metric("n_scored", scored)
    return scored


@flow(name="weekly-flywheel", schedule=CronSchedule(cron="0 2 * * 0"),
      description="Every Sunday 2AM: simulate -> fine-tune -> evaluate")
def weekly_flywheel(symbol: str = "CANBK", start_date: str = "2024-01-10",
                    n_rounds: int = 1, episodes: int = 50,
                    adapter_dir: str = "output/adapters",
                    data_dir: str = "data/rounds",
                    db: str = "trading_agent.db"):
    import mlflow
    current = None
    with mlflow.start_run(run_name=f"flywheel-{symbol}"):
        for rnd in range(1, n_rounds + 1):
            run_simulation_task(symbol, start_date, episodes, current)
            n = build_dataset_task(db, data_dir, rnd)
            mlflow.log_metric(f"r{rnd}_examples", n)
            if n < 10:
                continue
            current = finetune_task(data_dir, rnd, adapter_dir,
                                    "nvidia/Nemotron-Nano-4B-Instruct", current)
            m = evaluate_task(current, f"{data_dir}/round_{rnd:02d}/trading_test.jsonl")
            mlflow.log_metrics({f"r{rnd}_{k}": v for k, v in m.items()
                                 if isinstance(v, (int, float))})
    return current
