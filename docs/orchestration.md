# Workflow orchestration & reproducibility

**Short version:** FastAPI is the *API/UI* layer, not an orchestrator. For
standard workflow orchestration this project uses **Prefect** (runtime DAG with
retries/caching/observability) and **DVC** (reproducible, versioned pipeline).
Both are optional installs and both call the same `run.py` stage commands, so
nothing is duplicated.

## Why not "just FastAPI"?

FastAPI exposes buttons and status. Orchestration is a different job: sequence
stages, retry flaky steps, cache finished ones, schedule runs, and keep a run
history. That belongs to a workflow engine.

| Layer | Tool here | Role |
|---|---|---|
| API / UI | **FastAPI** (`app/main.py`) | trigger stages, show live progress |
| Runtime orchestration | **Prefect** (`modules/orchestrator/flows.py`) | DAG, retries, caching, logging, scheduling, UI |
| Reproducibility / versioning | **DVC** (`dvc.yaml`, `params.yaml`) | content-addressed stage DAG; re-run only what changed; version data+models+metrics |
| Native fallback | `modules/orchestrator/service.py` | zero-dependency in-process loop (used when Prefect isn't installed) |

The heavier cluster-grade alternatives (Airflow, Kubeflow, Dagster, Metaflow,
SageMaker/Vertex Pipelines) slot in at the Prefect layer if you outgrow a single
machine — the stage services (`run_datagen`, `run_split`, `run_training`,
`run_eval`) stay unchanged; only the wrapper changes.

## Prefect (runtime engine)

```bash
pip install prefect
python run.py flywheel --engine prefect --rounds 5 --episodes 50
```

`flows.py` declares each stage as a `@task` (with `retries`, `retry_delay`,
`cache_key_fn=task_input_hash`) and the loop as a `@flow`. If Prefect is not
installed, `run_prefect_flywheel` transparently falls back to the native
orchestrator, so CI and offline users are never blocked.

Benefits you get for free: automatic retries on a flaky stage, caching so a
re-run skips unchanged tasks, structured run logs, and (with `prefect server
start`) a web UI of every run.

## DVC (reproducible pipeline)

```bash
pip install dvc
dvc repro            # run only the stages whose inputs changed
dvc dag              # print the datagen → split → train → evaluate → governance graph
dvc metrics show     # compare eval metrics across git commits
```

`dvc.yaml` defines five stages with explicit `deps`/`outs` and reads
hyper-parameters from `params.yaml`. Change `lora_r` and `dvc repro` re-runs only
training + evaluation, not data generation — the standard "don't recompute what
didn't change" guarantee, plus full data/model lineage.

## How they fit together

- Day-to-day / interactive → **FastAPI UI** or `run.py`.
- Automated, retried, scheduled runs → **Prefect**.
- Reproducible, versioned, "prove how this model was built" → **DVC**.
