# API reference

All endpoints are served by `app/main.py` under the `/api` prefix. Interactive
OpenAPI docs are available at `/docs` (Swagger) and `/redoc` when the server is
running. The UI is served at `/ui`.

Start the server:

```bash
python run.py serve            # or: uvicorn app.main:app --port 8000
```

## Cross-stage

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness probe |
| GET | `/api/overview` | State of every stage (one call the UI polls) |

## Per stage

Every stage exposes `POST /<stage>/run` (launch/resume, non-blocking → 202-style
accepted) and `GET /<stage>/state` (progress). A `409` is returned if a job for
that stage is already running; a `422` for invalid input.

| Stage | Run | State | Extra |
|---|---|---|---|
| datagen | `POST /api/datagen/run` | `GET /api/datagen/state` | `GET /api/datagen/summary`, `GET /api/datagen/trades` |
| splitter | `POST /api/splitter/run` | `GET /api/splitter/state` | `GET /api/splitter/rounds`, `GET /api/splitter/dataset` |
| training | `POST /api/training/run` | `GET /api/training/state` | `GET /api/training/adapter` |
| evaluation | `POST /api/evaluation/run` | `GET /api/evaluation/state` | `GET /api/evaluation/metrics` |
| orchestrator | `POST /api/orchestrator/run` | `GET /api/orchestrator/state` | — |
| monitoring | `POST /api/monitoring/drift` | `GET /api/monitoring/state` | — |

## Example requests

```bash
# generate 50 episodes (resumes automatically)
curl -X POST localhost:8000/api/datagen/run \
  -H 'content-type: application/json' \
  -d '{"symbol":"CANBK","episodes":50,"backend":"fallback","resume":true}'

# build the round-1 dataset
curl -X POST localhost:8000/api/splitter/run \
  -H 'content-type: application/json' -d '{"round_num":1,"min_rr":1.3}'

# run the whole flywheel
curl -X POST localhost:8000/api/orchestrator/run \
  -H 'content-type: application/json' \
  -d '{"symbol":"CANBK","rounds":5,"episodes":50,"resume":true}'

# poll everything
curl localhost:8000/api/overview
```

## The UI

`app/ui/` is a dependency-free dashboard (SAP "Clean Minimal" light theme) that
polls these endpoints every few seconds and provides:

- **Overview** — status board for all six stages + a flywheel launcher.
- **Data Generation** — run/resume, live progress, recent generated trades.
- **Dataset Viewer** — browse per-round train/val/test examples.
- **Training** — run, live progress + a train/val **loss curve**.
- **Evaluation** — run, trading-metric cards + a **win-rate-by-round** chart.
