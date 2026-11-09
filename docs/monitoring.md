# Module: Monitoring (`modules/monitoring`)

Optional production-health stage. Detects when the live decision distribution
**drifts** away from the training distribution — the signal that the market
regime changed and the model should be retrained.

## What it does

`monitoring.py` loads a *reference* JSONL (the training decisions) and a
*current* JSONL (recent live decisions), extracts the numeric decision features
(action, confidence, risk_reward, SL, TP, reasoning length), and builds an
**Evidently** drift + data-quality report. `deploy_to_ollama.py` packages a
fused model back into Ollama with the extended context.

## Tech & libraries

| Library | Used for | Benefit |
|---|---|---|
| **Evidently** (optional) | Drift + data-quality report | Rich, standard drift analytics + HTML report |
| **pandas** | Feature frames | Easy column-wise comparison |
| **mlflow** (optional) | Log drift metrics | Central experiment tracking |

The heavy Evidently import stays inside the worker; if it is not installed the
service degrades gracefully instead of crashing the API.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/monitoring/drift` | Reference vs current drift report |
| GET | `/api/monitoring/state` | Progress / last summary |

## Example

```bash
curl -X POST localhost:8000/api/monitoring/drift \
  -H 'content-type: application/json' \
  -d '{"reference_path":"data/rounds/round_01/trading_train.jsonl",
       "current_path":"data/rounds/round_05/trading_train.jsonl"}'
```
