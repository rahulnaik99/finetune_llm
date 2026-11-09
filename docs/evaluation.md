# Module: Evaluation (`modules/evaluation`)

Measures how good a round's model is, on a **walk-forward** (future-dated) test
split the model never trained on.

## Two layers of metrics

### A) Trading metrics (pure math, always available)
Computed from the recorded P&L of the test-split trades (via the `_meta`
sidecar the splitter writes):

- `win_rate`, `avg_win`, `avg_loss`
- `expectancy` (avg P&L / trade)
- `profit_factor` (gross win / gross loss)
- `sharpe_per_trade`
- `total_pnl`, `max_drawdown`

### B) Model-quality (optional, needs an MLX model)
Runs the fine-tuned model over the test prompts and scores the generations:

- `json_validity`, `schema_ok`, `reasoning_ok`, `features_used_ok`
- `action_match` (agreement with the teacher's action)

## Why walk-forward

`SPLIT_MODE=time` guarantees the test dates are *after* the training dates, so
these numbers reflect performance on genuinely unseen future market conditions —
the honest measure for a trading model.

## Resumability

Each round's result is cached to `output/metrics/eval_round_NN.json`, and a
per-round summary is appended to the state history so the dashboard can chart
metrics **across rounds**.

## Tech & libraries

| Library | Used for | Benefit |
|---|---|---|
| **math / json** (stdlib) | Metric computation + JSON parsing | No dependency; trading metrics run anywhere |
| **mlx-lm** (optional) | Generate decisions for model-quality | Only needed when you ask for quality metrics |
| **mlflow** (optional) | Experiment tracking | Metrics land in MLflow when configured |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/evaluation/run` | Evaluate a round's test split |
| GET | `/api/evaluation/state` | Progress + per-round metric history |
| GET | `/api/evaluation/metrics?round_num=1` | Cached metrics for a round |

## CLI

```bash
python run.py eval --round 1
python run.py eval --round 1 --quality     # also run model-quality (needs MLX model)
```
