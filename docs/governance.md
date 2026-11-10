# Module: Governance (`modules/governance`)

Auto-generates the two artifacts the industry expects to ship with a model, from
**live facts** (the eval-metrics cache + the trade-log DB + config) so they never
go stale by hand.

## What it produces

- **Model Card** (`docs/generated/MODEL_CARD.md`) — Mitchell et al. style: what
  the model is, its intended use, headline metrics (win rate, expectancy, profit
  factor, Sharpe, **Sortino**, t-stat/significance, max drawdown), the
  **walk-forward robustness** summary, and limitations / responsible-use notes.
- **Dataset Datasheet** (`docs/generated/DATASHEET.md`) — Gebru et al. style:
  what the data is, where it came from (CSV prices / SEBI XBRL fundamentals /
  India VIX), how it was cleaned (quality filter, exact-hash de-dup, balancing),
  the walk-forward split, and known biases.

*Real-world analogy:* the nutrition label + allergy warnings on a food product —
you know what's inside and how it was made before you consume it.

## Files

| File | Role |
|---|---|
| `service.py` | `generate_model_card(round_num)`, `generate_datasheet()`, `generate_all()` |
| `api.py` | FastAPI router |

Facts are pulled from `output/metrics/eval_round_NN.json` (written by the
evaluation stage) and the `trade_log` table, so the card always reflects the
latest run.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/governance/generate` | (Re)generate both documents (optional `?round_num=`) |
| GET | `/api/governance/model_card` | The current Model Card (markdown) |
| GET | `/api/governance/datasheet` | The current Datasheet (markdown) |

## CLI

```bash
python run.py card            # writes docs/generated/MODEL_CARD.md + DATASHEET.md
```

Generated files live under `docs/generated/` and are git-ignored (regenerate any
time from the current metrics + data).
