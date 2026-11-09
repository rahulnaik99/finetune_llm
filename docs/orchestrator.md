# Module: Orchestrator (`modules/orchestrator`)

The **flywheel**, rebuilt on top of the four independent, resumable stages. It
runs the self-improving loop:

```
for each round:
    datagen → splitter → training → evaluation
    (the trained adapter becomes the backend for the next round's datagen)
```

## Why it is thin now

Because every stage is individually resumable (SQLite trade log, split
fingerprint, adapter-exists skip, cached metrics), the orchestrator only has to
remember **which round it finished**. State lives in
`output/state/orchestrator.json` as `last_round_done` + `current_adapter` + a
per-round results log.

On resume it jumps straight to the first unfinished round — **no data is
regenerated and no adapter retrained**. This replaces the original monolithic
`flywheel.py` loop, which restarted everything on any failure.

## Per-round logic

1. `datagen` with the current best adapter (or `fallback` in round 1); the
   accumulating episode target means early episodes are never redone.
2. `splitter` builds this round's dataset; rounds with `<10` train examples skip
   training (and are recorded as such).
3. `training` fine-tunes on the accumulated buffer, continuing from the previous
   adapter.
4. `evaluation` scores the new adapter on the walk-forward test split.

## Tech & libraries

| Library | Used for | Benefit |
|---|---|---|
| **In-process service calls** | Chain the 4 stages | Shared resume semantics, no subprocess overhead |
| `common.state` | Round-level checkpoint | Cheap, crash-safe resume |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/orchestrator/run` | Launch/resume the full flywheel |
| GET | `/api/orchestrator/state` | Progress across rounds + per-round results |

## CLI

```bash
python run.py flywheel --symbol CANBK --rounds 5 --episodes 50
python run.py flywheel --rounds 5 --episodes 50               # resumes from last round
python run.py flywheel --rounds 5 --episodes 50 --no-resume   # start over
```
