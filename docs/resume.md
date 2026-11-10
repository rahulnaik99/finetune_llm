# Resumability — "a break anywhere can resume without regenerating data"

This is the backbone requirement. Every stage persists enough state that an
interrupted pipeline continues from where it stopped.

## The mechanism

`common/state.py` provides a `StateStore` per stage, writing a single JSON file
under `output/state/<stage>.json`:

- **Atomic writes** — temp file + `os.replace`, so a crash mid-write never
  corrupts the file and the UI can safely read while a job writes.
- **Checkpoint payload** — stage-specific resume pointers.
- **History** — a timeline (training loss, eval metrics) for the dashboards.
- **Corruption-safe** — a partial/garbage file loads as a clean empty state
  instead of raising.

Heavy data itself never lives in state — it stays in SQLite (episodes), JSONL
(datasets) and on disk (adapters). State only tracks *progress* and *pointers*,
which is what makes resuming cheap.

## Per-stage resume contract

| Stage | Persisted state | What a resume skips |
|---|---|---|
| **datagen** | `episodes_done`, `date`, `capital` (per episode) | Re-runs only the missing episodes; the checkpoint is **reconciled against the real DB** (and reset if the symbol/date changed) so a stale state can't skip work |
| **splitter** | DB+params `fingerprint`, `counts` | Reuses the existing split if the DB is unchanged |
| **training** | adapter path per round | Skips a round whose adapter already exists; a non-zero exit is treated as failure (no half-written adapter) |
| **evaluation** | cached `eval_round_NN.json` + **test-set fingerprint** | Reuses the cache only if the test set is unchanged (`force` bypasses) |
| **orchestrator** | `last_round_done`, `current_adapter` | Jumps to the first unfinished round; a **failed round is retried** on the next resume, not skipped |

## Example: crash & resume

```bash
python run.py datagen --symbol CANBK --episodes 50
#  ... killed at episode 37 ...
python run.py datagen --symbol CANBK --episodes 50
#  → "37 already done" → generates episodes 38..50 only
```

```bash
python run.py flywheel --rounds 5 --episodes 50
#  ... killed during round 3 training ...
python run.py flywheel --rounds 5 --episodes 50
#  → rounds 1–2 already done → resumes at round 3 (datagen already satisfied,
#    split reused, training continues) — no identical data regenerated
```

## Forcing a fresh run

Pass `--no-resume` (CLI) / `"resume": false` (API) for datagen / training /
flywheel, or `--force` / `"force": true` for the splitter.

Tested by `tests/test_state.py` (roundtrip, cross-instance resume, history,
corruption safety).
