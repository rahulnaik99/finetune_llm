# Module: Training (`modules/training`)

Fine-tunes the base LLM into the trading student using **parameter-efficient**
methods, on whatever hardware is available.

## What it does

`train.py` **auto-detects hardware** and picks a backend:
- **Apple Silicon (M1–M5)** → **MLX-LoRA** (`mlx_lm`), no CUDA needed.
- **NVIDIA GPU** → **QLoRA** (4-bit) via `transformers` + `peft` + `trl` +
  `bitsandbytes`; fits a free Colab T4 (8 GB).

The training service:
- Merges **all** rounds' `trading_train.jsonl` into one buffer (continual
  learning — the model keeps everything it has seen).
- Starts from the **previous round's adapter** when available.
- Streams the child process's stdout, parses `Iter / train loss / val loss`,
  and appends them to the state history so the UI can draw the **loss curve**.

## Why LoRA / QLoRA

The base model has ~4B parameters. Full fine-tuning needs ~60 GB VRAM. LoRA
freezes the 4B weights and trains tiny low-rank `A×B` adapters (~0.8% of
params); QLoRA additionally 4-bit-quantises the frozen base. Result: fine-tuning
on consumer hardware at a fraction of the cost, with adapters that are a few MB
to ship.

## Resumability

Before training a round, the service checks for an existing adapter
(`output/adapters/round_NN/{fused_model,adapters,final_adapter}`). If found and
`resume=true`, it **skips** — no GPU hours wasted re-training a completed round.
MLX itself also checkpoints every `SAVE_EVERY` iterations.

## Files

| File | Role |
|---|---|
| `train.py` | Hardware auto-detect + MLX/QLoRA training, `params.yaml` + `.env` aware |
| `train_mlx.py` | Standalone MLX-LoRA backend (reference) |
| `train_qlora.py` | Standalone QLoRA backend (reference) |
| `service.py` | Resumable driver, buffer merge, loss parsing |
| `api.py` | FastAPI router |

## Tech & libraries

| Library | Used for | Benefit |
|---|---|---|
| **mlx-lm** | LoRA on Apple Silicon | Native Metal; no NVIDIA stack required |
| **transformers** | Model/tokenizer loading | Standard HF ecosystem |
| **peft** | LoRA/QLoRA adapters | Industry-standard PEFT implementation |
| **trl** (`SFTTrainer`) | Supervised fine-tuning loop | Handles chat formatting + packing |
| **bitsandbytes** | 4-bit quantisation (QLoRA) | Cuts VRAM ~4× |
| **PyYAML** | `params.yaml` hyper-parameters | Central, code-free tuning |
| **subprocess** (stdlib) | Launch `train.py` in isolation | Keeps heavy backends out of the API process |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/training/run` | Fine-tune a round (skips if adapter exists) |
| GET | `/api/training/state` | Progress + loss history (dashboard feed) |
| GET | `/api/training/adapter?round_num=1` | Resolved adapter path |

## CLI

```bash
python run.py train --round 1 --iters 600
python run.py train --round 1 --no-resume    # retrain even if adapter exists
```

## Key tunables (`.env` / `params.yaml`)

`BASE_MODEL`, `MLX_ITERS`, `LORA_R`, `MLX_NUM_LAYERS`, `EPOCHS`, `SAVE_EVERY`.
