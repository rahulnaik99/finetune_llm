"""
train_mlx.py
────────────
LoRA fine-tuning using Apple MLX — native M-series Silicon.
No CUDA, no bitsandbytes, no NVIDIA GPU needed.

Requirements:
    pip install mlx-lm

Supports: M1 / M2 / M3 / M4 / M5  (all unified memory Apple Silicon)

Speed on M5 Air 16GB:
  Nemotron-4B LoRA:   ~5-8 tokens/sec  ≈ 8-12h for 500 examples, 3 epochs
  For faster training: use Colab A100 (see colab_train.ipynb)

Usage:
    python train_mlx.py \
        --train data/trading_train.jsonl \
        --val   data/trading_val.jsonl \
        --model mlx-community/Nemotron-Nano-4B-Instruct-4bit \
        --output output/nemotron-trading-mlx
"""

import argparse, json, subprocess, sys, shutil
from pathlib import Path


# ── MLX-LM expects data in a specific format ─────────────────────────────────
def convert_to_mlx_format(jsonl_path: str, out_path: str) -> int:
    """
    MLX-LM needs JSONL where each line has a "text" field with the
    full conversation already formatted as a single string.
    Our pipeline already produces {"messages": [...]} — we just need
    to apply the chat template manually here.
    """
    # Nemotron / ChatML template
    def apply_chatml(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")  # generation prompt
        return "\n".join(parts)

    count = 0
    with open(jsonl_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            messages = ex.get("messages", [])
            if not messages:
                continue
            # For training we include the full conversation including assistant turn
            text = "\n".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                for m in messages
            )
            fout.write(json.dumps({"text": text}) + "\n")
            count += 1
    return count


def train_mlx(
    train_path: str,
    val_path: str | None,
    model: str,
    output_dir: str,
    lora_layers: int = 8,       # number of transformer layers to adapt (more = better but slower)
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_iterations: int = 600,   # ~3 epochs for 500 examples, batch_size=4 (500/4 * 3 ≈ 375, use 600)
    steps_per_eval: int = 50,
    save_every: int = 100,
    max_seq_length: int = 1024,  # keep lower on 16GB to avoid OOM
    grad_checkpoint: bool = True,
):
    """Run mlx_lm.lora training."""

    # ── Convert data format ──────────────────────────────────────────────────
    mlx_data_dir = Path(output_dir) / "mlx_data"
    mlx_data_dir.mkdir(parents=True, exist_ok=True)

    print("Converting data to MLX format...")
    n_train = convert_to_mlx_format(train_path, str(mlx_data_dir / "train.jsonl"))
    print(f"  Train: {n_train} examples")

    if val_path:
        n_val = convert_to_mlx_format(val_path, str(mlx_data_dir / "valid.jsonl"))
        print(f"  Val:   {n_val} examples")

    # MLX-LM also needs a test.jsonl (can be same as val)
    shutil.copy(
        str(mlx_data_dir / ("valid.jsonl" if val_path else "train.jsonl")),
        str(mlx_data_dir / "test.jsonl"),
    )

    # ── Build mlx_lm.lora command ────────────────────────────────────────────
    adapter_path = Path(output_dir) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",            model,
        "--train",
        "--data",             str(mlx_data_dir),
        "--adapter-path",     str(adapter_path),
        "--num-layers",       str(lora_layers),
        "--batch-size",       str(batch_size),
        "--learning-rate",    str(learning_rate),
        "--iters",            str(num_iterations),
        "--steps-per-eval",   str(steps_per_eval),
        "--save-every",       str(save_every),
        "--max-seq-length",   str(max_seq_length),
    ]

    if grad_checkpoint:
        cmd.append("--grad-checkpoint")   # saves ~30% memory at cost of ~10% speed

    print(f"\nStarting MLX LoRA training...")
    print(f"  Model:       {model}")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LoRA layers: {lora_layers}")
    print(f"  Output:      {adapter_path}")
    print(f"\n  Estimated time on M5 Air 16GB: "
          f"{num_iterations * batch_size * max_seq_length // 5 // 3600:.0f}-"
          f"{num_iterations * batch_size * max_seq_length // 3 // 3600:.0f} hours")
    print(f"  (run overnight or use Colab — see note below)\n")

    result = subprocess.run(cmd, check=True)

    # ── Fuse adapter into model (optional — for single-file deployment) ───────
    fused_path = Path(output_dir) / "fused_model"
    print(f"\nFusing adapter into model → {fused_path}")
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        model,
        "--adapter-path", str(adapter_path),
        "--save-path",    str(fused_path),
        "--de-quantize",  # optional: convert back to float16 for best quality
    ]
    try:
        subprocess.run(fuse_cmd, check=True)
        print(f"✓ Fused model saved → {fused_path}")
    except Exception as e:
        print(f"  Fuse step failed (adapter still works): {e}")

    print(f"\n✓ Training complete")
    print(f"  Adapter: {adapter_path}")
    print(f"  Use in Ollama: create a Modelfile pointing to the fused model")
    return str(adapter_path)


# ── Inference with fine-tuned MLX model ──────────────────────────────────────
def generate_mlx(model_path: str, system: str, user: str,
                 adapter_path: str | None = None, max_tokens: int = 512) -> str:
    """
    Run inference using the fine-tuned MLX model.
    Use this instead of the Ollama _ollama() call once training is done.
    """
    from mlx_lm import load, generate

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        verbose=False)
    return response


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",        required=True)
    p.add_argument("--val",          default=None)
    # MLX community provides pre-quantized models — use 4-bit for 16GB
    p.add_argument("--model",        default="mlx-community/Nemotron-Nano-4B-Instruct-4bit")
    p.add_argument("--output_dir",   default="output/nemotron-mlx")
    p.add_argument("--lora_layers",  type=int, default=8)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--iters",        type=int, default=600)
    p.add_argument("--max_seq_len",  type=int, default=1024)
    args = p.parse_args()

    train_mlx(
        train_path=args.train,
        val_path=args.val,
        model=args.model,
        output_dir=args.output_dir,
        lora_layers=args.lora_layers,
        batch_size=args.batch_size,
        num_iterations=args.iters,
        max_seq_length=args.max_seq_len,
    )
