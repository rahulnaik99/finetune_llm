"""
train.py
────────
Unified training entry point. Auto-detects hardware and picks the right method:

  Apple Silicon (M1-M5)  →  MLX  LoRA    (native, no CUDA needed)
  NVIDIA ≥ 16 GB VRAM    →  HF   LoRA    (bfloat16, no quantisation)
  NVIDIA  8-16 GB VRAM   →  HF   QLoRA   (8-bit base + LoRA)
  NVIDIA  < 8 GB VRAM    →  HF   QLoRA   (4-bit NF4 base + LoRA)
  CPU only               →  HF   LoRA    (slow, emergency fallback)

The difference between LoRA and QLoRA in one line:
  LoRA  = train tiny adapter matrices on top of the FULL-precision frozen model
  QLoRA = train the same adapters on top of a QUANTISED (compressed) frozen model
          → saves 50-70% VRAM at ~2% quality cost

Usage:
    python train.py --train data/trading_train.jsonl --val data/trading_val.jsonl
    python train.py --train data/trading_train.jsonl --force_method qlora_4bit
    python train.py --train data/trading_train.jsonl --force_method mlx
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path


# ════════════════════════════════════════════════════════════════════
#  STEP 0 — Hardware detection
# ════════════════════════════════════════════════════════════════════
def detect_hardware() -> dict:
    """
    Detect available hardware and recommend the best training method.
    Returns a dict with platform, vram_gb, recommended_method.
    """
    import platform
    info = {"platform": platform.system(), "machine": platform.machine()}

    # ── Apple Silicon ────────────────────────────────────────────────
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import subprocess
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            ram_gb = int(raw) // (1024 ** 3)
        except Exception:
            ram_gb = 16
        info.update({
            "type": "apple_silicon",
            "unified_memory_gb": ram_gb,
            "recommended_method": "mlx_lora",
            "reason": f"Apple Silicon with {ram_gb}GB unified memory. MLX is the native framework."
        })
        return info

    # ── NVIDIA GPU ────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_name = torch.cuda.get_device_name(0)
            if vram >= 16:
                method = "lora"
                reason = f"{device_name} has {vram:.0f}GB VRAM — enough for LoRA in bfloat16 (no quantisation)"
            elif vram >= 8:
                method = "qlora_8bit"
                reason = f"{device_name} has {vram:.0f}GB VRAM — using 8-bit base + LoRA adapters"
            else:
                method = "qlora_4bit"
                reason = f"{device_name} has {vram:.0f}GB VRAM — using 4-bit NF4 base + LoRA adapters (QLoRA)"
            info.update({
                "type": "nvidia_gpu",
                "vram_gb": round(vram, 1),
                "device_name": device_name,
                "recommended_method": method,
                "reason": reason,
            })
            return info
    except ImportError:
        pass

    # ── CPU fallback ──────────────────────────────────────────────────
    info.update({
        "type": "cpu",
        "recommended_method": "lora",
        "reason": "CPU only — LoRA in fp32. Will be slow (hours). Recommend Colab A100.",
    })
    return info


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — Chat template application
# ════════════════════════════════════════════════════════════════════
def apply_chat_template_to_jsonl(jsonl_path: str, tokenizer, out_path: str) -> int:
    """
    Convert {"messages": [...]} JSONL to {"text": "full conversation string"} JSONL.
    This step is the same regardless of LoRA vs QLoRA.
    """
    count = 0
    with open(jsonl_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            ex = json.loads(line.strip())
            if not ex.get("messages"):
                continue
            # Apply the model's native chat template (handles <|im_start|> etc.)
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            fout.write(json.dumps({"text": text}) + "\n")
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════
#  PATH A — MLX LoRA (Apple Silicon)
# ════════════════════════════════════════════════════════════════════
def train_mlx_lora(cfg: dict) -> str:
    """
    LoRA fine-tuning using Apple's MLX framework.
    On M5 Air 16GB: ~5-8 hours for 600 iterations on 500 examples.

    What MLX LoRA does:
      1. Loads the model in 4-bit quantisation (MLX's own format — not bitsandbytes)
      2. Adds LoRA adapter matrices to the attention layers
      3. Trains ONLY the adapter matrices via gradient descent
      4. Base model weights stay frozen throughout
    """
    import subprocess
    try:
        import mlx_lm
    except ImportError:
        raise ImportError("pip install mlx-lm")

    print("\n[MLX LoRA] Apple Silicon path")
    print(f"  Base model:  {cfg['model']}")
    print(f"  LoRA rank:   {cfg['lora_r']}  (adapters have {cfg['lora_r']} dimensions)")
    print(f"  Iterations:  {cfg['iters']}")
    print(f"  Batch size:  {cfg['batch_size']}")
    print(f"  Checkpoint:  every {cfg['save_every']} steps")

    # ── Convert data to MLX format ────────────────────────────────────
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(cfg["model"])
    mlx_data_dir = Path(cfg["output_dir"]) / "mlx_data"
    mlx_data_dir.mkdir(parents=True, exist_ok=True)

    val_src = cfg.get("val") or cfg["train"]
    if not cfg.get("val"):
        print("  WARNING: --val not provided. Using training data for val/test.")
        print("  Eval loss will be misleadingly low. Pass --val data/trading_val.jsonl")
    for split, src_path in [("train", cfg["train"]), ("valid", val_src), ("test", val_src)]:
        src = src_path  # reassign for the loop body below
        n = apply_chat_template_to_jsonl(src, tokenizer, str(mlx_data_dir / f"{split}.jsonl"))
        print(f"  {split}: {n} examples formatted")

    # ── Run mlx_lm.lora ──────────────────────────────────────────────
    adapter_path = Path(cfg["output_dir"]) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",          cfg["model"],
        "--train",
        "--data",           str(mlx_data_dir),
        "--adapter-path",   str(adapter_path),
        "--num-layers",     str(cfg.get("lora_layers", 8)),
        "--batch-size",     str(cfg["batch_size"]),
        "--learning-rate",  str(cfg["learning_rate"]),
        "--iters",          str(cfg["iters"]),
        "--steps-per-eval", "50",
        "--save-every",     str(cfg["save_every"]),
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--grad-checkpoint",
    ]

    # Resume from checkpoint if one exists
    ckpts = sorted(adapter_path.glob("*_adapters.safetensors"))
    if ckpts:
        last_ckpt = ckpts[-1]
        cmd += ["--resume-adapter-file", str(last_ckpt)]
        print(f"  Resuming from: {last_ckpt.name}")

    print(f"\n  Running: {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, check=True)

    # ── Fuse adapter into base model (optional — for vLLM serving) ───
    fused = Path(cfg["output_dir"]) / "fused_model"
    print(f"\n  Fusing adapter → {fused}")
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        cfg["model"],
        "--adapter-path", str(adapter_path),
        "--save-path",    str(fused),
    ]
    try:
        subprocess.run(fuse_cmd, check=True)
        print(f"  Fused model ready for inference: {fused}")
    except Exception as e:
        print(f"  Fuse skipped: {e} (adapter still usable directly)")

    return str(adapter_path)


# ════════════════════════════════════════════════════════════════════
#  PATH B — HuggingFace LoRA (no quantisation, ≥ 16 GB VRAM)
# ════════════════════════════════════════════════════════════════════
def train_hf_lora(cfg: dict) -> str:
    """
    Standard LoRA — frozen model in bfloat16 + trainable LoRA adapters.

    What LoRA does (the math in plain English):
      Original:  output = W × input       (W = 4096×4096, 16M params, frozen)
      LoRA adds: output = W×input + (B×A)×input
                          where A = 4096×r, B = r×4096, r = lora_r (e.g. 16)
      We only train A and B — that's 2 × 4096 × 16 = 131K params per layer
      vs 16M in the original layer = 0.8% overhead

    Advantage over QLoRA: no quantisation noise — slightly better quality
    Requirement: ≥ 16 GB VRAM to hold the full bfloat16 model + adapters
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset
    import torch

    print("\n[HF LoRA] Standard LoRA (bfloat16 base, no quantisation)")
    _print_lora_diagram(cfg["lora_r"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_r"] * 2,   # alpha = 2× r is the standard rule of thumb
        target_modules=cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return _run_sft_trainer(model, tokenizer, cfg, quantised=False)


# ════════════════════════════════════════════════════════════════════
#  PATH C — HuggingFace QLoRA (quantised base, < 16 GB VRAM)
# ════════════════════════════════════════════════════════════════════
def train_hf_qlora(cfg: dict, bits: int = 4) -> str:
    """
    QLoRA — quantised base model + LoRA adapters.

    What quantisation does:
      Normal float32: each weight = 32 bits
      bfloat16:       each weight = 16 bits  (2× memory saving)
      int8:           each weight =  8 bits  (4× saving, slight quality loss)
      NF4 (4-bit):    each weight =  4 bits  (8× saving, ~2% quality loss)

    QLoRA pipeline:
      1. Load base model in 4-bit NF4 (Normal Float 4 — optimal for LLM weights)
      2. Dequantise to bfloat16 for the forward pass
      3. Compute gradients
      4. Update ONLY the LoRA adapter (still full precision)
      5. Re-quantise base model for storage

    Net effect: 8 GB saves 70% memory vs full LoRA on bfloat16
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    import torch

    print(f"\n[HF QLoRA] {bits}-bit quantised base + LoRA adapters")
    print(f"  This is QLoRA: the frozen base is compressed to {bits}-bit,")
    print(f"  saving ~{70 if bits==4 else 50}% VRAM vs full bfloat16.")
    _print_lora_diagram(cfg["lora_r"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True if bits == 4 else False,
        load_in_8bit=True if bits == 8 else False,
        bnb_4bit_quant_type="nf4",            # NF4 outperforms fp4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16, # dequantise to bf16 for compute
        bnb_4bit_use_double_quant=True,        # quantise the quantisation constants too
    ) if bits in (4, 8) else None

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Prepare for k-bit training: cast layernorms to fp32, enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_r"] * 2,
        target_modules=cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return _run_sft_trainer(model, tokenizer, cfg, quantised=True)


# ════════════════════════════════════════════════════════════════════
#  SHARED — SFTTrainer (used by both LoRA and QLoRA paths)
# ════════════════════════════════════════════════════════════════════
def _run_sft_trainer(model, tokenizer, cfg: dict, quantised: bool) -> str:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    import torch

    train_ds = load_dataset("json", data_files=cfg["train"],  split="train")
    val_ds   = load_dataset("json", data_files=cfg.get("val", cfg["train"]), split="train")

    # Format: apply chat template to each example
    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    train_ds = train_ds.map(fmt)
    val_ds   = val_ds.map(fmt)

    output_dir = cfg["output_dir"]
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=quantised,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=cfg["save_every"],
        eval_steps=50,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="adamw_bnb_8bit" if quantised else "adamw_torch",
        report_to=["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else ["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=False,         # False = each example is independent (correct for trading data)
        args=args,
    )

    print(f"\n  Train examples:   {len(train_ds)}")
    print(f"  Effective batch:  {cfg['batch_size'] * cfg['grad_accum']}")
    steps_per_epoch = math.ceil(len(train_ds) / (cfg['batch_size'] * cfg['grad_accum']))
    print(f"  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total steps:      {steps_per_epoch * cfg['epochs']}")
    print()

    trainer.train()

    # Save adapter
    final = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"\n  Adapter saved → {final}")
    return str(final)


# ════════════════════════════════════════════════════════════════════
#  DIAGRAM helper
# ════════════════════════════════════════════════════════════════════
def _print_lora_diagram(r: int):
    d = 4096  # typical hidden dim for 4B model
    original_params = d * d
    lora_params = 2 * d * r
    pct = lora_params / original_params * 100
    print(f"""
  LoRA diagram (rank r={r}, hidden dim {d}):
  ┌──────────────────────────────────────────────────────┐
  │  Original: output = W × input                        │
  │            W = {d}×{d} = {original_params:,} params  [FROZEN]          │
  │                                                      │
  │  LoRA adds: output = W×input  +  B×A×input          │
  │             A = {d}×{r}  ({d*r:,} params)  [TRAINABLE]         │
  │             B = {r}×{d}  ({r*d:,} params)  [TRAINABLE]         │
  │             Total LoRA params: {lora_params:,} ({pct:.2f}% of layer)  │
  └──────────────────────────────────────────────────────┘""")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--train",          required=True,  help="Training JSONL path")
    p.add_argument("--val",            default=None,   help="Validation JSONL path")
    p.add_argument("--model",          default="nvidia/Nemotron-Nano-4B-Instruct")
    p.add_argument("--output_dir",     default="output/nemotron-trading")
    p.add_argument("--lora_r",         type=int,   default=16,    help="LoRA rank (higher = more capacity)")
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--learning_rate",  type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int,   default=1024)
    p.add_argument("--iters",          type=int,   default=600,   help="MLX only: training iterations")
    p.add_argument("--save_every",     type=int,   default=100,   help="Save checkpoint every N steps")
    p.add_argument("--force_method",   default=None,
                   choices=["mlx_lora","lora","qlora_8bit","qlora_4bit"],
                   help="Force a specific training method instead of auto-detecting")
    args = p.parse_args()

    cfg = vars(args)

    # ── Read params.yaml (override CLI defaults with central config) ──────────
    params_file = Path("params.yaml")
    if params_file.exists():
        try:
            import yaml
            p_cfg = yaml.safe_load(params_file.read_text())
            # Only override if user didn't explicitly pass the flag
            if args.lora_r == 16:          cfg["lora_r"]         = p_cfg.get("lora_r", 16)
            if args.epochs == 3:           cfg["epochs"]         = p_cfg.get("num_epochs", 3)
            if args.learning_rate == 2e-4: cfg["learning_rate"]  = p_cfg.get("learning_rate", 2e-4)
            if args.batch_size == 2:       cfg["batch_size"]     = p_cfg.get("batch_size", 2)
            if args.grad_accum == 8:       cfg["grad_accum"]     = p_cfg.get("grad_accum", 8)
            if args.max_seq_length == 1024:cfg["max_seq_length"] = p_cfg.get("max_seq_length", 1024)
            if args.iters == 600:          cfg["iters"]          = p_cfg.get("mlx_iters", 600)
            if args.save_every == 100:     cfg["save_every"]     = p_cfg.get("save_every", 100)
            cfg["target_modules"] = p_cfg.get("lora_target_modules", [
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
            cfg["lora_layers"] = p_cfg.get("mlx_lora_layers", 8)
            print(f"  Loaded hyperparameters from params.yaml")
        except ImportError:
            print("  PyYAML not installed (pip install pyyaml) — using CLI defaults")
        except Exception as e:
            print(f"  Could not read params.yaml: {e} — using CLI defaults")
    else:
        cfg["target_modules"] = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ]
        cfg["lora_layers"] = 8

    # ── Detect hardware ───────────────────────────────────────────────
    hw = detect_hardware()
    method = args.force_method or hw["recommended_method"]

    print("=" * 60)
    print("  TRADING AGENT — LLM Fine-tuning")
    print("=" * 60)
    print(f"  Hardware:  {hw['type'].replace('_',' ').title()}")
    print(f"  Method:    {method}")
    print(f"  Reason:    {hw.get('reason', '')}")
    print(f"  Model:     {cfg['model']}")
    print(f"  LoRA rank: {cfg['lora_r']}")
    print(f"  Output:    {cfg['output_dir']}")
    print("=" * 60)

    # ── MLflow run ────────────────────────────────────────────────────
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_experiment("trading-agent-finetune")
        mlflow.start_run(run_name=f"{method}-r{cfg['lora_r']}")
        mlflow.log_params({
            "method": method, "model": cfg["model"],
            "lora_r": cfg["lora_r"], "epochs": cfg.get("epochs", "—"),
            "batch_size": cfg["batch_size"], "learning_rate": cfg["learning_rate"],
            "hardware": hw["type"], "vram_gb": hw.get("vram_gb", hw.get("unified_memory_gb","—")),
        })
        mlflow_active = True
        print("  MLflow tracking active (mlflow ui to view)")
    except Exception:
        print("  MLflow not available — skipping tracking")

    # ── Train ─────────────────────────────────────────────────────────
    start = time.time()
    try:
        if method == "mlx_lora":
            adapter = train_mlx_lora(cfg)
        elif method == "lora":
            adapter = train_hf_lora(cfg)
        elif method == "qlora_8bit":
            adapter = train_hf_qlora(cfg, bits=8)
        elif method == "qlora_4bit":
            adapter = train_hf_qlora(cfg, bits=4)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed/3600:.1f}h")
        print(f"  Adapter: {adapter}")
        print(f"{'='*60}")

        if mlflow_active:
            import mlflow
            mlflow.log_metrics({"training_time_h": elapsed/3600})
            mlflow.log_param("adapter_path", adapter)
            mlflow.end_run()

    except Exception as e:
        if mlflow_active:
            import mlflow
            mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
