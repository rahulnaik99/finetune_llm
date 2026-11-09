"""
train_qlora.py
──────────────
QLoRA fine-tuning of nvidia/Nemotron-Nano-4B-Instruct on trading data.
Uses Unsloth (2× faster LoRA) with SFTTrainer from TRL.

Requirements:
    pip install unsloth transformers trl peft bitsandbytes datasets accelerate wandb

Run:
    # Single GPU (A10G / A100 / RTX 3090)
    python train_qlora.py --train data/trading_train.jsonl --val data/trading_val.jsonl

    # Free Colab T4 (slower but works):
    python train_qlora.py --train data/trading_train.jsonl --max_seq_length 1024 --batch_size 1

Memory guide:
    4-bit QLoRA:  ~5 GB VRAM   (T4, RTX 3060)
    8-bit LoRA:   ~9 GB VRAM   (RTX 3080, A10G)
    bf16 full:    ~18 GB VRAM  (A100 40GB)
"""
import argparse, json, os, math
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset, load_dataset


# ── config dataclass ─────────────────────────────────────────────────────────
class TrainConfig:
    # Model
    model_name:    str   = "nvidia/Nemotron-Nano-4B-Instruct"
    load_in_4bit:  bool  = True         # QLoRA (4-bit NF4) — set False for full 16-bit

    # LoRA
    lora_r:        int   = 16           # rank (higher = more capacity, more memory)
    lora_alpha:    int   = 32           # scaling factor (usually 2× rank)
    lora_dropout:  float = 0.05
    target_modules: list = [            # which projection matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Training
    max_seq_length:   int   = 2048
    batch_size:       int   = 2
    grad_accum:       int   = 8         # effective batch = batch_size × grad_accum = 16
    learning_rate:    float = 2e-4
    num_epochs:       int   = 3
    warmup_ratio:     float = 0.05
    weight_decay:     float = 0.01
    lr_scheduler:     str   = "cosine"
    fp16:             bool  = not torch.cuda.is_bf16_supported()
    bf16:             bool  = torch.cuda.is_bf16_supported()

    # Output
    output_dir:    str   = "output/nemotron-trading"
    run_name:      str   = f"trading-qlora-{datetime.now().strftime('%m%d-%H%M')}"
    save_steps:    int   = 50
    eval_steps:    int   = 50
    logging_steps: int   = 10


def load_chat_dataset(jsonl_path: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load JSONL → apply chat template → tokenise → return Dataset."""
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            # Apply the model's chat template to messages
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append({"text": text})

    dataset = Dataset.from_list(examples)
    print(f"  Loaded {len(dataset)} examples from {jsonl_path}")
    return dataset


def train(cfg: TrainConfig, train_path: str, val_path: str | None = None):
    # ── Try Unsloth first (2× faster, less memory) ────────────────────────────
    try:
        from unsloth import FastLanguageModel
        print("Using Unsloth for faster training")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=None,         # auto-detect
            load_in_4bit=cfg.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=cfg.target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",   # saves 30% VRAM
            random_state=42,
        )
        USE_UNSLOTH = True

    except ImportError:
        # ── Fall back to PEFT + bitsandbytes ─────────────────────────────────
        print("Unsloth not found → using PEFT + bitsandbytes")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",          # NF4 outperforms fp4 on LLMs
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,     # nested quant saves ~0.4 bits/param
        ) if cfg.load_in_4bit else None

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        USE_UNSLOTH = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = load_chat_dataset(train_path, tokenizer, cfg.max_seq_length)
    val_ds   = load_chat_dataset(val_path, tokenizer, cfg.max_seq_length) if val_path else None

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=not USE_UNSLOTH,     # Unsloth handles this itself
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if val_ds else None,
        evaluation_strategy="steps" if val_ds else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(val_ds),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["tensorboard"],
        optim="adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        seed=42,
    )

    # Only train on assistant turns (not system/user)
    # This is the key trick: we mask out the non-assistant tokens in the loss
    response_template = "<|assistant|>"   # Nemotron ChatML token
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )
    except Exception:
        collator = None   # fallback: train on full sequence

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        data_collator=collator,
        dataset_num_proc=2,
        packing=False,              # set True for short examples (< 512 tokens)
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training: {cfg.run_name}")
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Effective batch: {cfg.batch_size * cfg.grad_accum}")
    print(f"  Steps per epoch: {math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))}")

    trainer.train()

    # ── Save adapter weights ───────────────────────────────────────────────────
    adapter_path = Path(cfg.output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ Adapter saved → {adapter_path}")
    print("  To merge with base: see merge_adapter.py")
    print("  To use in inference: see inference.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",          required=True,  help="Path to training JSONL")
    p.add_argument("--val",            default=None,   help="Path to validation JSONL")
    p.add_argument("--model",          default="nvidia/Nemotron-Nano-4B-Instruct")
    p.add_argument("--output_dir",     default="output/nemotron-trading")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size",     type=int, default=2)
    p.add_argument("--epochs",         type=int, default=3)
    p.add_argument("--lora_r",         type=int, default=16)
    p.add_argument("--load_in_4bit",   action="store_true", default=True)
    args = p.parse_args()

    cfg = TrainConfig()
    cfg.model_name    = args.model
    cfg.output_dir    = args.output_dir
    cfg.max_seq_length= args.max_seq_length
    cfg.batch_size    = args.batch_size
    cfg.num_epochs    = args.epochs
    cfg.lora_r        = args.lora_r
    cfg.load_in_4bit  = args.load_in_4bit

    train(cfg, train_path=args.train, val_path=args.val)
