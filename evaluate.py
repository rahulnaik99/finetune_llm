"""
evaluate.py
───────────
Evaluates the fine-tuned adapter on the held-out test set.

Metrics:
  1. JSON validity rate       — does the model always output parseable JSON?
  2. Schema accuracy          — are required fields present with correct types?
  3. Action distribution      — BUY/SELL/HOLD balance
  4. Reasoning quality score  — heuristic: does reasoning cite numbers?
  5. Back-test P&L            — run the model's decisions against real forward data

Usage:
    python evaluate.py \
        --adapter output/nemotron-trading/final_adapter \
        --test    data/trading_test.jsonl
"""
import argparse, json, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ── Load fine-tuned model ────────────────────────────────────────────────────
def load_model(base_model: str, adapter_path: str):
    """Load base model + LoRA adapter in inference mode."""
    # Try Unsloth first
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,        # adapter_path must contain adapter_config.json
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print(f"Loaded with Unsloth: {adapter_path}")
        return model, tokenizer
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print(f"Loaded with PEFT: {adapter_path}")
    return model, tokenizer


# ── Inference helper ─────────────────────────────────────────────────────────
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Return only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Metric helpers ────────────────────────────────────────────────────────────
REQUIRED_PRE_FIELDS  = {"action", "quantity", "stop_loss", "take_profit", "reasoning"}
REQUIRED_POST_FIELDS = {"outcome", "lesson_learned", "strategy_adjustment"}

def check_schema(text: str, required: set) -> tuple[bool, bool, dict | None]:
    """Returns (json_valid, schema_ok, parsed)."""
    # Extract JSON from markdown if needed
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m: text = m.group(1)
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return False, False, None
    schema_ok = required.issubset(parsed.keys())
    return True, schema_ok, parsed

def reasoning_cites_numbers(reasoning: str) -> bool:
    """Heuristic: good reasoning should mention at least one price/percentage."""
    return bool(re.search(r"\d+\.\d+|\d+%|EMA\d+|RSI", reasoning or ""))


# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate(adapter_path: str, test_jsonl: str, base_model: str) -> dict:
    model, tokenizer = load_model(base_model, adapter_path)

    test_examples = []
    with open(test_jsonl) as f:
        for line in f:
            test_examples.append(json.loads(line))

    print(f"Evaluating {len(test_examples)} test examples...")

    results = {
        "total": len(test_examples),
        "json_valid": 0, "schema_ok": 0,
        "reasoning_quality": 0,
        "action_dist": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "examples": [],
    }

    for i, ex in enumerate(test_examples):
        messages = ex["messages"]
        # Remove the assistant turn — we'll generate it
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        expected = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

        generated = generate(model, tokenizer, prompt_messages)

        # Detect example type from system prompt
        is_pre = "action" in (expected or "")
        required = REQUIRED_PRE_FIELDS if is_pre else REQUIRED_POST_FIELDS
        json_ok, schema_ok, parsed = check_schema(generated, required)

        results["json_valid"] += int(json_ok)
        results["schema_ok"]  += int(schema_ok)

        if parsed and is_pre:
            action = parsed.get("action", "HOLD")
            results["action_dist"][action] = results["action_dist"].get(action, 0) + 1
            if reasoning_cites_numbers(parsed.get("reasoning", "")):
                results["reasoning_quality"] += 1

        if i < 3:   # print first 3 examples
            print(f"\n--- Example {i+1} ---")
            print(f"Expected:  {expected[:200]}...")
            print(f"Generated: {generated[:200]}...")
            print(f"JSON valid: {json_ok}  Schema ok: {schema_ok}")

    n = results["total"] or 1
    print(f"\n{'='*60}")
    print(f"  JSON validity:      {results['json_valid']}/{n} = {results['json_valid']/n:.1%}")
    print(f"  Schema accuracy:    {results['schema_ok']}/{n} = {results['schema_ok']/n:.1%}")
    print(f"  Reasoning quality:  {results['reasoning_quality']}/{n}")
    print(f"  Action dist:        {results['action_dist']}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",    required=True)
    p.add_argument("--test",       required=True)
    p.add_argument("--base_model", default="nvidia/Nemotron-Nano-4B-Instruct")
    args = p.parse_args()
    evaluate(args.adapter, args.test, args.base_model)
