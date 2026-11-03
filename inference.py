"""
inference.py
────────────
Drop-in replacement for the Ollama _ollama() call in the simulation notebook.
Loads the fine-tuned Nemotron-Nano adapter and generates decisions locally,
no Ollama server needed.

Usage in the notebook:
    from inference import FinetunedTrader
    trader = FinetunedTrader("output/nemotron-trading/final_adapter")
    # Then set FALLBACK_MODE = False and monkey-patch _ollama:
    import inference as _inf_module
    _inf_module._TRADER = trader
"""
import json, re, torch
from pathlib import Path


_TRADER = None   # singleton, loaded once


def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m: text = m.group(1)
    try:
        return json.loads(text.strip())
    except Exception:
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try: return json.loads(m2.group())
            except: pass
    return {}


class FinetunedTrader:
    """Wraps the fine-tuned adapter for single-call inference."""

    def __init__(self, adapter_path: str, base_model: str = "nvidia/Nemotron-Nano-4B-Instruct",
                 max_new_tokens: int = 512, temperature: float = 0.1):
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        print(f"Loading fine-tuned model from {adapter_path}...")
        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                adapter_path, max_seq_length=2048, dtype=None, load_in_4bit=True)
            FastLanguageModel.for_inference(self.model)
            print("  Loaded via Unsloth")
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.model.eval()
            print("  Loaded via PEFT")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def decide(self, system: str, user: str) -> dict:
        """Generate and parse trading decision JSON."""
        raw = self.generate(system, user)
        parsed = _parse_json(raw)
        parsed["_raw"] = raw
        return parsed


def get_trader(adapter_path: str = "output/nemotron-trading/final_adapter") -> FinetunedTrader:
    """Singleton — loads the model once."""
    global _TRADER
    if _TRADER is None:
        _TRADER = FinetunedTrader(adapter_path)
    return _TRADER
