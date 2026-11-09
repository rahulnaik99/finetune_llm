"""
inference_mlx.py
────────────────
Drop-in replacement for _ollama() using a locally fine-tuned MLX model.
Much faster than Ollama on Apple Silicon — uses the Neural Engine directly.

Usage in the simulation notebook:
    # Replace the OLLAMA_URL call with this:
    from inference_mlx import generate_mlx

    def _ollama(system, user, temp=0.2):
        return generate_mlx(
            model_path="output/nemotron-mlx/fused_model",
            system=system, user=user
        )
"""
import json, re

_MODEL = None
_TOKENIZER = None


def _load(model_path: str, adapter_path: str | None = None):
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from mlx_lm import load
        print(f"Loading MLX model: {model_path}")
        _MODEL, _TOKENIZER = load(model_path, adapter_path=adapter_path)
        print("Model loaded — ready for inference")
    return _MODEL, _TOKENIZER


def generate_mlx(
    model_path: str,
    system: str,
    user: str,
    adapter_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    from mlx_lm import generate

    model, tokenizer = _load(model_path, adapter_path)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    return response
