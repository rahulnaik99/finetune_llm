"""
inference_vllm.py
─────────────────
Production LLM inference using vLLM — 10-50× faster than Ollama.
vLLM uses PagedAttention and continuous batching — it handles 100s of
concurrent requests without queuing (unlike Ollama which serialises them).

Start the server:
    vllm serve output/nemotron-trading/merged \
        --port 8001 \
        --served-model-name nemotron-trading \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.85

Then set in model_config.json:
    "provider": "vllm"
    (and VLLM_BASE_URL=http://localhost:8001 in .env)

The stock scoring app's llm_factory.py already supports provider=vllm.
This file provides a standalone client for the simulation notebook.
"""
import json, re
from typing import Optional

_client = None   # openai.AsyncOpenAI singleton


def _get_client(base_url: str = "http://localhost:8001/v1"):
    global _client
    if _client is None:
        try:
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
            print(f"vLLM client ready → {base_url}")
        except ImportError:
            raise ImportError("pip install openai")
    return _client


async def generate_vllm(
    system: str,
    user: str,
    model: str = "nemotron-trading",
    base_url: str = "http://localhost:8001/v1",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Async generation via vLLM OpenAI-compatible API."""
    client = _get_client(base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_vllm_sync(system: str, user: str, **kwargs) -> str:
    """Synchronous wrapper — use in the simulation notebook."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(generate_vllm(system, user, **kwargs))
    except RuntimeError:
        return asyncio.run(generate_vllm(system, user, **kwargs))


# ── Batch inference (vLLM's main advantage over Ollama) ──────────────────────
async def batch_generate(
    prompts: list[tuple[str, str]],   # list of (system, user) tuples
    model: str = "nemotron-trading",
    base_url: str = "http://localhost:8001/v1",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> list[str]:
    """
    Send ALL prompts to vLLM simultaneously.
    vLLM batches them internally via continuous batching — massive throughput.
    Ollama would queue them sequentially (10 stocks × 5 agents = 50 serial calls).
    vLLM handles all 50 concurrently.
    """
    import asyncio
    client = _get_client(base_url)
    tasks = [
        client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":s}, {"role":"user","content":u}],
            temperature=temperature, max_tokens=max_tokens,
        )
        for s, u in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]


if __name__ == "__main__":
    import asyncio
    print("Testing vLLM batch inference...")
    prompts = [
        ("You are a trader. Output JSON only.", f"CANBK #{i}: BUY or SELL?")
        for i in range(5)
    ]
    try:
        results = asyncio.run(batch_generate(prompts))
        for i, r in enumerate(results):
            print(f"  Prompt {i+1}: {r[:80]}...")
        print("vLLM batch inference OK")
    except Exception as e:
        print(f"vLLM not running: {e}")
        print("Start with: vllm serve <model> --port 8001")
