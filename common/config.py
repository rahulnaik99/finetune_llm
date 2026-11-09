"""
config.py — central .env loader for the whole pipeline.

Loads key=value pairs from a .env file into environment variables so you can
configure everything (model, episodes, rounds, iterations, backend) without
editing any code. Works even if python-dotenv isn't installed (tiny fallback
parser). Import and call load_env() once at program start.

Usage:
    from config import load_env, env_str, env_int, env_bool
    load_env()
    MODEL = env_str("BASE_MODEL", "mlx-community/...")
    ROUNDS = env_int("ROUNDS", 5)
"""
from __future__ import annotations
import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """Load .env into os.environ. Existing env vars take precedence (setdefault)."""
    # Prefer python-dotenv if present
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except Exception:
        pass
    # Minimal fallback parser — no dependency needed
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split("#", 1)[0].strip().strip('"').strip("'")  # strip inline comments + quotes
        if key:
            os.environ.setdefault(key, val)


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")
