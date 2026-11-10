"""
common/params.py
────────────────
Single accessor for ``params.yaml`` so every stage resolves hyper-parameters
from ONE place. Values here are the authoritative defaults; an explicit CLI/API
value (i.e. not ``None``) still overrides. Falls back to an empty dict if PyYAML
or the file is missing, so callers always get a dict.
"""
from __future__ import annotations

from functools import lru_cache

from .paths import PROJECT_ROOT


@lru_cache(maxsize=1)
def load_params() -> dict:
    f = PROJECT_ROOT / "params.yaml"
    if not f.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(f.read_text()) or {}
    except Exception:
        return {}


def param(key: str, default=None):
    return load_params().get(key, default)
