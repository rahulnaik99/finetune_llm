"""
common/_bootstrap.py
────────────────────
Single source of truth for `sys.path` wiring.

The pipeline was refactored so every stage lives in its own folder
(`modules/<stage>/`). The heavy, domain-specific modules (``simulation``,
``data_pipeline``, ``evaluate``, ``train`` …) still import each other by their
*bare* module name exactly as they did before the restructure. To keep that
working — i.e. to restructure the project **without breaking it** — we add each
stage folder (and the shared ``common`` folder) to ``sys.path``.

The thin orchestration layer (``service.py`` / ``api.py`` inside every stage)
is always imported with its fully-qualified package path
(``modules.<stage>.api``) and therefore never collides across folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Order matters only for human readability; names are unique across folders.
_STAGE_DIRS = [
    ROOT,                              # project root → `import common`, `import modules`
    ROOT / "common",                  # `from config import ...`, `from state import ...`
    ROOT / "modules" / "datagen",     # simulation, zones, compact_prompt, fundamentals_xbrl…
    ROOT / "modules" / "splitter",    # data_pipeline
    ROOT / "modules" / "training",    # train, train_mlx, train_qlora
    ROOT / "modules" / "evaluation",  # evaluate
    ROOT / "modules" / "orchestrator",# flywheel
    ROOT / "modules" / "monitoring",  # monitoring, deploy_to_ollama
    ROOT / "scripts",
]


def setup_paths() -> None:
    """Idempotently add every stage folder to ``sys.path``."""
    for d in _STAGE_DIRS:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


setup_paths()
