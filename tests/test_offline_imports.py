"""
tests/test_offline_imports.py
─────────────────────────────
Regression guard for a bug that bit a user: the simulation module must import
and run its OFFLINE / rule-based path on a machine that has neither ``requests``
(only needed for the Ollama teacher) nor ``yfinance`` (only needed for online
prices/fundamentals) installed.

We run in a subprocess with those two packages hard-blocked, so a future bare
top-level ``import requests`` / ``import yfinance`` would fail this test even on
a CI box where both happen to be installed.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SNIPPET = """
import sys, builtins
_real = builtins.__import__
def _blk(name, *a, **k):
    if name.split('.')[0] in ("requests", "yfinance"):
        raise ModuleNotFoundError("blocked: " + name)
    return _real(name, *a, **k)
for m in ("requests", "yfinance"):
    sys.modules.pop(m, None)
builtins.__import__ = _blk
sys.path.insert(0, __ROOT__)
import common            # path bootstrap
import simulation        # must NOT require requests/yfinance at import time
assert simulation.requests is None
assert simulation.yf is None
assert hasattr(simulation, "_require_requests")
assert hasattr(simulation, "_require_yf")
from simulation import set_llm_backend   # offline backend selectable
print("OFFLINE_OK")
"""


def test_simulation_imports_without_requests_or_yfinance():
    code = _SNIPPET.replace("__ROOT__", repr(str(ROOT)))
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert "OFFLINE_OK" in proc.stdout, (
        "simulation failed to import offline:\n" + proc.stdout + proc.stderr)
