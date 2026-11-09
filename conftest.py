"""
conftest.py
───────────
Pytest bootstrap. Importing ``common`` wires every stage folder onto sys.path so
the heavy domain modules (``simulation``, ``data_pipeline``, ``evaluate`` …)
remain importable by their bare names after the restructure.
"""
import common  # noqa: F401  (side-effect: common._bootstrap.setup_paths)
