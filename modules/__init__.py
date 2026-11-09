"""Pipeline stages. Importing this package wires up sys.path for the bare-name
imports used by the heavy domain modules (see common/_bootstrap.py)."""
from common._bootstrap import setup_paths

setup_paths()
