"""Add python/ (this dir's parent) to sys.path so tests can import nets_v2 etc."""
import sys
import pathlib

_PYTHON_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
