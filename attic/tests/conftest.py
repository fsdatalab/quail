"""Put the repo root on sys.path so the attic tests can import both
attic.theory (never installed) and docengine (when not pip-installed),
under plain `pytest` as well as `python -m pytest`."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
