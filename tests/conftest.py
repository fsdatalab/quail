"""Put the repo root on sys.path so both import styles in the test
files work under plain `pytest` as well as `python -m pytest`:
bare imports of sibling test modules (`from test_exact import ...`)
and package-style imports (`from tests.test_client import ...`)."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
