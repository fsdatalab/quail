"""Put the repo root on sys.path so `import quail` resolves from a
source checkout under plain `pytest` as well as `python -m pytest`,
with no install step."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
