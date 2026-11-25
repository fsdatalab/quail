"""Put the project root on sys.path so tests run without an install."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DataFusion's physical plan, which explain() shows, splits rows into one
# partition per CPU by default
os.environ.setdefault("DATAFUSION_EXECUTION_TARGET_PARTITIONS", "4")
