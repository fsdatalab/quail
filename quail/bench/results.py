"""Write Modal run metadata."""

import json
from pathlib import Path


def write_json(path, data):
    """Replace a JSON file only after its contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)
