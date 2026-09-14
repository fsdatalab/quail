"""The pip requirement of quail-b for Modal images, from the pyproject pin."""

import tomllib
from pathlib import Path


def quail_b_requirement() -> str:
    """Return the pip requirement of quail-b at the commit pyproject.toml pins."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        # inside a container the module is imported from the mounted
        # package, with no pyproject.toml; the image is already built
        return "quail-b"
    source = tomllib.loads(pyproject.read_text())["tool"]["uv"]["sources"]
    return f"quail-b @ git+{source['quail-b']['git']}@{source['quail-b']['rev']}"
