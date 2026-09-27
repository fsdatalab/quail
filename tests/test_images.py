"""The Modal image settings against the repository's own pins."""

import tomllib
from pathlib import Path

from quail.bench import images


def test_image_uv_version_is_the_required_one():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        required = tomllib.load(f)["tool"]["uv"]["required-version"]
    assert required == f"=={images.UV_VERSION}"
