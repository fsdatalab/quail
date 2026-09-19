"""The pip requirements of quail and quail-b for Modal images.

The images mount the quail source instead of installing the package,
so they install quail's declared dependencies themselves, read from
the installed package metadata that pyproject.toml defines.
"""

import json
from importlib.metadata import distribution


def quail_requirements(without: tuple[str, ...] = ()) -> list[str]:
    """Return quail's pinned runtime requirements for a Linux image.

    Args:
        without: Distribution names to leave out, for an image that
            installs its own build of one (the SGLang image and vLLM).
    """
    from packaging.requirements import Requirement

    out = []
    for text in distribution("quail-engine").requires or ():
        req = Requirement(text)
        if req.name in without:
            continue
        if req.marker is not None and not req.marker.evaluate(
                {"sys_platform": "linux"}):
            continue
        out.append(f"{req.name}{req.specifier}")
    return out


def quail_b_requirement() -> str:
    """Return the pip requirement of the installed quail-b, at its exact commit.

    An installation from git records its origin (PEP 610), so the image
    installs the commit the environment runs, whatever pinned it.
    """
    quail_b = distribution("quail-b")
    origin = quail_b.read_text("direct_url.json")
    if origin is None:
        return f"quail-b=={quail_b.version}"
    origin = json.loads(origin)
    return f"quail-b @ git+{origin['url']}@{origin['vcs_info']['commit_id']}"
