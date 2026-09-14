"""The pip requirement of quail-b for Modal images."""

import json
from importlib.metadata import distribution


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
