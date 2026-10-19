"""Publish a file only once it is complete."""

import os
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def published(path: Path, *, overwrite: bool = True):
    """Yield a private path to write; move it to path on success, else remove it.

    Args:
        path: The destination, in a directory that exists.
        overwrite: Whether to replace a file already at path. When
            False, a file already there raises FileExistsError.
    """
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
