"""Publish complete files from private temporary paths."""

import os
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path, *, overwrite: bool = True):
    """Publish a temporary file on success and remove it on failure.

    Args:
        path: The destination, whose parent directory must exist.
        overwrite: Replace an existing destination. If False, publication
            raises FileExistsError when the destination already exists.
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
