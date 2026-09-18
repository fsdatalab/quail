"""Read benchmark files with Arrow's local and S3 filesystems."""

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pyarrow import fs

PUBLIC_BUCKET = "quail-bench"
GROUND_TRUTH_ROOT = "ground_truth/quailb/schema_v1"
_cache_dir = ContextVar("quail_b_cache_dir", default=None)


@contextmanager
def download_cache(directory=None):
    """Choose the download cache for the current benchmark operation."""
    token = _cache_dir.set(directory)
    try:
        yield
    finally:
        _cache_dir.reset(token)


def _cached_file(root, path):
    """Cache immutable published files; always refresh active collection pointers."""
    if root is not None and not str(root).startswith("s3://"):
        return None
    if Path(path).name.startswith("active_collection."):
        return None
    filesystem, _, source = _location(root, path)
    directory = _cache_dir.get() or os.environ.get("QUAIL_B_CACHE_DIR")
    if directory is None:
        directory = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        directory = directory / "quail-b"
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(source.encode()).hexdigest()
    cached = directory / key
    if not cached.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with filesystem.open_input_file(source) as remote:
                    shutil.copyfileobj(remote, stream)
                stream.flush()
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)
    return cached


def _location(root, path):
    root = f"s3://{PUBLIC_BUCKET}" if root is None else str(root)
    if root.startswith("s3://"):
        filesystem = fs.S3FileSystem(anonymous=True)
        base = root.removeprefix("s3://").rstrip("/")
    else:
        filesystem = fs.LocalFileSystem()
        base = Path(root).expanduser().resolve().as_posix()
    return filesystem, base, f"{base}/{path.lstrip('/')}"


def _read_bytes(root, path):
    cached = _cached_file(root, path)
    if cached is not None:
        return cached.read_bytes()
    filesystem, _, source = _location(root, path)
    with filesystem.open_input_file(source) as stream:
        return stream.read()


def _list_files(root, path):
    filesystem, base, source = _location(root, path)
    selector = fs.FileSelector(source, recursive=True, allow_not_found=True)
    return sorted(
        info.path.removeprefix(f"{base}/")
        for info in filesystem.get_file_info(selector)
        if info.type == fs.FileType.File
        and info.path.endswith((".json", ".parquet"))
    )
