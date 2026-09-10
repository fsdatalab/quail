"""Read benchmark files with Arrow's local and S3 filesystems."""

from pathlib import Path

from pyarrow import fs

PUBLIC_BUCKET = "quail-bench"
GROUND_TRUTH_ROOT = "ground_truth/quailb/schema_v1"


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
