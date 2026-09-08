"""Modal volumes shared by every worker function, no-ops outside Modal."""

import io
import json
import os
import time

import modal
import pyarrow as pa
from pyarrow import parquet as pq

hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

RESULTS_ROOT = "/results"
KERNEL_CACHE_ROOT = "/root/.cache/kernels"


def _mounted(root: str) -> bool:
    """Whether a volume is attached here: inside Modal, at its mount point."""
    return not modal.is_local() and os.path.isdir(root)


def commit_results() -> None:
    """Persist the results volume when it is attached to this container."""
    if _mounted(RESULTS_ROOT):
        results_vol.commit()


def commit_kernel_cache() -> None:
    """Persist the kernel cache volume when it is attached to this container."""
    if _mounted(KERNEL_CACHE_ROOT):
        kernel_cache.commit()


def run_record_path() -> str | None:
    """Return a fresh path under the results volume, or None without one."""
    if not (os.path.isdir(RESULTS_ROOT) and os.access(RESULTS_ROOT, os.W_OK)):
        return None
    runs = os.path.join(RESULTS_ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    return os.path.join(runs, f"run_{time.time_ns()}.json")


class ModalVolumeFiles:
    """Read and write the results volume through the Modal API.

    Works from any machine with Modal credentials, mounted or not. Has
    the same methods as the benchmark's file stores, so the runner can
    hand it to the label loader and the run-record writer.
    """

    def __init__(self, volume=results_vol):
        self.volume = volume

    def read_bytes(self, path: str) -> bytes:
        return b"".join(self.volume.read_file(path.lstrip("/")))

    def list_files(self, path: str) -> list[str]:
        return sorted(
            entry.path for entry in self.volume.listdir(
                path.lstrip("/"), recursive=True)
            if entry.path.endswith((".json", ".parquet")))

    def write_json(self, path: str, payload: dict) -> None:
        data = io.BytesIO(json.dumps(
            payload, indent=2, sort_keys=True).encode("utf-8"))
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(data, path.lstrip("/"))

    def write_parquet(self, path: str, table: pa.Table) -> None:
        data = io.BytesIO()
        pq.write_table(table, data, compression="zstd")
        data.seek(0)
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(data, path.lstrip("/"))
