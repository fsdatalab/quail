"""Modal volumes shared by every worker function, no-ops outside Modal."""

import os
import time

import modal

hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

RESULTS_ROOT = "/results"


def commit_results() -> None:
    """Persist the results volume when running inside Modal."""
    if not modal.is_local():
        results_vol.commit()


def commit_kernel_cache() -> None:
    """Persist the kernel cache volume when running inside Modal."""
    if not modal.is_local():
        kernel_cache.commit()


def run_record_path() -> str | None:
    """Return a fresh path under the results volume, or None without one."""
    if not (os.path.isdir(RESULTS_ROOT) and os.access(RESULTS_ROOT, os.W_OK)):
        return None
    runs = os.path.join(RESULTS_ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    return os.path.join(runs, f"run_{time.time_ns()}.json")
