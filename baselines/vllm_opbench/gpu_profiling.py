"""Nsight Systems (nsys) profiling for the vLLM-opbench worker.

Ported near-verbatim from /Users/adhariya/SQPE/gpu_profiling.py - this
module is entirely generic (no SQPE-specific coupling beyond its
config imports), so the port is a straight copy with vllm_opbench's
own config module.

Runs the Modal container runtime under `nsys profile` and uses CUDA
profiler API capture ranges so that ONLY a bounded, steady-state
window of one generate_batch() call is recorded - not the whole
container lifetime, and not the cold-start ramp-up of the batch
itself.

Important Modal-specific detail: the re-exec below launches

    python -m modal._container_entrypoint

NOT the file path directly - the file path form inserts /pkg/modal at
the front of sys.path, which makes modal/types.py shadow the standard
library's types.py and breaks Modal startup.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .config import (
    NSYS_CAPTURE_WARMUP_S,
    NSYS_CAPTURE_WINDOW_S,
    NSYS_OUTPUT_DIR,
    NSYS_REEXEC_SENTINEL,
    NSYS_TRACE_FLAGS,
)


def nsys_available() -> bool:
    return shutil.which("nsys") is not None


def maybe_reexec_under_nsys() -> None:
    """Re-launch the Modal container runtime under Nsight Systems.
    Call as the FIRST operation inside @modal.enter(), before anything
    CUDA-related happens. No-ops when PROFILE_GPU is False, when nsys
    isn't installed, or when already running under nsys."""
    from .config import PROFILE_GPU

    if not PROFILE_GPU:
        return
    if os.environ.get(NSYS_REEXEC_SENTINEL):
        return
    if not nsys_available():
        print("[gpu_profiling] PROFILE_GPU=True but nsys was not found in "
              "the container image; continuing without GPU profiling.",
              flush=True)
        return

    os.makedirs(NSYS_OUTPUT_DIR, exist_ok=True)
    try:
        version_out = subprocess.run(["nsys", "--version"], capture_output=True,
                                     text=True, timeout=10)
        print(f"[gpu_profiling] nsys version: "
              f"{version_out.stdout.strip() or version_out.stderr.strip()}",
              flush=True)
    except Exception as exc:
        print(f"[gpu_profiling] could not get nsys --version: {exc!r}",
              flush=True)

    timestamp = int(time.time())
    report_path = os.path.join(
        NSYS_OUTPUT_DIR,
        f"container_{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    modal_args = sys.argv[1:]

    nsys_cmd = [
        "nsys", "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat",
        f"--trace={NSYS_TRACE_FLAGS}",
        "--cuda-graph-trace=node",
        "--gpu-metrics-devices=cuda-visible",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace-fork-before-exec=true",
        f"--output={report_path}",
        "--force-overwrite=true",
        sys.executable, "-u", "-m", "modal._container_entrypoint",
        *modal_args,
    ]
    print("[gpu_profiling] re-exec'ing Modal runtime under nsys:", flush=True)
    print("  " + " ".join(nsys_cmd), flush=True)
    print(f"[gpu_profiling] expected report -> {report_path}.nsys-rep",
          flush=True)

    env = os.environ.copy()
    env[NSYS_REEXEC_SENTINEL] = "1"
    env["VLLM_OPBENCH_NSYS_REPORT_PATH"] = report_path
    env["PYTHONUNBUFFERED"] = "1"

    try:
        os.execvpe(nsys_cmd[0], nsys_cmd, env)
    except Exception as exc:
        print(f"[gpu_profiling] FAILED to exec nsys: {exc!r}", flush=True)
        raise


def under_nsys() -> bool:
    return bool(os.environ.get(NSYS_REEXEC_SENTINEL))


def current_report_path() -> str | None:
    if not under_nsys():
        return None
    return os.environ.get("VLLM_OPBENCH_NSYS_REPORT_PATH")


def start_capture() -> bool:
    """Start Nsight collection via cudaProfilerStart(). Non-fatal on
    failure so the benchmark keeps running either way."""
    if not under_nsys():
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        torch.cuda.profiler.start()
        return True
    except Exception as exc:
        print(f"[gpu_profiling] cudaProfilerStart FAILED: {exc!r}", flush=True)
        return False


def stop_capture() -> None:
    if not under_nsys():
        return
    try:
        import torch
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"[gpu_profiling] cudaProfilerStop FAILED: {exc!r}", flush=True)


class profiled_batch:
    """Arms a bounded, steady-state nsys capture window around a
    generate_batch() call instead of bracketing the whole call: a
    background thread waits warmup_s (letting the batch ramp up past
    cold-start admission), starts the capture, waits window_s or until
    the batch finishes (whichever first), then stops it. A batch that
    finishes during warmup captures nothing - there was no steady
    state to sample."""

    def __init__(self, do_profile: bool, profile_name: str | None = None,
                 warmup_s: float = NSYS_CAPTURE_WARMUP_S,
                 window_s: float = NSYS_CAPTURE_WINDOW_S):
        self.do_profile = do_profile
        self.profile_name = profile_name
        self.warmup_s = warmup_s
        self.window_s = window_s
        self.active = False
        # One-way latch: True once a capture window actually opened, and
        # stays True even after the window naturally closes (self.active
        # flips back to False then) - what callers need to know is "did a
        # trace get written for this call", not "is a capture live right
        # now". self.active alone under-reports both a batch that finished
        # after the window closed on its own AND a batch that finished
        # before start_capture() ever ran.
        self.captured = False
        self._nvtx_active = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _run(self):
        if self._stop_event.wait(self.warmup_s):
            return
        self.active = start_capture()
        if not self.active:
            return
        self.captured = True
        if self.profile_name:
            try:
                import torch
                torch.cuda.nvtx.range_push(self.profile_name)
                self._nvtx_active = True
            except Exception:
                pass
        self._stop_event.wait(self.window_s)
        self._end_capture()

    def _end_capture(self):
        if self._nvtx_active:
            try:
                import torch
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
            self._nvtx_active = False
        if self.active:
            stop_capture()
            self.active = False

    def __enter__(self):
        if not self.do_profile:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.warmup_s + self.window_s))
        return False

    def report_path(self) -> str | None:
        return current_report_path()
