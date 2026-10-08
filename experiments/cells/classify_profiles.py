r"""Capture torch.profiler snapshots of QUAIL-B classification queries.

Each (query, model, backend) runs in a fresh process on one H100, through
the benchmark's own `run_query`. The profiler covers one short window of
the query: it arms `offset` seconds after the first AI.CLASSIFY stage
starts and runs for `seconds`. It starts and stops at forward-pass
boundaries (Quail's chunks, vLLM's engine steps), on the thread that
runs them, so a window holds whole passes. Model loading, planning, and
the filter or join stages before the classification run unprofiled. The
run's own wall time includes profiler overhead; take headline seconds
from the benchmark runs.

Stock vLLM runs its engine core in the client process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) so the profiler sees its kernels. The
benchmark runs keep the engine core in its own process for Qwen3 and
Kai; DiffusionGemma already runs it in the client process.

One container per model, the models in parallel. The defaults are the
three classification queries farthest from SoL, the three models, and
both engines:

    log="results/profiles/$(date -u +%Y%m%dT%H%M%SZ)-classify-profiles.log"
    uv run modal run --detach experiments/cells/classify_profiles.py::profile \
      --queries IMDB-11,IMDB-14,BIO-6 --backends quail,stock_vllm \
      --sf 0.5 --offset 10 --seconds 5 2>&1 | tee "$log"

Fetch a container's summary later with
`modal.FunctionCall.from_id("<fc-...>").get()`. Each run writes, under
`/results/ablations/classify-profiles/<tag>/<model>/<backend>/<query>/`:

    trace.json.gz   the window's chrome trace (CPU and CUDA activity)
    window.json     stage times, where the window sits, GPU busy and
                    idle seconds in the window, and the kernels with
                    the most GPU time
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from quail.bench.images import gpu_image
from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)

PROFILE_ROOT = "/results/ablations/classify-profiles"
TOP_KERNELS = 25
TOP_GAPS = 10
MEASUREMENT_FIELDS = ("wall_s", "model_wall_s", "gpu_s", "chunks",
                      "planning_s", "fresh_tokens", "input_tokens")
image = gpu_image()


class Window:
    """Decide at pass boundaries when a timed capture starts and stops.

    The window arms `offset` seconds after the first classification
    stage starts and closes once `seconds` have passed since it
    started. Both edges land on a pass boundary, so the capture holds
    whole passes and runs a little longer than `seconds`. A window
    still open when the query ends is closed there and marked cut.
    """

    def __init__(self, offset: float, seconds: float):
        self.offset = offset
        self.seconds = seconds
        self.phase_started = None
        self.started = None
        self.ended = None
        self.passes = 0
        self.cut = False

    def phase(self, now: float) -> None:
        """Record the start of a classification stage."""
        if self.phase_started is None:
            self.phase_started = now

    def boundary(self, now: float) -> str | None:
        """Return "start" or "stop" when the capture changes at this boundary."""
        if self.phase_started is None or self.ended is not None:
            return None
        if self.started is None:
            if now - self.phase_started >= self.offset:
                self.started = now
                return "start"
            return None
        if now - self.started >= self.seconds:
            self.ended = now
            return "stop"
        self.passes += 1
        return None

    def finish(self, now: float) -> str | None:
        """Close a capture the query ended before `seconds` had passed."""
        if self.started is not None and self.ended is None:
            self.ended = now
            self.cut = True
            return "stop"
        return None

    def summary(self, origin: float) -> dict:
        """Describe the window in seconds after `origin`."""
        return {
            "offset_s": self.offset,
            "seconds": self.seconds,
            "armed": self.started is not None,
            "start_s": (None if self.started is None
                        else round(self.started - origin, 3)),
            "end_s": None if self.ended is None else round(self.ended - origin, 3),
            "passes": self.passes,
            "cut": self.cut,
        }


class Capture:
    """One torch.profiler session driven by a Window."""

    def __init__(self, torch, window: Window):
        self.torch = torch
        self.window = window
        self.profiler = None
        self.result = None

    def boundary(self) -> None:
        """Start or stop the profiler if the window says so at this boundary."""
        self._apply(self.window.boundary(time.perf_counter()))

    def finish(self) -> None:
        """Stop a profiler still running when the query ends."""
        self._apply(self.window.finish(time.perf_counter()))

    def _apply(self, action: str | None) -> None:
        if action == "start":
            self.profiler = self.torch.profiler.profile(activities=[
                self.torch.profiler.ProfilerActivity.CPU,
                self.torch.profiler.ProfilerActivity.CUDA])
            self.profiler.__enter__()
        elif action == "stop":
            # enqueued kernels must finish before collection stops
            self.torch.cuda.synchronize()
            self.profiler.__exit__(None, None, None)
            self.result = self.profiler
            self.profiler = None


def gpu_activity(events, top: int) -> dict:
    """Measure GPU busy and idle time over a window's device events.

    Args:
        events: Profiler events; device events carry `device_type`
            CUDA and a `time_range` in microseconds.
        top: Number of idle gaps to list, longest first.
    """
    from torch.autograd import DeviceType

    kernels = sorted((event.time_range.start, event.time_range.end, event.name)
                     for event in events
                     if event.device_type == DeviceType.CUDA)
    if not kernels:
        return {}
    start = kernels[0][0]
    end = max(k_end for _, k_end, _ in kernels)
    busy = 0.0
    gaps = []
    covered = start
    previous = "(window start)"
    for k_start, k_end, name in kernels:
        if k_start > covered:
            gaps.append((k_start - covered, covered - start, previous, name))
        if k_end > covered:
            busy += k_end - max(k_start, covered)
            covered = k_end
            previous = name
    gaps.sort(reverse=True)
    return {
        "span_s": round((end - start) / 1e6, 4),
        "busy_s": round(busy / 1e6, 4),
        "idle_s": round((end - start - busy) / 1e6, 4),
        "gaps": [{"ms": round(gap / 1e3, 2), "at_ms": round(at / 1e3, 1),
                  "after": before[:80], "before": after[:80]}
                 for gap, at, before, after in gaps[:top]],
    }


def top_kernels(averages, top: int) -> list[dict]:
    """List the kernels with the most device time and their share of it.

    Host operators also carry device time, the time of the kernels they
    launched, so only device-side entries count.

    Args:
        averages: The profiler's key averages.
        top: Number of kernels to list.
    """
    from torch.autograd import DeviceType

    rows = []
    for average in averages:
        device_us = getattr(average, "self_device_time_total",
                            getattr(average, "self_cuda_time_total", 0))
        if average.device_type == DeviceType.CUDA and device_us > 0:
            rows.append((device_us, average.count, average.key))
    total = sum(device_us for device_us, _, _ in rows) or 1
    rows.sort(reverse=True)
    return [{"name": name[:160], "calls": count,
             "device_s": round(device_us / 1e6, 4),
             "share": round(device_us / total, 4)}
            for device_us, count, name in rows[:top]]


def _cupti_preinit(torch) -> None:
    """Run one throwaway profiled kernel so CUPTI's start stays out of the window."""
    x = torch.ones(1024, device="cuda")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]):
        (x * 2.0).sum().item()
    torch.cuda.synchronize()


def _hook_quail(capture: Capture, phases: list, origin: float) -> None:
    """Mark Quail's classification stages and its forward passes."""
    import quail.backends.quail.executor.classify as classify_module
    import quail.backends.quail.executor.loop as loop_module

    classify = classify_module.QuailClassifier.classify
    forward = loop_module._forward

    def classify_hook(self, spec, *args, **kwargs):
        capture.window.phase(time.perf_counter())
        phase = {"stage": f"classify {spec.name}",
                 "start_s": round(time.perf_counter() - origin, 3)}
        phases.append(phase)
        try:
            return classify(self, spec, *args, **kwargs)
        finally:
            phase["end_s"] = round(time.perf_counter() - origin, 3)

    def forward_hook(*args, **kwargs):
        capture.boundary()
        return forward(*args, **kwargs)

    classify_module.QuailClassifier.classify = classify_hook
    loop_module._forward = forward_hook


def _hook_vllm(capture: Capture, phases: list, origin: float) -> None:
    """Mark the vLLM baseline's classification calls and its engine steps."""
    from vllm.v1.engine.llm_engine import LLMEngine

    import quail.backends.request as request_module

    classify = request_module.RequestModelExecution._classify
    step = LLMEngine.step

    def classify_hook(self, spec, bodies):
        capture.window.phase(time.perf_counter())
        phase = {"stage": f"classify {spec.output}",
                 "start_s": round(time.perf_counter() - origin, 3)}
        phases.append(phase)
        try:
            return classify(self, spec, bodies)
        finally:
            phase["end_s"] = round(time.perf_counter() - origin, 3)

    def step_hook(self, *args, **kwargs):
        capture.boundary()
        return step(self, *args, **kwargs)

    request_module.RequestModelExecution._classify = classify_hook
    LLMEngine.step = step_hook


def child(arguments: str) -> None:
    """Run one query under the windowed profiler and save its trace.

    Args:
        arguments: JSON of query_id, model, backend, sf, collection_id,
            offset, seconds, and out_dir.
    """
    import torch

    import quail
    from quail.bench.process_isolation import visible_gpu_uuids
    from quail.bench.quailb import run_query
    from quail.bench.results import write_json
    from quail_b.benchmark import load_benchmark

    args = json.loads(arguments)
    backend = args["backend"]
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend != "quail":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _cupti_preinit(torch)
    suite = load_benchmark([args["query_id"]], scale_factor=args["sf"],
                           data_dir=Path(DATA_DIR) / f"sf{args['sf']}",
                           collection_id=args["collection_id"] or None)
    spec = suite.queries[0]
    tables = {relation.table: suite.tables[relation.table]
              for relation in spec.info.relations}
    config = quail.EngineConfig(gpus=1, model=args["model"], backend=backend,
                                device="h100-sxm")
    window = Window(args["offset"], args["seconds"])
    capture = Capture(torch, window)
    phases = []
    with quail.Session(config) as session:
        origin = time.perf_counter()
        if backend == "quail":
            _hook_quail(capture, phases, origin)
        else:
            _hook_vllm(capture, phases, origin)
        output = run_query(session, spec, tables)
        capture.finish()
        wall_s = time.perf_counter() - origin
    record = {
        "query": args["query_id"], "model": args["model"], "backend": backend,
        "sf": args["sf"], "collection_id": args["collection_id"],
        "gpu_uuids": list(visible_gpu_uuids()),
        "vllm_engine_core": None if backend == "quail" else "client process",
        "profiled_wall_s": round(wall_s, 3),
        "measurements": {name: output.measurements.get(name)
                         for name in MEASUREMENT_FIELDS},
        "phases": phases,
        "window": window.summary(origin),
    }
    if capture.result is not None:
        record["gpu"] = gpu_activity(capture.result.events(), TOP_GAPS)
        record["kernels"] = top_kernels(capture.result.key_averages(),
                                        TOP_KERNELS)
        capture.result.export_chrome_trace(str(out_dir / "trace.json.gz"))
        record["trace"] = str(out_dir / "trace.json.gz")
    write_json(out_dir / "window.json", record)
    print("RESULT " + json.dumps(record, default=str), flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=VOLUMES)
def profile_model(model: str, query_ids: list[str], backends: list[str],
                  sf: float, collection_id: str, offset: float,
                  seconds: float, tag: str) -> str:
    """Profile every backend and query of one model, each in a fresh process.

    Args:
        model: Model name.
        query_ids: QUAIL-B query IDs to run.
        backends: Backend names, such as quail and stock_vllm.
        sf: Dataset scale factor.
        collection_id: Reference label collection.
        offset: Seconds into the first classification stage before the
            window arms.
        seconds: Seconds the window stays open, at least.
        tag: Output directory name under the profile root.

    Returns:
        The RESULT lines of every run, one per line.

    Raises:
        RuntimeError: A run's process failed.
    """
    lines = []
    for backend in backends:
        for query_id in query_ids:
            out_dir = f"{PROFILE_ROOT}/{tag}/{model}/{backend}/{query_id}"
            arguments = json.dumps(dict(
                query_id=query_id, model=model, backend=backend, sf=sf,
                collection_id=collection_id, offset=offset, seconds=seconds,
                out_dir=out_dir))
            command = [sys.executable, "-c",
                       "import sys, classify_profiles; "
                       "classify_profiles.child(sys.argv[1])", arguments]
            env = {**os.environ,
                   "PYTHONPATH": "/root:" + os.environ.get("PYTHONPATH", "")}
            started = time.perf_counter()
            done = subprocess.run(command, env=env, cwd="/tmp",
                                  capture_output=True, text=True)
            print(done.stdout[-60000:], flush=True)
            if done.returncode:
                print(done.stderr[-8000:], flush=True)
                raise RuntimeError(f"{query_id} on {backend} failed")
            lines += [line for line in done.stdout.splitlines()
                      if line.startswith("RESULT ")]
            print(f"{model} {backend} {query_id}: process "
                  f"{time.perf_counter() - started:.1f} s", flush=True)
            results_vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def profile(queries: str = "IMDB-11,IMDB-14,BIO-6",
            models: str = "qwen3-4b-fp8,diffusion-gemma-26b-a4b-fp8,"
                          "decision-2.0-kai-0.6b-bf16",
            backends: str = "quail,stock_vllm", sf: float = 0.5,
            offset: float = 10.0, seconds: float = 5.0, tag: str = ""):
    query_ids = [query.strip() for query in queries.split(",") if query.strip()]
    backend_names = [name.strip() for name in backends.split(",")
                     if name.strip()]
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data = ensure_data.spawn(sf, query_ids, "")
    print(f"function call id: {data.object_id} (data)", flush=True)
    collection = data.get()
    calls = {}
    for model in [name.strip() for name in models.split(",") if name.strip()]:
        calls[model] = profile_model.spawn(
            model, query_ids, backend_names, sf, collection, offset, seconds,
            tag)
        print(f"function call id: {calls[model].object_id} ({model})",
              flush=True)
    print(f"outputs under {PROFILE_ROOT}/{tag}/", flush=True)
    for model, call in calls.items():
        for line in call.get().splitlines():
            print(f"SUMMARY {model} " + line, flush=True)
