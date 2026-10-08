r"""Capture torch.profiler snapshots of QUAIL-B classification queries.

Each (query, model, backend) runs in a fresh process on one H100, through
the benchmark's own `run_query`. The profiler covers short windows of the
query's model execution: the first arms `offset` seconds after the first
model node starts, each stays open for `seconds`, and another arms
`every` seconds after the previous one started. Windows start and stop
at forward-pass boundaries (Quail's chunks, vLLM's engine steps), on the
thread that runs them, so a window holds whole passes. Model loading and
planning run unprofiled. Each window is saved with the stages it fell
in, so a report can pick the window in the stage that dominates the
query. The run's own wall time includes profiler overhead; take headline
seconds from the benchmark runs.

Stock vLLM runs its engine core in the client process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) so the profiler sees its kernels. The
benchmark runs keep the engine core in its own process for Qwen3 and
Kai; DiffusionGemma already runs it in the client process. Stock vLLM
refuses AI.CLASSIFY on DiffusionGemma, whose engine returns no text.

One container per model, the models in parallel. The defaults are the
three classification queries farthest from SoL, the three models, and
both engines:

    log="results/profiles/$(date -u +%Y%m%dT%H%M%SZ)-classify-profiles.log"
    uv run modal run --detach experiments/cells/classify_profiles.py::profile \
      --queries IMDB-11,IMDB-14,BIO-6 --backends quail,stock_vllm \
      --sf 0.5 --offset 10 --every 120 --seconds 5 2>&1 | tee "$log"

Fetch a container's summary later with
`modal.FunctionCall.from_id("<fc-...>").get()`. Each run writes, under
`/results/ablations/classify-profiles/<tag>/<model>/<backend>/<query>/`:

    trace-<n>.json.gz   window n's chrome trace (CPU and CUDA activity)
    window.json         stage times, each window's placement and
                        stages, its GPU busy and idle seconds, and the
                        kernels with the most GPU time
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
# CUPTI records the launch queue being full as a device event; it is
# not a kernel
QUEUE_MARKER = "Command Buffer Full"
image = gpu_image()


class Windows:
    """Decide at pass boundaries when timed captures start and stop.

    The first window arms `offset` seconds after model execution starts.
    A window closes once `seconds` have passed since it started, and the
    next arms `every` seconds after the previous one started; `every` of
    zero means one window. Both edges land on a pass boundary, so a
    capture holds whole passes and runs a little longer than `seconds`.
    A window still open when the query ends is closed there and marked
    cut.
    """

    def __init__(self, offset: float, every: float, seconds: float):
        self.offset = offset
        self.every = every
        self.seconds = seconds
        self.phase_started = None
        self.windows = []
        self.open = None

    def phase(self, now: float) -> None:
        """Record the start of model execution."""
        if self.phase_started is None:
            self.phase_started = now

    def boundary(self, now: float) -> str | None:
        """Return "start" or "stop" when a capture changes at this boundary."""
        if self.phase_started is None:
            return None
        if self.open is None:
            if not self.windows:
                due = self.phase_started + self.offset
            elif self.every:
                due = self.windows[-1]["started"] + self.every
            else:
                return None
            if now >= due:
                self.open = {"index": len(self.windows), "started": now,
                             "ended": None, "passes": 0, "cut": False}
                self.windows.append(self.open)
                return "start"
            return None
        if now - self.open["started"] >= self.seconds:
            self.open["ended"] = now
            self.open = None
            return "stop"
        self.open["passes"] += 1
        return None

    def finish(self, now: float) -> str | None:
        """Close a capture the query ended before `seconds` had passed."""
        if self.open is not None:
            self.open.update(ended=now, cut=True)
            self.open = None
            return "stop"
        return None

    def summary(self, origin: float, phases: list) -> list[dict]:
        """Describe each window in seconds after `origin`, with its stages.

        Args:
            origin: The time the query started.
            phases: Stage records with `stage`, `start_s`, and `end_s`
                in seconds after `origin`; a window lists the ones it
                overlaps.
        """
        rows = []
        for window in self.windows:
            start = window["started"] - origin
            end = window["ended"] - origin
            rows.append({
                "index": window["index"], "start_s": round(start, 3),
                "end_s": round(end, 3), "passes": window["passes"],
                "cut": window["cut"],
                "stages": [phase["stage"] for phase in phases
                           if phase["start_s"] < end
                           and phase.get("end_s", end) > start],
            })
        return rows


class Capture:
    """The torch.profiler sessions a Windows schedule drives."""

    def __init__(self, torch, windows: Windows, out_dir: Path):
        self.torch = torch
        self.windows = windows
        self.out_dir = out_dir
        self.profiler = None
        self.results = {}

    def boundary(self) -> None:
        """Start or stop the profiler if a window begins or ends here."""
        self._apply(self.windows.boundary(time.perf_counter()))

    def finish(self) -> None:
        """Stop a profiler still running when the query ends."""
        self._apply(self.windows.finish(time.perf_counter()))

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
            index = self.windows.windows[-1]["index"]
            path = self.out_dir / f"trace-{index}.json.gz"
            self.profiler.export_chrome_trace(str(path))
            self.results[index] = {
                "trace": str(path),
                "gpu": gpu_activity(self.profiler.events(), TOP_GAPS),
                "kernels": top_kernels(self.profiler.key_averages(),
                                       TOP_KERNELS),
            }
            self.profiler = None
            print(f"[profile] window {index} -> {path}", flush=True)


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
                     if event.device_type == DeviceType.CUDA
                     and not event.name.startswith(QUEUE_MARKER))
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
        if (average.device_type == DeviceType.CUDA and device_us > 0
                and not average.key.startswith(QUEUE_MARKER)):
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


def _timed(capture, phases: list, origin: float, name, function,
           start_phase: bool):
    """Wrap `function` to record a stage named by `name(args)`.

    Args:
        capture: The capture whose windows learn when execution starts.
        phases: The stage records to append to.
        origin: The time the query started.
        name: Callable from the wrapped function's arguments to the
            stage name.
        function: The function to wrap.
        start_phase: Whether this stage starts model execution.
    """
    def wrapper(*args, **kwargs):
        now = time.perf_counter()
        if start_phase:
            capture.windows.phase(now)
        phase = {"stage": name(*args, **kwargs),
                 "start_s": round(now - origin, 3)}
        phases.append(phase)
        try:
            return function(*args, **kwargs)
        finally:
            phase["end_s"] = round(time.perf_counter() - origin, 3)

    return wrapper


def _hook_quail(capture: Capture, phases: list, origin: float) -> None:
    """Mark Quail's model nodes, its stage loops, and its forward passes."""
    import quail.backends.quail.backend as backend_module
    import quail.backends.quail.executor.classify as classify_module
    import quail.backends.quail.executor.loop as loop_module
    import quail.backends.quail.executor.pipeline as pipeline_module
    import quail.backends.quail.executor.stages as stages_module

    backend_module.QuailModelExecution.execute = _timed(
        capture, phases, origin, lambda self, node, inputs: node.type_name,
        backend_module.QuailModelExecution.execute, True)
    backend_module.QuailModelExecution.execute_pipeline = _timed(
        capture, phases, origin, lambda self, pipeline, *rest: "pipeline",
        backend_module.QuailModelExecution.execute_pipeline, True)
    run_stages = _timed(
        capture, phases, origin,
        lambda *args, label=None, **kwargs: f"stages: {label}",
        stages_module.run_stages, False)
    # loop.py imports run_stages at call time; these two at import time
    stages_module.run_stages = run_stages
    pipeline_module.run_stages = run_stages
    classify_module.run_stages = run_stages
    forward = loop_module._forward

    def forward_hook(*args, **kwargs):
        capture.boundary()
        return forward(*args, **kwargs)

    loop_module._forward = forward_hook


def _hook_vllm(capture: Capture, phases: list, origin: float) -> None:
    """Mark the vLLM baseline's nodes, its classifications, and its steps."""
    from vllm.v1.engine.llm_engine import LLMEngine

    import quail.backends.request as request_module

    execution = request_module.RequestModelExecution
    execution.execute = _timed(
        capture, phases, origin, lambda self, node, inputs: node.type_name,
        execution.execute, True)
    execution._classify = _timed(
        capture, phases, origin,
        lambda self, spec, bodies: f"classify {spec.output}",
        execution._classify, False)
    step = LLMEngine.step

    def step_hook(self, *args, **kwargs):
        capture.boundary()
        return step(self, *args, **kwargs)

    LLMEngine.step = step_hook


def child(arguments: str) -> None:
    """Run one query under the windowed profiler and save its traces.

    Args:
        arguments: JSON of query_id, model, backend, sf, collection_id,
            offset, every, seconds, and out_dir.
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
    windows = Windows(args["offset"], args["every"], args["seconds"])
    capture = Capture(torch, windows, out_dir)
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
        "schedule": {"offset_s": args["offset"], "every_s": args["every"],
                     "seconds": args["seconds"]},
        "windows": [{**window, **capture.results.get(window["index"], {})}
                    for window in windows.summary(origin, phases)],
    }
    write_json(out_dir / "window.json", record)
    print("RESULT " + json.dumps(record, default=str), flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=VOLUMES)
def profile_model(model: str, query_ids: list[str], backends: list[str],
                  sf: float, collection_id: str, offset: float, every: float,
                  seconds: float, tag: str) -> str:
    """Profile every backend and query of one model, each in a fresh process.

    Args:
        model: Model name.
        query_ids: QUAIL-B query IDs to run.
        backends: Backend names, such as quail and stock_vllm.
        sf: Dataset scale factor.
        collection_id: Reference label collection.
        offset: Seconds into model execution before the first window arms.
        every: Seconds between window starts; zero for one window.
        seconds: Seconds a window stays open, at least.
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
                collection_id=collection_id, offset=offset, every=every,
                seconds=seconds, out_dir=out_dir))
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
            offset: float = 10.0, every: float = 120.0, seconds: float = 5.0,
            tag: str = ""):
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
            model, query_ids, backend_names, sf, collection, offset, every,
            seconds, tag)
        print(f"function call id: {calls[model].object_id} ({model})",
              flush=True)
    print(f"outputs under {PROFILE_ROOT}/{tag}/", flush=True)
    for model, call in calls.items():
        for line in call.get().splitlines():
            print(f"SUMMARY {model} " + line, flush=True)
