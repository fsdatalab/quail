"""Profile QUAIL-B query execution on Modal, excluding model startup.

Queries run in fresh processes. Modes are plain (no profiler), cprofile,
sample (Python stack sampling), trace (torch.profiler), timeline
(executor-step and GPU event timing), verify (graph/eager output checks),
and eager (graphs disabled, unchanged KV capacity). Each run prints timing,
allocation, accuracy, and result-digest fields, and saves the answer rows
beside the profiles. The env option sets child-process variables;
QUAIL_VLLM_EAGER=1 boots the vLLM baselines without CUDA graphs. A trace
run also saves 5 seconds of its trace, from 40% into the run, as a
Perfetto trace under /results/ablations/traces.

Pass QUAIL_OVERRIDES=module:NAME=json-value;... in env to set module
constants in the child, such as an executor constant under test, or
QUAIL_SWEEP=overrides|overrides|... to run each set as its own version
in one container, alternating order between repetitions.

Set QUAIL_BEFORE_SOURCE to a directory containing an earlier copy of
quail to compare both versions on the same GPU. Run order alternates
between repetitions.

    uv run modal run --detach experiments/cells/profile_query.py::profile
        --queries BIO-5,FEV-11 --modes plain,cprofile,trace
"""

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

from bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)
from quail.deploy.images import gpu_image
from quail.specs import modal_gpu_type

PROFILE_DIR = "/results/ablations/profiles"
# Perfetto traces, served by experiments/cells/trace_server.py
TRACE_DIR = "/results/ablations/traces"
# seconds of a trace run kept for Perfetto, starting 40% into the GPU
# activity; a whole run's trace is too large for the browser
TRACE_WINDOW_S = 5.0
# caching allocator counters: a retry flushes the cache after a failed
# cudaMalloc, which waits for every stream
ALLOCATOR_COUNTS = ("num_alloc_retries", "num_sync_all_streams",
                    "num_device_alloc", "num_device_free")
NODE_FIELDS = ("wall_s", "gpu_s", "chunks", "pack_s", "fresh_tokens",
               "suffix_tokens", "admission_s", "pass_sol_s", "pass_passes",
               "pass_decode_rows", "pass_prefill_tokens")
BEFORE_REMOTE = "/root/before"
BEFORE_LOCAL = os.environ.get("QUAIL_BEFORE_SOURCE", "")
image = gpu_image(*([(BEFORE_LOCAL, BEFORE_REMOTE)]
                    if modal.is_local() and BEFORE_LOCAL else []))
image = image.add_local_python_source("experiments")


def _answer_rows(output) -> dict:
    """Return each answer table's rows and the result rows, sorted."""
    tables = {**output.filter_answers, **output.join_answers,
              **(output.classify_answers or {}), **(output.map_answers or {}),
              **(output.extract_answers or {})}
    rows = {name: sorted(json.dumps(list(row.values()), default=str)
                         for row in tables[name].to_pylist())
            for name in sorted(tables)}
    rows[""] = sorted(json.dumps(list(row.values()), default=str)
                      for row in output.rows.to_pylist())
    return rows


def _digest(rows: dict) -> str:
    """Hash answer tables and result rows in document ID order."""
    import hashlib

    digest = hashlib.sha256()
    for name, table in rows.items():
        if name:
            digest.update(json.dumps([name, table]).encode())
    digest.update(json.dumps(rows[""]).encode())
    return digest.hexdigest()[:16]


def _eager_vllm() -> None:
    """Boot the vLLM baselines without torch.compile or CUDA graphs."""
    from quail.baselines import vllm as vllm_baseline

    llm_kwargs = vllm_baseline.VLLMEngine.llm_kwargs

    def eager_kwargs(self, spec, device=None):
        kwargs = llm_kwargs(self, spec, device)
        kwargs.pop("compilation_config", None)
        return {**kwargs, "enforce_eager": True}

    vllm_baseline.VLLMEngine.llm_kwargs = eager_kwargs
    print("vLLM baselines boot eager", flush=True)


def _forward_kernels(events):
    """Measure kernel envelopes and eager launches attributed to forward scopes."""
    from bisect import bisect_left
    from collections import Counter, defaultdict

    from torch.autograd import DeviceType

    # Kineto's CUDA annotations include custom and replayed kernels that
    # do not link individually to a PyTorch CPU operator.
    forwards, kernels = [], []
    launches, attention = defaultdict(Counter), Counter()
    for event in events:
        if event.device_type == DeviceType.CUDA:
            if event.name == "quail.forward":
                forwards.append(event.time_range)
            elif not event.is_user_annotation:
                kernels.append(event.time_range)
        elif event.name.startswith("quail.attention."):
            attention[event.name] += 1
        elif any(kind in event.name for kind in ("Launch", "Memcpy", "Memset")):
            parent = event.cpu_parent
            while parent is not None:
                if parent.name.startswith("quail.attention."):
                    launches[parent.name][event.name] += 1
                    break
                parent = parent.cpu_parent
    merged = []
    for value in sorted(kernels, key=lambda value: value.start):
        if merged and value.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], value.end)
        else:
            merged.append([value.start, value.end])
    ends = [end for _, end in merged]
    span = busy = 0.0
    for value in forwards:
        span += value.end - value.start
        index = bisect_left(ends, value.start)
        while index < len(merged) and merged[index][0] < value.end:
            start, end = merged[index]
            busy += max(0, min(end, value.end) - max(start, value.start))
            index += 1
    return {"passes": len(forwards), "span_s": span / 1e6,
            "busy_s": busy / 1e6, "idle_s": (span - busy) / 1e6,
            "total_kernels": len(kernels),
            "attention_dispatch": {
                name.removeprefix("quail.attention."): {
                    "layers": count, "launches": sum(
                        n for api, n in launches[name].items() if "Launch" in api),
                    "commands_per_layer": sum(launches[name].values()) / count,
                    "apis": dict(launches[name])}
                for name, count in attention.items()}}


def _kernel_lanes(events: list) -> list:
    """Move each GPU event that overlaps an earlier one on its stream to a lane.

    Kernels on one stream can overlap, as under programmatic dependent
    launch; Perfetto cannot nest a partly overlapping slice and reports
    an import error for each.

    Returns:
        The events, with one thread name event per added lane.
    """
    by_stream = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in (
                "kernel", "gpu_memcpy", "gpu_memset"):
            by_stream[(event["pid"], event["tid"])].append(event)
    names = []
    for (pid, tid), items in by_stream.items():
        ends = []
        for event in sorted(items, key=lambda item: item["ts"]):
            end = event["ts"] + event.get("dur", 0)
            lane = next((index for index, last in enumerate(ends)
                         if last <= event["ts"]), len(ends))
            if lane == len(ends):
                ends.append(end)
            ends[lane] = end
            if lane:
                event["tid"] = f"{tid} overlap {lane}"
        names.extend({"ph": "M", "name": "thread_name", "pid": pid,
                      "tid": f"{tid} overlap {lane}",
                      "args": {"name": f"stream {tid}, overlapping {lane}"}}
                     for lane in range(1, len(ends)))
    return events + names


def _graph_replays(events: list) -> list:
    """Add one slice per CUDA graph replay, on its own GPU row.

    The profiler records each kernel of a replay as its own slice; a
    replay's kernels share the correlation id of its cudaGraphLaunch.

    Returns:
        The events, with the replay slices and their row's name.
    """
    replays = defaultdict(list)
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "kernel" and args.get("graph id"):
            replays[(event["pid"], args["correlation"])].append(event)
    added = []
    for (pid, _), kernels in replays.items():
        start = min(kernel["ts"] for kernel in kernels)
        end = max(kernel["ts"] + kernel.get("dur", 0) for kernel in kernels)
        added.append({"ph": "X", "cat": "graph_replay", "pid": pid,
                      "tid": "graph replays", "ts": start, "dur": end - start,
                      "name": f"graph {kernels[0]['args']['graph id']}",
                      "args": {"kernels": len(kernels)}})
    added.extend({"ph": "M", "name": "thread_name", "pid": pid,
                  "tid": "graph replays", "args": {"name": "graph replays"}}
                 for pid in {pid for pid, _ in replays})
    return events + added


def _save_trace(prof, path: Path) -> dict:
    """Save a window of a torch.profiler run as a gzipped Perfetto trace.

    Returns:
        The trace's file name and the window's start and length in seconds
        from the first GPU kernel.
    """
    import gzip
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(raw))
        trace = json.loads(raw.read_text())
    events = trace["traceEvents"]
    kernels = [event["ts"] for event in events if event.get("cat") == "kernel"]
    if not kernels:
        return {}
    first, last = min(kernels), max(kernels)
    start = first + 0.4 * (last - first)
    end = start + TRACE_WINDOW_S * 1e6
    # the profiler's own span covers the whole run and would set the view
    trace["traceEvents"] = _graph_replays(_kernel_lanes([
        event for event in events
        if event.get("ph") == "M"
        or ("ts" in event and event.get("cat") != "Trace"
            and event["ts"] <= end
            and event["ts"] + event.get("dur", 0) >= start)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as out:
        json.dump(trace, out)
    return {"file": path.name, "start_s": round((start - first) / 1e6, 3),
            "window_s": TRACE_WINDOW_S}


def _gpu_gaps(prof, top: int) -> dict:
    """Measure GPU busy and idle time, including the longest idle gaps."""
    from torch.autograd import DeviceType

    events = list(prof.events())
    kernels = sorted((event.time_range.start, event.time_range.end, event.name)
                     for event in events
                     if event.device_type == DeviceType.CUDA
                     and not event.is_user_annotation)
    cpu = [event.time_range.start for event in events
           if event.device_type == DeviceType.CPU]
    if not kernels:
        return {}
    start = min(cpu + [kernels[0][0]])
    end = kernels[-1][1]
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
    by_kernel = {}
    for k_start, k_end, name in kernels:
        total, calls = by_kernel.get(name, (0.0, 0))
        by_kernel[name] = total + k_end - k_start, calls + 1
    ranked = sorted(by_kernel.items(), key=lambda item: -item[1][0])
    return {
        "kernels": [{"name": name, "s": round(total / 1e6, 4), "calls": calls}
                    for name, (total, calls) in ranked[:top]],
        "span_s": (end - start) / 1e6, "busy_s": busy / 1e6,
        "idle_s": (end - start - busy) / 1e6,
        "forward": _forward_kernels(events),
        "gaps": [{"ms": round(gap / 1e3, 2), "at_ms": round(at / 1e3, 1),
                  "after": before[:60], "before": after[:60]}
                 for gap, at, before, after in gaps[:top]],
    }


class Timeline:
    """Host spans of the executor's steps and GPU spans of each chunk.

    Wraps the chunk packer, the forward pass, the admission's next
    chunk, and the readouts' submit and result. A reference event
    recorded on an idle GPU at the window's start puts GPU times on
    the host clock.
    """

    def __init__(self, torch):
        import quail.backends.quail.executor.chunk as chunk
        import quail.backends.quail.executor.readout as readout
        import quail.backends.quail.executor.stage_executor as stage_executor
        from quail.backends.quail.executor import admission

        self.torch = torch
        self.entries = []
        self.active = False
        self._wrap(chunk, "pack_chunk", "pack")
        self._wrap(stage_executor, "forward_chunk", "forward", gpu=True)
        self._wrap(admission.JoinAdmission, "next_chunk", "next_chunk")
        for owner in (readout.AsyncLabelLogprobs, readout.AsyncAnswers,
                      readout.AsyncScores):
            self._wrap(owner, "submit", "submit", gpu=True)
            self._wrap(owner, "result", "result")

    def _wrap(self, owner, name, kind, gpu=False):
        original = getattr(owner, name)
        timeline = self

        def wrapper(*args, **kwargs):
            if not timeline.active:
                return original(*args, **kwargs)
            events = None
            if gpu:
                events = (timeline.torch.cuda.Event(enable_timing=True),
                          timeline.torch.cuda.Event(enable_timing=True))
                events[0].record()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                ended = time.perf_counter()
                if gpu:
                    events[1].record()
                rows = getattr(args[1], "shape", (None,))[0] if len(
                    args) > 1 and kind == "submit" else None
                if kind == "forward":
                    rows = args[2].tokens
                timeline.entries.append((kind, started, ended, events, rows))

        if isinstance(owner, type) and isinstance(
                owner.__dict__.get(name), staticmethod):
            wrapper = staticmethod(wrapper)
        setattr(owner, name, wrapper)

    def start(self):
        self.entries = []
        self.torch.cuda.synchronize()
        self.reference = self.torch.cuda.Event(enable_timing=True)
        self.reference.record()
        self.started = time.perf_counter()
        self.active = True

    def stop(self):
        self.torch.cuda.synchronize()
        self.ended = time.perf_counter()
        self.active = False

    def summary(self, top: int) -> dict:
        """Summarize host execution steps, GPU activity, and idle intervals."""
        host = {}
        gpu = []
        steps = []
        for kind, started, ended, events, rows in self.entries:
            host[kind] = host.get(kind, 0.0) + ended - started
            step = [kind, round((started - self.started) * 1e3, 1),
                    round((ended - started) * 1e3, 1)]
            if events is not None:
                step += [round(self.reference.elapsed_time(events[0]), 1),
                         round(events[0].elapsed_time(events[1]), 1), rows]
            if kind != "result":
                steps.append(step)
            if events is not None:
                gpu.append((self.reference.elapsed_time(events[0]) / 1e3,
                            self.reference.elapsed_time(events[1]) / 1e3, kind))
        window = self.ended - self.started
        gpu.sort()
        busy, covered, gaps = 0.0, 0.0, []
        for start, end, kind in gpu:
            if start > covered:
                gaps.append((start - covered, covered, kind))
            if end > covered:
                busy += end - max(start, covered)
                covered = end
        gaps.append((window - covered, covered, "(window end)"))
        gaps.sort(reverse=True)

        def doing(start, end):
            """Calculate host time per execution step within an interval."""
            spent = {}
            for kind, began, ended, *_ in self.entries:
                overlap = (min(ended - self.started, end)
                           - max(began - self.started, start))
                if overlap > 0:
                    spent[kind] = round(spent.get(kind, 0.0) + overlap, 4)
            return spent

        return {
            "window_s": round(window, 4), "gpu_busy_s": round(busy, 4),
            "gpu_idle_s": round(window - busy, 4),
            "host_s": {kind: round(value, 4) for kind, value in host.items()},
            "chunks": sum(1 for kind, *_ in self.entries if kind == "forward"),
            # kind, host start ms, host ms, GPU start ms, GPU ms, rows read
            "steps": steps,
            "gaps": [{"ms": round(gap * 1e3, 2), "at_ms": round(at * 1e3, 1),
                      "next": kind, "host": doing(at, at + gap)}
                     for gap, at, kind in gaps[:top]],
        }


class Sampler:
    """Sample the calling thread's Python stack from a second thread.

    A sample lands only when the sampling thread holds the GIL: about
    every millisecond while the calling thread waits with the GIL
    released, about every 5 ms (the interpreter's switch interval)
    while it runs Python. Each sample is weighted by the seconds since
    the one before, so both count at their real duration.
    """

    def __init__(self, interval: float = 0.001):
        import threading

        self.interval = interval
        self.target = threading.get_ident()
        self.stacks = {}
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.running = True
        self.started = time.perf_counter()
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.ended = time.perf_counter()

    def _loop(self):
        last = time.perf_counter()
        while self.running:
            frame = sys._current_frames().get(self.target)
            now = time.perf_counter()
            stack = []
            while frame is not None:
                code = frame.f_code
                name = code.co_filename.split("/root/")[-1]
                stack.append(f"{code.co_name} ({name}:{code.co_firstlineno})")
                frame = frame.f_back
            key = tuple(reversed(stack))
            self.stacks[key] = self.stacks.get(key, 0.0) + now - last
            last = now
            time.sleep(self.interval)

    def summary(self, top: int) -> dict:
        """Summarize sampled time by leaf frame and Quail frame."""
        leaf, inclusive, total = {}, {}, 0.0
        for stack, count in self.stacks.items():
            inside = [index for index, frame in enumerate(stack)
                      if frame.startswith(("run (", "serve_region ("))]
            if not inside:
                continue
            frames = stack[inside[0] + 1:]
            total += count
            if frames:
                leaf[frames[-1]] = leaf.get(frames[-1], 0) + count
            for frame in set(frames):
                if frame.split("(", 1)[-1].startswith(("quail/", "before/")):
                    inclusive[frame] = inclusive.get(frame, 0) + count

        def ranked(counts):
            return [[round(count * 1e3, 1), frame] for frame, count in sorted(
                counts.items(), key=lambda item: -item[1])[:top]]

        return {"window_s": round(self.ended - self.started, 4),
                "sampled_s": round(total, 4), "leaf_ms": ranked(leaf),
                "quail_inclusive_ms": ranked(inclusive)}


def _trace_calls(owner, name, window):
    """Attribute kernels to forwards and eager attention boundaries in traces."""
    original = getattr(owner, name)

    def traced(*args, **kwargs):
        if window.get("trace") is None:
            return original(*args, **kwargs)
        import torch
        chunk = args[-1]
        path = ("shared" if chunk.meta.get("shared") is not None
                else chunk.attention_mode)
        label = ("quail.forward" if name == "forward_chunk"
                 else f"quail.attention.{path}")
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, traced)


def _apply_overrides(text: str) -> None:
    """Set module constants from "module:NAME=value;..." for one run."""
    import importlib

    for item in filter(None, text.split(";")):
        target, value = item.split("=", 1)
        module, name = target.split(":")
        setattr(importlib.import_module(module), name, json.loads(value))
        print(f"override {module}.{name} = {value}", flush=True)


def _watch_compiles(window):
    """Log each Triton and FlashInfer CuTe DSL compile that runs during a run.

    Each entry gives the seconds from the run's start, the kernel, and
    the seconds the compile took.
    """
    import triton
    from triton import knobs

    started = {}

    def before(*, key, repr, fn, **_):
        started[str(key)] = time.perf_counter()

    def after(*, key, repr, fn, **_):
        if "start" not in window:
            return
        now = time.perf_counter()
        window["compiles"].append({
            "at_s": round(now - window["start"], 3),
            "kernel": f"{fn.module}.{fn.name}",
            "seconds": round(now - started.pop(str(key), now), 3),
            "signature": repr[:300]})

    knobs.runtime.jit_cache_hook = before
    knobs.runtime.jit_post_compile_hook = after
    del triton
    try:
        from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm90
    except ImportError:
        return
    original = delta_rule_sm90.cached_compile

    def cached_compile(func, *args, **kwargs):
        begun = time.perf_counter()
        compiled = original(func, *args, **kwargs)
        seconds = time.perf_counter() - begun
        # a cache hit returns in microseconds
        if seconds > 0.005 and "start" in window:
            window["compiles"].append({
                "at_s": round(begun - window["start"], 3),
                "kernel": f"cute:{type(func).__name__}",
                "seconds": round(seconds, 3),
                "signature": str(func._get_compile_key())[:300]})
        return compiled

    delta_rule_sm90.cached_compile = cached_compile


_WORKER_INSTALLED = False


def _describe_loaded_gpu(gpu) -> dict:
    """Return weight memory and module types of a loaded GPU pipeline."""
    import inspect

    pipeline = gpu.pipeline
    info = {"weights_bytes": gpu.weights_bytes, "pool_bytes": gpu.pool_bytes}
    attn_layer = getattr(pipeline, "attention_layer", None)
    gdn_layer = getattr(pipeline, "gdn_layer", None)
    if attn_layer is not None:
        attn = attn_layer.self_attn
        mlp = attn_layer.mlp
        info["attn"] = {
            "qkv_quant": type(attn.qkv_proj.quant_method).__name__,
            "o_quant": type(attn.o_proj.quant_method).__name__,
            "q_norm": type(attn.q_norm).__name__,
            "rotary": type(attn.rotary_emb).__name__,
            "rotary_dim": getattr(attn.rotary_emb, "rotary_dim", None),
            "mlp_gate_up_quant": type(mlp.gate_up_proj.quant_method).__name__,
            "mlp_down_quant": type(mlp.down_proj.quant_method).__name__,
            "mlp_act": type(mlp.act_fn).__name__,
            "input_norm": type(attn_layer.input_layernorm).__name__,
            "qkv_apply_src": inspect.getsource(attn.qkv_proj.quant_method.apply),
            "mlp_apply_src": inspect.getsource(mlp.gate_up_proj.quant_method.apply),
        }
        for label, module in (("qkv_proj", attn.qkv_proj),
                              ("gate_up_proj", mlp.gate_up_proj),
                              ("down_proj", mlp.down_proj)):
            info[label] = {
                key: (list(val.shape), str(val.dtype))
                for key, val in module.named_parameters(recurse=False)
            } | {
                key: (list(val.shape), str(val.dtype))
                for key, val in module.named_buffers(recurse=False)
            }
    if gdn_layer is not None:
        mixer = gdn_layer.linear_attn
        info["gdn"] = {
            "in_qkvz_quant": type(mixer.in_proj_qkvz.quant_method).__name__,
            "in_ba_quant": type(mixer.in_proj_ba.quant_method).__name__,
            "out_quant": type(mixer.out_proj.quant_method).__name__,
            "norm": type(mixer.norm).__name__,
            "norm_src": inspect.getsource(type(mixer.norm).forward),
        }
    try:
        from vllm import _custom_ops as ops
        info["vllm_ops"] = [
            name for name in dir(ops)
            if any(tag in name.lower() for tag in ("fp4", "fp8", "norm", "silu", "gdn"))
        ]
    except Exception:
        pass
    try:
        import flashinfer
        tags = ("fp4", "fp8", "norm", "silu", "gdn", "mm")
        info["flashinfer_ops"] = [
            name for name in dir(flashinfer)
            if any(tag in name.lower() for tag in tags)
        ]
    except Exception:
        pass
    return info


def register_quail_extension(registry) -> None:
    """Install profiling hooks inside the GPU worker process when active."""
    del registry
    global _WORKER_INSTALLED
    command_path = os.environ.get("QUAIL_PROFILE_CMD")
    output_path = os.environ.get("QUAIL_PROFILE_OUT")
    if _WORKER_INSTALLED or not command_path or not output_path:
        return
    _WORKER_INSTALLED = True
    import cProfile
    import io
    import pstats

    import quail.backends.quail.graph as graph_module
    from experiments.cells.piecewise_probe import PiecewiseProbe
    from quail.backends.quail.executor import cuda_graphs, stage_executor

    _apply_overrides(os.environ.get("QUAIL_OVERRIDES", ""))
    window = {}
    _watch_compiles(window)
    probe = PiecewiseProbe()
    _trace_calls(stage_executor, "forward_chunk", window)
    boundary = next((name for name in ("mixer", "attention")
                     if hasattr(getattr(cuda_graphs, "PiecewiseBuffers", None), name)),
                    None)
    if boundary is not None:
        _trace_calls(cuda_graphs.PiecewiseBuffers, boundary, window)
    original_serve_stages = graph_module.serve_stages
    original_stage_run = stage_executor.StageExecutor.run

    def tracked_stage_run(self):
        try:
            return original_stage_run(self)
        finally:
            if "stage_work" in window:
                window["stage_work"].append({
                    "label": self.label,
                    "pack_s": round(self.pack_s, 4),
                    "admission_s": round(self.admission_s, 4),
                    **dict(self.pass_work),
                })

    stage_executor.StageExecutor.run = tracked_stage_run

    def profiled_serve_stages(end, context, subgraph, stages, wanted,
                              hinted=None, first_rows=None):
        if not Path(command_path).is_file():
            return original_serve_stages(
                end, context, subgraph, stages, wanted, hinted, first_rows)
        spec = json.loads(Path(command_path).read_text())
        mode = spec["mode"]
        run = spec["run"]
        top = spec["top"]
        stats_prefix = spec["stats_prefix"]
        query_id = spec["query_id"]
        import torch

        profiler = cProfile.Profile() if mode == "cprofile" else None
        window["profiler"] = profiler
        window["trace"] = None
        window["timeline"] = None
        window["sampler"] = Sampler() if mode == "sample" else None
        if mode == "timeline":
            if "recorder" not in window:
                window["recorder"] = Timeline(torch)
            window["timeline"] = window["recorder"]
        if mode == "trace":
            window["trace"] = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA])
        probe.start(verify=mode == "verify", eager=mode == "eager",
                    work=mode == "work")
        window["compiles"] = []
        window["stage_work"] = []
        window["start"] = time.perf_counter()
        if window["sampler"] is not None:
            window["sampler"].start()
        before = torch.cuda.memory_stats()
        free_before = torch.cuda.mem_get_info()[0]
        if window["timeline"] is not None:
            window["timeline"].start()
        if window["trace"] is not None:
            window["trace"].__enter__()
        if profiler is not None:
            profiler.enable()
        try:
            return original_serve_stages(
                end, context, subgraph, stages, wanted, hinted, first_rows)
        finally:
            torch.cuda.synchronize()
            piecewise = probe.summary()
            if profiler is not None:
                profiler.disable()
            if window["trace"] is not None:
                window["trace"].__exit__(None, None, None)
            if window["timeline"] is not None:
                window["timeline"].stop()
            if window["sampler"] is not None:
                window["sampler"].stop()
            window.pop("start", None)
            after = torch.cuda.memory_stats()
            allocator = {
                name: after.get(name, 0) - before.get(name, 0)
                for name in ALLOCATOR_COUNTS}
            allocator["free_gib_before"] = round(free_before / 2**30, 2)
            allocator["reserved_gib_before"] = round(
                before.get("reserved_bytes.all.current", 0) / 2**30, 2)
            out = {
                "allocator": allocator,
                "piecewise": piecewise,
                "compiles": window["compiles"],
                "stage_work": window.pop("stage_work", []),
            }
            if run == 0:
                out["loaded_gpu"] = _describe_loaded_gpu(context.state["gpu"])
            if window["trace"] is not None:
                out["gpu_window"] = _gpu_gaps(window["trace"], top)
                prefix = Path(stats_prefix).name
                out["perfetto"] = _save_trace(
                    window["trace"], Path(TRACE_DIR) / f"{prefix}-run{run}.json.gz")
            if window["timeline"] is not None:
                out["timeline"] = window["timeline"].summary(top)
            if window["sampler"] is not None:
                out["samples"] = window["sampler"].summary(top)
            if profiler is not None:
                path = f"{stats_prefix}-run{run}.prof"
                profiler.dump_stats(path)
                text = io.StringIO()
                stats = pstats.Stats(profiler, stream=text)
                for order in ("tottime", "cumulative"):
                    text.write(f"\n== {query_id} run {run}: top {top} by {order}"
                               f" (stats at {path})\n")
                    stats.sort_stats(order).print_stats(top)
                out["cprofile_text"] = text.getvalue()
            with open(output_path, "a") as handle:
                handle.write(json.dumps(out, default=str) + "\n")

    graph_module.serve_stages = profiled_serve_stages


def child(arguments: str) -> None:
    """Run one query several times in a fresh session; print a RESULT line per run.

    Args:
        arguments: JSON of query_id, scale_factor, collection_id, model,
            device, modes (one per run: plain, cprofile, sample, trace,
            or timeline), top, and stats_prefix.
    """
    import tempfile

    import quail
    from bench.quailb import run_query
    from quail_b.benchmark import load_benchmark
    from quail_b.scoring import evaluate

    args = json.loads(arguments)
    _apply_overrides(os.environ.get("QUAIL_OVERRIDES", ""))
    if os.environ.get("QUAIL_VLLM_EAGER"):
        _eager_vllm()
    query_id, scale_factor = args["query_id"], args["scale_factor"]
    modes, top = args["modes"], args["top"]
    device = args.get("device", "h100-sxm")
    print(f"quail from {quail.__file__}", flush=True)
    suite = load_benchmark([query_id], scale_factor=scale_factor,
                           data_dir=Path(DATA_DIR) / f"sf{scale_factor}",
                           collection_id=args["collection_id"] or None)
    spec = suite.queries[0]
    sample_rows = args.get("sample_rows", 0)
    tables = {relation.table: (suite.tables[relation.table].slice(0, sample_rows)
                              if sample_rows else suite.tables[relation.table])
              for relation in spec.info.relations}
    with tempfile.TemporaryDirectory() as ipc_dir:
        command_path = Path(ipc_dir) / "command.json"
        output_path = Path(ipc_dir) / "output.jsonl"
        (Path(ipc_dir) / "sitecustomize.py").write_text(
            "from experiments.cells import profile_query\n"
            "profile_query.register_quail_extension(None)\n"
        )
        os.environ["QUAIL_PROFILE_CMD"] = str(command_path)
        os.environ["QUAIL_PROFILE_OUT"] = str(output_path)
        os.environ["PYTHONPATH"] = (
            f"{ipc_dir}:" + os.environ.get("PYTHONPATH", "")
        )
        config = quail.EngineConfig(gpus=1, model=args["model"],
                                    backend=args.get("backend", "quail"),
                                    device=device, gpu_timing=True)
        with quail.Session(config) as session:
            for run, mode in enumerate(modes):
                output_path.unlink(missing_ok=True)
                command_path.write_text(json.dumps({
                    "run": run, "mode": mode, "top": top,
                    "stats_prefix": args["stats_prefix"], "query_id": query_id,
                }))
                output = run_query(session, spec, tables)
                worker_records = ([json.loads(line)
                                   for line in output_path.read_text().splitlines()
                                   if line.strip()]
                                  if output_path.is_file() else [])
                worker = worker_records[-1] if worker_records else {}
                answer_rows = _answer_rows(output)
                answers_path = f"{args['stats_prefix']}-run{run}-answers.json"
                Path(answers_path).write_text(json.dumps(answer_rows))
                measures = output.measurements
                input_tokens = measures.get("input_tokens")
                wall_s = measures.get("wall_s")
                nodes = {key: {name: value[name] for name in NODE_FIELDS
                               if name in value}
                         for key, value in measures["node_metrics"].items()}
                record = {
                    "query": query_id, "run": run, "mode": mode,
                    "backend": config.backend, "device": device,
                    "input_tokens": input_tokens,
                    "input_tokens_per_second": (
                        input_tokens / wall_s
                        if input_tokens is not None and wall_s else None),
                    "cost_usd": (wall_s / 3600 * config.gpus
                                 * session.device.usd_per_hour
                                 if wall_s is not None else None),
                    "sample_rows": sample_rows,
                    "input_rows": {name: len(table) for name, table in tables.items()},
                    "quail": quail.__file__,
                    "digest": _digest(answer_rows),
                    "answers_path": answers_path,
                    **{name: measures.get(name) for name in (
                        "wall_s", "model_wall_s", "gpu_s", "chunks", "frontend_s",
                        "planning_s", "input_ready_s", "physical_prepare_s",
                        "finish_s", "answer_prepare_s", "collection_s",
                        "fresh_tokens", "coordinator_wall_s", "worker_total_s",
                        "kernel_cache_files")},
                    "nodes": nodes,
                    "allocator": worker.get("allocator"),
                    "piecewise": worker.get("piecewise"),
                    "compiles": worker.get("compiles", []),
                    "accuracy": evaluate(spec, output, suite.ground_truth, tables),
                }
                for extra_key in ("loaded_gpu", "gpu_window", "perfetto",
                                  "timeline", "samples", "stage_work"):
                    if extra_key in worker:
                        record[extra_key] = worker[extra_key]
                if len(worker_records) > 1:
                    record["regions"] = len(worker_records)
                print("RESULT " + json.dumps(record, default=str), flush=True)
                for region_record in worker_records:
                    if "cprofile_text" in region_record:
                        print(region_record["cprofile_text"], flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=6 * 3600,
              volumes=VOLUMES)
def profile_query(query_ids: list[str], scale_factor: float, collection_id: str,
                  model: str, modes: list[str], top: int, repeats: int,
                  tag: str, env: str = "", sample_rows: int = 0,
                  backend: str = "quail", device: str = "h100-sxm") -> str:
    """Profile each query in fresh processes and return its RESULT lines.

    Args:
        query_ids: QUAIL-B query IDs to run.
        scale_factor: Dataset scale factor.
        collection_id: Reference label collection.
        model: Model name.
        modes: Profiling modes, one per run within each fresh process.
        top: Number of entries displayed in profiler summaries.
        repeats: Number of repetitions per query and code version.
        tag: Prefix for saved profile filenames.
        env: Comma-separated KEY=VALUE variables for child processes.
        sample_rows: Keep the first rows of each input table; zero keeps all rows.
        backend: Engine backend, such as quail or stock_vllm.
        device: Registered device name. Defaults to ``h100-sxm``.

    Returns:
        Collected RESULT lines from all query processes.
    """
    env_items = [item for item in env.split(",") if item]
    # (name, source root, extra child variables) per code version
    versions = [("after", "/root", {})]
    if Path(BEFORE_REMOTE, "quail").is_dir():
        versions.insert(0, ("before", BEFORE_REMOTE, {}))
    sweep = dict(item.split("=", 1) for item in env_items).get("QUAIL_SWEEP")
    if sweep:
        # one version per override set, all in this container, so the
        # settings share one GPU and host
        versions = [(overrides, "/root", {"QUAIL_OVERRIDES": overrides})
                    for overrides in sweep.split("|")]
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    lines = []
    for repeat in range(repeats):
        order = versions if repeat % 2 == 0 else versions[::-1]
        for query_id in query_ids:
            for name, root, extra in order:
                prefix = (f"{PROFILE_DIR}/{tag}-{query_id}-{model}-"
                          f"{name.replace('/', '_').replace(':', '_')}-r{repeat}")
                arguments = json.dumps(
                    dict(
                        query_id=query_id,
                        scale_factor=scale_factor,
                        collection_id=collection_id,
                        model=model,
                        device=device,
                        backend=backend,
                        modes=modes,
                        top=top,
                        stats_prefix=prefix,
                        sample_rows=sample_rows,
                    )
                )
                command = [sys.executable, "-c",
                           "import sys; from experiments.cells import profile_query; "
                           "profile_query.child(sys.argv[1])", arguments]
                child_env = {**os.environ,
                       **dict(item.split("=", 1) for item in env_items), **extra,
                       "PYTHONPATH": f"{root}:/root:" + os.environ.get(
                           "PYTHONPATH", "")}
                started = time.perf_counter()
                done = subprocess.run(command, env=child_env, cwd="/tmp",
                                      capture_output=True, text=True)
                print(done.stdout[-60000:], flush=True)
                if done.returncode:
                    print(done.stderr[-8000:], flush=True)
                    raise RuntimeError(f"{query_id} on {name} failed")
                for line in done.stdout.splitlines():
                    if line.startswith("RESULT "):
                        record = json.loads(line[7:])
                        record.update(version=name, repeat=repeat)
                        lines.append(json.dumps(record))
                        # saved as each finishes, so a lost client loses nothing
                        with open(f"{PROFILE_DIR}/{tag}.jsonl", "a") as saved:
                            saved.write(lines[-1] + "\n")
                        results_vol.commit()
                print(f"{query_id} on {name}: process "
                      f"{time.perf_counter() - started:.1f} s", flush=True)
    results_vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def profile(queries: str = "BIO-5,FEV-11", scale_factor: float = 0.1,
            model: str = "qwen3-4b-fp8", device: str = "h100-sxm",
            modes: str = "plain,cprofile", top: int = 30, repeats: int = 1,
            tag: str = "window", env: str = "", sample_rows: int = 0,
            backend: str = "quail"):
    if sample_rows < 0:
        raise ValueError("sample_rows must be nonnegative")
    query_ids = [query.strip() for query in queries.split(",") if query.strip()]
    data = ensure_data.spawn(scale_factor, query_ids, "")
    print(f"function call id: {data.object_id} (data)", flush=True)
    collection = data.get()
    call = profile_query.with_options(gpu=modal_gpu_type(device)).spawn(
        query_ids, scale_factor, collection, model, modes.split(","), top,
        repeats, tag, env, sample_rows, backend, device)
    print(f"function call id: {call.object_id} (profile {queries})", flush=True)
    for line in call.get().splitlines():
        print("SUMMARY " + line, flush=True)
