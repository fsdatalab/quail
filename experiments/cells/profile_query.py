"""Profile QUAIL-B query execution on Modal, excluding model startup.

Queries run in fresh processes. Modes are plain (no profiler), cprofile,
sample (Python stack sampling), trace (torch.profiler), and timeline
(executor-step and GPU event timing). Each run prints timing, allocation,
and result-digest fields. The env option sets child-process variables.

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
from pathlib import Path

import modal

from quail.bench.images import gpu_image
from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)

PROFILE_DIR = "/results/ablations/profiles"
# caching allocator counters: a retry flushes the cache after a failed
# cudaMalloc, which waits for every stream
ALLOCATOR_COUNTS = ("num_alloc_retries", "num_sync_all_streams",
                    "num_device_alloc", "num_device_free")
NODE_FIELDS = ("wall_s", "gpu_s", "chunks", "pack_s", "fresh_tokens",
               "suffix_tokens")
BEFORE_REMOTE = "/root/before"
BEFORE_LOCAL = os.environ.get("QUAIL_BEFORE_SOURCE", "")
image = gpu_image(*([(BEFORE_LOCAL, BEFORE_REMOTE)]
                    if modal.is_local() and BEFORE_LOCAL else []))


def _digest(output) -> str:
    """Hash answer tables and result rows in document ID order."""
    import hashlib

    digest = hashlib.sha256()
    tables = {**output.filter_answers, **output.join_answers,
              **(output.classify_answers or {})}
    for name in sorted(tables):
        table = tables[name]
        rows = sorted(json.dumps(list(row.values()), default=str)
                      for row in table.to_pylist())
        digest.update(json.dumps([name, rows]).encode())
    rows = sorted(json.dumps(list(row.values()), default=str)
                  for row in output.rows.to_pylist())
    digest.update(json.dumps(rows).encode())
    return digest.hexdigest()[:16]


def _gpu_gaps(prof, top: int) -> dict:
    """Measure GPU busy and idle time, including the longest idle gaps."""
    from torch.autograd import DeviceType

    events = list(prof.events())
    kernels = sorted((event.time_range.start, event.time_range.end, event.name)
                     for event in events
                     if event.device_type == DeviceType.CUDA)
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
    return {
        "span_s": (end - start) / 1e6, "busy_s": busy / 1e6,
        "idle_s": (end - start - busy) / 1e6,
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
        import quail.backends.quail.executor.loop as loop
        import quail.backends.quail.executor.pack as pack
        import quail.backends.quail.executor.readout as readout

        self.torch = torch
        self.entries = []
        self.active = False
        self._wrap(chunk, "pack_chunk", "pack")
        self._wrap(loop, "_forward", "forward", gpu=True)
        self._wrap(pack.JoinAdmission, "next_chunk", "next_chunk")
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
                      if frame.startswith("run (profile_query.py")]
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


def child(arguments: str) -> None:
    """Run one query several times in a fresh session; print a RESULT line per run.

    Args:
        arguments: JSON of query_id, sf, collection_id, model, modes
            (one per run: plain, cprofile, sample, trace, or timeline),
            top, and
            stats_prefix.
    """
    import cProfile
    import io
    import pstats

    import quail
    import quail.backends.quail.graph as graph_module
    from quail.bench.quailb import run_query
    from quail_b.benchmark import load_benchmark

    args = json.loads(arguments)
    query_id, sf, modes, top = (args["query_id"], args["sf"], args["modes"],
                                args["top"])
    print(f"quail from {quail.__file__}", flush=True)
    window = {}

    class ProfiledRunner(graph_module.GenericRunner):
        def run(self, *args, **kwargs):
            profiler = window.get("profiler")
            traced = window.get("trace")
            timeline = window.get("timeline")
            sampler = window.get("sampler")
            if sampler is not None:
                sampler.start()
            import torch
            before = torch.cuda.memory_stats()
            free_before = torch.cuda.mem_get_info()[0]
            if timeline is not None:
                timeline.start()
            if traced is not None:
                traced.__enter__()
            if profiler is not None:
                profiler.enable()
            try:
                return super().run(*args, **kwargs)
            finally:
                if profiler is not None:
                    profiler.disable()
                if traced is not None:
                    torch.cuda.synchronize()
                    traced.__exit__(None, None, None)
                if timeline is not None:
                    timeline.stop()
                if sampler is not None:
                    sampler.stop()
                after = torch.cuda.memory_stats()
                window["allocator"] = {
                    name: after.get(name, 0) - before.get(name, 0)
                    for name in ALLOCATOR_COUNTS}
                window["allocator"]["free_gib_before"] = round(
                    free_before / 2**30, 2)
                window["allocator"]["reserved_gib_before"] = round(
                    before.get("reserved_bytes.all.current", 0) / 2**30, 2)

    graph_module.GenericRunner = ProfiledRunner
    suite = load_benchmark([query_id], scale_factor=sf,
                           data_dir=Path(DATA_DIR) / f"sf{sf}",
                           collection_id=args["collection_id"] or None)
    spec = suite.queries[0]
    tables = {relation.table: suite.tables[relation.table]
              for relation in spec._info.relations}
    config = quail.EngineConfig(gpus=1, model=args["model"], backend="quail",
                                device="h100-sxm", gpu_timing=True)
    with quail.Session(config) as session:
        for run, mode in enumerate(modes):
            profiler = cProfile.Profile() if mode == "cprofile" else None
            window["profiler"] = profiler
            window["trace"] = None
            window["timeline"] = None
            window["sampler"] = Sampler() if mode == "sample" else None
            if mode == "timeline":
                import torch
                if "recorder" not in window:
                    window["recorder"] = Timeline(torch)
                window["timeline"] = window["recorder"]
            if mode == "trace":
                import torch
                window["trace"] = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA])
            output = run_query(session, spec, tables)
            measures = output.measurements
            nodes = {key: {name: value[name] for name in NODE_FIELDS
                           if name in value}
                     for key, value in measures["node_metrics"].items()}
            record = {
                "query": query_id, "run": run, "mode": mode,
                "quail": quail.__file__,
                "digest": _digest(output),
                **{name: measures.get(name) for name in (
                    "wall_s", "model_wall_s", "gpu_s", "chunks", "frontend_s",
                    "planning_s", "input_ready_s", "physical_prepare_s",
                    "finish_s", "answer_prepare_s", "collection_s",
                    "fresh_tokens", "coordinator_wall_s", "worker_total_s",
                    "kernel_cache_files")},
                "nodes": nodes,
                "allocator": window.get("allocator"),
            }
            if window["trace"] is not None:
                record["gpu_window"] = _gpu_gaps(window["trace"], top)
            if window["timeline"] is not None:
                record["timeline"] = window["timeline"].summary(top)
            if window["sampler"] is not None:
                record["samples"] = window["sampler"].summary(top)
            print("RESULT " + json.dumps(record, default=str), flush=True)
            if profiler is not None:
                path = f"{args['stats_prefix']}-run{run}.prof"
                profiler.dump_stats(path)
                text = io.StringIO()
                stats = pstats.Stats(profiler, stream=text)
                for order in ("tottime", "cumulative"):
                    text.write(f"\n== {query_id} run {run}: top {top} by {order}"
                               f" (stats at {path})\n")
                    stats.sort_stats(order).print_stats(top)
                print(text.getvalue(), flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def profile_query(query_ids: list[str], sf: float, collection_id: str,
                  model: str, modes: list[str], top: int, repeats: int,
                  tag: str, env: str = "") -> str:
    """Profile each query in fresh processes and return its RESULT lines.

    Args:
        query_ids: QUAIL-B query IDs to run.
        sf: Dataset scale factor.
        collection_id: Reference label collection.
        model: Model name.
        modes: Profiling modes, one per run within each fresh process.
        top: Number of entries displayed in profiler summaries.
        repeats: Number of repetitions per query and code version.
        tag: Prefix for saved profile filenames.
        env: Comma-separated KEY=VALUE variables for child processes.

    Returns:
        Collected RESULT lines from all query processes.
    """
    env_items = [item for item in env.split(",") if item]
    versions = [("after", "/root")]
    if Path(BEFORE_REMOTE, "quail").is_dir():
        versions.insert(0, ("before", BEFORE_REMOTE))
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    lines = []
    for repeat in range(repeats):
        order = versions if repeat % 2 == 0 else versions[::-1]
        for query_id in query_ids:
            for name, root in order:
                prefix = (f"{PROFILE_DIR}/{tag}-{query_id}-{model}-{name}"
                          f"-r{repeat}")
                arguments = json.dumps(dict(
                    query_id=query_id, sf=sf, collection_id=collection_id,
                    model=model, modes=modes, top=top, stats_prefix=prefix))
                command = [sys.executable, "-c",
                           "import sys, profile_query; "
                           "profile_query.child(sys.argv[1])", arguments]
                child_env = {**os.environ,
                       **dict(item.split("=", 1) for item in env_items),
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
                print(f"{query_id} on {name}: process "
                      f"{time.perf_counter() - started:.1f} s", flush=True)
    results_vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def profile(queries: str = "BIO-5,FEV-11", sf: float = 0.1,
            model: str = "qwen3-4b-fp8", modes: str = "plain,cprofile",
            top: int = 30, repeats: int = 1, tag: str = "window",
            env: str = ""):
    query_ids = [query.strip() for query in queries.split(",") if query.strip()]
    data = ensure_data.spawn(sf, query_ids, "")
    print(f"function call id: {data.object_id} (data)", flush=True)
    collection = data.get()
    call = profile_query.spawn(query_ids, sf, collection, model,
                               modes.split(","), top, repeats, tag, env)
    print(f"function call id: {call.object_id} (profile {queries})", flush=True)
    for line in call.get().splitlines():
        print("SUMMARY " + line, flush=True)
