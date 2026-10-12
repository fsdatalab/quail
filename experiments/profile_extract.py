"""Where an extraction's decode idles: GPU bubbles and the host between them.

Runs one CUAD extraction query over the 45 contracts of scale factor
0.1 on Qwen3 4B through the Quail backend, twice: once unprofiled for
the wall the summary cites, and once under the torch profiler (CPU
and CUDA activities) with py-spy sampling the process. The stage
executor's chunks and answer reads are marked in the trace, so the
summary gives, for the read chunks and the answer rounds apart: the
wall, the GPU-active time, the idle share, and the longest gaps with
the host operations that ran during them. py-spy's samples give the
host functions by self time over the whole query.

Prediction: the answer rounds are about 35% idle, as run
20261012T060259Z measured one round at 44 ms wall with 28 ms of
kernels; the idle time is the per-layer kernel launches of the
decoder, which the forward pass issues from Python one layer at a
time, and the answers' reading and deciding between rounds; the read
chunks are under 10% idle, with a gap between chunks while the next
one is packed. py-spy's top self-time functions outside the forward
pass are the first-token scan and the stream tokenization.

    uv run modal run --detach -m experiments.profile_extract --query CUAD-3 \
        2>&1 | tee /tmp/profile_extract.log

Outputs on the quail-results volume:

    /results/ablations/extract-profile-<run>/<query>.json
    /results/ablations/extract-profile-<run>/<query>.chrome.json.gz
    /results/ablations/extract-profile-<run>/<query>.pyspy.txt
"""

import gzip
import json
import os
import subprocess
import time
from collections import Counter

import modal

from quail.bench.images import gpu_image

image = gpu_image(packages=("py-spy",))
app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
MODEL = "qwen3-4b-fp8"
TABLES = "/root/.cache/huggingface/quail-b-tables"
GAP_US = 500            # a GPU gap shorter than this is not a bubble


def _marked_executor(scheduler, torch):
    """Wrap the stage executor so the trace marks each chunk and each read."""
    original = {"run_one": scheduler._StageExecutor._run_one,
                "report": scheduler._StageExecutor._report}

    def run_one(self, part):
        kind = "read" if any(carried for *_, carried in part) else "round"
        if not isinstance(self.keys[0], tuple):
            kind = "warm"
        with torch.profiler.record_function(f"chunk:{kind}"):
            original["run_one"](self, part)

    def report(self, entry):
        with torch.profiler.record_function("report"):
            original["report"](self, entry)

    scheduler._StageExecutor._run_one = run_one
    scheduler._StageExecutor._report = report


def _query(session, query_id):
    """Build the benchmark query on the session."""
    from quail.bench import quailb
    from quail_b import get_query

    return quailb.build_query(session, get_query(query_id))


def _events(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _summarize(events):
    """Return per chunk kind the wall, the GPU-active time, and the gaps."""
    gpu = _merge([(e["ts"], e["ts"] + e["dur"]) for e in events
                  if e.get("ph") == "X"
                  and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")])
    marks = [e for e in events if e.get("ph") == "X"
             and e.get("cat") == "user_annotation"]
    cpu = [e for e in events if e.get("ph") == "X"
           and e.get("cat") in ("cpu_op", "user_annotation")]
    summary = {}
    for mark in marks:
        name = mark["name"]
        if not name.startswith("chunk:") or name == "chunk:warm":
            continue
        kind = name.split(":")[1]
        left, right = mark["ts"], mark["ts"] + mark["dur"]
        active = sum(min(right, b) - max(left, a) for a, b in gpu
                     if b > left and a < right)
        # the gaps: GPU-idle stretches inside the mark
        gaps = []
        at = left
        for a, b in gpu:
            if b <= left:
                continue
            if a >= right:
                break
            if a - at >= GAP_US:
                gaps.append((at, a))
            at = max(at, b)
        if right - at >= GAP_US:
            gaps.append((at, right))
        own = summary.setdefault(kind, {
            "chunks": 0, "wall_us": 0.0, "gpu_us": 0.0, "gaps": []})
        own["chunks"] += 1
        own["wall_us"] += mark["dur"]
        own["gpu_us"] += active
        for a, b in gaps:
            # the host operations running during the gap, longest first
            during = Counter()
            for e in cpu:
                s, t = e["ts"], e["ts"] + e["dur"]
                if t > a and s < b and e["name"] != name:
                    during[e["name"]] += min(b, t) - max(a, s)
            own["gaps"].append({
                "us": round(b - a), "at_us": round(a - left),
                "host": [(n, round(v)) for n, v in during.most_common(4)]})
    for kind, own in summary.items():
        own["idle_share"] = round(1 - own["gpu_us"] / max(own["wall_us"], 1), 3)
        own["gaps"].sort(key=lambda gap: -gap["us"])
        own["gap_us_total"] = sum(gap["us"] for gap in own["gaps"])
        own["gaps"] = own["gaps"][:12]
        own["wall_us"] = round(own["wall_us"])
        own["gpu_us"] = round(own["gpu_us"])
    reads = [e for e in marks if e["name"] == "report"]
    summary["report"] = {"count": len(reads),
                         "wall_us": round(sum(e["dur"] for e in reads))}
    return summary


def _pyspy_top(path, limit=25):
    """Return the functions by self samples from a py-spy raw dump."""
    self_time, total = Counter(), Counter()
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if not line or " " not in line:
                continue
            stack, count = line.rsplit(" ", 1)
            frames = stack.split(";")
            count = int(count)
            self_time[frames[-1]] += count
            for frame in set(frames):
                total[frame] += count
    samples = sum(self_time.values())
    return {"samples": samples,
            "self": [(name, count) for name, count in self_time.most_common(limit)],
            "total": [(name, count) for name, count in total.most_common(limit)]}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def profile(run: str, query_id: str) -> dict:
    """Profile one extraction query; save the trace, the samples, and a summary."""
    import pyarrow.parquet as pq
    import torch

    import quail
    from quail.backends.quail.executor import stages as scheduler
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets

    out = f"/results/ablations/extract-profile-{run}"
    os.makedirs(out, exist_ok=True)
    data_dir = build_sets(TABLES, 0.1)
    VOLUMES["/root/.cache/huggingface"].commit()
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    for name in ("contracts", "clause_types"):
        session.register(name, quail.DocumentProvider.from_table(
            pq.read_table(data_dir / f"{name}.parquet"), id_col="id"))
    _marked_executor(scheduler, torch)
    # an unprofiled pass gives the wall the summary cites and warms the kernels
    started = time.perf_counter()
    _query(session, query_id).run().collect()
    wall = time.perf_counter() - started
    print(f"{query_id}: {wall:.2f} s unprofiled", flush=True)
    # the profiled pass: the torch profiler and py-spy sampling this process
    raw = f"{out}/{query_id}.pyspy.txt"
    spy = subprocess.Popen(
        ["py-spy", "record", "--pid", str(os.getpid()), "--rate", "250",
         "--format", "raw", "--nonblocking", "--output", raw],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    started = time.perf_counter()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
        _query(session, query_id).run().collect()
        torch.cuda.synchronize()
    profiled_wall = time.perf_counter() - started
    spy.send_signal(2)      # SIGINT ends the recording and writes the file
    try:
        _, spy_err = spy.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        spy.kill()
        spy_err = b"py-spy did not stop"
    trace = f"{out}/{query_id}.chrome.json.gz"
    prof.export_chrome_trace(trace)
    session.close()
    summary = {"run": run, "query": query_id, "wall_s": round(wall, 3),
               "profiled_wall_s": round(profiled_wall, 3),
               "chunks": _summarize(_events(trace))}
    if os.path.exists(raw) and os.path.getsize(raw):
        summary["pyspy"] = _pyspy_top(raw)
    else:
        summary["pyspy_error"] = spy_err.decode(errors="replace")[-2000:]
    with open(f"{out}/{query_id}.json", "w") as stream:
        json.dump(summary, stream, indent=1)
    VOLUMES["/results"].commit()
    print(json.dumps(summary, indent=1)[:6000], flush=True)
    return summary


@app.local_entrypoint()
def main(query: str = "CUAD-3"):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    calls = {query_id: profile.spawn(run, query_id)
             for query_id in query.split(",")}
    for query_id, call in calls.items():
        print(f"{query_id}: function call id {call.object_id}", flush=True)
    print(f"run {run}; results under /results/ablations/extract-profile-{run}/",
          flush=True)
