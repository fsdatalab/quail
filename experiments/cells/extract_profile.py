"""Where an extraction's time goes: each chunk's parts, and one round's kernels.

run_profile asks the governing law and the cap on liability of the 45
CUAD contracts of scale factor 0.1 on Qwen3 4B, one query each, and
times every chunk of the stage executor in four parts, with the GPU
synchronized after each: packing the chunk on the host, the forward
pass, the readout, and reading the answers and deciding the next
tokens. Chunks are grouped into those that read documents and the
answer rounds. The synchronizing loses the overlap of one chunk's
reading with the next chunk's forward pass, so the parts sum to more
than the query time. Run 20261012T055945Z measured, for the cap on
liability, 4.5 s of forward passes over 4 read chunks, 4.2 s on the
host reading those chunks' answers, and 1.6 s of forward passes over
13 rounds.

With --kernels, run_kernels runs the second read chunk and the third
round of the cap on liability under the torch profiler and saves each
one's kernels by GPU time and operators by host time. Run
20261012T060259Z measured a read chunk of 110 thousand tokens at 1.5
s, the attention 42% and the fp8 matrix multiplies 39% of its
kernels, and a round of 272 rows at 44 ms wall with 28 ms of kernels.

Results are saved under /results/extract_pipeline_check/.

    uv run modal run --detach -m experiments.cells.extract_profile [--kernels]
        2>&1 | tee /tmp/extract_profile.log
"""

import json
import os
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image(("experiments", "/root/experiments"))
except ImportError:    # a container without the local quail package
    image = None

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


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_profile(run: str) -> dict:
    """Time each chunk's packing, forward pass, readout, and deciding."""
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor import loop
    from quail.backends.quail.executor import stages as scheduler
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets
    from quail_b.prompts import CUAD_QUESTIONS

    data_dir = build_sets(TABLES, 0.1)
    contracts = pq.read_table(data_dir / "contracts.parquet")
    chunks = []          # one record per chunk, in order
    current = {}
    executor = scheduler._StageExecutor
    original = {"build": executor._build, "submit": executor._submit,
                "report": executor._report, "run_one": executor._run_one,
                "forward": loop._forward}

    def timed(name):
        def wrap(self, *args, **kwargs):
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            out = original[name](self, *args, **kwargs)
            self.torch.cuda.synchronize()
            current[name] = time.perf_counter() - started
            return out
        return wrap

    def build(self, part):
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        chunk, entries = original["build"](self, part)
        current.clear()
        current["build"] = time.perf_counter() - started
        current["kind"] = ("read" if any(carried for *_, carried in part)
                           else "round")
        # the kernel warm-up runs stages of its own, over positional keys
        current["warm"] = not isinstance(self.keys[0], tuple)
        current["tokens"] = int(chunk.tokens)
        current["rows"] = int(sum(rows for _, rows in entries))
        current["stages"] = sorted({int(j) for _, j, *_ in part})
        return chunk, entries

    def forward(pipeline, arena, chunk):
        import torch

        torch.cuda.synchronize()
        started = time.perf_counter()
        out = original["forward"](pipeline, arena, chunk)
        torch.cuda.synchronize()
        current["forward"] = time.perf_counter() - started
        return out

    def run_one(self, part):
        original["run_one"](self, part)
        chunks.append(dict(current))

    def report(self, entry):
        started = time.perf_counter()
        original["report"](self, entry)
        # the entry's chunk is the last recorded one that lacks a report
        for record in reversed(chunks):
            if "report" not in record:
                record["report"] = time.perf_counter() - started
                break

    executor._build = build
    executor._submit = timed("submit")
    executor._report = report
    executor._run_one = run_one
    loop._forward = forward
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts, id_col="id"))
    results = {}
    for label, category in (("governing law", "Governing Law"),
                            ("cap on liability", "Cap On Liability")):
        chunks.clear()
        query = (session.docs("contracts").alias("c")
                 .ai_extract("c.text", CUAD_QUESTIONS[category], name="q0")
                 .select("c.id", "q0"))
        started = time.perf_counter()
        query.run().collect()
        wall = time.perf_counter() - started
        own = [c for c in chunks if c.get("tokens", 0) > 0 and not c["warm"]]
        summary = {}
        for kind in ("read", "round"):
            mine = [c for c in own if c["kind"] == kind]
            summary[kind] = {
                "chunks": len(mine),
                "tokens": sum(c["tokens"] for c in mine),
                "rows": sum(c["rows"] for c in mine),
                **{part: round(sum(c.get(part, 0.0) for c in mine), 3)
                   for part in ("build", "forward", "submit", "report")}}
        results[label] = {"wall_s": round(wall, 3), "summary": summary,
                          "chunks": own}
        print(label, json.dumps({"wall_s": round(wall, 3), **summary}),
              flush=True)
    session.close()
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}_profile.json"
    with open(path, "w") as f:
        json.dump({"run": run, "model": MODEL, "results": results}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **{k: {"wall_s": v["wall_s"], **v["summary"]}
                             for k, v in results.items()}}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_kernels(run: str) -> dict:
    """Profile the kernels of one read chunk and one answer round.

    Asks the cap on liability of the 45 contracts and runs the second
    read chunk and the third round under the torch profiler. Saves
    each one's kernels by GPU time and its operators by host time.
    Prediction: a round's kernels sum to under 30 ms of its 126 ms, and
    its host-side operator time fills the rest, so the rounds are
    launch-bound and CUDA graphs would cut them; a single kernel over
    half the round means that kernel is.
    """
    import pyarrow.parquet as pq
    import torch
    from torch.profiler import ProfilerActivity, profile

    import quail
    from quail.backends.quail.executor import loop
    from quail.backends.quail.executor import stages as scheduler
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets
    from quail_b.prompts import CUAD_QUESTIONS

    data_dir = build_sets(TABLES, 0.1)
    contracts = pq.read_table(data_dir / "contracts.parquet")
    executor = scheduler._StageExecutor
    original = {"build": executor._build, "forward": loop._forward}
    seen = {"read": 0, "round": 0}
    current = {}
    traces = {}

    def build(self, part):
        chunk, entries = original["build"](self, part)
        if isinstance(self.keys[0], tuple):
            kind = "read" if any(carried for *_, carried in part) else "round"
            seen[kind] += 1
            current["label"] = f"{kind} {seen[kind]}"
            current["tokens"] = int(chunk.tokens)
            current["rows"] = int(sum(rows for _, rows in entries))
        else:
            current.clear()
        return chunk, entries

    def forward(pipeline, arena, chunk):
        label = current.get("label")
        if label not in ("read 2", "round 3"):
            return original["forward"](pipeline, arena, chunk)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            out = original["forward"](pipeline, arena, chunk)
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        averages = prof.key_averages()
        by_gpu = averages.table(sort_by="cuda_time_total", row_limit=30)
        by_cpu = averages.table(sort_by="cpu_time_total", row_limit=30)
        kernel_s = sum(item.self_device_time_total for item in averages) / 1e6
        traces[label] = {"wall_s": round(wall, 4), "kernel_s": round(kernel_s, 4),
                         "tokens": current["tokens"], "rows": current["rows"],
                         "by_gpu": by_gpu, "by_cpu": by_cpu}
        print(label, f"{wall:.3f} s wall, {kernel_s:.3f} s of kernels",
              flush=True)
        print(by_gpu, flush=True)
        return out

    executor._build = build
    loop._forward = forward
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts, id_col="id"))
    query = (session.docs("contracts").alias("c")
             .ai_extract("c.text", CUAD_QUESTIONS["Cap On Liability"], name="q0")
             .select("c.id", "q0"))
    query.run().collect()
    session.close()
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}_kernels.json"
    with open(path, "w") as f:
        json.dump({"run": run, "model": MODEL, "traces": traces}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **{label: {k: v for k, v in trace.items()
                                     if k not in ("by_gpu", "by_cpu")}
                             for label, trace in traces.items()}}


@app.local_entrypoint()
def main(kernels: bool = False):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    function = run_kernels if kernels else run_profile
    call = function.spawn(run)
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"function call id {call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
