"""Does a pipelined extraction answer as the same extraction alone does?

Runs queries over the 45 CUAD contracts of scale factor 0.1 on Qwen3
4B, through the Quail backend, each a pipeline of extractions: A asks
the governing law, the agreement date, and the governing law again; B
the governing law alone; C the agreement date alone; D the governing
law twice; E the governing law then the agreement date. Every query
runs on the tree attention path and again on the unified path, and
each question is compared with the same question alone on its path.
For every contract and question it saves the lines the model
named, the none and best start log probabilities at the cue, and
the answer, under
/results/extract_pipeline_check/.

With --profile, run_profile asks the governing law and the cap on
liability of the 45 contracts, one query each, and times every chunk
of the stage executor in four parts, with the GPU synchronized after
each: packing the chunk on the host, the forward pass, the readout,
and reading the answers and deciding the next tokens. Chunks are
grouped into those that read documents and the answer rounds. The
synchronizing loses the overlap of one chunk's reading with the next
chunk's forward pass, so the parts sum to more than the query time.
Prediction: the read chunks take at least 6 s of CUAD-1's 7.8 s on
run 20261012T055523Z, with at least 4.5 s in the forward pass; the
rounds of the cap on liability take at most 8 chunks and 1.5 s. Rounds
over 2 s mean the per-round cost is the problem; read chunks with over
1 s outside the forward pass mean the host is.

Prediction: the pipeline reuses the document's KV exactly. A's first
question answers as B does on every contract but the few whose start
or lines decision is a near tie moved by chunk composition, with cue
log probabilities within 2 nats of B's; every later question differs
from the same question alone on no more contracts than that. A later
question that differs on many more contracts, with cue log
probabilities moved by tens of nats, means the KV it reads is not the
document's; which of D, E, and A fail, and whether the unified path
fails too, says whether the fault is the rewind, the frame rewrite,
or the tree path's writes.

    uv run modal run --detach -m experiments.cells.extract_pipeline_check
        2>&1 | tee /tmp/extract_pipeline_check.log
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


def _state(stages) -> list:
    """Each document's decisions for each question, from the executor."""
    out = []
    for doc in stages.docs:
        for q, qs in enumerate(doc.questions):
            out.append({
                "document": doc.index, "question": q, "path": qs.path,
                "range": qs.range, "none_lp": qs.none_lp, "best_lp": qs.best_lp,
                "answer": qs.answer, "span": qs.span, "steps": qs.steps,
                "written": len(qs.written)})
    return out


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_check(run: str) -> dict:
    """Run every query and compare the decisions question by question."""
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor import extract as executor
    from quail.backends.quail.executor.operators import extract as operator
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets
    from quail_b.prompts import CUAD_QUESTIONS

    data_dir = build_sets(TABLES, 0.1)
    VOLUMES["/root/.cache/huggingface"].commit()
    contracts = pq.read_table(data_dir / "contracts.parquet")
    captured = []
    original = executor.ExtractStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captured.append(self)

    executor.ExtractStages.__init__ = init
    run_stages = operator.run_stages
    forced = {"attention": None}

    def forced_run_stages(*args, **kwargs):
        if forced["attention"]:
            kwargs["attention_mode"] = forced["attention"]
        return run_stages(*args, **kwargs)

    operator.run_stages = forced_run_stages
    law = CUAD_QUESTIONS["Governing Law"]
    date = CUAD_QUESTIONS["Agreement Date"]
    queries = {"A": [law, date, law], "B": [law], "C": [date],
               "D": [law, law], "E": [law, date]}
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts, id_col="id"))
    results = {}
    for attention, names in (("tree", "ABCDE"), ("unified", "ABCDE")):
        forced["attention"] = attention
        for name in names:
            questions = queries[name]
            query = session.docs("contracts").alias("c")
            for q, question in enumerate(questions):
                query = query.ai_extract("c.text", question, name=f"q{q}")
            query = query.select("c.id", *(f"q{q}" for q in range(len(questions))))
            started = time.perf_counter()
            result = query.run()
            result.collect()
            wall = time.perf_counter() - started
            label = name if attention == "tree" else f"{name}.{attention}"
            results[label] = {"wall_s": wall, "questions": questions,
                              "attention": attention,
                              "fresh_tokens": result.report.get("fresh_tokens"),
                              "state": _state(captured[-1])}
            print(f"{label}: {wall:.1f} s, {result.report.get('fresh_tokens')} "
                  f"fresh tokens", flush=True)
    session.close()

    def rows(label, q):
        return {s["document"]: s for s in results[label]["state"]
                if s["question"] == q}

    pairs = {}
    for path, suffix in (("tree", ""), ("unified", ".unified")):
        standalone = {"law": (f"B{suffix}", 0), "date": (f"C{suffix}", 0)}
        for name, questions in queries.items():
            for q, question in enumerate(questions):
                if q == 0 and name in "BC":
                    continue
                alone = standalone["law" if question == law else "date"]
                pairs[f"{name}{suffix}.q{q} vs {alone[0]}"] = (
                    (f"{name}{suffix}", q), alone)
            if len(questions) > 1 and questions[0] == questions[-1]:
                pairs[f"{name}{suffix}.q{len(questions) - 1} vs q0"] = (
                    (f"{name}{suffix}", len(questions) - 1), (f"{name}{suffix}", 0))
    comparison = {}
    for label, (mine_ref, other_ref) in pairs.items():
        mine, other = rows(*mine_ref), rows(*other_ref)
        comparison[label] = {
            "documents": len(mine),
            "answers_differ": sum(mine[d]["answer"] != other[d]["answer"]
                                  for d in mine),
            "ranges_differ": sum(mine[d]["range"] != other[d]["range"]
                                 for d in mine),
            "max_none_lp_gap": round(max(abs(mine[d]["none_lp"] - other[d]["none_lp"])
                                         for d in mine), 4),
            "max_best_lp_gap": round(max(abs(mine[d]["best_lp"] - other[d]["best_lp"])
                                         for d in mine), 4)}
    summary = {"run": run, "model": MODEL, "comparison": comparison,
               "walls": {label: results[label]["wall_s"] for label in results},
               "fresh_tokens": {label: results[label]["fresh_tokens"]
                                for label in results}}
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_snapshot(run: str, document: int = 1) -> dict:
    """Compare one contract's document and frame KV rows across two askings.

    Runs query D, the governing law twice, on each attention path, and
    snapshots the K rows of the document and the first question's
    frame, on the first and the last layer, right after the first
    asking writes them and right before the second asking reads them.
    """
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor import stages as scheduler
    from quail.backends.quail.executor.operators import extract as operator
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets
    from quail_b.prompts import CUAD_QUESTIONS

    data_dir = build_sets(TABLES, 0.1)
    contracts = pq.read_table(data_dir / "contracts.parquet")
    snapshots = {}
    original = scheduler._StageExecutor._run_one
    per_question = 18       # tail, 7 lines steps, cue, step, 8 passes

    def rows_of(self, a, j):
        key = self.keys[a]
        f = self.prefix_lengths[a] + len(self.frames[j]) + 1
        return key, self.arena.capacity_rows(key)[:f]

    def snapshot(self, a, j, label):
        # the kernel warm-up runs stages of its own, over positional keys
        if len(self.stages) != 2 * per_question:
            return
        key = self.keys[a]
        if not isinstance(key, tuple) or not self.arena.is_resident(key):
            return
        key, rows = rows_of(self, a, j)
        layers = sorted({0, len(self.arena.k) - 1})
        self.torch.cuda.synchronize()
        taken = {}
        for layer in layers:
            k, v = self.arena.layer_kv(layer)
            index = rows.to(k.device)
            taken[layer] = (k.index_select(0, index).clone().cpu(),
                            v.index_select(0, index).clone().cpu())
        snapshots.setdefault(label, []).append((int(rows.numel()), taken))

    def run_one(self, part):
        for a, j, start, end, carried in part:
            if a == document and j == per_question:
                snapshot(self, a, j, "before second asking")
            if a == document and j == 1:
                snapshot(self, a, j, "before first lines step")
        original(self, part)
        for a, j, start, end, carried in part:
            if a == document and j == 0:
                snapshot(self, a, j, "after first asking")

    scheduler._StageExecutor._run_one = run_one
    run_stages = operator.run_stages
    forced = {"attention": None}

    def forced_run_stages(*args, **kwargs):
        if forced["attention"]:
            kwargs["attention_mode"] = forced["attention"]
        return run_stages(*args, **kwargs)

    operator.run_stages = forced_run_stages
    law = CUAD_QUESTIONS["Governing Law"]
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts, id_col="id"))
    report = {}
    for attention in ("tree", "unified"):
        forced["attention"] = attention
        snapshots.clear()
        query = (session.docs("contracts").alias("c")
                 .ai_extract("c.text", law, name="q0")
                 .ai_extract("c.text", law, name="q1")
                 .select("c.id", "q0", "q1"))
        query.run().collect()
        found = {label: len(taken) for label, taken in snapshots.items()}
        compared = {}
        after = snapshots.get("after first asking", [None])[0]
        for label in ("before first lines step", "before second asking"):
            later = snapshots.get(label, [None])[0]
            if after is None or later is None:
                continue
            rows, by_layer = after
            per_layer = {}
            for layer, (k0, v0) in by_layer.items():
                k1, v1 = later[1][layer]
                differ_k = (k0 != k1).any(dim=(1, 2)).nonzero().flatten().tolist()
                differ_v = (v0 != v1).any(dim=(1, 2)).nonzero().flatten().tolist()
                per_layer[layer] = {"rows": rows, "k_rows_differ": differ_k[:20],
                                    "k_differ_count": len(differ_k),
                                    "v_differ_count": len(differ_v)}
            compared[label] = per_layer
        report[attention] = {"snapshots": found, "compared": compared}
        print(attention, json.dumps(report[attention]), flush=True)
    session.close()
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}_snapshot.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **report}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_budgets(run: str) -> dict:
    """Ask the governing law alone under three chunk budgets.

    The budget changes only how the rows are batched. Answers that
    move with it show the extraction's decisions depend on the batch
    the rows share, which the pipeline also changes.
    """
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor import extract as executor
    from quail.cost import budgets
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets
    from quail_b.prompts import CUAD_QUESTIONS

    data_dir = build_sets(TABLES, 0.1)
    contracts = pq.read_table(data_dir / "contracts.parquet")
    captured = []
    original = executor.ExtractStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captured.append(self)

    executor.ExtractStages.__init__ = init
    default_budget = budgets.chunk_budget
    law = CUAD_QUESTIONS["Governing Law"]
    results = {}
    for budget in (None, 60_000, 30_000):
        budgets.chunk_budget = (default_budget if budget is None
                                else lambda model, device, b=budget: b)
        session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                             backend="quail"))
        session.register("contracts", quail.DocumentProvider.from_table(
            contracts, id_col="id"))
        query = (session.docs("contracts").alias("c")
                 .ai_extract("c.text", law, name="q0").select("c.id", "q0"))
        started = time.perf_counter()
        query.run().collect()
        label = "default" if budget is None else str(budget)
        results[label] = {"wall_s": time.perf_counter() - started,
                          "state": _state(captured[-1])}
        print(f"budget {label}: {results[label]['wall_s']:.1f} s", flush=True)
        session.close()
    base = {s["document"]: s for s in results["default"]["state"]}
    comparison = {}
    for label in ("60000", "30000"):
        mine = {s["document"]: s for s in results[label]["state"]}
        comparison[f"{label} vs default"] = {
            "answers_differ": sum(mine[d]["answer"] != base[d]["answer"]
                                  for d in mine),
            "ranges_differ": sum(mine[d]["range"] != base[d]["range"]
                                 for d in mine),
            "max_none_lp_gap": round(max(abs(mine[d]["none_lp"] - base[d]["none_lp"])
                                         for d in mine), 4),
            "max_best_lp_gap": round(max(abs(mine[d]["best_lp"] - base[d]["best_lp"])
                                         for d in mine), 4)}
    summary = {"run": run, "comparison": comparison,
               "walls": {k: v["wall_s"] for k, v in results.items()}}
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}_budgets.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


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
def main(snapshot: bool = False, budgets: bool = False, profile: bool = False,
         kernels: bool = False):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    function = (run_kernels if kernels else run_profile if profile
                else run_snapshot if snapshot else run_budgets if budgets
                else run_check)
    call = function.spawn(run)
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"function call id {call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
