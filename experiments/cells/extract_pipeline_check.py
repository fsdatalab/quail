"""Does a pipelined extraction answer as the same extraction alone does?

Runs two queries over the 45 CUAD contracts of scale factor 0.1 on
Qwen3 4B, through the Quail backend: A asks the governing law, the
agreement date, and the governing law again, as one pipeline; B asks
the governing law alone. For every contract and question it saves
the lines the model named, the none and best start log probabilities
at the cue, and the answer, under /results/extract_pipeline_check/.

Prediction: the pipeline reuses the document's KV exactly. A's first
question answers as B does on every contract but the few whose start
or lines decision is a near tie moved by chunk composition, with cue
log probabilities within 0.1 of B's; A's third question, over KV the
two questions before it wrote after and rewound, differs from B on no
more contracts than the first does. A third question that differs on
many more contracts, or whose cue log probabilities move by whole
nats, means the rewound KV is not the document's.

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
            top = sorted(qs.cue_lp.items(), key=lambda item: -item[1])[:5]
            out.append({
                "document": doc.index, "question": q, "path": qs.path,
                "range": qs.range, "none_lp": qs.none_lp, "best_lp": qs.best_lp,
                "top_cue": [[int(t), round(v, 4)] for t, v in top],
                "answer": qs.answer, "span": qs.span, "passes": qs.passes})
    return out


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run_check(run: str) -> dict:
    """Run both queries and compare the decisions question by question."""
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor import extract as executor
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
    law = CUAD_QUESTIONS["Governing Law"]
    date = CUAD_QUESTIONS["Agreement Date"]
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend="quail"))
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts, id_col="id"))
    results = {}
    for name, questions in (("A", [law, date, law]), ("B", [law])):
        query = session.docs("contracts").alias("c")
        for q, question in enumerate(questions):
            query = query.ai_extract("c.text", question, name=f"q{q}")
        query = query.select("c.id", *(f"q{q}" for q in range(len(questions))))
        started = time.perf_counter()
        result = query.run()
        result.collect()
        wall = time.perf_counter() - started
        stages = captured[-1]
        results[name] = {"wall_s": wall, "questions": questions,
                         "fresh_tokens": result.report.get("fresh_tokens"),
                         "state": _state(stages)}
        print(f"{name}: {wall:.1f} s, {result.report.get('fresh_tokens')} "
              f"fresh tokens", flush=True)
    session.close()

    def rows(name, q):
        return {s["document"]: s for s in results[name]["state"]
                if s["question"] == q}

    b = rows("B", 0)
    comparison = {}
    for label, (name, q) in {"A.q0 vs B": ("A", 0), "A.q2 vs B": ("A", 2),
                             "A.q2 vs A.q0": ("A", 2)}.items():
        other = rows("A", 0) if label == "A.q2 vs A.q0" else b
        mine = rows(name, q)
        answers = sum(mine[d]["answer"] != other[d]["answer"] for d in mine)
        ranges = sum(mine[d]["range"] != other[d]["range"] for d in mine)
        none_gap = max(abs(mine[d]["none_lp"] - other[d]["none_lp"]) for d in mine)
        best_gap = max(abs(mine[d]["best_lp"] - other[d]["best_lp"]) for d in mine)
        comparison[label] = {"documents": len(mine), "answers_differ": answers,
                             "ranges_differ": ranges,
                             "max_none_lp_gap": round(none_gap, 4),
                             "max_best_lp_gap": round(best_gap, 4)}
    summary = {"run": run, "model": MODEL, "comparison": comparison,
               "walls": {name: results[name]["wall_s"] for name in results},
               "fresh_tokens": {name: results[name]["fresh_tokens"]
                                for name in results}}
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs("/results/extract_pipeline_check", exist_ok=True)
    path = f"/results/extract_pipeline_check/{run}.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.local_entrypoint()
def main():
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = run_check.spawn(run)
    print(f"function call id {call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
