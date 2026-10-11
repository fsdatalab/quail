"""AI.EXTRACT on a GPU: the executor and the vLLM baseline on the measured items.

Runs the operator through a Quail session, on the Quail backend and on
stock vLLM, over the same 300 SQuAD and 300 CUAD items the measurement
cell (extract_spans.py) scored, one query per distinct question, and
compares the answers with that cell's saved run by item id.

Run 20261011T050258Z, with the cell's span scoring, gave 80.1 F1 on
SQuAD (cell 81.8, stock vLLM 80.3) and 32.8 on CUAD with trim and 41.0
without (cell 33.6, stock vLLM 52.0): the scored spans started late or
stopped early on CUAD's clauses, 32 words at the median.

Prediction for the decoder rule, which starts at the model's likeliest
start and ends a span where stopping beats copying: CUAD trim F1
within 5 points of stock vLLM's 52.0, SQuAD within 2 points of 80.1,
and fewer fresh tokens on CUAD than that run's 768,598, as one start
is fed instead of up to eight. Stock vLLM is unchanged.

    uv run modal run --detach -m experiments.cells.extract_check \
        2>&1 | tee /tmp/extract_check.log

Results go to /results/extract_check/<run>_<backend>_<dataset>_<trim>.json
on the quail-results volume.
"""

import json
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
# the measurement cell's final runs, the ones with the template prompts
CELL_RUNS = {"squad": "20261011T010106Z", "cuad": "20261011T005925Z"}
N_ITEMS = 300


def _groups(questions) -> dict:
    """Group the items by question text: one query asks one question."""
    groups = {}
    for item in questions:
        groups.setdefault(item["question"], []).append(item)
    return groups


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


@app.function(image=image, gpu="H100!", memory=65536, timeout=7200,
              volumes=VOLUMES)
def check(dataset: str, backend: str, trim: bool, n: int, run: str) -> dict:
    """Answer every item through the session and score the answers."""
    import os

    import pyarrow as pa

    import quail
    from experiments.cells.extract_spans import _f1, _questions
    from quail.planner.plan import EngineConfig, Refusal

    questions = _questions(dataset, n)
    cell_path = f"/results/extract_spans/{CELL_RUNS[dataset]}_{MODEL}_{dataset}.json"
    with open(cell_path) as f:
        cell = {item["id"]: item for item in json.load(f)["items"]}
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm",
                                         backend=backend))
    items = []
    queries = []
    started = time.perf_counter()
    for index, (question, members) in enumerate(_groups(questions).items()):
        name = f"docs{index}"
        session.register(name, quail.DocumentProvider.from_table(pa.table({
            "id": [m["id"] for m in members],
            "text": [m["context"] for m in members]}), id_col="id"))
        query = (session.docs(name).alias("d")
                 .ai_extract("d.text", question, name="answer", trim=trim)
                 .select("d.id", "answer"))
        plan = query.plan()
        if isinstance(plan, Refusal):
            raise RuntimeError(f"refused: {plan.reasons}")
        result = query.run()
        rows = result.collect().to_pydict()
        answers = dict(zip(rows["d.id"], rows["answer"]))
        spans = dict(zip(rows["d.id"], rows["answer_span"]))
        report = result.report
        metrics = report.get("node_metrics", {})
        extract = next((m for k, m in metrics.items()
                        if k.startswith("ai-extract")), {})
        queries.append({
            "question": question, "documents": len(members),
            "wall_s": report["wall_s"], "estimated_s": plan.estimated_seconds,
            "fresh_tokens": report.get("fresh_tokens"),
            "cached_tokens": report.get("cached_tokens"),
            "extension": {k: v for k, v in extract.items()
                          if isinstance(v, (int, float, str))}})
        for m in members:
            answer = answers.get(m["id"])
            em, f1 = _f1(answer or "", m["answers"])
            saved = cell.get(m["id"], {})
            cell_answer = saved.get("scored", {}).get("k8", {})
            items.append({
                "id": m["id"], "answer": answer, "span": spans.get(m["id"]),
                "em": em, "f1": f1, "cell_path": saved.get("path"),
                "cell_answer": cell_answer.get("text"),
                "cell_em_f1": cell_answer.get("em_f1"),
                "same_as_cell": (answer or "").strip() == (
                    cell_answer.get("text") or "").strip()})
        print(f"{index + 1}/{len(_groups(questions))} {dataset} {backend} "
              f"{report['wall_s']:.1f} s, f1 so far "
              f"{_mean(i['f1'] for i in items):.3f}", flush=True)
    seconds = time.perf_counter() - started
    session.close()
    by_path = {}
    for path in ("located", "scored", "fallback"):
        subset = [i for i in items if i["cell_path"] == path]
        if subset:
            by_path[path] = {
                "n": len(subset), "em": 100 * _mean(i["em"] for i in subset),
                "f1": 100 * _mean(i["f1"] for i in subset),
                "cell_f1": 100 * _mean(i["cell_em_f1"][1] for i in subset
                                       if i["cell_em_f1"]),
                "same_as_cell": _mean(i["same_as_cell"] for i in subset)}
    summary = {
        "run": run, "dataset": dataset, "backend": backend, "trim": trim,
        "model": MODEL, "n": len(items), "queries": len(queries),
        "seconds": seconds,
        "query_wall_s": sum(q["wall_s"] for q in queries),
        "estimated_s": sum(q["estimated_s"] for q in queries),
        "fresh_tokens": sum(q["fresh_tokens"] or 0 for q in queries),
        "em": 100 * _mean(i["em"] for i in items),
        "f1": 100 * _mean(i["f1"] for i in items),
        "cell_f1_k8": 100 * _mean(i["cell_em_f1"][1] for i in items
                                  if i["cell_em_f1"]),
        "same_as_cell": _mean(i["same_as_cell"] for i in items),
        "null_answers": sum(i["answer"] is None for i in items),
        "by_cell_path": by_path,
        "cell_run": CELL_RUNS[dataset],
    }
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs("/results/extract_check", exist_ok=True)
    path = (f"/results/extract_check/{run}_{backend}_{dataset}_"
            f"{'trim' if trim else 'lines'}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "queries": queries, "items": items},
                  f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.local_entrypoint()
def main(n: int = N_ITEMS, backends: str = "quail,stock_vllm"):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    calls = {}
    for dataset, backend, trim in (("squad", "quail", True),
                                   ("cuad", "quail", True),
                                   ("cuad", "quail", False),
                                   ("squad", "stock_vllm", True),
                                   ("cuad", "stock_vllm", True)):
        if backend not in backends.split(","):
            continue
        call = check.spawn(dataset, backend, trim, n, run)
        calls[(dataset, backend, trim)] = call
        print(f"{dataset} {backend} trim={trim}: function call id "
              f"{call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
