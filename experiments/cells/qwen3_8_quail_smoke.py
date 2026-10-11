"""Run a filter and a join on Qwen3.8 27B through the Quail backend.

The filter asks two planted questions of 200 synthetic documents; the
join matches 12 reports to 36 candidates by a planted color. Both
read the answers off the state Quail saves for each document: the
filter's second stage starts from the kept state, and the join's
partners start from the anchor's. The cell prints each query's
explain and compares the rows with the planted answers.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/qwen3_8_quail_smoke.py \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/qwen3_8-quail-smoke.log

Writes /results/ablations/qwen3_8_27b_quail_smoke_<run id>.json.
"""

import json
import time
import uuid

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

MODEL = "qwen3.8-27b-fp8"
SEED = 20260818
DOCUMENTS = 200
RATES = (0.6, 0.5)
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")
FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
            "then FLAG_9 has value FALSE.\nInstruction: output only the value "
            "of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}=")
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")
REPORTS, CANDIDATES = 12, 36


def _write_filter_corpus(path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(SEED)
    flags = rng.random((DOCUMENTS, len(RATES))) < np.array(RATES)
    bodies = []
    for row in flags:
        line = " ".join(f"FLAG_{j + 1}={'TRUE' if flag else 'FALSE'}"
                        for j, flag in enumerate(row))
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({"id": [f"d{i}" for i in range(DOCUMENTS)],
                             "body": bodies}), path)
    return flags


def _write_join_corpora(reports_path, candidates_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = len(COLORS)
    reports = [FILLER * 30 + f"\n\nThe dominant color in this scene is "
               f"{COLORS[i % keys]}." for i in range(REPORTS)]
    candidates = [f"The candidate color is {COLORS[j % keys]}."
                  for j in range(CANDIDATES)]
    pq.write_table(pa.table({"id": [f"r{i}" for i in range(REPORTS)],
                             "report": reports}), reports_path)
    pq.write_table(pa.table({"id": [f"c{j}" for j in range(CANDIDATES)],
                             "body": candidates}), candidates_path)
    return {(f"r{i}", f"c{j}") for i in range(REPORTS) for j in range(CANDIDATES)
            if i % keys == j % keys}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3 * 3600,
              volumes=volumes)
def smoke(run_id: str) -> dict:
    """Run the filter and the join on the Quail backend and save the summary."""
    import tempfile
    from pathlib import Path

    import quail
    from quail.planner.plan import EngineConfig

    tmp = tempfile.mkdtemp()
    flags = _write_filter_corpus(f"{tmp}/docs.parquet")
    pairs = _write_join_corpora(f"{tmp}/reports.parquet", f"{tmp}/cands.parquet")
    session = quail.Session(EngineConfig(
        model=MODEL, backend="quail", device="h100-sxm", gpus=1))
    for name, column in (("docs", "body"), ("reports", "report"),
                         ("cands", "body")):
        session.register(name, quail.DocumentProvider.from_parquet(
            f"{tmp}/{name}.parquet", id_col="id"))
    summary = {"run_id": run_id, "model": MODEL, "backend": "quail"}

    q1 = FILTER_Q.replace("{j}", "1")
    q2 = FILTER_Q.replace("{j}", "2")
    query = session.sql(f"""
        SELECT d.id FROM docs d
        WHERE AI_FILTER(PROMPT('{{0}}{q1}', d.body), {{'selectivity': 0.6}})
          AND AI_FILTER(PROMPT('{{0}}{q2}', d.body), {{'selectivity': 0.5}})
    """)
    print(query.explain(), flush=True)
    started = time.perf_counter()
    result = query.run()
    planted = {f"d{i}" for i in range(DOCUMENTS) if flags[i].all()}
    got = {row[0] for row in result.to_rows()}
    summary["filter"] = dict(
        seconds_with_boot=time.perf_counter() - started,
        planted_survivors=len(planted), rows=len(got),
        agree_with_planted=len(got & planted), report=result.report)
    print(json.dumps({k: v for k, v in summary["filter"].items()
                      if k != "report"}), flush=True)

    join = (session.docs("reports").alias("r")
            .ai_join(session.docs("cands").alias("c"),
                     quail.prompt(
                         "Judge strictly from {0} whether it says its "
                         "dominant color is the color named in {1}. "
                         "Answer TRUE if it does, FALSE otherwise."
                         "\nANSWER=",
                         quail.col("r.report"), quail.col("c.body")),
                     selectivity=1 / 6)
            .select("r.id", "c.id"))
    print(join.explain(), flush=True)
    started = time.perf_counter()
    matched = set(join.run().to_rows())
    summary["join"] = dict(
        seconds=time.perf_counter() - started, pairs_returned=len(matched),
        planted_pairs=len(pairs), agree_with_planted=len(matched & pairs))
    print(json.dumps(summary["join"]), flush=True)
    session.close()
    path = Path(f"/results/ablations/qwen3_8_27b_quail_smoke_{run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    results_vol.commit()
    summary["path"] = str(path)
    return summary


@app.local_entrypoint()
def main(prediction: str):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    call = smoke.spawn(run_id)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2, default=str), flush=True)
