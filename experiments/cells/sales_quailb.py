r"""The QUAIL-B SALES queries on Decision-2.0-Kai-0.6B.

The five SALES queries run through Quail's QUAIL-B runner (`run_query`) at
scale factor 1.0: 10,088 CRMArena-Pro sales calls on 3,460 deals. Their
reference labels are pending and the calls carry no labels of their own,
so the summary gives answer counts and rates per query, and the answer
tables are saved for reading by hand.

The SALES queries are on the quail-bench branch, not the pinned release,
so the cell mounts a local quail-bench checkout named by QUAIL_BENCH_DIR
and puts it first on the import path.

The table is built on the CPU with the branch's builder, then copied to
the volume:

    uv run modal volume put quail-results sales_calls.parquet \
        /sales/inputs/sf1.0/sales_calls.parquet

Prediction:

- SALES-1 flags 40% to 70% of calls; B2C calls, which compare dealers
  more often, are flagged more than B2B calls.
- SALES-2 puts price first and support or service second; no concern is
  under 10% of calls.
- SALES-3 answers TRUE for 50% to 80% of the 6,628 next-call pairs.
- SALES-4: the discount filter keeps 40% to 70% of calls, and 5% to 20%
  of those also commit to buying.
- SALES-5's 25 rows are calls where the customer agrees to terms or asks
  for the contract.
- Each query finishes in under 2 minutes after startup.

    QUAIL_BENCH_DIR=../quail-bench uv run modal run --detach \
        experiments/cells/sales_quailb.py 2>&1 | tee results/sales_quailb.log

The summary is written to /results/sales/<run>_sales_quailb.json and each
query's answers to /results/sales/<run>_<query>_<operator>.parquet.
"""

import json
import os
import time

import modal

QUAIL_BENCH_DIR = os.environ.get("QUAIL_BENCH_DIR")
REMOTE_BENCH = "/root/quail-bench"
try:
    from quail.bench.images import gpu_image
    quail_image = gpu_image(*(
        [(os.path.join(QUAIL_BENCH_DIR, "quail_b"), f"{REMOTE_BENCH}/quail_b")]
        if QUAIL_BENCH_DIR else []))
except ImportError:    # the local entrypoint may lack the GPU stack
    quail_image = None

MODEL = "decision-2.0-kai-0.6b-bf16"
INPUT = "/results/sales/inputs/sf1.0/sales_calls.parquet"
QUERIES = ("SALES-1", "SALES-2", "SALES-3", "SALES-4", "SALES-5")

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}


def _rate(answers):
    """TRUE count and share of a list of booleans."""
    true = int(sum(answers))
    return {"n": len(answers), "true": true,
            "share": round(true / len(answers), 4) if answers else None}


@app.function(image=quail_image, gpu="H100!", memory=98304, timeout=7200,
              volumes=VOLUMES)
def run_sales(run: str) -> dict:
    """Run the five SALES queries and save their answers."""
    import sys

    if os.path.isdir(REMOTE_BENCH):
        sys.path.insert(0, REMOTE_BENCH)

    import pyarrow.parquet as pq

    import quail
    from quail.bench.quailb import run_query
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b.queries import queries

    specs = queries(include_pending=True)
    calls = pq.read_table(INPUT)
    tables = {"sales_calls": calls}
    frame = calls.to_pandas()
    domain_of = dict(zip(frame.id, frame.domain))
    stage_of = dict(zip(frame.id, frame.deal_stage))

    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm"))
    tok = session.tokenizer
    doc_tokens = dict(zip(frame.id, (len(tok(text))
                                     for text in frame.transcript)))
    out = {"run": run, "model": MODEL, "input": INPUT,
           "quail_b": sys.modules["quail_b"].__file__, "queries": {}}
    os.makedirs("/results/sales", exist_ok=True)
    for qid in QUERIES:
        spec = specs[qid]
        output = run_query(session, spec, tables)
        report = output.measurements
        summary = {"rows": output.rows.num_rows,
                   "runtime_s": round(output.runtime_s, 2),
                   "query_time_s": report["wall_s"],
                   "boot_s": report.get("boot_s"),
                   "fresh_tokens": report.get("fresh_tokens")}
        prompts = {o.id: o.prompt for o in (
            *spec.info.filters, *spec.info.joins, *spec.info.scores,
            *spec.info.classifies)}
        answers = {**output.filter_answers, **output.join_answers,
                   **(output.score_answers or {}),
                   **(output.classify_answers or {})}
        input_tokens = 0
        for operator_id, table in answers.items():
            aliases = [name for name in table.column_names
                       if name not in ("answer", "label", "score")]
            prompt_tokens = len(tok(prompts[operator_id].replace(
                "{0}", "").replace("{1}", "")))
            input_tokens += sum(
                prompt_tokens + sum(doc_tokens[i] for i in row)
                for row in zip(*(table.column(a).to_pylist()
                                 for a in aliases)))
            pq.write_table(table, f"/results/sales/{run}_{qid}_"
                           f"{operator_id}.parquet")
        summary["input_tokens_approx"] = input_tokens
        summary["input_tokens_per_second"] = round(
            input_tokens / report["wall_s"], 1)
        summary["usd_per_query"] = round(
            report["wall_s"] / 3600 * H100_USD_PER_HOUR, 4)

        for operator_id, table in output.filter_answers.items():
            ids = table.column("c").to_pylist()
            answer = table.column("answer").to_pylist()
            summary[f"filter {operator_id}"] = _rate(answer)
            summary[f"filter {operator_id} by domain"] = {
                domain: _rate([a for a, i in zip(answer, ids)
                               if domain_of[i] == domain])
                for domain in ("b2b", "b2c")}
        for operator_id, table in output.join_answers.items():
            summary[f"join {operator_id}"] = _rate(
                table.column("answer").to_pylist())
        for operator_id, table in (output.classify_answers or {}).items():
            ids = table.column("c").to_pylist()
            labels = table.column("label").to_pylist()
            summary[f"classify {operator_id} by domain"] = {
                domain: {label: sum(1 for lab, i in zip(labels, ids)
                                    if lab == label
                                    and domain_of[i] == domain)
                         for label in sorted(set(labels))}
                for domain in ("b2b", "b2c")}
        for operator_id, table in (output.score_answers or {}).items():
            scores = table.column("score").to_pylist()
            ranked = sorted(scores, reverse=True)
            summary[f"score {operator_id}"] = {
                "n": len(scores),
                "over_0.5": sum(score > 0.5 for score in scores),
                "25th_highest": ranked[24] if len(ranked) >= 25 else None,
                "median": ranked[len(ranked) // 2] if ranked else None}
        rows = output.rows.to_pandas()
        if "c" in rows:
            summary["row stages"] = {
                stage: int(sum(stage_of[i] == stage for i in rows.c))
                for stage in sorted({stage_of[i] for i in rows.c})}
        summary["result"] = rows.head(10).to_dict(orient="records")
        pq.write_table(output.rows, f"/results/sales/{run}_{qid}_rows.parquet")
        out["queries"][qid] = summary
        print(f"[sales] {qid}: {json.dumps(summary, default=str)}",
              flush=True)
    session.close()
    path = f"/results/sales/{run}_sales_quailb.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    results.commit()
    return {"path": path}


@app.local_entrypoint()
def main():
    """Spawn the run and print its function call id."""
    call = run_sales.spawn(time.strftime("%Y%m%d-%H%M%S"))
    print(f"[sales] run_sales function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2, default=str), flush=True)
