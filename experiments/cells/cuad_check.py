"""The CUAD queries through QUAIL-B, on Quail and stock vLLM.

Runs CUAD-1 to CUAD-15 at one scale factor on one model, one backend
per container, against the published labels. Each run saves a QUAIL-B
run record and report under
/results/benchmarks/quailb/cuad-<run>-<model>-<backend>/. The tables
are downloaded once into the Hugging Face cache volume so that a
backend's refusals are found before its engine boots: a decision model
refuses the nine extraction queries, and stock vLLM refuses CUAD-11's
GROUP BY.

Prediction for Qwen3 4B, from the 2,500-character contract windows of
run 20261011T060458Z (Quail span F1 47.5 with locating on, stock vLLM
52.0): on the extraction queries Quail's span token F1 lands within 10
points of stock vLLM's, lower on the clause fields (CUAD-3, 7, 8, 9)
than on the short ones (CUAD-1, 2), and both engines agree with the
lawyers' Yes or No answers on the filters within 3 points of each
other. Measured on run 20261011T183123Z: Quail trailed stock vLLM by
up to 32 points on the clause fields (CUAD-3: 0.23 against 0.55) and
matched it within 3 points on the filters. Run 20261011T235454Z, on
the published labels, repeated every number; Quail took 15 to 19 s on
the single extractions against 7 to 15 s for stock vLLM, and 46 s
against 13 s on CUAD-10, where it read each contract three times.

Prediction for the shared document KV (one key per document and
rendering, retained after each extraction): CUAD-10 falls from 46 s to
under 22 s and from 1.07 million fresh tokens to under 400 thousand,
with every answer unchanged; CUAD-7, a filter then an extraction,
keeps its 18.7 s and its answers, since a filter's KV holds the plain
document and the extraction shows it numbered. Measured on run
20261012T001413Z: CUAD-10 took 42.4 s and 800 thousand fresh tokens,
answers unchanged. The arena holds 362 thousand tokens and the
retention cap leaves 141 thousand retained, so 38% of the contracts
stayed resident between the extractions.

Prediction for the extraction pipeline (one read per document, every
question asked while it is resident): CUAD-10 reads each contract
once, under 400 thousand fresh tokens, and takes about 35 s, the
single read of 5 s plus three questions' steps of about 10 s each,
with every answer unchanged. CUAD-1 and CUAD-7 keep their times and
answers within a second. Measured on run 20261012T003911Z: CUAD-10
took 23.7 s and 368 thousand fresh tokens; CUAD-1 and CUAD-7 kept
their answers; CUAD-10's second and third questions changed on 7 and
8 contracts. Runs 20261012T005434Z, 010321Z, and 010614Z
(experiments/cells/extract_pipeline_check.py) showed the KV those
questions read is bit-identical to what the first question wrote, and
that the same question alone moves on 0 to 3 contracts when only the
chunk budget changes: the span rule's decisions sat on near ties.

Prediction for the answer written token by token within the located
lines, each token chosen among the document's continuations: span
token F1 lands within 3 points of stock vLLM's on every extraction
query of run 20261011T235454Z (CUAD-1 0.35, CUAD-2 0.85, CUAD-3 0.55,
CUAD-4 0.63, CUAD-7 0.75, CUAD-8 0.48, CUAD-9 0.26, CUAD-10 0.48),
since both write the answer greedily and the stock path keeps only
answers it finds in the text. Time follows the answer's tokens: the
short fields (CUAD-1, 2, 10) stay within 2 s of their last runs, and
the clause fields (CUAD-3, 7, 8, 9), about 60 tokens, take up to 60 s
at today's cost per round. Measured on run 20261012T011602Z, with
locating on: F1 unchanged from the span rule on every query (CUAD-1
0.28, CUAD-2 0.61, CUAD-3 0.21), 13 to 19 s. Run 011923Z, locating
off: within 3 points of stock vLLM on CUAD-1, 2, 4, 9, and 10 (0.37,
0.83, 0.62, 0.29, 0.49) and below on the clause fields (CUAD-3 0.41,
CUAD-7 0.58, CUAD-8 0.36); the lines the model names are wrong often
enough to cost 5 to 22 points on every query, so locating is off by
default. That run took 30 to 126 s per query from a first-token scan
per position and streams tokenized to the contract's end; run
013438Z, with the scan by word and streams tokenized a window at a
time, took 7.9 to 14.0 s, within a second of stock vLLM on each query.

Prediction for the null rule (the answer is none when the model's
likeliest token continues no position, as the stock path's is when
its written answer is not in the text): CUAD-3 and CUAD-7 close to
within 5 points of stock vLLM (0.55, 0.75), since stock vLLM returned
null on 16 of 45 contracts for CUAD-3 where the reference is often
null; CUAD-1 and CUAD-2 stay within 2 points of 0.37 and 0.83.
Measured on run 20261012T014442Z: CUAD-3 0.57, CUAD-7 0.67, CUAD-8
0.50, CUAD-1 0.37, CUAD-2 0.81, and every query within 2 s of stock
vLLM; Quail now returns null on all 16 contracts stock vLLM does for
CUAD-3. CUAD-4, the parties, fell from 0.62 to 0.26: the rule fired
in the middle of 27 answers, where the model would write "and"
between two names the contract lists apart, and stock vLLM keeps
such an answer as written (locate_span finds no span; the text is
scored).

Prediction for the rule at the first token only, the answer later
following the document's continuation as before: CUAD-4 returns to
at least 0.58, and CUAD-3, CUAD-7, and CUAD-8 hold within 3 points of
run 014442Z, if their added nulls came at the first token. Measured
on run 20261012T015932Z: CUAD-4 0.65 (exact match 0.24), CUAD-3 0.56,
CUAD-1 0.37, CUAD-2 0.83, CUAD-9 0.35, CUAD-10 0.49; CUAD-7 fell to
0.60 and CUAD-8 to 0.40, so their added nulls came in the middle of
the answer, where the model would rather paraphrase than copy on.

Prediction for ending the answer where the model's writing would
leave the document, and none only when that is at the cue: CUAD-4
holds at or above 0.62, with exact match above 0.24 since a contract
is scored against its best-matching party; CUAD-7 and CUAD-8 land
between the two runs, at or above 0.63 and 0.45; CUAD-1, 2, 3, 9,
and 10 hold within 3 points of run 015932Z.

Prediction for Decision-2.0-Kai 0.6B: the planner refuses the nine
extraction queries, so six run. Its agreement with the lawyers on the
two filter queries lands within 10 points of Qwen3 4B's (CUAD-5 0.71,
CUAD-6 0.27 on run 20261011T183123Z), the governing law
classifications (CUAD-11, 12) within 10 points of Qwen3 4B's 0.91, and
the join queries (CUAD-13, 14) keep Qwen3 4B's shape, recall above 0.9
and precision below 0.3. Every query finishes in under 10 s on Quail.

    uv run modal run --detach -m experiments.cells.cuad_check
        [--model decision-2.0-kai-0.6b-bf16] [--backends quail]
        [--sf 0.1] [--only CUAD-5,CUAD-13] [--locate]
        2>&1 | tee /tmp/cuad_check.log
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
TABLES = "/root/.cache/huggingface/quail-b-tables"
QUERIES = tuple(f"CUAD-{i}" for i in range(1, 16))


def _summary(record) -> dict:
    """Pull the numbers to report from each query of a run record."""
    out = {}
    for item in record["queries"]:
        metrics = item.get("metrics") or {}
        accuracy = metrics.get("accuracy") or {}
        out[item["id"]] = {
            "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "answer_accuracy": (accuracy.get("answer_accuracy") or {}).get(
                "accuracy"),
            "label_accuracy": (accuracy.get("label_accuracy") or {}).get(
                "accuracy"),
            "span_token_f1": (accuracy.get("span_accuracy") or {}).get(
                "token_f1"),
            "span_exact_match": (accuracy.get("span_accuracy") or {}).get(
                "exact_match"),
            "output_precision": (accuracy.get("output_accuracy") or {}).get(
                "precision"),
            "output_recall": (accuracy.get("output_accuracy") or {}).get(
                "recall"),
            "input_tokens": metrics.get("input_tokens"),
            "fresh_tokens": metrics.get("fresh_tokens"),
            "regret_tokens": metrics.get("regret_tokens"),
            "input_tokens_per_second": metrics.get("input_tokens_per_second"),
            "cost_usd": metrics.get("cost_usd"),
            "error": item.get("error"),
        }
    return out


@app.function(image=image, gpu="H100!", memory=98304, timeout=7200,
              volumes=VOLUMES)
def run_cuad(run: str, model: str, backend: str, sf: float = 0.1,
             only=None, locate: bool = False) -> dict:
    """Run the CUAD queries on one backend and save the QUAIL-B record."""
    import quail
    from quail.bench import quailb
    from quail_b.data import build_sets

    quailb.EXTRACT_OPTIONS.update(locate=locate)
    queries = list(only or QUERIES)
    data_dir = build_sets(TABLES, sf)
    VOLUMES["/root/.cache/huggingface"].commit()
    output = (f"/results/benchmarks/quailb/cuad-{run}-{model}-{backend}"
              f"{'-located' if locate else ''}")
    started = time.perf_counter()
    try:
        record = quailb.run_suite(
            queries, sf=sf,
            config=quail.EngineConfig(model=model, device="h100-sxm",
                                      backend=backend),
            data_dir=data_dir, output_dir=output)
    finally:
        VOLUMES["/results"].commit()
    summary = {"run": run, "model": model, "backend": backend, "sf": sf,
               "output": output, "locate": locate,
               "total_s": time.perf_counter() - started,
               "skipped": record.get("skipped_queries", {}),
               "queries": _summary(record)}
    print(json.dumps(summary), flush=True)
    return summary


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", backends: str = "quail,stock_vllm",
         sf: float = 0.1, only: str = "", locate: bool = False):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    queries = [q for q in only.split(",") if q] or None
    for backend in backends.split(","):
        call = run_cuad.spawn(run, model, backend, sf, queries, locate)
        print(f"{backend}: function call id {call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
