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
and 10 hold within 3 points of run 015932Z. Measured on run
20261012T020405Z: CUAD-4 0.79 (exact match 0.49), above stock vLLM's
0.63; CUAD-1 0.36, CUAD-2 0.83, CUAD-3 0.55, CUAD-9 0.33, CUAD-10
0.49, as predicted; CUAD-7 0.60 and CUAD-8 0.40, unchanged from run
015932Z, so that part of the prediction failed. The filters in front
of CUAD-7 and CUAD-8 returned the same rows on all three runs. On
CUAD-7 the null rule had answered none on 7 more of the 39 extracted
contracts than ending the answer does; ending it changed 3 answers
and moved the score by 0.3 points. CUAD-8 extracts 10 contracts, so
its 10 points are one contract that the null rule answered none,
matching the reference. Stock vLLM reaches 0.75 on CUAD-7 by
writing none on more contracts than Quail's cue does.

Prediction for feeding the document's next 16 tokens as a draft each
round and verifying them by the rows after them: every answer stays
as on run 020405Z but for the 0 to 3 contracts per query that move
between runs; the clause fields drop from 10.2 s to about 6 s
(CUAD-3), 13.0 to about 8 s (CUAD-7), and 8.9 to about 6 s (CUAD-8),
since a 60-token answer takes 4 to 8 rounds instead of 60; the short
fields (CUAD-1, 2, 4, 10) move by less than 1 s. Measured on run
20261012T055523Z: every score within a point of run 020405Z (CUAD-3
0.54, CUAD-4 0.80, the rest equal); CUAD-3 8.5 s, CUAD-7 11.6 s,
CUAD-8 7.4 s, less than predicted; CUAD-1 7.8 s, CUAD-2 7.3 s, CUAD-4
6.7 s, CUAD-9 7.6 s, CUAD-10 11.9 s. The short fields fell by 0.2 to
2.5 s too, so the rounds were not the whole gap: a query with answers
of a few tokens takes 7 to 8 s, and run 20261012T055945Z
(experiments/cells/extract_pipeline_check.py --profile) times the
read chunks and the rounds apart. It measured, for the cap on
liability: 4.5 s of forward passes over 4 read chunks, with the
attention and the fp8 matrix multiplies near their peak (run
20261012T060259Z --kernels); 4.2 s on the host reading those chunks'
answers, nearly all the first-token scan; 1.6 s of forward passes over
13 rounds; and, before the run, 2.2 s on the host preparing the
documents, measured locally.

Prediction for batch tokenization, the scan vectorized, and streams
tokenized on demand (preparation 2.5 s to 0.9 s and the scan 3.3 s to
0.6 s, measured locally over the 45 contracts): every query 1.5 to
2.5 s faster than run 055523Z, CUAD-1 about 6 s and CUAD-3 about 6.5
s, with every answer unchanged but for 0 to 3 contracts per query.
Measured on run 20261012T061618Z: CUAD-1 6.4 s, CUAD-2 5.2 s, CUAD-3
5.6 s, CUAD-4 5.3 s, CUAD-7 9.5 s, CUAD-8 6.6 s, CUAD-9 6.6 s, and
CUAD-10 6.1 s, every one faster than stock vLLM's run 235454Z (6.9,
8.7, 10.3, 10.0, 14.8, 11.4, 10.2, 12.7 s); CUAD-10, three questions
over each contract, fell 5.8 s since each question's scan and the
preparation were paid three times. Scores within a point of run
055523Z (CUAD-3 0.53, CUAD-10 0.49, the rest equal).

Prediction for scale factor 1.0, 450 contracts and about 3.4 million
document tokens, on both backends: Quail reads at the 75 thousand
tokens per second its read chunks run, so the short fields (CUAD-1,
2, 4) take 55 to 65 s and the clause fields and CUAD-10 up to 80 s,
the rounds riding inside the 31 read chunks; stock vLLM, which read
at 49 thousand tokens per second at scale factor 0.1, takes 70 to 100
s per query. Span F1 lands within 5 points of the scale factor 0.1
values for both backends, since the contracts differ. Measured on run
20261012T061833Z, Quail then stock vLLM: CUAD-1 51.0 s (0.35) and
57.4 s (0.36); CUAD-2 48.7 s (0.75) and 56.2 s (0.78); CUAD-3 49.6 s
(0.57) and 69.4 s (0.65); CUAD-4 48.9 s (0.82) and 61.7 s (0.64);
CUAD-7 89.6 s (0.59) and 113.1 s (0.71); CUAD-8 62.3 s (0.45) and
104.1 s (0.54); CUAD-9 63.1 s (0.35) and 95.6 s (0.44); CUAD-10 53.7
s (0.45) and 170.4 s (0.47). Quail read at 67 to 70 thousand input
tokens per second and stock vLLM at 49 to 60 thousand. The short
fields came in under the predicted 55 s; CUAD-7 took 89.6 s, over the
predicted 80, since its filter passes most contracts and the
extraction reads them again under its own prompt head. Stock vLLM's
CUAD-10 took 170 s, over the predicted 100, as it feeds each of the
three questions' prompts whole. The 45-contract scores were noisier
than predicted: at 450 contracts stock vLLM leads on the clause
fields by 6 to 12 points (CUAD-3, 7, 8, 9) and Quail on CUAD-4 by 18.

Prediction for a draft of 32 tokens instead of 16, at scale factor
0.1: the clause fields take about half the rounds (the cap on
liability took 13 on run 055945Z), so CUAD-3 falls from 5.6 s to
about 5.0 s and CUAD-7 from 9.5 s to about 8.8 s; the short fields
move by less than 0.3 s; every answer unchanged but for 0 to 3
contracts per query. Measured on run 20261012T063539Z: every time
within 0.3 s of run 061618Z and every score equal, so the draft's
length does not set the round count, and the draft stays 16 tokens.

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
