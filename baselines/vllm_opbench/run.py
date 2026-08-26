"""vLLM-opbench orchestrator: drives WorkerH100 (see worker.py) over
queries built from quail's own document sets and predicates. Runs
entirely on Modal. Both CPU preparation and GPU generation happen
remotely in the existing "quail-milestone1" app. No local Python
execution is needed to launch this.

Four queries, mirroring an existing quailb.py query each so the
numbers are directly comparable - one filter shape, one join shape
per dataset (IMDB, BioDEX, FEVER):
    filter-reports   - BIO-1's shape: F7 alone, over `reports`.
    join-reports     - BIO-2's shape: REACTION, reports x terms.
    join-claims      - FEV-2's shape: SUPPORT, claims x evidence.
    join-imdb        - IMDB-2's shape: DISCUSS_ASPECT, reviews x aspects.

Run (from the repository root; tee to a file per house rule):

    uv run modal run -m baselines.vllm_opbench.run::main \\
        2>&1 | tee results/vllm_opbench_run.log

    # one query only:
    uv run modal run -m baselines.vllm_opbench.run::main \\
        --query join-reports 2>&1 | tee results/vllm_opbench_run.log

Spawn, not remote: the function call id prints before anything waits
on the result, so a dropped local connection doesn't lose the run -
re-fetch with modal.FunctionCall.from_id("<id>").get(). Results land
on the "quail-results" volume under /results/vllm_opbench/<timestamp>/,
not in any local JSON file.
"""

import json
import time
from pathlib import Path

import modal

from . import operators
from .config import DATA_DIR, FILTER_MAX_TOKENS, MODEL_NAMES, SF
from .worker import WorkerH100, app, hf_cache_vol

# CPU-only: this function never touches the GPU itself, it just builds
# data/prompts and calls WorkerH100 (a separate, GPU-backed class on
# this same app) via .remote(). Needs quail's own build_sets deps
# (pyarrow/numpy/datasets/huggingface_hub) plus transformers for the
# tokenizer - not vllm, which only WorkerH100's own image needs.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]", "pandas", "pyarrow",
                "numpy", "datasets", "transformers")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # add_local_* must be last: Modal requires it after every build step.
    .add_local_python_source("quail")
    .add_local_python_source("baselines")
)

results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)


def _tokenizer_for(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def build_queries(data_dir: str, sf: float):
    from quail.bench.quailb import DISCUSS_ASPECT, F7, REACTION, SUPPORT

    report_ids, report_texts = operators.read_table(
        data_dir, sf, "reports", "id", "report")
    term_ids, term_texts = operators.read_table(
        data_dir, sf, "terms", "id", "term")
    claim_ids, claim_texts = operators.read_table(
        data_dir, sf, "claims", "id", "claim")
    evidence_ids, evidence_texts = operators.read_table(
        data_dir, sf, "evidence", "id", "text")
    review_ids, review_texts = operators.read_table(
        data_dir, sf, "reviews", "id", "body")
    aspect_ids, aspect_texts = operators.read_table(
        data_dir, sf, "aspects", "id", "aspect")

    return {
        "filter-reports": ("filter", operators.Filter("F7", F7),
                          (report_ids, report_texts)),
        "join-reports": ("join", operators.Join("REACTION", REACTION,
                                                  anchor=0),
                        (report_ids, report_texts, term_ids, term_texts)),
        "join-claims": ("join", operators.Join("SUPPORT", SUPPORT,
                                                 anchor=1),
                       (claim_ids, claim_texts, evidence_ids, evidence_texts)),
        "join-imdb": ("join", operators.Join("DISCUSS_ASPECT",
                                               DISCUSS_ASPECT, anchor=0),
                     (review_ids, review_texts, aspect_ids, aspect_texts)),
    }


def call_operator(worker, kind, op, data, tokenizer, true_ids, false_ids,
                  log: list, do_profile: bool = False):
    build_t0 = time.time()
    if kind == "filter":
        ids, texts = data
        print(f"[vllm_opbench] {op.name} ({kind}): building {len(texts)} "
              f"prompts...", flush=True)
        prompts = op.build_prompts(texts, tokenizer)
        doc_ids = list(ids)
    else:
        left_ids, left_texts, right_ids, right_texts = data
        print(f"[vllm_opbench] {op.name} ({kind}): building "
              f"{len(left_texts)}x{len(right_texts)}="
              f"{len(left_texts) * len(right_texts)} prompts...", flush=True)
        prompts, pairs = op.build_prompts(left_texts, right_texts, tokenizer)
        # per_request[i] corresponds to prompts[i] by construction (both
        # come from the same build order), but nothing on disk said so -
        # doc_ids makes that explicit instead of leaving it as an
        # unstated positional assumption for whoever reads the .jsonl.
        doc_ids = [(left_ids[li], right_ids[ri]) for li, ri in pairs]
    build_s = time.time() - build_t0
    print(f"[vllm_opbench] {op.name} ({kind}): built {len(prompts)} prompts "
          f"in {build_s:.1f}s, submitting to worker...", flush=True)

    t0 = time.time()
    result = worker.generate_batch.remote(
        prompts, true_ids, false_ids, FILTER_MAX_TOKENS,
        do_profile=do_profile, profile_name=op.name)
    rpc_wall_s = time.time() - t0
    print(f"[vllm_opbench] {op.name} ({kind}): worker call returned after "
          f"{rpc_wall_s:.1f}s, post-processing...", flush=True)

    n = len(prompts)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in result["per_request"])
    total_output_tokens = sum(r["output_tokens"] for r in result["per_request"])
    # "fresh" = tokens vLLM's own prefix cache didn't already have resident,
    # i.e. total prompt tokens minus the oracle's cache-hit token count.
    # total_prompt_tokens alone (pre-discount) is NOT fresh_tokens - printing
    # it under that name previously understated the baseline's actual cache
    # effectiveness whenever there were real hits.
    fresh_prompt_tokens = (total_prompt_tokens
                          - result["oracle_regret"]["oracle_hit_tokens"])
    entry = dict(
        operator=op.name, kind=kind, n_prompts=n,
        anchor=(op.anchor if kind == "join" else None),
        # rpc_wall_time_s: time.time() around worker.generate_batch.remote()
        # from the orchestrator - includes Modal RPC/scheduling overhead.
        # generate_wall_time_s: result["wall_time_s"], timed inside the
        # worker around llm.generate() alone - no RPC overhead. The two are
        # not interchangeable; a prior hand-built comparison (vllm_opbench_
        # vs_quail.json) used rpc_wall_time_s under a "worker_wall_s" label,
        # which reads as the tighter generate-only number - see cleanup note
        # in that file's regenerating script.
        rpc_wall_time_s=rpc_wall_s, generate_wall_time_s=result["wall_time_s"],
        build_s=build_s,
        total_prompt_tokens=total_prompt_tokens,
        fresh_prompt_tokens=fresh_prompt_tokens,
        total_output_tokens=total_output_tokens,
        req_per_s=n / result["wall_time_s"] if result["wall_time_s"] else None,
        oracle_regret=result["oracle_regret"],
        vllm_metrics=result["vllm_metrics"],
        trace_path=result.get("trace_path"),
    )
    log.append(dict(entry=entry, per_request=result["per_request"],
                    timeseries=result["timeseries"], doc_ids=doc_ids))
    req_per_s_str = (f"{entry['req_per_s']:.1f}"
                     if entry["req_per_s"] is not None else "n/a")
    print(f"[vllm_opbench] {op.name} ({kind}): n={n} "
          f"generate_wall={result['wall_time_s']:.2f}s "
          f"req/s={req_per_s_str} "
          f"prompt_tokens={total_prompt_tokens} "
          f"fresh_tokens={fresh_prompt_tokens}", flush=True)
    return entry


@app.function(image=orchestrator_image, timeout=3600,
             volumes={"/root/.cache/huggingface": hf_cache_vol,
                      "/results": results_vol})
def run_baseline(model: str = "qwen3-4b", query_id: str | None = None,
                 gpu: str = "H100!", quantization: str = "fp8",
                 profile: bool = False) -> dict:
    """CPU-side prep (build_sets, tokenizing, prompt building) plus
    driving WorkerH100 for each query - all inside one Modal function
    on the "vllm-opbench" app, so instantiating WorkerH100 and calling
    its .remote() methods here works the same way it would from local
    code with an open app.run() context: this function is already
    executing inside a live invocation of that same app."""
    from quail.bench.quailb import build_sets

    build_sets(DATA_DIR, SF)
    tokenizer = _tokenizer_for(MODEL_NAMES[model])
    true_ids, false_ids = operators.true_false_ids(tokenizer)

    queries = build_queries(DATA_DIR, SF)
    ids = [query_id] if query_id else list(queries)

    worker = WorkerH100(model=model, quantization=quantization)
    print(f"[vllm_opbench] warming up {model} on {gpu} ({quantization})",
          flush=True)
    worker.warmup.remote(true_ids, false_ids)

    out_dir = Path("/results/vllm_opbench") / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for qid in ids:
        kind, op, data = queries[qid]
        log: list = []
        entry = call_operator(worker, kind, op, data, tokenizer,
                              true_ids, false_ids, log, do_profile=profile)
        summary.append(entry)
        with open(out_dir / f"{qid}.jsonl", "w") as f:
            for row in log:
                f.write(json.dumps(row) + "\n")
        # Commit after every query, not just at the end: a later query in
        # this same call (OOM, RPC timeout, KeyError) must not take
        # already-computed, GPU-time-costing results down with it.
        with open(out_dir / "summary.json", "w") as f:
            json.dump(dict(model=model, gpu=gpu, quantization=quantization,
                           sf=SF, queries=summary), f, indent=2)
        results_vol.commit()
    print(f"[vllm_opbench] saved {out_dir}/summary.json", flush=True)
    return dict(out_dir=str(out_dir), n_queries=len(summary))


@app.local_entrypoint()
def main(model: str = "qwen3-4b", query: str = "", gpu: str = "H100!",
        quantization: str = "fp8", profile: bool = False):
    fc = run_baseline.spawn(model=model, query_id=(query or None), gpu=gpu,
                            quantization=quantization, profile=profile)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
