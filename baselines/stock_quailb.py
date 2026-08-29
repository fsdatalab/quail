"""Run the stock vLLM join client on the three QUAIL-B joins.

The GPU work calls ``baselines.stock.run_join_grouped``. Results are written
to the ``quail-results`` Modal volume.

    uv run modal run -m baselines.stock_quailb::stock_main \
        --model qwen3-4b-fp8
"""

import json
import time
from pathlib import Path

import modal

from baselines.vllm_opbench import operators
from baselines.vllm_opbench.config import DATA_DIR, MODEL_NAMES, SF
from baselines.vllm_opbench.run import (
    WorkerH100,
    _tokenizer_for,
    app,
    build_queries,
    hf_cache_vol,
    orchestrator_image,
    results_vol,
)

STOCK_MODELS = {"qwen3-4b-fp8", "qwen3-32b-fp8"}
JOIN_QUERY_IDS = ("join-reports", "join-claims", "join-imdb")
SUBMISSION = (
    "one request per pair, anchor-major, one llm.generate call, "
    "prefix caching on"
)


def _pair_ids(op, data, members, n_prefixes):
    left_ids, _, right_ids, _ = data
    partner = 1 - op.anchor
    pairs = []
    for anchor_idx in range(n_prefixes):
        for member in members:
            indices = [None, None]
            indices[op.anchor] = anchor_idx
            indices[partner] = member[0]
            pairs.append((left_ids[indices[0]], right_ids[indices[1]]))
    return pairs


@app.function(
    image=orchestrator_image,
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol,
             "/results": results_vol},
)
def run_stock_baseline(model: str = "qwen3-4b-fp8",
                       query_id: str | None = None,
                       gpu: str = "H100!") -> dict:
    """Run current QUAIL-B join prompts through the stock join client."""
    from quail.bench.quailb import build_sets

    if model not in STOCK_MODELS:
        raise ValueError(
            f"stock QUAIL-B model must be one of {sorted(STOCK_MODELS)}; "
            f"got {model!r}")
    if gpu != WorkerH100.GPU:
        raise ValueError(
            f"stock QUAIL-B requires gpu={WorkerH100.GPU!r}; got {gpu!r}")
    if query_id and query_id not in JOIN_QUERY_IDS:
        raise ValueError(
            f"stock QUAIL-B query must be one of {JOIN_QUERY_IDS}; "
            f"got {query_id!r}")

    build_sets(DATA_DIR, SF)
    tokenizer = _tokenizer_for(MODEL_NAMES[model])
    true_ids, false_ids = operators.true_false_ids(tokenizer)
    queries = build_queries(DATA_DIR, SF)
    ids = [query_id] if query_id else list(JOIN_QUERY_IDS)

    worker = WorkerH100(model=model, quantization="checkpoint")
    print(f"[stock_quailb] warming up {model} on {gpu}", flush=True)
    worker.warmup.remote(true_ids, false_ids)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("/results/stock_quailb") / f"{timestamp}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "runner": "baselines.stock.run_join_grouped",
        "submission": SUBMISSION,
        "model": model,
        "model_name": MODEL_NAMES[model],
        "gpu": gpu,
        "weight_dtype": "fp8",
        "weight_source": "prequantized checkpoint",
        "kv_cache_dtype": "bfloat16",
        "sf": SF,
        "queries": [],
    }

    for qid in ids:
        kind, op, data = queries[qid]
        if kind != "join":
            raise ValueError(f"{qid} is not a join")
        _, left_texts, _, right_texts = data
        print(
            f"[stock_quailb] {qid}: building "
            f"{len(left_texts)}x{len(right_texts)}="
            f"{len(left_texts) * len(right_texts)} prompts",
            flush=True,
        )
        build_t0 = time.time()
        prefixes, suffixes, members = op.build_grouped_inputs(
            left_texts, right_texts, tokenizer)
        pairs = _pair_ids(op, data, members, len(prefixes))
        build_s = time.time() - build_t0

        warmup = worker.generate_stock_join_batch.remote(
            prefixes[:2], suffixes[:32], true_ids, false_ids)
        print(
            f"[stock_quailb] {qid}: 64-pair warmup took "
            f"{warmup['wall']:.2f}s",
            flush=True,
        )
        rpc_t0 = time.time()
        result = worker.generate_stock_join_batch.remote(
            prefixes, suffixes, true_ids, false_ids)
        rpc_wall_s = time.time() - rpc_t0
        n_prompts = len(prefixes) * len(suffixes)
        entry = {
            "query_id": qid,
            "operator": op.name,
            "kind": kind,
            "anchor": op.anchor,
            "n_prompts": n_prompts,
            "build_s": build_s,
            "warmup_wall_time_s": warmup["wall"],
            "rpc_wall_time_s": rpc_wall_s,
            "generate_wall_time_s": result["wall"],
            "req_per_s": n_prompts / result["wall"],
            "total_prompt_tokens": result["prompt_tokens"],
            "fresh_prompt_tokens": result["fresh_tokens"],
            "cached_prompt_tokens": result["cached_tokens"],
            "true_answers": sum(result["answers"]),
            "max_num_seqs": result["max_num_seqs"],
            "max_num_batched_tokens": result["max_num_batched_tokens"],
            "block_size": result["block_size"],
        }
        summary["queries"].append(entry)
        with (out_dir / f"{qid}.jsonl").open("w") as output:
            for pair, answer in zip(pairs, result["answers"]):
                output.write(json.dumps({
                    "left_id": pair[0],
                    "right_id": pair[1],
                    "answer": answer,
                }) + "\n")
        with (out_dir / "summary.json").open("w") as output:
            json.dump(summary, output, indent=2)
        results_vol.commit()
        print(
            f"[stock_quailb] {qid}: "
            f"generate_wall={result['wall']:.2f}s, "
            f"requests/s={entry['req_per_s']:.1f}, "
            f"fresh_tokens={result['fresh_tokens']}",
            flush=True,
        )

    path = f"{out_dir}/summary.json"
    print(f"[stock_quailb] saved {path}", flush=True)
    return {"result_path": path, "n_queries": len(summary["queries"])}


@app.local_entrypoint()
def stock_main(model: str = "qwen3-4b-fp8", query: str = "",
               gpu: str = "H100!"):
    call = run_stock_baseline.spawn(
        model=model, query_id=(query or None), gpu=gpu)
    print(f"function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)
