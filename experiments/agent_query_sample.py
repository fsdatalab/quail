"""Compare two models on a candidate agent trace filter.

Run the experiment through the existing Modal app and save the log:

    mkdir -p results/benchmark
    sample_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-agent2-sample.log"
    uv run modal run --detach experiments/agent_query_sample.py 2>&1 | tee "$sample_log"

The experiment writes its summary and individual answers to:

    /results/ablations/agent2_implemented_fix_sample.json
    /results/ablations/agent2_implemented_fix_sample.parquet
    /results/ablations/agent2_identified_cause_sample_rows.parquet
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

from quail.specs import H100_USD_PER_HOUR

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
SAMPLE_SIZE = 200
SCALE_FACTOR = 0.1
DATA_DIR = "/results/quailb_data"
OUTPUT_ROOT = Path("/results/ablations")
SAMPLE_PATH = OUTPUT_ROOT / "agent2_identified_cause_sample_rows.parquet"
ANSWER_PATH = OUTPUT_ROOT / "agent2_implemented_fix_sample.parquet"
SUMMARY_PATH = OUTPUT_ROOT / "agent2_implemented_fix_sample.json"

CANDIDATE_PROMPT = (
    "Judge strictly from the agent trace above whether, by the end of the "
    "trace, the agent has implemented a plausible fix that directly addresses "
    "the reported issue. A fix must include a code or configuration change "
    "whose purpose is to correct the issue. Inspection, reproduction, tests "
    "without a fix, and unrelated edits do not count.\n\n{0}\n\nInstruction: "
    "answer TRUE if the agent has implemented a plausible fix that directly "
    "addresses the reported issue. Answer FALSE otherwise."
)

PREDICTION_TEXT = (
    "On the same fixed sample of 200 trace snapshots, Qwen3 32B will label "
    "35 to 60 percent true. Qwen3 4B selectivity will be within 10 percentage "
    "points of Qwen3 32B, and the two models will agree on at least 75 "
    "percent of documents."
)

ACCEPTANCE = {
    "qwen3_32b_selectivity_min": 0.25,
    "qwen3_32b_selectivity_max": 0.60,
    "selectivity_gap_max": 0.10,
    "agreement_min": 0.75,
}

MODELS = {
    "qwen3-32b-fp8": {
        "repo": "Qwen/Qwen3-32B-FP8",
        "revision": "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
        "gpu_memory_utilization": 0.85,
    },
    "qwen3-4b-fp8": {
        "repo": "Qwen/Qwen3-4B-FP8",
        "revision": "96b30dc13593a244a5e59e84687309f53c375cfa",
        "gpu_memory_utilization": 0.91,
    },
}

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    # quail's pinned dependencies and the dev group (quail-b among
    # them) from uv.lock; the quail source is mounted after
    .uv_sync(groups=["dev"])
    .env({
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail")
)

data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(groups=["dev"], extra_options="--no-install-package vllm")
    .add_local_python_source("quail")
)

finalize_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(groups=["dev"], extra_options="--no-install-package vllm")
    .add_local_python_source("quail")
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)


def _sample_rank(row_id: str) -> bytes:
    value = f"20260818:agent2-identified-cause:{row_id}"
    return hashlib.sha256(value.encode()).digest()


def stable_sample_rows(rows: list[dict], size: int) -> list[dict]:
    """Select a stable uniform sample of documents."""
    if size > len(rows):
        raise ValueError(f"sample size {size} exceeds {len(rows)} rows")
    return sorted(rows, key=lambda row: _sample_rank(str(row["id"])))[:size]


def _render_prompt(document: str) -> str:
    from quail.logical import ColumnRef, bind_prompt

    ref = ColumnRef("t", "agent_traces", "trace")
    prompt = bind_prompt(CANDIDATE_PROMPT, (ref,))
    if not prompt.tail.startswith("{0}"):
        raise ValueError("candidate prompt does not use the filter layout")
    return prompt.preamble + document + prompt.tail.replace("{0}", "", 1)


def _sample_hash(rows: list[dict]) -> str:
    payload = [
        {
            "id": row["id"],
            "trace_sha256": hashlib.sha256(row["trace"].encode()).hexdigest(),
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_function_calls(value: str) -> dict[str, str]:
    """Parse the completed function calls needed by the result writer."""
    calls = {}
    for item in value.split(","):
        name, function_call_id = item.split("=", 1)
        calls[name.strip()] = function_call_id.strip()
    expected = {"sample", *MODELS}
    if set(calls) != expected:
        raise ValueError(
            f"function calls must contain exactly {sorted(expected)}")
    return calls


@app.function(
    image=data_image,
    memory=8192,
    timeout=1800,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def prepare_sample() -> str:
    """Build the corpus and save the fixed sample."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quail_b import data

    results_vol.reload()
    data_dir = data.build_sets(DATA_DIR, sf=SCALE_FACTOR)
    rows = pq.read_table(
        Path(data_dir) / "agent_traces.parquet",
        columns=["id", "trace", "trajectory_id", "turn_index", "token_count"],
    ).to_pylist()
    sample = stable_sample_rows(rows, SAMPLE_SIZE)
    if any("[USER]\n" not in row["trace"] for row in sample):
        raise ValueError("sample contains a trace without a user issue")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("id", pa.string()),
        ("trace", pa.string()),
        ("trajectory_id", pa.string()),
        ("turn_index", pa.int32()),
        ("token_count", pa.int32()),
    ])
    temp = SAMPLE_PATH.with_name(SAMPLE_PATH.name + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(sample, schema=schema),
        temp,
        compression="zstd",
        use_dictionary=False,
    )
    os.replace(temp, SAMPLE_PATH)
    results_vol.commit()
    result = {
        "data_seed": data.DATA_SEED,
        "cache_schema_version": data.CACHE_SCHEMA_VERSION,
        "source_revision": data.SOURCE_REVISIONS[
            "TIGER-Lab/SWE-Next-SFT-Trajectories"],
        "sample_size": len(sample),
        "sample_full_hash": _sample_hash(sample),
        "sample_volume_path": str(SAMPLE_PATH),
        "unique_trajectories": len({row["trajectory_id"] for row in sample}),
        "min_turn": min(row["turn_index"] for row in sample),
        "max_turn": max(row["turn_index"] for row in sample),
        "mean_tokens": sum(row["token_count"] for row in sample) / len(sample),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    },
)
def judge_sample(model_name: str, sample_full_hash: str) -> str:
    """Judge the saved sample with one model."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from quail import true_false_ids
    from quail_b import data

    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    results_vol.reload()
    rows = pq.read_table(SAMPLE_PATH).to_pylist()
    if _sample_hash(rows) != sample_full_hash:
        raise ValueError("saved sample hash changed")
    prompts = [_render_prompt(row["trace"]) for row in rows]
    spec = MODELS[model_name]

    t_total = time.perf_counter()
    t_boot = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"])
    true_ids, false_ids = true_false_ids(tokenizer)
    allowed = sorted(true_ids | false_ids)
    llm = LLM(
        model=spec["repo"],
        revision=spec["revision"],
        tokenizer_revision=spec["revision"],
        seed=data.DATA_SEED,
        kv_cache_dtype="auto",
        max_model_len=32_768,
        max_num_batched_tokens=25_305,
        max_num_seqs=4_096,
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed,
        logprobs=len(allowed),
        seed=data.DATA_SEED,
    )
    boot_s = time.perf_counter() - t_boot
    t_model = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    model_wall_s = time.perf_counter() - t_model

    answers = []
    prompt_tokens = 0
    for row, output in zip(rows, outputs):
        token_id = int(output.outputs[0].token_ids[0])
        if token_id in true_ids:
            answer = True
        elif token_id in false_ids:
            answer = False
        else:
            raise ValueError(f"unexpected answer token {token_id}")
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if prompt_ids is not None:
            prompt_tokens += len(prompt_ids)
        answers.append({
            "id": row["id"],
            "answer": answer,
            "selected_token_id": token_id,
        })

    total_wall_s = time.perf_counter() - t_total
    result = {
        "model": model_name,
        "model_repo": spec["repo"],
        "model_revision": spec["revision"],
        "sample_full_hash": sample_full_hash,
        "requests": len(answers),
        "prompt_tokens": prompt_tokens,
        "true_rows": sum(row["answer"] for row in answers),
        "selectivity": sum(row["answer"] for row in answers) / len(answers),
        "boot_s": round(boot_s, 2),
        "model_wall_s": round(model_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
        "cost_with_boot_usd": total_wall_s / 3600 * H100_USD_PER_HOUR,
        "answers": answers,
    }
    kernel_cache.commit()
    print(json.dumps({
        key: value for key, value in result.items() if key != "answers"
    }, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


def _comparison(reference: dict, candidate: dict) -> dict:
    reference_answers = {
        row["id"]: bool(row["answer"]) for row in reference["answers"]
    }
    candidate_answers = {
        row["id"]: bool(row["answer"]) for row in candidate["answers"]
    }
    if set(reference_answers) != set(candidate_answers):
        raise ValueError("model answers cover different documents")

    true_positive = true_negative = false_positive = false_negative = 0
    for row_id, expected in reference_answers.items():
        observed = candidate_answers[row_id]
        if expected and observed:
            true_positive += 1
        elif expected:
            false_negative += 1
        elif observed:
            false_positive += 1
        else:
            true_negative += 1
    total = len(reference_answers)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0)
    recall = true_positive / recall_denominator if recall_denominator else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "agreement": (true_positive + true_negative) / total,
        "selectivity_gap": abs(
            reference["selectivity"] - candidate["selectivity"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


@app.function(
    image=finalize_image,
    memory=4096,
    timeout=600,
    volumes={"/results": results_vol},
)
def save_result(sample_json: str, results_json: str) -> str:
    """Save the combined summary and individual answers."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sample = json.loads(sample_json)
    results = {
        result["model"]: result
        for result in json.loads(results_json)
    }
    reference = results["qwen3-32b-fp8"]
    candidate = results["qwen3-4b-fp8"]
    comparison = _comparison(reference, candidate)
    accepted = (
        ACCEPTANCE["qwen3_32b_selectivity_min"]
        <= reference["selectivity"]
        <= ACCEPTANCE["qwen3_32b_selectivity_max"]
        and comparison["selectivity_gap"]
        <= ACCEPTANCE["selectivity_gap_max"]
        and comparison["agreement"] >= ACCEPTANCE["agreement_min"]
    )

    reference_answers = {row["id"]: row for row in reference["answers"]}
    candidate_answers = {row["id"]: row for row in candidate["answers"]}
    answer_rows = [
        {
            "id": row_id,
            "qwen3_32b_answer": reference_answers[row_id]["answer"],
            "qwen3_4b_answer": candidate_answers[row_id]["answer"],
            "agree": (
                reference_answers[row_id]["answer"]
                == candidate_answers[row_id]["answer"]
            ),
            "qwen3_32b_token_id": reference_answers[row_id][
                "selected_token_id"],
            "qwen3_4b_token_id": candidate_answers[row_id][
                "selected_token_id"],
        }
        for row_id in sorted(reference_answers)
    ]
    schema = pa.schema([
        ("id", pa.string()),
        ("qwen3_32b_answer", pa.bool_()),
        ("qwen3_4b_answer", pa.bool_()),
        ("agree", pa.bool_()),
        ("qwen3_32b_token_id", pa.int64()),
        ("qwen3_4b_token_id", pa.int64()),
    ])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    answer_temp = ANSWER_PATH.with_name(ANSWER_PATH.name + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(answer_rows, schema=schema),
        answer_temp,
        compression="zstd",
        use_dictionary=True,
    )
    os.replace(answer_temp, ANSWER_PATH)

    model_summaries = {
        model_name: {
            key: value for key, value in result.items() if key != "answers"
        }
        for model_name, result in results.items()
    }
    summary = {
        "cell": "agent2_implemented_fix_sample",
        "prediction": PREDICTION_TEXT,
        "candidate_prompt": CANDIDATE_PROMPT,
        "acceptance_thresholds": ACCEPTANCE,
        "accepted": accepted,
        "sample": sample,
        "models": model_summaries,
        "comparison": comparison,
        "total_gpu_cost_with_boot_usd": sum(
            result["cost_with_boot_usd"] for result in results.values()),
        "summary_volume_path": str(SUMMARY_PATH),
        "answers_volume_path": str(ANSWER_PATH),
    }
    summary_temp = SUMMARY_PATH.with_name(SUMMARY_PATH.name + ".tmp")
    with open(summary_temp, "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    os.replace(summary_temp, SUMMARY_PATH)
    results_vol.commit()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return json.dumps(summary, sort_keys=True)


@app.local_entrypoint()
def main(finalize_from: str | None = None):
    """Run the sample preparation, both judges, and result writer."""
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    if finalize_from:
        calls = parse_function_calls(finalize_from)
        sample_json = modal.FunctionCall.from_id(calls["sample"]).get()
        results_json = json.dumps([
            json.loads(modal.FunctionCall.from_id(calls[model_name]).get())
            for model_name in MODELS
        ])
        save_call = save_result.spawn(sample_json, results_json)
        print(f"function call id (save result): {save_call.object_id}",
              flush=True)
        print(save_call.get(), flush=True)
        return

    sample_call = prepare_sample.spawn()
    print(f"function call id (prepare sample): {sample_call.object_id}",
          flush=True)
    sample_json = sample_call.get()
    sample = json.loads(sample_json)

    calls = {
        model_name: judge_sample.spawn(
            model_name, sample["sample_full_hash"])
        for model_name in MODELS
    }
    for model_name, call in calls.items():
        print(f"function call id ({model_name}): {call.object_id}", flush=True)
    results_json = json.dumps([
        json.loads(calls[model_name].get()) for model_name in MODELS
    ])
    save_call = save_result.spawn(sample_json, results_json)
    print(f"function call id (save result): {save_call.object_id}", flush=True)
    print(save_call.get(), flush=True)
