"""Confirm the typed engine on the existing Modal app.

Run from the repository root and tee every line:

    uv run modal run experiments/extensible_engine_confirmation.py \
      --prediction "State the expected results before starting." \
      2>&1 | tee results/extensible-engine-confirmation.log

The cell writes these files to the quail-results volume:

    /results/ablations/extensible_engine_confirmation_4b.json
    /results/ablations/extensible_engine_confirmation_4b_2gpu.json
    /results/ablations/extensible_engine_confirmation_32b.json
"""

import hashlib
import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "transformers>=5.2.0",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets",
        "sqlglot",
    )
    .env({
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail", "quail_b")
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(
    "quail-results", create_if_missing=True
)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True
)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}


def _fingerprint(table) -> str:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    if table.column_names:
        table = table.sort_by([
            (column, "ascending") for column in table.column_names
        ]).combine_chunks()
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _save(name: str, value: dict) -> str:
    path = f"/results/ablations/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)
    results_vol.commit()
    return path


def _query_record(query_id, query, gpu_count):
    from quail.runtime.local import execute_worker_query
    from quail.specs import H100_USD_PER_HOUR

    result = execute_worker_query(query)
    table = result.collect()
    report = result.report
    join_stages = [
        stage for stage in report["stages"] if stage["op"] == "join"
    ]
    filter_stages = [
        stage for stage in report["stages"] if stage["op"] == "filter"
    ]
    pairs = sum(stage["tuples"] for stage in join_stages)
    documents = sum(stage["evaluated"] for stage in filter_stages)
    work = pairs if join_stages else documents
    return {
        "query": query_id,
        "model": query.session.model.name,
        "gpus": gpu_count,
        "wall_s": report["wall_s"],
        "throughput": work / report["wall_s"],
        "throughput_unit": (
            "document pairs/second" if join_stages else "documents/second"
        ),
        "usd_per_query": (
            report["wall_s"] / 3600 * gpu_count * H100_USD_PER_HOUR
        ),
        "evaluated_documents": documents,
        "evaluated_document_pairs": pairs,
        "fresh_tokens": report["fresh_tokens"],
        "regret_tokens": report.get("regret_tokens"),
        "kv_manager": report.get("kv_manager"),
        "expected_join_plan": report.get("expected_join_plan"),
        "executed_join_plan": report.get("executed_join_plan"),
        "rows": len(table),
        "result_sha256": _fingerprint(table),
        "boot": report.get("boot"),
        "result_volume_path": report.get("result_volume_path"),
    }


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=3600,
    volumes=volumes,
)
def confirm_4b(
    prediction: str,
    output_name: str = "extensible_engine_confirmation_4b",
    query_ids: str = "BIO-2,AGENT-1",
) -> str:
    import torch

    from quail.bench.quailb import queries, register_sets
    from quail.planner.plan import EngineConfig
    from quail.runtime.session import Session
    from quail_b.data import build_sets

    data = build_sets("/results/quailb_data", 0.1)
    session = Session(EngineConfig(model="qwen3-4b-fp8", gpus=1))
    register_sets(session, data)
    definitions = queries(session)
    selected = [query_id.strip() for query_id in query_ids.split(",")
                if query_id.strip()]
    unknown = set(selected) - {"BIO-2", "AGENT-1"}
    if unknown:
        raise ValueError(f"unknown 4B queries: {sorted(unknown)}")
    started = time.time()
    records = [
        _query_record(query_id, definitions[query_id][1](),
                      1)
        for query_id in selected
    ]
    result = {
        "prediction": prediction,
        "gpu": "H100!",
        "gpu_device_name": torch.cuda.get_device_name(0),
        "elapsed_s": time.time() - started,
        "queries": records,
    }
    result["volume_path"] = _save(output_name, result)
    return json.dumps(result, indent=2)


def _small_session(model: str, gpus: int):
    import pyarrow as pa

    import quail
    from quail.planner.plan import EngineConfig

    session = quail.Session(EngineConfig(model=model, gpus=gpus))
    session.register("left_docs", quail.DocumentProvider.from_table(
        pa.table({
            "id": [f"l{index}" for index in range(8)],
            "body": [f"Issue report {index} about a failing test"
                     for index in range(8)],
        }),
        id_col="id",
        identity=f"confirmation-left-{model}",
    ))
    session.register("right_docs", quail.DocumentProvider.from_table(
        pa.table({
            "id": [f"r{index}" for index in range(4)],
            "body": [f"Patch {index} changes a related test"
                     for index in range(4)],
        }),
        id_col="id",
        identity=f"confirmation-right-{model}",
    ))
    return session


@app.function(
    image=image,
    gpu="H100!:2",
    memory=131072,
    timeout=3600,
    volumes=volumes,
)
def confirm_4b_2gpu(prediction: str) -> str:

    session = _small_session("qwen3-4b-fp8", 2)
    query = session.sql("""
        SELECT l.id, r.id
        FROM left_docs l
        JOIN right_docs r
          ON AI_FILTER(PROMPT(
            'Could document {0} and document {1} describe the same test? ',
            l.body, r.body), {'selectivity': 0.5})
    """)
    result = {
        "prediction": prediction,
        "query": _query_record("small-join", query, 2),
    }
    result["volume_path"] = _save(
        "extensible_engine_confirmation_4b_2gpu", result
    )
    return json.dumps(result, indent=2)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=3600,
    volumes=volumes,
)
def confirm_32b(prediction: str) -> str:

    session = _small_session("qwen3-32b-fp8", 1)
    query = session.sql("""
        SELECT l.id
        FROM left_docs l
        WHERE AI_FILTER(PROMPT(
          'Does document {0} mention a failing test?', l.body),
          {'selectivity': 0.5})
    """)
    result = {
        "prediction": prediction,
        "query": _query_record("small-filter", query, 1),
    }
    result["volume_path"] = _save(
        "extensible_engine_confirmation_32b", result
    )
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main(
    prediction: str = "",
    runs: str = "4b,4b_2gpu,32b",
    output_suffix: str = "",
    query_ids: str = "BIO-2,AGENT-1",
):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    selected = [name.strip() for name in runs.split(",") if name.strip()]
    functions = {
        "4b": lambda: confirm_4b.spawn(
            prediction,
            f"extensible_engine_confirmation_4b{output_suffix}",
            query_ids,
        ),
        "4b_2gpu": lambda: confirm_4b_2gpu.spawn(prediction),
        "32b": lambda: confirm_32b.spawn(prediction),
    }
    unknown = set(selected) - set(functions)
    if unknown:
        raise ValueError(f"unknown runs: {sorted(unknown)}")
    calls = {name: functions[name]() for name in selected}
    for name, call in calls.items():
        print(f"function call id: {call.object_id} ({name})", flush=True)
    for name, call in calls.items():
        print(f"waiting for {name}", flush=True)
        print(call.get(), flush=True)
