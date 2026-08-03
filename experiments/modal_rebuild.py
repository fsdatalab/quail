"""Custom DocEngine runtime experiments on one Modal H100."""

import modal


app = modal.App("docengine-rebuild")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub",
        "pandas",
        "pyarrow",
        "numpy",
    )
    .env({
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_python_source("docengine")
)
hf_cache = modal.Volume.from_name(
    "docengine-hf-cache",
    create_if_missing=True,
)

MODEL = "Qwen/Qwen3-4B-FP8"
MODEL_REVISION = "96b30dc13593a244a5e59e84687309f53c375cfa"
DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
WORKLOAD_SEED = 20260731
GROUND_TRUTH_SEED = 424249


def _documents(n_docs):
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        frames.append(pd.read_parquet(path)["text"])
    pool = list(frames[0]) + list(frames[1])
    indices = sorted(
        np.random.default_rng(WORKLOAD_SEED).choice(
            len(pool),
            size=10_000,
            replace=False,
        )
    )
    return [pool[index] for index in indices[:n_docs]]


def _flags_line(flags):
    return "\n\n[FLAGS] " + " ".join(
        f"FLAG_{stage + 1}={'YES' if value else 'NO'}"
        for stage, value in enumerate(flags)
    )


def _question(stage):
    return (
        "\n\nExample: if the line said [FLAGS] FLAG_9=NO, then FLAG_9 "
        f"has value NO.\nInstruction: output only the value of FLAG_{stage} "
        f"from the [FLAGS] line above.\nFLAG_{stage}="
    )


@app.function(
    image=image,
    gpu="H100!",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def custom_smoke(
    n_docs: int = 8,
    n_filters: int = 2,
    k: int = 1,
) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams

    from docengine.configs import QWEN3_4B_FP8
    from docengine.runtime.batching import (
        BatchLimits,
        VariableLengthBatchPacker,
    )
    from docengine.runtime.custom import DocEngineRuntime
    from docengine.runtime.kv import KVPageAllocator
    from docengine.runtime.protocol import (
        FilterQuery,
        GroundTruthLabels,
        evaluate_answers,
    )
    from docengine.runtime.trace import TraceRecorder
    from docengine.runtime.vllm_runner import (
        VLLMModelRunner,
        initialize_model_executor,
    )

    documents = _documents(n_docs)
    labels_array = (
        np.random.default_rng(GROUND_TRUTH_SEED)
        .random((n_docs, n_filters)) < 0.8
    ).astype(int)
    bodies = [
        document + _flags_line(flags)
        for document, flags in zip(documents, labels_array)
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
    )
    body_ids = tokenizer(
        bodies,
        add_special_tokens=False,
    )["input_ids"]
    question_ids = [
        tokenizer(
            _question(stage + 1),
            add_special_tokens=False,
        )["input_ids"]
        for stage in range(n_filters)
    ]
    yes_ids = set()
    for text in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        if ids:
            yes_ids.add(ids[0])
    query = FilterQuery.from_sequences(
        body_ids,
        question_ids,
        yes_ids,
    )
    ground_truth = GroundTruthLabels.from_sequences(labels_array.tolist())
    ground_truth.validate_for(query)

    vllm_config = EngineArgs(
        model=MODEL,
        revision=MODEL_REVISION,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.90,
        max_model_len=4608,
        enforce_eager=True,
        disable_log_stats=True,
        attention_backend="FLASHINFER",
    ).create_engine_config()
    executor, kv_config = initialize_model_executor(vllm_config)
    page_size = vllm_config.cache_config.block_size
    kv = KVPageAllocator(
        total_pages=kv_config.num_blocks,
        page_size_tokens=page_size,
        bytes_per_token=QWEN3_4B_FP8.kappa,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        skip_clone=True,
    )
    runner = VLLMModelRunner(
        query=query,
        kv=kv,
        model_executor=executor,
        sampling_params=sampling,
        kv_cache_groups=len(kv_config.kv_cache_groups),
    )
    trace = TraceRecorder()
    runtime = DocEngineRuntime(
        query=query,
        runner=runner,
        packer=VariableLengthBatchPacker(BatchLimits(
            max_new_tokens=16_384,
            max_sequences=1_024,
            max_temporary_bytes=2 * (1 << 30),
        )),
        kv=kv,
        trace=trace,
        speculation_k=k,
    )
    result = runtime.run()
    evaluation = evaluate_answers(result.answers, ground_truth)
    try:
        executor.shutdown()
    except Exception:
        pass
    properties = torch.cuda.get_device_properties(0)
    return {
        "phase": "custom-smoke",
        "n_docs": n_docs,
        "n_filters": n_filters,
        "k": k,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "vllm_version": __import__("vllm").__version__,
        "kv_cache_dtype": "fp8",
        "kv_page_size": page_size,
        "kv_pages": kv_config.num_blocks,
        "steps": result.steps,
        "wall_ns": result.wall_ns,
        "survivors": list(result.survivors),
        "answers": {
            f"{document},{stage}": answer
            for (document, stage), answer in result.answers.items()
        },
        "accuracy": evaluation.accuracy,
        "wrong": [list(item) for item in evaluation.wrong],
        "ground_truth_used_by_runtime": False,
        "trace": [
            {
                "step": row.step,
                "prefill_tokens": row.prefill_tokens,
                "decode_tokens": row.decode_tokens,
                "sequence_count": row.sequence_count,
                "duration_ns": row.duration_ns,
                "planning_ns": row.planning_ns,
                "unused_capacity_reason": row.unused_capacity_reason,
            }
            for row in trace.records
        ],
        "gpu": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        },
        "cache_reset_confirmed": True,
    }


@app.local_entrypoint()
def main(
    phase: str = "custom-smoke",
    n_docs: int = 8,
    n_filters: int = 2,
    k: int = 1,
    out: str = "results/runs",
):
    import sys

    from docengine.runtime.artifacts import RunMetadata, write_run_artifact

    if phase != "custom-smoke":
        raise SystemExit(f"unknown phase {phase}")
    data = custom_smoke.remote(n_docs, n_filters, k)
    metadata = RunMetadata.create(
        phase=phase,
        config={"n_docs": n_docs, "n_filters": n_filters, "k": k},
        seeds={
            "workload": WORKLOAD_SEED,
            "ground_truth": GROUND_TRUTH_SEED,
        },
        model_revision=MODEL_REVISION,
        dataset_revision=DATASET_REVISION,
        gpu=data["gpu"],
        cache_reset_confirmed=True,
        command=tuple(sys.argv),
    )
    directory = write_run_artifact(out, metadata, data)
    print(f"saved immutable run {directory}")
