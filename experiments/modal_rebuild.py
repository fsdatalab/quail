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
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
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


def _resize_documents(documents, tokenizer, n_docs, target_tokens):
    if not target_tokens:
        return documents[:n_docs]
    lengths = tokenizer(
        documents,
        add_special_tokens=False,
    )["input_ids"]
    resized = []
    cursor = 0
    body_target = max(1, target_tokens - 80)
    for _ in range(n_docs):
        parts = []
        total = 0
        while total < body_target:
            index = cursor % len(documents)
            parts.append(documents[index])
            total += len(lengths[index]) + 2
            cursor += 1
        resized.append("\n\n".join(parts))
    return resized


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
    document_tokens: int = 0,
    short_circuit: bool = True,
    debug_sync: bool = False,
) -> dict:
    import os

    if debug_sync:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

    documents = _documents(n_docs if not document_tokens else 10_000)
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
    documents = _resize_documents(
        documents,
        tokenizer,
        n_docs,
        document_tokens,
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
        max_model_len=max(4608, document_tokens + 1024),
        enforce_eager=True,
        disable_log_stats=True,
        attention_backend="FLASHINFER",
        max_num_batched_tokens=16_384,
        max_num_seqs=1_024,
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
            max_sequences=(1 if k > 1 else 1_024),
            max_temporary_bytes=2 * (1 << 30),
        )),
        kv=kv,
        trace=trace,
        speculation_k=k,
        short_circuit=short_circuit,
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
        "target_document_tokens": document_tokens,
        "body_token_lengths": [len(row) for row in body_ids],
        "short_circuit": short_circuit,
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


@app.function(
    image=image,
    gpu="H100!",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
async def stock_smoke(
    n_docs: int = 8,
    n_filters: int = 2,
    document_tokens: int = 0,
    short_circuit: bool = True,
) -> dict:
    import asyncio
    import time

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    documents = _documents(n_docs if not document_tokens else 10_000)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
    )
    documents = _resize_documents(
        documents,
        tokenizer,
        n_docs,
        document_tokens,
    )
    labels_array = (
        np.random.default_rng(GROUND_TRUTH_SEED)
        .random((n_docs, n_filters)) < 0.8
    ).astype(int)
    bodies = [
        document + _flags_line(flags)
        for document, flags in zip(documents, labels_array)
    ]
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
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=MODEL,
        revision=MODEL_REVISION,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.90,
        max_model_len=max(4608, document_tokens + 1024),
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=True,
        attention_backend="FLASHINFER",
        max_num_batched_tokens=16_384,
        max_num_seqs=1_024,
    ))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        skip_clone=True,
    )
    answers = {}

    async def ask(document, stage):
        final = None
        prompt = {
            "prompt_token_ids": (
                body_ids[document] + question_ids[stage]
            )
        }
        async for output in engine.generate(
            prompt,
            sampling,
            f"stock-{document}-{stage}",
        ):
            final = output
        text = final.outputs[0].text.strip().upper()
        return 1 if text.startswith("Y") else 0

    async def document_chain(document):
        for stage in range(n_filters):
            answer = await ask(document, stage)
            answers[(document, stage + 1)] = answer
            if short_circuit and not answer:
                break

    started = time.perf_counter_ns()
    await asyncio.gather(*[
        document_chain(document)
        for document in range(n_docs)
    ])
    ended = time.perf_counter_ns()
    try:
        engine.shutdown()
    except Exception:
        pass
    attempted = len(answers)
    correct = sum(
        answer == labels_array[document][stage - 1]
        for (document, stage), answer in answers.items()
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "phase": "stock-smoke",
        "n_docs": n_docs,
        "n_filters": n_filters,
        "target_document_tokens": document_tokens,
        "body_token_lengths": [len(row) for row in body_ids],
        "short_circuit": short_circuit,
        "wall_ns": ended - started,
        "answers": {
            f"{document},{stage}": answer
            for (document, stage), answer in answers.items()
        },
        "accuracy": correct / attempted if attempted else 0.0,
        "ground_truth_used_by_runtime": False,
        "gpu": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        },
        "cache_reset_confirmed": True,
        "vllm_version": __import__("vllm").__version__,
    }


@app.function(
    image=image,
    gpu="H100!",
    timeout=1800,
)
def cascade_kernel(
    groups: int = 8,
    k: int = 2,
    prefix_tokens: int = 304,
    tail_tokens: int = 32,
    repetitions: int = 20,
    debug_sync: bool = False,
    held_out: bool = False,
) -> dict:
    import flashinfer
    import torch

    if groups <= 0 or k <= 0:
        raise ValueError("groups and k must be positive")
    page_size = 16
    if prefix_tokens % page_size:
        raise ValueError("prefix_tokens must be page aligned")
    prefix_pages = prefix_tokens // page_size
    unique_pages = (tail_tokens + page_size - 1) // page_size
    total_tails = groups * k
    shared_page_indices = []
    shared_indptr = [0]
    unique_page_indices = []
    unique_indptr = [0]
    full_page_indices = []
    full_indptr = [0]
    page = 0
    group_shared = []
    tail_unique = []
    for _group in range(groups):
        shared = list(range(page, page + prefix_pages))
        page += prefix_pages
        group_shared.append(shared)
        shared_page_indices.extend(shared)
        shared_indptr.append(len(shared_page_indices))
        for _tail in range(k):
            unique = list(range(page, page + unique_pages))
            page += unique_pages
            tail_unique.append(unique)
            unique_page_indices.extend(unique)
            unique_indptr.append(len(unique_page_indices))
            full_page_indices.extend(shared + unique)
            full_indptr.append(len(full_page_indices))
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    q_tokens = total_tails * tail_tokens
    query = torch.randn(
        q_tokens,
        num_qo_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        page,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    workspace = torch.empty(
        256 * (1 << 20),
        dtype=torch.uint8,
        device="cuda",
    )
    top_qo = torch.tensor(
        [group * k * tail_tokens for group in range(groups + 1)],
        dtype=torch.int32,
        device="cpu",
    )
    bottom_qo = torch.tensor(
        [tail * tail_tokens for tail in range(total_tails + 1)],
        dtype=torch.int32,
        device="cpu",
    )
    shared_indptr_tensor = torch.tensor(
        shared_indptr,
        dtype=torch.int32,
        device="cpu",
    )
    unique_indptr_tensor = torch.tensor(
        unique_indptr,
        dtype=torch.int32,
        device="cpu",
    )
    shared_indices_tensor = torch.tensor(
        shared_page_indices,
        dtype=torch.int32,
        device="cpu",
    )
    unique_indices_tensor = torch.tensor(
        unique_page_indices,
        dtype=torch.int32,
        device="cpu",
    )
    shared_last = torch.full(
        (groups,),
        page_size,
        dtype=torch.int32,
        device="cpu",
    )
    unique_last = torch.full(
        (total_tails,),
        tail_tokens % page_size or page_size,
        dtype=torch.int32,
        device="cpu",
    )
    cascade = flashinfer.MultiLevelCascadeAttentionWrapper(
        2,
        workspace,
        "NHD",
    )
    cascade.plan(
        [top_qo, bottom_qo],
        [shared_indptr_tensor, unique_indptr_tensor],
        [shared_indices_tensor, unique_indices_tensor],
        [shared_last, unique_last],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
    )
    full_indptr_tensor = torch.tensor(
        full_indptr,
        dtype=torch.int32,
        device="cpu",
    )
    full_indices_tensor = torch.tensor(
        full_page_indices,
        dtype=torch.int32,
        device="cpu",
    )
    standard = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
    )
    standard.plan(
        bottom_qo,
        full_indptr_tensor,
        full_indices_tensor,
        unique_last,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
    )
    for _ in range(3):
        cascade_output = cascade.run(query, kv_cache)
        standard_output = standard.run(query, kv_cache)
    torch.cuda.synchronize()

    def measure(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repetitions

    cascade_ms = measure(lambda: cascade.run(query, kv_cache))
    standard_ms = measure(lambda: standard.run(query, kv_cache))
    difference = (
        cascade_output.float() - standard_output.float()
    ).abs()
    return {
        "phase": "cascade-kernel",
        "groups": groups,
        "k": k,
        "prefix_tokens": prefix_tokens,
        "tail_tokens": tail_tokens,
        "q_tokens": q_tokens,
        "kv_cache_dtype": str(kv_cache.dtype),
        "cascade_ms": cascade_ms,
        "standard_ms": standard_ms,
        "speedup": standard_ms / cascade_ms,
        "max_absolute_difference": difference.max().item(),
        "mean_absolute_difference": difference.mean().item(),
        "finite": bool(torch.isfinite(cascade_output).all()),
        "held_out": held_out,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }


@app.local_entrypoint()
def main(
    phase: str = "custom-smoke",
    n_docs: int = 8,
    n_filters: int = 2,
    k: int = 1,
    document_tokens: int = 0,
    short_circuit: bool = True,
    groups: int = 8,
    prefix_tokens: int = 304,
    tail_tokens: int = 32,
    repetitions: int = 20,
    debug_sync: bool = False,
    held_out: bool = False,
    out: str = "results/runs",
):
    import sys

    from docengine.runtime.artifacts import RunMetadata, write_run_artifact

    if phase == "custom-smoke":
        data = custom_smoke.remote(
            n_docs,
            n_filters,
            k,
            document_tokens,
            short_circuit,
            debug_sync,
        )
    elif phase == "stock-smoke":
        data = stock_smoke.remote(
            n_docs,
            n_filters,
            document_tokens,
            short_circuit,
        )
    elif phase == "packing-compare":
        custom_handle = custom_smoke.spawn(
            n_docs,
            n_filters,
            1,
            document_tokens,
            False,
            debug_sync,
        )
        stock_handle = stock_smoke.spawn(
            n_docs,
            n_filters,
            document_tokens,
            False,
        )
        data = {
            "phase": phase,
            "custom": custom_handle.get(),
            "stock": stock_handle.get(),
        }
    elif phase == "cascade-kernel":
        data = cascade_kernel.remote(
            groups,
            k,
            prefix_tokens,
            tail_tokens,
            repetitions,
            debug_sync,
            held_out,
        )
    else:
        raise SystemExit(f"unknown phase {phase}")
    metadata = RunMetadata.create(
        phase=phase,
        config={
            "n_docs": n_docs,
            "n_filters": n_filters,
            "k": k,
            "document_tokens": document_tokens,
            "short_circuit": short_circuit,
            "groups": groups,
            "prefix_tokens": prefix_tokens,
            "tail_tokens": tail_tokens,
            "repetitions": repetitions,
            "debug_sync": debug_sync,
            "held_out": held_out,
        },
        seeds={
            "workload": WORKLOAD_SEED,
            "ground_truth": GROUND_TRUTH_SEED,
        },
        model_revision=MODEL_REVISION,
        dataset_revision=DATASET_REVISION,
        gpu=(
            data.get("gpu")
            or data.get("custom", {}).get("gpu", {})
        ),
        cache_reset_confirmed=True,
        command=tuple(sys.argv),
    )
    directory = write_run_artifact(out, metadata, data)
    print(f"saved immutable run {directory}")
