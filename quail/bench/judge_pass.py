"""Build the sf=0.1 QUAIL-B reference labels on Modal, one H100 per workload.

The labeling itself is `quail_b.labeling`, which runs on one GPU; this
module wraps it in Modal functions that mount the results volume.

    uv run modal run -m quail.bench.judge_pass

Reuse labels after an unrelated table changes in a new corpus:

    uv run modal run --detach -m quail.bench.judge_pass \
      --reuse-from-collection <collection> \
      --target-corpus <corpus> \
      --relabeled-workloads lepard
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from quail.runtime.volumes import hf_cache, kernel_cache, results_vol
from quail_b import labeling
from quail_b.labeling import (
    PREDICTION_TEXT,
    REUSE_PREDICTION_TEXT,
    SCALE_FACTOR,
    WORKLOADS,
)
from quail_b.store import GROUND_TRUTH_ROOT

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail_b")
)

data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow")
    .add_local_python_source("quail_b")
)

# Building the corpus reads the source datasets off HuggingFace, so it
# needs more than parquet - but not vllm. Its own image keeps
# prepare_corpus light, where data_image is only enough to read parquet
# back.
corpus_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow", "pandas", "huggingface_hub",
                 "datasets", "transformers>=5.2.0")
    .add_local_python_source("quail_b")
)

# Experiment cells attach to this existing app so its caches remain useful.
app = modal.App("quail-milestone1")

# Inside a container the labels live on the volume and every part
# file is committed as it lands; on the launching machine these
# assignments are harmless.
labeling.ROOT = Path("/results") / GROUND_TRUTH_ROOT
labeling.after_write = results_vol.commit


def parse_function_calls(value: str) -> dict[str, str]:
    calls = {}
    for item in value.split(","):
        try:
            workload, function_call_id = item.split("=", 1)
        except ValueError as exc:
            raise ValueError(
                "function calls must use workload=fc-id") from exc
        workload = workload.strip()
        function_call_id = function_call_id.strip()
        if workload in calls:
            raise ValueError(f"duplicate workload {workload!r}")
        calls[workload] = function_call_id
    missing = set(WORKLOADS) - set(calls)
    unknown = set(calls) - set(WORKLOADS)
    if missing or unknown:
        raise ValueError(
            f"function calls have missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
    if any(not value.startswith("fc-") for value in calls.values()):
        raise ValueError("every function call id must start with fc-")
    return calls


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def compact_ground_truth(collection_id: str) -> str:
    results_vol.reload()
    result = labeling.compact_ground_truth(collection_id)
    results_vol.commit()
    return json.dumps(result, sort_keys=True)


@app.function(
    image=corpus_image, memory=4096, timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/results": results_vol})
def prepare_corpus(sf: float = SCALE_FACTOR) -> str:
    results_vol.reload()
    result = labeling.prepare_corpus(sf)
    results_vol.commit()
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image, gpu="H100!", memory=98304, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/root/.cache/kernels": kernel_cache,
             "/results": results_vol})
def judge_workload(corpus_id: str, workload: str) -> str:
    """Label one workload's predicates on one GPU."""
    results_vol.reload()
    partial = labeling.judge_workload(corpus_id, workload)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(partial, sort_keys=True)


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def finalize_collection(sf: float, corpus_id: str, partials: str) -> str:
    """Assemble five workloads and activate their ground truth."""
    results_vol.reload()
    summary = labeling.finalize_collection(sf, corpus_id, json.loads(partials))
    results_vol.commit()
    return json.dumps(summary, sort_keys=True)


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def activate_reused_collection(
        sf: float, target_corpus_id: str, source_collection_id: str,
        relabeled_workloads: str) -> str:
    """Build one collection from new labels and verified unchanged tables."""
    results_vol.reload()
    summary = labeling.activate_reused_collection(
        sf, target_corpus_id, source_collection_id, relabeled_workloads)
    results_vol.commit()
    return json.dumps(summary, sort_keys=True)


@app.local_entrypoint()
def main(sf: float = SCALE_FACTOR, compact_collection: str | None = None,
         only: str | None = None, finalize_from: str | None = None,
         reuse_from_collection: str | None = None,
         relabeled_workloads: str = "lepard",
         target_corpus: str | None = None):
    """Run the five workloads side by side, then activate the result.

    ``--only imdb,fever`` restricts the pass to those workloads; the
    finalize step is skipped because a collection needs all label sets.
    """
    if compact_collection:
        call = compact_ground_truth.spawn(compact_collection)
        print(f"function call id: {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    prediction = (REUSE_PREDICTION_TEXT if reuse_from_collection
                  else PREDICTION_TEXT)
    print(f"PREDICTION: {prediction}", flush=True)

    if reuse_from_collection and target_corpus:
        corpus_id = target_corpus
        prepared = None
    else:
        call = prepare_corpus.spawn(sf)
        print(f"function call id (prepare_corpus): {call.object_id}",
              flush=True)
        prepared = json.loads(call.get())
        corpus_id = prepared["corpus"]["corpus_id"]
    if reuse_from_collection:
        call = activate_reused_collection.spawn(
            sf, corpus_id, reuse_from_collection, relabeled_workloads)
        print("function call id (activate_reused_collection): "
              f"{call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    if finalize_from:
        partials = {}
        for workload, function_call_id in parse_function_calls(
                finalize_from).items():
            result = modal.FunctionCall.from_id(function_call_id).get()
            partial = json.loads(result)
            if partial["workload"] != workload:
                raise ValueError(
                    f"{function_call_id} returned workload "
                    f"{partial['workload']!r}, expected {workload!r}")
            partials[workload] = partial
        call = finalize_collection.spawn(
            sf, corpus_id, json.dumps(partials))
        print(f"function call id (finalize_collection): {call.object_id}",
              flush=True)
        print(call.get(), flush=True)
        return
    if prepared["complete"] and not only:
        print(f"collection {prepared['collection_id']} is already complete",
              flush=True)
        return

    names = ([w.strip() for w in only.split(",")] if only
             else list(WORKLOADS))
    unknown = [w for w in names if w not in WORKLOADS]
    if unknown:
        raise ValueError(f"unknown workloads: {unknown}")

    calls = {w: judge_workload.spawn(corpus_id, w) for w in names}
    for w, c in calls.items():
        print(f"function call id (judge_workload {w}): {c.object_id}",
              flush=True)
    partials = {}
    for w, c in calls.items():
        partials[w] = json.loads(c.get())
        print(f"[main] {w} finished in "
              f"{partials[w]['total_wall_s']:.1f}s", flush=True)

    if only:
        print("--only was given, so the collection is not finalized",
              flush=True)
        return
    call = finalize_collection.spawn(sf, corpus_id, json.dumps(partials))
    print(f"function call id (finalize_collection): {call.object_id}",
          flush=True)
    print(call.get(), flush=True)
