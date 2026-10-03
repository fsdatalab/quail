"""Startup and weight memory of the two 0.6B bf16 models, each in a fresh container.

Times the vLLM import apart from loading the weights, and measures the
GPU bytes the loaded model holds (after the output head is reduced to
its answer rows), for Decision-2.0-Kai-0.6B and Qwen3 Reranker 0.6B.

Prediction: the vLLM import takes most of the startup for both models
and about the same time for each; loading the weights takes under 10 s.
The decision model holds about 1.19e9 bytes, the spec's w_mem_bytes
less its fp32 head file.

    uv run modal run experiments/cells/load_costs.py \
        2>&1 | tee /tmp/load_costs.log

The summary is written to /results/load_costs/<run>.json on the
quail-results volume.
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

MODELS = ("decision-2.0-kai-0.6b-bf16", "qwen3-reranker-0.6b-bf16")

app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": results,
}


@app.function(image=image, gpu="H100!", memory=65536, timeout=1800,
              volumes=VOLUMES)
def load(model: str) -> dict:
    """Seconds to import vLLM and to load the weights; GPU bytes held."""
    t0 = time.perf_counter()
    import torch
    import vllm  # noqa: F401
    import_s = time.perf_counter() - t0

    from transformers import AutoTokenizer

    from quail.backends.quail.executor.model import checkpoint_path, load_model
    from quail.logical import true_false_ids
    from quail.specs import MODELS as SPECS

    spec = SPECS[model]
    t0 = time.perf_counter()
    path = checkpoint_path(spec.hf_name, spec.revision)
    files_s = time.perf_counter() - t0
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    true_ids, false_ids = true_false_ids(tokenizer)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    load_model(path, answer_token_ids=true_ids | false_ids)
    torch.cuda.synchronize()
    return {"model": model, "import_vllm_s": import_s, "model_files_s": files_s,
            "load_model_s": time.perf_counter() - t0,
            "weight_bytes": torch.cuda.memory_allocated() - before,
            "spec_w_resident": spec.W_resident}


@app.local_entrypoint()
def main():
    run = time.strftime("%Y%m%d-%H%M%S")
    calls = [load.spawn(model) for model in MODELS]
    for call in calls:
        print(f"load function call id: {call.object_id}", flush=True)
    rows = [call.get() for call in calls]
    print(json.dumps(rows, indent=2), flush=True)
    save.remote(run, rows)


@app.function(image=image, volumes={"/results": results}, timeout=300)
def save(run: str, rows: list) -> str:
    import os

    path = f"/results/load_costs/{run}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    results.commit()
    return path
