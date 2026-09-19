"""Run the GPU tests under tests/gpu on Modal.

    uv run modal run experiments/run_gpu_tests.py 2>&1 | tee results/gpu-tests.log

Prints pytest's output and fails when a test fails.
"""

import modal

# the uv the images sync with; pyproject.toml requires this version
UV_VERSION = "0.12.13"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    # quail's pinned dependencies and the dev group (pytest among
    # them) from uv.lock; the quail source and the tests are mounted
    .uv_sync(groups=["dev"], uv_version=UV_VERSION)
    .env({
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail")
    .add_local_dir("tests/gpu", remote_path="/root/tests/gpu")
)

app = modal.App("quail-milestone1")
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def run_tests(keyword: str = "") -> int:
    """Run pytest over tests/gpu and return its exit code."""
    import pytest

    args = ["/root/tests/gpu", "-x", "-q", "-s", "-p", "no:cacheprovider"]
    if keyword:
        args += ["-k", keyword]
    return int(pytest.main(args))


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def parity_report() -> str:
    """Where Quail's forward departs from stock vLLM's, layer by layer.

    Stock records every layer's output for the test prompts; Quail
    runs its stack cut after each layer. Prints the median relative
    error per layer, the per-layer-input setting, and the answer
    margins at the last prompt row.
    """
    import json
    import subprocess
    import sys

    import torch

    sys.path.insert(0, "/root/tests/gpu")
    import test_diffusion_gemma_forward as t

    path = "/tmp/stock_layers.pt"
    subprocess.run([sys.executable, t.__file__, path, "all"], check=True)
    stock = torch.load(path)
    prompts = t._prompt_ids()

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.cost.budgets import PAGE_TOKENS
    from quail.specs import MODELS

    spec = MODELS[t.MODEL]
    model = load_model(spec.hf_name, max_batched_tokens=8192,
                       moe_backend=spec.moe_backend)
    arena = KVArena(n_layers=spec.layers, n_pages=2048,
                    page_tokens=PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=2048)
    pipeline = build_pipeline(spec, model, arena)
    layer0 = pipeline.layers[0]
    report = {
        "hidden_size_per_layer_input": getattr(
            layer0, "hidden_size_per_layer_input", None),
        "layer_scalar_0": float(layer0.layer_scalar),
        "per_layer": [],
    }
    groups = []
    for index, ids in enumerate(prompts):
        key = ("review", index)
        split = len(ids) - 8
        arena.activate(key, len(ids), capacity_tokens=len(ids) + 64,
                       base_tokens=split)
        groups.append(dict(key=key, prefix=ids[:split], f=split,
                           suffixes=[ids[split:]]))
    chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                       canvas=pipeline.canvas_ids,
                       answer_row=pipeline.canvas_answer_row)
    all_layers = list(pipeline.layers)

    def rows_after(layer_index):
        pipeline.layers = all_layers[:layer_index + 1]
        with torch.inference_mode():
            hidden = pipeline.backbone_rows(chunk)
        out, offset = [], 0
        for ids in prompts:
            out.append(hidden[offset:offset + len(ids)].float().cpu())
            offset += len(ids) + len(pipeline.canvas_ids)
        return out

    for layer_index in range(len(all_layers)):
        ours = rows_after(layer_index)
        rel = torch.cat([
            (o - s[layer_index]).norm(dim=-1)
            / s[layer_index].norm(dim=-1).clamp_min(1e-6)
            for o, s in zip(ours, stock)])
        first = torch.cat([
            ((o - s[layer_index]).norm(dim=-1)
             / s[layer_index].norm(dim=-1).clamp_min(1e-6))[:4]
            for o, s in zip(ours, stock)])
        entry = dict(layer=layer_index, median=round(rel.median().item(), 4),
                     p99=round(rel.quantile(0.99).item(), 4),
                     first_rows_median=round(first.median().item(), 4))
        report["per_layer"].append(entry)
        print(json.dumps(entry), flush=True)
    pipeline.layers = all_layers
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(keyword: str = "", parity: bool = False):
    if parity:
        call = parity_report.spawn()
        print(f"function call id: {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    call = run_tests.spawn(keyword)
    print(f"function call id: {call.object_id}", flush=True)
    code = call.get()
    print(f"pytest exit code: {code}", flush=True)
    if code:
        raise SystemExit(code)
