r"""Time one DiffusionGemma chunk component by component.

The profiles put DiffusionGemma's filter at about 11 documents per
second on IMDB-3, some 18 times slower per row than Qwen3 4B, where
the cost model expects under 2 times. This cell packs one chunk of
synthetic documents, runs the pipeline once to warm up, then runs it
again with each component wrapped in a synchronizing timer: the
embedding, and per layer the norms, qkv projection, rotary, attention,
o projection, dense MLP, router, and experts. Sliding and
full-attention layers are summed separately.

    uv run modal run experiments/diffusion_gemma_layer_timing.py \
      --prediction "..." 2>&1 | tee results/diffusion-gemma-layer-timing.log

Writes /results/ablations/diffusion_gemma_layer_timing.json.
"""

import json
import os
import time

import modal

from quail.bench.requirements import quail_b_requirement

MODEL = "diffusion-gemma-26b-a4b-fp8"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "transformers>=5.8.0",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets>=5.0.1",
        "sqlglot>=27.0",
        "gigatoken>=0.10.0",
        quail_b_requirement(),
    )
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
)

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": results_vol,
}


def _save(name: str, value: dict) -> str:
    path = f"/results/ablations/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)
    results_vol.commit()
    return path


def _timer(torch, totals, name):
    def wrap(function):
        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = function(*args, **kwargs)
            torch.cuda.synchronize()
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - t0
            return out
        return timed
    return wrap


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def time_chunk(prediction: str, docs: int = 110, doc_tokens: int = 300,
               wide_head_kernel: str = "triton") -> str:
    import torch

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[MODEL]
    device = DEVICES["h100-sxm"]
    chunk_tokens = budgets.chunk_budget(spec, device)
    model = load_model(spec.hf_name, max_batched_tokens=chunk_tokens)
    full_pages, sliding_pages = budgets.arena_pages(spec, device, chunk_tokens)
    arena = KVArena(n_layers=spec.layers, n_pages=full_pages,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=sliding_pages)
    pipeline = build_pipeline(spec, model, arena,
                              wide_head_kernel=wide_head_kernel)
    tail = list(range(100, 116))
    groups = []
    for index in range(docs):
        key = ("t", index)
        arena.activate(key, doc_tokens + len(tail),
                       capacity_tokens=doc_tokens + len(tail) + 256,
                       base_tokens=doc_tokens)
        prefix = [1000 + (i % 500) for i in range(doc_tokens)]
        groups.append(dict(key=key, prefix=prefix, f=doc_tokens,
                           suffixes=[tail]))

    def run():
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                           canvas=pipeline.canvas_ids,
                           answer_row=pipeline.canvas_answer_row)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pipeline.forward_chunk(chunk)
        torch.cuda.synchronize()
        for key in chunk.temporary_keys:
            arena.free_key(key)
        return chunk.tokens, time.perf_counter() - t0

    rows, warm_s = run()
    rows, plain_s = run()
    print(f"{wide_head_kernel}: {rows} rows in {plain_s:.2f} s, "
          f"{rows / plain_s:.0f} rows/s", flush=True)

    totals = {}
    engine = pipeline.engine
    original_attention = engine.attention_unified

    def attention(q3, k3, v3, meta, **kwargs):
        name = "attention_sliding" if kwargs.get("window") else "attention_full"
        return _timer(torch, totals, name)(original_attention)(
            q3, k3, v3, meta, **kwargs)

    def wrap_forward(module, name):
        # a module's forward is a plain attribute; the module itself
        # cannot be replaced by a function
        module.forward = _timer(torch, totals, name)(module.forward)

    engine.attention_unified = attention
    wrap_forward(pipeline.embed, "embed")
    wrap_forward(pipeline.final_norm, "final_norm")
    for layer in pipeline.layers:
        attn = layer.self_attn
        kind = "sliding" if attn.is_sliding else "full"
        wrap_forward(attn.qkv_proj, f"qkv_proj_{kind}")
        wrap_forward(attn.o_proj, f"o_proj_{kind}")
        wrap_forward(attn.rotary_emb, "rotary")
        for norm in (attn.q_norm, attn.k_norm, attn.v_norm):
            wrap_forward(norm, "qkv_norms")
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm",
                     "post_feedforward_layernorm_1",
                     "pre_feedforward_layernorm_2",
                     "post_feedforward_layernorm_2"):
            wrap_forward(getattr(layer, name), "layer_norms")
        wrap_forward(layer.mlp, "dense_mlp")
        wrap_forward(layer.router, "router")
        wrap_forward(layer.moe, "experts")
    rows, timed_s = run()
    accounted = sum(totals.values())
    result = {
        "prediction": prediction,
        "wide_head_kernel": wide_head_kernel,
        "docs": docs, "doc_tokens": doc_tokens, "rows": rows,
        "chunk_tokens": chunk_tokens,
        "warm_s": warm_s, "plain_s": plain_s, "timed_s": timed_s,
        "rows_per_s_plain": rows / plain_s,
        "components_s": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
        "accounted_s": accounted,
        "unaccounted_s": timed_s - accounted,
    }
    result["volume_path"] = _save(
        f"diffusion_gemma_layer_timing_{wide_head_kernel}", result)
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 110, doc_tokens: int = 300,
         kernels: str = "triton,fa4"):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    calls = {kernel: time_chunk.spawn(prediction, docs, doc_tokens, kernel)
             for kernel in kernels.split(",")}
    for kernel, call in calls.items():
        print(f"function call id: {call.object_id} ({kernel})", flush=True)
    for kernel, call in calls.items():
        print(call.get(), flush=True)
