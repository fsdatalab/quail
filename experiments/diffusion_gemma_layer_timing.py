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
               wide_head_kernel: str = "triton",
               random_tokens: bool = False) -> str:
    import numpy as np
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
    # random ids route the experts as broadly as real text does; the
    # repeating pattern sends every row to the same few experts
    rng = np.random.default_rng(1)
    for index in range(docs):
        key = ("t", index)
        arena.activate(key, doc_tokens + len(tail),
                       capacity_tokens=doc_tokens + len(tail) + 256,
                       base_tokens=doc_tokens)
        if random_tokens:
            prefix = [int(t) for t in rng.integers(1000, spec.vocab - 1000,
                                                   doc_tokens)]
        else:
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
        "random_tokens": random_tokens,
        "docs": docs, "doc_tokens": doc_tokens, "rows": rows,
        "chunk_tokens": chunk_tokens,
        "warm_s": warm_s, "plain_s": plain_s, "timed_s": timed_s,
        "rows_per_s_plain": rows / plain_s,
        "components_s": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
        "accounted_s": accounted,
        "unaccounted_s": timed_s - accounted,
    }
    suffix = "_random" if random_tokens else ""
    result["volume_path"] = _save(
        f"diffusion_gemma_layer_timing_{wide_head_kernel}{suffix}", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def time_loop(prediction: str, reviews: int = 1000) -> str:
    """Run the filter loop over IMDB reviews with its phase counters on."""
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    from quail.backends.quail.executor import loop
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS
    from quail_b.data import build_sets

    spec = MODELS[MODEL]
    device = DEVICES["h100-sxm"]
    chunk_tokens = budgets.chunk_budget(spec, device)
    model = load_model(spec.hf_name, max_batched_tokens=chunk_tokens)
    full_pages, sliding_pages = budgets.arena_pages(spec, device, chunk_tokens,
                                                    mean_doc_tokens=300)
    arena = KVArena(n_layers=spec.layers, n_pages=full_pages,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=sliding_pages)
    pipeline = build_pipeline(spec, model, arena)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    rows = AnswerRows.from_tokenizer(torch, torch.nn.functional, model, tokenizer)
    answers = AsyncAnswers(torch, rows)
    data = build_sets("/results/quailb_data", 0.1)
    table = pq.read_table(f"{data}/reviews.parquet")
    texts = table.column("body").to_pylist()[:reviews]

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    docs = [tok(spec.turn_prefix + "DOCUMENT:\n" + text) for text in texts]
    question = tok("\n\nYou are performing a data processing task. Evaluate "
                   "TRUE or FALSE for the following question: Does the "
                   "review mention a positive aspect?\nANSWER:"
                   + spec.turn_suffix)
    with torch.inference_mode():
        loop.run_filter(torch, arena, pipeline, answers, docs[:120],
                        [question], chunk_tokens, arena_writes=True)
        torch.cuda.synchronize()
        timing = {}
        started = time.perf_counter()
        _answers, spans, tokens = loop.run_filter(
            torch, arena, pipeline, answers, docs, [question], chunk_tokens,
            timing=timing, arena_writes=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
    gpu_s = sum(e0.elapsed_time(e1) for _, e0, e1 in spans) / 1000
    result = {
        "prediction": prediction,
        "reviews": len(docs), "fresh_tokens": tokens,
        "doc_tokens_mean": sum(map(len, docs)) / len(docs),
        "wall_s": wall, "gpu_s": gpu_s, "chunks": len(spans),
        "rows_per_s": tokens / wall,
        "cpu_phases_s": dict(sorted(
            ((k, v) for k, v in timing.items() if isinstance(v, float)),
            key=lambda kv: -kv[1])),
        "n_chunks_counter": timing.get("n_chunks"),
    }
    result["volume_path"] = _save("diffusion_gemma_loop_timing", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def micro(prediction: str, rows: int = 62920) -> str:
    """Time norm and rotary calls through the modules and as raw kernels."""
    import torch
    from vllm import _custom_ops as ops

    from quail.backends.quail.executor.model import load_model
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    model = load_model(spec.hf_name, max_batched_tokens=rows)
    layer = model.model.layers[0]
    full = model.model.layers[5]
    x = torch.randn(rows, 2816, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda")

    def timed(name, function, repeats=5):
        function()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            function()
        torch.cuda.synchronize()
        return name, (time.perf_counter() - t0) / repeats

    norm = layer.input_layernorm
    out = torch.empty_like(x)
    results = dict([
        timed("norm_module_ms", lambda: norm(x)),
        timed("norm_raw_kernel_ms", lambda: torch.ops._C.rms_norm(
            out, x, norm.weight, norm.variance_epsilon)),
        timed("norm_native_torch_ms", lambda: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                    + norm.variance_epsilon)
        ).to(x.dtype) * norm.weight),
    ])
    for name, attn in (("sliding", layer.self_attn), ("full", full.self_attn)):
        H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        q = torch.randn(rows, H * D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(rows, KH * D, device="cuda", dtype=torch.bfloat16)
        rope = attn.rotary_emb
        results.update([
            timed(f"rotary_module_{name}_ms", lambda: rope(positions, q, k)),
            timed(f"rotary_raw_kernel_{name}_ms", lambda: ops.rotary_embedding(
                positions, q, k, D, rope.cos_sin_cache.to(q.dtype),
                rope.is_neox_style)),
            timed(f"qk_norm_module_{name}_ms", lambda: (
                attn.q_norm(q.unflatten(-1, (H, D))),
                attn.k_norm(k.unflatten(-1, (KH, D))))),
        ])
        results[f"rotary_class_{name}"] = type(rope).__name__
        results[f"cos_sin_cache_{name}"] = (
            str(rope.cos_sin_cache.dtype), list(rope.cos_sin_cache.shape))
    results = {key: (round(value * 1000, 3) if isinstance(value, float) else value)
               for key, value in results.items()}
    results["prediction"] = prediction
    results["rows"] = rows
    results["norm_dispatch"] = type(norm).__name__
    results["volume_path"] = _save("diffusion_gemma_micro_timing", results)
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 110, doc_tokens: int = 300,
         kernels: str = "triton,fa4", random_tokens: bool = False,
         runs: str = "chunk,loop"):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    calls = {}
    if "chunk" in runs:
        for kernel in kernels.split(","):
            calls[kernel] = time_chunk.spawn(prediction, docs, doc_tokens,
                                             kernel, random_tokens)
    if "loop" in runs:
        calls["loop"] = time_loop.spawn(prediction)
    if "micro" in runs:
        calls["micro"] = micro.spawn(prediction)
    for name, call in calls.items():
        print(f"function call id: {call.object_id} ({name})", flush=True)
    for name, call in calls.items():
        print(call.get(), flush=True)
