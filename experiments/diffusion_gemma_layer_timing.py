r"""Time DiffusionGemma's kernels on one H100, cell by cell.

    uv run modal run experiments/diffusion_gemma_layer_timing.py \
      --prediction "..." --runs chunk,loop 2>&1 | tee results/<stamp>.log

Cells, chosen with --runs:

- chunk: pack one chunk of synthetic documents and time each
  component of the pipeline with a synchronizing timer (the norms,
  projections, rotary, attention, dense MLP, router, and experts),
  sliding and full-attention layers summed separately. --profile
  saves a Chrome trace.
- gemms: the dense fp8 GEMM kernels at the chunk's row count.
- tiles: the fused MoE kernel's tile configurations at 32k and 64k
  rows; the winners are the table in executor/moe_configs.py.
- loop: the filter loop over IMDB reviews, end to end.
- micro: the fused elementwise kernels against their unfused parts.
- stock: one stock vLLM prefill pass under the profiler, for the
  kernel-per-layer figure.

Each cell prints its Modal function call id and writes its record to
/results/ablations/diffusion_gemma_<cell>.json on the quail-results
volume.
"""

import json
import os
import time

import modal

from quail.bench.images import gpu_image

MODEL = "diffusion-gemma-26b-a4b-fp8"

image = gpu_image()

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
def gemms(prediction: str, rows: int = 34870) -> str:
    """Time the dense fp8 GEMM shapes of one layer on two kernels.

    vLLM's CUTLASS scaled_mm against cuBLASLt through torch._scaled_mm,
    both with per-row activation scales and per-channel weight scales.
    """
    import torch
    from vllm import _custom_ops as ops

    from quail.backends.quail.executor.model import load_model
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[MODEL]
    device = DEVICES["h100-sxm"]
    model = load_model(spec.hf_name,
                       max_batched_tokens=budgets.chunk_budget(spec, device),
                       moe_backend="triton")
    layers = model.model.layers
    linears = {
        "qkv_sliding": layers[0].self_attn.qkv_proj,
        "o_sliding": layers[0].self_attn.o_proj,
        "qkv_full": layers[5].self_attn.qkv_proj,
        "o_full": layers[5].self_attn.o_proj,
        "gate_up": layers[0].mlp.gate_up_proj,
        "down": layers[0].mlp.down_proj,
    }

    def timed(fn, repeats=20):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    result = {"prediction": prediction, "rows": rows, "shapes": {}}
    peak = device.peak_flops
    for name, module in linears.items():
        weight = module.weight            # (K, N) fp8, column-major B
        scale_b = module.weight_scale     # (N, 1) float32
        k, n = weight.shape
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        cutlass = timed(lambda: ops.cutlass_scaled_mm(
            xq, weight, scale_a=xs, scale_b=scale_b,
            out_dtype=torch.bfloat16))
        ref = ops.cutlass_scaled_mm(xq, weight, scale_a=xs, scale_b=scale_b,
                                    out_dtype=torch.bfloat16)
        entry = {"K": k, "N": n, "cutlass_us": cutlass * 1e6,
                 "ideal_us": 2 * rows * k * n / peak * 1e6}
        try:
            b = weight.t().contiguous().t() if not weight.is_contiguous() \
                else weight
            out = torch._scaled_mm(xq, b, scale_a=xs, scale_b=scale_b.t(),
                                   out_dtype=torch.bfloat16)
            cublas = timed(lambda: torch._scaled_mm(
                xq, b, scale_a=xs, scale_b=scale_b.t(),
                out_dtype=torch.bfloat16))
            entry["cublaslt_us"] = cublas * 1e6
            entry["cublaslt_rel"] = ((out.float() - ref.float()).pow(2).mean()
                                     .sqrt() / ref.float().pow(2).mean()
                                     .sqrt()).item()
        except Exception as error:  # noqa: BLE001 - report the shape
            entry["cublaslt_error"] = repr(error)[:200]
        result["shapes"][name] = entry
        print(name, {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in entry.items()}, flush=True)
    result["volume_path"] = _save("diffusion_gemma_gemms", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def moe_tiles(prediction: str, docs: int = 110, doc_tokens: int = 300) -> str:
    """Time the experts under a sweep of Triton fused MoE tile configs."""
    import itertools

    import numpy as np
    import torch
    from vllm.model_executor.layers.fused_moe import override_config

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[MODEL]
    device = DEVICES["h100-sxm"]
    chunk_tokens = budgets.chunk_budget(spec, device)
    model = load_model(spec.hf_name, max_batched_tokens=chunk_tokens,
                       moe_backend="triton")
    full_pages, sliding_pages = budgets.arena_pages(spec, device, chunk_tokens)
    arena = KVArena(n_layers=spec.layers, n_pages=full_pages,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=sliding_pages)
    pipeline = build_pipeline(spec, model, arena)
    tail = list(range(100, 116))
    rng = np.random.default_rng(1)
    groups = []
    for index in range(docs):
        key = ("t", index)
        arena.activate(key, doc_tokens + len(tail),
                       capacity_tokens=doc_tokens + len(tail) + 256,
                       base_tokens=doc_tokens)
        prefix = [int(t) for t in rng.integers(1000, spec.vocab - 1000,
                                               doc_tokens)]
        groups.append(dict(key=key, prefix=prefix, f=doc_tokens,
                           suffixes=[tail]))
    totals = {}
    for layer in pipeline.layers:
        layer.moe.forward = _timer(torch, totals, "experts")(layer.moe.forward)

    def run():
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                           canvas=pipeline.canvas_ids,
                           answer_row=pipeline.canvas_answer_row)
        totals.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pipeline.forward_chunk(chunk)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        for key in chunk.temporary_keys:
            arena.free_key(key)
        return chunk.tokens, seconds, totals.get("experts", 0.0)

    rows, _, _ = run()
    rows, plain_s, plain_experts = run()
    trials = [{"config": "shipped", "chunk_s": plain_s,
               "experts_s": plain_experts}]
    print(f"shipped: chunk {plain_s:.3f} s, experts {plain_experts:.3f} s",
          flush=True)
    grid = itertools.product((64, 128), (128, 256), (128, 256), (8, 16, 32),
                             (4, 8), (2, 3, 4))
    for bm, bn, bk, gm, warps, stages in grid:
        config = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                  "GROUP_SIZE_M": gm, "num_warps": warps,
                  "num_stages": stages}
        try:
            with override_config(config):
                run()
                _, chunk_s, experts_s = run()
        except Exception as error:  # noqa: BLE001 - a tile can be invalid
            trials.append({"config": config, "error": repr(error)[:160]})
            continue
        trials.append({"config": config, "chunk_s": chunk_s,
                       "experts_s": experts_s})
        print(f"{config}: chunk {chunk_s:.3f} s, experts {experts_s:.3f} s",
              flush=True)
    timed = [t for t in trials if "experts_s" in t]
    best = min(timed, key=lambda t: t["experts_s"])
    result = {
        "prediction": prediction, "rows": rows, "docs": docs,
        "shipped_experts_s": plain_experts, "shipped_chunk_s": plain_s,
        "best": best, "trials": trials,
    }
    result["volume_path"] = _save("diffusion_gemma_moe_tiles", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def time_chunk(prediction: str, docs: int = 110, doc_tokens: int = 300,
               random_tokens: bool = False, moe_backend: str = "auto",
               profile: bool = False) -> str:
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
    model = load_model(spec.hf_name, max_batched_tokens=chunk_tokens,
                       moe_backend=moe_backend)
    full_pages, sliding_pages = budgets.arena_pages(spec, device, chunk_tokens)
    arena = KVArena(n_layers=spec.layers, n_pages=full_pages,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=sliding_pages)
    pipeline = build_pipeline(spec, model, arena)
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
    print(f"{rows} rows in {plain_s:.2f} s, {rows / plain_s:.0f} rows/s",
          flush=True)
    kernels = None
    if profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torch_profile
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            run()
        events = [e for e in prof.key_averages()
                  if getattr(e, "self_device_time_total", 0) > 0]
        events.sort(key=lambda e: -e.self_device_time_total)
        total_us = sum(e.self_device_time_total for e in events)
        kernels = {
            "total_gpu_s": total_us / 1e6,
            "top": [{"name": e.key[:120], "calls": e.count,
                     "gpu_s": e.self_device_time_total / 1e6}
                    for e in events[:40]],
        }
        for row in kernels["top"]:
            print(f"{row['gpu_s']:7.3f} s {row['calls']:6d}  {row['name']}",
                  flush=True)

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
    import os

    from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs
    moe = pipeline.layers[-1].moe
    quant = getattr(moe, "quant_method", None)
    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER", "")
    configs = get_moe_configs(128, 704, "fp8_w8a8") or {}
    result = {
        "moe_config_folder": folder,
        "moe_config_folder_files": sorted(os.listdir(folder))[:5]
        if folder and os.path.isdir(folder) else None,
        "moe_config_keys": sorted(configs),
        "moe_config_32768": configs.get(32768),
        "prediction": prediction,
        "moe_backend": moe_backend,
        "moe_backend_used": str(getattr(quant, "fp8_backend", None)),
        "moe_kernel": type(getattr(quant, "moe_kernel", None)).__name__,
        "kernels": kernels,
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
    if moe_backend != "auto":
        suffix += f"_{moe_backend}"
    result["volume_path"] = _save(
        f"diffusion_gemma_layer_timing{suffix}", result)
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


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def stock_kernels(prediction: str, n_docs: int = 512,
                  doc_tokens: int = 300) -> str:
    """Record the kernels stock vLLM's compiled graph runs for this model.

    Boots stock vLLM at its defaults, profiles one prefill-heavy pass
    over filter-shaped prompts of random document tokens, and records
    the kernels plus the resolved compilation config.
    """
    import os as _os

    # the v1 engine runs the model in a child process by default,
    # where this process's profiler cannot see the kernels
    _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import random

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from quail.logical import bind_prompt, render_filter_prompt_ids
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    rng = random.Random(0)
    vocab = tokenizer.vocab_size
    bodies = [[rng.randrange(1000, vocab) for _ in range(doc_tokens)]
              for _ in range(n_docs)]
    prompt = bind_prompt("Is {0} a positive review?", ("body",), tok,
                         turn=spec.turn)
    prompts = [dict(prompt_token_ids=render_filter_prompt_ids(prompt, b, tok))
               for b in bodies]

    llm = LLM(model=spec.hf_name, gpu_memory_utilization=0.92,
              enable_prefix_caching=False, disable_log_stats=True)
    # a diffusion model rejects temperature and min_tokens
    sampling = SamplingParams(max_tokens=1)
    config = llm.llm_engine.vllm_config
    comp = config.compilation_config
    report = dict(
        prediction=prediction,
        model=spec.hf_name,
        optimization_level=int(config.optimization_level),
        compilation_mode=str(comp.mode),
        custom_ops=list(comp.custom_ops),
        enabled_custom_ops=dict(comp.enabled_custom_ops),
        disabled_custom_ops=dict(comp.disabled_custom_ops),
        pass_config={
            k: bool(getattr(comp.pass_config, k))
            for k in ("fuse_norm_quant", "fuse_act_quant",
                      "fuse_attn_quant", "enable_qk_norm_rope_fusion")
            if getattr(comp.pass_config, k, None) is not None},
        attention_backend=str(getattr(config.attention_config, "backend", "")),
        kernels=[])

    llm.generate(prompts[:32], sampling)   # warm outside the profile
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        llm.generate(prompts, sampling)
    rows = []
    total_us = 0.0
    for ev in prof.key_averages():
        # op wrapper rows repeat a kernel's name with the same device
        # time; keep the device-side rows only
        if "CUDA" not in str(getattr(ev, "device_type", "")):
            continue
        cuda_us = (getattr(ev, "self_device_time_total", 0)
                   or getattr(ev, "self_cuda_time_total", 0))
        if not cuda_us:
            continue
        total_us += cuda_us
        rows.append((round(cuda_us / 1e3, 2), ev.count, ev.key[:120]))
    rows.sort(reverse=True)
    report["cuda_busy_s"] = round(total_us / 1e6, 2)
    report["kernels"] = [dict(ms=ms, n=n, name=k) for ms, n, k in rows[:80]]
    report["volume_path"] = _save("diffusion_gemma_stock_kernels", report)
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 110, doc_tokens: int = 300,
         random_tokens: bool = False, runs: str = "chunk,loop",
         moe_backends: str = "auto", profile: bool = False):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    calls = {}
    if "chunk" in runs:
        for backend in moe_backends.split(","):
            calls[f"chunk/{backend}"] = time_chunk.spawn(
                prediction, docs, doc_tokens, random_tokens, backend, profile)
    if "gemms" in runs:
        calls["gemms"] = gemms.spawn(prediction)
    if "tiles" in runs:
        calls["tiles"] = moe_tiles.spawn(prediction, docs, doc_tokens)
    if "loop" in runs:
        calls["loop"] = time_loop.spawn(prediction)
    if "micro" in runs:
        calls["micro"] = micro.spawn(prediction)
    if "stock" in runs:
        calls["stock"] = stock_kernels.spawn(prediction)
    for name, call in calls.items():
        print(f"function call id: {call.object_id} ({name})", flush=True)
    for name, call in calls.items():
        print(call.get(), flush=True)
