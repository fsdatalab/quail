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
def primitives(prediction: str, docs: int = 20, doc_tokens: int = 300) -> str:
    """Check each fused primitive against the reference ops on real weights."""
    import numpy as np
    import torch
    from vllm import _custom_ops as ops

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
    pipeline = build_pipeline(spec, model, arena, fused=False)
    engine = pipeline.engine
    layers = pipeline.layers
    n = docs * doc_tokens
    torch.manual_seed(0)
    x = torch.randn(n, spec.hidden, device="cuda", dtype=torch.bfloat16) * 3
    r = torch.randn(n, spec.hidden, device="cuda", dtype=torch.bfloat16) * 3
    positions = torch.arange(n, device="cuda") % doc_tokens
    result = {"prediction": prediction}

    def rel(a, b):
        a, b = a.float(), b.float()
        return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()

    with torch.inference_mode():
        norm = layers[0].input_layernorm
        # (a) norm then quant, one kernel against two
        ref = engine.norm_rows(x, norm.weight, norm.variance_epsilon)
        rq, rs = ops.scaled_fp8_quant(ref, use_per_token_if_dynamic=True)
        fq, fs = engine.norm_quant_rows(x, norm.weight, norm.variance_epsilon)
        result["norm_quant_rel"] = rel(fq.float() * fs, rq.float() * rs)
        result["norm_quant_scale_shape"] = list(fs.shape)
        # with the residual: the kernel adds x into r first
        r2 = r.clone()
        fq, fs = engine.norm_quant_rows(x, norm.weight, norm.variance_epsilon,
                                        r2)
        summed = (x.float() + r.float()).to(torch.bfloat16)
        ref = engine.norm_rows(summed, norm.weight, norm.variance_epsilon)
        rq, rs = ops.scaled_fp8_quant(ref, use_per_token_if_dynamic=True)
        result["norm_quant_residual_rel"] = rel(fq.float() * fs, rq.float() * rs)
        result["norm_quant_residual_updated_rel"] = rel(r2, summed)
        # (b) the direct fp8 GEMM against the module
        qkv_proj = layers[0].self_attn.qkv_proj
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        result["fp8_linear_rel"] = rel(engine.fp8_linear(qkv_proj, xq, xs),
                                       qkv_proj(x)[0])
        result["weight_scale_shape"] = list(qkv_proj.weight_scale.shape)
        result["weight_shape"] = list(qkv_proj.weight.shape)
        # (c) fused qk norm and rotary on a sliding and a full layer
        for name, layer in (("sliding", layers[0]), ("full", layers[5])):
            attn = layer.self_attn
            H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
            qkv = attn.qkv_proj(x)[0]
            q, k, _ = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q = engine.norm_rows(q.reshape(n, H, D), attn.q_norm.weight,
                                 attn.q_norm.variance_epsilon).view(n, H * D)
            k = engine.norm_rows(k.reshape(n, KH, D), attn.k_norm.weight,
                                 attn.k_norm.variance_epsilon).view(n, KH * D)
            rope = attn.rotary_emb
            q, k = engine.rope_inplace(positions, q, k, D, rope.cos_sin_cache,
                                       rope.is_neox_style)
            fq, fk = engine.qk_norm_rope_heads(
                qkv, positions, n_q=H, n_kv=KH, head_dim=D,
                q_weight=attn.q_norm.weight, k_weight=attn.k_norm.weight,
                eps=attn.q_norm.variance_epsilon,
                cos_sin_cache=rope.cos_sin_cache)
            result[f"qk_{name}_q_rel"] = rel(fq, q)
            result[f"qk_{name}_k_rel"] = rel(fk, k)
            result[f"qk_{name}_neox"] = bool(rope.is_neox_style)
            result[f"qk_{name}_cache"] = list(rope.cos_sin_cache.shape)
            result[f"qk_{name}_eps"] = (attn.q_norm.variance_epsilon,
                                        attn.k_norm.variance_epsilon)
        # (g) the four fused elementwise kernels against the reference ops
        mlp = layers[0].mlp
        gate_up = mlp.gate_up_proj(x)[0]
        ref = mlp.act_fn(gate_up)
        rq, rs = ops.scaled_fp8_quant(ref, use_per_token_if_dynamic=True)
        fq, fs = engine.gelu_mul_quant(gate_up)
        result["gelu_quant_rel"] = rel(fq.float() * fs, rq.float() * rs)
        result["gelu_quant_scale_rel"] = rel(fs, rs)
        result["down_from_fused_rel"] = rel(
            engine.fp8_linear(mlp.down_proj, fq, fs), mlp.down_proj(ref)[0])
        norm = layers[1].input_layernorm
        scale = float(layers[0].layer_scalar)
        r2 = r.clone()
        fq, fs = engine.scale_add_norm_quant(x, r2, scale, norm.weight,
                                             norm.variance_epsilon)
        summed = (r.float() * scale + x.float()).to(torch.bfloat16)
        ref = engine.norm_rows(summed, norm.weight, norm.variance_epsilon)
        rq, rs = ops.scaled_fp8_quant(ref, use_per_token_if_dynamic=True)
        result["scale_norm_quant_rel"] = rel(fq.float() * fs, rq.float() * rs)
        result["scale_norm_quant_residual_rel"] = rel(r2, summed)
        pre = layers[0].pre_feedforward_layernorm_2
        router = layers[0].router
        o1, o2 = engine.norm_rows2(x, pre.weight, router.quail_scale,
                                   pre.variance_epsilon)
        result["norm2_a_rel"] = rel(o1, engine.norm_rows(
            x, pre.weight, pre.variance_epsilon))
        result["norm2_b_rel"] = rel(o2, engine.norm_rows(
            x, router.quail_scale, pre.variance_epsilon))
        for name, layer in (("sliding", layers[0]), ("full", layers[5])):
            attn = layer.self_attn
            H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
            qkv = attn.qkv_proj(x)[0]
            _, _, v_ref = qkv.split([attn.q_size, attn.kv_size, attn.kv_size],
                                    -1)
            v_ref = engine.norm_rows(
                v_ref.reshape(n, KH, D),
                torch.ones(D, dtype=v_ref.dtype, device=v_ref.device),
                attn.v_norm.variance_epsilon).view(n, KH * D)
            rope = attn.rotary_emb
            fq, fk, fv = engine.qkv_norm_rope_heads(
                qkv, positions, n_q=H, n_kv=KH, head_dim=D,
                q_weight=attn.q_norm.weight, k_weight=attn.k_norm.weight,
                eps=attn.q_norm.variance_epsilon,
                cos_sin_cache=rope.cos_sin_cache)
            q2, k2 = engine.qk_norm_rope_heads(
                qkv, positions, n_q=H, n_kv=KH, head_dim=D,
                q_weight=attn.q_norm.weight, k_weight=attn.k_norm.weight,
                eps=attn.q_norm.variance_epsilon,
                cos_sin_cache=rope.cos_sin_cache)
            result[f"qkv_{name}_q_rel"] = rel(fq, q2)
            result[f"qkv_{name}_k_rel"] = rel(fk, k2)
            result[f"qkv_{name}_v_rel"] = rel(fv, v_ref)
        # (d) the residual-add norm
        norm = layers[0].post_feedforward_layernorm
        h, r2 = x.clone(), r.clone()
        engine.fused_add_rms_norm(h, r2, norm)
        summed = (x.float() + r.float()).to(torch.bfloat16)
        result["fused_add_norm_rel"] = rel(
            h, engine.norm_rows(summed, norm.weight, norm.variance_epsilon))
        result["fused_add_residual_rel"] = rel(r2, summed)
        # (e) the scalar fold alone, on the reference path
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

        def run(pipe):
            chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                               canvas=pipe.canvas_ids,
                               answer_row=pipe.canvas_answer_row)
            rows = pipe.forward_chunk(chunk)
            for key in chunk.temporary_keys:
                arena.free_key(key)
            return rows.float()

        # (f) the reference layer step by step against the fused steps,
        # on layer 0, before the fold touches the weights
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                           canvas=pipeline.canvas_ids,
                           answer_row=pipeline.canvas_answer_row)
        meta = chunk.meta
        pos = chunk.positions
        hidden0 = pipeline.embed(chunk.input_ids) * pipeline.normalizer
        layer = layers[0]
        attn = layer.self_attn
        from vllm.forward_context import set_forward_context
        with set_forward_context(None, pipeline.vllm_config,
                                 num_tokens=hidden0.shape[0]):
            meta["layer"] = 0
            h1 = pipeline._norm(hidden0, layer.input_layernorm)
            a = pipeline._attention(attn, h1, pos, meta)
            a_n = pipeline._norm(a, layer.post_attention_layernorm)
            mid = a_n + hidden0
            f = pipeline._norm(mid, layer.pre_feedforward_layernorm)
            d = layer.mlp(f)
            d_n = pipeline._norm(d, layer.post_feedforward_layernorm_1)
            r_in = pipeline._norm(mid, layer.pre_feedforward_layernorm_2)
            logits = pipeline._router_logits(layer.router, mid)
            routed = layer.moe(r_in, logits)
            routed_n = pipeline._norm(routed, layer.post_feedforward_layernorm_2)
            summed = pipeline._norm(d_n + routed_n,
                                    layer.post_feedforward_layernorm)
            out_ref = (summed + mid) * layer.layer_scalar
            meta["layer"] = 0
            out_layer = pipeline._layer(layer, hidden0, pos, meta)
            result["step_layer_vs_steps_rel"] = rel(out_layer, out_ref)
            # the fused steps
            meta["layer"] = 0
            xq, xs = engine.norm_quant_rows(
                hidden0, layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon)
            qkv = engine.fp8_linear(attn.qkv_proj, xq, xs)
            result["step_qkv_rel"] = rel(qkv, attn.qkv_proj(h1)[0])
            # the fused attention applies o_proj itself
            a2 = pipeline._attention_fused(attn, qkv, pos, meta)
            result["step_attention_rel"] = rel(a2, a)
            a2_n = pipeline._norm(a2, layer.post_attention_layernorm)
            r = hidden0.clone()
            xq, xs = engine.norm_quant_rows(
                a2_n, layer.pre_feedforward_layernorm.weight,
                layer.pre_feedforward_layernorm.variance_epsilon, r)
            result["step_mid_rel"] = rel(r, mid)
            gu = engine.fp8_linear(layer.mlp.gate_up_proj, xq, xs)
            d2, _ = layer.mlp.down_proj(layer.mlp.act_fn(gu))
            result["step_dense_rel"] = rel(d2, d)
            d2_n = pipeline._norm(d2, layer.post_feedforward_layernorm_1)
            r2_in = pipeline._norm(r, layer.pre_feedforward_layernorm_2)
            logits2, _ = layer.router.proj(engine.norm_rows(
                r, layer.router.quail_scale, layer.router.norm.variance_epsilon))
            result["step_router_rel"] = rel(logits2, logits)
            routed2 = layer.moe(r2_in, logits2)
            result["step_routed_rel"] = rel(routed2, routed)
            routed2_n = pipeline._norm(routed2, layer.post_feedforward_layernorm_2)
            engine.fused_add_rms_norm(d2_n, routed2_n,
                                      layer.post_feedforward_layernorm)
            result["step_summed_rel"] = rel(d2_n, summed)
            r.mul_(layer.layer_scalar)
            out2 = d2_n * layer.layer_scalar + r
            result["step_output_rel"] = rel(out2, out_ref)
            # cumulative over the first layers, reference first
            refs = []
            h = hidden0.clone()
            meta["layer"] = 0
            for lay in layers[:3]:
                h = pipeline._layer(lay, h, pos, meta)
                refs.append(h.clone())
            result["layer_scalars"] = [float(lay.layer_scalar)
                                       for lay in layers[:4]]
            fused = build_pipeline(spec, model, arena, fused=True)
            all_layers = fused.layers
            for k in range(1, 4):
                fused.layers = all_layers[:k]
                meta["layer"] = 0
                out_k = fused._layers_fused(hidden0.clone(), pos, meta)
                result[f"cumulative_{k}_rel"] = rel(out_k, refs[k - 1])
            fused.layers = all_layers
        for key in chunk.temporary_keys:
            arena.free_key(key)
    result["volume_path"] = _save("diffusion_gemma_primitives_check", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def check(prediction: str, docs: int = 110, doc_tokens: int = 300,
          wide_head_kernel: str = "triton") -> str:
    """Compare the fused forward path against the reference path.

    Builds the reference pipeline first, since the fused one folds the
    layer scalars into the model's norm weights.
    """
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

    def run(pipeline):
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                           canvas=pipeline.canvas_ids,
                           answer_row=pipeline.canvas_answer_row)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            rows = pipeline.forward_chunk(chunk)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        for key in chunk.temporary_keys:
            arena.free_key(key)
        return rows.float(), seconds

    answer = model.quail_answer_weights.float()
    reference = build_pipeline(spec, model, arena, fused=False,
                               wide_head_kernel="triton")
    ref_rows, _ = run(reference)
    ref_rows, ref_s = run(reference)
    fused = build_pipeline(spec, model, arena, fused=True,
                           wide_head_kernel=wide_head_kernel)
    fused_rows, _ = run(fused)
    fused_rows, fused_s = run(fused)
    ref_logits = ref_rows @ answer.T
    fused_logits = fused_rows @ answer.T
    diff = (fused_rows - ref_rows).abs()
    result = {
        "prediction": prediction,
        "wide_head_kernel": wide_head_kernel,
        "docs": docs, "doc_tokens": doc_tokens,
        "reference_s": ref_s, "fused_s": fused_s,
        "speedup": ref_s / fused_s,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "reference_rms": ref_rows.pow(2).mean().sqrt().item(),
        "relative_rms_diff": (diff.pow(2).mean().sqrt()
                              / ref_rows.pow(2).mean().sqrt()).item(),
        "answer_agreement": (ref_logits.argmax(1)
                             == fused_logits.argmax(1)).float().mean().item(),
        "logit_gap_max_abs_diff": (
            (ref_logits[:, 0] - ref_logits[:, 1])
            - (fused_logits[:, 0] - fused_logits[:, 1])).abs().max().item(),
    }
    result["volume_path"] = _save(
        f"diffusion_gemma_fused_check_{wide_head_kernel}", result)
    return json.dumps(result, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def time_chunk(prediction: str, docs: int = 110, doc_tokens: int = 300,
               wide_head_kernel: str = "triton",
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
        "wide_head_kernel": wide_head_kernel,
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


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def fa4_tiles(prediction: str, prefix: int = 448, suffix: int = 54,
              partners: int = 12, anchors: int = 54) -> str:
    """Time the join attention on the full-layer shape, option by option.

    Synthetic IMDB-2 layout: each anchor's prefix is shared by its
    partners, and each pair's suffix rows see the prefix and the
    pair's earlier rows. 16 query heads, 2 KV heads, 512-wide heads,
    k = v as in the model's full-attention layers. Also times the
    sliding-layer shape (16 query heads, 8 KV heads, 256-wide heads,
    window 1024) on FlashAttention 3 and 4.
    """
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    page = 16
    if prefix % page:
        raise ValueError("prefix must be a whole number of pages")
    pairs = anchors * partners
    rows = pairs * suffix
    own_pages = -(-suffix // page)
    prefix_pages = prefix // page
    n_pages = anchors * prefix_pages + pairs * own_pages
    dev = "cuda"

    def layout(H, KH, D):
        q = torch.randn(rows, H, D, device=dev, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(n_pages, page, KH, D, device=dev,
                         dtype=torch.bfloat16) * 0.1
        table = torch.empty(pairs, prefix_pages + own_pages,
                            device=dev, dtype=torch.int32)
        for a in range(anchors):
            for j in range(partners):
                pair = a * partners + j
                table[pair, :prefix_pages] = torch.arange(
                    a * prefix_pages, (a + 1) * prefix_pages)
                first = anchors * prefix_pages + pair * own_pages
                table[pair, prefix_pages:] = torch.arange(
                    first, first + own_pages)
        # the same rows unpaged: per pair [prefix | own rows]
        flat = kv.view(n_pages * page, KH, D)
        row_ids = []
        for pair in range(pairs):
            pages = table[pair].tolist()
            ids = [p * page + t for p in pages for t in range(page)]
            row_ids.extend(ids[:prefix + suffix])
        row_ids = torch.tensor(row_ids, device=dev)
        unpaged = flat.index_select(0, row_ids)
        own_ids = row_ids.view(pairs, -1)[:, prefix:].reshape(-1)
        own = flat.index_select(0, own_ids)
        cu_q = torch.arange(0, rows + 1, suffix, device=dev,
                            dtype=torch.int32)
        cu_k = torch.arange(0, pairs * (prefix + suffix) + 1, prefix + suffix,
                            device=dev, dtype=torch.int32)
        used = torch.full((pairs,), prefix + suffix, device=dev,
                          dtype=torch.int32)
        cu_anchor = torch.arange(0, rows + 1, partners * suffix, device=dev,
                                 dtype=torch.int32)
        anchor_table = table[::partners, :prefix_pages].contiguous()
        anchor_used = torch.full((anchors,), prefix, device=dev,
                                 dtype=torch.int32)
        return dict(q=q, kv=kv, table=table, unpaged=unpaged, own=own,
                    cu_q=cu_q, cu_k=cu_k, used=used, cu_anchor=cu_anchor,
                    anchor_table=anchor_table, anchor_used=anchor_used)

    def flops(H, D, own=True, shared=True):
        per_pair = ((suffix * prefix if shared else 0)
                    + (suffix * (suffix + 1) // 2 if own else 0))
        return pairs * per_pair * H * D * 4

    def timed(fn, repeats=10):
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - report, keep going
            return f"failed: {type(error).__name__}: {str(error)[:160]}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    def paged_fa(L, version, window=None, **knobs):
        def run():
            if not knobs:
                return flash_attn_varlen_func(
                    L["q"], L["kv"], L["kv"], max_seqlen_q=suffix,
                    cu_seqlens_q=L["cu_q"], max_seqlen_k=prefix + suffix,
                    block_table=L["table"], seqused_k=L["used"],
                    causal=True, fa_version=version, softmax_scale=1.0,
                    return_softmax_lse=True,
                    window_size=list(window) if window else (-1, -1))
            return _flash_attn_fwd(
                L["q"], L["kv"], L["kv"], cu_seqlens_q=L["cu_q"],
                seqused_k=L["used"], max_seqlen_q=suffix,
                max_seqlen_k=prefix + suffix, page_table=L["table"],
                softmax_scale=1.0, causal=True, num_splits=0,
                return_lse=True, **knobs)
        return run

    def unpaged_fa(L, version):
        def run():
            return flash_attn_varlen_func(
                L["q"], L["unpaged"], L["unpaged"], max_seqlen_q=suffix,
                cu_seqlens_q=L["cu_q"], max_seqlen_k=prefix + suffix,
                cu_seqlens_k=L["cu_k"], causal=True, fa_version=version,
                softmax_scale=1.0, return_softmax_lse=True)
        return run

    def call_a(L, version):
        def run():
            return flash_attn_varlen_func(
                L["q"], L["own"], L["own"], max_seqlen_q=suffix,
                cu_seqlens_q=L["cu_q"], max_seqlen_k=suffix,
                cu_seqlens_k=L["cu_q"], causal=True, fa_version=version,
                softmax_scale=1.0, return_softmax_lse=True)
        return run

    def call_b(L, version):
        def run():
            return flash_attn_varlen_func(
                L["q"], L["kv"], L["kv"], max_seqlen_q=partners * suffix,
                cu_seqlens_q=L["cu_anchor"], max_seqlen_k=prefix,
                block_table=L["anchor_table"], seqused_k=L["anchor_used"],
                causal=False, fa_version=version, softmax_scale=1.0,
                return_softmax_lse=True)
        return run

    def triton_paged(L, window=None):
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )
        out = torch.empty_like(L["q"])

        def run():
            unified_attention(
                q=L["q"], k=L["kv"], v=L["kv"], out=out,
                cu_seqlens_q=L["cu_q"], max_seqlen_q=suffix,
                seqused_k=L["used"], max_seqlen_k=prefix + suffix,
                softmax_scale=1.0, causal=True,
                window_size=window or (-1, -1), block_table=L["table"],
                softcap=0.0, q_descale=None, k_descale=None, v_descale=None)
        return run

    result = {"prediction": prediction, "prefix": prefix, "suffix": suffix,
              "partners": partners, "anchors": anchors, "rows": rows}
    full = layout(16, 2, 512)
    variants = {
        "fa4_paged_default": paged_fa(full, 4),
        "fa4_unpaged_default": unpaged_fa(full, 4),
        "fa4_paged_pack_gqa_off": paged_fa(full, 4, pack_gqa=False),
        "fa4_paged_tile_128x64": paged_fa(full, 4, tile_mn=(128, 64)),
        "fa4_paged_tile_64x128": paged_fa(full, 4, tile_mn=(64, 128)),
        "fa4_paged_tile_128x128": paged_fa(full, 4, tile_mn=(128, 128)),
        "fa4_paged_tile_128x64_pack_off": paged_fa(
            full, 4, tile_mn=(128, 64), pack_gqa=False),
        "fa4_paged_pv_rs": paged_fa(full, 4, mma_pv_is_rs=True),
        "fa4_paged_no_overlap": paged_fa(full, 4, intra_wg_overlap=False),
        "fa4_call_a_own_rows": call_a(full, 4),
        "fa4_call_b_prefix": call_b(full, 4),
        "triton_paged": triton_paged(full),
    }
    full_flops = flops(16, 512)
    result["full_flops"] = full_flops
    # call A covers each pair's own rows, call B the shared prefix
    part = {"fa4_call_a_own_rows": flops(16, 512, shared=False),
            "fa4_call_b_prefix": flops(16, 512, own=False)}
    for name, fn in variants.items():
        seconds = timed(fn)
        if isinstance(seconds, str):
            result[name] = seconds
        else:
            work = part.get(name, full_flops)
            result[name] = {"ms": round(seconds * 1e3, 3),
                            "tflops": round(work / seconds / 1e12, 1)}
        print(name, result[name], flush=True)
    a, b = result["fa4_call_a_own_rows"], result["fa4_call_b_prefix"]
    if isinstance(a, dict) and isinstance(b, dict):
        result["fa4_two_call_sum_ms"] = round(a["ms"] + b["ms"], 3)
    del full
    torch.cuda.empty_cache()

    sliding = layout(16, 8, 256)
    window = (1023, 0)
    sliding_flops = flops(16, 256)
    result["sliding_flops"] = sliding_flops
    for name, fn in {
        "sliding_fa3_paged_window": paged_fa(sliding, 3, window),
        "sliding_fa4_paged_window": paged_fa(sliding, 4, window),
        "sliding_fa3_paged_full": paged_fa(sliding, 3),
        "sliding_triton_paged_window": triton_paged(sliding, window),
    }.items():
        seconds = timed(fn)
        if isinstance(seconds, str):
            result[name] = seconds
        else:
            result[name] = {"ms": round(seconds * 1e3, 3),
                            "tflops": round(sliding_flops / seconds / 1e12, 1)}
        print(name, result[name], flush=True)
    result["h100_bf16_dense_peak_tflops"] = 989
    result["gb_kv_full_per_token"] = 2 * 512 * 2 / 1e9
    result["softmax_scale"] = 1.0
    result["volume_path"] = _save("diffusion_gemma_fa4_tiles", result)
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 110, doc_tokens: int = 300,
         kernels: str = "triton,fa4", random_tokens: bool = False,
         runs: str = "chunk,loop", moe_backends: str = "auto",
         profile: bool = False):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    calls = {}
    if "chunk" in runs:
        for kernel in kernels.split(","):
            for backend in moe_backends.split(","):
                calls[f"{kernel}/{backend}"] = time_chunk.spawn(
                    prediction, docs, doc_tokens, kernel, random_tokens,
                    backend, profile)
    if "gemms" in runs:
        calls["gemms"] = gemms.spawn(prediction)
    if "tiles" in runs:
        calls["tiles"] = moe_tiles.spawn(prediction, docs, doc_tokens)
    if "primitives" in runs:
        calls["primitives"] = primitives.spawn(prediction, docs, doc_tokens)
    if "check" in runs:
        for kernel in kernels.split(","):
            calls[f"check/{kernel}"] = check.spawn(prediction, docs,
                                                  doc_tokens, kernel)
    if "loop" in runs:
        calls["loop"] = time_loop.spawn(prediction)
    if "micro" in runs:
        calls["micro"] = micro.spawn(prediction)
    if "fa4" in runs:
        calls["fa4"] = fa4_tiles.spawn(prediction)
    for name, call in calls.items():
        print(f"function call id: {call.object_id} ({name})", flush=True)
    for name, call in calls.items():
        print(call.get(), flush=True)
