"""Diagnose the kernel-compile stalls: what compiles, and the exact
size buckets, from the library's own selector (issue #25 follow-up).

Three questions, one GPU run (4B, cold container-local caches):

1. Attribution: with the library's JIT debug logging on and the
   warmup ladder disabled, run BIO-F3 and record which kernels
   compile during the tail chunks (the 9-13 s stalls). New files in
   the cache directories during the query are the ground truth.
2. Bucket map: DeepGEMM picks a kernel configuration per (M, N, K)
   with an ordinary Python function before compiling. Call it for
   every M from 1 to the chunk budget, for all four layer shapes of
   both models, and record where the chosen configuration changes.
   That is the exact warm list a principled warmup needs. Saved to
   /results/ablations/dg_buckets_<model>.json.
3. Bucket = compile key check: for a few buckets, run a bare matmul
   at the bucket's first M (expect a compile) then at a later M in
   the same bucket (expect none). If the second is fast, warming one
   M per bucket provably covers the bucket.

If the selector's import path differs in this library version, the
run reports the package's API surface instead of failing, so the
next attempt can adapt.

    uv run modal run ablations/dg_buckets.py 2>&1 | tee results/dg_buckets.log
"""

import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("quail", "baselines")
)

app = modal.App("quail-milestone1")   # house rule: no new app names
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)

SELECTOR_PATHS = (
    ("deep_gemm.jit_kernels.gemm", "get_best_configs"),
    ("deep_gemm.jit_kernels.heuristics", "get_best_configs"),
    ("deep_gemm.utils.layout", "get_best_configs"),
    ("deep_gemm", "get_best_configs"),
)


def _shapes(spec):
    """The four linear projections' (name, N, K) at one model."""
    return (("qkv", (spec.n_q + 2 * spec.n_kv) * spec.d_head,
             spec.hidden),
            ("o", spec.hidden, spec.n_q * spec.d_head),
            ("gate_up", spec.ffn_width, spec.hidden),
            ("down", spec.hidden, spec.intermediate))


def _find_selector():
    import importlib
    for mod, name in SELECTOR_PATHS:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        fn = getattr(m, name, None)
        if fn is not None:
            return f"{mod}.{name}", fn
    return None, None


def _api_dump():
    import deep_gemm
    out = dict(version=getattr(deep_gemm, "__version__", "?"),
               file=getattr(deep_gemm, "__file__", "?"),
               top=sorted(d for d in dir(deep_gemm)
                          if not d.startswith("_")))
    try:
        from deep_gemm import jit_kernels
        out["jit_kernels"] = sorted(d for d in dir(jit_kernels)
                                    if not d.startswith("_"))
    except ImportError:
        pass
    return out


def _call_selector(fn, m, n, k, num_sms):
    """Adapt to the selector's signature by parameter name."""
    import inspect
    params = inspect.signature(fn).parameters
    kwargs = {}
    for name in params:
        if name in ("m",):
            kwargs["m"] = m
        elif name in ("n",):
            kwargs["n"] = n
        elif name in ("k",):
            kwargs["k"] = k
        elif "num_groups" in name:
            kwargs[name] = 1
        elif "sms" in name:
            kwargs[name] = num_sms
    return fn(**kwargs)


def _enumerate(fn, spec, budget, num_sms):
    """First M of every configuration bucket, per layer shape."""
    out = {}
    for name, n, k in _shapes(spec):
        firsts, last_key = [], None
        for m in range(1, budget + 1):
            key = repr(_call_selector(fn, m, n, k, num_sms))
            if key != last_key:
                firsts.append(m)
                last_key = key
        out[name] = firsts
    return out


@app.function(image=image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def diagnose(run_query: bool = True) -> str:
    # cold, container-local caches + the library's JIT debug log,
    # set before anything imports the JIT machinery
    os.environ["DG_JIT_DEBUG"] = "1"
    for var in ("DG_CACHE_DIR", "DG_JIT_CACHE_DIR"):
        os.environ[var] = "/tmp/dg-cold"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton-cold"

    import quail.executor.loop as loop_mod
    loop_mod.TINY_WARM_TOKENS = ()   # attribution wants the stalls

    import torch

    from quail.planner import budgets
    from quail.planner.calibrate import _boot, resolve_pair
    from quail.specs import MODELS

    result = dict(selector=None, buckets={}, verify=[], query={})
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    result["num_sms"] = num_sms

    sel_name, sel = _find_selector()
    result["selector"] = sel_name
    if sel is None:
        result["api"] = _api_dump()
        print(f"[dg] selector not found; api={result['api']}",
              flush=True)
    else:
        print(f"[dg] selector: {sel_name}", flush=True)
        for model_name in ("qwen3-4b-fp8", "qwen3-32b-fp8"):
            spec = MODELS[model_name]
            budget = budgets.chunk_budget(
                spec, resolve_pair(model_name, "h100-sxm")[1])
            t0 = time.perf_counter()
            firsts = _enumerate(sel, spec, budget, num_sms)
            union = sorted({m for f in firsts.values() for m in f})
            result["buckets"][model_name] = dict(
                budget=budget,
                per_shape_counts={s: len(f) for s, f in firsts.items()},
                union_count=len(union),
                union_under_4096=len([m for m in union if m <= 4096]),
                enum_s=round(time.perf_counter() - t0, 1))
            path = f"/results/ablations/dg_buckets_{model_name}.json"
            os.makedirs("/results/ablations", exist_ok=True)
            with open(path, "w") as f:
                json.dump(dict(model=model_name, budget=budget,
                               num_sms=num_sms, selector=sel_name,
                               first_m_per_bucket=firsts,
                               union=union), f)
            print(f"[dg] {model_name}: buckets per shape "
                  f"{result['buckets'][model_name]['per_shape_counts']}"
                  f" union {len(union)} (<=4096: "
                  f"{result['buckets'][model_name]['union_under_4096']})"
                  f" -> {path}", flush=True)
        results_vol.commit()

    if not run_query:
        return json.dumps(result)

    spec, device = resolve_pair("qwen3-4b-fp8", "h100-sxm")
    (torch, tokenizer, pipeline, arena, async_ans,
     budget) = _boot(spec, device)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    # 3) bucket = compile key: bare matmul at a bucket's first M
    # (compile), then a mid-bucket M (should be warm)
    if sel is not None:
        layer = pipeline.layers[0]
        lin = layer.mlp.gate_up_proj
        firsts = json.load(open(
            "/results/ablations/dg_buckets_qwen3-4b-fp8.json")
        )["first_m_per_bucket"]["gate_up"]
        pairs = [(a, (a + b - 1) // 2 + 1) for a, b in
                 zip(firsts, firsts[1:]) if b - a >= 8][:5]
        with torch.inference_mode():
            # one throwaway matmul absorbs one-time library setup so
            # the first pair times only its own compile
            x = torch.randn(4096, lin.weight.shape[1], device="cuda",
                            dtype=torch.bfloat16)
            q0, s0 = pipeline.quant(x)
            pipeline.gemm(q0, s0, lin)
            torch.cuda.synchronize()
            for first_m, mid_m in pairs:
                times = []
                for m in (first_m, mid_m):
                    x = torch.randn(m, lin.weight.shape[1],
                                    device="cuda",
                                    dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    q, s = pipeline.quant(x)
                    pipeline.gemm(q, s, lin)
                    torch.cuda.synchronize()
                    times.append(round(time.perf_counter() - t0, 3))
                result["verify"].append(
                    dict(first_m=first_m, first_s=times[0],
                         mid_m=mid_m, mid_s=times[1]))
                print(f"[dg] verify bucket@{first_m}: first "
                      f"{times[0]}s, mid@{mid_m} {times[1]}s",
                      flush=True)

    # 4) attribution: standard warmup (no ladder), then BIO-F3 with
    # cache-directory snapshots around the measured run
    import re

    from quail.bench.quailb import F7, F8, F9, _biodex_rows
    from quail.executor.attention import FILTER_ATTENTION
    from quail.executor.loop import run_filter, warm_kernels
    from quail.logical import SHARED_PRE, ColumnRef, bind_prompt

    # BIO-F3's inputs, built the way the sweep cell builds them
    bio = _biodex_rows(200)
    pre = tok(SHARED_PRE)

    def stage_ids(template):
        p = bind_prompt(template, (ColumnRef("r", "r", "report"),), tok)
        return tok(re.sub(r"\{\d+\}", "", p.tail))

    bodies = [pre + tok(t) for t, _ in bio]
    qids_stages = [stage_ids(t) for t in (F7, F8, F9)]

    def listing():
        out = {}
        for d in ("/tmp/dg-cold", "/tmp/triton-cold"):
            files = set()
            for root, _, names in os.walk(d):
                files |= {os.path.join(root, x) for x in names}
            out[d] = files
        return out

    qsuf = tok("\n\nAnswer TRUE or FALSE.\nANSWER:")
    doc = (tok("The document discusses a clinical finding. ")
           * 20)[:512]
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, [doc] * 64,
                     [qsuf], budget)
    torch.cuda.synchronize()
    before = listing()
    trace = []
    with torch.inference_mode():
        pipeline.attention_mode = FILTER_ATTENTION
        _, spans, tokens = run_filter(
            torch, arena, pipeline, async_ans, bodies,
            qids_stages, budget, trace=trace, arena_writes=True)
    torch.cuda.synchronize()
    after = listing()
    chunks = [dict(tokens=tr["tokens"],
                   gpu_ms=round(e0.elapsed_time(e1), 1))
              for (_, e0, e1), tr in zip(spans, trace)]
    new = {d: sorted(os.path.basename(p) for p in after[d] - before[d])
           for d in after}
    result["query"] = dict(
        chunks=chunks,
        stalls=[c for c in chunks if c["gpu_ms"] > 500],
        new_dg_files=new["/tmp/dg-cold"],
        new_triton_files=new["/tmp/triton-cold"])
    print(f"[dg] tail chunks: {[c for c in chunks if c['tokens'] < 2000]}",
          flush=True)
    print(f"[dg] new dg files during query: {new['/tmp/dg-cold']}",
          flush=True)
    print(f"[dg] new triton files during query: "
          f"{new['/tmp/triton-cold']}", flush=True)
    return json.dumps(result)


@app.local_entrypoint()
def run(no_query: bool = False):
    call = diagnose.spawn(not no_query)
    print(f"[dg] fc={call.object_id}", flush=True)
    print(call.get(), flush=True)
