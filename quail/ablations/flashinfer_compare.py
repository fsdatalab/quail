"""FlashInfer versus FlashAttention-3 on Quail's attention shapes
(issue #24, "evaluate off-the-shelf alternatives").

Two cells:

- probe_flashinfer: report the FlashInfer version and API surface in
  the exact vLLM 0.26.0 image, before any benchmark depends on a
  wrapper name or signature. Ran 2026-08-21: flashinfer 0.6.14 is in
  the image (results/flashinfer_probe.json).
- bench: time the attention paths on synthetic tensors shaped like
  the real workloads, FA3 against FlashInfer, isolated from the rest
  of the forward pass. Each timing includes everything downstream
  that differs between paths (KV scatter, merge, FP8 quantization),
  so the numbers compare what o_proj actually waits for.

The decision rule from the issue: if an off-the-shelf kernel is
within 5% of our path, prefer the off-the-shelf one for
maintainability.

Shapes (Qwen3 4B: 32 q heads, 8 kv heads, head_dim 128, page 16):

- filter_fresh: 340 groups of (fresh document 180-460 tokens +
  question 32 tokens), ~110k fresh tokens - the stage-1 filter chunk,
  where ~90% of all filter tokens are spent.
- filter_cached: the same groups with the documents already resident,
  question suffixes only - the stage-2+ rewind chunk.
- join: 10 anchors of 3,530 resident tokens, 26 suffixes of 42 tokens
  each - the BioDEX 10x256 chunk geometry.
- join_fanout: 1 anchor, 256 suffixes - the high-fan-out join. This
  shape also fits FlashInfer's cascade wrapper (their two-level
  shared-prefix decomposition), which needs one prefix shared by
  every query in the batch, so it cannot run the multi-anchor shapes.

Prediction, stated before the run: FlashInfer's prefill kernels and
FA3 are the same class of Hopper tensor-core kernel, so expect the
two-call stacks within tens of percent of each other; our unified
path is a single FA3 call with no merge, so on the filter shapes
FlashInfer has to beat one FA3 kernel launch outright to displace
it. Cascade should behave like our two-call path (it is the same
decomposition) with a different merge implementation.

A third cell, bench_tuned, is the fairness pass: the same shapes
with every backend string the wrappers accept forced in turn
(instead of "auto"), plus a hybrid variant whose two FlashInfer
attention calls are merged by our fused merge_quant kernel, so
kernel quality is measured separately from fusion.

Run from the quail/ directory (tee per house rule):

    uv run modal run ablations/flashinfer_compare.py::run_probe \
        2>&1 | tee results/flashinfer_probe.log
    uv run modal run ablations/flashinfer_compare.py::run_bench \
        2>&1 | tee results/flashinfer_bench.log
    uv run modal run ablations/flashinfer_compare.py::run_bench_tuned \
        2>&1 | tee results/flashinfer_tuned.log
"""

import json
import os

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "baselines")
)

# House rule: never create new Modal app names.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})


def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


@app.function(timeout=900, **GPU_KW)
def probe_flashinfer() -> str:
    """FlashInfer's presence, version, and the entry points the
    benchmark uses, from the exact image."""
    import importlib
    import inspect

    report = {}
    try:
        fi = importlib.import_module("flashinfer")
        report["version"] = getattr(fi, "__version__", "unknown")
        report["file"] = getattr(fi, "__file__", None)
    except Exception as exc:
        report["import_error"] = repr(exc)
        return _write(report, "flashinfer_probe")

    names = ("BatchPrefillWithPagedKVCacheWrapper",
             "BatchPrefillWithRaggedKVCacheWrapper",
             "MultiLevelCascadeAttentionWrapper",
             "BatchAttention",
             "merge_state", "merge_states", "merge_state_in_place",
             "single_prefill_with_kv_cache")
    entries = {}
    for name in names:
        value = getattr(fi, name, None)
        if value is None:
            entries[name] = None
            continue
        try:
            sig = str(inspect.signature(value))
        except (TypeError, ValueError):
            sig = "class"
        entries[name] = sig
    report["entries"] = entries
    for cls_name in ("BatchPrefillWithPagedKVCacheWrapper",
                     "MultiLevelCascadeAttentionWrapper"):
        cls = getattr(fi, cls_name, None)
        if cls is None:
            continue
        methods = {}
        for m in ("__init__", "plan", "run"):
            fn = getattr(cls, m, None)
            if fn is None:
                methods[m] = None
                continue
            try:
                methods[m] = str(inspect.signature(fn))
            except (TypeError, ValueError):
                methods[m] = "?"
        report[cls_name] = methods
    return _write(report, "flashinfer_probe")


@app.local_entrypoint()
def run_probe():
    print(probe_flashinfer.remote())


# ------------------------------------------------------- the benchmark

def _median_ms(torch, fn, iters=20):
    """Median wall of fn() over iters, CUDA-event timed, after 3
    unmeasured calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    times.sort()
    return times[len(times) // 2]


def _case_builder(q_heads):
    """The synthetic chunk builder at the engine's geometry. Shared
    by bench and bench_tuned so both cells time identical tensors
    (same seeds, same arena layout)."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline

    torch.manual_seed(7)
    H, KH, D = q_heads, 8, 128
    PAGE = 16

    weight = SimpleNamespace(shape=(4096, 4096))
    attn = SimpleNamespace(
        num_heads=H, num_kv_heads=KH, head_dim=D,
        rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(
        self_attn=attn,
        mlp=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=weight)))
    model = SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=None, norm=None))
    rng = np.random.default_rng(7)

    def build_case(name, kept_lens, suffix_counts, suffix_len, fresh):
        """One packed chunk. fresh=True packs each prefix's tokens in
        the chunk (the stage-1 shape: KV scattered during the pass);
        fresh=False pre-fills the arena and streams suffixes only."""
        pages_needed = sum(-(-(k + suffix_len) // PAGE)
                           for k in kept_lens)
        arena = KVArena(n_layers=1, n_pages=pages_needed + 8,
                        page_tokens=PAGE, n_kv=KH, d_head=D,
                        dtype=torch.bfloat16)
        pipeline = Pipeline(model, arena, kernels="quail",
                            attention_mode="merge_quant")
        groups = []
        n_tokens = 0
        for i, kept in enumerate(kept_lens):
            key = (name, i)
            assert arena.alloc(
                key, kept, capacity_tokens=kept + suffix_len) is not None
            if fresh:
                groups.append(dict(
                    key=key, prefix=[1] * kept, f=kept,
                    suffixes=[[2] * suffix_len
                              for _ in range(suffix_counts[i])]))
                n_tokens += kept
            else:
                rows = arena.rows_gpu(key)
                arena.k[0].index_copy_(
                    0, rows, torch.randn(kept, KH, D, device="cuda",
                                         dtype=torch.bfloat16))
                arena.v[0].index_copy_(
                    0, rows, torch.randn(kept, KH, D, device="cuda",
                                         dtype=torch.bfloat16))
                groups.append(dict(
                    key=key, prefix=None, f=kept,
                    suffixes=[[2] * suffix_len
                              for _ in range(suffix_counts[i])]))
            n_tokens += suffix_counts[i] * suffix_len
        q = torch.randn(n_tokens, H, D, device="cuda",
                        dtype=torch.bfloat16)
        k3 = torch.randn(n_tokens, KH, D, device="cuda",
                         dtype=torch.bfloat16).contiguous()
        v3 = torch.randn_like(k3).contiguous()
        return dict(name=name, arena=arena, pipeline=pipeline,
                    groups=groups, q=q, k=k3, v=v3, fresh=fresh,
                    n_tokens=n_tokens, kept_lens=kept_lens,
                    suffix_counts=suffix_counts, suffix_len=suffix_len)

    return SimpleNamespace(torch=torch, np=np, rng=rng, H=H, KH=KH,
                           D=D, PAGE=PAGE, build_case=build_case)


def _bench_cases(S):
    """The four workload shapes, identical in bench and bench_tuned."""
    rng = S.rng
    return [
        S.build_case("filter_fresh",
                     [int(x) for x in rng.integers(180, 460, size=340)],
                     [1] * 340, 32, fresh=True),
        S.build_case("filter_cached",
                     [int(x) for x in rng.integers(180, 460, size=340)],
                     [1] * 340, 32, fresh=False),
        S.build_case("join", [3530] * 10, [26] * 10, 42, fresh=False),
        S.build_case("join_fanout", [3530], [256], 42, fresh=False),
    ]


@app.function(timeout=2400, **GPU_KW)
def bench(q_heads: int = 32) -> str:
    """q_heads=32 is the 4B geometry (4:1 GQA); 64 is the 32B
    geometry (8:1 GQA)."""
    import numpy as np
    import torch

    from quail.executor.loop import pack_chunk

    S = _case_builder(q_heads)
    H, KH, D, PAGE = S.H, S.KH, S.D, S.PAGE

    # ---- the FA3 sides: the engine's own paths over pack_chunk ----

    def fa3_variants(case):
        out = {}
        n = case["n_tokens"]
        q_flat = case["q"].view(n, -1)
        k_flat = case["k"].view(n, -1)
        v_flat = case["v"].view(n, -1)
        pipeline = case["pipeline"]
        arena = case["arena"]

        mq_chunk = pack_chunk(torch, arena, case["groups"],
                              pinned=True, attention_mode="merge_quant")

        def run_mq():
            mq_chunk["meta"]["layer"] = 0
            return pipeline.attention_merge_quant(
                q_flat, k_flat, v_flat, mq_chunk["meta"])
        out["fa3_merge_quant"] = _median_ms(torch, run_mq)

        if all(c == 1 for c in case["suffix_counts"]):
            uni_chunk = pack_chunk(torch, arena, case["groups"],
                                   pinned=True,
                                   attention_mode="unified")

            def run_uni():
                uni_chunk["meta"]["layer"] = 0
                o = pipeline.attention_unified(
                    q_flat, k_flat, v_flat, uni_chunk["meta"])
                return pipeline.quant(o)
            out["fa3_unified_plus_quant"] = _median_ms(torch, run_uni)
        return out

    # ---- the FlashInfer sides -------------------------------------

    def fi_variants(case, fi):
        out = {}
        errors = {}
        arena = case["arena"]
        pipeline = case["pipeline"]
        name = case["name"]
        n = case["n_tokens"]
        kept = case["kept_lens"]
        counts = case["suffix_counts"]
        slen = case["suffix_len"]
        q3, k3, v3 = case["q"], case["k"], case["v"]
        n_pages = arena.accounting.n_pages
        kcache = arena.k[0].view(n_pages, PAGE, KH, D)
        vcache = arena.v[0].view(n_pages, PAGE, KH, D)

        def pages_of(i):
            return arena.accounting.owned[(name, i)]

        def i32(x):
            # plan() inputs stay on the host; the wrapper stages them
            return torch.tensor(x, dtype=torch.int32)

        def ws():
            return torch.empty(160 * 1024 * 1024, dtype=torch.uint8,
                               device="cuda")

        # -- one causal paged call over kept + current (unified) --
        if all(c == 1 for c in counts):
            try:
                uni_chunk = pack_chunk(torch, arena, case["groups"],
                                       pinned=True,
                                       attention_mode="unified")
                uni = uni_chunk["meta"]["unified"]
                kv_lens = [k + slen for k in kept]
                page_counts = [-(-x // PAGE) for x in kv_lens]
                wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
                    ws(), "NHD")
                # q rows per entry: the fresh shape packs prefix +
                # suffix as this entry's queries; the cached shape
                # packs the suffix only
                q_lens = ([k + slen for k in kept] if case["fresh"]
                          else [slen] * len(kept))
                wrapper.plan(
                    i32([0] + list(np.cumsum(q_lens))),
                    i32([0] + list(np.cumsum(page_counts))),
                    i32([p for i in range(len(kept))
                         for p in pages_of(i)[:page_counts[i]]]),
                    i32([(x - 1) % PAGE + 1 for x in kv_lens]),
                    H, KH, D, PAGE, causal=True,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16)

                def run_fi_unified():
                    pipeline.kv_row_scatter(k3, v3, uni["src"],
                                            uni["dst"], 0)
                    o = wrapper.run(q3, (kcache, vcache))
                    return pipeline.quant(o.view(n, -1))
                out["fi_paged_causal_plus_quant"] = _median_ms(
                    torch, run_fi_unified)
                # correctness spot check against the FA3 unified path
                uni_chunk["meta"]["layer"] = 0
                ref = pipeline.attention_unified(
                    q3.view(n, -1), k3.view(n, -1), v3.view(n, -1),
                    uni_chunk["meta"])
                pipeline.kv_row_scatter(k3, v3, uni["src"],
                                        uni["dst"], 0)
                got = wrapper.run(q3, (kcache, vcache)).view(n, -1)
                out["fi_unified_max_abs_vs_fa3"] = float(
                    (got.float() - ref.float()).abs().max().item())
            except Exception as exc:
                errors["fi_paged_causal"] = repr(exc)

        # -- two calls plus FlashInfer's merge --
        try:
            chunk = pack_chunk(torch, arena, case["groups"],
                               pinned=True,
                               attention_mode="merge_quant")
            meta = chunk["meta"]
            cross = meta["cross"]
            rows = cross["rows"]
            ragged = fi.BatchPrefillWithRaggedKVCacheWrapper(
                ws(), "NHD")
            cu_a = meta["cu_a"].cpu()
            ragged.plan(cu_a, cu_a, H, KH, D, causal=True,
                        q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16)
            page_counts = [-(-k // PAGE) for k in kept]
            paged = fi.BatchPrefillWithPagedKVCacheWrapper(ws(), "NHD")
            paged.plan(
                cross["cu_q"].cpu(),
                i32([0] + list(np.cumsum(page_counts))),
                i32([p for i in range(len(kept))
                     for p in pages_of(i)[:page_counts[i]]]),
                i32([(k - 1) % PAGE + 1 for k in kept]),
                H, KH, D, PAGE, causal=False,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16)

            def run_fi_two_call():
                if meta["kv_src"] is not None:
                    pipeline.kv_row_scatter(k3, v3, meta["kv_src"],
                                            meta["kv_dst"], 0)
                oa, lse_a = ragged.run(q3, k3, v3, return_lse=True)
                q_suf = q3.index_select(0, rows)
                ob, lse_b = paged.run(q_suf, (kcache, vcache),
                                      return_lse=True)
                merged, _ = fi.merge_state(
                    oa.index_select(0, rows),
                    lse_a.index_select(0, rows), ob, lse_b)
                o = oa.index_copy_(0, rows, merged)
                return pipeline.quant(o.view(n, -1))
            out["fi_two_call_merge_plus_quant"] = _median_ms(
                torch, run_fi_two_call)
        except Exception as exc:
            errors["fi_two_call"] = repr(exc)

        # -- cascade: FlashInfer's shared-prefix decomposition --
        # (one prefix shared by every query in the batch, so only the
        # single-anchor fan-out shape fits)
        if len(kept) == 1 and not case["fresh"]:
            try:
                total_q = counts[0] * slen
                anchor_pages = -(-kept[0] // PAGE)
                suffix_pages = -(-slen // PAGE)
                # a combined cache: the anchor's pages first, then
                # page-aligned scratch pages for each suffix's own KV
                cas_pages = anchor_pages + counts[0] * suffix_pages
                cache = torch.zeros(cas_pages, 2, PAGE, KH, D,
                                    dtype=torch.bfloat16,
                                    device="cuda")
                flat_k = cache[:, 0].reshape(cas_pages * PAGE, KH, D)
                flat_v = cache[:, 1].reshape(cas_pages * PAGE, KH, D)
                src_rows = arena.rows_gpu((name, 0))
                flat_k[:kept[0]] = arena.k[0].index_select(0, src_rows)
                flat_v[:kept[0]] = arena.v[0].index_select(0, src_rows)
                dst = torch.cat([
                    torch.arange((anchor_pages + i * suffix_pages)
                                 * PAGE,
                                 (anchor_pages + i * suffix_pages)
                                 * PAGE + slen, device="cuda")
                    for i in range(counts[0])])
                src = torch.arange(total_q, device="cuda")
                wrapper = fi.MultiLevelCascadeAttentionWrapper(
                    2, ws(), "NHD")
                wrapper.plan(
                    [i32([0, total_q]),
                     i32([0] + list(np.cumsum([slen] * counts[0])))],
                    [i32([0, anchor_pages]),
                     i32([0] + list(np.cumsum([suffix_pages]
                                              * counts[0])))],
                    [i32(list(range(anchor_pages))),
                     i32(list(range(anchor_pages, cas_pages)))],
                    [i32([(kept[0] - 1) % PAGE + 1]),
                     i32([(slen - 1) % PAGE + 1] * counts[0])],
                    H, KH, D, PAGE, causal=True,
                    q_data_type="bfloat16", kv_data_type="bfloat16")
                row = KH * D

                def run_cascade():
                    pipeline._triton_kernels()["kv_scatter"][
                        (total_q,)](
                        k3, v3, flat_k, flat_v, src, dst,
                        ROW=row, ROW_POW2=1 << (row - 1).bit_length())
                    o = wrapper.run(q3, cache)
                    return pipeline.quant(o.view(n, -1))
                out["fi_cascade_plus_quant"] = _median_ms(
                    torch, run_cascade)
            except Exception as exc:
                errors["fi_cascade"] = repr(exc)
        if errors:
            out["errors"] = errors
        return out

    cases = _bench_cases(S)

    try:
        import flashinfer as fi
        fi_version = fi.__version__
    except Exception as exc:
        fi = None
        fi_version = repr(exc)

    report = dict(
        cell="flashinfer_bench", flashinfer=fi_version,
        q_heads=H, kv_heads=KH,
        prediction=(
            "Two-call stacks within tens of percent of each other; "
            "FA3 unified stays ahead on the filter shapes unless "
            "FlashInfer's single paged causal kernel is outright "
            "faster; cascade behaves like the two-call path. Adopt "
            "FlashInfer only if within 5% or better."),
        note=("Times are per-layer attention milliseconds including "
              "KV scatter, merge, and FP8 quantization - what o_proj "
              "waits for. us_per_fresh_token divides by the chunk's "
              "fresh token count."),
        shapes={})
    with torch.inference_mode():
        for case in cases:
            row = dict(
                n_tokens=case["n_tokens"],
                kept_tokens=int(sum(case["kept_lens"])),
                fresh=case["fresh"],
                fa3=fa3_variants(case),
                flashinfer=(fi_variants(case, fi) if fi else {}))
            us = {}
            for side in ("fa3", "flashinfer"):
                for k, v in row[side].items():
                    if isinstance(v, float) and "max_abs" not in k:
                        us[k] = round(v * 1e3 / case["n_tokens"], 3)
            row["us_per_fresh_token"] = us
            report["shapes"][case["name"]] = row
            print(f"[flashinfer_bench] {case['name']}: "
                  f"{json.dumps(us)}", flush=True)
            if "fi_unified_max_abs_vs_fa3" in row["flashinfer"]:
                print(f"[flashinfer_bench] {case['name']} "
                      f"fi-vs-fa3 max_abs "
                      f"{row['flashinfer']['fi_unified_max_abs_vs_fa3']}",
                      flush=True)
    tag = "" if H == 32 else f"_{H}h"
    return _write(report, f"flashinfer_bench{tag}")


@app.local_entrypoint()
def run_bench(q_heads: int = 32):
    print(bench.remote(q_heads))


# ------------------------------------------------- the fairness pass

@app.function(timeout=3600, **GPU_KW)
def bench_tuned(q_heads: int = 32) -> str:
    """Give FlashInfer its best configuration (issue #24 follow-up):
    force each backend string the 0.6.14 wrappers accept instead of
    "auto", and add a hybrid two-call variant whose two FlashInfer
    attention calls are merged by our fused merge_quant Triton
    kernel - separating kernel quality from fusion. Page size stays
    16 on both stacks: it is an arena property the engine sets, and
    both sides run the same value."""
    import numpy as np
    import torch

    from quail.executor.loop import pack_chunk

    S = _case_builder(q_heads)
    H, KH, D, PAGE = S.H, S.KH, S.D, S.PAGE

    import flashinfer as fi

    BACKENDS = ("auto", "fa2", "fa3", "cutlass", "trtllm-gen")
    cases = _bench_cases(S)

    report = dict(
        cell="flashinfer_tuned", flashinfer=fi.__version__,
        q_heads=H, kv_heads=KH, backends_tried=list(BACKENDS),
        prediction=(
            "Forcing a backend moves the filter-shape gap little "
            "(auto already picks per shape); the hybrid variant "
            "removes roughly the fusion share of the join gap but "
            "the ragged+paged pair still trails merge_quant; "
            "nothing reaches the 5% adoption bar."),
        note=("Same timing method as flashinfer_bench: plan() and "
              "JIT compilation outside the timed region, median of "
              "20 CUDA-event runs after 3 warmups, times include KV "
              "scatter, merge, and FP8 quantization."),
        shapes={})

    with torch.inference_mode():
        for case in cases:
            name = case["name"]
            arena = case["arena"]
            pipeline = case["pipeline"]
            n = case["n_tokens"]
            kept = case["kept_lens"]
            counts = case["suffix_counts"]
            slen = case["suffix_len"]
            q3, k3, v3 = case["q"], case["k"], case["v"]
            q_flat, k_flat, v_flat = (q3.view(n, -1), k3.view(n, -1),
                                      v3.view(n, -1))
            n_pages = arena.accounting.n_pages
            kcache = arena.k[0].view(n_pages, PAGE, KH, D)
            vcache = arena.v[0].view(n_pages, PAGE, KH, D)

            def pages_of(i):
                return arena.accounting.owned[(name, i)]

            def i32(x):
                return torch.tensor(x, dtype=torch.int32)

            def ws():
                return torch.empty(160 * 1024 * 1024,
                                   dtype=torch.uint8, device="cuda")

            row = dict(n_tokens=n, fa3={}, paged_causal={},
                       paged_causal_max_abs={},
                       two_call_merge_state={},
                       two_call_fused_merge={}, rejected={})

            # same-run FA3 references
            mq_chunk = pack_chunk(torch, arena, case["groups"],
                                  pinned=True,
                                  attention_mode="merge_quant")

            def run_mq():
                mq_chunk["meta"]["layer"] = 0
                return pipeline.attention_merge_quant(
                    q_flat, k_flat, v_flat, mq_chunk["meta"])
            row["fa3"]["merge_quant"] = _median_ms(torch, run_mq)

            unified_ok = all(c == 1 for c in counts)
            if unified_ok:
                uni_chunk = pack_chunk(torch, arena, case["groups"],
                                       pinned=True,
                                       attention_mode="unified")
                uni = uni_chunk["meta"]["unified"]

                def run_uni():
                    uni_chunk["meta"]["layer"] = 0
                    o = pipeline.attention_unified(
                        q_flat, k_flat, v_flat, uni_chunk["meta"])
                    return pipeline.quant(o)
                row["fa3"]["unified_plus_quant"] = _median_ms(
                    torch, run_uni)
                uni_chunk["meta"]["layer"] = 0
                ref = pipeline.attention_unified(
                    q_flat, k_flat, v_flat, uni_chunk["meta"])

                kv_lens = [k + slen for k in kept]
                page_counts = [-(-x // PAGE) for x in kv_lens]
                q_lens = (kv_lens if case["fresh"]
                          else [slen] * len(kept))
                uni_plan_args = (
                    i32([0] + list(np.cumsum(q_lens))),
                    i32([0] + list(np.cumsum(page_counts))),
                    i32([p for i in range(len(kept))
                         for p in pages_of(i)[:page_counts[i]]]),
                    i32([(x - 1) % PAGE + 1 for x in kv_lens]))

            # the two-call plan inputs, shared by both merge variants
            sp_chunk = pack_chunk(torch, arena, case["groups"],
                                  pinned=True,
                                  attention_mode="merge_quant")
            meta = sp_chunk["meta"]
            cross = meta["cross"]
            rows_idx = cross["rows"]
            source = cross["source"]
            cu_a = meta["cu_a"].cpu()
            kept_pages = [-(-k // PAGE) for k in kept]
            paged_plan_args = (
                cross["cu_q"].cpu(),
                i32([0] + list(np.cumsum(kept_pages))),
                i32([p for i in range(len(kept))
                     for p in pages_of(i)[:kept_pages[i]]]),
                i32([(k - 1) % PAGE + 1 for k in kept]))

            for be in BACKENDS:
                # one causal paged call (the unified equivalent)
                if unified_ok:
                    try:
                        wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
                            ws(), "NHD", backend=be)
                        wrapper.plan(
                            *uni_plan_args, H, KH, D, PAGE,
                            causal=True,
                            q_data_type=torch.bfloat16,
                            kv_data_type=torch.bfloat16)

                        def run_fi_unified():
                            pipeline.kv_row_scatter(
                                k3, v3, uni["src"], uni["dst"], 0)
                            o = wrapper.run(q3, (kcache, vcache))
                            return pipeline.quant(o.view(n, -1))
                        row["paged_causal"][be] = _median_ms(
                            torch, run_fi_unified)
                        pipeline.kv_row_scatter(k3, v3, uni["src"],
                                                uni["dst"], 0)
                        got = wrapper.run(q3, (kcache, vcache))
                        row["paged_causal_max_abs"][be] = float(
                            (got.view(n, -1).float()
                             - ref.float()).abs().max().item())
                        del wrapper
                    except Exception as exc:
                        row["rejected"][f"paged_causal[{be}]"] = \
                            repr(exc)

                # two calls, merged by their merge_state and by our
                # fused kernel
                try:
                    ragged = fi.BatchPrefillWithRaggedKVCacheWrapper(
                        ws(), "NHD", backend=be)
                    ragged.plan(cu_a, cu_a, H, KH, D, causal=True,
                                q_data_type=torch.bfloat16,
                                kv_data_type=torch.bfloat16)
                    paged = fi.BatchPrefillWithPagedKVCacheWrapper(
                        ws(), "NHD", backend=be)
                    paged.plan(*paged_plan_args, H, KH, D, PAGE,
                               causal=False,
                               q_data_type=torch.bfloat16,
                               kv_data_type=torch.bfloat16)

                    def run_two_call():
                        if meta["kv_src"] is not None:
                            pipeline.kv_row_scatter(
                                k3, v3, meta["kv_src"],
                                meta["kv_dst"], 0)
                        oa, la = ragged.run(q3, k3, v3,
                                            return_lse=True)
                        q_suf = q3.index_select(0, rows_idx)
                        ob, lb = paged.run(q_suf, (kcache, vcache),
                                           return_lse=True)
                        merged, _ = fi.merge_state(
                            oa.index_select(0, rows_idx),
                            la.index_select(0, rows_idx), ob, lb)
                        o = oa.index_copy_(0, rows_idx, merged)
                        return pipeline.quant(o.view(n, -1))
                    row["two_call_merge_state"][be] = _median_ms(
                        torch, run_two_call)

                    def run_two_call_fused():
                        if meta["kv_src"] is not None:
                            pipeline.kv_row_scatter(
                                k3, v3, meta["kv_src"],
                                meta["kv_dst"], 0)
                        oa, la = ragged.run(q3, k3, v3,
                                            return_lse=True)
                        q_suf = q3.index_select(0, rows_idx)
                        ob, lb = paged.run(q_suf, (kcache, vcache),
                                           return_lse=True)
                        la_t = (la if la.shape[0] == n
                                else la.transpose(0, 1))
                        lb_t = (lb
                                if lb.shape[0] == rows_idx.shape[0]
                                else lb.transpose(0, 1))
                        return pipeline.merge_attn_quant(
                            oa, la_t, ob, lb_t, source)
                    row["two_call_fused_merge"][be] = _median_ms(
                        torch, run_two_call_fused)
                    del ragged, paged
                except Exception as exc:
                    row["rejected"][f"two_call[{be}]"] = repr(exc)

            us = {}
            for group in ("fa3", "paged_causal",
                          "two_call_merge_state",
                          "two_call_fused_merge"):
                for k, v in row[group].items():
                    key = k if group == "fa3" else f"{group}[{k}]"
                    us[key] = round(v * 1e3 / n, 3)
            row["us_per_fresh_token"] = us
            report["shapes"][name] = row
            print(f"[flashinfer_tuned] {name}: {json.dumps(us)}",
                  flush=True)
            if row["rejected"]:
                print(f"[flashinfer_tuned] {name} rejected: "
                      f"{list(row['rejected'])}", flush=True)

    tag = "" if H == 32 else f"_{H}h"
    return _write(report, f"flashinfer_tuned{tag}")


@app.local_entrypoint()
def run_bench_tuned(q_heads: int = 32):
    print(bench_tuned.remote(q_heads))
