"""Milestone 1: the committed filter and join results, reproduced on
the new executor. Three cells, run in order; a failed gate stops the
milestone and reopens the design's executor section.

  probe    correctness gates before any timed run: attention math vs
           an fp32 reference, packed answers vs unpacked per-pair
           references, kept-KV replay, multi-group and mixed
           fresh/kept chunks, paged vs gather cross-attention, and a
           rate read at the large-chunk geometry. Every parity gate
           requires 0 disagreements.

  filter   the committed 10k-document five-filter workload
           (filter_cells_bf16.json is the bf16-KV reference: rewind
           wall ~38.0 s, survivors 1873, answered 23381).
           PREDICTION, stated before the run: ~3.84M fresh tokens at
           the 121k tok/s packed rate -> ~32 s, survivors near 1873.

  join     the committed 256k-pair 2-way BioDEX join
           (join2way.json: packed wall 103.5-103.6 s, 8,417,425
           fresh tokens, 77 chunks, 177,831 YES).
           PREDICTION: within 10% of 103.6 s, yes count near 177,831.

Run from the quail/ directory (tee to a file per house rule):

    uv run modal run tests/gpu/milestone1.py::run_probe  2>&1 | tee results/m1_probe.log
    uv run modal run tests/gpu/milestone1.py::run_filter 2>&1 | tee results/m1_filter.log
    uv run modal run tests/gpu/milestone1.py::run_join   2>&1 | tee results/m1_join.log
    uv run modal run tests/gpu/milestone1.py::run_join3  2>&1 | tee results/m1_join3.log
"""

import json
import os
from pathlib import Path

import modal

from corpus import MODEL

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# vllm pinned for reproducibility against the committed walls (same
# kernels, same loader). Nothing here reaches vLLM private surface;
# the pin is measurement hygiene, not an API dependency.
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          # variable chunk shapes fragment the caching allocator;
          # expandable segments returns that memory to the pool
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # JIT artifacts persist on the kernel-cache volume so each
          # DeepGEMM/Triton configuration compiles once ever, not
          # once per container (both env spellings for DeepGEMM
          # version drift)
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "corpus", "baselines")
    # the package's data files: add_local_python_source ships only
    # .py, and the calibrate cell reads the anchor JSON in-container
    .add_local_dir("quail/calibration",
                   remote_path="/root/quail/calibration")
)

# House rule: never create new Modal app names - caches and warm state
# ride on the app, so new GPU cells attach to this app (or the
# worker's "quail-engine"), never to a fresh one.
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


def _boot():
    """Model, pipeline, arena, answerers - the one executor, sized by
    the plan arithmetic (chunk budget at the kernel cap, arena from
    the admission budget, bf16 KV)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import Answerer, AsyncAnswers
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_model(MODEL)
    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    arena_tok = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    spec = QWEN3_4B_FP8
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)
    return (torch, F, tokenizer, model, pipeline, arena, answerer,
            async_ans, exec_budget, arena_tok)


def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/m1", exist_ok=True)
    with open(f"/results/m1/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return json.dumps(result)


# -------------------------------------------------------------- probe

@app.function(timeout=2400, **GPU_KW)
def probe() -> str:
    import math

    import torch

    from corpus import biodex_sample
    from quail.executor.loop import pack_chunk

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    result = {"exec_budget": exec_budget, "arena_tokens": arena_tok,
              "gates": {}}

    # ---- 1. attention math in isolation: the two-call paged merge
    # against a plain fp32 reference on random [prefix | 3 suffixes]
    H, KH, D = pipeline.num_q_heads, pipeline.num_kv_heads, \
        pipeline.head_dim
    f0, sufs0 = 256, [32, 32, 32]
    n0 = f0 + sum(sufs0)
    gen = torch.Generator(device="cuda").manual_seed(7)
    q0 = torch.randn(n0, H, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    k0 = torch.randn(n0, KH, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    v0 = torch.randn(n0, KH, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    arena.alloc("m0", f0)
    cu = [0, f0]
    for s in sufs0:
        cu.append(cu[-1] + s)
    table, _ = arena.block_table(["m0"])
    meta0 = dict(
        layer=0, paged=True,
        kv_src=torch.arange(f0, dtype=torch.int64, device="cuda"),
        kv_dst=arena.rows_gpu("m0")[:f0],
        cu_a=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        max_a=f0,
        cross=dict(
            rows=torch.arange(f0, n0, device="cuda"),
            cu_q=torch.tensor([0, n0 - f0], dtype=torch.int32,
                              device="cuda"),
            max_q=n0 - f0, keys=["m0"],
            used=torch.tensor([f0], dtype=torch.int32, device="cuda"),
            max_used=f0, table=table,
            cu_k=torch.tensor([0, f0], dtype=torch.int32,
                              device="cuda")))
    with torch.inference_mode():
        shared0 = pipeline.attention(
            q0.reshape(n0, H * D), k0.reshape(n0, KH * D),
            v0.reshape(n0, KH * D), meta0).view(n0, H, D)

        def ref_rows(qr, kr, vr):
            kk = kr.repeat_interleave(H // KH, dim=1).float()
            vv = vr.repeat_interleave(H // KH, dim=1).float()
            s = torch.einsum("qhd,khd->hqk", qr.float(), kk)
            s = s / math.sqrt(D)
            nq = qr.shape[0]
            mask = torch.triu(torch.ones(nq, nq, device="cuda",
                                         dtype=torch.bool), 1)
            s.masked_fill_(mask[None], float("-inf"))
            return torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)

        worst = 0.0
        off = f0
        for s in sufs0:
            idx = torch.tensor(list(range(f0))
                               + list(range(off, off + s)),
                               device="cuda")
            ref = ref_rows(q0.index_select(0, idx),
                           k0.index_select(0, idx),
                           v0.index_select(0, idx))[f0:]
            got = shared0[off:off + s].float()
            worst = max(worst, (ref - got).abs().max().item())
            off += s
    arena.free_key("m0")
    result["gates"]["attention_math_max_diff"] = round(worst, 4)

    # ---- data for the answer gates
    data = biodex_sample(tokenizer, n_reports=4)
    prefix = data["prefixes"][0]
    sufs = data["suffixes"][:64]
    result["lengths"] = dict(
        prefix=len(prefix),
        suffix_mean=round(sum(map(len, sufs)) / len(sufs), 1))

    def fresh_group(key, p, sfx):
        return dict(key=key, prefix=p, f=len(p), suffixes=sfx)

    def kept_group(key, f, sfx):
        return dict(key=key, prefix=None, f=f, suffixes=sfx)

    with torch.inference_mode():
        # 2. shared (paged) vs unshared per-pair references
        arena.alloc("r0", len(prefix))
        shared_chunk = pack_chunk(torch, arena,
                                  [fresh_group("r0", prefix, sufs)])
        normed_shared = pipeline.forward_chunk(shared_chunk)
        shared_answers = answerer(normed_shared)

        unshared_answers, unshared_rows = [], []
        for suf in sufs:
            one = pack_chunk(torch, arena,
                             [dict(key="ref", prefix=prefix + suf,
                                   f=len(prefix) + len(suf),
                                   suffixes=[])])
            one["final_indices"] = torch.tensor(
                [len(prefix) + len(suf) - 1], device="cuda")
            normed = pipeline.forward_chunk(one)
            unshared_rows.append(normed)
            unshared_answers.extend(answerer(normed))

        # 3. kept-KV replay: the same suffixes against the pages only
        kept_chunk = pack_chunk(torch, arena,
                                [kept_group("r0", len(prefix), sufs)])
        kept_answers = answerer(pipeline.forward_chunk(kept_chunk))

        # 6. paged vs gather on the same kept chunk
        gather_chunk = pack_chunk(torch, arena,
                                  [kept_group("r0", len(prefix), sufs)])
        gather_chunk["meta"]["paged"] = False
        gather_answers = answerer(pipeline.forward_chunk(gather_chunk))

        # 4. multi-group: two reports in ONE chunk answer as alone
        p1 = data["prefixes"][1]
        sufs2 = data["suffixes"][64:96]
        arena.alloc("a0", len(prefix))
        alone0 = pipeline.forward_chunk(pack_chunk(
            torch, arena, [fresh_group("a0", prefix, sufs2)]))
        arena.free_key("a0")
        arena.alloc("a1", len(p1))
        alone1 = pipeline.forward_chunk(pack_chunk(
            torch, arena, [fresh_group("a1", p1, sufs2)]))
        arena.free_key("a1")
        arena.alloc("b0", len(prefix))
        arena.alloc("b1", len(p1))
        both = pipeline.forward_chunk(pack_chunk(
            torch, arena, [fresh_group("b0", prefix, sufs2),
                           fresh_group("b1", p1, sufs2)]))
        arena.free_key("b0")
        arena.free_key("b1")
        multi_expect = answerer(alone0) + answerer(alone1)
        multi_got = answerer(both)
        dis_multi = sum(x != y for x, y in zip(multi_got, multi_expect))

        # 5. mixed gate: a fresh group and a kept group in ONE chunk
        arena.alloc("m1", len(p1))
        mixed = pipeline.forward_chunk(pack_chunk(
            torch, arena,
            [fresh_group("m1", p1, sufs2),
             kept_group("r0", len(prefix), sufs2)]))
        arena.free_key("m1")
        mixed_expect = answerer(alone1) + answerer(alone0)
        mixed_got = answerer(mixed)
        dis_mixed = sum(x != y for x, y in zip(mixed_got, mixed_expect))
    arena.free_key("r0")

    gap = (normed_shared.float()
           - torch.cat(unshared_rows).float()).abs().max().item()
    result["gates"]["shared_vs_unshared_max_hidden_gap"] = round(gap, 4)
    dis_su = sum(a != b for a, b in zip(shared_answers,
                                        unshared_answers))
    dis_ks = sum(a != b for a, b in zip(kept_answers, shared_answers))
    dis_pg = sum(a != b for a, b in zip(gather_answers, kept_answers))
    result["gates"].update(
        shared_vs_unshared_disagreements=int(dis_su),
        kept_vs_shared_disagreements=int(dis_ks),
        paged_vs_gather_disagreements=int(dis_pg),
        multi_group_disagreements=int(dis_multi),
        mixed_kept_fresh_disagreements=int(dis_mixed),
    )

    # ---- 7. rate at the large-chunk geometry: 2 reports x all terms
    import time

    from quail.executor.pack import plan_groups
    data_full = biodex_sample(tokenizer, n_reports=2)
    with torch.inference_mode():
        chunks = []
        for r, p in enumerate(data_full["prefixes"]):
            key = f"rate{r}"
            arena.alloc(key, len(p))
            first = True
            for start, end in plan_groups(
                    len(p), [len(s) for s in data_full["suffixes"]],
                    exec_budget):
                g = (fresh_group(key, p,
                                 data_full["suffixes"][start:end])
                     if first else
                     kept_group(key, len(p),
                                data_full["suffixes"][start:end]))
                first = False
                chunks.append(pack_chunk(torch, arena, [g]))
        for c in chunks[:2]:
            pipeline.forward_chunk(c)     # deepgemm + shape warmup
        torch.cuda.synchronize()
        walls = []
        for _ in range(2):
            t0 = time.perf_counter()
            for c in chunks:
                pipeline.forward_chunk(c)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
        wall = min(walls)
        tokens = sum(c["tokens"] for c in chunks)
        result["rate"] = dict(chunks=len(chunks), tokens=tokens,
                              wall_s=round(wall, 3),
                              tok_s=round(tokens / wall, 1))
    for r in range(2):
        arena.free_key(f"rate{r}")
    result["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    result["pass"] = (worst < 0.05 and dis_su == 0 and dis_ks == 0
                      and dis_pg == 0 and dis_multi == 0
                      and dis_mixed == 0)
    return _write(result, "probe")


# ------------------------------------------------------------- filter

@app.function(timeout=3600, **GPU_KW)
def filter_run(n_docs: int = 10000, reps: int = 2,
               budget: int = 0, timing: bool = False) -> str:
    """budget overrides the chunk budget: the reference cell at the
    ladder's largest measured point (25,305) tells a rate change at
    large chunks apart from a slow loop. timing adds a per-phase CPU
    breakdown of run_filter to each rep (host-side timers only)."""
    import time

    from corpus import build_corpus
    from quail.executor.loop import run_filter

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    if budget:
        exec_budget = budget
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    corpus_tokens = sum(len(b) for b in body_ids)
    report = dict(
        cell="m1_filter", n_docs=n_docs, corpus_tokens=corpus_tokens,
        n_filters=len(q_ids), exec_budget=exec_budget,
        arena_tokens=arena_tok, kv="bf16",
        prediction=("~3.84M fresh tokens at the 121k tok/s packed "
                    "rate -> ~32 s; committed bf16 chain reference "
                    "38.0 s, survivors 1873, answered 23381"),
        reference=dict(artifact="filter_cells_bf16.json",
                       wall_s=38.0, survivors=1873, answered=23381,
                       wrong=6229),
        runs=[])
    print(f"[m1_filter] {report['prediction']}", flush=True)

    # boot-side warmup: one budget-sized chunk, so DeepGEMM and
    # Triton compile their full-size configurations outside the
    # measured walls (a small warmup chunk left ~52 s of JIT inside
    # rep 0)
    from quail.executor.loop import warm_kernels
    t_warm = time.perf_counter()
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, body_ids,
                     q_ids, exec_budget)
    torch.cuda.synchronize()
    kernel_cache.commit()    # keep the compiles even if the run dies
    report["warmup_s"] = round(time.perf_counter() - t_warm, 2)
    print(f"[m1_filter] warmup {report['warmup_s']} s", flush=True)

    for rep in range(reps):
        torch.cuda.reset_peak_memory_stats()
        chunk_trace = []
        timers = {} if timing else None
        t0 = time.perf_counter()
        with torch.inference_mode():
            answers, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                exec_budget, trace=chunk_trace, timing=timers)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        answered = sum(len(v) for v in answers.values())
        survivors = [d for d, row in answers.items()
                     if len(row) == len(q_ids) and all(row)]
        wrong = 0
        for d, row in answers.items():
            for j, bit in enumerate(row):
                if bit != int(flags[d][j]):
                    wrong += 1
        chunk_ms = [round(e0.elapsed_time(e1), 1) for _, e0, e1 in spans]
        slow = sorted(range(len(chunk_ms)), key=lambda i: -chunk_ms[i])
        row = dict(rep=rep, wall=round(wall, 2), fresh_tokens=tokens,
                   tok_s=round(tokens / wall, 1), answered=answered,
                   survivors=len(survivors), wrong=wrong,
                   chunks=len(spans),
                   gpu_s=round(sum(chunk_ms) / 1e3, 2),
                   chunk_ms_first8=chunk_ms[:8],
                   slow_chunks=[dict(idx=i, ms=chunk_ms[i],
                                     **chunk_trace[i])
                                for i in slow[:8]],
                   peak_gib=round(
                       torch.cuda.max_memory_allocated() / 2**30, 2))
        if timers is not None:
            row["cpu_phase_s"] = {k: round(v, 3)
                                  for k, v in sorted(timers.items())}
        report["runs"].append(row)
        print(f"[m1_filter] {row}", flush=True)
    return _write(report, "filter")


# --------------------------------------------------------------- join

@app.function(timeout=7200, **GPU_KW)
def join_run(n_reports: int = 100, reps: int = 2) -> str:
    import time

    from corpus import biodex_sample
    from quail.executor.loop import run_join

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes, suffixes = data["prefixes"], data["suffixes"]
    n_terms = len(suffixes)
    report = dict(
        cell="m1_join", n_reports=n_reports, n_terms=n_terms,
        pairs=n_reports * n_terms, exec_budget=exec_budget,
        arena_tokens=arena_tok, kv="bf16",
        prediction=("within 10% of the committed packed 103.6 s; "
                    "~8.42M fresh tokens, ~77 chunks, yes near "
                    "177,831"),
        reference=dict(artifact="join2way.json", wall_s=103.6,
                       fresh_tokens=8417425, chunks=77, yes=177831),
        lengths=dict(
            prefix_mean=round(sum(map(len, prefixes)) / n_reports, 1),
            suffix_mean=round(sum(len(s) for s in suffixes)
                              / n_terms, 1)),
        runs=[])
    print(f"[m1_join] {report['prediction']}", flush=True)

    with torch.inference_mode():
        run_join(torch, arena, pipeline, async_ans, prefixes[:2],
                 [suffixes[:64]], exec_budget)     # warmup

    for rep in range(reps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            ans, spans, tokens = run_join(
                torch, arena, pipeline, async_ans, prefixes,
                [suffixes], exec_budget)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        answers = []
        for a in range(n_reports):
            answers.extend(ans[0][a])
        row = dict(rep=rep, wall=round(wall, 2), chunks=len(spans),
                   fresh_tokens=tokens,
                   tok_s=round(tokens / wall, 1), yes=sum(answers),
                   peak_gib=round(
                       torch.cuda.max_memory_allocated() / 2**30, 2))
        report["runs"].append(row)
        print(f"[m1_join] {row}", flush=True)
    return _write(report, "join")


# ---------------------------------------------------------- the 3-way

@app.function(timeout=7200, **GPU_KW)
def join3_run() -> str:
    """The replay gate: the committed planted 3-way chain. Gating and
    dedup on the new executor must reproduce the nested-loop reference
    exactly, and stage-2 pairs must equal survivors x |C|."""
    import time

    from corpus import N_B, N_C, nway_corpus, nway_truth
    from quail.executor.loop import run_join
    from quail.executor.pack import (assemble, brute_force_triples,
                                     gate)

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    b_prefix, a_suffix, c_suffix = nway_corpus(tokenizer)
    truth1, _ = nway_truth()
    report = dict(
        cell="m1_join3",
        prediction=("triples equal the nested-loop replay reference; "
                    "stage-2 pairs = survivors x 100; total wall near "
                    "the committed ~70 s"),
        reference=dict(artifact="join_nway3.json"))

    with torch.inference_mode():
        run_join(torch, arena, pipeline, async_ans, b_prefix[:1],
                 [a_suffix[:8], c_suffix[:8]], exec_budget,
                 group_size=1)      # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ans, spans, tokens = run_join(
            torch, arena, pipeline, async_ans, b_prefix,
            [a_suffix, c_suffix], exec_budget, group_size=1)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    ans1, ans2 = ans[0], ans[1]
    survivors = gate(ans1)
    staged = assemble(ans1, ans2)
    reference = brute_force_triples(ans1, ans2)
    stage2_pairs = sum(len(row) for row in ans2.values())
    wrong1 = sum(ans1[b][a] != truth1[b][a]
                 for b in range(N_B) for a in range(len(ans1[b])))
    report["result"] = dict(
        total_wall_s=round(wall, 1), fresh_tokens=tokens,
        chunks=len(spans),
        survivors=len(survivors),
        stage2_pairs=stage2_pairs,
        stage2_pairs_expected=len(survivors) * N_C,
        triples=len(staged),
        triples_match_replay=staged == reference,
        stage1_model_vs_planted_wrong=wrong1,
        planted_expected_survivors=N_B - 20,
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    return _write(report, "join3")


# ---------------------------------------------------------- the store

@app.function(timeout=5400, image=image, gpu="H100!", memory=327680,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def filter_store_run(n_docs: int = 5000, capacity_gb: int = 250,
                     warm_reps: int = 2) -> str:
    """The store gate: the five-filter workload cold (with offload),
    then warm (restore instead of recompute).

    PREDICTION, from the committed persist result (chain 10k: 44.3 s
    cold offload pass, 16.3 s per warm pass, 1.9-2.1x): at 5,000
    documents (~1.6M corpus tokens, ~236 GB of bf16 KV) the cold pass
    lands near half the 10k wall (~20 s) plus the offload tail, and
    the warm passes land near half the cold wall - restore streams on
    the side channel while the question chunks compute.

    bf16 at 10k docs needs ~472 GB of pinned host memory, past this
    container's 320 GB, so the gate runs at 5,000 documents - the
    same per-token economics, a corpus the pool holds whole."""
    import time

    from corpus import build_corpus
    from quail.executor.kvstore import PinnedStore
    from quail.executor.loop import run_filter
    from quail.specs import QWEN3_4B_FP8

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    corpus_tokens = sum(len(b) for b in body_ids)
    kappa = QWEN3_4B_FP8.kappa

    t_store = time.perf_counter()
    store = PinnedStore(
        capacity_tokens=int(capacity_gb * 1e9) // int(kappa),
        n_layers=QWEN3_4B_FP8.layers, n_kv=QWEN3_4B_FP8.n_kv,
        d_head=QWEN3_4B_FP8.d_head,
        max_doc_tokens=max(len(b) for b in body_ids))
    store_init_s = round(time.perf_counter() - t_store, 2)

    from quail.executor.loop import warm_kernels
    t_warm = time.perf_counter()
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, body_ids,
                     q_ids, exec_budget)
    torch.cuda.synchronize()
    kernel_cache.commit()

    report = dict(
        cell="m1_filter_store", n_docs=n_docs,
        corpus_tokens=corpus_tokens,
        store_capacity_gb=capacity_gb,
        store_kv_gb=round(corpus_tokens * kappa / 1e9, 1),
        store_init_s=store_init_s,
        warmup_s=round(time.perf_counter() - t_warm, 2),
        prediction=("cold ~20 s plus offload tail; warm passes near "
                    "half the cold wall (committed persist: 1.9-2.1x)"),
        runs=[])
    print(f"[m1_store] init {store_init_s} s; "
          f"{report['store_kv_gb']} GB of KV into a {capacity_gb} GB "
          f"pool; {report['prediction']}", flush=True)

    for rep in range(1 + warm_reps):
        stats = {}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            answers, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                exec_budget, store=store, store_hash="m1",
                store_min_tokens=1, stats=stats)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        survivors = [d for d, row in answers.items()
                     if len(row) == len(q_ids) and all(row)]
        row = dict(rep=rep, kind="cold" if rep == 0 else "warm",
                   wall=round(wall, 2), fresh_tokens=tokens,
                   tok_s=round(tokens / wall, 1),
                   survivors=len(survivors), chunks=len(spans),
                   peak_gib=round(
                       torch.cuda.max_memory_allocated() / 2**30, 2),
                   **stats)
        report["runs"].append(row)
        print(f"[m1_store] {row}", flush=True)
    cold = report["runs"][0]["wall"]
    warm = min(r["wall"] for r in report["runs"][1:])
    report["warm_speedup"] = round(cold / warm, 2)
    return _write(report, "filter_store")


# ----------------------------------------------------------- profiling

@app.function(timeout=3600, **GPU_KW)
def profile_filter_run(n_docs: int = 3000) -> str:
    """Torch-profile a steady-state filter run to name the ~1.3
    us/token gap between the executor (9.9 us/token measured) and the
    no-KV ladder ceiling (8.26). KV writes, paged reads, and the LSE
    merge only account for ~0.2 of it; this trace decides among:
    GEMMs under peak at our shapes, oversized elementwise chains, or
    scheduling gaps (kernel sum well under region wall)."""
    import time

    from corpus import build_corpus
    from quail.executor.loop import run_filter, warm_kernels

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, body_ids,
                     q_ids, exec_budget)
        run_filter(torch, arena, pipeline, async_ans, body_ids, q_ids,
                   exec_budget)      # unprofiled reference pass
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            answers, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                exec_budget)
        torch.cuda.synchronize()
    region_wall = time.perf_counter() - t0

    cats = dict(gemm=0.0, attention=0.0, triton_fused=0.0, quant=0.0,
                copies=0.0, other=0.0)
    rows = []
    for ev in prof.key_averages():
        cuda_us = getattr(ev, "self_device_time_total", 0) or \
            getattr(ev, "self_cuda_time_total", 0)
        if not cuda_us:
            continue
        name = ev.key
        low = name.lower()
        if "deep_gemm" in low or "sm90_fp8" in low or "gemm" in low:
            cats["gemm"] += cuda_us
        elif "flash" in low or "attn" in low:
            cats["attention"] += cuda_us
        elif any(k in low for k in ("silu_mul", "add_rms", "qk_norm")):
            cats["triton_fused"] += cuda_us
        elif "quant" in low:
            cats["quant"] += cuda_us
        elif "memcpy" in low or "copy" in low:
            cats["copies"] += cuda_us
        else:
            cats["other"] += cuda_us
        rows.append((round(cuda_us / 1e6, 3), ev.count, name[:90]))
    rows.sort(reverse=True)
    busy_s = sum(cats.values()) / 1e6
    result = dict(
        n_docs=n_docs, fresh_tokens=tokens, chunks=len(spans),
        region_wall_s=round(region_wall, 2),
        cuda_busy_s=round(busy_s, 2),
        gap_wall_minus_busy_s=round(region_wall - busy_s, 2),
        us_per_token=round(region_wall * 1e6 / tokens, 2),
        category_s={k: round(v / 1e6, 2) for k, v in cats.items()},
        ideal_gemm_s=round(2 * 3.6e9 * tokens / 1.979e15, 2),
        top_kernels=[dict(s=s, n=n, name=k) for s, n, k in rows[:25]])
    prof.export_chrome_trace("/results/m1/torchprof_filter.json.gz")
    return _write(result, "profile_filter")


@app.local_entrypoint()
def run_profile_filter(n_docs: int = 3000,
                       out: str = "results/profile_filter.json"):
    _save(profile_filter_run.remote(n_docs), out)


# ---------------------------------------------------------- calibrate

@app.function(timeout=3600, **GPU_KW)
def calibrate_run(tokens_per_point: int = 1_500_000) -> str:
    """Measure the planner's calibration constants for this
    (model, device) pair - the `quail calibrate` step the calibration
    module's docstring names.

    a and a2 come from a document-length sweep: run the packed filter
    at fixed lengths, take seconds per fresh token at each, and fit
    t(h) = a + a2*h - exactly the form restore_crossover_tokens and
    the store break-even consume. The channel probes re-measure
    pinned copy bandwidth. q_kv is NOT measured: the fp8 arena path
    is not implemented, so the value carried through is the loaded
    (anchor or spec-scaled) one, and the provenance says so.
    """
    import time

    from corpus import question, token_stream
    from quail.executor.loop import run_filter, warm_kernels
    from quail.planner.calibration import fit_affine, load_calibration
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    spec, device = QWEN3_4B_FP8, H100_SXM
    lengths = (256, 1024, 4096, 8192)

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()
    q_ids = [tokenizer(question(1),
                       add_special_tokens=False)["input_ids"]]
    stream = token_stream(tokenizer, max(lengths) + tokens_per_point)
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans,
                     [stream[:512]] * 64, q_ids, exec_budget)
    torch.cuda.synchronize()

    points, rows = [], []
    for h in lengths:
        n_docs = max(8, tokens_per_point // h)
        body_ids = [stream[i * h:(i + 1) * h] for i in range(n_docs)]
        t0 = time.perf_counter()
        with torch.inference_mode():
            _, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                exec_budget)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        gpu_s = sum(e0.elapsed_time(e1) for _, e0, e1 in spans) / 1e3
        t_wall = wall / tokens
        points.append((h, t_wall))
        rows.append(dict(doc_tokens=h, n_docs=n_docs,
                         fresh_tokens=tokens, wall_s=round(wall, 2),
                         gpu_s=round(gpu_s, 2),
                         us_per_token_wall=round(t_wall * 1e6, 3),
                         us_per_token_gpu=round(gpu_s / tokens * 1e6,
                                                3)))
        print(f"[calibrate] {rows[-1]}", flush=True)

    a, a2 = fit_affine(points)
    a2 = max(a2, 0.0)    # a noisy flat sweep must not go negative

    # channel probes: 2 GiB timed copies, pinned and unpinned
    channels = {}
    buf_bytes = 2 << 30
    dev_buf = torch.empty(buf_bytes, dtype=torch.uint8, device="cuda")
    for pinned in (True, False):
        host = torch.empty(buf_bytes, dtype=torch.uint8,
                           pin_memory=pinned)
        name = "pinned" if pinned else "unpinned"
        for tag, src, dst in ((f"{name}_h2d", host, dev_buf),
                              (f"{name}_d2h", dev_buf, host)):
            dst.copy_(src)                       # first-touch warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize()
            channels[tag] = round(
                3 * buf_bytes / (time.perf_counter() - t0), 0)
        del host
    del dev_buf

    loaded = load_calibration(spec, device)
    result = dict(
        model=spec.name, device=device.name,
        a_s_per_token=a, a2_s_per_token2=a2,
        q_kv_s_per_token=loaded.q_kv_s_per_token,
        provenance=dict(
            a=("wall seconds per fresh token, length sweep "
               f"{list(lengths)} at ~{tokens_per_point} tokens per "
               "point, affine fit intercept"),
            a2="affine fit slope of the same sweep",
            q_kv=("not measured: fp8 arena not implemented; carried "
                  f"from '{loaded.source}'")),
        points=rows,
        channels_measured_bytes_per_s=channels,
        loaded_before=dict(a=loaded.a_s_per_token,
                           a2=loaded.a2_s_per_token2,
                           source=loaded.source))
    return _write(result, "calibrate")


@app.local_entrypoint()
def run_calibrate(out: str = "results/calibrate.json",
                  commit: bool = False):
    """--commit writes the three constants into
    quail/calibration/{model}_{device}.json, where load_calibration
    reads them; without it the measurement only lands in results/."""
    payload = calibrate_run.remote()
    _save(payload, out)
    if commit:
        d = json.loads(payload)
        keep = {k: d[k] for k in
                ("model", "device", "a_s_per_token",
                 "a2_s_per_token2", "q_kv_s_per_token", "provenance")}
        dest = (Path(__file__).resolve().parents[2] / "quail"
                / "calibration" / f"{d['model']}_{d['device']}.json")
        with open(dest, "w") as f:
            json.dump(keep, f, indent=2)
        print(f"committed {dest}")


# ----------------------------------------------------------- baselines

@app.function(timeout=5400, **GPU_KW)
def baseline_filter_run(n_docs: int = 10000, reps: int = 2) -> str:
    """Stock vLLM on the committed five-filter workload: the
    pipelined per-(document, stage) client under the plan's token
    budget, prefix caching on - the strongest stock client the
    exploration built.

    PREDICTION: the committed stock band, 39.2-43.2 s per rep
    (bf16 40.8/39.2/39.5; fp8 42.8-43.2), against the packed
    executor's measured 39.4-39.9 s."""
    from baselines.stock import run_filter_chain
    from corpus import MODEL, build_corpus
    from vllm import LLM, SamplingParams
    from quail.executor.loop import yes_no_ids

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    yes, no = yes_no_ids(tokenizer)

    # the committed BF16-KV stock knobs (our engine's KV is bf16, so
    # the fp8 config's 749,782 budget oversubscribes the ~473k-token
    # pool 1.6x - measured tonight at 47-79 s of thrash): budget
    # 374,891, 2,648 sequences, 25,305 step tokens, prefix caching on
    llm = LLM(model=MODEL, max_num_batched_tokens=25_305,
              max_num_seqs=2648, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=sorted(yes | no))
    engine = llm.llm_engine
    budget = 374_891     # the committed bf16-KV admission budget
    report = dict(cell="baseline_filter", n_docs=n_docs,
                  submission="separate requests per (document, stage), "
                             "pipelined, document-cap admission",
                  budget_tokens=budget, step_tokens=25_305,
                  prediction="committed bf16 stock band 39.2-40.8 s",
                  runs=[])
    # warm the engine (kernel compile, allocator)
    run_filter_chain(engine, sampling, body_ids[:64], q_ids, budget,
                     tag="w", yes_ids=yes)
    for rep in range(reps):
        r = run_filter_chain(engine, sampling, body_ids, q_ids,
                             budget, tag=f"r{rep}", yes_ids=yes)
        row = dict(rep=rep, wall=round(r["wall"], 2),
                   requests=r["requests"],
                   fresh_tokens=r["prompt_tokens"] - r["cached_tokens"],
                   survivors=len(r["survivors"]))
        report["runs"].append(row)
        print(f"[baseline_filter] {row}", flush=True)
    return _write(report, "baseline_filter")


@app.function(timeout=5400, **GPU_KW)
def baseline_join_run(n_reports: int = 60, n_cands: int = 1200,
                      reps: int = 2) -> str:
    """Stock vLLM on the dispatch gate's 72k-pair synthetic join:
    one request per pair, anchor-major, prefix caching on.

    PREDICTION: fresh ~2.7M tokens at the committed stock effective
    rate (~17k tok/s) -> 2.5-3 min per rep, against the packed
    executor's measured 31.1 s on one GPU (the committed BioDEX
    shape measured 4.1x)."""
    from baselines.stock import run_join_grouped
    from corpus import MODEL
    from vllm import LLM, SamplingParams
    from quail.executor.loop import yes_no_ids

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    filler = ("The projector hummed while the reel changed and nobody "
              "in the back row noticed the splice. ")
    colors = ("blue", "red", "green", "yellow", "purple", "orange")

    def tok(t):
        return tokenizer(t, add_special_tokens=False)["input_ids"]

    pre = tok("You will be shown a scene report and one candidate "
              "color. Decide from the report's own words.\n\nREPORT:\n")
    prefixes = [pre + tok(filler * 110
                          + f"\n\nThe dominant color in this scene is "
                            f"{colors[i % 6]}.")
                for i in range(n_reports)]
    suffixes = [tok(f"\n\nCANDIDATE:\nThe candidate color is "
                    f"{colors[j % 6]}.\nInstruction: answer YES if "
                    f"the report says its dominant color is the "
                    f"candidate color, NO otherwise.\nANSWER=")
                for j in range(n_cands)]
    yes, no = yes_no_ids(tokenizer)
    mean_pair = (sum(map(len, prefixes)) * n_cands
                 + n_reports * sum(map(len, suffixes))) \
        // (n_reports * n_cands) + 1
    max_seqs = max(64, min(4096, 749_782 // mean_pair))
    # 0.92, same as quail's pool fraction. The committed join2way
    # stock arm ran 0.88 only because its container booted two
    # executors back to back; standalone, stock gets the full pool.
    llm = LLM(model=MODEL, max_num_batched_tokens=25_305,
              max_num_seqs=max_seqs, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=sorted(yes | no))
    report = dict(cell="baseline_join", n_reports=n_reports,
                  n_cands=n_cands, pairs=n_reports * n_cands,
                  submission="one request per pair, anchor-major, "
                             "prefix caching on",
                  max_num_seqs=max_seqs, step_tokens=25_305,
                  prediction="~2.5-3 min per rep vs the packed 31.1 s",
                  runs=[])
    run_join_grouped(llm, sampling, prefixes[:2], suffixes[:32], yes)
    for rep in range(reps):
        llm.reset_prefix_cache()
        r = run_join_grouped(llm, sampling, prefixes, suffixes, yes)
        row = dict(rep=rep, wall=round(r["wall"], 2),
                   fresh_tokens=r["fresh_tokens"],
                   tok_s=round(r["fresh_tokens"] / r["wall"], 1),
                   yes=sum(r["answers"]))
        report["runs"].append(row)
        print(f"[baseline_join] {row}", flush=True)
    return _write(report, "baseline_join")


# ----------------------------------------------------- join diagnostics

@app.function(timeout=1800, **GPU_KW)
def debug_join() -> str:
    """Pair-predicate diagnostic: the same pairs through the
    trivially-correct single-segment causal path vs the packed
    multi-group path. Disagreement would mean an executor bug on this
    shape; agreement means the answers are the model's.

    Measured twice (plain question and few-shot example): the 4B
    answers YES to every constrained one-token equality judgment on
    BOTH paths, 0 disagreements - the executor is exonerated, the
    checkpoint cannot judge symbolic equality. Content-style
    predicates (the BioDEX shape) discriminate."""
    import torch

    from quail.executor.loop import pack_chunk

    (torch, F, tokenizer, model, pipeline, arena, answerer, async_ans,
     exec_budget, arena_tok) = _boot()

    filler = ("The projector hummed while the reel changed and nobody "
              "in the back row noticed the splice. ")
    colors = ("blue", "red", "green", "yellow", "purple", "orange")
    pre_t = ("You will be shown a scene report and one candidate "
             "color. Decide from the report's own words.\n\nREPORT:\n")
    mid_t = "\n\nCANDIDATE:\n"
    tail_t = ("\nExample: if the report says the dominant color is "
              "green and the candidate color is green, the answer is "
              "YES. If the report says green and the candidate color "
              "is blue, the answer is NO.\nInstruction: answer YES if "
              "the report says its dominant color is the candidate "
              "color, NO otherwise.\nANSWER=")

    def tok(t):
        return tokenizer(t, add_special_tokens=False)["input_ids"]

    reports = [tok(filler * 30
                   + f"\n\nThe dominant color in this scene is "
                     f"{colors[i % 6]}.") for i in range(2)]
    cands = [tok(f"The candidate color is {colors[j % 6]}.")
             for j in range(12)]
    pre, mid, tail = tok(pre_t), tok(mid_t), tok(tail_t)
    prefixes = [pre + r for r in reports]
    suffixes = [mid + c + tail for c in cands]
    planted = [[1 if i % 6 == j % 6 else 0 for j in range(12)]
               for i in range(2)]

    result = {}
    with torch.inference_mode():
        # (a) single-segment reference: one pair per chunk, plain
        # causal attention, no paging anywhere
        ref = []
        for i, p in enumerate(prefixes):
            row = []
            for s in suffixes:
                one = pack_chunk(torch, arena,
                                 [dict(key=f"x{i}", prefix=p + s,
                                       f=len(p) + len(s), suffixes=[])])
                one["final_indices"] = torch.tensor(
                    [len(p) + len(s) - 1], device="cuda")
                row.extend(answerer(pipeline.forward_chunk(one)))
            ref.append(row)

        # (b) packed: both anchors and all suffixes in one chunk
        for i, p in enumerate(prefixes):
            arena.alloc(f"a{i}", len(p))
        packed_chunk = pack_chunk(
            torch, arena,
            [dict(key=f"a{i}", prefix=p, f=len(p), suffixes=suffixes)
             for i, p in enumerate(prefixes)])
        bits = answerer(pipeline.forward_chunk(packed_chunk))
        packed = [bits[:12], bits[12:]]
        for i in range(2):
            arena.free_key(f"a{i}")

    result["reference_rows"] = ref
    result["packed_rows"] = packed
    result["planted_rows"] = planted
    result["ref_vs_packed_disagreements"] = sum(
        a != b for ra, rp in zip(ref, packed)
        for a, b in zip(ra, rp))
    result["ref_vs_planted_wrong"] = sum(
        a != b for ra, rp in zip(ref, planted)
        for a, b in zip(ra, rp))
    return _write(result, "debug_join")


# ---------------------------------------------------------- entrypoints

def _save(payload, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(json.loads(payload), f, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def run_probe(out: str = "results/m1_probe.json"):
    _save(probe.remote(), out)


@app.local_entrypoint()
def run_filter(n_docs: int = 10000, reps: int = 2, budget: int = 0,
               out: str = "results/m1_filter.json"):
    _save(filter_run.remote(n_docs, reps, budget), out)


@app.local_entrypoint()
def run_filter_timing(n_docs: int = 3000, reps: int = 1,
                      out: str = "results/m1_filter_timing.json"):
    _save(filter_run.remote(n_docs, reps, 0, True), out)


@app.local_entrypoint()
def run_join(n_reports: int = 100, reps: int = 2,
             out: str = "results/m1_join.json"):
    _save(join_run.remote(n_reports, reps), out)


@app.local_entrypoint()
def run_join3(out: str = "results/m1_join3.json"):
    _save(join3_run.remote(), out)


@app.local_entrypoint()
def run_debug_join(out: str = "results/debug_join.json"):
    _save(debug_join.remote(), out)


@app.local_entrypoint()
def run_filter_store(n_docs: int = 5000, capacity_gb: int = 250,
                     warm_reps: int = 2,
                     out: str = "results/m1_filter_store.json"):
    _save(filter_store_run.remote(n_docs, capacity_gb, warm_reps), out)


@app.local_entrypoint()
def run_baseline_filter(n_docs: int = 10000, reps: int = 2,
                        out: str = "results/baseline_filter.json"):
    _save(baseline_filter_run.remote(n_docs, reps), out)


@app.local_entrypoint()
def run_baseline_join(out: str = "results/baseline_join.json"):
    _save(baseline_join_run.remote(), out)
