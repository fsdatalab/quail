"""Forward-pass ablation ladder: stock vLLM to the current packed executor,
one change per rung, on the 10,000-document five-filter workload.

Rungs A0/A1 are stock vLLM (fp8/bf16 KV). A2 is the packed executor with
vLLM's own between-GEMM kernels. A3 adds our three Triton kernels.

    uv run modal run ablations/forward_pass.py::run_all
"""

import json
import os

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# Same image and pins as tests/gpu/milestone1.py: vllm==0.26.0 is
# measurement hygiene (same kernels, same loader), not an API
# dependency.
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
    # corpus.py lives next to the milestone cells; mount it beside the
    # ablation cell rather than moving shared workload code
    .add_local_dir("tests/gpu", remote_path="/root/gpu_tests")
)

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
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

# the committed bf16-KV admission budget, shared by both stock rungs
# so KV dtype is the only variable between them
STOCK_BUDGET = 374_891
STOCK_STEP_TOKENS = 25_305
STOCK_MAX_SEQS = 2648


def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


def _wrong_count(answers_by_doc, flags):
    """answers_by_doc: doc -> list of stage bits. flags: doc -> planted
    0/1 per stage."""
    wrong = 0
    for d, row in answers_by_doc.items():
        for j, bit in enumerate(row):
            if bit != int(flags[d][j]):
                wrong += 1
    return wrong


# --------------------------------------------------------- stock rungs

STOCK_PREDICTIONS = {
    "A0": "42.8-43.2 s (exploration fp8 band)",
    "A1": "38.7-39.0 s (banked baseline_filter4.json)",
}


@app.function(timeout=5400, **GPU_KW)
def stock_rung(rung: str, kv_dtype: str, n_docs: int = 10000,
               reps: int = 2) -> str:
    """One stock rung: the pipelined stock client over the five-filter
    workload. kv_dtype is "fp8" for A0 and "auto" (bf16) for A1."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from baselines.stock import run_filter_chain
    from corpus import MODEL, build_corpus
    from quail.executor.loop import true_false_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    true, false = true_false_ids(tokenizer)

    report = dict(
        cell="ablation_stock", rung=rung, kv=kv_dtype, n_docs=n_docs,
        workload="committed 10k five-filter, pipelined "
                 "per-(document, stage) client, prefix caching on",
        budget_tokens=STOCK_BUDGET, step_tokens=STOCK_STEP_TOKENS,
        max_num_seqs=STOCK_MAX_SEQS,
        prediction=STOCK_PREDICTIONS[rung],
        runs=[])
    print(f"[ablation_stock] {rung} prediction: "
          f"{STOCK_PREDICTIONS[rung]}", flush=True)

    llm = LLM(model=MODEL, kv_cache_dtype=kv_dtype,
              max_num_batched_tokens=STOCK_STEP_TOKENS,
              max_num_seqs=STOCK_MAX_SEQS,
              gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=sorted(true | false))
    engine = llm.llm_engine
    # warm the engine (kernel compile, allocator) outside the
    # measured reps
    run_filter_chain(engine, sampling, body_ids[:64], q_ids,
                     STOCK_BUDGET, tag="w", true_ids=true)
    for rep in range(reps):
        r = run_filter_chain(engine, sampling, body_ids, q_ids,
                             STOCK_BUDGET, tag=f"{rung}{rep}",
                             true_ids=true)
        by_doc = {}
        for (i, j), bit in r["answers"].items():
            by_doc.setdefault(i, {})[j] = bit
        by_doc = {i: [stages[j] for j in sorted(stages)]
                  for i, stages in by_doc.items()}
        row = dict(rung=rung, kv=kv_dtype, rep=rep,
                   wall=round(r["wall"], 2),
                   requests=r["requests"],
                   fresh_tokens=r["prompt_tokens"] - r["cached_tokens"],
                   cached_tokens=r["cached_tokens"],
                   survivors=len(r["survivors"]),
                   wrong=_wrong_count(by_doc, flags))
        report["runs"].append(row)
        print(f"[ablation_stock] {row}", flush=True)
    return _write(report, f"stock_{rung.lower()}")


# -------------------------------------------------------- packed rungs

# name, Pipeline kernels, pinned staging, the change the rung adds.
# Pinned staging is on in both rungs: it is part of the packed base
# configuration, not a rung variable.
PACKED_RUNGS = (
    ("A2", "vllm", True,
     "packed executor, vLLM's between-GEMM kernels, kept KV, "
     "110,376-token chunks, pinned staging"),
    ("A3", "quail", True,
     "+ our three Triton kernels (the current executor)"),
)

PACKED_PREDICTIONS = {
    "A2": "~42.9 s: this rung measured 43.9 s before the direct KV "
          "write (kv_row_scatter); the write fix is worth ~1.0 s",
    "A3": "~33.5 s: 34.5 s before the KV write fix; the filter gate "
          "with the fix already measured 33.5/33.6 s "
          "(m1_filter_kvscatter.json)",
}

# banked current-executor counts (results/m1_filter.json); A3 must
# reproduce them exactly
BANKED_A3 = dict(survivors=4645, wrong=0, answered=40052)


@app.function(timeout=5400, **GPU_KW)
def packed_rungs(n_docs: int = 10000, reps: int = 2) -> str:
    """A2 and A3 in one container, one boot: same loop, same arena,
    same attention path, same pinned staging; only the kernel set
    moves between rungs."""
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from corpus import MODEL, build_corpus
    from quail.executor.arena import KVArena
    from quail.executor.attention import (FILTER_ATTENTION,
                                          Pipeline)
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     warm_kernels)
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
    pipeline = Pipeline(model, arena, attention_mode=FILTER_ATTENTION)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)

    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    corpus_tokens = sum(len(b) for b in body_ids)

    report = dict(
        cell="ablation_packed", n_docs=n_docs,
        corpus_tokens=corpus_tokens, n_filters=len(q_ids),
        exec_budget=exec_budget, arena_tokens=arena_tok, kv="bf16",
        rungs={name: dict(change=change,
                          prediction=PACKED_PREDICTIONS[name])
               for name, _, _, change in PACKED_RUNGS},
        runs={})
    print(f"[ablation_packed] predictions: "
          f"{json.dumps(PACKED_PREDICTIONS)}", flush=True)

    # one warmup covers every rung: the DeepGEMM configuration sweep
    # is kernel-mode independent, and the budget-sized pass warms the
    # Triton kernels and the attention path
    t_warm = time.perf_counter()
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, exec_budget,
                     model_name=MODEL)
    torch.cuda.synchronize()
    kernel_cache.commit()
    report["warmup_s"] = round(time.perf_counter() - t_warm, 2)
    print(f"[ablation_packed] warmup {report['warmup_s']} s",
          flush=True)

    answers_by_rung = {}
    for name, kernels, pinned, _change in PACKED_RUNGS:
        pipeline.kernels = kernels
        with torch.inference_mode():
            # per-rung warm, unmeasured: first-call op init for the
            # vLLM CUDA ops stays out of the measured reps
            run_filter(torch, arena, pipeline, async_ans,
                       body_ids[:256], q_ids, exec_budget,
                       pinned=pinned, arena_writes=True)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                torch.cuda.reset_peak_memory_stats()
                timers = {}
                t0 = time.perf_counter()
                answers, spans, tokens = run_filter(
                    torch, arena, pipeline, async_ans, body_ids,
                    q_ids, exec_budget, timing=timers, pinned=pinned,
                    arena_writes=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                answered = sum(len(v) for v in answers.values())
                survivors = [d for d, row in answers.items()
                             if len(row) == len(q_ids) and all(row)]
                row = dict(
                    rung=name, kernels=kernels, pinned=pinned, rep=rep,
                    wall=round(wall, 2), fresh_tokens=tokens,
                    tok_s=round(tokens / wall, 1), answered=answered,
                    survivors=len(survivors),
                    wrong=_wrong_count(answers, flags),
                    chunks=len(spans),
                    gpu_s=round(sum(e0.elapsed_time(e1)
                                    for _, e0, e1 in spans) / 1e3, 2),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2),
                    cpu_phase_s={k: round(v, 3) for k, v
                                 in sorted(timers.items())})
                rows.append(row)
                print(f"[ablation_packed] {row}", flush=True)
        answers_by_rung[name] = answers
        report["runs"][name] = rows

    # gates: A3 is the current executor, so its counts must equal the
    # banked ones exactly. A2 runs different quant kernels, so its
    # answers are reported against A3, not gated to zero.
    def disagreements(x, y):
        return sum(bit_x != bit_y
                   for d in x for bit_x, bit_y in zip(x[d], y[d]))

    a2, a3 = answers_by_rung["A2"], answers_by_rung["A3"]
    last_a3 = report["runs"]["A3"][-1]
    report["gates"] = dict(
        a2_vs_a3_disagreements=disagreements(a2, a3),
        a3_vs_banked=dict(
            measured={k: last_a3[k] for k in BANKED_A3},
            banked=BANKED_A3),
    )
    report["pass"] = all(last_a3[k] == v for k, v in BANKED_A3.items())
    return _write(report, "packed")


# ---------------------------------------------------- attention paths

ATTENTION_PATHS = (
    ("merge_quant", "two-call attention, fused merge and FP8 quant"),
    ("unified", "one causal paged attention call, separate quant"),
)


def _boot_executor(model):
    """Boot the packed executor and return (spec, tokenizer, arena,
    pipeline, answerer, async_answers, budget, arena_tokens)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import Answerer, AsyncAnswers
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model_mod = load_model(spec.hf_name)
    chunk = budgets.chunk_budget(spec, H100_SXM)
    arena_tok = budgets.arena_tokens(spec, H100_SXM, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model_mod, arena, kernels="quail",
                        attention_mode="merge_quant")
    answerer = Answerer(torch, F, model_mod, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    budget = min(chunk, pipeline.max_chunk_tokens)
    return (spec, tokenizer, arena, pipeline, answerer, async_ans,
            budget, arena_tok)


def _disagreements(left, right, n_stages=None):
    """Count answer differences between two {doc: bits} maps.

    Returns:
        (total, by_stage); by_stage is [] unless n_stages is given.
    """
    total = 0
    by_stage = [0] * (n_stages or 0)
    for d in set(left) | set(right):
        a, b = left.get(d, []), right.get(d, [])
        for j, (x, y) in enumerate(zip(a, b)):
            if x != y:
                total += 1
                if n_stages:
                    by_stage[j] += 1
        total += abs(len(a) - len(b))
    return total, by_stage


@app.function(timeout=1200, **GPU_KW)
def attention_parity(q_heads: int = 32) -> str:
    """Compare unified paged attention with a contiguous reference.

    q_heads=32 is the 4B geometry; q_heads=64 is the 32B geometry.
    """
    import math
    from types import SimpleNamespace

    import torch

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import pack_chunk

    heads = q_heads
    kv_heads = 8
    head_dim = 128
    page_tokens = 16

    weight = SimpleNamespace(shape=(4096, 4096))
    attn = SimpleNamespace(
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
        rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(
        self_attn=attn,
        mlp=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=weight)))
    model = SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=None, norm=None))

    def fragmented_arena(pages_needed):
        n_pages = pages_needed * 2 + 8
        arena = KVArena(
            n_layers=1, n_pages=n_pages, page_tokens=page_tokens,
            n_kv=kv_heads, d_head=head_dim, dtype=torch.bfloat16)
        for i in range(n_pages):
            assert arena.alloc(("filler", i), 1) is not None
        free_keys = list(range(0, n_pages, 2))[:pages_needed]
        for i in free_keys:
            arena.free_key(("filler", i))
        return arena

    def cuda_i32(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    def error(left, right):
        delta = (left.float() - right.float()).abs()
        flat = delta.flatten()
        # kthvalue instead of quantile: quantile rejects tensors over
        # 2^24 elements, which the 64-head many-page case exceeds
        k = max(1, int(0.99 * flat.numel()))
        return dict(
            max_abs=float(delta.max().item()),
            mean_abs=float(delta.mean().item()),
            p99_abs=float(flat.kthvalue(k).values.item()),
            fraction_over_1e_2=float((delta > 1e-2).float().mean().item()),
            fraction_over_5e_2=float((delta > 5e-2).float().mean().item()))

    def pytorch_reference(q_parts, k_parts, v_parts, prefix_lengths):
        outputs = []
        group = heads // kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        for q, k, v, prefix in zip(
                q_parts, k_parts, v_parts, prefix_lengths):
            q_len = q.shape[0]
            k_len = k.shape[0]
            k_gqa = k.repeat_interleave(group, dim=1)
            v_gqa = v.repeat_interleave(group, dim=1)
            scores = torch.einsum(
                "qhd,khd->hqk", q.float(), k_gqa.float()) * scale
            q_pos = prefix + torch.arange(q_len, device="cuda")
            k_pos = torch.arange(k_len, device="cuda")
            mask = k_pos[None, :] <= q_pos[:, None]
            scores.masked_fill_(~mask[None, :, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hqk,khd->qhd", probs, v_gqa.float())
            outputs.append(out.to(torch.bfloat16))
        return torch.cat(outputs)

    def run_case(name, specs, fresh):
        needed = sum(math.ceil((f + s) / page_tokens)
                     for f, s in specs)
        arena = fragmented_arena(needed)
        pipeline = Pipeline(model, arena, kernels="quail",
                            attention_mode="merge_quant")
        groups = []
        q_parts = []
        k_parts = []
        v_parts = []
        prefix_lengths = []
        q_packed = []
        k_packed = []
        v_packed = []

        for i, (prefix_len, suffix_len) in enumerate(specs):
            key = (name, i)
            assert arena.alloc(
                key, prefix_len,
                capacity_tokens=prefix_len + suffix_len) is not None
            if fresh:
                count = prefix_len + suffix_len
                q = torch.randn(
                    count, heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
                k = torch.randn(
                    count, kv_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
                v = torch.randn_like(k)
                groups.append(dict(
                    key=key, prefix=[1] * prefix_len, f=prefix_len,
                    suffixes=[[2] * suffix_len]))
                q_parts.append(q)
                k_parts.append(k)
                v_parts.append(v)
                prefix_lengths.append(0)
                q_packed.append(q)
                k_packed.append(k)
                v_packed.append(v)
            else:
                q = torch.randn(
                    suffix_len, heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
                current_k = torch.randn(
                    suffix_len, kv_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
                current_v = torch.randn_like(current_k)
                cached_k = torch.randn(
                    prefix_len, kv_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
                cached_v = torch.randn_like(cached_k)
                rows = arena.rows_gpu(key)
                arena.k[0].index_copy_(0, rows, cached_k)
                arena.v[0].index_copy_(0, rows, cached_v)
                groups.append(dict(
                    key=key, prefix=None, f=prefix_len,
                    suffixes=[[2] * suffix_len]))
                q_parts.append(q)
                k_parts.append(torch.cat((cached_k, current_k)))
                v_parts.append(torch.cat((cached_v, current_v)))
                prefix_lengths.append(prefix_len)
                q_packed.append(q)
                k_packed.append(current_k)
                v_packed.append(current_v)

        q = torch.cat(q_packed).contiguous()
        k = torch.cat(k_packed).contiguous()
        v = torch.cat(v_packed).contiguous()
        q_flat = q.view(q.shape[0], -1)
        k_flat = k.view(k.shape[0], -1)
        v_flat = v.view(v.shape[0], -1)

        unified_chunk = pack_chunk(
            torch, arena, groups, pinned=True, attention_mode="unified")
        unified_chunk["meta"]["layer"] = 0
        unified = pipeline.attention_unified(
            q_flat, k_flat, v_flat, unified_chunk["meta"])

        cu_q = [0]
        cu_k = [0]
        for qp, kp in zip(q_parts, k_parts):
            cu_q.append(cu_q[-1] + qp.shape[0])
            cu_k.append(cu_k[-1] + kp.shape[0])
        contiguous, _ = pipeline._fa(
            torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts),
            cuda_i32(cu_q), cuda_i32(cu_k),
            max(qp.shape[0] for qp in q_parts),
            max(kp.shape[0] for kp in k_parts), causal=True)
        reference = pytorch_reference(
            q_parts, k_parts, v_parts, prefix_lengths)
        unified = unified.view_as(contiguous)

        physical_pages = [arena.accounting.owned[(name, i)]
                          for i in range(len(specs))]
        return dict(
            name=name, fresh=fresh, specs=specs,
            physical_pages=physical_pages,
            unified_vs_contiguous=error(unified, contiguous),
            unified_vs_pytorch=error(unified, reference),
            contiguous_vs_pytorch=error(contiguous, reference))

    def safe_case(name, specs, fresh):
        """One case, or its error: an edge case that crashes must not
        take the rest of the sweep with it."""
        try:
            out = run_case(name, specs, fresh)
            torch.cuda.synchronize()
            return out
        except Exception as exc:
            return dict(name=name, fresh=fresh, specs=specs,
                        error=repr(exc))

    torch.manual_seed(12345)
    cases = [
        safe_case("fresh_page_edges", [(15, 1), (16, 7), (17, 11)], True),
        safe_case("cached_page_edges", [(15, 1), (16, 7), (17, 11)], False),
        safe_case("cached_long", [(63, 32), (129, 13), (511, 7)], False),
        # Edge cases: documents spanning hundreds of pages, documents
        # shorter than the suffix, one-token documents, a wide chunk
        # of small groups, and (last: may be refused) empty document.
        safe_case("cached_many_pages", [(2049, 37), (4097, 15)], False),
        safe_case("fresh_many_pages", [(2049, 37)], True),
        safe_case("tiny_docs", [(1, 5), (2, 30), (3, 3)], False),
        safe_case("fresh_tiny_docs", [(1, 5), (2, 30), (3, 3)], True),
        safe_case("suffix_dominates", [(7, 300), (5, 111)], False),
        safe_case("many_groups", [(15, 2), (16, 3), (17, 2)] * 16, False),
        safe_case("fresh_empty_doc", [(0, 9)], True),
    ]
    torch.cuda.synchronize()
    report = dict(
        cell="attention_parity", q_heads=heads, kv_heads=kv_heads,
        seed=12345,
        interpretation=(
            "Unified versus contiguous checks the paged mask and row mapping. "
            "Both calls are also compared with an explicit float32 causal "
            "attention reference."),
        cases=cases)
    tag = "" if heads == 32 else f"_{heads}h"
    return _write(report, f"attention_parity{tag}")


@app.function(timeout=2400, **GPU_KW)
def attention_end_to_end_parity(n_docs: int = 256,
                                model: str = "qwen3-4b-fp8") -> str:
    """Compare filter answers with full prompt recomputation."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import build_corpus
    from quail.executor.loop import pack_chunk, run_filter

    (spec, tokenizer, arena, pipeline, answerer, async_answers,
     budget, _) = _boot_executor(model)
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)

    def full_prompt_reference():
        answers = {d: [] for d in range(n_docs)}
        live = list(range(n_docs))
        pipeline.attention_mode = "merge_quant"
        for stage, question in enumerate(question_ids):
            stage_bits = {}
            start = 0
            while start < len(live):
                end = start
                tokens = 0
                while end < len(live):
                    size = len(body_ids[live[end]]) + len(question)
                    if end > start and tokens + size > budget:
                        break
                    tokens += size
                    end += 1
                docs = live[start:end]
                prompts = [body_ids[d] + question for d in docs]
                groups = [dict(
                    key=("reference", stage, d), prefix=prompt,
                    f=len(prompt), suffixes=[])
                    for d, prompt in zip(docs, prompts)]
                packed = pack_chunk(
                    torch, arena, groups, pinned=True,
                    attention_mode="merge_quant")
                final_rows = []
                row = 0
                for prompt in prompts:
                    row += len(prompt)
                    final_rows.append(row - 1)
                packed["final_indices"] = torch.tensor(
                    final_rows, dtype=torch.int64, device="cuda")
                bits = answerer(pipeline.forward_chunk(packed))
                for d, bit in zip(docs, bits):
                    answers[d].append(bit)
                    stage_bits[d] = bit
                start = end
            live = [d for d in live if stage_bits[d]]
        return answers

    outputs = {}
    with torch.inference_mode():
        for mode in ("merge_quant", "unified"):
            pipeline.attention_mode = mode
            outputs[mode], _, _ = run_filter(
                torch, arena, pipeline, async_answers,
                body_ids, question_ids, budget, arena_writes=True)
        outputs["full_prompt"] = full_prompt_reference()
    torch.cuda.synchronize()

    comparisons = {}
    reference = outputs["full_prompt"]
    for mode in ("merge_quant", "unified"):
        count, by_stage = _disagreements(outputs[mode], reference,
                                         len(question_ids))
        comparisons[mode] = dict(
            disagreements=count, disagreements_by_stage=by_stage,
            answered=sum(len(row) for row in outputs[mode].values()),
            wrong=_wrong_count(outputs[mode], flags))
    merge_unified, merge_unified_by_stage = _disagreements(
        outputs["merge_quant"], outputs["unified"], len(question_ids))
    comparisons["merge_quant_vs_unified"] = dict(
        disagreements=merge_unified,
        disagreements_by_stage=merge_unified_by_stage)
    report = dict(
        cell="attention_end_to_end_parity", model=spec.name,
        n_docs=n_docs,
        reference=(
            "Every document and stage recomputes the complete document plus "
            "question as one causal sequence without paging or an LSE merge."),
        comparisons=comparisons,
        pass_unified=comparisons["unified"]["disagreements"] == 0)
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(report, f"attention_end_to_end_parity{tag}")


@app.function(timeout=3600, **GPU_KW)
def join_attention_paths(n_reports: int = 10, n_terms: int = 256,
                         reps: int = 1,
                         model: str = "qwen3-4b-fp8") -> str:
    """Compare both attention paths on the join workload."""
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import biodex_sample
    from quail.executor.loop import run_join

    (spec, tokenizer, arena, pipeline, _, async_answers,
     budget, arena_tokens) = _boot_executor(model)
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes = data["prefixes"]
    suffixes = data["suffixes"][:n_terms]
    modes = ("merge_quant", "unified_packed")
    report = dict(
        cell="join_attention_paths", model=spec.name,
        n_reports=n_reports,
        n_terms=n_terms, pairs=n_reports * n_terms, reps=reps,
        budget=budget, arena_tokens=arena_tokens,
        prediction=(
            "packed unified should be close to merge_quant. It avoids "
            "the merge kernel, but adds private page-table rows and "
            "copies the partial anchor page for each suffix."),
        runs={}, comparisons={})
    print(f"[join_attention_paths] {report['prediction']}", flush=True)

    def run_mode(mode):
        pipeline.attention_mode = (
            "unified" if mode == "unified_packed" else mode)
        answers, spans, tokens = run_join(
            torch, arena, pipeline, async_answers, prefixes,
            [suffixes], budget)
        return answers[0], len(spans), tokens

    outputs = {}
    for mode in modes:
        with torch.inference_mode():
            # Warm the exact measured shape. This covers FlashAttention,
            # DeepGEMM, quantization, scatter, and boundary-copy kernels.
            run_mode(mode)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                answers, n_chunks, tokens = run_mode(mode)
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
                flat = [bit for anchor in range(n_reports)
                        for bit in answers[anchor]]
                row = dict(
                    mode=mode, rep=rep, wall=round(wall, 3),
                    fresh_tokens=tokens,
                    us_per_token=round(wall * 1e6 / tokens, 3),
                    chunks=n_chunks, yes=sum(flat),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2))
                rows.append(row)
                print(f"[join_attention_paths] {row}", flush=True)
        report["runs"][mode] = rows
        outputs[mode] = flat

    baseline = outputs["merge_quant"]
    for mode in modes[1:]:
        report["comparisons"][mode] = dict(
            disagreements=sum(a != b for a, b
                              in zip(baseline, outputs[mode])),
            wall_delta_s=round(
                report["runs"][mode][-1]["wall"]
                - report["runs"]["merge_quant"][-1]["wall"], 3))
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(
        report, f"join_attention_paths_packed{tag}_{n_reports}x{n_terms}")


@app.function(timeout=1800, **GPU_KW)
def packed_unified_join_parity(n_reports: int = 10,
                               n_terms: int = 256,
                               model: str = "qwen3-4b-fp8") -> str:
    """Compare one packed unified join with one-partner batches."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import biodex_sample
    from quail.executor.loop import pack_chunk, run_join

    (spec, tokenizer, arena, pipeline, _, async_answers,
     budget, _) = _boot_executor(model)
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes = data["prefixes"]
    suffixes = data["suffixes"][:n_terms]
    prediction = (
        "The packed unified batch and the one-partner unified batches "
        "will return the same answer for every pair.")
    print(f"[packed_unified_join_parity] {prediction}", flush=True)

    pipeline.attention_mode = "unified"
    with torch.inference_mode():
        packed, _, _ = run_join(
            torch, arena, pipeline, async_answers, prefixes,
            [suffixes], budget)
        packed = packed[0]

        for a, prefix in enumerate(prefixes):
            assert arena.alloc(a, len(prefix)) is not None
        reference = {a: [] for a in range(n_reports)}
        for partner, suffix in enumerate(suffixes):
            groups = [dict(
                key=a,
                prefix=prefixes[a] if partner == 0 else None,
                f=len(prefixes[a]),
                suffixes=[suffix]) for a in range(n_reports)]
            chunk = pack_chunk(
                torch, arena, groups, pinned=True,
                attention_mode="unified")
            normed = pipeline.forward_chunk(chunk)
            bits = async_answers.result(async_answers.submit(normed))
            for a, bit in enumerate(bits):
                reference[a].append(bit)
        for a in range(n_reports):
            arena.free_key(a)

    differences = sum(
        x != y for a in range(n_reports)
        for x, y in zip(packed[a], reference[a]))
    report = dict(
        cell="packed_unified_join_parity", model=spec.name,
        n_reports=n_reports, n_terms=n_terms,
        pairs=n_reports * n_terms, prediction=prediction,
        packed_true=sum(map(sum, packed.values())),
        one_partner_true=sum(map(sum, reference.values())),
        pair_differences=differences,
        passed=differences == 0)
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(
        report, f"packed_unified_join_parity{tag}_{n_reports}x{n_terms}")


@app.function(timeout=5400, **GPU_KW)
def attention_paths(n_docs: int = 10000, reps: int = 2,
                    model: str = "qwen3-4b-fp8") -> str:
    """Compare the two attention paths on the filter workload.

    All paths share one model, one arena, one container, and the current
    Quail kernels. Each path gets an unmeasured warmup before its measured
    repetitions. model picks the spec ("qwen3-4b-fp8" or
    "qwen3-32b-fp8"); results for a non-default model write to a
    suffixed file.
    """
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import build_corpus
    from quail.executor.loop import run_filter, warm_kernels

    (spec, tokenizer, arena, pipeline, _, async_ans,
     exec_budget, arena_tok) = _boot_executor(model)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)

    predictions = {
        "merge_quant": "the two-call reference for this comparison",
        "unified": "0.2 to 0.4 us/token faster than merge_quant",
    }
    report = dict(
        cell="attention_paths", model=spec.name, n_docs=n_docs,
        reps=reps,
        exec_budget=exec_budget, arena_tokens=arena_tok,
        paths={name: description for name, description in ATTENTION_PATHS},
        predictions=predictions, runs={}, comparisons={})
    print(f"[attention_paths] predictions: {json.dumps(predictions)}",
          flush=True)

    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, exec_budget,
                     model_name=spec.hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()

    answers_by_path = {}
    for mode, _ in ATTENTION_PATHS:
        pipeline.attention_mode = mode
        with torch.inference_mode():
            run_filter(torch, arena, pipeline, async_ans,
                       body_ids[:min(256, n_docs)], q_ids, exec_budget,
                       arena_writes=True)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                timers = {}
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                answers, spans, tokens = run_filter(
                    torch, arena, pipeline, async_ans, body_ids,
                    q_ids, exec_budget, timing=timers,
                    arena_writes=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                answered = sum(len(v) for v in answers.values())
                survivors = sum(len(row) == len(q_ids) and all(row)
                                for row in answers.values())
                row = dict(
                    mode=mode, rep=rep, wall=round(wall, 3),
                    fresh_tokens=tokens,
                    us_per_token=round(wall * 1e6 / tokens, 3),
                    answered=answered, survivors=survivors,
                    wrong=_wrong_count(answers, flags),
                    chunks=len(spans),
                    gpu_s=round(sum(e0.elapsed_time(e1)
                                    for _, e0, e1 in spans) / 1e3, 3),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2),
                    cpu_phase_s={k: round(v, 3) for k, v
                                 in sorted(timers.items())})
                rows.append(row)
                print(f"[attention_paths] {row}", flush=True)
        report["runs"][mode] = rows
        answers_by_path[mode] = answers

    baseline = answers_by_path["merge_quant"]
    for mode in ("unified",):
        report["comparisons"][mode] = dict(
            disagreements=_disagreements(
                baseline, answers_by_path[mode])[0],
            wall_delta_s=round(
                report["runs"][mode][-1]["wall"]
                - report["runs"]["merge_quant"][-1]["wall"], 3))
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(report, f"attention_paths{tag}")


# ---------------------------------------------------------- profiling

def _categorize(name):
    """Kernel name -> time bucket. Order matters: our Triton kernel
    names contain substrings ("add_rms", "quant") that also appear in
    vLLM's op names, so the Triton names are checked first."""
    low = name.lower()
    if "deep_gemm" in low or "sm90_fp8" in low or "gemm" in low:
        return "gemm"
    if "flash" in low or "attn" in low:
        return "attention"
    if any(k in low for k in ("silu_mul_quant", "add_rms_norm_quant",
                              "qk_norm_rope")):
        return "triton_fused"
    if any(k in low for k in ("rms_norm", "rotary", "silu_and_mul")):
        return "vllm_elementwise"
    if "quant" in low:
        return "quant"
    if any(k in low for k in ("memcpy", "copy", "index", "cat",
                              "gather", "scatter")):
        return "copies"
    return "other"


@app.function(timeout=3600, **GPU_KW)
def profile_packed(n_docs: int = 3000) -> str:
    """Per-kernel-category GPU time for A2 and A3 in one container."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from corpus import MODEL, build_corpus
    from quail.executor.arena import KVArena
    from quail.executor.attention import (FILTER_ATTENTION,
                                          Pipeline)
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     warm_kernels)
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
    pipeline = Pipeline(model, arena, attention_mode=FILTER_ATTENTION)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)

    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, exec_budget,
                     model_name=MODEL)
    torch.cuda.synchronize()
    kernel_cache.commit()

    result = {"n_docs": n_docs, "exec_budget": exec_budget,
          "stock_reference_us_per_token": dict(
              gemm=5.521, attention=0.813, kv_write_fp8=0.427,
              small_kernels=4.004, total=10.77,
              note="banked engine profile at 25,305-token steps, "
                   "fp8 KV (fusion_ab.json control cell)"),
          "rungs": {}}
    for name, kernels, pinned, _ in PACKED_RUNGS:
        pipeline.kernels = kernels
        with torch.inference_mode():
            # unprofiled reference: the true rate
            _, _, tokens = run_filter(torch, arena, pipeline,
                                      async_ans, body_ids, q_ids,
                                      exec_budget, pinned=pinned,
                                      arena_writes=True)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                run_filter(torch, arena, pipeline, async_ans, body_ids,
                           q_ids, exec_budget, pinned=pinned,
                           arena_writes=True)
                torch.cuda.synchronize()

        cats, counts = {}, {}
        rows = []
        for ev in prof.key_averages():
            cuda_us = getattr(ev, "self_device_time_total", 0) or \
                getattr(ev, "self_cuda_time_total", 0)
            if not cuda_us:
                continue
            # skip op wrappers ("_C::...", "aten::...") and runtime
            # events: they carry the same device time as the kernel
            # they launched and would double-count it
            if (ev.key.startswith(("_C::", "aten::"))
                    or "Command Buffer" in ev.key):
                continue
            cat = _categorize(ev.key)
            cats[cat] = cats.get(cat, 0.0) + cuda_us
            counts[cat] = counts.get(cat, 0) + ev.count
            rows.append((round(cuda_us / 1e6, 3), ev.count,
                         ev.key[:90]))
        rows.sort(reverse=True)
        busy = sum(cats.values())
        result["rungs"][name] = dict(
            fresh_tokens=tokens,
            cuda_busy_s=round(busy / 1e6, 2),
            us_per_token=round(busy / tokens, 2),
            category_s={k: round(v / 1e6, 2) for k, v
                        in sorted(cats.items())},
            category_us_per_token={k: round(v / tokens, 2)
                                   for k, v in sorted(cats.items())},
            category_launches={k: counts[k] for k in sorted(counts)},
            top_kernels=[dict(s=s, n=n, name=k) for s, n, k
                         in rows[:25]])
        prof.export_chrome_trace(
            f"/results/ablations/profile_{name.lower()}.json.gz")
        print(f"[profile_packed] {name}: "
              f"{json.dumps(result['rungs'][name]['category_us_per_token'])}",
              flush=True)
    return _write(result, "profile_packed")


# ---------------------------------------------------------- entrypoint

def _save(payload, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(json.loads(payload), f, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def run_profile(out: str = "results/ablation_profile.json"):
    _save(profile_packed.remote(), out)


@app.function(timeout=900, **GPU_KW)
def probe_attention_api() -> str:
    """Report the cache-attention entry points in the vLLM image."""
    import importlib
    import inspect

    modules = (
        "vllm.vllm_flash_attn",
        "vllm.vllm_flash_attn.flash_attn_interface",
        "vllm.vllm_flash_attn.flash_attn_interface_fa3",
    )
    report = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            report[module_name] = {"error": repr(exc)}
            continue
        names = sorted(name for name in dir(module)
                       if "cache" in name.lower() or "flash_attn" in name)
        entries = {}
        for name in names:
            value = getattr(module, name)
            try:
                signature = str(inspect.signature(value))
            except (TypeError, ValueError):
                signature = None
            entries[name] = signature
        report[module_name] = {
            "file": getattr(module, "__file__", None),
            "entries": entries,
        }
    print(json.dumps(report, indent=2), flush=True)
    return json.dumps(report)


@app.local_entrypoint()
def probe_api():
    print(probe_attention_api.remote())


@app.local_entrypoint()
def run_attention_paths(n_docs: int = 10000, reps: int = 2,
                        model: str = "qwen3-4b-fp8"):
    print(attention_paths.remote(n_docs, reps, model))


@app.local_entrypoint()
def run_attention_parity(q_heads: int = 32):
    result = attention_parity.remote(q_heads)
    print(result)


@app.local_entrypoint()
def run_attention_end_to_end_parity(n_docs: int = 256,
                                    model: str = "qwen3-4b-fp8"):
    result = attention_end_to_end_parity.remote(n_docs, model)
    print(result)


@app.local_entrypoint()
def run_join_attention_paths(n_reports: int = 10, n_terms: int = 256,
                             reps: int = 1,
                             model: str = "qwen3-4b-fp8"):
    handle = join_attention_paths.spawn(n_reports, n_terms, reps, model)
    print(f"join_attention_paths fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_packed_unified_join_parity(n_reports: int = 10,
                                   n_terms: int = 256,
                                   model: str = "qwen3-4b-fp8"):
    handle = packed_unified_join_parity.spawn(
        n_reports, n_terms, model)
    print(f"packed_unified_join_parity fc: {handle.object_id}",
          flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_join_paths_sweep(model: str = "qwen3-4b-fp8"):
    """Run the join ablation at three fan-out shapes."""
    handles = [(n_r, n_t,
                join_attention_paths.spawn(n_r, n_t, reps, model))
               for n_r, n_t, reps in
               ((10, 256, 2), (1, 2560, 1), (100, 256, 1))]
    for n_r, n_t, h in handles:
        print(f"join {n_r}x{n_t} fc: {h.object_id}", flush=True)
    for n_r, n_t, h in handles:
        print(h.get())


@app.local_entrypoint()
def run_all(n_docs: int = 10000, reps: int = 2,
            out: str = "results/ablation_forward.json"):
    a0_h = stock_rung.spawn("A0", "fp8", n_docs, reps)
    a1_h = stock_rung.spawn("A1", "auto", n_docs, reps)
    packed_h = packed_rungs.spawn(n_docs, reps)
    merged = dict(stock={"A0": json.loads(a0_h.get()),
                         "A1": json.loads(a1_h.get())},
                  packed=json.loads(packed_h.get()))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"saved {out}")
