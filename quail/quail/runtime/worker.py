"""The Modal worker: one container, one H100, one executor.

Takes the coordinator's payload (token ids and the planned settings,
nothing else), runs the filter chains and join stages on the packed
executor, gates between stages next to the GPU, and returns the raw
answer rows. The coordinator assembles tuples, replay-checks, and
projects - it never sees a tensor.

Consecutive full-join stages sharing one anchor run as a single
multi-stage run_join call, so the anchor's KV is computed in stage 1
and read again in stage 2 - the cross-stage reuse the executor's kept
KV exists for. exists/anti stages run alone; their early-stop
optimization is not built yet, so they stream the full partner list
and the keep rule is applied to the answers.
"""

import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # JIT artifacts persist on the kernel-cache volume so each
          # DeepGEMM/Triton configuration compiles once ever, not
          # once per container
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail")
)

# House rule: never create new Modal app names - caches and warm state
# ride on the app. The engine worker lives here, permanently.
app = modal.App("quail-engine")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


@app.function(image=image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute(payload: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     run_join, warm_kernels)
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    if payload["kv_dtype"] != "bf16":
        raise NotImplementedError(
            "the fp8 KV arena is not built yet; the planner only "
            "picks fp8 under warm-store pressure, which needs the "
            "store (a later step)")

    spec = MODELS[payload["model"]]
    device = DEVICES["h100-sxm"]

    t_boot = time.perf_counter()
    from quail.executor.model import load_model
    model = load_model(spec.hf_name)
    chunk = budgets.chunk_budget(spec, device)
    arena_tok = budgets.arena_tokens(spec, device, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena)
    # the worker has no tokenizer: the YES/NO token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, F, model, payload["yes_ids"],
                                payload["no_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    budget = min(chunk, pipeline.max_chunk_tokens,
                 payload["chunk_tokens"])

    docs = payload["docs"]
    total_tokens = 0
    out_filters = {}
    survivors = {alias: list(range(len(d))) for alias, d in docs.items()}

    with torch.inference_mode():
        # boot-side warmup: one budget-sized chunk compiles the
        # kernels' full-size configurations outside the measured wall
        # (a tiny warmup chunk compiles the wrong DeepGEMM configs and
        # leaves ~50 s of JIT inside the first query)
        first_alias = next(iter(docs))
        warm_q = (next(iter(payload["filters"].values()))[0]
                  if payload["filters"] else [1, 2, 3])
        warm_kernels(torch, arena, pipeline, async_ans,
                     docs[first_alias], [warm_q], budget)
    torch.cuda.synchronize()
    kernel_cache.commit()    # keep the compiles even if the run dies
    boot_s = time.perf_counter() - t_boot   # load + compile, together

    t0 = time.perf_counter()
    with torch.inference_mode():
        for alias, qids in payload["filters"].items():
            answers, _, tokens = run_filter(
                torch, arena, pipeline, async_ans, docs[alias], qids,
                budget)
            total_tokens += tokens
            out_filters[alias] = {int(d): row
                                  for d, row in answers.items()}
            survivors[alias] = sorted(
                d for d, row in answers.items()
                if len(row) == len(qids) and all(row))

        out_joins = []
        for group in _stage_groups(payload["joins"]):
            anchor_alias = group[0]["anchor"]
            anchors_glob = list(survivors[anchor_alias])
            stage_suffixes = []
            partner_globs = []
            for j in group:
                partners = list(survivors[j["partner"]])
                partner_globs.append(partners)
                stage_suffixes.append(
                    [j["mid"] + docs[j["partner"]][p] + j["tail"]
                     for p in partners])
            prefixes = [group[0]["pre"] + docs[anchor_alias][a]
                        for a in anchors_glob]
            ans, _, tokens = run_join(
                torch, arena, pipeline, async_ans, prefixes,
                stage_suffixes, budget,
                group_size=1 if len(group) > 1 else None)
            total_tokens += tokens
            for si, j in enumerate(group):
                out_joins.append(dict(
                    rows={int(a): row for a, row in ans[si].items()},
                    anchor_index=anchors_glob,
                    partner_index=partner_globs[si]))
            # gate the anchor set for stages after this group
            last = ans[len(group) - 1]
            kept = {anchors_glob[a] for a, row in last.items()
                    if any(row)}
            if group[-1]["semantics"] == "anti":
                survivors[anchor_alias] = [
                    a for a in anchors_glob if a not in kept]
            else:
                survivors[anchor_alias] = sorted(kept)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    report = dict(filters=out_filters, joins=out_joins,
                  wall_s=round(wall, 2), boot_s=round(boot_s, 2),
                  fresh_tokens=total_tokens,
                  peak_gib=round(
                      torch.cuda.max_memory_allocated() / 2**30, 2))
    os.makedirs("/results/runs", exist_ok=True)
    with open(f"/results/runs/run_{int(time.time())}.json", "w") as f:
        json.dump(dict(wall_s=report["wall_s"], boot_s=report["boot_s"],
                       fresh_tokens=total_tokens), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return report


def _stage_groups(joins):
    """Consecutive full stages sharing an anchor run as one gated
    multi-stage call; everything else runs alone."""
    groups, current = [], []
    for j in joins:
        if (current and j["semantics"] == "full"
                and current[-1]["semantics"] == "full"
                and current[0]["anchor"] == j["anchor"]
                and current[0]["pre"] == j["pre"]):
            current.append(j)
        else:
            if current:
                groups.append(current)
            current = [j]
    if current:
        groups.append(current)
    return groups


class _PayloadAnswerer:
    """The Answerer, built from YES/NO token ids shipped in the
    payload instead of a tokenizer."""

    def __init__(self, torch, F, model, yes_ids, no_ids):
        self.F = F
        self.allowed = sorted(set(yes_ids) | set(no_ids))
        sel = torch.tensor(self.allowed, device="cuda")
        self.weights = model.lm_head.weight.index_select(0, sel).to(
            torch.bfloat16)
        self.yes_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in set(yes_ids)],
            device="cuda")
        self.no_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in set(no_ids)],
            device="cuda")

    def __call__(self, normed):
        scores = self.F.linear(normed, self.weights)
        yes = scores.index_select(1, self.yes_cols).amax(dim=1)
        no = scores.index_select(1, self.no_cols).amax(dim=1)
        return (yes > no).int().cpu().tolist()
