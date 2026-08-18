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


# Process-global state: the container IS the session-side cache. A
# warm container keeps the loaded model, the arena, and the pinned KV
# store across execute() calls, which is what makes a session's later
# queries boot in milliseconds and restore instead of recompute.
_BOOTED = {}      # model name -> dict(model, arena, pipeline, budget)
_STORE = None     # one PinnedStore per container, shared


@app.function(image=image, gpu="H100!", timeout=7200, memory=98304,
              scaledown_window=300,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute(payload: dict) -> dict:
    global _STORE
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.kvstore import PinnedStore
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     run_join, warm_kernels)
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    if payload["kv_dtype"] != "bf16":
        raise NotImplementedError(
            "the fp8 KV arena is not built yet; the planner only "
            "picks fp8 under warm-store pressure at scales the store "
            "cannot hold in bf16")

    spec = MODELS[payload["model"]]
    device = DEVICES["h100-sxm"]
    docs = payload["docs"]

    t_boot = time.perf_counter()
    booted = _BOOTED.get(spec.name)
    if booted is None:
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
        booted = dict(model=model, arena=arena, pipeline=pipeline,
                      warmed=False)
        _BOOTED[spec.name] = booted
    model, arena, pipeline = (booted["model"], booted["arena"],
                              booted["pipeline"])
    chunk = budgets.chunk_budget(spec, device)
    # the worker has no tokenizer: the YES/NO token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, F, model, payload["yes_ids"],
                                payload["no_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    budget = min(chunk, pipeline.max_chunk_tokens,
                 payload["chunk_tokens"])

    if not booted["warmed"]:
        with torch.inference_mode():
            # boot-side warmup: the dense token sweep plus one
            # budget-sized chunk, so every kernel configuration
            # compiles outside measured walls
            first_alias = next(iter(docs))
            warm_q = (next(iter(payload["filters"].values()))[0]
                      if payload["filters"] else [1, 2, 3])
            warm_kernels(torch, arena, pipeline, async_ans,
                         docs[first_alias], [warm_q], budget)
        torch.cuda.synchronize()
        kernel_cache.commit()   # keep the compiles even if the run dies
        booted["warmed"] = True

    store_cfg = payload.get("store")
    if store_cfg is not None and _STORE is None:
        max_doc = max((len(d) for ds in docs.values() for d in ds),
                      default=0)
        _STORE = PinnedStore(
            capacity_tokens=int(store_cfg["capacity_bytes"]
                                // spec.kappa),
            n_layers=spec.layers, n_kv=spec.n_kv, d_head=spec.d_head,
            max_doc_tokens=max(max_doc, 4096), dtype=torch.bfloat16)
    boot_s = time.perf_counter() - t_boot   # load + compile + store

    total_tokens = 0
    out_filters = {}
    store_stats = {}
    survivors = {alias: list(range(len(d))) for alias, d in docs.items()}

    t0 = time.perf_counter()
    with torch.inference_mode():
        for alias, qids in payload["filters"].items():
            stats = {}
            answers, _, tokens = run_filter(
                torch, arena, pipeline, async_ans, docs[alias], qids,
                budget,
                store=_STORE if store_cfg else None,
                store_hash=(store_cfg["hashes"][alias]
                            if store_cfg else None),
                store_min_tokens=(store_cfg["min_doc_tokens"]
                                  if store_cfg else 1),
                stats=stats)
            store_stats[alias] = stats
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
                  store=(store_stats if store_cfg else None),
                  peak_gib=round(
                      torch.cuda.max_memory_allocated() / 2**30, 2))
    os.makedirs("/results/runs", exist_ok=True)
    with open(f"/results/runs/run_{int(time.time())}.json", "w") as f:
        json.dump(dict(wall_s=report["wall_s"], boot_s=report["boot_s"],
                       fresh_tokens=total_tokens), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return report


# ------------------------------------------------- multi-GPU dispatch
#
# One executor per GPU, as its own child process (its own CUDA
# context, arena, and store slice). The parent is the in-container
# coordinator: it splits the payload with quail.runtime.coordinator,
# runs the filter round, merges survivors, runs the join round, and
# merges the answers - no network hop anywhere between rounds.
# Children persist across execute calls, so their models stay loaded
# and their store slices stay warm for the whole session.

_CHILDREN = []      # [(process, connection)] in GPU order


def _child_main(gpu_idx, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    state = {}
    while True:
        try:
            kind, data = conn.recv()
        except EOFError:
            break
        if kind == "shutdown":
            break
        try:
            if kind == "filters":
                conn.send(("ok", _child_filters(state, data)))
            elif kind == "joins":
                conn.send(("ok", _child_joins(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.kvstore import PinnedStore
    from quail.executor.loop import AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[sub["model"]]
    device = DEVICES["h100-sxm"]
    if "pipeline" not in state:
        model = load_model(spec.hf_name)
        chunk = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        state.update(torch=torch, F=F, model=model, arena=arena,
                     pipeline=Pipeline(model, arena), spec=spec,
                     chunk=chunk, warmed=False, store=None)
    answerer = _PayloadAnswerer(torch, F, state["model"],
                                sub["yes_ids"], sub["no_ids"])
    state["async_ans"] = AsyncAnswers(torch, answerer)
    state["budget"] = min(state["chunk"],
                          state["pipeline"].max_chunk_tokens,
                          sub["chunk_tokens"])
    if not state["warmed"]:
        docs = next((d for d in sub.get("docs", {}).values() if d),
                    None)
        if docs is None:
            docs = sub.get("anchor_docs") or [[1, 2, 3]]
        warm_q = (next(iter(sub["filters"].values()))[0]
                  if sub.get("filters") else [1, 2, 3])
        with torch.inference_mode():
            warm_kernels(torch, state["arena"], state["pipeline"],
                         state["async_ans"], docs, [warm_q],
                         state["budget"])
        torch.cuda.synchronize()
        kernel_cache.commit()
        state["warmed"] = True
    store_cfg = sub.get("store")
    if store_cfg is not None and state.get("store") is None:
        max_doc = max((len(d) for ds in sub.get("docs", {}).values()
                       for d in ds), default=0)
        state["store"] = PinnedStore(
            capacity_tokens=int(store_cfg["capacity_bytes"]
                                // state["spec"].kappa
                                // sub.get("workers", 1)),
            n_layers=state["spec"].layers, n_kv=state["spec"].n_kv,
            d_head=state["spec"].d_head,
            max_doc_tokens=max(max_doc, 4096), dtype=torch.bfloat16)


def _child_filters(state, sub):
    import time as _time

    from quail.executor.loop import run_filter

    t0 = _time.perf_counter()
    _child_boot(state, sub)
    boot_s = _time.perf_counter() - t0
    torch = state["torch"]
    store_cfg = sub.get("store")
    out = dict(filters={}, survivors={}, fresh_tokens=0, store={},
               boot_s=round(boot_s, 2))
    t0 = _time.perf_counter()
    with torch.inference_mode():
        for alias, qids in sub["filters"].items():
            stats = {}
            index = sub["doc_index"][alias]
            answers, _, tokens = run_filter(
                torch, state["arena"], state["pipeline"],
                state["async_ans"], sub["docs"][alias], qids,
                state["budget"],
                store=state["store"] if store_cfg else None,
                store_hash=(store_cfg["hashes"][alias]
                            if store_cfg else None),
                store_min_tokens=(store_cfg["min_doc_tokens"]
                                  if store_cfg else 1),
                stats=stats, store_ids=index)
            out["store"][alias] = stats
            out["fresh_tokens"] += tokens
            out["filters"][alias] = {index[d]: row
                                     for d, row in answers.items()}
            out["survivors"][alias] = sorted(
                index[d] for d, row in answers.items()
                if len(row) == len(qids) and all(row))
        # an alias with no filters survives whole
        for alias, index in sub["doc_index"].items():
            out["survivors"].setdefault(alias, sorted(index))
    torch.cuda.synchronize()
    out["wall_s"] = round(_time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    import time as _time

    from quail.executor.loop import run_join

    _child_boot(state, sub)
    torch = state["torch"]
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    live = list(range(len(anchors_glob)))     # local anchor positions
    out_joins, tokens_total = [], 0
    t0 = _time.perf_counter()
    with torch.inference_mode():
        for group in _stage_groups(sub["joins"]):
            stage_suffixes, partner_globs = [], []
            for j in group:
                partner = sub["partners"][j["partner"]]
                partner_globs.append(list(partner["index"]))
                stage_suffixes.append(
                    [j["mid"] + doc + j["tail"]
                     for doc in partner["docs"]])
            prefixes = [group[0]["pre"] + anchor_docs[a] for a in live]
            ans, _, tokens = run_join(
                torch, state["arena"], state["pipeline"],
                state["async_ans"], prefixes, stage_suffixes,
                state["budget"],
                group_size=1 if len(group) > 1 else None)
            tokens_total += tokens
            for si, j in enumerate(group):
                out_joins.append(dict(
                    rows={int(a): row for a, row in ans[si].items()},
                    anchor_index=[anchors_glob[live[a]]
                                  for a in range(len(live))],
                    partner_index=partner_globs[si]))
            last = ans[len(group) - 1]
            kept = {a for a, row in last.items() if any(row)}
            if group[-1]["semantics"] == "anti":
                live = [live[a] for a in range(len(live))
                        if a not in kept]
            else:
                live = [live[a] for a in sorted(kept)]
    torch.cuda.synchronize()
    return dict(joins=out_joins, fresh_tokens=tokens_total,
                wall_s=round(_time.perf_counter() - t0, 2))


def _ensure_children(k):
    import multiprocessing as mp
    if len(_CHILDREN) >= k:
        return
    ctx = mp.get_context("spawn")
    for gpu in range(len(_CHILDREN), k):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_child_main, args=(gpu, child_conn),
                           daemon=True)
        proc.start()
        _CHILDREN.append((proc, parent_conn))


def _round(kind, subs):
    """Send one round to the children and collect, failing loudly
    with the child's traceback."""
    for (_, conn), sub in zip(_CHILDREN, subs):
        conn.send((kind, sub))
    outs = []
    for (_, conn), _sub in zip(_CHILDREN, subs):
        status, data = conn.recv()
        if status != "ok":
            raise RuntimeError(f"GPU child failed:\n{data}")
        outs.append(data)
    return outs


def _execute_multi(payload: dict) -> dict:
    import time as _time

    from quail.runtime import coordinator

    k = payload["workers"]
    shards = payload.get("shards", {})
    _ensure_children(k)
    t0 = _time.perf_counter()
    fouts = _round("filters",
                   coordinator.filter_round_payloads(payload, shards, k))
    merged = coordinator.merge_filter_round(fouts)
    out_joins = []
    if payload["joins"]:
        jsubs = coordinator.join_round_payloads(
            payload, shards, k, merged["survivors"])
        jouts = _round("joins", jsubs)
        out_joins = coordinator.merge_join_round(jouts)
        merged["fresh_tokens"] += sum(o["fresh_tokens"] for o in jouts)
    wall = _time.perf_counter() - t0
    boot_s = max(o["boot_s"] for o in fouts)
    report = dict(filters=merged["filters"], joins=out_joins,
                  wall_s=round(wall - boot_s, 2),
                  boot_s=round(boot_s, 2),
                  fresh_tokens=merged["fresh_tokens"],
                  store=merged["store"] or None,
                  peak_gib=max(o["peak_gib"] for o in fouts))
    results_vol.commit()
    kernel_cache.commit()
    return report


@app.function(image=image, gpu="H100!:2", timeout=7200, memory=131072,
              scaledown_window=300,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_2(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:4", timeout=7200, memory=196608,
              scaledown_window=300,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_4(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:8", timeout=7200, memory=262144,
              scaledown_window=300,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_8(payload: dict) -> dict:
    return _execute_multi(payload)


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
