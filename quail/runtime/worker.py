"""Modal worker: runs filter chains and join stages on the GPU,
returns raw answer rows.
"""

import itertools
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
# warm container keeps the loaded model and the arena across
# execute() calls, which is what makes a session's later queries boot
# in milliseconds.
_BOOTED = {}      # model name -> dict(model, arena, pipeline)


@app.function(image=image, gpu="H100!", timeout=7200, memory=98304,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute(payload: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     run_join, warm_kernels)
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    if payload["kv_dtype"] != "bf16":
        raise ValueError(
            f"KV is always bf16; got {payload['kv_dtype']!r}")

    spec = MODELS[payload["model"]]
    device = DEVICES["h100-sxm"]

    t_boot = time.perf_counter()
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    booted = _BOOTED.get(spec.name)
    if booted is None:
        from quail.executor.model import load_model
        t0 = time.perf_counter()
        model = load_model(spec.hf_name)
        boot["load_model_s"] = time.perf_counter() - t0
        # budgets.* is tiny CPU; fold into arena_s so the four phases
        # cover the cold-load span without a leftover residual
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(model, arena,
                            attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        booted = dict(model=model, arena=arena, pipeline=pipeline,
                      warmed=False)
        _BOOTED[spec.name] = booted
        boot["kind"] = "cold"
    model, arena, pipeline = (booted["model"], booted["arena"],
                              booted["pipeline"])
    # the worker has no tokenizer: the TRUE/FALSE token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, F, model, payload["true_ids"],
                                payload["false_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    chunk_tokens = payload["chunk_tokens"]

    if not booted["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            # boot-side warmup on synthetic tokens: the compile pass
            # once ever per stack+model+budget (marker on the kernel
            # cache volume), the millisecond-loads touch pass on
            # every container after that
            warm = warm_kernels(torch, arena, pipeline, async_ans,
                                chunk_tokens,
                                model_name=spec.hf_name)
        torch.cuda.synchronize()
        kernel_cache.commit()   # keep the compiles even if the run dies
        boot["warm_kernels_s"] = time.perf_counter() - t0
        boot["warm_tier"] = warm["tier"]
        booted["warmed"] = True
        boot["kind"] = "cold"

    boot_s = time.perf_counter() - t_boot   # load + compile
    for k in ("load_model_s", "arena_s", "pipeline_s",
              "warm_kernels_s"):
        boot[k] = round(boot[k], 2)
    boot["boot_s"] = round(boot_s, 2)
    state = dict(model=model, arena=arena, pipeline=pipeline,
                 spec=spec, torch=torch, F=F)
    report = _execute_single(state, payload)
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    os.makedirs("/results/runs", exist_ok=True)
    with open(f"/results/runs/run_{int(time.time())}.json", "w") as f:
        json.dump(dict(wall_s=report["wall_s"], boot_s=report["boot_s"],
                       boot_kind=report["boot_kind"], boot=boot,
                       fresh_tokens=report["fresh_tokens"]), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return report


def _execute_single(state, payload: dict) -> dict:
    """Single-GPU execution core.

    Args:
        state: Dict with model, arena, pipeline, spec, torch, F.
        payload: The coordinator's payload dict.
    """
    import torch.nn.functional as F  # noqa: F401 (state carries it)

    from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
    from quail.executor.loop import AsyncAnswers, run_filter, run_join
    from quail.planner.decide import pick_runtime_anchor
    from quail.runtime.coordinator import (derive_plan_nodes,
                                           filter_keep_map,
                                           filter_round_limit,
                                           gate_group, stage_for_anchor,
                                           thin_survivors)

    torch = state["torch"]
    arena, pipeline = state["arena"], state["pipeline"]
    docs = payload["docs"]
    # the engine preamble: prepended to every KV-owning document
    # (filter scans, join anchors) so their KV is identical across
    # operators; a partner document rides in the suffix raw
    pre = payload.get("pre_ids") or []
    answerer = _PayloadAnswerer(torch, state["F"], state["model"],
                                payload["true_ids"], payload["false_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    chunk_tokens = payload["chunk_tokens"]

    total_tokens = 0
    out_filters = {}
    survivors = {alias: list(range(len(d))) for alias, d in docs.items()}
    # None when the payload has joins: LIMIT caps output rows, and a
    # join fans one survivor into zero or many rows
    limit = filter_round_limit(payload)

    filter_writes = payload["filter_arena_writes"]
    filter_keep = filter_keep_map(payload)
    kept = {}    # alias -> {global doc id: prefix tokens held}
    sweep_kept_keys(arena)    # a dead earlier query may have left KV
    t0 = time.perf_counter()
    with torch.inference_mode():
        pipeline.attention_mode = FILTER_ATTENTION
        for alias, qids in payload["filters"].items():
            fkeep = filter_keep.get(alias) or {}
            keep = bool(fkeep.get("keep"))
            kept_ids = []
            answers, _, tokens = run_filter(
                torch, arena, pipeline, async_ans,
                [pre + d for d in docs[alias]], qids,
                chunk_tokens, limit=limit,
                keys=[("kv", alias, g)
                      for g in range(len(docs[alias]))],
                keep=keep,
                keep_extra_tokens=fkeep.get("frame_tokens", 0),
                kept_out=kept_ids,
                arena_writes=filter_writes[alias])
            if keep:
                kept[alias] = {g: len(pre) + len(docs[alias][g])
                               for g in kept_ids}
            total_tokens += tokens
            out_filters[alias] = {int(d): row
                                  for d, row in answers.items()}
            survivors[alias] = sorted(
                d for d, row in answers.items()
                if len(row) == len(qids) and all(row))

        out_joins = []
        finished_full = []      # full stage outputs, for barriers
        pipeline.attention_mode = JOIN_ATTENTION
        nodes = payload.get("plan_nodes") or derive_plan_nodes(
            payload["joins"])
        join_nodes_ahead = [n for n in nodes if n["op"] == "JoinGroup"]
        for node in nodes:
            if node["op"] == "Barrier":
                # barrier thinning; on one GPU there is no shard step
                thin_survivors(finished_full, survivors)
                continue
            if node["op"] != "JoinGroup":
                continue
            join_nodes_ahead = join_nodes_ahead[1:]
            specs = [payload["joins"][i] for i in node["stage_idxs"]]
            anchor_alias = node["anchor"]
            if len(specs) == 1 and specs[0].get("anchor_free"):
                # a one-stage group re-picks its anchor from the
                # measured live counts, crediting KV already resident
                resident = {
                    a: float(sum(kept.get(a, {}).get(g, 0)
                                 for g in survivors[a]))
                    for a in specs[0]["aliases"]}
                anchor_alias = pick_runtime_anchor(
                    specs[0],
                    {a: [len(docs[a][g]) for g in survivors[a]]
                     for a in specs[0]["aliases"]},
                    len(pre), chunk_tokens,
                    resident_tokens=resident)
            group = [stage_for_anchor(s, anchor_alias) for s in specs]
            anchors_glob = list(survivors[anchor_alias])
            # kept KV of documents a barrier already thinned away has
            # no reader left
            free_kept(arena, kept, anchor_alias,
                      drop=set(kept.get(anchor_alias, ()))
                      - set(anchors_glob))
            stage_suffixes = []
            tuple_globs = []
            for j in group:
                tuples = [list(t) for t in itertools.product(
                    *[survivors[p] for p in j["partners"]])]
                tuple_globs.append(tuples)
                stage_suffixes.append(
                    [_tuple_suffix(j, docs, t) for t in tuples])
            prefixes = [pre + docs[anchor_alias][a]
                        for a in anchors_glob]
            keep_sem = (group[-1]["semantics"]
                        if node.get("keep_anchor_kv")
                        and anchor_alias == node["anchor"] else None)
            ans, _, tokens = run_join(
                torch, arena, pipeline, async_ans, prefixes,
                stage_suffixes, chunk_tokens,
                stage_frames=[j.get("frame") or [] for j in group],
                group_size=1 if len(group) > 1 else None,
                anchor_keys=[("kv", anchor_alias, g)
                             for g in anchors_glob],
                keep_semantics=keep_sem,
                evict=make_evict(arena, kept, anchor_alias))
            if keep_sem:
                kept[anchor_alias] = {
                    g: len(prefixes[i])
                    for i, g in enumerate(anchors_glob)
                    if ("kv", anchor_alias, g) in arena.accounting.owned}
            else:
                kept.pop(anchor_alias, None)
            # kept KV whose consumer groups are all behind us is done
            ahead = {n2["anchor"] for n2 in join_nodes_ahead}
            for alias in [a for a in kept if a not in ahead]:
                free_kept(arena, kept, alias)
            total_tokens += tokens
            for si, j in enumerate(group):
                stage_out = dict(
                    rows={int(a): row for a, row in ans[si].items()},
                    anchor_index=anchors_glob,
                    partner_index=tuple_globs[si],
                    anchor=anchor_alias,
                    partners=list(j["partners"]))
                out_joins.append(stage_out)
                if j["semantics"] == "full":
                    finished_full.append(stage_out)
            # gate the anchor set for stages after this group
            survivors[anchor_alias] = gate_group(
                out_joins[-1], group[-1]["semantics"])
        for alias in list(kept):
            free_kept(arena, kept, alias)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return dict(filters=out_filters, joins=out_joins,
                wall_s=round(wall, 2),
                fresh_tokens=total_tokens,
                peak_gib=round(
                    torch.cuda.max_memory_allocated() / 2**30, 2))



# ------------------------------------------------- multi-GPU dispatch
#
# One executor per GPU, as its own child process (its own CUDA
# context and arena). The parent is the in-container coordinator: it
# splits the payload with quail.runtime.coordinator, runs the filter
# round, merges survivors, runs the join round, and merges the
# answers - no network hop anywhere between rounds. Children persist
# across execute calls, so their models stay loaded for the whole
# session.

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
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[sub["model"]]
    device = DEVICES["h100-sxm"]
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    t_boot = time.perf_counter()
    if "pipeline" not in state:
        t0 = time.perf_counter()
        model = load_model(spec.hf_name)
        boot["load_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(model, arena,
                            attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        state.update(torch=torch, F=F, model=model, arena=arena,
                     pipeline=pipeline, spec=spec, warmed=False)
        boot["kind"] = "cold"
    answerer = _PayloadAnswerer(torch, F, state["model"],
                                sub["true_ids"], sub["false_ids"])
    state["async_ans"] = AsyncAnswers(torch, answerer)
    state["chunk_tokens"] = sub["chunk_tokens"]
    if not state["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            warm = warm_kernels(torch, state["arena"],
                                state["pipeline"],
                                state["async_ans"],
                                state["chunk_tokens"],
                                model_name=state["spec"].hf_name)
        torch.cuda.synchronize()
        kernel_cache.commit()
        boot["warm_kernels_s"] = time.perf_counter() - t0
        boot["warm_tier"] = warm["tier"]
        state["warmed"] = True
        boot["kind"] = "cold"
    for k in ("load_model_s", "arena_s", "pipeline_s",
              "warm_kernels_s"):
        boot[k] = round(boot[k], 2)
    boot["boot_s"] = round(time.perf_counter() - t_boot, 2)
    state["boot"] = boot


def _child_filters(state, sub):
    import time as _time

    from quail.executor.attention import FILTER_ATTENTION
    from quail.executor.loop import run_filter

    _child_boot(state, sub)
    boot = state["boot"]
    torch = state["torch"]
    # a new query begins: nothing kept for the previous one may stay
    sweep_kept_keys(state["arena"])
    state["kept"] = {}
    out = dict(filters={}, survivors={}, kept={}, fresh_tokens=0,
               boot_s=boot["boot_s"], boot_kind=boot["kind"],
               boot=boot)
    pre = sub.get("pre_ids") or []
    limit = sub.get("limit")
    filter_writes = sub["filter_arena_writes"]
    filter_keep = sub.get("filter_keep") or {}
    t0 = _time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = FILTER_ATTENTION
        for alias, qids in sub["filters"].items():
            index = sub["doc_index"][alias]
            fkeep = filter_keep.get(alias) or {}
            kept_ids = []
            answers, _, tokens = run_filter(
                torch, state["arena"], state["pipeline"],
                state["async_ans"],
                [pre + d for d in sub["docs"][alias]], qids,
                state["chunk_tokens"], limit=limit,
                keys=[("kv", alias, g) for g in index],
                keep=bool(fkeep.get("keep")),
                keep_extra_tokens=fkeep.get("frame_tokens", 0),
                kept_out=kept_ids,
                arena_writes=filter_writes[alias])
            if fkeep.get("keep"):
                state["kept"][alias] = {
                    index[d]: len(pre) + len(sub["docs"][alias][d])
                    for d in kept_ids}
                out["kept"][alias] = sorted(state["kept"][alias])
            out["fresh_tokens"] += tokens
            out["filters"][alias] = {index[d]: row
                                     for d, row in answers.items()}
            out["survivors"][alias] = sorted(
                index[d] for d, row in answers.items()
                if len(row) == len(qids) and all(row))
    torch.cuda.synchronize()
    out["wall_s"] = round(_time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    import time as _time

    from quail.executor.attention import JOIN_ATTENTION
    from quail.executor.loop import run_join

    _child_boot(state, sub)
    torch = state["torch"]
    pre = sub.get("pre_ids") or []
    anchor_alias = sub["anchor_alias"]
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    kept = state.setdefault("kept", {})
    # kept KV whose consumer groups are behind us, and kept anchors a
    # barrier thinned away, have no reader left
    for alias in sub.get("drop_kept") or ():
        free_kept(state["arena"], kept, alias)
    free_kept(state["arena"], kept, anchor_alias,
              drop=set(kept.get(anchor_alias, ()))
              - set(anchors_glob))
    # one anchor group per round: the parent walks the plan's nodes,
    # thins at barriers, and re-shards; the child just runs the group
    group = sub["joins"]
    keep_sem = sub.get("keep_semantics")
    out_joins, tokens_total = [], 0
    t0 = _time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = JOIN_ATTENTION
        stage_suffixes, tuple_globs = [], []
        for j in group:
            # partner docs ride keyed by local position; tuples
            # combine one local position per partner alias
            locals_ = [range(len(sub["partners"][p]["index"]))
                       for p in j["partners"]]
            combos = list(itertools.product(*locals_))
            tuple_globs.append(
                [[sub["partners"][p]["index"][c]
                  for p, c in zip(j["partners"], combo)]
                 for combo in combos])
            part_docs = {p: sub["partners"][p]["docs"]
                         for p in j["partners"]}
            stage_suffixes.append(
                [_tuple_suffix(j, part_docs, combo)
                 for combo in combos])
        prefixes = [pre + d for d in anchor_docs]
        ans, _, tokens = run_join(
            torch, state["arena"], state["pipeline"],
            state["async_ans"], prefixes, stage_suffixes,
            state["chunk_tokens"],
            stage_frames=[j.get("frame") or [] for j in group],
            group_size=1 if len(group) > 1 else None,
            anchor_keys=[("kv", anchor_alias, g)
                         for g in anchors_glob],
            keep_semantics=keep_sem,
            evict=make_evict(state["arena"], kept, anchor_alias))
        if keep_sem:
            kept[anchor_alias] = {
                g: len(prefixes[i])
                for i, g in enumerate(anchors_glob)
                if ("kv", anchor_alias, g)
                in state["arena"].accounting.owned}
        else:
            kept.pop(anchor_alias, None)
        tokens_total += tokens
        for si, j in enumerate(group):
            out_joins.append(dict(
                rows={int(a): row for a, row in ans[si].items()},
                anchor_index=anchors_glob,
                partner_index=tuple_globs[si]))
    if sub.get("final_group"):
        for alias in list(kept):
            free_kept(state["arena"], kept, alias)
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
    """Send one round to children and collect results."""
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

    from quail.planner.decide import pick_runtime_anchor
    from quail.runtime import coordinator

    k = payload["workers"]
    shards = payload.get("shards", {})
    _ensure_children(k)
    # None when the payload has joins: LIMIT caps output rows, and a
    # join fans one survivor into zero or many rows
    limit = coordinator.filter_round_limit(payload)
    t0 = _time.perf_counter()
    fouts = _round("filters",
                   coordinator.filter_round_payloads(payload, shards, k))
    merged = coordinator.merge_filter_round(fouts, limit=limit)
    survivors = {a: list(v) for a, v in merged["survivors"].items()}
    for alias, ds in payload["docs"].items():
        survivors.setdefault(alias, list(range(len(ds))))
    # global ids of documents whose KV the children hold
    kept = {}
    for out in fouts:
        for alias, ids in (out.get("kept") or {}).items():
            kept.setdefault(alias, set()).update(ids)
    out_joins = []
    finished_full = []      # full stage outputs, for barriers
    pre_len = len(payload.get("pre_ids") or [])
    docs = payload["docs"]
    nodes = payload.get("plan_nodes") or coordinator.derive_plan_nodes(
        payload["joins"])
    join_nodes_ahead = [n for n in nodes if n["op"] == "JoinGroup"]
    prior_shards = {}    # alias -> anchor shards its kept KV sits on
    for node in nodes:
        if node["op"] == "Barrier":
            # barrier thinning; the re-shard itself happens when the
            # next group's payloads are built over the thinned sets
            coordinator.thin_survivors(finished_full, survivors)
            continue
        if node["op"] != "JoinGroup":
            continue
        join_nodes_ahead = join_nodes_ahead[1:]
        specs = [payload["joins"][i] for i in node["stage_idxs"]]
        anchor = node["anchor"]
        if len(specs) == 1 and specs[0].get("anchor_free"):
            # a one-stage group re-picks its anchor from the measured
            # live counts, crediting KV the children already hold
            resident = {
                a: float(sum(pre_len + len(docs[a][g])
                             for g in survivors[a]
                             if g in kept.get(a, ())))
                for a in specs[0]["aliases"]}
            anchor = pick_runtime_anchor(
                specs[0],
                {a: [len(docs[a][g]) for g in survivors[a]]
                 for a in specs[0]["aliases"]},
                pre_len, payload["chunk_tokens"],
                resident_tokens=resident)
        group = [coordinator.stage_for_anchor(s, anchor)
                 for s in specs]
        keep_sem = (group[-1]["semantics"]
                    if node.get("keep_anchor_kv")
                    and anchor == node["anchor"] else None)
        ahead = {n2["anchor"] for n2 in join_nodes_ahead}
        drop = [a for a in kept if a not in ahead and a != anchor]
        jsubs = coordinator.join_group_payloads(
            payload, k, survivors, group, prior_shards=prior_shards)
        for sub in jsubs:
            sub["keep_semantics"] = keep_sem
            sub["drop_kept"] = drop
            sub["final_group"] = not join_nodes_ahead
        for a in drop:
            kept.pop(a, None)
        jouts = _round("joins", jsubs)
        stage_outs = coordinator.merge_join_round(jouts)
        merged["fresh_tokens"] += sum(o["fresh_tokens"] for o in jouts)
        for stage_out, j in zip(stage_outs, group):
            stage_out["anchor"] = anchor
            stage_out["partners"] = list(j["partners"])
            out_joins.append(stage_out)
            if j["semantics"] == "full":
                finished_full.append(stage_out)
        survivors[anchor] = coordinator.gate_group(
            stage_outs[-1], group[-1]["semantics"])
        if keep_sem and join_nodes_ahead:
            kept[anchor] = set(survivors[anchor])
            prior_shards[anchor] = [list(sub["anchor_index"])
                                    for sub in jsubs]
        else:
            kept.pop(anchor, None)
        if not join_nodes_ahead:
            kept.clear()
    wall = _time.perf_counter() - t0
    boot_s = max(o["boot_s"] for o in fouts)
    # forward the breakdown from the child that owned the max boot
    slowest = max(fouts, key=lambda o: o["boot_s"])
    report = dict(filters=merged["filters"], joins=out_joins,
                  wall_s=round(wall - boot_s, 2),
                  boot_s=round(boot_s, 2),
                  boot_kind=slowest.get("boot_kind"),
                  boot=slowest.get("boot"),
                  fresh_tokens=merged["fresh_tokens"],
                  peak_gib=max(o["peak_gib"] for o in fouts))
    results_vol.commit()
    kernel_cache.commit()
    return report


@app.function(image=image, gpu="H100!:2", timeout=7200, memory=131072,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_2(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:4", timeout=7200, memory=196608,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_4(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:8", timeout=7200, memory=262144,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_8(payload: dict) -> dict:
    return _execute_multi(payload)


def _tuple_suffix(join, docs, member) -> list:
    """Build one tuple's suffix: partner documents with block labels, then the answer cue."""
    out = []
    for alias, g in zip(join["partners"], member):
        out += join["labels"][alias]
        out += docs[alias][g]
    out += join["tail"]
    return out


# ------------------------------------------------- kept KV management
#
# Document KV kept across operators lives in the arena under
# ("kv", alias, global doc id) keys. The filter round keeps survivor
# KV for aliases the plan will anchor; a join group keeps its
# surviving anchors when a later group re-uses the table. Everything
# kept is freed the moment its last reader is behind us, and swept
# defensively at the start of the next query - the arena outlives a
# query, kept KV must not.

def _pages(arena):
    """The page accounting: KVArena nests it, PageArena is it."""
    return getattr(arena, "accounting", arena)


def sweep_kept_keys(arena):
    """Free any ("kv", alias, doc) keys an earlier query left resident."""
    for key in [k for k in list(_pages(arena).owned)
                if isinstance(k, tuple) and len(k) == 3
                and k[0] == "kv"]:
        arena.free_key(key)


def free_kept(arena, kept, alias, drop=None):
    """Free one alias's kept KV, or just the given global doc ids."""
    held = kept.get(alias)
    if not held:
        kept.pop(alias, None)
        return
    ids = list(held) if drop is None else [g for g in drop
                                           if g in held]
    for g in ids:
        key = ("kv", alias, g)
        if key in _pages(arena).owned:
            arena.free_key(key)
        held.pop(g, None)
    if not held:
        kept.pop(alias, None)


def make_evict(arena, kept, current_alias):
    """Eviction under arena pressure, for run_join's anchor allocs.

    Victims accumulate smallest first - freed pages are linear in
    document length while the recompute a later group then pays grows
    faster - then any victim the later, larger ones made redundant is
    dropped again, so no document is evicted for pages the allocation
    does not need."""
    def evict(pages_needed):
        candidates = sorted(
            (tokens, alias, g)
            for alias, held in kept.items() if alias != current_alias
            for g, tokens in held.items()
            if ("kv", alias, g) in _pages(arena).owned)
        victims, total = [], 0
        for tokens, alias, g in candidates:
            pages = len(_pages(arena).owned[("kv", alias, g)])
            victims.append((pages, alias, g))
            total += pages
            if total >= pages_needed:
                break
        for entry in list(victims):
            if total - entry[0] >= pages_needed:
                victims.remove(entry)
                total -= entry[0]
        for _, alias, g in victims:
            arena.free_key(("kv", alias, g))
            kept[alias].pop(g, None)
            if not kept[alias]:
                kept.pop(alias)
        return bool(victims)
    return evict


class _PayloadAnswerer:
    """Answerer using TRUE/FALSE token ids from the payload."""

    def __init__(self, torch, F, model, true_ids, false_ids):
        self.F = F
        self.allowed = sorted(set(true_ids) | set(false_ids))
        sel = torch.tensor(self.allowed, device="cuda")
        self.weights = model.lm_head.weight.index_select(0, sel).to(
            torch.bfloat16)
        self.true_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed)
             if t in set(true_ids)], device="cuda")
        self.false_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed)
             if t in set(false_ids)], device="cuda")

    def __call__(self, normed):
        scores = self.F.linear(normed, self.weights)
        t = scores.index_select(1, self.true_cols).amax(dim=1)
        f = scores.index_select(1, self.false_cols).amax(dim=1)
        return (t > f).int().cpu().tolist()
