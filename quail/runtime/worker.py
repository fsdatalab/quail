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
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow")
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


def _execute_payload(payload: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import (
        AsyncAnswers,
        warm_kernels,
    )
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
    result_path = f"/results/runs/run_{time.time_ns()}.json"
    report["result_volume_path"] = result_path
    with open(result_path, "w") as f:
        json.dump(dict(
            wall_s=report["wall_s"], boot_s=report["boot_s"],
            boot_kind=report["boot_kind"], boot=boot,
            fresh_tokens=report["fresh_tokens"],
            join_optimizer=report.get("join_optimizer"),
            kv_manager=report.get("kv_manager")), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return report


@app.function(image=image, gpu="H100!", timeout=7200, memory=98304,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute(payload: dict) -> dict:
    return _execute_payload(payload)


def _execute_single(state, payload: dict) -> dict:
    """Single-GPU execution core.

    Args:
        state: Dict with model, arena, pipeline, spec, torch, F.
        payload: The coordinator's payload dict.
    """
    import torch.nn.functional as F  # noqa: F401 (state carries it)

    from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
    from quail.executor.loop import AsyncAnswers, run_filter, run_join
    from quail.planner.joins import search_joins, summarize_alias
    from quail.planner.sol import prefix_recompute_seconds
    from quail.runtime.coordinator import (
        filter_round_limit,
        gate_group,
        retain_aliases,
        runtime_nodes,
        search_specs,
        stage_for_anchor,
        thin_survivors,
    )
    from quail.runtime.tokens import chain_tokens, decode_payload_documents
    from quail.specs import DEVICES

    torch = state["torch"]
    arena, pipeline = state["arena"], state["pipeline"]
    model_spec = state["spec"]
    device = DEVICES["h100-sxm"]
    docs = decode_payload_documents(payload["docs"])
    # the engine preamble: prepended to every KV-owning document
    # (filter scans, join anchors) so their KV is identical across
    # operators; a partner document rides in the suffix raw
    pre = payload.get("pre_ids") or []
    answerer = _PayloadAnswerer(torch, state["F"], state["model"],
                                payload["true_ids"], payload["false_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    chunk_tokens = payload["chunk_tokens"]

    # the arena outlives a query; nothing from the last one may stay
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()

    def retention_value(alias, document):
        return prefix_recompute_seconds(
            len(pre) + len(docs[alias][document]), model_spec, device)

    total_tokens = 0
    out_filters = {}
    survivors = {alias: list(range(len(d))) for alias, d in docs.items()}
    # None when the payload has joins: LIMIT caps output rows, and a
    # join fans one survivor into zero or many rows
    limit = filter_round_limit(payload)
    filter_writes = payload["filter_arena_writes"]
    retain = retain_aliases(payload)
    t0 = time.perf_counter()
    with torch.inference_mode():
        pipeline.attention_mode = FILTER_ATTENTION
        for alias, qids in payload["filters"].items():
            keep = (range(len(docs[alias])) if alias in retain
                    else ())
            answers, _, tokens = run_filter(
                torch, arena, pipeline, async_ans,
                [chain_tokens(pre, d) for d in docs[alias]], qids,
                chunk_tokens, limit=limit,
                arena_writes=filter_writes[alias],
                arena_keys=[(alias, d)
                            for d in range(len(docs[alias]))],
                retain_survivors=keep,
                retention_values={d: retention_value(alias, d)
                                  for d in keep})
            total_tokens += tokens
            out_filters[alias] = {int(d): row
                                  for d, row in answers.items()}
            survivors[alias] = sorted(
                d for d, row in answers.items()
                if len(row) == len(qids) and all(row))

        kv_stats = dict(
            retained_after_filters=len(arena.accounting.retained),
            retained_pages_after_filters=arena.accounting.retained_pages,
            join_anchor_hits=0,
            join_anchor_misses=0)

        out_joins = []
        finished_full = []      # full stage outputs, for barriers
        pipeline.attention_mode = JOIN_ATTENTION
        all_specs = search_specs(payload["joins"])
        remaining = set(range(len(payload["joins"])))
        already_joined = set()
        optimizer_runs = []
        optimizer_sequence = []

        def possible_anchors(indices):
            out = set()
            for index in indices:
                spec = all_specs[index]
                if spec["semantics"] == "full" \
                        and spec.get("anchor_free"):
                    out.update(spec["aliases"])
                else:
                    out.add(spec["anchor"])
            return out

        def next_group():
            specs = [all_specs[i] for i in sorted(remaining)]
            involved = sorted({a for spec in specs
                               for a in spec["aliases"]})
            found = search_joins(
                specs,
                {a: float(len(survivors[a])) for a in involved},
                {a: summarize_alias(
                    (len(docs[a][g]) for g in survivors[a]),
                    resident_flags=(
                        (a, g) in arena.accounting.owned
                        for g in survivors[a]))
                 for a in involved},
                {},
                len(pre), chunk_tokens, model_spec, device,
                fixed_order=payload.get("order_rule") == "as_written",
                arena_tokens=float(arena.accounting.n_pages
                                   * arena.accounting.page_tokens),
                page_tokens=arena.accounting.page_tokens,
                already_joined=already_joined)
            if found is not None:
                optimizer_runs.append(found)
                nodes = runtime_nodes(found["seq"], payload["joins"])
                return next(node for node in nodes
                            if node["op"] == "JoinGroup")

            ordered = sorted(
                remaining,
                key=lambda i: payload["joins"][i].get("written_pos", i))
            first = ordered[0]
            anchor = payload["joins"][first]["anchor"]
            group = [first]
            if payload["joins"][first]["semantics"] == "full":
                for index in ordered[1:]:
                    join = payload["joins"][index]
                    if join["semantics"] != "full" \
                            or join["anchor"] != anchor:
                        break
                    group.append(index)
            return dict(op="JoinGroup", anchor=anchor,
                        stage_idxs=tuple(group))

        for key in [key for key in list(arena.accounting.retained)
                    if key[0] not in possible_anchors(remaining)]:
            arena.free_key(key)

        while remaining:
            node = next_group()
            optimizer_sequence.extend(
                (payload["joins"][index].get("written_pos", index),
                 node["anchor"])
                for index in node["stage_idxs"])
            group_specs = [payload["joins"][i]
                           for i in node["stage_idxs"]]
            anchor_alias = node["anchor"]
            group = [stage_for_anchor(s, anchor_alias)
                     for s in group_specs]
            anchors_glob = list(survivors[anchor_alias])
            alive_now = set(anchors_glob)
            for key in [k for k in list(arena.accounting.retained)
                        if k[0] == anchor_alias
                        and k[1] not in alive_now]:
                arena.free_key(key)
            stage_suffixes = []
            tuple_globs = []
            for j in group:
                tuples = [list(t) for t in itertools.product(
                    *[survivors[p] for p in j["partners"]])]
                tuple_globs.append(tuples)
                stage_suffixes.append(
                    [_tuple_suffix(j, docs, t) for t in tuples])
            prefixes = [chain_tokens(pre, docs[anchor_alias][g])
                        for g in anchors_glob]
            anchor_keys = [(anchor_alias, g) for g in anchors_glob]
            kv_stats["join_anchor_hits"] += sum(
                key in arena.accounting.owned for key in anchor_keys)
            kv_stats["join_anchor_misses"] += sum(
                key not in arena.accounting.owned
                for key in anchor_keys)
            remaining.difference_update(node["stage_idxs"])
            future_anchors = possible_anchors(remaining)

            def anchor_done(a, row):
                key = anchor_keys[a]
                matched = any(row)
                alive = (not matched
                         if group[-1]["semantics"] == "anti"
                         else matched)
                if alive and anchor_alias in future_anchors:
                    arena.retain(key, len(prefixes[a]),
                                 retention_value(anchor_alias,
                                                 anchors_glob[a]))
                else:
                    arena.free_key(key)

            ans, _, tokens = run_join(
                torch, arena, pipeline, async_ans, prefixes,
                stage_suffixes, chunk_tokens,
                stage_frames=[j.get("frame") or [] for j in group],
                anchor_keys=anchor_keys, anchor_done=anchor_done)
            total_tokens += tokens
            for si, j in enumerate(group):
                stage_out = dict(
                    rows={int(a): row for a, row in ans[si].items()},
                    anchor_index=anchors_glob,
                    partner_index=tuple_globs[si],
                    anchor=anchor_alias,
                    partners=list(j["partners"]),
                    semantics=j["semantics"],
                    selectivity=j.get("selectivity"),
                    written_pos=j.get("written_pos"))
                out_joins.append(stage_out)
                if j["semantics"] == "full":
                    finished_full.append(stage_out)
            # gate the anchor set for stages after this group, and
            # settle any anchor run_join never called back (an anchor
            # with no live suffixes never enters a chunk)
            survivors[anchor_alias] = gate_group(
                out_joins[-1], group[-1]["semantics"])
            alive_after = set(survivors[anchor_alias])
            for g, key, prefix in zip(anchors_glob, anchor_keys,
                                      prefixes):
                if key not in arena.accounting.owned:
                    continue
                if g in alive_after and anchor_alias in future_anchors:
                    arena.retain(key, len(prefix),
                                 retention_value(anchor_alias, g))
                else:
                    arena.free_key(key)
            for key in [k for k in list(arena.accounting.retained)
                        if k[0] not in future_anchors]:
                arena.free_key(key)
            before = {a: set(ids) for a, ids in survivors.items()}
            thin_survivors(finished_full, survivors)
            for alias, old_ids in before.items():
                for document in old_ids - set(survivors[alias]):
                    key = (alias, document)
                    if key in arena.accounting.owned:
                        arena.free_key(key)
            for index in node["stage_idxs"]:
                already_joined.update(all_specs[index]["aliases"])
        for key in list(arena.accounting.owned):
            arena.free_key(key)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return dict(filters=out_filters, joins=out_joins,
                wall_s=round(wall, 2),
                fresh_tokens=total_tokens,
                join_optimizer=(None if not optimizer_runs else dict(
                    states=sum(run["states"] for run in optimizer_runs),
                    generated=sum(run["generated"]
                                  for run in optimizer_runs),
                    replans=len(optimizer_runs),
                    sequence=[list(step)
                              for step in optimizer_sequence])),
                kv_manager=dict(
                    **kv_stats,
                    evicted_keys=arena.evicted_keys,
                    evicted_pages=arena.evicted_pages,
                    evicted_value_seconds=arena.evicted_value),
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
    from quail.planner.sol import prefix_recompute_seconds
    from quail.specs import DEVICES
    from quail.runtime.tokens import chain_tokens

    _child_boot(state, sub)
    boot = state["boot"]
    torch = state["torch"]
    arena = state["arena"]
    device = DEVICES["h100-sxm"]
    # a new query begins: nothing kept for the previous one may stay
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    out = dict(filters={}, survivors={}, retained={}, fresh_tokens=0,
               boot_s=boot["boot_s"], boot_kind=boot["kind"],
               boot=boot)
    pre = sub.get("pre_ids") or []
    limit = sub.get("limit")
    filter_writes = sub["filter_arena_writes"]
    retain = set(sub.get("retain_aliases") or ())
    t0 = _time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = FILTER_ATTENTION
        for alias, qids in sub["filters"].items():
            index = sub["doc_index"][alias]
            keep = (range(len(sub["docs"][alias]))
                    if alias in retain else ())
            answers, _, tokens = run_filter(
                torch, arena, state["pipeline"],
                state["async_ans"],
                [chain_tokens(pre, d) for d in sub["docs"][alias]], qids,
                state["chunk_tokens"], limit=limit,
                arena_writes=filter_writes[alias],
                arena_keys=[(alias, g) for g in index],
                retain_survivors=keep,
                retention_values={
                    d: prefix_recompute_seconds(
                        len(pre) + len(sub["docs"][alias][d]),
                        state["spec"], device)
                    for d in keep})
            if alias in retain:
                out["retained"][alias] = sorted(
                    key[1] for key in arena.accounting.retained
                    if key[0] == alias)
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
    from quail.planner.sol import prefix_recompute_seconds
    from quail.specs import DEVICES
    from quail.runtime.tokens import chain_tokens

    _child_boot(state, sub)
    torch = state["torch"]
    arena = state["arena"]
    device = DEVICES["h100-sxm"]
    pre = sub.get("pre_ids") or []
    anchor_alias = sub["anchor_alias"]
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    # kept KV whose consumer groups are behind us, and kept anchors a
    # barrier thinned off this shard, have no reader here
    drop = set(sub.get("drop_kept") or ())
    alive_now = set(anchors_glob)
    for key in list(arena.accounting.retained):
        stale = key[0] == anchor_alias and key[1] not in alive_now
        if key[0] in drop or stale:
            arena.free_key(key)
    # one anchor group per round: the parent runs the join search,
    # walks its nodes, thins at barriers, and re-shards; the child
    # just runs the group
    group = sub["joins"]
    retain_anchor = bool(sub.get("retain_anchor"))
    hits = sum(1 for g in anchors_glob
               if (anchor_alias, g) in arena.accounting.owned)
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
        prefixes = [chain_tokens(pre, d) for d in anchor_docs]
        anchor_keys = [(anchor_alias, g) for g in anchors_glob]

        def value(a):
            return prefix_recompute_seconds(
                len(prefixes[a]), state["spec"], device)

        def anchor_done(a, row):
            matched = any(row)
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(anchor_keys[a], len(prefixes[a]),
                             value(a))
            else:
                arena.free_key(anchor_keys[a])

        ans, _, tokens = run_join(
            torch, arena, state["pipeline"],
            state["async_ans"], prefixes, stage_suffixes,
            state["chunk_tokens"],
            stage_frames=[j.get("frame") or [] for j in group],
            anchor_keys=anchor_keys, anchor_done=anchor_done)
        # settle anchors run_join never called back (no live suffixes)
        last = ans[-1] if ans else {}
        for a, key in enumerate(anchor_keys):
            if key not in arena.accounting.owned:
                continue
            matched = any(last.get(a, []))
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(key, len(prefixes[a]), value(a))
            else:
                arena.free_key(key)
        tokens_total += tokens
        for si, j in enumerate(group):
            out_joins.append(dict(
                rows={int(a): row for a, row in ans[si].items()},
                anchor_index=anchors_glob,
                partner_index=tuple_globs[si]))
    if sub.get("final_group"):
        for key in list(arena.accounting.owned):
            arena.free_key(key)
    torch.cuda.synchronize()
    retained = {}
    for alias, document in arena.accounting.retained:
        retained.setdefault(alias, []).append(document)
    return dict(joins=out_joins, fresh_tokens=tokens_total,
                retained={alias: sorted(documents)
                          for alias, documents in retained.items()},
                kv_round=dict(hits=hits,
                              misses=len(anchors_glob) - hits),
                kv_totals=dict(
                    evicted_keys=arena.evicted_keys,
                    evicted_pages=arena.evicted_pages,
                    evicted_value_seconds=arena.evicted_value),
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

    from quail.planner import budgets
    from quail.planner.joins import search_joins, summarize_alias
    from quail.runtime import coordinator
    from quail.runtime.tokens import decode_payload_documents
    from quail.specs import DEVICES, MODELS

    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
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
    retained = {}
    for out in fouts:
        for alias, ids in (out.get("retained") or {}).items():
            retained.setdefault(alias, set()).update(ids)
    out_joins = []
    finished_full = []      # full stage outputs, for barriers
    pre_len = len(payload.get("pre_ids") or [])
    docs = payload["docs"]
    model_spec = MODELS[payload["model"]]
    device = DEVICES["h100-sxm"]
    all_specs = coordinator.search_specs(payload["joins"])
    remaining = set(range(len(payload["joins"])))
    already_joined = set()
    optimizer_runs = []
    optimizer_sequence = []

    def possible_anchors(indices):
        out = set()
        for index in indices:
            spec = all_specs[index]
            if spec["semantics"] == "full" and spec.get("anchor_free"):
                out.update(spec["aliases"])
            else:
                out.add(spec["anchor"])
        return out

    def next_group():
        specs = [all_specs[i] for i in sorted(remaining)]
        involved = sorted({a for spec in specs
                           for a in spec["aliases"]})
        found = search_joins(
            specs,
            {a: float(len(survivors[a])) for a in involved},
            {a: summarize_alias(
                (len(docs[a][g]) for g in survivors[a]),
                resident_flags=(g in retained.get(a, ())
                                for g in survivors[a]))
             for a in involved},
            {},
            pre_len, payload["chunk_tokens"], model_spec, device,
            fixed_order=payload.get("order_rule") == "as_written",
            arena_tokens=float(budgets.arena_tokens(
                model_spec, device, payload["chunk_tokens"])) * k,
            page_tokens=budgets.PAGE_TOKENS,
            already_joined=already_joined)
        if found is not None:
            optimizer_runs.append(found)
            nodes = coordinator.runtime_nodes(found["seq"],
                                               payload["joins"])
            return next(node for node in nodes
                        if node["op"] == "JoinGroup")

        ordered = sorted(
            remaining,
            key=lambda i: payload["joins"][i].get("written_pos", i))
        first = ordered[0]
        anchor = payload["joins"][first]["anchor"]
        group = [first]
        if payload["joins"][first]["semantics"] == "full":
            for index in ordered[1:]:
                join = payload["joins"][index]
                if join["semantics"] != "full" \
                        or join["anchor"] != anchor:
                    break
                group.append(index)
        return dict(op="JoinGroup", anchor=anchor,
                    stage_idxs=tuple(group))

    prior_shards = {}    # alias -> anchor shards its kept KV sits on
    kv_stats = dict(
        retained_after_filters=sum(len(v) for v in retained.values()),
        join_anchor_hits=0, join_anchor_misses=0)
    child_totals = [None] * k
    while remaining:
        node = next_group()
        optimizer_sequence.extend(
            (payload["joins"][index].get("written_pos", index),
             node["anchor"])
            for index in node["stage_idxs"])
        specs_group = [payload["joins"][i] for i in node["stage_idxs"]]
        anchor = node["anchor"]
        group = [coordinator.stage_for_anchor(s, anchor)
                 for s in specs_group]
        remaining.difference_update(node["stage_idxs"])
        future = possible_anchors(remaining)
        drop = [a for a in retained if a not in future and a != anchor]
        for a in drop:
            retained.pop(a, None)
        jsubs = coordinator.join_group_payloads(
            payload, k, survivors, group, prior_shards=prior_shards)
        for sub in jsubs:
            sub["retain_anchor"] = anchor in future
            sub["drop_kept"] = drop
            sub["final_group"] = not remaining
        jouts = _round("joins", jsubs)
        stage_outs = coordinator.merge_join_round(jouts)
        merged["fresh_tokens"] += sum(o["fresh_tokens"] for o in jouts)
        for i, o in enumerate(jouts):
            kv_stats["join_anchor_hits"] += o["kv_round"]["hits"]
            kv_stats["join_anchor_misses"] += o["kv_round"]["misses"]
            child_totals[i] = o["kv_totals"]
        for stage_out, j in zip(stage_outs, group):
            stage_out["anchor"] = anchor
            stage_out["partners"] = list(j["partners"])
            stage_out["semantics"] = j["semantics"]
            stage_out["selectivity"] = j.get("selectivity")
            stage_out["written_pos"] = j.get("written_pos")
            out_joins.append(stage_out)
            if j["semantics"] == "full":
                finished_full.append(stage_out)
        survivors[anchor] = coordinator.gate_group(
            stage_outs[-1], group[-1]["semantics"])
        retained = {}
        placement = {}
        for worker, out in enumerate(jouts):
            for alias, ids in (out.get("retained") or {}).items():
                retained.setdefault(alias, set()).update(ids)
                placement.setdefault(alias, [[] for _ in range(k)])
                placement[alias][worker] = list(ids)
        prior_shards = placement
        coordinator.thin_survivors(finished_full, survivors)
        for alias in list(retained):
            retained[alias].intersection_update(survivors[alias])
            if not retained[alias]:
                retained.pop(alias)
                prior_shards.pop(alias, None)
            else:
                alive = retained[alias]
                prior_shards[alias] = [
                    [document for document in shard if document in alive]
                    for shard in prior_shards[alias]
                ]
        for index in node["stage_idxs"]:
            already_joined.update(all_specs[index]["aliases"])
    for totals in child_totals:
        for key, v in (totals or {}).items():
            kv_stats[key] = kv_stats.get(key, 0) + v
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
                  join_optimizer=(None if not optimizer_runs else dict(
                      states=sum(run["states"] for run in optimizer_runs),
                      generated=sum(run["generated"]
                                    for run in optimizer_runs),
                      replans=len(optimizer_runs),
                      sequence=[list(step)
                                for step in optimizer_sequence])),
                  kv_manager=kv_stats,
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


def _tuple_suffix(join, docs, member):
    """Build the partner blocks and answer cue for one tuple."""
    from quail.runtime.tokens import chain_tokens

    parts = []
    for alias, g in zip(join["partners"], member):
        parts.extend((join["labels"][alias], docs[alias][g]))
    parts.append(join["tail"])
    return chain_tokens(*parts)


class _PayloadAnswerer:
    """Answerer using TRUE/FALSE token ids from the payload."""

    def __init__(self, torch, F, model, true_ids, false_ids):
        self.F = F
        self.allowed = sorted(set(true_ids) | set(false_ids))
        # the head weight lives on the CPU when untied (moved there at
        # load); slice where it lives, keep only the slice on the GPU
        weight = model.lm_head.weight
        sel = torch.tensor(self.allowed, device=weight.device)
        self.weights = weight.index_select(0, sel).to(
            device="cuda", dtype=torch.bfloat16)
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
