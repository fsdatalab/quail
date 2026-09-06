"""Quail model execution inside a compute worker.

One GPU runs in the worker process. Several GPUs run one child process
per GPU, each with its own CUDA context and arena, coordinated by the
worker process over pipes.
"""

import gc
import itertools
import json
import os
import time

from quail.backends.base import GpuContext
from quail.backends.quail.coordinator import stage_for_anchor
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.graph import (
    _join_round_kv,
    _tuple_suffix,
    execute_single_graph,
)
from quail.execution import PhysicalResponse
from quail.executor.arena import KVArena
from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION, Pipeline
from quail.executor.loop import AsyncAnswers, warm_kernels
from quail.executor.model import load_model
from quail.physical import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    decode_graph,
    DocumentInput,
    PackedFilter,
)
from quail.planner import budgets
from quail.runtime.runner import ExecutionContext
from quail.runtime.tokens import (
    chain_tokens,
    decode_payload_documents,
    DocumentPrefixes,
)


# GPU child processes, one per H100, kept alive across queries so their
# models stay loaded for the whole session.
_CHILDREN: list = []


def quail_runtime_payload(request, graph) -> dict:
    """Build private Quail scheduler state from a standard request."""
    envelope = request.plan
    docs = {}
    for node in graph.nodes:
        if not isinstance(node, DocumentInput):
            continue
        docs[node.alias] = request.inputs[node.input_id].documents
    return {
        "physical_plan": envelope,
        "model": envelope["model"],
        "workers": envelope["workers"],
        "docs": docs,
        **dict(envelope["settings"]),
    }


def execute_quail_request(context):
    """Run one Quail request from a backend execution context."""
    payload = quail_runtime_payload(context.request, context.graph)
    if context.gpu_count == 1:
        backend = context.registry.backend(context.request.plan["backend"])
        return execute_quail_payload(
            payload, context.registry, context.graph, backend,
            context.runtime_state,
        )
    return execute_quail_multi(payload, context.registry, context.graph)


def _release_vllm_parallel_state() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError:
        return

    destroy_model_parallel()
    destroy_distributed_environment()


def release_booted_models(runtime_state: dict) -> dict:
    """Release Quail GPU state before another engine uses this process."""
    import torch

    released = len(runtime_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for state in runtime_state.values():
        if not isinstance(state, dict):
            continue
        close = state.get("close")
        if callable(close):
            close()
    runtime_state.clear()
    _release_vllm_parallel_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {
        "models_released": released,
        "vllm_parallel_state_released": True,
        "cuda_allocated_bytes": (
            int(torch.cuda.memory_allocated())
            if torch.cuda.is_available() else 0),
        "cuda_reserved_bytes": (
            int(torch.cuda.memory_reserved())
            if torch.cuda.is_available() else 0),
    }


def _boot_gpu(state, backend, spec, device, gpu_index, workers,
              chunk_tokens):
    """Load the model, arena, and pipeline once per GPU executor."""
    import torch
    import torch.nn.functional as F


    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    if "model_execution" not in state:
        t0 = time.perf_counter()
        model = load_model(spec.hf_name, revision=spec.revision)
        boot["load_model_s"] = time.perf_counter() - t0
        # budgets.* is tiny CPU; fold into arena_s so the four phases
        # cover the cold-load span without a leftover residual
        t0 = time.perf_counter()
        budget = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, budget)
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
        execution = backend.start(GpuContext(
            gpu_index=gpu_index,
            gpu_count=workers,
            model=spec,
            device=device,
            query_settings={"chunk_tokens": chunk_tokens},
        ))
        execution.bind_loaded_model(
            model=model, arena=arena, pipeline=pipeline
        )
        state.update(torch=torch, F=F, model_execution=execution,
                     model=model, arena=arena, pipeline=pipeline,
                     spec=spec, warmed=False)
        boot["kind"] = "cold"
    return boot


def _bind_query(state, true_ids, false_ids, chunk_tokens):
    """Attach the answerer and chunk budget for one query."""

    torch = state["torch"]
    # the worker has no tokenizer: the TRUE/FALSE token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, state["F"], state["model"],
                                true_ids, false_ids)
    state["async_ans"] = AsyncAnswers(torch, answerer)
    state["chunk_tokens"] = chunk_tokens
    state["model_execution"].bind_query(
        torch=torch,
        async_answers=state["async_ans"],
        chunk_tokens=chunk_tokens,
    )


def _warm(state, boot):
    """Compile and touch the kernels once per container."""
    from quail.runtime.volumes import kernel_cache

    if state["warmed"]:
        return
    torch = state["torch"]
    t0 = time.perf_counter()
    with torch.inference_mode():
        # boot-side warmup on synthetic tokens: the compile pass once
        # ever per stack+model+budget (marker on the kernel cache
        # volume), the millisecond-loads touch pass on every container
        # after that
        warm = warm_kernels(torch, state["arena"], state["pipeline"],
                            state["async_ans"], state["chunk_tokens"],
                            model_name=state["spec"].hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()   # keep the compiles even if the run dies
    boot["warm_kernels_s"] = time.perf_counter() - t0
    boot["warm_tier"] = warm["tier"]
    state["warmed"] = True
    boot["kind"] = "cold"


def _finish_boot(boot, t_boot):
    for k in ("load_model_s", "arena_s", "pipeline_s", "warm_kernels_s"):
        boot[k] = round(boot[k], 2)
    boot["boot_s"] = round(time.perf_counter() - t_boot, 2)


def execute_quail_payload(payload, registry, graph, backend, runtime_state):
    """Execute one Quail payload on the worker's own GPU."""
    from quail.runtime.volumes import kernel_cache, results_vol

    spec = registry.model(payload["model"])
    device = registry.device(payload["physical_plan"]["device"])

    t_boot = time.perf_counter()
    boot_key = (backend.name, spec.name)
    state = runtime_state.setdefault(boot_key, {})
    boot = _boot_gpu(state, backend, spec, device, 0, payload["workers"],
                     payload["chunk_tokens"])
    _bind_query(state, payload["true_ids"], payload["false_ids"],
                payload["chunk_tokens"])
    _warm(state, boot)
    _finish_boot(boot, t_boot)

    report = execute_single(state, payload, registry, graph)
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
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
            regret_tokens=report.get("regret_tokens"),
            node_metrics=report.get("node_metrics"),
            join_optimizer=report.get("join_optimizer"),
            kv_manager=report.get("kv_manager")), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return PhysicalResponse(outputs, report)


def execute_single(state, payload: dict, registry, graph) -> dict:
    """Execute the typed Quail graph on one GPU."""
    runtime_state = {
        **state,
        "docs": decode_payload_documents(payload["docs"]),
        "runtimes": registry.runtimes,
        "model_spec": state["spec"],
        "device": registry.device(payload["physical_plan"]["device"]),
        "chunk_tokens": payload["chunk_tokens"],
    }
    return execute_single_graph(runtime_state, payload, graph)


# ------------------------------------------------- multi-GPU dispatch
#
# One executor per GPU, as its own child process. The parent is the
# in-container coordinator: it splits the payload with
# quail.backends.quail.coordinator, runs the filter round, merges
# survivors, runs the join round, and merges the answers, with no
# network hop between rounds.

def _child_main(gpu_idx, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    state = {"gpu_index": gpu_idx}
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
    # a spawned child rebuilds its registry from the plan's manifest;
    # quail.builtins imports this module's backend, so it stays here
    from quail.builtins import registry_from_manifest

    envelope = sub["physical_plan"]
    registry = registry_from_manifest(envelope["extensions"])
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    t_boot = time.perf_counter()
    boot = _boot_gpu(state, backend, spec, device, state["gpu_index"],
                     sub["workers"], sub["chunk_tokens"])
    _bind_query(state, sub["true_ids"], sub["false_ids"],
                sub["chunk_tokens"])
    state["registry"] = registry
    state["runtime_context"] = ExecutionContext(
        runtimes=registry.runtimes,
        model_execution=state["model_execution"],
    )
    _warm(state, boot)
    _finish_boot(boot, t_boot)
    state["boot"] = boot


def _child_filters(state, sub):

    _child_boot(state, sub)
    boot = state["boot"]
    torch = state["torch"]
    arena = state["arena"]
    if sub.get("start_query", True):
        _reset_child_query(state)
    out = dict(filters={}, survivors={}, retained={}, fresh_tokens=0,
               boot_s=boot["boot_s"], boot_kind=boot["kind"],
               boot=boot)
    pre = sub.get("pre_ids") or []
    filter_limit = sub.get("filter_limit")
    t0 = time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = FILTER_ATTENTION
        node_id = sub.get("node_id")
        if node_id is not None:
            graph = decode_graph(
                sub["physical_plan"]["graph"],
                state["registry"].codecs,
            )
            node = graph.node(node_id)
            if not isinstance(node, PackedFilter):
                raise TypeError(
                    f"child filter received {node.type_name!r}")
            alias = node.alias
            index = sub["doc_index"][alias]
            runtime_context = state["runtime_context"]
            result = runtime_context.runtimes[node.runtime_key].execute(
                node,
                {
                    "documents": DocumentPrefixes(
                        pre,
                        sub["docs"][alias],
                        range(len(sub["docs"][alias])),
                    ),
                    "document_ids": index,
                    "limit": filter_limit,
                    "retain_survivors": node.keep_kv,
                },
                runtime_context,
            )
            answers = result.outputs[f"filter_answers:{alias}"]
            if node.keep_kv:
                out["retained"][alias] = sorted(
                    key[1] for key in arena.accounting.retained
                    if key[0] == alias)
            out["fresh_tokens"] += result.metrics.fresh_tokens
            # answers is keyed by document position; every answered
            # document's prefix was computed once in this query
            state["seen"].update((alias, document)
                                 for document in answers)
            out["filters"][alias] = {
                int(document): row for document, row in answers.items()
            }
            out["survivors"][alias] = list(
                result.outputs[f"ids:{alias}"]
            )
    torch.cuda.synchronize()
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):

    _child_boot(state, sub)
    torch = state["torch"]
    arena = state["arena"]
    if sub.get("start_query", False):
        _reset_child_query(state)
    pre = sub.get("pre_ids") or []
    registry = state["registry"]
    encoded_node = sub["physical_node"]
    node = registry.codecs[encoded_node["type"]].decode(encoded_node)
    if not isinstance(node, AnchoredJoin):
        raise TypeError(f"child join received {node.type_name!r}")
    graph = decode_graph(
        sub["physical_plan"]["graph"], registry.codecs
    )
    adaptive = next(
        physical_node for physical_node in graph.nodes
        if isinstance(physical_node, AdaptiveJoinPlan)
    )
    group = [
        stage_for_anchor(adaptive.join_specs[index], node.anchor)
        for index in node.stage_idxs
    ]
    anchor_alias = node.anchor
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
    retain_anchor = bool(sub.get("retain_anchor"))
    seen = state.setdefault("seen", set())
    out_joins, tokens_total = [], 0
    t0 = time.perf_counter()
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
        round_kv = _join_round_kv(
            anchor_keys, [len(p) for p in prefixes],
            arena.accounting.owned, seen)

        def anchor_done(a, row):
            matched = any(row)
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(anchor_keys[a], len(prefixes[a]))
            else:
                arena.free_key(anchor_keys[a])

        runtime_context = state["runtime_context"]
        result = runtime_context.runtimes[node.runtime_key].execute(
            node,
            {
                "prefixes": prefixes,
                "stage_suffixes": stage_suffixes,
                "stage_frames": [
                    join.get("frame") or [] for join in group
                ],
                "anchor_keys": anchor_keys,
                "anchor_done": anchor_done,
                "anchor_ids": anchors_glob,
                "partner_indices": {
                    stage.written_pos: tuples
                    for stage, tuples in zip(node.stages, tuple_globs)
                },
                "group": group,
            },
            runtime_context,
        )
        ans = result.metrics.extension["answers"]
        seen.update(anchor_keys)
        # settle anchors run_join never called back (no live suffixes)
        last = ans[-1] if ans else {}
        for a, key in enumerate(anchor_keys):
            if key not in arena.accounting.owned:
                continue
            matched = any(last.get(a, []))
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(key, len(prefixes[a]))
            else:
                arena.free_key(key)
        tokens_total += result.metrics.fresh_tokens
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
                kv_round=round_kv,
                kv_totals=dict(
                    evicted_keys=arena.evicted_keys,
                    evicted_pages=arena.evicted_pages,
                    evicted_prefix_tokens=arena.evicted_prefix_tokens),
                wall_s=round(time.perf_counter() - t0, 2))


def _reset_child_query(state):
    """Clear KV and counters before a child starts a new query."""
    arena = state["arena"]
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    state["seen"] = set()


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


def execute_quail_multi(payload, registry, graph):
    """Execute the typed Quail graph across GPU child processes."""
    from quail.runtime.volumes import kernel_cache, results_vol

    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
    gpu_count = payload["workers"]
    _ensure_children(gpu_count)
    report = execute_distributed_graph(
        payload,
        graph,
        gpu_count,
        _round,
        registry.model(payload["model"]),
        registry.device(payload["physical_plan"]["device"]),
        registry.runtimes,
        registry,
    )
    results_vol.commit()
    kernel_cache.commit()
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    return PhysicalResponse(outputs, report)


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
