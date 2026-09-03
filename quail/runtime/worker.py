"""Run physical requests on Modal GPU workers."""

import gc
import itertools
import json
import os
import time
from dataclasses import dataclass, field

import modal

def build_worker_image(
    local_python_sources=(),
    pip_packages=(),
    runtime_package="vllm==0.26.0",
):
    """Build the Modal image containing Quail and registered extensions."""
    packages = tuple(dict.fromkeys((
        "sqlglot>=27.0",
        "bpe-qwen>=0.1.5",
        "datasets>=5.0.1",
        *pip_packages,
    )))
    sources = tuple(dict.fromkeys(("quail", *local_python_sources)))
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .pip_install(
            runtime_package,
            "huggingface_hub",
            "numpy",
            "pyarrow",
        )
        .pip_install(*packages)
        .env({"VLLM_LOGGING_LEVEL": "WARNING",
              "VLLM_USE_FLASHINFER_SAMPLER": "0",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
              "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
              "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
        .add_local_python_source(*sources)
    )


hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


@dataclass
class _WorkerRuntime:
    booted: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


_RUNTIME = _WorkerRuntime()


def _validate_physical_request(request):
    """Validate and decode a physical execution request."""
    from quail.execution import PhysicalRequest
    from quail.extensions import registry_from_modules
    from quail.physical import DocumentInput, check_plan_envelope, decode_graph

    if not isinstance(request, PhysicalRequest):
        raise TypeError("the worker needs a PhysicalRequest")
    envelope = request.plan
    check_plan_envelope(envelope)
    registry = registry_from_modules(tuple(envelope["extension_modules"]))
    backend = registry.backend(envelope["backend"])
    graph = decode_graph(envelope["graph"], registry.codecs)
    graph.validate(runtime_keys=set(registry.runtimes))
    graph.validate_backend(envelope["backend"])
    needed_inputs = {
        node.input_id for node in graph.nodes
        if isinstance(node, DocumentInput)
    }
    missing = needed_inputs - set(request.inputs)
    extra = set(request.inputs) - needed_inputs
    if missing or extra:
        raise ValueError(
            "execution request has wrong input bindings; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return request, registry, graph, backend


def _quail_runtime_payload(request, graph) -> dict:
    """Build private Quail scheduler state from a standard request."""
    from quail.physical import DocumentInput
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


def release_booted_models() -> dict:
    """Release Quail GPU state before another engine uses this process."""
    import torch

    released = len(_RUNTIME.booted)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for state in _RUNTIME.booted.values():
        if not isinstance(state, dict):
            continue
        close = state.get("close")
        if callable(close):
            close()
    _RUNTIME.booted.clear()
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


def _execute_physical(request):
    """Run the backend selected by a registered physical plan."""
    from quail.backends import BackendExecutionContext
    from quail.execution import PhysicalResponse

    request, registry, graph, backend = _validate_physical_request(request)
    gpu_count = request.gpu_count
    response = backend.execute_request(BackendExecutionContext(
        request=request,
        graph=graph,
        registry=registry,
        gpu_count=gpu_count,
        graph_executor=lambda: (
            _execute_quail_payload(
                _quail_runtime_payload(request, graph),
                registry,
                graph,
                backend,
            )
            if gpu_count == 1 else
            _execute_quail_multi(
                _quail_runtime_payload(request, graph),
                registry,
                graph,
            )
        ),
        runtime_state=_RUNTIME.booted,
    ))
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a model backend must return PhysicalResponse")
    if (
        response.metrics.get("result_volume_path") is None
        and os.path.isdir("/results")
        and os.access("/results", os.W_OK)
    ):
        metrics = dict(response.metrics)
        os.makedirs("/results/runs", exist_ok=True)
        result_path = f"/results/runs/run_{time.time_ns()}.json"
        metrics["result_volume_path"] = result_path
        with open(result_path, "w") as output:
            json.dump({
                key: metrics.get(key)
                for key in (
                    "backend",
                    "wall_s",
                    "boot_s",
                    "boot_kind",
                    "boot",
                    "fresh_tokens",
                    "cached_tokens",
                    "regret_tokens",
                    "peak_gib",
                    "node_metrics",
                    "backend_metrics",
                )
            }, output)
        results_vol.commit()
        response = PhysicalResponse(response.outputs, metrics)
    return response


def execute_worker_query(query, physical_executor=None):
    """Execute one query inside its current worker process."""
    from quail.execution import PhysicalResponse
    from quail.planner.plan import Refusal
    from quail.runtime.session import RefusalError

    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    plan.graph.validate(runtime_keys=set(query.session.registry.runtimes))
    plan.graph.validate_backend(plan.backend)
    if plan.workers > 8:
        raise NotImplementedError(
            "more than 8 GPUs means multiple containers; the "
            "multi-container coordinator is a later step"
        )
    request = query._prepare_physical()
    started = time.perf_counter()
    response = (physical_executor or _execute_physical)(request)
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a physical executor must return PhysicalResponse")
    return query.finish(response, time.perf_counter() - started)


def _execute_logical_query(value, gpu_count: int):
    """Read query sources, plan the query, and execute it."""
    from quail.catalog import DocumentProvider
    from quail.extensions import registry_from_modules
    from quail.planner.plan import EngineConfig
    from quail.runtime.session import Query, Session

    config_value = value["config"]
    if not isinstance(config_value, EngineConfig):
        raise TypeError("a worker query needs an EngineConfig")
    requested_gpus = int(config_value.gpus)
    if requested_gpus != gpu_count:
        raise ValueError(
            f"query needs {requested_gpus} GPUs but worker has {gpu_count}"
        )
    started = time.perf_counter()
    registry = registry_from_modules(tuple(value["extension_modules"]))
    providers = {}
    for name, source in value["sources"].items():
        if "remote" in source:
            provider = registry.open_source(source["remote"])
        elif "table" in source:
            provider = DocumentProvider.from_table(
                source["table"], id_col=str(source["id_col"])
            )
        else:
            raise ValueError(f"query source {name!r} has no location")
        providers[name] = provider
    session = Session(
        config_value,
        device=str(value["device"]),
        registry=registry,
    )
    for name, provider in providers.items():
        session.register(name, provider)
    query = Query(session, value["logical_plan"], order=value["order"])
    result = execute_worker_query(query)
    result.report["worker_total_s"] = round(
        time.perf_counter() - started, 4
    )
    return result


def _execute_quail_payload(payload, registry, graph, backend) -> dict:
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import (
        AsyncAnswers,
        warm_kernels,
    )
    from quail.planner import budgets
    spec = registry.model(payload["model"])
    device = registry.device(payload["physical_plan"]["device"])

    t_boot = time.perf_counter()
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    boot_key = (backend.name, spec.name)
    booted = _RUNTIME.booted.get(boot_key)
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
        from quail.backends import GpuContext
        execution = backend.start(GpuContext(
            gpu_index=0,
            gpu_count=payload["workers"],
            model=spec,
            device=device,
            query_settings={
                "chunk_tokens": payload["chunk_tokens"],
            },
        ))
        execution.bind_loaded_model(
            model=model, arena=arena, pipeline=pipeline
        )
        booted = dict(execution=execution, warmed=False)
        _RUNTIME.booted[boot_key] = booted
        boot["kind"] = "cold"
    execution = booted["execution"]
    model = execution.state["model"]
    arena = execution.state["arena"]
    pipeline = execution.state["pipeline"]
    # the worker has no tokenizer: the TRUE/FALSE token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, F, model, payload["true_ids"],
                                payload["false_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    chunk_tokens = payload["chunk_tokens"]
    execution.bind_query(
        torch=torch,
        async_answers=async_ans,
        chunk_tokens=chunk_tokens,
    )

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
    state = dict(model_execution=execution,
                 model=model, arena=arena, pipeline=pipeline,
                 spec=spec, torch=torch, F=F)
    report = _execute_single(state, payload, registry, graph)
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
    from quail.execution import PhysicalResponse

    return PhysicalResponse(outputs, report)


def _execute_single(state, payload: dict, registry, graph) -> dict:
    """Execute the typed Quail graph on one GPU."""
    from quail.runtime.quail_graph import execute_single_graph
    from quail.runtime.tokens import decode_payload_documents
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
# One executor per GPU, as its own child process (its own CUDA
# context and arena). The parent is the in-container coordinator: it
# splits the payload with quail.runtime.coordinator, runs the filter
# round, merges survivors, runs the join round, and merges the
# answers - no network hop anywhere between rounds. Children persist
# across execute calls, so their models stay loaded for the whole
# session.

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
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.extensions import registry_from_modules
    from quail.planner import budgets
    envelope = sub["physical_plan"]
    registry = registry_from_modules(tuple(envelope["extension_modules"]))
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    t_boot = time.perf_counter()
    if "model_execution" not in state:
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
        from quail.backends import GpuContext
        execution = backend.start(GpuContext(
            gpu_index=state["gpu_index"],
            gpu_count=sub["workers"],
            model=spec,
            device=device,
            query_settings={
                "chunk_tokens": sub["chunk_tokens"],
            },
        ))
        execution.bind_loaded_model(
            model=model, arena=arena, pipeline=pipeline
        )
        state.update(torch=torch, F=F, model_execution=execution,
                     model=model, arena=arena, pipeline=pipeline,
                     spec=spec, warmed=False)
        boot["kind"] = "cold"
    answerer = _PayloadAnswerer(torch, F, state["model"],
                                sub["true_ids"], sub["false_ids"])
    state["async_ans"] = AsyncAnswers(torch, answerer)
    state["chunk_tokens"] = sub["chunk_tokens"]
    state["model_execution"].bind_query(
        torch=torch,
        async_answers=state["async_ans"],
        chunk_tokens=state["chunk_tokens"],
    )
    from quail.runtime.runner import ExecutionContext
    state["registry"] = registry
    state["runtime_context"] = ExecutionContext(
        runtimes=registry.runtimes,
        model_execution=state["model_execution"],
    )
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
    from quail.physical import PackedFilter, decode_graph
    from quail.runtime.tokens import DocumentPrefixes

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
    t0 = _time.perf_counter()
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
    out["wall_s"] = round(_time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    import time as _time

    from quail.executor.attention import JOIN_ATTENTION
    from quail.physical import AdaptiveJoinPlan, AnchoredJoin, decode_graph
    from quail.runtime.coordinator import stage_for_anchor
    from quail.runtime.quail_graph import _join_round_kv, _tuple_suffix
    from quail.runtime.tokens import chain_tokens

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
                wall_s=round(_time.perf_counter() - t0, 2))


def _reset_child_query(state):
    """Clear KV and counters before a child starts a new query."""
    arena = state["arena"]
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    state["seen"] = set()


def _ensure_children(k):
    import multiprocessing as mp
    if len(_RUNTIME.children) >= k:
        return
    ctx = mp.get_context("spawn")
    for gpu in range(len(_RUNTIME.children), k):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_child_main, args=(gpu, child_conn),
                           daemon=True)
        proc.start()
        _RUNTIME.children.append((proc, parent_conn))


def _round(kind, subs):
    """Send one round to children and collect results."""
    for (_, conn), sub in zip(_RUNTIME.children, subs):
        conn.send((kind, sub))
    outs = []
    for (_, conn), _sub in zip(_RUNTIME.children, subs):
        status, data = conn.recv()
        if status != "ok":
            raise RuntimeError(f"GPU child failed:\n{data}")
        outs.append(data)
    return outs


def _execute_quail_multi(payload, registry, graph) -> dict:
    """Execute the typed Quail graph across GPU child processes."""
    from quail.runtime.quail_distributed import execute_distributed_graph
    from quail.runtime.tokens import decode_payload_documents

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
    from quail.execution import PhysicalResponse

    return PhysicalResponse(outputs, report)


@dataclass(frozen=True)
class ModalWorker:
    """Modal app and query functions for one extension set."""

    app: object
    execute_1: object
    execute_2: object
    execute_4: object
    execute_8: object

    def function(self, gpu_count: int):
        """Return the function for one supported GPU count."""
        functions = {
            1: self.execute_1,
            2: self.execute_2,
            4: self.execute_4,
            8: self.execute_8,
        }
        try:
            return functions[gpu_count]
        except KeyError as error:
            raise ValueError("Modal supports 1, 2, 4, or 8 GPUs") from error


def modal_worker(
    local_python_sources=(),
    pip_packages=(),
    *,
    secrets=(),
    runtime_package="vllm==0.26.0",
) -> ModalWorker:
    """Return Modal Functions containing requested extensions."""
    local_python_sources = tuple(local_python_sources)
    pip_packages = tuple(pip_packages)
    secrets = tuple(secrets)

    # Every function stays in the existing app so it shares image and volume
    # caches with earlier Quail workers.
    worker_app = modal.App("quail-engine")
    worker_image = build_worker_image(
        local_python_sources,
        pip_packages,
        runtime_package,
    )
    volumes = {
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    }

    def define_function(name, gpu, gpu_count, memory):
        @worker_app.function(
            name=name,
            serialized=True,
            image=worker_image,
            gpu=gpu,
            memory=memory,
            secrets=secrets,
            max_containers=1,
            scaledown_window=300,
            startup_timeout=120,
            timeout=21600,
            volumes=volumes,
        )
        def execute(value):
            result = _execute_logical_query(value, gpu_count)
            return result.collect(), result.report

        return execute

    execute_1 = define_function("execute_1", "H100!", 1, 98304)
    execute_2 = define_function("execute_2", "H100!:2", 2, 131072)
    execute_4 = define_function("execute_4", "H100!:4", 4, 196608)
    execute_8 = define_function("execute_8", "H100!:8", 8, 262144)

    return ModalWorker(
        worker_app,
        execute_1,
        execute_2,
        execute_4,
        execute_8,
    )


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
