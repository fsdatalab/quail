"""Quail model execution inside a compute worker.

One GPU runs in the worker process. Several GPUs run one child process
per GPU, each with its own CUDA context and arena, coordinated by the
worker process over pipes.
"""

import gc
import itertools
import json
import time

from quail.backends.base import GpuContext
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.graph import (
    _join_round_kv,
    _tuple_suffix,
    execute_single_graph,
)
from quail.backends.quail.retention import apply_retention, retain_after_join
from quail.execution import PhysicalResponse
from quail.executor.arena import KVArena
from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION, Pipeline
from quail.executor.loop import AsyncAnswers, warm_kernels
from quail.executor.model import answer_weights, load_model, resolve_model_path
from quail.physical import (
    AnchoredJoin,
    DocumentInput,
    PackedFilter,
    decode_graph,
)
from quail.planner import budgets
from quail.progress import say, set_gpu_index
from quail.runtime.runner import ExecutionContext
from quail.runtime.tokens import (
    DocumentPrefixes,
    chain_tokens,
    decode_payload_documents,
)

# GPU child processes, one per GPU, kept alive across queries so their
# models stay loaded for the whole session.
_CHILDREN: list = []


class LoadedGpu:
    """One model loaded on one GPU, reusable across queries."""

    def __init__(self, backend, context, answer_token_ids, *,
                 model_path=None):
        import torch
        import torch.nn.functional as F

        spec, device = context.model, context.device
        gpu_index = context.gpu_index
        self.torch = torch
        self.F = F
        self.spec = spec
        self._warmed = False
        self.prepared_boot = None

        set_gpu_index(gpu_index)
        free, total = torch.cuda.mem_get_info()
        say(f"loading {spec.hf_name} onto GPU {gpu_index}, "
            f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free")

        t0 = time.perf_counter()
        self.model = load_model(model_path or spec.hf_name,
                                revision=None if model_path else spec.revision,
                                answer_token_ids=answer_token_ids)
        self.load_model_s = time.perf_counter() - t0

        # cuBLAS allocates its handle outside PyTorch's caching allocator.
        # At the warm-up peak the allocator holds every free byte, so
        # create the handle now while memory is available.
        torch.cuda.current_blas_handle()
        probe = torch.ones(8, 64, device="cuda", dtype=torch.bfloat16)
        F.linear(probe, probe)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        budget = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, budget)
        self.arena = KVArena(n_layers=spec.layers,
                             n_pages=arena_tok // budgets.PAGE_TOKENS,
                             page_tokens=budgets.PAGE_TOKENS,
                             n_kv=spec.n_kv, d_head=spec.d_head,
                             dtype=torch.bfloat16)
        self.arena_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.pipeline = Pipeline(self.model, self.arena,
                                 attention_mode=FILTER_ATTENTION)
        self.pipeline_s = time.perf_counter() - t0

        self.execution = backend.start(context)
        self.execution.bind_loaded_model(
            model=self.model, arena=self.arena, pipeline=self.pipeline)

        self.async_ans = None
        self.chunk_tokens = None

    def bind_query(self, true_ids, false_ids, chunk_tokens):
        """Attach the answerer and chunk budget for one query."""
        answerer = _PayloadAnswerer(self.torch, self.F, self.model,
                                    true_ids, false_ids)
        self.async_ans = AsyncAnswers(self.torch, answerer)
        self.chunk_tokens = chunk_tokens
        self.execution.bind_query(
            torch=self.torch,
            async_answers=self.async_ans,
            chunk_tokens=chunk_tokens,
        )

    def warm(self):
        """Compile and warm kernels once.

        Returns:
            Tuple of (seconds spent warming, compilation tier).
        """
        if self._warmed:
            return 0.0, None
        from quail.runtime.volumes import commit_kernel_cache

        t0 = time.perf_counter()
        with self.torch.inference_mode():
            warm = warm_kernels(self.torch, self.arena, self.pipeline,
                                self.async_ans, self.chunk_tokens,
                                model_name=self.spec.hf_name)
        self.torch.cuda.synchronize()
        commit_kernel_cache()
        warm_s = time.perf_counter() - t0
        self._warmed = True
        return warm_s, warm["tier"]

    def close(self):
        """Release GPU resources held by the execution context."""
        close_fn = getattr(self.execution, "close", None)
        if callable(close_fn):
            close_fn()


def _boot_record(gpu, cold, warm_s, warm_tier, t_boot):
    """Assemble a boot timing dict from a loaded GPU and warm results."""
    kind = "cold" if cold or warm_tier else "warm"
    boot = {
        "kind": kind,
        "load_model_s": round(gpu.load_model_s if cold else 0.0, 2),
        "arena_s": round(gpu.arena_s if cold else 0.0, 2),
        "pipeline_s": round(gpu.pipeline_s if cold else 0.0, 2),
        "warm_kernels_s": round(warm_s, 2),
        "boot_s": round(time.perf_counter() - t_boot, 2),
    }
    if warm_tier:
        boot["warm_tier"] = warm_tier
    return boot


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


def _boot_for_query(runtime_state, backend, spec, device, workers,
                    chunk_tokens, true_ids, false_ids):
    """Load or reuse a GPU, bind a query, warm kernels."""
    key = (backend.name, spec.name)
    gpu = runtime_state.get(key)
    t_boot = time.perf_counter()
    if gpu is None:
        gpu = LoadedGpu(backend, GpuContext(
            gpu_index=0, gpu_count=workers, model=spec, device=device,
            query_settings={},
        ), true_ids + false_ids)
        runtime_state[key] = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(true_ids, false_ids, chunk_tokens)
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return gpu, boot


def prepare_quail_request(context) -> None:
    """Boot one GPU from the plan alone; the documents can arrive later."""
    if context.gpu_count != 1:
        return
    envelope = context.request.plan
    settings = dict(envelope["settings"])
    registry = context.registry
    backend = registry.backend(envelope["backend"])
    spec = registry.model(envelope["model"])
    device = registry.device(envelope["device"])
    gpu, boot = _boot_for_query(
        context.runtime_state, backend, spec, device,
        envelope["workers"], settings["chunk_tokens"],
        settings["true_ids"], settings["false_ids"])
    gpu.prepared_boot = boot


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
        if isinstance(state, LoadedGpu):
            state.close()
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


def execute_quail_payload(payload, registry, graph, backend, runtime_state):
    """Execute one Quail payload on the worker's own GPU."""
    from quail.runtime.volumes import (
        commit_kernel_cache,
        commit_results,
        run_record_path,
    )

    spec = registry.model(payload["model"])
    device = registry.device(payload["physical_plan"]["device"])

    key = (backend.name, spec.name)
    gpu = runtime_state.get(key)
    boot = None
    if isinstance(gpu, LoadedGpu):
        boot = gpu.prepared_boot
        gpu.prepared_boot = None
    if boot is None:
        gpu, boot = _boot_for_query(
            runtime_state, backend, spec, device, payload["workers"],
            payload["chunk_tokens"], payload["true_ids"], payload["false_ids"])
    say("running the query")

    state = _gpu_state(gpu)
    report = execute_single(state, payload, registry, graph)
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    result_path = run_record_path()
    report["result_volume_path"] = result_path
    if result_path is not None:
        with open(result_path, "w") as f:
            json.dump(dict(
                wall_s=report["wall_s"], boot_s=report["boot_s"],
                boot_kind=report["boot_kind"], boot=boot,
                fresh_tokens=report["fresh_tokens"],
                regret_tokens=report.get("regret_tokens"),
                node_metrics=report.get("node_metrics"),
                executed_join_plan=report.get("executed_join_plan", []),
                kv_manager=report.get("kv_manager")), f)
    commit_results()
    commit_kernel_cache()    # persist any JIT artifacts this run built
    return PhysicalResponse(outputs, report)


def _gpu_state(gpu):
    """Build a state dict from a LoadedGpu for graph execution."""
    return {
        "torch": gpu.torch, "F": gpu.F,
        "model_execution": gpu.execution,
        "model": gpu.model, "arena": gpu.arena,
        "pipeline": gpu.pipeline, "spec": gpu.spec,
    }


def execute_single(state, payload: dict, registry, graph) -> dict:
    """Execute the typed Quail graph on one GPU."""
    runtime_state = {
        **state,
        "docs": decode_payload_documents(payload["docs"]),
        "runtimes": registry.runtimes,
        "model_spec": state.get("spec") or state.get("model_spec"),
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
    set_gpu_index(gpu_idx)
    state = {"gpu_index": gpu_idx, "gpu": None}
    while True:
        try:
            kind, data = conn.recv()
        except EOFError:
            break
        if kind == "shutdown":
            break
        try:
            if kind == "boot":
                state["registry"] = data["registry"]
                state["model_path"] = data["model_path"]
                conn.send(("ok", _child_boot(state, data)))
            elif kind == "filters":
                conn.send(("ok", _child_filters(state, data)))
            elif kind == "joins":
                conn.send(("ok", _child_joins(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    envelope = sub["physical_plan"]
    registry = state["registry"]
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    t_boot = time.perf_counter()
    gpu = state.get("gpu")
    if gpu is None:
        gpu = LoadedGpu(backend, GpuContext(
            gpu_index=state["gpu_index"], gpu_count=sub["workers"],
            model=spec, device=device, query_settings={},
        ), sub["true_ids"] + sub["false_ids"],
            model_path=state["model_path"])
        state["gpu"] = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(sub["true_ids"], sub["false_ids"],
                   sub["chunk_tokens"])
    state["runtime_context"] = ExecutionContext(
        runtimes=registry.runtimes, model_execution=gpu.execution,
    )
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return boot


def _child_filters(state, sub):
    gpu = state.get("gpu")
    torch = gpu.torch if gpu else state["torch"]
    arena = gpu.arena if gpu else state["arena"]
    if sub.get("start_query", True):
        _reset_child_query(state)
        config = sub.get("retention", {})
        apply_retention(arena, config, config.get("initial", {}))
    out = dict(filters={}, survivors={}, retained={}, fresh_tokens=0)
    pre = sub.get("pre_ids") or []
    filter_limit = sub.get("filter_limit")
    t0 = time.perf_counter()
    with torch.inference_mode():
        pipeline = gpu.pipeline if gpu else state["pipeline"]
        pipeline.attention_mode = FILTER_ATTENTION
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
            state["seen"].update((alias, document)
                                 for document in answers)
            out["filters"][alias] = {
                int(document): row for document, row in answers.items()
            }
            out["survivors"][alias] = list(
                result.outputs[f"ids:{alias}"]
            )
    for alias, document in arena.accounting.retained:
        out["retained"].setdefault(alias, []).append(document)
    out["retained"] = {alias: sorted(set(documents))
                       for alias, documents in out["retained"].items()}
    torch.cuda.synchronize()
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    gpu = state.get("gpu")
    torch = gpu.torch if gpu else state["torch"]
    arena = gpu.arena if gpu else state["arena"]
    if sub.get("start_query", False):
        _reset_child_query(state)
    pre = sub.get("pre_ids") or []
    registry = state["registry"]
    encoded_node = sub["physical_node"]
    node = registry.codecs[encoded_node["type"]].decode(encoded_node)
    if not isinstance(node, AnchoredJoin):
        raise TypeError(f"child join received {node.type_name!r}")
    group = [stage.runtime_spec() for stage in node.stages]
    anchor_alias = node.anchor
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    config = sub.get("retention", {})
    live = {alias: partner["index"] for alias, partner in sub["partners"].items()}
    live[anchor_alias] = anchors_glob
    apply_retention(arena, config,
                    config.get("before", {}).get(node.node_id, {}), live)
    retain_anchor = bool(sub.get("retain_anchor"))
    seen = state.setdefault("seen", set())
    out_joins, tokens_total = [], 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        pipeline = gpu.pipeline if gpu else state["pipeline"]
        pipeline.attention_mode = JOIN_ATTENTION
        stage_suffixes, tuple_globs = [], []
        for j in group:
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
                retain_after_join(arena, anchor_keys[a], len(prefixes[a]), config,
                                  config.get("after", {}).get(node.node_id, {}))
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
        last = ans[-1] if ans else {}
        for a, key in enumerate(anchor_keys):
            if key not in arena.accounting.owned:
                continue
            matched = any(last.get(a, []))
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                retain_after_join(arena, key, len(prefixes[a]), config,
                                  config.get("after", {}).get(node.node_id, {}))
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
    gpu = state.get("gpu")
    arena = gpu.arena if gpu else state["arena"]
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
    from quail.runtime.volumes import commit_kernel_cache, commit_results

    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
    gpu_count = payload["workers"]
    spec = registry.model(payload["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    _ensure_children(gpu_count)
    setup = {key: payload[key] for key in (
        "model", "physical_plan", "workers", "chunk_tokens", "true_ids", "false_ids",
    )}
    setup.update(registry=registry, model_path=model_path)
    boots = _round("boot", [setup] * gpu_count)
    boot_s = round(time.perf_counter() - started, 2)
    say(f"all {gpu_count} GPUs ready; running the query")
    report = execute_distributed_graph(
        payload,
        graph,
        gpu_count,
        _round,
        spec,
        registry.device(payload["physical_plan"]["device"]),
        registry.runtimes,
        registry,
    )
    slowest = max(boots, key=lambda boot: boot["boot_s"])
    report["boot_s"] = boot_s
    report["boot_kind"] = "cold" if any(b["kind"] == "cold" for b in boots) else "warm"
    report["boot"] = {
        **slowest,
        "model_files_s": round(model_files_s, 2),
        "boot_s": boot_s,
    }
    commit_results()
    commit_kernel_cache()
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    return PhysicalResponse(outputs, report)


class _PayloadAnswerer:
    """Answerer using TRUE/FALSE token ids from the payload."""

    def __init__(self, torch, F, model, true_ids, false_ids):
        self.F = F
        self.allowed = sorted(set(true_ids) | set(false_ids))
        self.weights = answer_weights(model, self.allowed)
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
