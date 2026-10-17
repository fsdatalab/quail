"""Load and run Quail models on the process's GPUs.

One GPU runs in the calling process. Several GPUs use one child process
per GPU, each with its own CUDA context and arena.
"""

import gc
import time
from dataclasses import dataclass
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.model_loading import load_model, resolve_model_path
from quail.backends.quail.executor.models import build_pipeline
from quail.backends.quail.executor.operators.extract import execute_extractions
from quail.backends.quail.executor.operators.join import join_inputs
from quail.backends.quail.executor.readout import (
    AnswerRows,
    AsyncAnswers,
    AsyncDecisions,
    AsyncDecisionScores,
    DecisionHead,
)
from quail.backends.quail.executor.state import GpuRuntime
from quail.backends.quail.executor.warmup import warm_kernels
from quail.backends.quail.graph import execute_single_graph
from quail.backends.quail.retention import apply_retention
from quail.catalog.tokens import (
    DocumentPrefixes,
    decode_payload_documents,
)
from quail.cost import budgets
from quail.execution.runner import ExecutionContext
from quail.execution.types import PhysicalResponse
from quail.physical import (
    AiFilter,
    AiJoin,
    Scan,
    decode_graph,
)
from quail.progress import say, set_gpu_index

# Children outlive sessions so later queries can reuse their loaded models.
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

        budget = budgets.chunk_budget(spec, device)
        t0 = time.perf_counter()
        self.model = load_model(model_path or spec.hf_name,
                                revision=None if model_path else spec.revision,
                                answer_token_ids=answer_token_ids,
                                max_batched_tokens=budget,
                                moe_backend=spec.moe_backend)
        self.load_model_s = time.perf_counter() - t0

        # cuBLAS allocates its handle outside PyTorch's caching allocator.
        # At the warm-up peak the allocator holds every free byte, so
        # create the handle now while memory is available.
        torch.cuda.current_blas_handle()
        probe = torch.ones(8, 64, device="cuda", dtype=torch.bfloat16)
        F.linear(probe, probe)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        full_pages, sliding_pages = budgets.arena_pages(spec, device, budget)
        self.arena = KVArena(num_layers=spec.layers,
                             num_pages=full_pages,
                             page_tokens=budgets.PAGE_TOKENS,
                             num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                             dtype=torch.bfloat16,
                             layer_kv=spec.kv_shapes,
                             sliding_layers=spec.sliding_layer_set,
                             sliding_window=spec.sliding_window,
                             num_sliding_pages=sliding_pages)
        self.arena_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.decision_head = None
        if spec.role == "decision":
            path = resolve_model_path(model_path or spec.hf_name,
                                      None if model_path else spec.revision)
            self.decision_head = DecisionHead.load(torch, F, path)
            self.decision_offsets = decision_offsets(spec, path)
        self.pipeline = build_pipeline(spec, self.model, self.arena)
        self.pipeline_s = time.perf_counter() - t0

        self.execution = backend.start(context)
        self.execution.bind_loaded_model(
            model=self.model, arena=self.arena, pipeline=self.pipeline,
            decision_head=self.decision_head)

        self.async_ans = None
        self.chunk_tokens = None

    def bind_query(self, true_ids, false_ids, chunk_tokens, arena_pages=None):
        """Attach the answer rows, readout, and chunk budget for one query.

        arena_pages resizes the KV pools to the plan's split; nothing
        survives in the arena between queries.
        """
        if arena_pages is not None:
            graphs = getattr(self.pipeline, "decode_graphs", None)
            if (graphs is not None and tuple(arena_pages)
                    != (self.arena.num_pages, self.arena.num_sliding_pages)):
                self.torch.cuda.synchronize()
                graphs.invalidate()
            self.arena.resize(*arena_pages, free_resident=True)
        rows = AnswerRows(self.torch, self.F, self.model, true_ids, false_ids)
        async_scores = None
        if self.decision_head is not None:
            self.async_ans = AsyncDecisions(
                self.torch, self.decision_head, self.decision_offsets)
            async_scores = AsyncDecisionScores(
                self.torch, self.decision_head, self.decision_offsets)
        else:
            self.async_ans = AsyncAnswers(self.torch, rows)
        self.chunk_tokens = chunk_tokens
        self.execution.bind_query(
            torch=self.torch,
            async_answers=self.async_ans,
            answer_rows=rows,
            chunk_tokens=chunk_tokens,
            async_scores=async_scores,
        )

    def warm(self):
        """Compile and warm kernels once.

        Returns:
            Tuple of (seconds spent warming, compilation tier).
        """
        t0 = time.perf_counter()
        tier = None
        with self.torch.inference_mode():
            if not self._warmed:
                warm = warm_kernels(self.torch, self.arena, self.pipeline,
                                    self.async_ans, self.chunk_tokens,
                                    model_name=self.spec.hf_name,
                                    model=self.model)
                tier = warm["tier"]
            graphs = getattr(self.pipeline, "decode_graphs", None)
            if graphs is not None and graphs.prepare():
                tier = tier or "graphs"
        if tier is None:
            return 0.0, None
        self.torch.cuda.synchronize()
        warm_s = time.perf_counter() - t0
        self._warmed = True
        return warm_s, tier

    def close(self):
        """Release GPU resources held by the execution context."""
        close_fn = getattr(self.execution, "close", None)
        if callable(close_fn):
            close_fn()


def decision_offsets(spec, path) -> tuple[int, ...]:
    """The rows a decision readout reads, as distances before the last row."""
    from gigatoken import Tokenizer

    from quail.planner.prompt_lowering import answer_row_offsets

    tokenizer = Tokenizer(path).as_hf()
    offsets = answer_row_offsets(
        spec.prompt_layout,
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        spec.turn_suffix)
    if len(offsets) != 3:
        raise ValueError(
            f"{spec.name}: the decision readout reads a no row, a yes row, "
            f"and the last row; layout {spec.prompt_layout!r} gives "
            f"{len(offsets)} rows")
    return offsets


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
    documents = {}
    texts = {}
    for node in graph.nodes:
        if not isinstance(node, Scan):
            continue
        documents[node.alias] = request.inputs[node.input_id].documents
        if request.inputs[node.input_id].texts is not None:
            texts[node.alias] = request.inputs[node.input_id].texts
    return {
        "physical_plan": envelope,
        "model": envelope["model"],
        "gpus": envelope["gpus"],
        "documents": documents,
        "texts": texts,
        "columns": request.column_tables(),
        **dict(envelope["settings"]),
    }


def _single_gpu_context(registry, envelope):
    """Build the GPU context for the single-GPU boot path."""
    return GpuContext(
        gpu_index=0,
        gpus=envelope["gpus"],
        model=registry.model(envelope["model"]),
        device=registry.device(envelope["device"]),
        query_settings={},
    )


def _boot_for_query(runtime_state, backend, gpu_context,
                    chunk_tokens, true_ids, false_ids, arena_pages=None):
    """Load or reuse a GPU, bind a query, warm kernels."""
    key = (backend.name, gpu_context.model.name)
    gpu = runtime_state.get(key)
    t_boot = time.perf_counter()
    if gpu is None:
        loaded = [name for _, name in runtime_state if name != key[1]]
        if loaded:
            # one model's weights and KV arena take the GPU; another
            # model cannot load beside them
            say(f"releasing {', '.join(loaded)} to load {key[1]}")
            release_booted_models(runtime_state)
        gpu = LoadedGpu(backend, gpu_context, true_ids + false_ids)
        runtime_state[key] = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(true_ids, false_ids, chunk_tokens, arena_pages)
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return gpu, boot


def prepare_quail_request(context) -> None:
    """Boot one GPU from the plan alone; the documents can arrive later."""
    if context.gpus != 1:
        return
    envelope = context.request.plan
    settings = dict(envelope["settings"])
    registry = context.registry
    gpu, boot = _boot_for_query(
        context.runtime_state, registry.backend(envelope["backend"]),
        _single_gpu_context(registry, envelope), settings["chunk_tokens"],
        settings["true_ids"], settings["false_ids"],
        settings.get("arena_pages"))
    gpu.prepared_boot = boot


def execute_quail_request(context):
    """Run one Quail request from a backend execution context."""
    payload = quail_runtime_payload(context.request, context.graph)
    if context.gpus == 1:
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
    gpu_context = _single_gpu_context(registry, payload["physical_plan"])

    key = (backend.name, gpu_context.model.name)
    gpu = runtime_state.get(key)
    boot = None
    if isinstance(gpu, LoadedGpu):
        boot = gpu.prepared_boot
        gpu.prepared_boot = None
    if boot is None:
        gpu, boot = _boot_for_query(
            runtime_state, backend, gpu_context, payload["chunk_tokens"],
            payload["true_ids"], payload["false_ids"],
            payload.get("arena_pages"))
    say("running the query")

    report = execute_single(gpu, payload, registry, graph)
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    return PhysicalResponse(outputs, report)


def execute_single(gpu: GpuRuntime, payload: dict, registry, graph) -> dict:
    """Execute the typed Quail graph on one GPU."""
    runtime_state = {
        "gpu": gpu,
        "documents": decode_payload_documents(payload["documents"]),
        "texts": payload.get("texts", {}),
        "columns": payload.get("columns", {}),
        "functions": registry.functions,
        "runtimes": registry.runtimes,
        "device": registry.device(payload["physical_plan"]["device"]),
        "chunk_tokens": payload["chunk_tokens"],
        "gpu_timing": payload.get("gpu_timing", False),
    }
    return execute_single_graph(runtime_state, payload, graph)


@dataclass
class ChildState:
    """GPU resources and query inputs owned by one worker process."""

    gpu_index: int
    gpu: LoadedGpu | None = None
    registry: Any = None
    model_path: str | None = None
    runtime_context: ExecutionContext | None = None
    score_documents: Any = None


def _child_main(gpu_index, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    set_gpu_index(gpu_index)
    state = ChildState(gpu_index)
    while True:
        try:
            kind, data = conn.recv()
        except EOFError:
            break
        if kind == "shutdown":
            break
        try:
            if kind == "boot":
                state.registry = data["registry"]
                state.model_path = data["model_path"]
                conn.send(("ok", _child_boot(state, data)))
            elif kind == "filters":
                conn.send(("ok", _child_filters(state, data)))
            elif kind == "scores":
                gpu = state.gpu
                inputs = data["inputs"]
                if "documents" in inputs:
                    state.score_documents = inputs["documents"]
                inputs["documents"] = state.score_documents
                with gpu.torch.inference_mode():
                    result = gpu.execution.execute(data["node"], inputs)
                gpu.torch.cuda.synchronize()
                conn.send(("ok", result))
            elif kind == "joins":
                conn.send(("ok", _child_joins(state, data)))
            elif kind == "extract":
                conn.send(("ok", _child_extract(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    envelope = sub["physical_plan"]
    registry = state.registry
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    t_boot = time.perf_counter()
    gpu = state.gpu
    if gpu is None:
        gpu = LoadedGpu(backend, GpuContext(
            gpu_index=state.gpu_index, gpus=sub["gpus"],
            model=spec, device=device, query_settings={},
        ), sub["true_ids"] + sub["false_ids"],
            model_path=state.model_path)
        state.gpu = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(sub["true_ids"], sub["false_ids"],
                   sub["chunk_tokens"], sub.get("arena_pages"))
    state.runtime_context = ExecutionContext(
        runtimes=registry.runtimes, model_execution=gpu.execution,
    )
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return boot


def _execute_predicates(context, nodes, inputs):
    """Dispatch one predicate or a pipeline of predicates on the worker."""
    if len(nodes) == 1:
        node = nodes[0]
        return {node.node_id: context.runtimes[node.runtime_key].execute(
            node, inputs, context)}
    return context.model_execution.execute_predicates(nodes, inputs)


def _child_filters(state, sub):
    gpu = state.gpu
    torch = gpu.torch
    arena = gpu.arena
    if sub.get("start_query", True):
        _reset_child_query(state)
        config = sub.get("retention", {})
        apply_retention(arena, config, config.get("initial", {}))
    results = {}
    filter_limit = sub.get("filter_limit")
    t0 = time.perf_counter()
    with torch.inference_mode():
        node_id = sub.get("node_id")
        if node_id is not None:
            graph = decode_graph(
                sub["physical_plan"]["graph"],
                state.registry.codecs,
            )
            nodes = tuple(graph.node(member)
                          for member in sub.get("node_ids", (node_id,)))
            node = nodes[0]
            if not isinstance(node, AiFilter):
                raise TypeError(
                    f"child filter received {node.type_name!r}")
            alias = node.alias
            index = sub["document_index"][alias]
            runtime_context = state.runtime_context
            results = _execute_predicates(
                runtime_context, nodes,
                {
                    "documents": DocumentPrefixes(
                        node.document_head,
                        sub["documents"][alias],
                        range(len(sub["documents"][alias])),
                    ),
                    "document_ids": index,
                    "limit": filter_limit,
                    "stop_groups": sub.get("stop_groups"),
                    "gpu_timing": sub.get("gpu_timing", False),
                },
            )
    return _child_results(torch, arena, results, wall_started=t0)


def _child_results(torch, arena, results, *, wall_started=None, **metrics):
    """Collect worker outputs and measurements after GPU work completes."""
    torch.cuda.synchronize()
    retained = {}
    for alias, document in arena.retained_keys():
        if isinstance(alias, tuple) and alias[0] == "extract":
            alias = alias[1]
        retained.setdefault(alias, []).append(document)
    return {
        "results": results,
        "retained": {alias: sorted(documents)
                     for alias, documents in retained.items()},
        "kv_totals": {
            "evicted_keys": arena.evicted_keys,
            "evicted_pages": arena.evicted_pages,
            "evicted_prefix_tokens": arena.evicted_prefix_tokens,
        },
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        **({"wall_s": round(time.perf_counter() - wall_started, 2)}
           if wall_started is not None else {}),
        **metrics,
    }


def _child_extract(state, sub):
    """Extract answers on one worker and report its retained document KV."""
    gpu = state.gpu
    with gpu.torch.inference_mode():
        results = execute_extractions(gpu.execution.query, sub["nodes"], sub["inputs"])
    return _child_results(gpu.torch, gpu.arena, results)


def _child_joins(state, sub):
    gpu = state.gpu
    torch = gpu.torch
    arena = gpu.arena
    if sub.get("start_query", False):
        _reset_child_query(state)
    registry = state.registry
    nodes = tuple(registry.codecs[value["type"]].decode(value)
                  for value in sub["physical_nodes"])
    node = nodes[0]
    if not isinstance(node, AiJoin):
        raise TypeError(f"child join received {node.type_name!r}")
    anchor_alias = node.anchor
    anchors_glob = list(sub["anchor_index"])
    anchor_documents = sub["anchor_documents"]
    config = sub.get("retention", {})
    expected_rows = {
        alias: partner["index"] for alias, partner in sub["partners"].items()
    }
    expected_rows[anchor_alias] = anchors_glob
    apply_retention(arena, config,
                    config.get("before", {}).get(node.node_id, {}), expected_rows)
    t0 = time.perf_counter()
    with torch.inference_mode():
        prefixes = DocumentPrefixes(node.document_head, anchor_documents,
                                    range(len(anchor_documents)))
        prepared = join_inputs(
            nodes, {alias: partner["documents"]
                    for alias, partner in sub["partners"].items()},
            expected_rows, prefixes, arena, config, sub.get("pairs", {}))
        round_kv = prepared["kv_round"]
        prepared["gpu_timing"] = sub.get("gpu_timing", False)
        results = _execute_predicates(state.runtime_context, nodes, prepared)
    if sub.get("final_group"):
        for key in arena.resident_keys():
            arena.free_key(key)
    return _child_results(torch, arena, results, wall_started=t0, kv_round=round_kv)


def _reset_child_query(state):
    """Clear KV and counters before a child starts a new query."""
    gpu = state.gpu
    arena = gpu.arena
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()


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
    payload = dict(payload)
    payload["documents"] = decode_payload_documents(payload["documents"])
    gpus = payload["gpus"]
    spec = registry.model(payload["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    _ensure_children(gpus)
    setup = {key: payload[key] for key in (
        "model", "physical_plan", "gpus", "chunk_tokens", "true_ids", "false_ids",
    )}
    setup["arena_pages"] = payload.get("arena_pages")
    setup.update(registry=registry, model_path=model_path)
    boots = _round("boot", [setup] * gpus)
    boot_s = round(time.perf_counter() - started, 2)
    say(f"all {gpus} GPUs ready; running the query")
    report = execute_distributed_graph(
        payload,
        graph,
        gpus,
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
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    return PhysicalResponse(outputs, report)
