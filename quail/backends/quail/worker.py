"""Load and run Quail models on the process's GPUs.

One GPU runs in the calling process. Several GPUs use one child process
per GPU, each with its own CUDA context and arena.
"""

import atexit
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.model_loading import load_model
from quail.backends.quail.executor.models import build_pipeline
from quail.backends.quail.executor.readout import (
    AnswerRows,
    AsyncAnswers,
    AsyncDecisions,
    AsyncDecisionScores,
    DecisionHead,
)
from quail.backends.quail.executor.state import GpuRuntime
from quail.backends.quail.executor.warmup import warm_kernels
from quail.backends.quail.graph import (
    execute_single_graph,
    execution_context,
    execution_measurements,
)
from quail.backends.resources import release_models
from quail.catalog.tokens import (
    decode_payload_documents,
)
from quail.cost import budgets
from quail.execution.runner import ExecutionContext
from quail.execution.types import PhysicalResponse
from quail.physical import (
    Scan,
    decode_graph,
)
from quail.progress import say, set_gpu_index
from quail.specs.files import resolve_model_path


class LoadedGpu:
    """One model loaded on one GPU, reusable across queries."""

    def __init__(self):
        self.execution = None
        self.model = self.arena = self.pipeline = None
        self.decision_head = self.async_ans = None
        self.chunk_tokens = None
        self._warmed = False
        self.prepared_boot = None

    def load(self, backend, context, answer_token_ids, *, model_path=None):
        """Load model weights, allocate KV, and bind the execution context."""
        import torch
        import torch.nn.functional as F

        spec, device = context.model, context.device
        gpu_index = context.gpu_index
        self.torch = torch
        self.F = F
        self.spec = spec

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
        del probe

        t0 = time.perf_counter()
        full_pages, sliding_pages = budgets.arena_pages(spec, device, budget)
        self.arena = KVArena(n_layers=spec.layers,
                             n_pages=full_pages,
                             page_tokens=budgets.PAGE_TOKENS,
                             n_kv=spec.n_kv, d_head=spec.d_head,
                             dtype=torch.bfloat16,
                             layer_kv=spec.kv_shapes,
                             sliding_layers=spec.sliding_layer_set,
                             sliding_window=spec.sliding_window,
                             n_sliding_pages=sliding_pages)
        self.arena_s = time.perf_counter() - t0

        t0 = time.perf_counter()
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

    def bind_query(self, true_ids, false_ids, chunk_tokens, arena_pages=None):
        """Attach the answer rows, readout, and chunk budget for one query.

        arena_pages resizes the KV pools to the plan's split; nothing
        survives in the arena between queries.
        """
        if arena_pages is not None:
            graphs = getattr(self.pipeline, "decode_graphs", None)
            if (graphs is not None and tuple(arena_pages)
                    != (self.arena.n_pages, self.arena.n_sliding_pages)):
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
        """Release the execution context and the loaded GPU resources."""
        close_fn = getattr(self.execution, "close", None)
        if callable(close_fn):
            close_fn()
        self.execution = None
        self.model = self.arena = self.pipeline = None
        self.decision_head = self.async_ans = None


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
    docs = {}
    texts = {}
    for node in graph.nodes:
        if not isinstance(node, Scan):
            continue
        docs[node.alias] = request.inputs[node.input_id].documents
        if request.inputs[node.input_id].texts is not None:
            texts[node.alias] = request.inputs[node.input_id].texts
    return {
        "physical_plan": envelope,
        "model": envelope["model"],
        "workers": envelope["workers"],
        "docs": docs,
        "texts": texts,
        "columns": request.column_tables(),
        **dict(envelope["settings"]),
    }


def _single_gpu_context(registry, envelope):
    """Build the GPU context for the single-GPU boot path."""
    return GpuContext(
        gpu_index=0,
        gpu_count=envelope["workers"],
        model=registry.model(envelope["model"]),
        device=registry.device(envelope["device"]),
        query_settings={},
    )


def _boot_for_query(runtime_state, backend, gpu_context,
                    chunk_tokens, true_ids, false_ids, arena_pages=None):
    """Load or reuse a GPU, bind a query, warm kernels."""
    gpu, boot = _boot_gpu(
        runtime_state.pop("gpu", None), backend, gpu_context,
        chunk_tokens, true_ids, false_ids, arena_pages)
    runtime_state["gpu"] = gpu
    return gpu, boot


def _boot_gpu(gpu, backend, context, chunk_tokens, true_ids, false_ids,
              arena_pages=None, *, model_path=None):
    """Prepare a GPU for a query, releasing its resources if preparation fails."""
    t_boot = time.perf_counter()
    cold = gpu is None
    with ExitStack() as rollback:
        if cold:
            gpu = LoadedGpu()
        rollback.callback(release_models, {"gpu": gpu})
        if cold:
            gpu.load(backend, context, true_ids + false_ids, model_path=model_path)
        gpu.bind_query(true_ids, false_ids, chunk_tokens, arena_pages)
        warm_s, warm_tier = gpu.warm()
        boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
        say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
        rollback.pop_all()
    return gpu, boot


def prepare_quail_request(context) -> None:
    """Boot the GPUs from the plan alone; the documents can arrive later."""
    envelope = context.request.plan
    registry = context.registry
    if context.gpu_count != 1:
        workers, boot = _boot_workers(envelope, registry)
        workers.prepared_boot = boot
        context.runtime_state["workers"] = workers
        return
    settings = dict(envelope["settings"])
    gpu, boot = _boot_for_query(
        context.runtime_state, registry.backend(envelope["backend"]),
        _single_gpu_context(registry, envelope), settings["chunk_tokens"],
        settings["true_ids"], settings["false_ids"],
        settings.get("arena_pages"))
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
    return execute_quail_multi(
        payload, context.registry, context.graph,
        workers=context.runtime_state.get("workers"))


def release_quail_request(state: dict) -> None:
    """Release the GPU workers and the model loaded in this process."""
    _close_children()
    release_models(state)


def execute_quail_payload(payload, registry, graph, backend, runtime_state):
    """Execute one Quail payload on the worker's own GPU."""
    gpu_context = _single_gpu_context(registry, payload["physical_plan"])

    gpu = runtime_state.get("gpu")
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
    return execute_single_graph(
        _execution_state(gpu, payload, registry), payload, graph)


def _execution_state(gpu, payload, registry):
    return {
        "gpu": gpu,
        "docs": decode_payload_documents(payload["docs"]),
        "texts": payload.get("texts", {}),
        "columns": payload.get("columns", {}),
        "functions": registry.functions,
        "runtimes": registry.runtimes,
        "device": registry.device(payload["physical_plan"]["device"]),
        "chunk_tokens": payload["chunk_tokens"],
        "gpu_timing": payload.get("gpu_timing", False),
    }


@dataclass
class ChildState:
    """GPU resources and query inputs owned by one worker process."""

    gpu_index: int
    gpu: LoadedGpu | None = None
    registry: Any = None
    model_path: str | None = None
    runtime_context: ExecutionContext | None = None


def _child_main(gpu_idx, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    set_gpu_index(gpu_idx)
    state = ChildState(gpu_idx)
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
            elif kind == "execute":
                conn.send(("ok", _child_execute(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    envelope = sub["physical_plan"]
    registry = state.registry
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    gpu, state.gpu = state.gpu, None
    state.runtime_context = None
    state.gpu, boot = _boot_gpu(
        gpu, backend, GpuContext(
            gpu_index=state.gpu_index, gpu_count=sub["workers"],
            model=spec, device=device, query_settings={},
        ), sub["chunk_tokens"], sub["true_ids"], sub["false_ids"],
        sub.get("arena_pages"), model_path=state.model_path)
    return boot


def _child_execute(state, sub):
    """Run a graph group with the same context and executor as a local query."""
    if "query" in sub:
        payload = sub["query"]
        graph = decode_graph(payload["physical_plan"]["graph"], state.registry.codecs)
        state.runtime_context = execution_context(
            _execution_state(state.gpu, payload, state.registry), payload, graph)
    context = state.runtime_context
    context.state["filter_limit"] = sub["limit"]
    nodes, inputs = sub["nodes"], sub["inputs"]
    gpu = state.gpu
    with gpu.torch.inference_mode():
        if sub["pipeline"]:
            results = context.run_pipeline(nodes, inputs, context)
        else:
            (node,) = nodes
            results = {node.node_id: context.runtimes[node.runtime_key].execute(
                node, inputs[node.node_id], context)}
    gpu.torch.cuda.synchronize()
    if sub["final"]:
        for key in gpu.arena.resident_keys():
            gpu.arena.free_key(key)
    return {"results": results, "retained": tuple(gpu.arena.retained_keys()),
            **execution_measurements(context)}


class GpuWorkers:
    """Own the GPU processes and pipes for one engine configuration."""

    def __init__(self, count, configuration):
        import multiprocessing as mp

        self.configuration = configuration
        self.prepared_boot = None
        self.children = []
        context = mp.get_context("spawn")
        try:
            for gpu in range(count):
                parent_conn, child_conn = context.Pipe()
                try:
                    process = context.Process(
                        target=_child_main, args=(gpu, child_conn), daemon=True)
                    process.start()
                except BaseException:
                    parent_conn.close()
                    raise
                finally:
                    child_conn.close()
                self.children.append((process, parent_conn))
        except BaseException:
            self.close(terminate=True)
            raise

    @property
    def alive(self):
        return bool(self.children) and all(
            process.is_alive() for process, _ in self.children)

    def close(self, *, terminate=False):
        """Close idle workers normally, or terminate an interrupted round."""
        self.prepared_boot = None
        children, self.children = self.children, []
        for _, connection in children:
            connection.close()
        deadline = time.monotonic() + 5.0
        for process, _ in children:
            if not terminate:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
            process.join()

    def round(self, kind, subs):
        """Send one round and collect results, releasing workers on failure."""
        try:
            for (_, connection), sub in zip(self.children, subs):
                connection.send((kind, sub))
            outputs = []
            for (_, connection), _ in zip(self.children, subs):
                status, data = connection.recv()
                if status != "ok":
                    raise RuntimeError(f"GPU child failed:\n{data}")
                outputs.append(data)
            return outputs
        except BaseException:
            self.close(terminate=True)
            raise


_WORKERS: GpuWorkers | None = None


def _close_children():
    global _WORKERS
    workers, _WORKERS = _WORKERS, None
    if workers is not None:
        workers.close()


atexit.register(_close_children)


def _ensure_children(count, configuration):
    global _WORKERS
    if (_WORKERS is None or _WORKERS.configuration != configuration
            or not _WORKERS.alive):
        _close_children()
        _WORKERS = GpuWorkers(count, configuration)
    return _WORKERS


def _boot_workers(envelope, registry):
    """Load or reuse GPU workers and prepare them for the plan."""
    gpu_count = envelope["workers"]
    spec = registry.model(envelope["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    device = registry.device(envelope["device"])
    workers = _ensure_children(
        gpu_count, (spec, device, envelope["backend"], model_path, gpu_count))
    settings = envelope["settings"]
    setup = {key: settings[key] for key in ("chunk_tokens", "true_ids", "false_ids")}
    setup.update(
        model=envelope["model"], physical_plan=envelope, workers=gpu_count,
        arena_pages=settings.get("arena_pages"), registry=registry,
        model_path=model_path)
    boots = workers.round("boot", [setup] * gpu_count)
    boot_s = round(time.perf_counter() - started, 2)
    slowest = max(boots, key=lambda boot: boot["boot_s"])
    return workers, {
        **slowest,
        "kind": "cold" if any(boot["kind"] == "cold" for boot in boots) else "warm",
        "model_files_s": round(model_files_s, 2),
        "boot_s": boot_s,
    }


def execute_quail_multi(payload, registry, graph, workers=None):
    """Execute the typed Quail graph across GPU child processes."""
    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
    envelope = payload["physical_plan"]
    if workers is not None and workers.prepared_boot is not None:
        boot, workers.prepared_boot = workers.prepared_boot, None
    else:
        workers, boot = _boot_workers(envelope, registry)
    gpu_count = payload["workers"]
    say(f"all {gpu_count} GPUs ready; running the query")
    report = execute_distributed_graph(
        payload,
        graph,
        gpu_count,
        workers.round,
        registry.device(envelope["device"]),
        registry.runtimes,
        registry,
    )
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    return PhysicalResponse(outputs, report)
