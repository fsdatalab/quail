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

# Children outlive sessions so later queries can reuse their loaded models.
_WORKERS = None


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
                                moe_backend=spec.moe_backend,
                                text_only=spec.vision_tower,
                                num_layers=(spec.layers if spec.checkpoint_layers
                                            else None))
        self.load_model_s = time.perf_counter() - t0
        self.weights_bytes = int(torch.cuda.memory_allocated())

        # cuBLAS allocates its handle outside PyTorch's caching allocator.
        # At the warm-up peak the allocator holds every free byte, so
        # create the handle now while memory is available.
        torch.cuda.current_blas_handle()
        probe = torch.ones(8, 64, device="cuda", dtype=torch.bfloat16)
        F.linear(probe, probe)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        full_pages, sliding_pages = budgets.arena_pages(spec, device, budget)
        self.device = device
        self.arena = KVArena(num_layers=spec.layers,
                             num_pages=full_pages,
                             page_tokens=device.page_tokens,
                             num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                             dtype=torch.bfloat16,
                             layer_kv=spec.kv_shapes,
                             sliding_layers=spec.sliding_layer_set,
                             sliding_window=spec.sliding_window,
                             num_sliding_pages=sliding_pages,
                             **checkpoint_arena_args(spec, device, budget,
                                                     full_pages))
        self.arena_s = time.perf_counter() - t0
        self.pool_bytes = pool_bytes(self.arena)

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
            slots = checkpoint_arena_args(
                self.spec, self.device, chunk_tokens, arena_pages[0])
            graphs = self.pipeline.decode_graphs
            if (graphs is not None and tuple(arena_pages)
                    != (self.arena.num_pages, self.arena.num_sliding_pages)):
                self.torch.cuda.synchronize()
                graphs.invalidate()
            self.arena.resize(*arena_pages, free_resident=True,
                              **{k: v for k, v in slots.items()
                                 if k == "num_state_slots"})
            self.pool_bytes = pool_bytes(self.arena)
        self.torch.cuda.reset_peak_memory_stats()
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
            graphs = self.pipeline.decode_graphs
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
        self.execution.close()


def checkpoint_arena_args(spec, device, chunk_tokens, kv_pages) -> dict:
    """KVArena arguments for the checkpoint slots; empty without state layers."""
    if not spec.state_layers:
        return {}
    layers = len(spec.state_layers)
    return {
        "num_state_slots": budgets.checkpoint_slots(
            spec, device, chunk_tokens, kv_pages),
        "gdn_state_shape": (layers, spec.gdn_value_heads,
                            spec.gdn_value_dim, spec.gdn_key_dim),
        "conv_state_shape": (layers, spec.conv_width - 1,
                             spec.conv_state_channels),
    }


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


def pool_bytes(arena) -> dict:
    """Return the bytes of an arena's KV pages and checkpoint state.

    Args:
        arena: A KVArena, or any object with the k, v, gdn_state and
            conv_state tensors.

    Returns:
        {"kv": ..., "state": ...} in bytes. State is the Gated DeltaNet
        recurrent state plus the convolution columns, or 0.
    """
    def size(tensor):
        return 0 if tensor is None else int(
            tensor.numel() * tensor.element_size())

    return {
        "kv": sum(size(t) for t in list(arena.k) + list(arena.v)),
        "state": size(arena.gdn_state) + size(arena.conv_state),
    }



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
    """Load or reuse the process's GPU, bind a query, warm kernels."""
    gpu, boot = _boot_gpu(runtime_state.pop("gpu", None), backend, gpu_context,
                          chunk_tokens, true_ids, false_ids, arena_pages)
    runtime_state["gpu"] = gpu
    return gpu, boot


def _boot_gpu(gpu, backend, gpu_context, chunk_tokens, true_ids, false_ids,
              arena_pages=None, *, model_path=None):
    """Prepare a GPU for a query, releasing it when preparation fails.

    A gpu of None loads the model; otherwise the loaded GPU is reused.
    """
    t_boot = time.perf_counter()
    cold = gpu is None
    state = {}
    with ExitStack() as rollback:
        # a failed boot must leave the GPU free for the next attempt
        rollback.callback(release_models, state)
        if cold:
            gpu = LoadedGpu(backend, gpu_context, true_ids + false_ids,
                            model_path=model_path)
        state["gpu"] = gpu
        gpu.bind_query(true_ids, false_ids, chunk_tokens, arena_pages)
        warm_s, warm_tier = gpu.warm()
        boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
        say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
        rollback.pop_all()
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


def release_quail_request(state: dict) -> None:
    """Release the GPU child processes and the model loaded in this process."""
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
    state = _execution_state(gpu, payload, registry)
    return execute_single_graph(state, payload, graph)


def _execution_state(gpu, payload, registry) -> dict:
    """Return the single-GPU executor's state for one worker and query."""
    return {
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


@dataclass
class ChildState:
    """GPU resources and the bound query of one worker process."""

    gpu_index: int
    gpu: LoadedGpu | None = None
    registry: Any = None
    model_path: str | None = None
    runtime_context: ExecutionContext | None = None


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
            gpu_index=state.gpu_index, gpus=sub["gpus"],
            model=spec, device=device, query_settings={}),
        sub["chunk_tokens"], sub["true_ids"], sub["false_ids"],
        sub.get("arena_pages"), model_path=state.model_path)
    return boot


def _child_execute(state, sub) -> dict:
    """Run one graph group on this worker's share of its documents.

    The first group of a query carries the query itself; the worker
    binds it to the same execution context a single GPU uses.
    """
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
        if sub["pipeline"] and context.run_pipeline is not None:
            results = context.run_pipeline(nodes, inputs, context)
        else:
            # an executor without pipelines runs the group node by node
            results = {}
            for node in nodes:
                node_inputs = dict(inputs[node.node_id])
                for port in node.inputs:
                    if port.source.node_id in results:
                        node_inputs[port.name] = results[
                            port.source.node_id].outputs[port.source.port]
                results[node.node_id] = context.runtimes[node.runtime_key].execute(
                    node, node_inputs, context)
    gpu.torch.cuda.synchronize()
    if sub["final"]:
        for key in gpu.arena.resident_keys():
            gpu.arena.free_key(key)
    return {"results": results, "retained": tuple(gpu.arena.retained_keys()),
            **execution_measurements(context)}


class GpuWorkers:
    """The child process and pipe of each GPU, for one configuration.

    Args:
        count: How many GPUs, one child each.
        configuration: The model, device, backend, and GPU count the
            children were started for.
    """

    def __init__(self, count, configuration):
        import multiprocessing as mp

        self.configuration = configuration
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
    def alive(self) -> bool:
        return bool(self.children) and all(
            process.is_alive() for process, _ in self.children)

    def round(self, kind, subs):
        """Send one round to the children and collect their results."""
        for (_, conn), sub in zip(self.children, subs):
            conn.send((kind, sub))
        # every reply is read before an error is raised, so the pipes
        # stay in step with the children for the next round
        replies = [conn.recv() for (_, conn), _sub in zip(self.children, subs)]
        for status, data in replies:
            if status != "ok":
                raise RuntimeError(f"GPU child failed:\n{data}")
        return [data for _, data in replies]

    def close(self, *, terminate=False):
        """Stop every child and close its pipe."""
        children, self.children = self.children, []
        for process, conn in children:
            if not terminate and process.is_alive():
                try:
                    conn.send(("shutdown", None))
                except (BrokenPipeError, OSError):
                    pass
            conn.close()
        for process, _ in children:
            if terminate:
                process.terminate()
            process.join()
            process.close()


def _close_children():
    global _WORKERS
    workers, _WORKERS = _WORKERS, None
    if workers is not None:
        workers.close()


atexit.register(_close_children)


def _ensure_children(count, configuration) -> GpuWorkers:
    """Return live children for the configuration, replacing any others."""
    global _WORKERS
    if (_WORKERS is None or _WORKERS.configuration != configuration
            or not _WORKERS.alive):
        _close_children()
        _WORKERS = GpuWorkers(count, configuration)
    return _WORKERS


def execute_quail_multi(payload, registry, graph):
    """Execute the typed Quail graph across GPU child processes."""
    payload = dict(payload)
    payload["documents"] = decode_payload_documents(payload["documents"])
    gpus = payload["gpus"]
    spec = registry.model(payload["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    envelope = payload["physical_plan"]
    workers = _ensure_children(gpus, (
        payload["model"], envelope["device"], envelope["backend"], gpus))
    setup = {key: payload[key] for key in (
        "model", "physical_plan", "gpus", "chunk_tokens", "true_ids", "false_ids",
    )}
    setup["arena_pages"] = payload.get("arena_pages")
    setup.update(registry=registry, model_path=model_path)
    boots = workers.round("boot", [setup] * gpus)
    boot_s = round(time.perf_counter() - started, 2)
    say(f"all {gpus} GPUs ready; running the query")
    report = execute_distributed_graph(
        payload, graph, gpus, workers.round,
        registry.device(payload["physical_plan"]["device"]),
        registry.runtimes, registry)
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
