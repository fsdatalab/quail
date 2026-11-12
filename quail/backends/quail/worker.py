"""Load and run Quail models on the machine's GPUs.

Each GPU has one worker process, with its own CUDA context and arena,
which keeps its model loaded between queries. Rust starts the workers
and sends each its partition of a region's stages
(``quail_python::workers``); this module is the process that runs.
"""

import atexit
import json
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import _quail_native

from quail.backends.base import GpuContext
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.cuda_graphs import PiecewiseGraphs
from quail.backends.quail.executor.model_loading import load_model
from quail.backends.quail.executor.models import build_pipeline
from quail.backends.quail.executor.readout import (
    AnswerRows,
    AsyncAnswers,
    AsyncDecisions,
    AsyncDecisionScores,
    DecisionHead,
)
from quail.backends.quail.executor.warmup import warm_kernels
from quail.backends.quail.graph import (
    payload_tables,
    run_planned_region,
    serve_region,
)
from quail.backends.resources import freeze_loaded_objects, release_models
from quail.common import budgets
from quail.physical.plan import graph_from_envelope
from quail.progress import (
    say,
    set_answer_sink,
    set_gpu_index,
    set_progress_sink,
)
from quail.specs.files import resolve_model_path

# The workers and the configuration they load. They outlive sessions,
# so later queries reuse their loaded models.
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
                                            else None),
                                quantize_fp8=(spec.arch == "lfm3"
                                              and spec.weight_precision == "fp8"))
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
        self.pipeline.piecewise_graphs = PiecewiseGraphs.for_specs(
            self.pipeline, spec, device)
        self.pipeline_s = time.perf_counter() - t0

        tokenizer = None
        if spec.role not in ("reranker", "decision"):
            say(f"loading the tokenizer of {spec.name} for AI.MAP and AI.EXTRACT")
            tokenizer = model_tokenizer(spec, model_path)
        self.execution = backend.start(context)
        self.execution.bind_loaded_model(
            model=self.model, arena=self.arena, pipeline=self.pipeline,
            decision_head=self.decision_head, tokenizer=tokenizer)

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
            graphs = self.pipeline.piecewise_graphs
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
            graphs = self.pipeline.piecewise_graphs
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


def model_tokenizer(spec, model_path: str = ""):
    """Load a model's tokenizer, with the ids that end its answers.

    Args:
        spec: The model spec.
        model_path: A local checkpoint that replaces the spec's Hugging
            Face repository at its pinned revision.
    """
    from quail.common.tokens import Tokenizer
    from quail.specs.files import resolve_model_path

    path = resolve_model_path(model_path or spec.hf_name,
                              None if model_path else spec.revision,
                              include_weights=False)
    return Tokenizer(path, spec.turn_suffix)


def decision_offsets(spec, path) -> tuple[int, ...]:
    """The rows a decision readout reads, as distances before the last row."""
    from quail.common.tokens import Tokenizer, model_answers

    _, _, offsets = model_answers(spec, Tokenizer(path))
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

    # a KVArena's recurrent state has one more row than its slots
    rows = getattr(arena, "gdn_state_rows", None)
    return {
        "kv": sum(size(t) for t in list(arena.k) + list(arena.v)),
        "state": (size(arena.gdn_state if rows is None else rows)
                  + size(arena.conv_state)),
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
        if cold:
            freeze_loaded_objects()
        boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
        say(f"model ready after {boot['boot_s']} s ({boot['kind']} start)")
        rollback.pop_all()
    return gpu, boot


def release_quail_request(state: dict) -> None:
    """Release the GPU worker processes and the models they loaded."""
    _close_workers()
    state.clear()


def _execution_state(gpu, payload, registry) -> dict:
    """Return the executor's state for one GPU and query."""
    return {
        "gpu": gpu,
        "documents": payload["documents"],
        "texts": payload.get("texts", {}),
        "columns": payload.get("columns", {}),
        "runtimes": registry.runtimes,
        "device": registry.device(payload["physical_plan"]["device"]),
        "chunk_tokens": payload["chunk_tokens"],
        "gpu_timing": payload.get("gpu_timing", False),
    }


@dataclass
class WorkerState:
    """The GPU resources of one worker process."""

    gpu_index: int
    registry_of: Any = None
    gpu: LoadedGpu | None = None
    registry: Any = None


def main(path: str, gpu_index: int, registry_of, describe) -> None:
    """Run a GPU worker: answer Quail's messages until it is done with us.

    Args:
        path: The socket Quail listens at for this worker.
        gpu_index: Which of the session's GPUs this worker runs.
        registry_of: Callable(extension module names) that returns the
            registry of the session's models, devices, and backends.
        describe: Callable(configuration) that returns what Quail plans
            the configuration's regions with.
    """
    set_gpu_index(gpu_index)
    state = WorkerState(gpu_index, registry_of=registry_of)
    end = _quail_native.WorkerEnd(path)
    while (message := end.next()) is not None:
        try:
            if message[0] == "describe":
                end.described(json.dumps(describe(json.loads(message[1]))))
            elif message[0] == "boot":
                end.booted(json.dumps(_boot_worker(state, json.loads(message[1]))))
            elif message[0] == "region":
                _serve_region(state, json.loads(message[1]), message[2], end)
        except Exception:
            end.fail(traceback.format_exc())


def _boot_worker(state, setup) -> dict:
    """Prepare this worker's GPU for a plan; returns the boot record."""
    if state.registry is None:
        state.registry = state.registry_of(setup["extensions"])
    registry = state.registry
    spec = registry.model(setup["model"])
    gpu, state.gpu = state.gpu, None
    state.gpu, boot = _boot_gpu(
        gpu, registry.backend(setup["backend"]), GpuContext(
            gpu_index=state.gpu_index, gpus=setup["gpus"], model=spec,
            device=registry.device(setup["device"]), query_settings={}),
        setup["chunk_tokens"], setup["true_ids"], setup["false_ids"],
        setup.get("arena_pages"),
        # a setup without the model's files has the worker find them
        model_path=(setup["model_path"]
                    or resolve_model_path(spec.hf_name, spec.revision)))
    return boot


def _serve_region(state, region, tables, end) -> None:
    """Serve one region's stages on this worker's GPU until the region ends."""
    payload = {**region["query"], **payload_tables(tables)}
    graph = graph_from_envelope(payload["physical_plan"])
    progress, answers = region["listening"]
    # the listeners are in the region's process: each event is sent there
    if progress:
        set_progress_sink(lambda *event: end.ask(json.dumps({"progress": event})))
    if answers:
        set_answer_sink(lambda event: end.ask(json.dumps({"answer": [event]})))
    try:
        serve_region(_execution_state(state.gpu, payload, state.registry),
                     payload, graph, end, region["wanted"], region["hinted"])
    finally:
        set_progress_sink(None)
        set_answer_sink(None)


def _close_workers():
    global _WORKERS
    found, _WORKERS = _WORKERS, None
    if found is not None:
        found[1].close()


atexit.register(_close_workers)


def _ensure_workers(gpus, configuration):
    """Return live workers for the configuration, replacing any others.

    Args:
        gpus: How many GPUs, one worker process each.
        configuration: The model, device, backend, and GPU count the
            workers load.
    """
    global _WORKERS
    if (_WORKERS is None or _WORKERS[0] != configuration
            or not _WORKERS[1].alive()):
        _close_workers()
        workers = _quail_native.Workers(gpus)
        try:
            workers.spawn([sys.executable, "-m", "quail.worker"])
            workers.connect()
        except BaseException:
            workers.close()
            raise
        _WORKERS = (configuration, workers)
    return _WORKERS[1]


def _boot_workers(envelope, registry) -> dict:
    """Start or reuse each GPU's worker and prepare it for the plan.

    Returns:
        The workers under "workers", and the boot record of the run:
        the slowest worker's, with the seconds the model's files and the
        whole boot took.
    """
    gpus = envelope["gpus"]
    spec = registry.model(envelope["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    workers = _ensure_workers(gpus, (
        envelope["model"], envelope["device"], envelope["backend"], gpus))
    settings = envelope["settings"]
    setup = {
        **{key: envelope[key] for key in ("model", "device", "backend", "gpus")},
        **{key: settings[key] for key in ("chunk_tokens", "true_ids", "false_ids")},
        "arena_pages": settings.get("arena_pages"),
        "model_path": model_path,
        # a worker imports the session's extensions by module name
        "extensions": list(registry.extension_modules),
    }
    boots = [json.loads(boot) for boot in workers.boot(json.dumps(setup))]
    boot_s = round(time.perf_counter() - started, 2)
    say(f"all {gpus} GPUs ready" if gpus > 1 else "the GPU is ready")
    return {
        "workers": workers,
        "boot": {
            **max(boots, key=lambda boot: boot["boot_s"]),
            "kind": ("cold" if any(boot["kind"] == "cold" for boot in boots)
                     else "warm"),
            "model_files_s": round(model_files_s, 2),
            "boot_s": boot_s,
        },
    }


def prepare_quail_request(context) -> None:
    """Boot the GPUs from the plan alone; the documents can arrive later."""
    context.runtime_state["booted"] = _boot_workers(
        context.request.plan, context.registry)


def run_quail_region(envelope, registry, runtime_state, **region):
    """Run a planned region on the GPUs' worker processes.

    Args:
        envelope: The plan's envelope.
        registry: The session's registry.
        runtime_state: The backend's state, which may hold a boot
            prepared for this region.
        **region: What ``run_planned_region`` reads of the region.

    Returns:
        The region's rows, and the run's report with its outputs under
        "_outputs".
    """
    # a boot prepared for this region is used once
    booted = (runtime_state.pop("booted", None)
              or _boot_workers(envelope, registry))
    try:
        rows, report = run_planned_region(
            booted["workers"], envelope, registry.device(envelope["device"]),
            **region)
    except BaseException:
        # a worker may be partway through the region: start afresh
        _close_workers()
        raise
    boot = booted["boot"]
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    return rows, report
