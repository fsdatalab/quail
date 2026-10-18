"""The request backend: engine support, planning entry, boot, and the graph run."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.resources import release_models
from quail.baselines.execution import RequestModelExecution
from quail.baselines.nodes import RequestExecution
from quail.baselines.planning import plan_request_backend
from quail.execution.metrics import scalar_node_metrics
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.execution.runtimes import ModelNodeRuntime
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import Scan
from quail.planner.optimizer.candidates import SupportResult


def execute_request_graph(context, backend, engine_state, boot):
    """Run a request backend over its compute subgraph."""
    try:
        import torch
    except ImportError:
        torch = None


    envelope = context.request.plan
    documents = {
        node.alias: context.request.inputs[node.input_id].documents
        for node in context.graph.nodes
        if isinstance(node, Scan)
    }
    document_texts = {
        node.alias: context.request.inputs[node.input_id].texts
        for node in context.graph.nodes
        if isinstance(node, Scan)
        and context.request.inputs[node.input_id].texts is not None
    }
    settings = dict(envelope["settings"])
    model_execution = backend.start(GpuContext(
        gpu_index=0,
        gpus=1,
        model=context.registry.model(envelope["model"]),
        device=context.registry.device(envelope["device"]),
        query_settings={
            **settings,
            **engine_state,
            "documents": documents,
            "document_texts": document_texts,
        },
    ))
    if engine_state["client"].reset_prefix_cache() is False:
        raise RuntimeError("request engine did not reset its prefix cache")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    compute_graph = compute_subgraph(context.graph)
    run = GenericRunner().run(
        compute_graph,
        ExecutionContext(
            runtimes=context.registry.runtimes,
            model_execution=model_execution,
            sources={
                **{alias: range(len(value))
                   for alias, value in documents.items()},
                **context.request.relations,
            },
            functions=context.registry.functions,
        ),
    )
    metrics = run.metrics
    peak_bytes = (
        int(torch.cuda.max_memory_allocated())
        if torch is not None and torch.cuda.is_available() else 0
    )
    request_result = run.nodes["request-model"]
    report = {
        "backend": backend.name,
        "wall_s": round(metrics.wall_s, 2),
        "boot_s": boot["boot_s"],
        "boot_kind": boot["kind"],
        "boot": boot,
        "fresh_tokens": metrics.fresh_tokens,
        "cached_tokens": metrics.cached_tokens,
        "peak_gib": round(peak_bytes / 2**30, 2),
        "node_metrics": scalar_node_metrics(run.nodes),
        "backend_metrics": dict(request_result.metrics.extension),
    }
    return PhysicalResponse(
        export_physical_outputs(compute_graph, run),
        report,
    )


def request_runtimes() -> dict:
    """Return runtimes for the request backends' physical node."""
    return {RequestExecution.runtime_key: ModelNodeRuntime()}


SUPPORTED_DEVICES = frozenset({"h100-sxm", "rtx-pro-6000-blackwell-server"})


def _warm_boot() -> dict:
    return {
        "kind": "warm",
        "llm_init_s": 0.0,
        "weight_load_s": None,
        "kv_profile_s": None,
        "boot_s": 0.0,
    }


@dataclass
class LoadedRequestEngine:
    """One request engine loaded for a backend configuration.

    Attributes:
        settings: The client, sampling parameters, and capacity the
            engine adapter's boot returned.
        resources: Owns the engine; closing it stops the engine.
        prepared_boot: The boot record of a prepare_request not yet
            consumed by an execute_request.
    """

    settings: dict = field(default_factory=dict)
    resources: ExitStack = field(default_factory=ExitStack)
    prepared_boot: dict | None = None

    def close(self) -> None:
        self.resources.close()


@dataclass(frozen=True)
class RequestBackend:
    """A model backend that submits one request per prompt to an engine.

    The engine adapter boots vLLM or SGLang and returns a client with
    generate, reset_prefix_cache, and run_filter_sequence. The submission
    strategies decide how filter stages and join tuples become
    requests.
    """

    name: str
    engine: Any
    filter_submission: str
    join_submission: str = "anchor-major"
    locates_answers_in_text = False

    @property
    def engine_name(self) -> str:
        return self.engine.kind

    @property
    def runtime_package(self) -> str:
        return self.engine.runtime_package

    @property
    def reads_document_text(self) -> bool:
        return bool(getattr(self.engine, "reads_document_text", False))

    def supports(self, model, device, gpus: int) -> SupportResult:
        label = self.engine.label
        roles = getattr(self.engine, "roles", frozenset({"generative"}))
        if model.role not in roles:
            return SupportResult.reject(
                f"{label} does not support model {model.name!r}: "
                f"it serves {', '.join(sorted(roles))} models"
            )
        if device.name not in SUPPORTED_DEVICES:
            return SupportResult.reject(
                f"{label} does not support device {device.name!r}"
            )
        if gpus != 1:
            return SupportResult.reject(
                f"the {label} request backends use one model copy on one GPU"
            )
        if model.canvas_tokens == 1 and not getattr(self.engine, "reads_logprobs",
                                                     False):
            return SupportResult.reject(
                f"{label} does not support model {model.name!r}: its answer is "
                f"read from the first canvas row's logprobs, which {label} "
                f"does not return")
        return SupportResult.accept()

    def plan(self, region, context):
        return plan_request_backend(
            region,
            context,
            backend_name=self.name,
            filter_submission=self.filter_submission,
            join_submission=self.join_submission,
            scores_labels=getattr(self.engine, "scores_labels", False),
        )

    def start(self, context):
        return RequestModelExecution(context)

    def _load_engine(self, context) -> tuple[LoadedRequestEngine, dict]:
        """Boot the engine into the runtime state, or reuse the one there."""
        engine = context.runtime_state.get("engine")
        if engine is not None:
            return engine, _warm_boot()
        envelope = context.request.plan
        model = context.registry.model(envelope["model"])
        allowed_ids = sorted(set(
            envelope["settings"]["true_ids"]
        ) | set(envelope["settings"]["false_ids"]))
        started = time.perf_counter()
        with ExitStack() as rollback:
            # a failed boot must leave the GPU free for the next attempt
            rollback.callback(release_models, context.runtime_state)
            engine = LoadedRequestEngine()
            context.runtime_state["engine"] = engine
            engine.settings, boot = self.engine.boot(
                model, allowed_ids, resources=engine.resources)
            rollback.pop_all()
        return engine, {**boot, "boot_s": round(time.perf_counter() - started, 2)}

    def prepare_request(self, context) -> None:
        """Boot and warm the engine before the query's documents are ready."""
        engine, boot = self._load_engine(context)
        engine.prepared_boot = boot

    def execute_request(self, context):
        engine, boot = self._load_engine(context)
        if engine.prepared_boot is not None:
            boot, engine.prepared_boot = engine.prepared_boot, None
        return execute_request_graph(context, self, engine.settings, boot)

    def release_request(self, state) -> None:
        """Stop the engine loaded for this configuration."""
        release_models(state)
