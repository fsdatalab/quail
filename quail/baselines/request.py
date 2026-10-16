"""The request backend: engine support, planning entry, boot, and the graph run."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from quail.backends.base import GpuContext
from quail.baselines.execution import RequestModelExecution
from quail.baselines.nodes import RequestExecution
from quail.baselines.planning import plan_request_backend
from quail.execution.metrics import scalar_node_metrics
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.execution.runtimes import ModelNodeRuntime
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import Scan
from quail.planner.optimizer.candidates import SupportResult
from quail.specs import DEVICES


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
        gpu_count=1,
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


SUPPORTED_DEVICES = frozenset(DEVICES)


def _warm_boot() -> dict:
    return {
        "kind": "warm",
        "llm_init_s": 0.0,
        "weight_load_s": None,
        "kv_profile_s": None,
        "boot_s": 0.0,
    }


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
    def runtime_package(self) -> str:
        return self.engine.runtime_package

    @property
    def reads_document_text(self) -> bool:
        return bool(getattr(self.engine, "reads_document_text", False))

    def supports(self, model, device, gpu_count: int) -> SupportResult:
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
        if gpu_count != 1:
            return SupportResult.reject(
                f"the {label} request backends use one model copy on one GPU"
            )
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

    def execute_request(self, context):
        envelope = context.request.plan
        model = context.registry.model(envelope["model"])
        state_key = ("request-engine", self.engine.kind, model.name)
        engine_state = context.runtime_state.get(state_key)
        if engine_state is None:
            allowed_ids = sorted(set(
                envelope["settings"]["true_ids"]
            ) | set(envelope["settings"]["false_ids"]))
            device = context.registry.device(envelope["device"])
            boot_call = self.engine.boot
            if "device" in inspect.signature(boot_call).parameters:
                engine_state, boot = boot_call(model, allowed_ids, device)
            else:
                engine_state, boot = boot_call(model, allowed_ids)
            context.runtime_state[state_key] = engine_state
        else:
            boot = _warm_boot()
        return execute_request_graph(context, self, engine_state, boot)
