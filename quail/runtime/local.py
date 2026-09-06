"""Run a logical query request in the current process.

The Modal worker and the in-process compute provider both come here.
"""

import json
import os
import time
from dataclasses import dataclass, field

from quail.backends import BackendExecutionContext
from quail.builtins import registry_from_manifest
from quail.execution import PhysicalRequest, PhysicalResponse
from quail.physical import DocumentInput, check_plan_envelope, decode_graph
from quail.planner.plan import Refusal
from quail.runtime.compute import QueryRequest
from quail.runtime.result import QueryResult
from quail.runtime.session import Query, RefusalError, Session


@dataclass
class _WorkerRuntime:
    """Per process state that backends keep between queries."""

    booted: dict = field(default_factory=dict)


_RUNTIME = _WorkerRuntime()


def _validate_physical_request(request):
    """Validate and decode a physical execution request."""

    if not isinstance(request, PhysicalRequest):
        raise TypeError("the worker needs a PhysicalRequest")
    envelope = request.plan
    check_plan_envelope(envelope)
    registry = registry_from_manifest(envelope["extensions"])
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


def _execute_physical(request):
    """Run the backend selected by a registered physical plan."""

    request, registry, graph, backend = _validate_physical_request(request)
    response = backend.execute_request(BackendExecutionContext(
        request=request,
        graph=graph,
        registry=registry,
        gpu_count=request.gpu_count,
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
        from quail.runtime.volumes import results_vol

        results_vol.commit()
        response = PhysicalResponse(response.outputs, metrics)
    return response


def execute_worker_query(query, physical_executor=None):
    """Execute one query inside its current worker process."""

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


def execute_query_request(
    request: QueryRequest, physical_executor=None
) -> QueryResult:
    """Plan and run one logical query request in this process."""
    started = time.perf_counter()
    registry = registry_from_manifest(request.extensions)
    session = Session(request.config, device=request.device,
                      registry=registry)
    for name, provider in request.providers.items():
        session.register(name, provider)
    query = Query(session, request.logical_plan, order=request.order)
    result = execute_worker_query(query, physical_executor)
    result.report["worker_total_s"] = round(
        time.perf_counter() - started, 4
    )
    return result
