"""Run a query in the current process."""

import os
import time

from quail.backends import BackendExecutionContext
from quail.execution import PhysicalRequest, PhysicalResponse
from quail.physical import (
    Scan,
    check_plan_envelope,
    decode_graph,
    validate_streams,
)
from quail.planner.plan import Refusal
from quail.runtime.session import RefusalError

# Loaded models outlive individual sessions.
_BACKEND_STATE: dict = {}

def gpu_problem() -> str | None:
    """Return why this process cannot run a model, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    # the default check initializes the CUDA runtime and marks every
    # later fork bad, which breaks vLLM's forked engine process; the
    # NVML based check does not touch the runtime
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible to this process"
    return None


def _validate_physical_request(request, registry):
    """Validate and decode a physical execution request."""
    if not isinstance(request, PhysicalRequest):
        raise TypeError("execution requires a PhysicalRequest")
    envelope = request.plan
    check_plan_envelope(envelope)
    backend = registry.backend(envelope["backend"])
    graph = decode_graph(envelope["graph"], registry.codecs)
    graph.validate(runtime_keys=set(registry.runtimes))
    graph.validate_backend(envelope["backend"])
    validate_streams(graph)
    needed_inputs = {
        node.input_id for node in graph.nodes
        if isinstance(node, Scan)
    }
    missing = needed_inputs - set(request.inputs)
    extra = set(request.inputs) - needed_inputs
    if missing or extra:
        raise ValueError(
            "execution request has wrong input bindings; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return graph, backend


def _execute_physical(request, registry):
    """Run the backend selected by a registered physical plan."""
    graph, backend = _validate_physical_request(request, registry)
    response = backend.execute_request(BackendExecutionContext(
        request=request,
        graph=graph,
        registry=registry,
        gpu_count=request.gpu_count,
        runtime_state=_BACKEND_STATE,
    ))
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a model backend must return PhysicalResponse")
    return response


def execute_query(query, physical_executor=None, plan=None):
    """Execute one query in its session's process.

    Args:
        query: The Query to run.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.
        plan: An edited PhysicalPlan to run instead of the planner's.
    """
    total_started = time.perf_counter()
    if plan is not None:
        if isinstance(plan, Refusal) or not hasattr(plan, "graph"):
            raise TypeError("plan must be a PhysicalPlan")
        query.plan()            # tokenization and pair tables first
        query._plan = plan
    if physical_executor is None:
        problem = gpu_problem()
        if problem is not None:
            raise RuntimeError(
                f"cannot run the model in this process: {problem}. "
                "Run your Session in a process with a CUDA GPU "
                "and the backend installed."
            )
    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    plan.graph.validate(runtime_keys=set(query.session.registry.runtimes))
    plan.graph.validate_backend(plan.backend)
    if plan.workers > 8:
        raise NotImplementedError("query execution supports at most 8 GPUs")
    if physical_executor is None:
        # Backend startup must not wait for background tokenization.
        _prepare_backend(plan, query.session.registry)
    request = query._prepare_physical()
    started = time.perf_counter()
    response = (
        physical_executor(request) if physical_executor is not None
        else _execute_physical(request, query.session.registry)
    )
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a physical executor must return PhysicalResponse")
    result = query.finish(response, time.perf_counter() - started)
    result.report["token_wait_s"] = round(query.token_wait_s, 4)
    result.report["worker_total_s"] = round(
        time.perf_counter() - total_started, 4)
    return result


def _prepare_backend(plan, registry) -> None:
    """Let the backend boot from the plan alone, before documents arrive."""
    backend = registry.backend(plan.backend)
    prepare = getattr(backend, "prepare_request", None)
    if prepare is None:
        return
    envelope = plan.to_envelope(registry.codecs)
    prepare(BackendExecutionContext(
        request=PhysicalRequest(envelope, {}),
        graph=plan.graph,
        registry=registry,
        gpu_count=plan.workers,
        runtime_state=_BACKEND_STATE,
    ))
