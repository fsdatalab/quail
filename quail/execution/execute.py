"""Run a query in the current process."""

import os
import time

from quail.backends import BackendExecutionContext
from quail.execution.session import RefusalError
from quail.execution.types import (
    PhysicalRequest,
    PhysicalResponse,
    TokenizedInput,
)
from quail.pdf import PDFInput
from quail.physical import (
    PDFScan,
    PhysicalScan,
    TextScan,
    check_plan_envelope,
    decode_graph,
    validate_streams,
)
from quail.planner.plan import Refusal

# Loaded models outlive individual sessions.
_BACKEND_STATE: dict = {}

def gpu_problem() -> str | None:
    """Return why this process cannot run a model, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    # the default check initializes CUDA, which breaks vLLM's forked engine
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
    scans = [node for node in graph.nodes if isinstance(node, PhysicalScan)]
    needed_inputs = {node.input_id for node in scans}
    missing = needed_inputs - set(request.inputs)
    extra = set(request.inputs) - needed_inputs
    if missing or extra:
        raise ValueError(
            "execution request has wrong input bindings; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for node in scans:
        check_scan_input(node, request.inputs[node.input_id])
    return graph, backend


# the input each scan kind binds
SCAN_INPUT_TYPES = {TextScan: TokenizedInput, PDFScan: PDFInput}


def check_scan_input(node: PhysicalScan, value) -> None:
    """Fail when a scan's bound input is of the wrong kind or shape."""
    expected = SCAN_INPUT_TYPES.get(type(node))
    if expected is None or not isinstance(value, expected):
        raise TypeError(
            f"scan {node.node_id!r} ({type(node).__name__}) is bound to "
            f"{type(value).__name__}; it needs "
            f"{expected.__name__ if expected else 'a known input type'}")
    if len(value) != node.n_docs:
        raise ValueError(
            f"scan {node.node_id!r} plans {node.n_docs} documents but its "
            f"input holds {len(value)}")
    if isinstance(node, PDFScan) and (
            value.row_mode != node.row_mode
            or value.visual_tokens != node.visual_tokens):
        raise ValueError(
            f"scan {node.node_id!r} plans row_mode={node.row_mode!r} at "
            f"{node.visual_tokens} visual tokens, but its input has "
            f"row_mode={value.row_mode!r} at {value.visual_tokens}")


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
