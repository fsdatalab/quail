"""Run a query in the current process."""

import atexit
import os
import time

from quail.backends import BackendExecutionContext
from quail.execution.types import PhysicalRequest, PhysicalResponse
from quail.logical import WithCTE
from quail.physical import (
    ExecutionLocation,
    Scan,
    check_plan_envelope,
    decode_graph,
)
from quail.planner.plan import Refusal, RefusalError


class _BackendState:
    """The engine loaded in this process, for one configuration at a time.

    A configuration is the engine, model, device, and GPU count a plan
    names. Binding a different one releases the loaded engine first:
    one model's weights and KV take the GPU.
    """

    def __init__(self):
        self.configuration = None
        self.state = {}
        self.release = None

    def bind(self, envelope, backend) -> dict:
        """Return the runtime state for the plan, releasing another engine."""
        configuration = (getattr(backend, "engine_name", backend.name),
                         envelope["model"], envelope["device"], envelope["gpus"])
        if configuration != self.configuration:
            self.close()
            self.configuration = configuration
            self.release = getattr(backend, "release_request", None)
        return self.state

    def close(self) -> None:
        if self.release is not None:
            self.release(self.state)
        self.state.clear()
        self.configuration = self.release = None


# Loaded models outlive individual sessions.
_BACKEND_STATE = _BackendState()
atexit.register(_BACKEND_STATE.close)

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
        gpus=request.gpus,
        runtime_state=_BACKEND_STATE.bind(request.plan, backend),
    ))
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a model backend must return PhysicalResponse")
    return response


def execute_query(query, physical_executor=None, plan=None,
                  on_backend_ready=None, on_execution_ready=None):
    """Execute one query in its session's process.

    Args:
        query: The Query to run.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.
        plan: An edited PhysicalPlan to run instead of the planner's.
        on_backend_ready: Optional callback after model preparation.
        on_execution_ready: Optional callback after backend preparation and
            immediately before query execution.
    """
    if isinstance(query.logical.root, WithCTE):
        from quail.api.stages import execute_stages

        if plan is not None:
            raise RuntimeError("edited query stage plans are not supported")
        return execute_stages(query, physical_executor, on_backend_ready,
                              on_execution_ready)
    total_started = time.perf_counter()
    if plan is not None:
        if isinstance(plan, Refusal) or not hasattr(plan, "graph"):
            raise TypeError("plan must be a PhysicalPlan")
        query.plan()            # tokenization and pair tables first
        query._plan = plan
    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    plan.graph.validate(runtime_keys=set(query.session.registry.runtimes))
    plan.graph.validate_backend(plan.backend)
    if plan.gpus > 8:
        raise NotImplementedError("query execution supports at most 8 GPUs")
    has_compute = any(node.backend is not None or
                      node.location is ExecutionLocation.GPU_EXECUTOR
                      for node in plan.nodes)
    if has_compute and physical_executor is None:
        problem = gpu_problem()
        if problem is not None:
            raise RuntimeError(
                f"cannot run the model in this process: {problem}. "
                "Run your Session in a process with a CUDA GPU "
                "and the backend installed."
            )
    if has_compute and physical_executor is None:
        # Backend startup must not wait for background tokenization.
        _prepare_backend(plan, query.session.registry)
    if has_compute and on_backend_ready is not None:
        on_backend_ready()
    query.wait_for_tokens()
    physical_prepare_started = time.perf_counter()
    request = query._prepare_physical() if has_compute else None
    physical_prepare_s = time.perf_counter() - physical_prepare_started
    if on_execution_ready is not None:
        on_execution_ready()
    started = time.perf_counter()
    if has_compute:
        response = (physical_executor(request) if physical_executor is not None
                    else _execute_physical(request, query.session.registry))
    else:
        response = PhysicalResponse(
            {}, {"wall_s": 0.0, "fresh_tokens": 0, "cached_tokens": 0})
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a physical executor must return PhysicalResponse")
    coordinator_wall_s = time.perf_counter() - started
    finish_started = time.perf_counter()
    result = query.finish(response, coordinator_wall_s)
    finish_s = time.perf_counter() - finish_started
    model_wall_s = result.report["wall_s"]
    result.report.update(
        planning_s=round(query.planning_s, 4),
        input_ready_s=round(query.input_ready_s, 4),
        physical_prepare_s=round(physical_prepare_s, 4),
        model_wall_s=model_wall_s,
        finish_s=round(finish_s, 4),
    )
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
        gpus=plan.gpus,
        runtime_state=_BACKEND_STATE.bind(envelope, backend),
    ))
