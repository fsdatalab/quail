"""Compute providers for logical query execution."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from quail.catalog import ScanRequest, TableProvider
from quail.extensions import ExtensionRegistry
from quail.logical import LogicalPlan
from quail.logical_rules import push_down_projection
from quail.planner import collect_operators
from quail.planner.plan import EngineConfig
from quail.runtime.result import QueryResult


@dataclass(frozen=True)
class QueryRequest:
    """A logical query submitted to a compute provider."""

    logical_plan: LogicalPlan
    providers: Mapping[str, TableProvider]
    config: EngineConfig
    order: str | None = None
    registry: ExtensionRegistry = field(
        default_factory=ExtensionRegistry.with_built_ins)
    # the caller's Query, reused by an in-process provider; remote ones ignore it
    planned_query: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logical_plan, LogicalPlan):
            raise TypeError("query request needs a LogicalPlan")
        if not self.providers:
            raise ValueError("query request needs at least one provider")
        if not all(isinstance(name, str) and name for name in self.providers):
            raise TypeError("query request providers need nonempty names")
        if not self.config.device:
            raise ValueError("query request needs a device")

    @property
    def gpu_count(self) -> int:
        """Return the requested GPU count."""
        return int(self.config.gpus)


class ComputeProvider(Protocol):
    """Run a logical query through one compute provider."""

    def execute(self, request: QueryRequest) -> QueryResult: ...

    def close(self) -> None: ...


LOCAL_KERNEL_CACHE = "~/.cache/quail/kernels"

# same cache layout as the Modal image; values already in the environment win
_LOCAL_ENV_DEFAULTS = (
    ("VLLM_CACHE_ROOT", "vllm"),
    ("DG_CACHE_DIR", "deep_gemm"),
    ("DG_JIT_CACHE_DIR", "deep_gemm"),
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
)


def default_local_caches() -> str:
    """Point every kernel cache at one directory and return its root."""
    root = os.path.expanduser(LOCAL_KERNEL_CACHE)
    for name, sub in _LOCAL_ENV_DEFAULTS:
        os.environ.setdefault(name, os.path.join(root, sub))
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return root


def local_gpu_problem() -> str | None:
    """Return why this process cannot run a model, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible to this process"
    return None


class InProcessComputeProvider:
    """Run logical queries on the GPU in the current process.

    The default provider. A fake physical executor skips the GPU check.
    """

    def __init__(self, physical_executor=None):
        self._physical_executor = physical_executor

    def execute(self, request: QueryRequest) -> QueryResult:
        if self._physical_executor is None:
            # before torch loads, so the allocator setting takes effect
            default_local_caches()
            problem = local_gpu_problem()
            if problem is not None:
                raise RuntimeError(
                    f"cannot run the model in this process: {problem}. "
                    "Pass compute_provider=quail.ModalComputeProvider() "
                    "to Session to run on Modal instead."
                )
        # local imports the session module, which imports this one
        from quail.runtime.local import execute_query_request

        return execute_query_request(request, self._physical_executor)

    def close(self) -> None:
        return None


def _modal_request(request: QueryRequest) -> dict:
    """Prepare one request for a Modal Function."""
    scans, _, _ = collect_operators(
        LogicalPlan(push_down_projection(request.logical_plan.root)))
    needed = {
        name: {provider.id_col}
        for name, provider in request.providers.items()
    }
    for scan in scans:
        needed[scan.provider].add(scan.column)
        needed[scan.provider].update(scan.columns)

    sources = {}
    for name, provider in request.providers.items():
        remote = provider.remote_source()
        if remote is not None:
            sources[name] = {"remote": dict(remote)}
            continue
        columns = tuple(
            column for column in provider.columns if column in needed[name]
        )
        reader = provider.scan(ScanRequest(columns=columns))
        try:
            table = reader.read_all()
        finally:
            reader.close()
        sources[name] = {
            "table": table,
            "id_col": provider.id_col,
        }

    return {
        "logical_plan": request.logical_plan,
        "sources": sources,
        "config": request.config,
        "order": request.order,
        "registry": request.registry,
    }


class ModalComputeProvider:
    """Run logical queries with Modal Functions."""

    def __init__(self, *, secrets=(), detach: bool = False,
                 local_python_sources: tuple[str, ...] = (),
                 pip_packages: tuple[str, ...] = (),
                 initialize_worker: Callable[[ExtensionRegistry], None] | None = None):
        self._app_context = None
        self._worker = None
        self._worker_key = None
        self._functions = {}
        self._modal_secrets = tuple(secrets)
        self._detach = bool(detach)
        self._local_python_sources = tuple(local_python_sources)
        self._pip_packages = tuple(pip_packages)
        self._initialize_worker = initialize_worker

    def _worker_for(self, backend_name: str, registry: ExtensionRegistry):
        """Return Modal Functions containing the requested extensions."""
        local_sources = tuple(dict.fromkeys(self._local_python_sources))
        pip_packages = tuple(dict.fromkeys(self._pip_packages))
        backend = registry.backend(backend_name)
        runtime_package = getattr(
            backend, "runtime_package", "vllm==0.26.0"
        )
        key = (backend_name, runtime_package, local_sources, pip_packages)
        if self._app_context is not None and key != self._worker_key:
            self.close()
        if self._app_context is None:
            # the worker module imports modal, which is slow to load
            # and only needed by this provider
            from quail.runtime import worker

            self._worker = worker.modal_worker(
                local_python_sources=local_sources,
                pip_packages=pip_packages,
                secrets=self._modal_secrets,
                runtime_package=runtime_package,
            )
            self._app_context = self._worker.app.run(detach=self._detach)
            self._app_context.__enter__()
            self._worker_key = key
        return self._worker

    def _function(self, gpu_count: int, backend_name: str, registry: ExtensionRegistry):
        worker = self._worker_for(backend_name, registry)
        function_key = (backend_name, gpu_count)
        function = self._functions.get(function_key)
        if function is None:
            function = worker.function(gpu_count)
            function.update_autoscaler(min_containers=1)
            self._functions[function_key] = function
        return function

    def execute(self, request: QueryRequest) -> QueryResult:
        """Execute one logical query with a Modal Function."""
        if request.config.device != "h100-sxm":
            raise ValueError(
                "ModalComputeProvider currently provisions H100s only; "
                "use InProcessComputeProvider on an RTX PRO 6000 host")
        function = self._function(
            request.gpu_count,
            request.config.backend,
            request.registry,
        )
        call = function.spawn(_modal_request(request), self._initialize_worker)
        print(f"function call id: {call.object_id}", flush=True)
        result = call.get()
        if not isinstance(result, QueryResult):
            raise TypeError("a Modal worker must return a QueryResult")
        return result

    def close(self) -> None:
        """Release the selected Modal Functions."""
        try:
            for function in self._functions.values():
                function.update_autoscaler(min_containers=0)
        finally:
            if self._app_context is not None:
                self._app_context.__exit__(None, None, None)
            self._functions = {}
            self._app_context = None
            self._worker = None
            self._worker_key = None
