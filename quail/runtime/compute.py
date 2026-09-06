"""Compute providers for logical query execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa

from quail.builtins import registry_from_modules
from quail.catalog import ScanRequest, TableProvider
from quail.extensions import ExtensionPackage
from quail.logical import LogicalPlan
from quail.planner import collect_operators
from quail.planner.plan import EngineConfig
from quail.runtime.result import QueryResult



@dataclass(frozen=True)
class QueryRequest:
    """A logical query submitted to a compute provider."""

    logical_plan: LogicalPlan
    providers: Mapping[str, TableProvider]
    config: EngineConfig
    device: str
    order: str | None = None
    extensions: tuple[ExtensionPackage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.logical_plan, LogicalPlan):
            raise TypeError("query request needs a LogicalPlan")
        if not self.providers:
            raise ValueError("query request needs at least one provider")
        if not all(isinstance(name, str) and name for name in self.providers):
            raise TypeError("query request providers need nonempty names")
        if not self.device:
            raise ValueError("query request needs a device")

    @property
    def gpu_count(self) -> int:
        """Return the requested GPU count."""
        return int(self.config.gpus)


class ComputeProvider(Protocol):
    """Run a logical query through one compute provider."""

    def execute(self, request: QueryRequest) -> QueryResult: ...

    def close(self) -> None: ...


class InProcessComputeProvider:
    """Run logical queries in the current process.

    This is the provider for code that already runs where the GPUs
    are, such as the benchmark runner inside a Modal function, and for
    tests that fake the physical executor.
    """

    def __init__(self, physical_executor=None):
        self._physical_executor = physical_executor

    def execute(self, request: QueryRequest) -> QueryResult:
        # local imports the session module, which imports this one
        from quail.runtime.local import execute_query_request

        return execute_query_request(request, self._physical_executor)

    def close(self) -> None:
        return None


def _modal_request(request: QueryRequest) -> dict:
    """Prepare one request for a Modal Function."""

    scans, _, _ = collect_operators(request.logical_plan)
    needed = {
        name: {provider.id_col}
        for name, provider in request.providers.items()
    }
    for scan in scans:
        needed[scan.provider].add(scan.column)
    for column in request.logical_plan.output_schema():
        needed[column.provider].add(column.column)

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
        "device": request.device,
        "order": request.order,
        "extension_modules": tuple(
            extension.module for extension in request.extensions
        ),
    }


class ModalComputeProvider:
    """Run logical queries with Modal Functions."""

    def __init__(self, *, secrets=(), detach: bool = False):
        self._app_context = None
        self._worker = None
        self._extension_key = None
        self._functions = {}
        self._modal_secrets = tuple(secrets)
        self._detach = bool(detach)

    def _worker_for(self, backend_name: str, extensions=()):
        """Return Modal Functions containing the requested extensions."""

        local_sources = tuple(dict.fromkeys(
            source
            for extension in extensions
            for source in extension.local_python_sources
        ))
        pip_packages = tuple(dict.fromkeys(
            package
            for extension in extensions
            for package in extension.pip_packages
        ))
        registry = registry_from_modules(tuple(
            extension.module for extension in extensions
        ))
        backend = registry.backend(backend_name)
        runtime_package = getattr(
            backend, "runtime_package", "vllm==0.26.0"
        )
        key = (backend_name, runtime_package, local_sources, pip_packages)
        if self._app_context is not None and key != self._extension_key:
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
            self._extension_key = key
        return self._worker

    def _function(self, gpu_count: int, backend_name: str, extensions=()):
        worker = self._worker_for(backend_name, extensions)
        function_key = (backend_name, gpu_count)
        function = self._functions.get(function_key)
        if function is None:
            function = worker.function(gpu_count)
            function.update_autoscaler(min_containers=1)
            self._functions[function_key] = function
        return function

    def execute(self, request: QueryRequest) -> QueryResult:
        """Execute one logical query with a Modal Function."""
        function = self._function(
            request.gpu_count,
            request.config.backend,
            request.extensions,
        )
        call = function.spawn(_modal_request(request))
        print(f"function call id: {call.object_id}", flush=True)
        table, report = call.get()
        if not isinstance(table, pa.Table):
            raise TypeError("a Modal worker must return an Arrow table")
        if not isinstance(report, Mapping):
            raise TypeError("a Modal worker must return a report mapping")
        return QueryResult.from_table(table, report=dict(report))

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
            self._extension_key = None
