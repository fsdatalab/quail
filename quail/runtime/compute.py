"""Compute providers for remote physical plan execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ComputeProvider(Protocol):
    """Run a prepared physical plan in a remote compute process."""

    def execute(
        self,
        payload: Mapping[str, Any],
        gpu_count: int,
        extensions: Sequence[Any],
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class ModalComputeProvider:
    """Run physical plans in the existing Quail Modal app."""

    def __init__(self):
        self._app_context = None
        self._worker = None
        self._extension_key = None

    def worker(self, extensions=()):
        """Return Modal functions containing the requested extensions."""
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
        key = (local_sources, pip_packages)
        if self._app_context is not None and key != self._extension_key:
            self.close()
        if self._app_context is None:
            from quail.runtime import worker
            self._worker = worker.modal_worker(
                local_python_sources=local_sources,
                pip_packages=pip_packages,
            )
            self._app_context = self._worker.app.run()
            self._app_context.__enter__()
            self._extension_key = key
        return self._worker

    def execute(self, payload, gpu_count, extensions=()):
        """Execute one request on the selected Modal GPU function."""
        worker = self.worker(extensions)
        function = (
            worker.execute if gpu_count == 1 else
            worker.execute_2 if gpu_count == 2 else
            worker.execute_4 if gpu_count == 4 else
            worker.execute_8 if gpu_count == 8 else None
        )
        if function is None:
            raise ValueError("Modal supports 1, 2, 4, or 8 GPUs")
        call = function.spawn(dict(payload))
        print(f"function call id: {call.object_id}", flush=True)
        return call.get()

    def close(self) -> None:
        """Stop the active Modal app context."""
        if self._app_context is not None:
            self._app_context.__exit__(None, None, None)
        self._app_context = None
        self._worker = None
        self._extension_key = None
