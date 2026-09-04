"""Per session extension registration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from quail.physical import NodeCodec, built_in_codecs


def _register(values: dict, name: str, value: Any, kind: str) -> None:
    if name in values:
        raise ValueError(f"duplicate {kind} registration {name!r}")
    values[name] = value


@dataclass(frozen=True)
class ExtensionPackage:
    """One extension package available in every execution process."""

    module: str
    local_python_sources: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()


@dataclass
class ExtensionRegistry:
    """Extensions visible to one session."""

    logical_rules: dict[str, Any] = field(default_factory=dict)
    physical_planners: dict[str, Any] = field(default_factory=dict)
    physical_rules: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)
    codecs: dict[str, NodeCodec] = field(default_factory=dict)
    runtimes: dict[str, Any] = field(default_factory=dict)
    observer_factories: dict[str, Any] = field(default_factory=dict)
    source_readers: dict[str, Any] = field(default_factory=dict)
    extension_packages: list[ExtensionPackage] = field(default_factory=list)

    @classmethod
    def with_built_ins(cls) -> "ExtensionRegistry":
        """Create a registry containing Quail's built in extensions."""
        return built_in_registry()

    def register_backend(self, backend: Any) -> None:
        _register(self.backends, backend.name, backend, "backend")

    def register_model(self, model: Any) -> None:
        _register(self.models, model.name, model, "model")

    def register_device(self, device: Any) -> None:
        _register(self.devices, device.name, device, "device")

    def register_codec(self, codec: NodeCodec) -> None:
        _register(self.codecs, codec.type_name, codec, "node codec")

    def register_runtime(self, name: str, runtime: Any) -> None:
        _register(self.runtimes, name, runtime, "node runtime")

    def register_observer(self, name: str, factory: Any) -> None:
        _register(
            self.observer_factories, name, factory, "execution observer"
        )

    def register_logical_rule(self, name: str, rule: Any) -> None:
        _register(self.logical_rules, name, rule, "logical rule")

    def register_source_reader(self, name: str, reader: Any) -> None:
        """Register a remote table source reader."""
        _register(self.source_readers, name, reader, "source reader")

    def open_source(self, value) -> Any:
        """Open a registered remote table source."""
        source_type = value.get("type")
        try:
            reader = self.source_readers[source_type]
        except KeyError as error:
            raise ValueError(
                f"unknown remote source type {source_type!r}; known "
                f"source types are {sorted(self.source_readers)}"
            ) from error
        return reader(value)

    def register_physical_planner(self, name: str, planner: Any) -> None:
        _register(self.physical_planners, name, planner, "physical planner")

    def register_physical_rule(self, name: str, rule: Any) -> None:
        _register(self.physical_rules, name, rule, "physical rule")

    def load_extension(
        self,
        module: str,
        *,
        local_python_sources: tuple[str, ...] = (),
        pip_packages: tuple[str, ...] = (),
    ) -> None:
        """Import an extension and record its remote Python packages."""
        if module in self.extension_modules:
            raise ValueError(f"duplicate extension module {module!r}")
        loaded = importlib.import_module(module)
        register = getattr(loaded, "register_quail_extension", None)
        if register is None:
            raise ValueError(
                f"extension module {module!r} has no "
                "register_quail_extension(registry) function"
            )
        register(self)
        self.extension_packages.append(ExtensionPackage(
            module=module,
            local_python_sources=tuple(local_python_sources),
            pip_packages=tuple(pip_packages),
        ))

    @property
    def extension_modules(self) -> tuple[str, ...]:
        """Return extension modules in registration order."""
        return tuple(package.module for package in self.extension_packages)

    def backend(self, name: str) -> Any:
        try:
            return self.backends[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model backend {name!r}; "
                f"known backends are {sorted(self.backends)}") from error

    def model(self, name: str) -> Any:
        try:
            return self.models[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model {name!r}; known models are "
                f"{sorted(self.models)}"
            ) from error

    def device(self, name: str) -> Any:
        try:
            return self.devices[name]
        except KeyError as error:
            raise ValueError(
                f"unknown device {name!r}; known devices are "
                f"{sorted(self.devices)}"
            ) from error

    def new_observers(self) -> tuple[Any, ...]:
        """Create fresh execution observers for one query."""
        return tuple(factory() for factory in self.observer_factories.values())


def built_in_registry() -> ExtensionRegistry:
    """Create one registry with Quail built ins."""
    from quail.backends import (
        QuailBackend,
        SGLangBackend,
        pipelined_vllm_backend,
        stock_vllm_backend,
    )
    from quail.backends.quail import quail_runtimes
    from quail.backends.request import request_runtimes
    from quail.catalog import built_in_source_readers
    from quail.runtime.runner import built_in_runtimes
    from quail.specs import DEVICES, MODELS

    registry = ExtensionRegistry()
    for model in MODELS.values():
        registry.register_model(model)
    for device in DEVICES.values():
        registry.register_device(device)
    registry.register_backend(QuailBackend())
    registry.register_backend(stock_vllm_backend())
    registry.register_backend(pipelined_vllm_backend())
    registry.register_backend(SGLangBackend())
    for codec in built_in_codecs():
        registry.register_codec(codec)
    for runtimes in (built_in_runtimes(), quail_runtimes(), request_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime_key, runtime)
    for name, reader in built_in_source_readers().items():
        registry.register_source_reader(name, reader)
    return registry


def registry_from_modules(modules: tuple[str, ...]) -> ExtensionRegistry:
    """Rebuild a registry from importable extension modules."""
    registry = built_in_registry()
    for module in modules:
        registry.load_extension(module)
    return registry
