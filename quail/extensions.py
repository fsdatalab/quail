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
    codecs: dict[str, NodeCodec] = field(default_factory=dict)
    runtimes: dict[str, Any] = field(default_factory=dict)
    table_provider_factories: dict[str, Any] = field(default_factory=dict)
    extension_packages: list[ExtensionPackage] = field(default_factory=list)

    @classmethod
    def with_built_ins(cls) -> "ExtensionRegistry":
        """Create a registry containing Quail's built in extensions."""
        return built_in_registry()

    def register_backend(self, backend: Any) -> None:
        _register(self.backends, backend.name, backend, "backend")

    def register_codec(self, codec: NodeCodec) -> None:
        _register(self.codecs, codec.type_name, codec, "node codec")

    def register_runtime(self, name: str, runtime: Any) -> None:
        _register(self.runtimes, name, runtime, "node runtime")

    def register_logical_rule(self, name: str, rule: Any) -> None:
        _register(self.logical_rules, name, rule, "logical rule")

    def register_physical_planner(self, name: str, planner: Any) -> None:
        _register(self.physical_planners, name, planner, "physical planner")

    def register_physical_rule(self, name: str, rule: Any) -> None:
        _register(self.physical_rules, name, rule, "physical rule")

    def register_table_provider(self, name: str, factory: Any) -> None:
        _register(
            self.table_provider_factories, name, factory, "table provider"
        )

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


def built_in_registry() -> ExtensionRegistry:
    """Create one registry with Quail built ins."""
    from quail.backends import QuailBackend
    from quail.runtime.runner import built_in_runtimes

    registry = ExtensionRegistry()
    registry.register_backend(QuailBackend())
    for codec in built_in_codecs():
        registry.register_codec(codec)
    for runtime_key, runtime in built_in_runtimes().items():
        registry.register_runtime(runtime_key, runtime)
    return registry


def registry_from_modules(modules: tuple[str, ...]) -> ExtensionRegistry:
    """Rebuild a registry from importable extension modules."""
    registry = built_in_registry()
    for module in modules:
        registry.load_extension(module)
    return registry
