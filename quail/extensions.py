"""Per session extension registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quail.physical import NodeCodec, built_in_codecs


def _register(values: dict, name: str, value: Any, kind: str) -> None:
    if name in values:
        raise ValueError(f"duplicate {kind} registration {name!r}")
    values[name] = value


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
