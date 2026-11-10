"""Per session extension registration."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from quail.backends.base import ModelBackend
    from quail.execution.runner import ExecutionObserver, NodeRuntime
    from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class LogicalRule:
    """A logical optimizer rule that DataFusion runs over a query's plan.

    The name is the rule's ``OptimizerRule::name()``: one of Quail's
    rules or DataFusion's default rules, listed in
    ``quail_datafusion::logical_rule``. A registry lists the rules a
    session applies, by name, in the order they apply.
    """

    name: str
    kind: ClassVar[str] = "logical"


@dataclass(frozen=True)
class PhysicalRule:
    """A physical rule of the Rust planner, which rewrites a physical graph.

    The rules are in ``crates/quail-planner/src/physical_rules.rs``.
    """

    name: str
    kind: ClassVar[str] = "physical"


_KINDS = {
    "logical_rule", "physical_rule",
    "backend",
    "model", "device", "runtime", "observer",
}


class ExtensionRegistry:
    """Register and look up the extensions visible to one session."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], object] = {}
        self._modules: dict[str, None] = {}

    @classmethod
    def with_built_ins(cls) -> ExtensionRegistry:
        """Create a registry containing Quail's built in extensions."""
        # Built in backends import this module.
        from quail.builtins import built_in_registry

        return built_in_registry()

    def _values(self, kind: str) -> Mapping[str, Any]:
        return MappingProxyType({
            name: value
            for (registered_kind, name), value in self._registrations.items()
            if registered_kind == kind
        })

    @property
    def logical_rules(self) -> Mapping[str, LogicalRule]:
        return self._values("logical_rule")

    @property
    def physical_rules(self) -> Mapping[str, PhysicalRule]:
        return self._values("physical_rule")

    @property
    def rules(self) -> Mapping[str, LogicalRule | PhysicalRule]:
        """Return both rule categories in their shared registration order."""
        return MappingProxyType({
            name: value
            for (kind, name), value in self._registrations.items()
            if kind in ("logical_rule", "physical_rule")
        })

    @property
    def backends(self) -> Mapping[str, ModelBackend]:
        return self._values("backend")

    @property
    def models(self) -> Mapping[str, ModelSpec]:
        return self._values("model")

    @property
    def devices(self) -> Mapping[str, DeviceSpec]:
        return self._values("device")

    @property
    def runtimes(self) -> Mapping[str, NodeRuntime]:
        return self._values("runtime")

    @property
    def observer_factories(self) -> Mapping[str, Callable[[], ExecutionObserver]]:
        return self._values("observer")

    def _check_name(self, kind: str, name: str) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown extension kind {kind!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"a {kind} needs a nonempty string name")
        if (kind, name) in self._registrations:
            raise ValueError(f"duplicate {kind} registration {name!r}")

    def _add(self, kind: str, name: str, value: object) -> ExtensionRegistry:
        self._check_name(kind, name)
        self._registrations[kind, name] = value
        return self

    def register_backend(self, backend: ModelBackend) -> ExtensionRegistry:
        return self._add("backend", backend.name, backend)

    def register_model(self, model: ModelSpec) -> ExtensionRegistry:
        return self._add("model", model.name, model)

    def register_device(self, device: DeviceSpec) -> ExtensionRegistry:
        return self._add("device", device.name, device)

    def register_runtime(
        self, runtime: NodeRuntime, *, key: str | None = None,
    ) -> ExtensionRegistry:
        """Register the runtime for one physical node type."""
        return self._add(
            "runtime", getattr(runtime, "runtime_key", None) if key is None else key,
            runtime,
        )

    def register_observer(
        self, factory: Callable[[], ExecutionObserver], *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a factory that creates an observer for each query."""
        return self._add(
            "observer", getattr(factory, "name", None) if name is None else name,
            factory,
        )

    def _register_rule(self, rule, name, *, kind, rule_type):
        if not isinstance(rule, rule_type):
            raise TypeError(
                f"{kind} needs a {rule_type.__name__}: Quail's rules are "
                f"written in Rust and registered by name")
        name = rule.name if name is None else name
        self._check_name(kind, name)
        if name in self.rules:
            raise ValueError(f"duplicate optimizer rule registration {name!r}")
        return self._add(kind, name, rule)

    def register_logical_rule(
        self, rule: LogicalRule, *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a logical rule of the Rust planner by name."""
        return self._register_rule(
            rule, name, kind="logical_rule", rule_type=LogicalRule)

    def register_physical_rule(
        self, rule: PhysicalRule, *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a physical rule of the Rust planner by name, in apply order."""
        return self._register_rule(
            rule, name, kind="physical_rule", rule_type=PhysicalRule)

    def load_extension(self, module: str | ModuleType) -> ExtensionRegistry:
        """Call a module's register_quail_extension function once.

        Args:
            module: Extension module or importable module name.
        """
        loaded = (
            module if isinstance(module, ModuleType)
            else importlib.import_module(module)
        )
        name = loaded.__name__
        if name in self._modules:
            raise ValueError(f"duplicate extension module {name!r}")
        register = getattr(loaded, "register_quail_extension", None)
        if not callable(register):
            raise ValueError(
                f"extension module {name!r} has no "
                "register_quail_extension(registry) function"
            )
        registrations = self._registrations.copy()
        modules = self._modules.copy()
        self._modules[name] = None
        try:
            register(self)
        except BaseException:
            self._registrations = registrations
            self._modules = modules
            raise
        return self

    @property
    def extension_modules(self) -> tuple[str, ...]:
        """Return loaded module names in registration order."""
        return tuple(self._modules)

    def backend(self, name: str) -> ModelBackend:
        try:
            return self.backends[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model backend {name!r}; "
                f"known backends are {sorted(self.backends)}") from error

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model {name!r}; known models are "
                f"{sorted(self.models)}"
            ) from error

    def device(self, name: str) -> DeviceSpec:
        try:
            return self.devices[name]
        except KeyError as error:
            raise ValueError(
                f"unknown device {name!r}; known devices are "
                f"{sorted(self.devices)}"
            ) from error

    def new_observers(self) -> tuple[ExecutionObserver, ...]:
        """Create fresh execution observers for one query."""
        return tuple(factory() for factory in self.observer_factories.values())
