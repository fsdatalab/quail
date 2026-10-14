"""Per session extension registration."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any

from quail.physical import NodeCodec, PhysicalNode

if TYPE_CHECKING:
    from quail.backends.base import ModelBackend
    from quail.execution.runner import ExecutionObserver, NodeRuntime
    from quail.planner.optimizer.candidates import PhysicalPlanner
    from quail.planner.optimizer.rules import OptimizerRule
    from quail.specs import DeviceSpec, ModelSpec


_KINDS = {
    "universal_rule", "physical_planner", "cost_based_rule",
    "backend",
    "model", "device", "codec", "runtime", "observer", "function",
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
    def universal_rules(self) -> Mapping[str, OptimizerRule]:
        return self._values("universal_rule")

    @property
    def physical_planners(self) -> Mapping[str, PhysicalPlanner]:
        return self._values("physical_planner")

    @property
    def cost_based_rules(self) -> Mapping[str, OptimizerRule]:
        return self._values("cost_based_rule")

    @property
    def rules(self) -> Mapping[str, OptimizerRule]:
        """Return both rule categories in their shared registration order."""
        return MappingProxyType({
            name: value
            for (kind, name), value in self._registrations.items()
            if kind in ("universal_rule", "cost_based_rule")
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
    def codecs(self) -> Mapping[str, NodeCodec]:
        return self._values("codec")

    @property
    def runtimes(self) -> Mapping[str, NodeRuntime]:
        return self._values("runtime")

    @property
    def observer_factories(self) -> Mapping[str, Callable[[], ExecutionObserver]]:
        return self._values("observer")

    @property
    def functions(self) -> Mapping[str, Callable[..., Any]]:
        """User functions a query's Apply nodes call, by name."""
        return self._values("function")

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

    def register_node(
        self,
        node_type: type[PhysicalNode],
        *,
        runtime: NodeRuntime,
        codec: NodeCodec | None = None,
    ) -> ExtensionRegistry:
        """Register a physical node's codec and runtime together.

        Args:
            node_type: Physical node class with a type_name and runtime_key.
            runtime: Implementation that executes the node.
            codec: Node codec. Defaults to NodeCodec(node_type).
        """
        if not isinstance(node_type, type) or not issubclass(node_type, PhysicalNode):
            raise TypeError("register_node needs a PhysicalNode class")
        codec = NodeCodec(node_type) if codec is None else codec
        if codec.node_type is not node_type:
            raise ValueError("the codec must describe the registered node class")
        self._check_name("codec", codec.type_name)
        self._check_name("runtime", node_type.runtime_key)
        self._registrations["codec", codec.type_name] = codec
        self._registrations["runtime", node_type.runtime_key] = runtime
        return self

    def register_codec(self, codec: NodeCodec) -> ExtensionRegistry:
        return self._add("codec", codec.type_name, codec)

    def register_runtime(
        self, runtime: NodeRuntime, *, key: str | None = None,
    ) -> ExtensionRegistry:
        """Register the runtime for one physical node type."""
        return self._add(
            "runtime", getattr(runtime, "runtime_key", None) if key is None else key,
            runtime,
        )

    def register_function(
        self, function: Callable[..., Any], *, name: str,
    ) -> ExtensionRegistry:
        """Register a Python function for a query's apply() nodes.

        The function takes a dict of Arrow tables keyed by alias and
        returns ids or pairs; see quail.logical.Apply.
        """
        if not callable(function):
            raise TypeError("register_function needs a callable")
        return self._add("function", name, function)

    def register_observer(
        self, factory: Callable[[], ExecutionObserver], *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a factory that creates an observer for each query."""
        return self._add(
            "observer", getattr(factory, "name", None) if name is None else name,
            factory,
        )

    def _register_rule(self, rule, name, *, cost_based):
        name = rule.name if name is None else name
        kind = "cost_based_rule" if cost_based else "universal_rule"
        self._check_name(kind, name)
        if name in self.rules:
            raise ValueError(f"duplicate optimizer rule registration {name!r}")
        if getattr(rule, "target", None) not in ("logical", "physical"):
            raise ValueError("a rule target must be 'logical' or 'physical'")
        if getattr(rule, "cost_based", None) is not cost_based:
            raise ValueError(f"{kind} needs cost_based={cost_based}")
        return self._add(kind, name, rule)

    def register_universal_rule(
        self, rule: OptimizerRule, *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a rule that applies without comparing costs."""
        return self._register_rule(rule, name, cost_based=False)

    def register_physical_planner(
        self, planner: PhysicalPlanner, *, name: str | None = None,
    ) -> ExtensionRegistry:
        return self._add(
            "physical_planner", planner.name if name is None else name, planner,
        )

    def register_cost_based_rule(
        self, rule: OptimizerRule, *, name: str | None = None,
    ) -> ExtensionRegistry:
        """Register a rule that proposes or prices alternatives."""
        return self._register_rule(rule, name, cost_based=True)

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
