"""Per session extension registration."""

from __future__ import annotations

import importlib
import pickle
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Mapping

from quail.physical import NodeCodec

# registration kind -> the registry field that holds it
_KINDS = {
    "logical_rule": "logical_rules",
    "physical_planner": "physical_planners",
    "physical_rule": "physical_rules",
    "backend": "backends",
    "model": "models",
    "device": "devices",
    "codec": "codecs",
    "runtime": "runtimes",
    "observer": "observer_factories",
    "source_reader": "source_readers",
}


@dataclass(frozen=True)
class ExtensionPackage:
    """One extension module loaded through its entry point function."""

    module: str
    local_python_sources: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Registration:
    """One registered object, pickled for a compute worker."""

    kind: str
    name: str
    payload: bytes


@dataclass(frozen=True)
class ExtensionManifest:
    """What a compute worker needs to rebuild a session's registry.

    Entry point modules are imported and asked to register themselves.
    Registered objects travel pickled; their modules must be importable
    in the worker, which the local Python sources and pip packages
    arrange.
    """

    modules: tuple[str, ...] = ()
    registrations: tuple[Registration, ...] = ()
    local_python_sources: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()

    def to_value(self) -> dict:
        """Return the plain mapping that crosses a process boundary."""
        return {
            "modules": list(self.modules),
            "registrations": [
                [item.kind, item.name, item.payload]
                for item in self.registrations
            ],
            "local_python_sources": list(self.local_python_sources),
            "pip_packages": list(self.pip_packages),
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "ExtensionManifest":
        check_manifest_value(value)
        return cls(
            modules=tuple(value["modules"]),
            registrations=tuple(
                Registration(kind, name, payload)
                for kind, name, payload in value["registrations"]
            ),
            local_python_sources=tuple(value["local_python_sources"]),
            pip_packages=tuple(value["pip_packages"]),
        )


def check_manifest_value(value: Any) -> None:
    """Validate an extension manifest mapping."""
    required = {"modules", "registrations", "local_python_sources",
                "pip_packages"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            f"extension manifest must have fields {sorted(required)}")
    for key in ("modules", "local_python_sources", "pip_packages"):
        items = value[key]
        if not isinstance(items, list) or not all(
                isinstance(item, str) and item for item in items):
            raise ValueError(f"extension manifest {key} must be strings")
    modules = value["modules"]
    if len(modules) != len(set(modules)):
        raise ValueError("extension manifest has duplicate modules")
    for item in value["registrations"]:
        if (not isinstance(item, (list, tuple)) or len(item) != 3
                or item[0] not in _KINDS or not isinstance(item[1], str)
                or not isinstance(item[2], bytes)):
            raise ValueError(
                "extension manifest registrations must be "
                "[kind, name, payload] entries")


def _module_of(value: Any) -> str:
    module = getattr(value, "__module__", None)
    if module is None:
        module = type(value).__module__
    return module


def _top_level_source(module_name: str) -> str | None:
    """Return the local top-level package a module ships as, or None.

    Quail's own package is in every worker image. A package installed
    from PyPI is declared through pip packages instead.
    """
    top = module_name.split(".")[0]
    if top == "quail":
        return None
    if top == "__main__":
        raise ValueError(
            "an extension registered from __main__ cannot travel to a "
            "compute worker; define it in an importable module")
    module = sys.modules.get(top)
    path = getattr(module, "__file__", None) or ""
    if "site-packages" in path or "dist-packages" in path:
        return None
    return top


@dataclass
class ExtensionRegistry:
    """Extensions visible to one session.

    Register objects; names come from the objects unless given. Every
    object registered after the built-ins is pickled into the manifest
    a compute worker rebuilds the registry from, and its top-level
    package is copied into the worker image.
    """

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
    # (kind, name, object) for every registration a worker must redo
    shipped: list[tuple[str, str, Any]] = field(default_factory=list)
    record: bool = True

    @classmethod
    def with_built_ins(cls) -> "ExtensionRegistry":
        """Create a registry containing Quail's built in extensions."""
        # quail.builtins imports every built in backend, and those
        # import this module; the convenience constructor is the one
        # place the dependency runs the other way
        from quail.builtins import built_in_registry

        return built_in_registry()

    # ---- registration ------------------------------------------------

    def _add(self, kind: str, name: Any, value: Any) -> "ExtensionRegistry":
        if not isinstance(name, str) or not name:
            raise ValueError(f"a {kind} needs a nonempty string name")
        values = getattr(self, _KINDS[kind])
        if name in values:
            raise ValueError(f"duplicate {kind} registration {name!r}")
        values[name] = value
        if self.record:
            self.shipped.append((kind, name, value))
        return self

    def register_backend(self, backend: Any) -> "ExtensionRegistry":
        return self._add("backend", backend.name, backend)

    def register_model(self, model: Any) -> "ExtensionRegistry":
        return self._add("model", model.name, model)

    def register_device(self, device: Any) -> "ExtensionRegistry":
        return self._add("device", device.name, device)

    def register_codec(self, codec: NodeCodec) -> "ExtensionRegistry":
        return self._add("codec", codec.type_name, codec)

    def register_runtime(self, runtime: Any, *,
                         key: str | None = None) -> "ExtensionRegistry":
        """Register the runtime for one physical node type."""
        return self._add(
            "runtime", key or getattr(runtime, "runtime_key", None), runtime)

    def register_observer(self, factory: Any, *,
                          name: str | None = None) -> "ExtensionRegistry":
        """Register an execution observer factory, usually its class."""
        return self._add(
            "observer", name or getattr(factory, "name", None), factory)

    def register_logical_rule(self, rule: Any, *,
                              name: str | None = None) -> "ExtensionRegistry":
        return self._add(
            "logical_rule", name or getattr(rule, "name", None), rule)

    def register_source_reader(self, reader: Any, *,
                               source_type: str | None = None,
                               ) -> "ExtensionRegistry":
        """Register a reader for one remote table source type."""
        return self._add(
            "source_reader",
            source_type or getattr(reader, "source_type", None), reader)

    def register_physical_planner(self, planner: Any, *,
                                  name: str | None = None,
                                  ) -> "ExtensionRegistry":
        return self._add(
            "physical_planner", name or getattr(planner, "name", None),
            planner)

    def register_physical_rule(self, rule: Any, *,
                               name: str | None = None) -> "ExtensionRegistry":
        return self._add(
            "physical_rule", name or getattr(rule, "name", None), rule)

    def load_extension(
        self,
        module,
        *,
        local_python_sources: tuple[str, ...] | None = None,
        pip_packages: tuple[str, ...] = (),
    ) -> "ExtensionRegistry":
        """Import an extension module and call its entry point.

        The module's register_quail_extension(registry) registers what
        it provides. A compute worker imports the module by name and
        calls the same function, so the module's objects do not travel
        pickled.

        Args:
            module: The extension module, or its importable name.
            local_python_sources: Local packages the compute provider
                copies into the worker image. Defaults to the module's
                top-level package.
            pip_packages: Packages the compute provider installs.
        """
        if isinstance(module, ModuleType):
            loaded = module
            module = module.__name__
        else:
            loaded = importlib.import_module(module)
        if module in self.extension_modules:
            raise ValueError(f"duplicate extension module {module!r}")
        if local_python_sources is None:
            source = _top_level_source(module)
            local_python_sources = () if source is None else (source,)
        register = getattr(loaded, "register_quail_extension", None)
        if register is None:
            raise ValueError(
                f"extension module {module!r} has no "
                "register_quail_extension(registry) function"
            )
        recording = self.record
        self.record = False
        try:
            register(self)
        finally:
            self.record = recording
        self.extension_packages.append(ExtensionPackage(
            module=module,
            local_python_sources=tuple(local_python_sources),
            pip_packages=tuple(pip_packages),
        ))
        return self

    @property
    def extension_modules(self) -> tuple[str, ...]:
        """Return entry point extension modules in registration order."""
        return tuple(package.module for package in self.extension_packages)

    # ---- shipping to workers -----------------------------------------

    def manifest(self) -> ExtensionManifest:
        """Return what a compute worker needs to rebuild this registry."""
        registrations = []
        sources = []
        for kind, name, value in self.shipped:
            try:
                payload = pickle.dumps(value)
            except Exception as error:
                raise ValueError(
                    f"{kind} {name!r} cannot travel to a compute worker "
                    f"({error}); give it picklable state, or register it "
                    "from a module entry point with load_extension"
                ) from error
            registrations.append(Registration(kind, name, payload))
            source = _top_level_source(_module_of(value))
            if source is not None:
                sources.append(source)
        for package in self.extension_packages:
            sources.extend(package.local_python_sources)
        return ExtensionManifest(
            modules=self.extension_modules,
            registrations=tuple(registrations),
            local_python_sources=tuple(dict.fromkeys(sources)),
            pip_packages=tuple(dict.fromkeys(
                package
                for item in self.extension_packages
                for package in item.pip_packages
            )),
        )

    def restore(self, manifest: ExtensionManifest) -> "ExtensionRegistry":
        """Redo a manifest's registrations on this registry."""
        for module in manifest.modules:
            self.load_extension(module)
        for item in manifest.registrations:
            self._add(item.kind, item.name, pickle.loads(item.payload))
        return self

    # ---- lookup --------------------------------------------------------

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
