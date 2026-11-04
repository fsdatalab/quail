"""Extension registration, ordering, and rollback tests."""

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import ClassVar

import pytest

from quail.builtins import built_in_registry
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.runtimes import ScanRuntime
from quail.extensions import PhysicalRule
from quail.physical import NodeCodec, PhysicalGraph, PortRef, Scan


@dataclass(frozen=True)
class CustomInput(Scan):
    type_name: ClassVar[str] = "test.custom_input"
    runtime_key: ClassVar[str] = "test.custom_input_runtime"


def _module(monkeypatch, name, register):
    module = ModuleType(name)
    module.register_quail_extension = register
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _added(registry):
    built_in = list(built_in_registry().rules)
    return [name for name in registry.rules if name not in built_in]


def test_extension_order_nesting_and_rollback(monkeypatch):
    def register(registry):
        assert "double" in registry.rules
        registry.register_physical_rule(PhysicalRule("add_three"))

    registry = (
        built_in_registry()
        .register_physical_rule(PhysicalRule("double"))
        .load_extension(_module(monkeypatch, "test_order_extension", register))
        .register_physical_rule(PhysicalRule("double_again"))
    )
    assert _added(registry) == ["double", "add_three", "double_again"]
    with pytest.raises(TypeError, match="registered by name"):
        registry.register_physical_rule(object())

    calls = []

    def register_inner(registry):
        calls.append("inner")
        registry.register_physical_rule(PhysicalRule("inner"))

    inner = _module(monkeypatch, "test_inner_extension", register_inner)

    def register_outer(registry):
        calls.append("outer")
        registry.register_physical_rule(PhysicalRule("before"))
        registry.load_extension(inner)
        registry.register_physical_rule(PhysicalRule("after"))

    outer = _module(monkeypatch, "test_outer_extension", register_outer)
    registry = built_in_registry().load_extension(outer)
    built_in = list(built_in_registry().rules)
    assert [name for name in registry.rules
            if name not in built_in] == ["before", "inner", "after"]
    assert registry.extension_modules == (outer.__name__, inner.__name__)
    assert calls == ["outer", "inner"]

    # a failed registration rolls the registry back
    registry = built_in_registry().register_physical_rule(PhysicalRule("existing"))

    def register(registry):
        registry.register_physical_rule(PhysicalRule("added"))
        registry.register_physical_rule(PhysicalRule("existing"))

    module = _module(monkeypatch, "test_failed_extension", register)
    with pytest.raises(ValueError, match="duplicate physical_rule"):
        registry.load_extension(module)
    built_in = list(built_in_registry().rules)
    assert [name for name in registry.rules
            if name not in built_in] == ["existing"]
    assert registry.extension_modules == ()

    module.register_quail_extension = (
        lambda registry: registry.register_physical_rule(PhysicalRule("added")))
    registry.load_extension(module)
    assert [name for name in registry.rules
            if name not in built_in] == ["existing", "added"]

    def recursive(registry):
        registry.register_physical_rule(PhysicalRule("added"))
        registry.load_extension("test_recursive_extension")

    module = _module(monkeypatch, "test_recursive_extension", recursive)
    registry = built_in_registry()
    with pytest.raises(ValueError, match="duplicate extension module"):
        registry.load_extension(module)
    assert list(registry.rules) == built_in
    assert not registry.extension_modules


def test_node_registration_validation_and_execution():
    for existing in ["codec", "runtime"]:
        registry = built_in_registry()
        if existing == "codec":
            registry.register_codec(NodeCodec(CustomInput))
        else:
            registry.register_runtime(ScanRuntime(), key=CustomInput.runtime_key)
        codecs, runtimes = dict(registry.codecs), dict(registry.runtimes)
        with pytest.raises(ValueError, match=f"duplicate {existing}"):
            registry.register_node(CustomInput, runtime=ScanRuntime())
        assert dict(registry.codecs) == codecs
        assert dict(registry.runtimes) == runtimes

    registry = built_in_registry().register_node(CustomInput, runtime=ScanRuntime())
    node = CustomInput(node_id="custom", alias="d", input_id="d", num_documents=2)
    codec = registry.codecs[node.type_name]
    assert codec.decode(codec.encode(node)) == node
    assert isinstance(registry.runtimes[node.runtime_key], ScanRuntime)
    result = GenericRunner().run(
        PhysicalGraph((node,), PortRef("custom", "rows:d")),
        ExecutionContext(runtimes=registry.runtimes, sources={"d": [3, 7]}),
    )
    assert result.value.ids["d"].to_pylist() == [3, 7]

    registry = built_in_registry()
    with pytest.raises(ValueError, match="codec must describe"):
        registry.register_node(
            CustomInput, runtime=ScanRuntime(), codec=NodeCodec(Scan),
        )
    assert CustomInput.type_name not in registry.codecs
    assert CustomInput.runtime_key not in registry.runtimes
