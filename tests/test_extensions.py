"""Extension registration, ordering, and rollback tests."""

import sys
from dataclasses import dataclass, replace
from types import ModuleType
from typing import ClassVar

import pytest

from quail.builtins import built_in_registry
from quail.physical import NodeCodec, PhysicalGraph, PortRef, Scan
from quail.planning import apply_physical_rules
from quail.runtime.runner import ExecutionContext, GenericRunner, ScanRuntime


@dataclass(frozen=True)
class ChangeCount:
    name: str
    factor: int = 1
    offset: int = 0

    def rewrite(self, graph, context):
        return PhysicalGraph(tuple(
            replace(node, n_docs=node.n_docs * self.factor + self.offset)
            for node in graph.nodes
        ), graph.root)


@dataclass(frozen=True)
class CustomInput(Scan):
    type_name: ClassVar[str] = "test.custom_input"
    runtime_key: ClassVar[str] = "test.custom_input_runtime"


def _module(monkeypatch, name, register):
    module = ModuleType(name)
    module.register_quail_extension = register
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _count(registry):
    graph = PhysicalGraph(
        (Scan(node_id="input", alias="d", input_id="d", n_docs=1),),
        PortRef("input", "ids:d"),
    )
    rewritten, _ = apply_physical_rules(
        graph, tuple(registry.physical_rules.values()), None,
    )
    return rewritten.nodes[0].n_docs


def test_extension_order_and_nested_registration(monkeypatch):
    with monkeypatch.context() as patch:
        def register(registry):
            assert "double" in registry.physical_rules
            registry.register_physical_rule(ChangeCount("add_three", offset=3))

        module = _module(patch, "test_order_extension", register)
        registry = (
            built_in_registry()
            .register_physical_rule(ChangeCount("double", factor=2))
            .load_extension(module)
            .register_physical_rule(ChangeCount("double_again", factor=2))
        )
        assert _count(registry) == 10

    with monkeypatch.context() as patch:
        calls = []

        def register_inner(registry):
            calls.append("inner")
            registry.register_physical_rule(ChangeCount("inner"))

        inner = _module(patch, "test_inner_extension", register_inner)

        def register_outer(registry):
            calls.append("outer")
            registry.register_physical_rule(ChangeCount("before"))
            registry.load_extension(inner)
            registry.register_physical_rule(ChangeCount("after"))

        outer = _module(patch, "test_outer_extension", register_outer)
        registry = built_in_registry().load_extension(outer)
        assert list(registry.physical_rules) == ["before", "inner", "after"]
        assert registry.extension_modules == (outer.__name__, inner.__name__)
        assert calls == ["outer", "inner"]


def test_failed_extension_registration_rolls_back(monkeypatch):
    with monkeypatch.context() as patch:
        registry = built_in_registry().register_physical_rule(ChangeCount("existing"))

        def register(registry):
            registry.register_physical_rule(ChangeCount("added"))
            registry.register_physical_rule(ChangeCount("existing"))

        module = _module(patch, "test_failed_extension", register)
        with pytest.raises(ValueError, match="duplicate physical_rule"):
            registry.load_extension(module)
        assert list(registry.physical_rules) == ["existing"]
        assert registry.extension_modules == ()

        def retry(registry):
            registry.register_physical_rule(ChangeCount("added"))

        module.register_quail_extension = retry
        registry.load_extension(module)
        assert list(registry.physical_rules) == ["existing", "added"]

    with monkeypatch.context() as patch:
        def register(registry):
            registry.register_physical_rule(ChangeCount("added"))
            registry.load_extension("test_recursive_extension")

        module = _module(patch, "test_recursive_extension", register)
        registry = built_in_registry()
        with pytest.raises(ValueError, match="duplicate extension module"):
            registry.load_extension(module)
        assert not registry.physical_rules
        assert not registry.extension_modules


def test_node_registration_validation_and_execution():
    for existing in ["codec", "runtime"]:
        registry = built_in_registry()
        if existing == "codec":
            registry.register_codec(NodeCodec(CustomInput))
        else:
            registry.register_runtime(
                ScanRuntime(), key=CustomInput.runtime_key)
        codecs, runtimes = dict(registry.codecs), dict(registry.runtimes)
        with pytest.raises(ValueError, match=f"duplicate {existing}"):
            registry.register_node(CustomInput, runtime=ScanRuntime())
        assert dict(registry.codecs) == codecs
        assert dict(registry.runtimes) == runtimes

    registry = built_in_registry().register_node(
        CustomInput, runtime=ScanRuntime(),
    )
    node = CustomInput(node_id="custom", alias="d", input_id="d", n_docs=2)
    codec = registry.codecs[node.type_name]
    assert codec.decode(codec.encode(node)) == node
    assert isinstance(registry.runtimes[node.runtime_key], ScanRuntime)
    result = GenericRunner().run(
        PhysicalGraph((node,), PortRef("custom", "ids:d")),
        ExecutionContext(runtimes=registry.runtimes, sources={"d": [3, 7]}),
    )
    assert result.value == [3, 7]

    registry = built_in_registry()
    with pytest.raises(ValueError, match="codec must describe"):
        registry.register_node(
            CustomInput, runtime=ScanRuntime(), codec=NodeCodec(Scan),
        )
    assert CustomInput.type_name not in registry.codecs
    assert CustomInput.runtime_key not in registry.runtimes
