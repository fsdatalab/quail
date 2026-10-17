"""Extension registration, ordering, and rollback tests."""

import sys
from dataclasses import dataclass, replace
from types import ModuleType
from typing import ClassVar

import pytest

from quail.builtins import built_in_registry
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.runtimes import ScanRuntime
from quail.physical import NodeCodec, PhysicalGraph, PortRef, Scan
from quail.planner.optimizer.graph_refinement import rewrite_graph
from quail.planner.optimizer.rules import LogicalRule, PhysicalRule


@dataclass(frozen=True)
class ChangeCount(PhysicalRule):
    name: str
    factor: int = 1
    offset: int = 0

    def rewrite(self, graph, context):
        return PhysicalGraph(tuple(
            replace(node, num_documents=node.num_documents * self.factor + self.offset)
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
        (Scan(node_id="input", alias="d", input_id="d", num_documents=1),),
        PortRef("input", "ids:d"),
    )
    rewritten, _ = rewrite_graph(
        graph, tuple(registry.rules.values()), None)
    return rewritten.nodes[0].num_documents


def test_extension_order_nesting_and_rollback(monkeypatch):
    def register(registry):
        assert "double" in registry.rules
        registry.register_physical_rule(ChangeCount("add_three", offset=3))

    registry = (
        built_in_registry()
        .register_physical_rule(ChangeCount("double", factor=2))
        .load_extension(_module(monkeypatch, "test_order_extension", register))
        .register_physical_rule(ChangeCount("double_again", factor=2))
    )
    assert _count(registry) == 10

    calls = []

    def register_inner(registry):
        calls.append("inner")
        registry.register_physical_rule(ChangeCount("inner"))

    inner = _module(monkeypatch, "test_inner_extension", register_inner)

    def register_outer(registry):
        calls.append("outer")
        registry.register_physical_rule(ChangeCount("before"))
        registry.load_extension(inner)
        registry.register_physical_rule(ChangeCount("after"))

    outer = _module(monkeypatch, "test_outer_extension", register_outer)
    registry = built_in_registry().load_extension(outer)
    built_in = list(built_in_registry().rules)
    assert [name for name in registry.rules
            if name not in built_in] == ["before", "inner", "after"]
    assert registry.extension_modules == (outer.__name__, inner.__name__)
    assert calls == ["outer", "inner"]

    # a failed registration rolls the registry back
    registry = built_in_registry().register_physical_rule(ChangeCount("existing"))

    def register(registry):
        registry.register_physical_rule(ChangeCount("added"))
        registry.register_physical_rule(ChangeCount("existing"))

    module = _module(monkeypatch, "test_failed_extension", register)
    with pytest.raises(ValueError, match="duplicate physical_rule"):
        registry.load_extension(module)
    built_in = list(built_in_registry().rules)
    assert [name for name in registry.rules
            if name not in built_in] == ["existing"]
    assert registry.extension_modules == ()

    module.register_quail_extension = (
        lambda registry: registry.register_physical_rule(ChangeCount("added")))
    registry.load_extension(module)
    assert [name for name in registry.rules
            if name not in built_in] == ["existing", "added"]

    def recursive(registry):
        registry.register_physical_rule(ChangeCount("added"))
        registry.load_extension("test_recursive_extension")

    module = _module(monkeypatch, "test_recursive_extension", recursive)
    registry = built_in_registry()
    with pytest.raises(ValueError, match="duplicate extension module"):
        registry.load_extension(module)
    assert list(registry.rules) == built_in
    assert not registry.extension_modules


class KeepTree(LogicalRule):
    def __init__(self, name):
        self.name = name

    def rewrites(self, root, context):
        raise AssertionError("logical rules must not run on physical graphs")


def test_rule_types_preserve_registration_order_within_each_stage():
    registry = built_in_registry()
    registry.register_physical_rule(ChangeCount("double", factor=2))
    registry.register_logical_rule(KeepTree("keep_tree"))
    registry.register_physical_rule(ChangeCount("add_three", offset=3))
    registry.register_physical_rule(ChangeCount("double_again", factor=2))

    assert list(registry.rules)[-4:] == [
        "double", "keep_tree", "add_three", "double_again"]
    assert list(registry.physical_rules)[-3:] == [
        "double", "add_three", "double_again"]
    assert list(registry.logical_rules)[-1:] == ["keep_tree"]
    assert _count(registry) == 10

    before = dict(registry.rules)
    with pytest.raises(ValueError, match="duplicate optimizer rule"):
        registry.register_logical_rule(KeepTree("double"))
    assert dict(registry.rules) == before


def test_rule_registration_validates_base_class():
    registry = built_in_registry()
    before = dict(registry.rules)
    with pytest.raises(TypeError, match="needs a LogicalRule subclass"):
        registry.register_logical_rule(ChangeCount("wrong_type"))
    with pytest.raises(TypeError, match="needs a PhysicalRule subclass"):
        registry.register_physical_rule(KeepTree("wrong_type"))
    with pytest.raises(TypeError, match="needs a PhysicalRule subclass"):
        registry.register_physical_rule(object())
    assert dict(registry.rules) == before


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
        PhysicalGraph((node,), PortRef("custom", "ids:d")),
        ExecutionContext(runtimes=registry.runtimes, sources={"d": [3, 7]}),
    )
    assert result.value.to_pylist() == [3, 7]

    registry = built_in_registry()
    with pytest.raises(ValueError, match="codec must describe"):
        registry.register_node(
            CustomInput, runtime=ScanRuntime(), codec=NodeCodec(Scan),
        )
    assert CustomInput.type_name not in registry.codecs
    assert CustomInput.runtime_key not in registry.runtimes
