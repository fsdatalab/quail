"""Extension registration, ordering, and rollback tests."""

import sys
from types import ModuleType

import pyarrow as pa
import pytest

from quail.builtins import built_in_registry
from quail.execution.metrics import NodeResult
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.extensions import PhysicalRule
from quail.physical import (
    ExtensionNode,
    GraphValidationError,
    OutputPort,
    PhysicalGraph,
    PortRef,
    ValueType,
)

CUSTOM_INPUT = "test.custom_input"


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


def test_an_extension_node_round_trips_and_runs_with_its_registered_runtime():
    class SourceRuntime:
        def execute(self, node, inputs, context):
            return NodeResult({f"rows:{node.alias}": Rows(
                {node.alias: pa.array(context.sources[node.input_id])})})

    registry = built_in_registry().register_runtime(SourceRuntime(), key=CUSTOM_INPUT)
    with pytest.raises(ValueError, match="duplicate runtime"):
        registry.register_runtime(SourceRuntime(), key=CUSTOM_INPUT)
    node = ExtensionNode(
        CUSTOM_INPUT, "custom",
        outputs=(OutputPort("rows:d", ValueType.ROWS, schema=("d",)),),
        alias="d", input_id="d")
    graph = PhysicalGraph((node,), PortRef("custom", "rows:d"))
    assert PhysicalGraph.from_json(graph.to_json()) == graph
    result = GenericRunner().run(
        graph, ExecutionContext(runtimes=registry.runtimes, sources={"d": [3, 7]}))
    assert result.value.ids["d"].to_pylist() == [3, 7]
    with pytest.raises(GraphValidationError, match="unregistered runtimes"):
        graph.validate(runtime_keys=set(built_in_registry().runtimes))
    with pytest.raises(ValueError, match="one of Quail's node types"):
        ExtensionNode("quail.scan", "custom")
