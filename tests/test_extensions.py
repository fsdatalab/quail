"""Extension registration and ordinary Python transport tests."""

import importlib
import pickle
import subprocess
import sys
from dataclasses import dataclass, replace
from types import ModuleType
from typing import ClassVar

import pytest

from modal._serialization import deserialize, serialize

from quail.builtins import built_in_registry
from quail.extensions import ExtensionRegistry
from quail.physical import DocumentInput, NodeCodec, PhysicalGraph, PortRef
from quail.planning import apply_physical_rules
from quail.runtime.runner import DocumentInputRuntime, ExecutionContext, GenericRunner


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
class CustomInput(DocumentInput):
    type_name: ClassVar[str] = "test.custom_input"
    runtime_key: ClassVar[str] = "test.custom_input_runtime"


def _module(monkeypatch, name, register):
    module = ModuleType(name)
    module.register_quail_extension = register
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _count(registry):
    graph = PhysicalGraph(
        (DocumentInput(node_id="input", alias="d", input_id="d", n_docs=1),),
        PortRef("input", "ids:d"),
    )
    rewritten, _ = apply_physical_rules(
        graph, tuple(registry.physical_rules.values()), None,
    )
    return rewritten.nodes[0].n_docs


def test_mixed_registration_order_preserves_rule_behavior(monkeypatch):
    def register(registry):
        assert "double" in registry.physical_rules
        registry.register_physical_rule(ChangeCount("add_three", offset=3))

    module = _module(monkeypatch, "test_order_extension", register)
    registry = (
        built_in_registry()
        .register_physical_rule(ChangeCount("double", factor=2))
        .load_extension(module)
        .register_physical_rule(ChangeCount("double_again", factor=2))
    )
    restored = deserialize(serialize(registry), None)
    assert list(restored.physical_rules) == list(registry.physical_rules)
    assert _count(restored) == _count(registry) == 10


def test_nested_extensions_are_registered_once(monkeypatch):
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
    restored = deserialize(serialize(registry), None)
    assert list(restored.physical_rules) == ["before", "inner", "after"]
    assert restored.extension_modules == (outer.__name__, inner.__name__)
    assert calls == ["outer", "inner"]


def test_failed_module_load_leaves_no_partial_registrations(monkeypatch):
    registry = built_in_registry().register_physical_rule(ChangeCount("existing"))

    def register(registry):
        registry.register_physical_rule(ChangeCount("added"))
        registry.register_physical_rule(ChangeCount("existing"))

    module = _module(monkeypatch, "test_failed_extension", register)
    with pytest.raises(ValueError, match="duplicate physical_rule"):
        registry.load_extension(module)
    assert list(registry.physical_rules) == ["existing"]
    assert registry.extension_modules == ()
    module.register_quail_extension = lambda registry: registry.register_physical_rule(
        ChangeCount("added"))
    registry.load_extension(module)
    assert list(registry.physical_rules) == ["existing", "added"]


def test_recursive_module_load_is_rejected_and_rolled_back(monkeypatch):
    def register(registry):
        registry.register_physical_rule(ChangeCount("added"))
        registry.load_extension("test_recursive_extension")

    module = _module(monkeypatch, "test_recursive_extension", register)
    registry = built_in_registry()
    with pytest.raises(ValueError, match="duplicate extension module"):
        registry.load_extension(module)
    assert not registry.physical_rules
    assert not registry.extension_modules


def test_lookup_tables_cannot_bypass_registration():
    registry = built_in_registry()
    with pytest.raises(TypeError):
        registry.physical_rules["lost"] = ChangeCount("lost")
    with pytest.raises(AttributeError):
        registry.physical_rules = {"lost": ChangeCount("lost")}
    with pytest.raises(TypeError):
        ExtensionRegistry(physical_rules={"lost": ChangeCount("lost")})
    registry.register_physical_rule(ChangeCount("saved"))
    assert list(deserialize(serialize(registry), None).physical_rules) == ["saved"]


@pytest.mark.parametrize("existing", ["codec", "runtime"])
def test_register_node_duplicate_leaves_registry_unchanged(existing):
    registry = built_in_registry()
    if existing == "codec":
        registry.register_codec(NodeCodec(CustomInput))
    else:
        registry.register_runtime(DocumentInputRuntime(), key=CustomInput.runtime_key)
    codecs, runtimes = dict(registry.codecs), dict(registry.runtimes)
    with pytest.raises(ValueError, match=f"duplicate {existing}"):
        registry.register_node(CustomInput, runtime=DocumentInputRuntime())
    assert dict(registry.codecs) == codecs
    assert dict(registry.runtimes) == runtimes


def test_register_node_round_trip_and_runtime():
    registry = built_in_registry().register_node(
        CustomInput, runtime=DocumentInputRuntime(),
    )
    restored = deserialize(serialize(registry), None)
    node = CustomInput(node_id="custom", alias="d", input_id="d", n_docs=2)
    codec = restored.codecs[node.type_name]
    assert codec.decode(codec.encode(node)) == node
    assert isinstance(restored.runtimes[node.runtime_key], DocumentInputRuntime)
    result = GenericRunner().run(
        PhysicalGraph((node,), PortRef("custom", "ids:d")),
        ExecutionContext(runtimes=restored.runtimes, sources={"d": [3, 7]}),
    )
    assert result.value == [3, 7]


def test_register_node_rejects_a_codec_for_another_class():
    registry = built_in_registry()
    with pytest.raises(ValueError, match="codec must describe"):
        registry.register_node(
            CustomInput, runtime=DocumentInputRuntime(), codec=NodeCodec(DocumentInput),
        )
    assert CustomInput.type_name not in registry.codecs
    assert CustomInput.runtime_key not in registry.runtimes


def test_explicit_empty_name_is_rejected():
    registry = built_in_registry()
    with pytest.raises(ValueError, match="nonempty string name"):
        registry.register_physical_rule(ChangeCount("valid"), name="")


def test_registry_crosses_a_fresh_process_without_replaying_modules(tmp_path, monkeypatch):
    source = tmp_path / "fresh_extension.py"
    source.write_text('''
from dataclasses import dataclass

calls = 0

@dataclass
class Rule:
    name: str
    amount: int

    def rewrite(self, graph, context):
        return None

def register_quail_extension(registry):
    global calls
    calls += 1
    first = registry.physical_rules["first"]
    registry.register_physical_rule(Rule("second", first.amount + 1))
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("fresh_extension")
    try:
        registry = (
            built_in_registry()
            .register_physical_rule(module.Rule("first", 7))
            .load_extension(module)
            .register_physical_rule(module.Rule("third", 9))
        )
        process = subprocess.run(
            [sys.executable, "-c", '''
import pickle
import sys
sys.path.insert(0, sys.argv[1])
from modal._serialization import deserialize
registry = deserialize(sys.stdin.buffer.read(), None)
import fresh_extension
values = [(rule.name, rule.amount) for rule in registry.physical_rules.values()]
sys.stdout.buffer.write(pickle.dumps((values, fresh_extension.calls)))
''', str(tmp_path)],
            input=serialize(registry), capture_output=True, timeout=30,
        )
        assert process.returncode == 0, process.stderr.decode()
        values, worker_calls = pickle.loads(process.stdout)
        assert values == [("first", 7), ("second", 8), ("third", 9)]
        assert module.calls == 1
        assert worker_calls == 0
    finally:
        sys.modules.pop("fresh_extension", None)
