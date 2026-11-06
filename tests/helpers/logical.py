"""A region's logical plan, as Rust writes it in JSON, read for tests."""

import json
from types import SimpleNamespace

from quail.physical import decode_plan_value

NODE_PREFIX = "quail.logical."


def _read(value):
    """A logical node as a namespace of its fields; any other value decoded."""
    tag = value.get("type", "") if isinstance(value, dict) else ""
    if isinstance(tag, str) and tag.startswith(NODE_PREFIX):
        fields = {key: _read(item) for key, item in value.items() if key != "type"}
        return SimpleNamespace(kind=tag.rsplit(".", 1)[-1], **fields)
    if isinstance(value, list):
        return tuple(_read(item) for item in value)
    return decode_plan_value(json.dumps(value))


def read_plan(logical: str) -> SimpleNamespace:
    """Return the plan: a namespace whose root is its root node.

    Each node is a namespace with its kind, such as "Scan", and its fields;
    expressions are Quail's expression types.
    """
    return _read(json.loads(logical))


def walk(logical: str | SimpleNamespace) -> list[SimpleNamespace]:
    """Return every node of the plan, each child before its parent.

    Args:
        logical: The plan as JSON, or as `read_plan` returns it.
    """
    found = []

    def visit(node):
        children = ((node.left, node.right) if node.kind == "Join"
                    else (node.input,) if hasattr(node, "input") else ())
        for child in children:
            visit(child)
        found.append(node)

    plan = read_plan(logical) if isinstance(logical, str) else logical
    visit(plan.root)
    return found


def kinds(logical: str) -> list[str]:
    """Return the kind of every node, each child before its parent."""
    return [node.kind for node in walk(logical)]


def nodes(logical: str, kind: str) -> list[SimpleNamespace]:
    """Return the plan's nodes of one kind, each child before its parent."""
    return [node for node in walk(logical) if node.kind == kind]


def join_outer_input(join: SimpleNamespace) -> SimpleNamespace:
    """Return the tree a SemanticJoin extends, before its new tables."""
    node = join.input
    while node.kind == "Join":
        node = node.left
    return node


def join_conditions(join: SimpleNamespace) -> tuple:
    """Return the equalities under one SemanticJoin, in written order."""
    conditions = []

    def visit(node):
        if node.kind == "Join":
            visit(node.left)
            conditions.extend(node.on)

    visit(join.input)
    return tuple(conditions)
