"""Logical node and optimizer extension tests."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical import ColumnRef, LogicalPlan, Project, Scan
from quail.planner.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
)


@dataclass(frozen=True)
class TaggedInput:
    """Logical test node defined outside the built in node list."""

    input: object
    tag: str

    type_name: ClassVar[str] = "test.tagged_input"

    def children(self):
        return (self.input,)

    def expressions(self):
        return (self.tag,)

    def output_schema(self):
        return self.input.output_schema()

    def validate(self):
        if not self.tag:
            raise ValueError("tag cannot be empty")

    def with_children(self, children):
        if len(children) != 1:
            raise ValueError("TaggedInput needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions):
        if len(expressions) != 1:
            raise ValueError("TaggedInput needs one tag")
        return replace(self, tag=expressions[0])

    def explain_fields(self):
        return {"tag": self.tag}


class RenameTag:
    name = "rename_tag"

    def rewrite(self, node, context):
        if isinstance(node, TaggedInput) and node.tag == "old":
            return replace(node, tag="new")
        return None


def test_custom_logical_node_walks_and_rewrites_without_generic_changes():
    field = ColumnRef("r", "reviews", "review")
    scan = Scan("reviews", "r", "review")
    tagged = TaggedInput(scan, "old")
    plan = LogicalPlan(Project(tagged, (field,)))

    optimized, changed = apply_logical_rules(
        plan,
        (RenameTag(),),
        LogicalPlanningContext(catalog=None, engine_config=None),
    )

    assert [node.type_name for node in optimized.walk()] == [
        "quail.scan",
        "test.tagged_input",
        "quail.logical_project",
    ]
    assert optimized.root.input.tag == "new"
    assert changed == ("rename_tag",)
