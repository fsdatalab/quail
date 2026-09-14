"""Logical node and optimizer extension tests."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical import ColumnRef, LogicalPlan, Project, Scan
from quail.planner.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
    rewrite_bottom_up,
)

CONTEXT = LogicalPlanningContext(catalog=None, engine_config=None)


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
    """Rename one tag value to another, one node at a time."""

    def __init__(self, old, new):
        self.name = f"rename_{old}_{new}"
        self.old, self.new = old, new

    def rewrite(self, root, context):
        def rename(node):
            if isinstance(node, TaggedInput) and node.tag == self.old:
                return replace(node, tag=self.new)
            return node

        rewritten = rewrite_bottom_up(root, rename)
        return None if rewritten is root else rewritten


def _tagged_plan(tag):
    field = ColumnRef("r", "reviews", "review")
    scan = Scan("reviews", "r", "review")
    return LogicalPlan(Project(TaggedInput(scan, tag), (field,)))


def test_custom_logical_node_walks_and_rewrites_without_generic_changes():
    plan = _tagged_plan("old")

    optimized, changed = apply_logical_rules(
        plan, (RenameTag("old", "new"),), CONTEXT)

    assert [node.type_name for node in optimized.walk()] == [
        "quail.scan",
        "test.tagged_input",
        "quail.logical_project",
    ]
    assert optimized.root.input.tag == "new"
    assert changed == ("rename_old_new",)


def test_rules_repeat_until_a_round_changes_nothing():
    # The second rule only matches after the first has run, and it is
    # listed first, so a single round would miss it.
    rules = (RenameTag("mid", "new"), RenameTag("old", "mid"))

    optimized, changed = apply_logical_rules(_tagged_plan("old"), rules, CONTEXT)

    assert optimized.root.input.tag == "new"
    assert changed == ("rename_old_mid", "rename_mid_new")


def test_rules_that_never_settle_stop_at_the_pass_cap():
    # Each round ends on a different tag than it started, so the rounds
    # would repeat forever: a -> b -> c | c -> a | a -> b -> c | ...
    rules = (RenameTag("a", "b"), RenameTag("c", "a"), RenameTag("b", "c"))

    optimized, changed = apply_logical_rules(
        _tagged_plan("a"), rules, CONTEXT, max_passes=3)

    assert optimized.root.input.tag == "c"
    assert len(changed) == 5


def test_unchanged_plan_keeps_its_root_object():
    plan = _tagged_plan("new")

    optimized, changed = apply_logical_rules(
        plan, (RenameTag("old", "new"),), CONTEXT)

    assert optimized.root is plan.root
    assert changed == ()
