"""Logical node and optimizer extension tests."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical import ColumnRef, LogicalPlan, Project, Scan
from quail.planner.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
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
        pass

    def with_children(self, children):
        (child,) = children
        return replace(self, input=child)

    def with_expressions(self, expressions):
        (tag,) = expressions
        return replace(self, tag=tag)

    def explain_fields(self):
        return {"tag": self.tag}


class RenameTag:
    """Rename one tag value to another, one node at a time."""

    def __init__(self, old, new):
        self.name = f"rename_{old}_{new}"
        self.old, self.new = old, new

    def rewrite(self, root, context):
        def visit(node):
            children = tuple(visit(child) for child in node.children())
            if children != node.children():
                node = node.with_children(children)
            if isinstance(node, TaggedInput) and node.tag == self.old:
                return replace(node, tag=self.new)
            return node

        rewritten = visit(root)
        return None if rewritten is root else rewritten


def _tagged_plan(tag):
    field = ColumnRef("r", "reviews", "review")
    scan = Scan("reviews", "r", "review")
    return LogicalPlan(Project(TaggedInput(scan, tag), (field,)))


REWRITES = [
    ("old", [("old", "new")], "new", ("rename_old_new",)),
    # the second rule only matches after the first has run, and it is
    # listed first, so a single round would miss it
    ("old", [("mid", "new"), ("old", "mid")], "new",
     ("rename_old_mid", "rename_mid_new")),
    ("new", [("old", "new")], "new", ()),
]


def test_custom_logical_node_rewrites_until_a_round_changes_nothing():
    for start, rules, tag, changed in REWRITES:
        case = f"{start} {rules}"
        plan = _tagged_plan(start)
        optimized, applied = apply_logical_rules(
            plan, tuple(RenameTag(old, new) for old, new in rules), CONTEXT)

        assert [node.type_name for node in optimized.walk()] == [
            "quail.scan", "test.tagged_input", "quail.logical_project"], case
        assert optimized.root.input.tag == tag, case
        assert applied == changed, case
        assert (optimized.root is plan.root) == (not changed), case


def test_rules_that_never_settle_stop_at_the_pass_cap():
    # each round ends on a different tag than it started, so the rounds
    # would repeat forever: a -> b -> c | c -> a | a -> b -> c | ...
    rules = (RenameTag("a", "b"), RenameTag("c", "a"), RenameTag("b", "c"))

    optimized, changed = apply_logical_rules(
        _tagged_plan("a"), rules, CONTEXT, max_passes=3)

    assert optimized.root.input.tag == "c"
    assert len(changed) == 5


class RecordContext:
    """Keep the context a session hands its logical rules."""

    name = "record_context"

    def __init__(self):
        self.seen = []

    def rewrite(self, root, context):
        self.seen.append(context)
        return None


def test_session_rules_read_statistics_from_the_context():
    import pyarrow as pa

    import quail

    recorder = RecordContext()
    session = quail.Session(
        quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=str.split)
    session.registry.register_logical_rule(recorder)
    session.register("docs", quail.DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one", "two three"]}),
        id_col="id"))
    query = session.sql(
        "SELECT d.id FROM docs d WHERE AI_FILTER(PROMPT('ok {0}', d.body))")
    query.plan()
    query.wait_for_tokens()

    # the rules ran before the physical planner, on document lengths
    # estimated while the background thread tokenized the table
    assert len(recorder.seen) >= 1
    context = recorder.seen[0]
    assert context.catalog is session.catalog
    assert context.engine_config is session.config
    assert context.model is session.model
    assert context.device is session.device
    assert (context.gpu_count, context.backend) == (1, "quail")
    assert list(context.document_tokens["d"]) == [1, 2]
    assert context.tokenizer is session.tokenizer
    assert dict(context.pair_fractions) == {}
    assert context.physical_context().document_tokens is context.document_tokens
    session.close()
