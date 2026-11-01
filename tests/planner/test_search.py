"""Search compares physical candidates after all registered rules."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.catalog import DocumentProvider
from quail.logical import extracted_above_joins
from quail.physical import AiExtract, PhysicalGraph
from quail.planner.optimize import search_candidates, search_plan
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.rules import PhysicalRule
from quail.planner.plan import EngineConfig, Refusal
from quail.planner.validation import ClassifyRefusedError


@pytest.fixture()
def query(tmp_path):
    session = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    shared = "The following agreement is made between the parties. " * 4
    for name, texts in (("documents", [shared + "The lease ends in June."] * 20),
                        ("questions", ["leases"])):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table({"id": range(len(texts)), "text": texts}), path)
        session.register(name, DocumentProvider.from_parquet(str(path), id_col="id"))
    value = session.sql(
        "SELECT d.id, q.id, AI.EXTRACT(d.text, {'end': 'When does it end?'}) AS x "
        "FROM documents d JOIN questions q "
        "ON AI_FILTER(PROMPT('Does {0} answer {1}?', d.text, q.text))")
    yield value
    session.close()


def context_for(query):
    query.plan()
    return PlanningContext(
        model=query.session.model, device=query.session.device, gpus=1,
        document_tokens=query._document_tokens, backend="quail",
        tokenizer=query.session.tokenizer)


class PlacementCost(PhysicalRule):
    name = "test_placement_cost"

    def __init__(self, *, refuse=False, exact=False):
        self.refuse = refuse
        self.exact = exact
        self.seen = []

    def rewrite(self, graph, context):
        lifted = bool(extracted_above_joins(context.logical_plan.root))
        assert "retention" in context.settings
        assert "test_seen" not in context.settings
        context.settings["test_seen"] = lifted
        self.seen.append(lifted)
        if lifted and self.refuse:
            raise ClassifyRefusedError("test candidate cannot run", 2, 1)
        exact = all(lengths.store is not None
                    for lengths in context.document_tokens.values())
        if self.exact and not exact:
            return None
        return PhysicalGraph(tuple(
            replace(node, spec=replace(node.spec,
                                      estimated_seconds=100.0 if lifted else 0.0))
            if isinstance(node, AiExtract) else node for node in graph.nodes
        ), graph.root)


def test_graph_refinement_can_reverse_placement_choice(query):
    context = context_for(query)
    registry = query.session.registry
    original, lifted = search_candidates(query._logical_input, context, registry)
    assert lifted.estimated_seconds < original.estimated_seconds
    rule = PlacementCost()
    registry.register_physical_rule(rule)
    chosen = search_plan(query._logical_input, context, registry)
    assert rule.seen == [False, True]
    assert context.settings == {}
    original, lifted = chosen.settings["candidates"]
    assert original["selected"] and not lifted["selected"]
    assert original["estimated_seconds"] < lifted["estimated_seconds"]
    assert "should_lift_maps" in lifted["rewrites"]
    assert "test_seen" not in context.settings
    text = query.explain(verbose=True)
    assert "should_lift_maps" in text.split("  physical:")[0]
    assert "rewrite logical candidates" in text
    assert "lowest estimated cost after graph refinement" in text


def test_refused_candidate_does_not_hide_a_runnable_plan(query):
    context = context_for(query)
    query.session.registry.register_physical_rule(PlacementCost(refuse=True))
    plan = search_plan(query._logical_input, context, query.session.registry)
    assert not isinstance(plan, Refusal)
    original, lifted = plan.settings["candidates"]
    assert original["selected"]
    assert lifted["estimated_seconds"] is None
    assert lifted["refusal"] is not None






def test_logical_alternatives_retain_original_and_deduplicate(query):
    from quail.planner.rules.logical import ShouldLiftMaps

    context = context_for(query)
    registry = query.session.registry

    class CheckCandidateContext(ShouldLiftMaps):
        def rewrites(self, root, context):
            assert context.logical_plan.root is root
            assert context.settings == {}
            context.settings["tree_rule_setting"] = True
            return super().rewrites(root, context)

    registry.register_logical_rule(CheckCandidateContext(), name="duplicate_lift")
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_logical_rule(ShouldLiftMaps())
    original, lifted = search_candidates(query._logical_input, context, registry)
    assert not extracted_above_joins(original.logical_plan.root)
    assert "should_lift_maps" not in original.rewrites
    assert "should_lift_maps" in lifted.rewrites
    assert context.settings == {}
    assert "tree_rule_setting" not in original.plan.settings
    assert "tree_rule_setting" not in lifted.plan.settings










