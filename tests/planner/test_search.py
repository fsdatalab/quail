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
from quail.planner.optimizer.candidates import ModelRegion
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.rules import LogicalRule, PhysicalRule
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
        exact = all(hasattr(lengths, "_store")
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


def test_backend_only_lowers_the_given_alternative(query):
    context = context_for(query)
    backend = query.session.registry.backend("quail")
    (candidate,) = backend.plan(ModelRegion(query._logical_input), context)
    assert candidate.logical_plan == query._logical_input
    assert "retention" not in candidate.plan.settings
    assert "candidates" not in candidate.plan.settings
    assert len(search_candidates(
        query._logical_input, context, query.session.registry)) == 2


def test_prefix_sharing_is_priced_for_every_placement(query):
    query.token_inputs()
    context = context_for(query)
    registry = query.session.registry
    candidates = search_candidates(query._logical_input, context, registry)
    assert len(candidates) == 2
    for candidate in candidates:
        (raw,) = registry.backend("quail").plan(
            ModelRegion(candidate.logical_plan),
            replace(context, logical_plan=candidate.logical_plan))
        (extract,) = [node for node in candidate.plan.nodes
                     if isinstance(node, AiExtract)]
        assert extract.spec.share_prefixes
        assert "retention" in candidate.plan.settings
        assert candidate.estimated_seconds < raw.estimated_seconds
        # the sharing extraction is priced over its fresh tokens
        assert candidate.plan.estimates[extract.node_id]["seconds"] < (
            extract.spec.estimated_seconds)


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


def test_exact_tokenization_reconsiders_placement(query):
    rule = PlacementCost(exact=True)
    query.session.registry.register_physical_rule(rule)
    first = query.plan()
    assert first.settings["candidates"][1]["selected"]
    assert extracted_above_joins(query.logical.root)
    assert query.logical is first.logical_plan
    request = query._prepare_physical()
    final = query.plan()
    assert final.settings["candidates"][0]["selected"]
    assert not extracted_above_joins(query.logical.root)
    assert query.logical is final.logical_plan
    assert request.plan["settings"]["candidates"] == final.settings["candidates"]
    assert rule.seen == [False, True, False, True]
    assert final.settings["test_seen"] is False
    assert query._replanning_s > 0


def test_exact_tokenization_preserves_explicit_physical_edits(query):
    first = query.plan()
    edited = replace(first, settings={**first.settings, "user_edit": True})
    query._plan = edited
    query.wait_for_tokens()
    assert query.plan().settings["user_edit"] is True


def test_logical_alternatives_finish_before_physical_rules(query):
    context = context_for(query)
    events = []

    class RecordTrees(LogicalRule):
        name = "record_trees"

        def rewrites(self, root, context):
            events.append("logical")
            return [root]

    class RecordGraphs(PhysicalRule):
        name = "record_graphs"

        def rewrite(self, graph, context):
            events.append("physical")
            return None

    registry = query.session.registry
    registry.register_physical_rule(RecordGraphs())
    registry.register_logical_rule(RecordTrees())
    candidates = search_candidates(query._logical_input, context, registry)
    assert len(candidates) == 2
    first_physical = events.index("physical")
    assert first_physical > 0
    assert events[:first_physical] == ["logical"] * first_physical
    assert events[first_physical:] == ["physical", "physical"]


def test_a_filter_above_the_joins_thins_nothing_before_them(query):
    from quail.logical import classified_above_joins
    from quail.planner.search.statistics import (
        cached_statistics,
        expected_rows_after_filters,
    )

    session = query.session
    built = (session.docs("documents").alias("d")
             .ai_filter(quail.prompt("Signed? {0}", quail.col("d.text")),
                        selectivity=0.5)
             .ai_join(session.docs("questions").alias("q"), quail.prompt(
                 "Pair? {0} {1}", quail.col("d.text"), quail.col("q.text")),
                 anchor="d", selectivity=0.1)
             .ai_classify(quail.prompt("Kind? {0}", quail.col("d.text")),
                          ["a", "b"], name="kind")
             .label_in("kind", ["a"], selectivity=0.3)
             .select("d.id", "q.id"))
    context = context_for(built)
    expected = {}
    for candidate in search_candidates(built._logical_input, context,
                                       session.registry):
        logical = candidate.logical_plan
        statistics = cached_statistics(
            logical, {}, model=session.model, device=session.device,
            document_tokens=built._document_tokens, context=context)
        lifted = bool(classified_above_joins(logical.root))
        expected[lifted] = expected_rows_after_filters(logical, statistics)
    assert expected == {False: {"d": 3.0, "q": 1.0}, True: {"d": 10.0, "q": 1.0}}
