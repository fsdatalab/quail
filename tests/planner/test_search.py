"""Search compares physical candidates after all registered rules."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.catalog import DocumentProvider
from quail.logical import classified_above_joins, extracted_above_joins
from quail.physical import AiClassify, AiExtract, AiJoin, Filter, Foreign, PhysicalGraph
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
        model=query.session.model, device=query.session.device, gpu_count=1,
        document_tokens=query._doc_tokens, backend="quail",
        tokenizer=query.session.tokenizer)


def _upstream_ids(graph, node):
    """Return every node supplying an input to node, directly or indirectly."""
    pending = [port.source.node_id for port in node.inputs]
    found = set()
    while pending:
        node_id = pending.pop()
        if node_id not in found:
            found.add(node_id)
            pending.extend(port.source.node_id
                           for port in graph.node(node_id).inputs)
    return found


@pytest.mark.parametrize("kind", ["classify", "extract"])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("callback_kind", ["per_batch", "barrier"])
def test_map_placement_preserves_callback_inputs(query, kind, position, callback_kind):
    session = query.session

    def keep_last(rows):
        return rows["d"].ids["d"].slice(19)

    builder = session.docs("documents").alias("d")
    names = []
    for index in range(3):
        if index == position:
            builder.apply(keep_last, columns=[quail.col("d.id")], kind=callback_kind)
        if index == 2:
            break
        name = f"answer{index}"
        if kind == "classify":
            builder.ai_classify(
                quail.prompt(f"Which {index}? {{0}}", quail.col("d.text")),
                ["a", "b"], name=name)
            builder.label_in(name, ["a"])
        else:
            builder.ai_extract("d.text", {"end": f"When does {index} end?"}, name=name)
            builder.where(quail.col(f"{name}.end") != "")
        names.append(name)
    other = (session.docs("questions").alias("q")
             .ai_classify(quail.prompt("Which? {0}", quail.col("q.text")),
                          ["a", "b"], name="question_kind"))
    built = (builder.ai_join(other, quail.prompt(
        "Does {0} answer {1}?", quail.col("d.text"), quail.col("q.text")))
        .select("d.id", *names, "question_kind"))
    candidates = search_candidates(built._logical_input, context_for(built),
                                   session.registry)
    other_lifted = False
    for candidate in candidates:
        assert not isinstance(candidate.plan, Refusal)
        graph = candidate.plan.graph
        (callback,) = [node for node in graph.nodes if isinstance(node, Foreign)]

        before = _upstream_ids(graph, callback)
        maps = [node for node in graph.nodes
                if isinstance(node, (AiClassify, AiExtract))]
        for index, name in enumerate(names):
            (mapped,) = [node for node in maps if node.spec.name.startswith(name)]
            assert (mapped.node_id in before) == (index < position)
            if index >= position:
                assert callback.node_id in _upstream_ids(graph, mapped)
        other_lifted |= "q" in classified_above_joins(candidate.logical_plan.root)
    assert other_lifted


@pytest.mark.parametrize("semantics", ["exists", "anti"])
def test_map_placement_keeps_inner_label_filters_before_existence_tests(
        query, semantics):
    session = query.session
    outer = (session.docs("documents").alias("d")
             .ai_classify(quail.prompt("Which? {0}", quail.col("d.text")),
                          ["a", "b"], name="document_kind")
             .label_in("document_kind", ["a"]))
    inner = (session.docs("questions").alias("q")
             .ai_classify(quail.prompt("Which? {0}", quail.col("q.text")),
                          ["a", "b"], name="question_kind")
             .label_in("question_kind", ["a"]))
    built = (outer.ai_join(inner, quail.prompt(
        "Does {0} answer {1}?", quail.col("d.text"), quail.col("q.text")),
        semantics=semantics).select("d.id", "document_kind"))
    candidates = search_candidates(built._logical_input, context_for(built),
                                   session.registry)
    outer_lifted = False
    for candidate in candidates:
        assert not isinstance(candidate.plan, Refusal)
        graph = candidate.plan.graph
        (join,) = [node for node in graph.nodes if isinstance(node, AiJoin)]
        maps = {node.spec.name: node for node in graph.nodes
                if isinstance(node, AiClassify)}
        assert maps["question_kind"].node_id in _upstream_ids(graph, join)
        (condition,) = [node for node in graph.nodes
                        if isinstance(node, Filter) and node.column == "question_kind"]
        assert condition.node_id in _upstream_ids(graph, join)
        outer_lifted |= join.node_id in _upstream_ids(graph, maps["document_kind"])
    assert outer_lifted


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


def test_filter_pricing_reuses_shared_prefix_work(query, monkeypatch):
    from types import SimpleNamespace

    from quail.physical import AiFilter
    from quail.planner.graph import pricing

    query.plan()
    query.wait_for_tokens()
    context = context_for(query)
    lengths = context.document_tokens["d"]
    statistics = SimpleNamespace(
        stats={"d": SimpleNamespace(mean_doc_tokens=sum(lengths) / len(lengths))})
    pricer = pricing.PlanPricer(context, statistics, {}, {})
    node = AiFilter(node_id="filter:d", alias="d", preamble_tokens=2,
                    question_tokens=5, expected_docs=20, share_prefixes=True)
    cases = [replace(node, expected_docs=live, question_tokens=question)
             for live in (0, 0.5, 10, 20, 40) for question in (5, 10)]
    expected = []
    for case in cases:
        context.memo.clear()
        expected.append(pricer._filter_work(case, False))
    context.memo.clear()
    calls = []
    original = pricing.ask

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing, "ask", counted)
    assert [pricer._filter_work(case, False) for case in cases] == expected
    assert 0 < len(calls) <= 2 * len(lengths)


@pytest.mark.parametrize("kind", ["per_batch", "barrier"])
@pytest.mark.parametrize("retain", [False, True])
def test_filter_pricing_uses_the_pipeline_and_retained_kv(query, kind, retain):
    from quail.physical import AiFilter

    def preserve(rows):
        return rows["d"]

    built = (query.session.docs("documents").alias("d")
             .ai_filter(quail.prompt("First? {0}", quail.col("d.text")),
                        selectivity=1.0)
             .apply(preserve, columns=[quail.col("d.id")], kind=kind)
             .ai_filter(quail.prompt("Second? {0}", quail.col("d.text")),
                        selectivity=1.0)
             .select("d.id", order="as_written"))
    plan = built.plan()
    first, second = [node for node in plan.nodes if isinstance(node, AiFilter)]
    graph = PhysicalGraph(tuple(replace(node, keep_kv=retain) if node is first else node
                                for node in plan.nodes), plan.root)
    prices = plan.estimator._priced(graph)
    work = prices[second.node_id][0].work
    prefix = (second.preamble_tokens
              + plan.estimator.statistics.stats["d"].mean_doc_tokens)
    assert work.tokens == pytest.approx(second.expected_docs * (
        second.question_tokens + (prefix if kind == "barrier" and not retain else 0)))
    assert first.expected_docs == second.expected_docs == 20


@pytest.mark.parametrize("selectivity", [0.0, 0.25, 1.0])
def test_join_search_prices_later_filters_over_join_survivors(query, selectivity):
    from quail.planner.search.operator_order import choose_step_orders, filter_works

    def preserve(rows):
        return rows["d"]

    built = (query.session.docs("documents").alias("d")
             .ai_join(query.session.docs("questions").alias("q"), quail.prompt(
                 "Pair? {0} {1}", quail.col("d.text"), quail.col("q.text")),
                 anchor="d", selectivity=selectivity)
             .apply_table(preserve, columns=[quail.col("d.id")])
             .ai_filter(quail.prompt("Keep? {0}", quail.col("d.text")))
             .select("d.id", "q.id", order="as_written"))
    plan = built.plan()
    context, stats = plan.estimator.context, plan.estimator.statistics
    logical = plan.logical_plan
    steps = choose_step_orders(logical.operators(), stats,
                               rule="as_written", context=context)
    work = filter_works(logical, stats, context.model, steps)["d"]
    fresh = (stats.pre + stats.stats["d"].mean_doc_tokens
             + context.lowered(logical.operators().filters["d"][0].prompt).tail_tokens)
    assert work.tokens == pytest.approx(20 * selectivity * fresh)
