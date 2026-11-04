"""Search lowers each logical alternative and keeps the cheapest refined plan."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.logical import columns_above_joins
from quail.planner import native
from quail.planner.context import PlanningContext
from quail.planner.optimize import search_plan


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


def alternative_plans(query):
    """The refined plan of each logical alternative, in search order."""
    context = context_for(query)
    registry = query.session.registry
    return [native.search_plan((alternative,), context, registry)
            for alternative in native.logical_alternatives(
                query._logical_input, registry, context)]


def test_search_compares_both_extraction_placements_and_keeps_the_cheaper(query):
    context = context_for(query)
    registry = query.session.registry
    original, lifted = native.logical_alternatives(
        query._logical_input, registry, context)
    assert not columns_above_joins(original.plan.root)
    assert "should_lift_column_operators" not in original.rules
    assert "should_lift_column_operators" in lifted.rules
    first, second = alternative_plans(query)
    chosen = search_plan(query._logical_input, context, registry)
    compared = chosen.settings["candidates"]
    assert [candidate["rewrites"] for candidate in compared] == [
        original.rules, lifted.rules]
    assert [candidate["estimated_seconds"] for candidate in compared] == [
        first.estimated_seconds, second.estimated_seconds]
    (selected,) = [index for index, candidate in enumerate(compared)
                   if candidate["selected"]]
    assert chosen.estimated_seconds == min(
        first.estimated_seconds, second.estimated_seconds)
    assert chosen.estimated_seconds == (first, second)[selected].estimated_seconds
    text = query.explain(verbose=True)
    assert "should_lift_column_operators" in text.split("  physical:")[0]
    assert "rewrite logical candidates" in text


def test_map_answer_tests_follow_maps_on_both_sides_of_join_placement(query,
                                                                      tmp_path):
    from helpers.oracle import Oracle, finish

    from quail.physical import AiMap, Filter
    from quail.physical.pipelines import build_pipelines

    # distinct documents, so each gets its own answer
    path = tmp_path / "leases.parquet"
    pq.write_table(pa.table({"id": range(20), "text": [
        f"Lease {index} ends in June." for index in range(20)]}), path)
    query.session.register("leases", DocumentProvider.from_parquet(
        str(path), id_col="id"))
    mapped = query.session.sql(
        "SELECT d.id, q.id, AI.MAP(d.text, 'Name the topic.', 16) AS topic "
        "FROM leases d JOIN questions q "
        "ON AI_FILTER(PROMPT('Does {0} answer {1}?', d.text, q.text)) "
        "WHERE HINT(topic = 'leases', {'selectivity': 0.25})")
    candidates = alternative_plans(mapped)
    assert len(candidates) == 2
    for candidate in candidates:
        (node,) = [node for node in candidate.nodes if isinstance(node, AiMap)]
        (test,) = [node for node in candidate.nodes if isinstance(node, Filter)]
        assert test.inputs[0].source.node_id == node.node_id
        assert test.selectivity == 0.25
        pipeline = next(members for members in build_pipelines(candidate.graph).values()
                        if node in members)
        assert pipeline[pipeline.index(node) + 1] == test
        mapped._plan = candidate
        oracle = Oracle(
            joins=lambda node, assignment: True,
            maps=lambda spec, document: "leases" if document == 1 else "other")
        assert finish(mapped, oracle).collect().to_pylist() == [
            {"d.id": 1, "q.id": 0, "topic": "leases"}]


