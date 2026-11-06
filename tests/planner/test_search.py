"""Search lowers each logical alternative and keeps the cheapest refined plan."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.logical import walk
from helpers.planning import placements, search
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider


@pytest.fixture()
def query(tmp_path):
    session = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER)
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


COLUMN_KINDS = {"SemanticColumn", "SemanticClassify", "SemanticExtract", "SemanticMap"}


def columns_above_joins(logical):
    """Return the one-table column operators that sit above a SemanticJoin."""
    return [node for node in walk(logical)
            if node.kind in COLUMN_KINDS and len(node.call.aliases()) == 1
            and any(below.kind == "SemanticJoin"
                    for below in walk(SimpleNamespace(root=node.input)))]


def alternative_plans(query):
    """The refined plan of each logical alternative, in search order."""
    fields = query.plan().pricing["fields"]
    return [search((alternative,), fields, query.session.tokenizer)
            for alternative in placements(query._logical_input)]


def test_search_compares_both_extraction_placements_and_keeps_the_cheaper(query):
    original, lifted = placements(query._logical_input)
    assert not columns_above_joins(original.plan)
    assert columns_above_joins(lifted.plan)
    assert "should_lift_column_operators" not in original.rules
    assert "should_lift_column_operators" in lifted.rules
    first, second = alternative_plans(query)
    chosen = query.plan()
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


