"""A region on several GPUs runs as stages, one partition per GPU."""

import json

import _quail_native
import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.quailb_data import standin_sets
from helpers.sessions import TOKENIZER

import quail
from bench.quailb import queries, register_tables
from quail.api.config import EngineConfig
from quail.api.execute import execute_query
from quail.execution.runner import compute_subgraph

TEXTS = ["No date", "June", "No date", "August"]

ANTI_JOIN = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('good: {0}', r.body))
      AND NOT EXISTS (SELECT 1 FROM aspects a
                      WHERE AI_FILTER(PROMPT('match {0} {1}', r.body, a.aspect)))
"""


def _oracle():
    return Oracle(
        filters=lambda alias, document, position: (document + position) % 3 != 0,
        joins=lambda node, assignment: sum(assignment.values()) % 2 == 0,
        labels=lambda spec, ids: spec.labels[sum(ids) % len(spec.labels)])


def _run(tmp_path, gpus, build):
    """Run a query on the oracle; return its rows and what it asked.

    What it asked is each filter's documents, each score or label's rows,
    and each join's pairs.
    """
    with quail.Session(
        EngineConfig(gpus=gpus, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER,
    ) as session:
        register_tables(session, tmp_path)
        oracle = _oracle()
        result = execute_query(build(session),
                               physical_executor=oracle_executor(oracle))
        pairs = [tuple(row.values())
                 for position, table in sorted(result.answer_tables["joins"].items())
                 for row in table.drop_columns([table.column_names[-1]])
                 .append_column("position", pa.array([position] * table.num_rows))
                 .to_pylist()]
        calls = [(name, tuple(row)) for name, rows in oracle.calls for row in rows]
        return (sorted(map(str, result.collect().to_pylist())), oracle.asked,
                calls, pairs)


@pytest.mark.parametrize("query_id", [
    "FEV-9", "IMDB-2", "IMDB-4", "IMDB-15", "CUAD-10", "REL-AGENT-3"])
def test_a_query_on_two_gpus_asks_each_question_once_and_returns_the_rows_of_one(
        tmp_path, query_id):
    standin_sets(tmp_path)
    one, two = (_run(tmp_path, gpus, lambda session: queries(session)[query_id][1]())
                for gpus in (1, 2))
    rows, asked, calls, pairs = two
    assert rows == one[0]
    assert sorted(asked) == sorted(one[1])
    assert sorted(calls) == sorted(one[2])
    assert len(set(pairs)) == len(pairs) and sorted(pairs) == sorted(one[3])


def test_an_anti_join_on_two_gpus_keeps_the_anchors_of_one(tmp_path):
    standin_sets(tmp_path)
    one, two = (_run(tmp_path, gpus, lambda session: session.sql(ANTI_JOIN))
                for gpus in (1, 2))
    assert two[0] == one[0] and sorted(two[3]) == sorted(one[3])


def test_a_limit_on_two_gpus_returns_as_many_rows_each_a_row_without_it(tmp_path):
    standin_sets(tmp_path)
    sql = "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"
    every = _run(tmp_path, 1, lambda session: session.sql(sql))[0]
    for gpus in (1, 2):
        rows = _run(tmp_path, gpus, lambda session: session.sql(f"{sql} LIMIT 3"))[0]
        assert len(rows) == 3 and set(rows) <= set(every)


@pytest.mark.parametrize("query_id,expected", [
    # the reviews' filters run with their join; aspects go whole
    ("IMDB-4", [("r", ["ai_filter:r", "ai_filter:r:0", "ai_join:r"],
                 [("scan:a", False), ("scan:r", True)])]),
    # each partner's filter is a stage; the barrier's stage hashes c2
    # again and reads the join answers whole
    ("FEV-9", [
        ("c1", ["ai_filter:c1"], [("scan:c1", True)]),
        ("e1", ["ai_filter:e1", "ai_join:e1", "ai_join:e1:predicate:1"],
         [("ai_filter:c1", False), ("ai_filter:c2", False), ("scan:e1", True)]),
        ("c2", ["ai_filter:c2"], [("scan:c2", True)]),
        ("e2", ["ai_filter:e2"], [("scan:e2", True)]),
        ("c2", ["barrier:c2", "ai_join:c2"],
         [("ai_filter:c1", False), ("ai_filter:c2", True), ("ai_filter:e2", False),
          ("ai_join:e1", False), ("ai_join:e1:predicate:1", False)])]),
])
def test_a_join_hashes_its_anchors_and_broadcasts_its_partners(
        tmp_path, query_id, expected):
    standin_sets(tmp_path)
    with quail.Session(
        EngineConfig(gpus=2, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER,
    ) as session:
        register_tables(session, tmp_path)
        graph = compute_subgraph(queries(session)[query_id][1]().plan().graph)
    stages = json.loads(_quail_native.region_stages(graph.to_json()))
    assert [(stage["key"], stage["nodes"],
             sorted({(spec["node_id"], spec["hashed"]) for spec in stage["inputs"]}))
            for stage in stages] == expected


@pytest.fixture
def contracts():
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=4),
        tokenizer=TOKENIZER,
    ) as session:
        session.register("contracts", quail.DocumentProvider.from_table(
            pa.table({"id": range(len(TEXTS)), "text": TEXTS}), id_col="id"))
        yield session


def test_copies_of_a_text_on_several_gpus_are_asked_once(contracts):
    query = contracts.sql(
        "SELECT c.id FROM contracts c "
        "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text))")
    oracle = Oracle(filters=lambda alias, document, position: True)
    result = execute_query(query, physical_executor=oracle_executor(oracle))
    # the two "No date" contracts are one text, asked on one GPU
    assert sorted(document for _, document in oracle.asked) == [0, 1, 3]
    assert result.to_rows() == [(0,), (1,), (2,), (3,)]
    assert len(result.report["memory_workers"]) == 4


def test_an_extraction_after_a_test_on_several_gpus_reads_the_survivors(contracts):
    query = contracts.sql(
        "SELECT c.id, AI.EXTRACT(c.text, {'ends': 'When does it end?'}) AS lease "
        "FROM contracts c WHERE lease.ends IS NOT NULL", order="as_written")
    oracle = Oracle(answers=[None if text == "No date" else text for text in TEXTS])
    result = execute_query(query, physical_executor=oracle_executor(oracle))
    assert result.collect().column("c.id").to_pylist() == [1, 3]
