"""A region on several GPUs runs as stages, one partition per GPU."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.quailb_data import standin_sets
from helpers.sessions import TOKENIZER
from test_kv_reads import named

import quail
from bench.quailb import queries, register_tables
from quail.api.config import EngineConfig
from quail.api.execute import execute_query

TEXTS = ["No date", "June", "No date", "August"]

ANTI_JOIN = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('good: {0}', r.body))
      AND NOT EXISTS (SELECT 1 FROM aspects a
                      WHERE AI_FILTER(PROMPT('match {0} {1}', r.body, a.aspect)))
"""


# the stage lines of FEV-9's explain text on two GPUs
FEV9_STAGES_TEXT = [
    "  Stage 1: ai_filter:c1; split by c1 id: scan:c1",
    "  Stage 2: ai_filter:e1 -> ai_join:e1 -> ai_join:e1:predicate:1; "
    "split by e1 id: scan:e1; sent whole to every GPU: ai_filter:c1 rows:c1, "
    "ai_filter:c2 rows:c2; waits for stages 1, 3",
    "  Stage 3: ai_filter:c2; split by c2 id: scan:c2",
    "  Stage 4: ai_filter:e2; split by e2 id: scan:e2",
    "  Stage 5: barrier:c2 -> ai_join:c2; split by c2 id: ai_filter:c2 rows:c2; "
    "sent whole to every GPU: ai_join:e1 join_answers:0, "
    "ai_join:e1:predicate:1 join_answers:1, ai_filter:c1 rows:c1, "
    "ai_join:e1:predicate:1 rows:e1, ai_filter:e2 rows:e2; "
    "waits for stages 1, 2, 3, 4",
]


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


def check_two_gpus_ask_each_question_once_and_return_the_rows_of_one(
        tmp_path, query_id):
    one, two = (_run(tmp_path, gpus, lambda session: queries(session)[query_id][1]())
                for gpus in (1, 2))
    rows, asked, calls, pairs = two
    assert rows == one[0]
    assert sorted(asked) == sorted(one[1])
    assert sorted(calls) == sorted(one[2])
    assert len(set(pairs)) == len(pairs) and sorted(pairs) == sorted(one[3])


def test_a_query_on_two_gpus_asks_each_question_once_and_returns_the_rows_of_one(
        tmp_path):
    standin_sets(tmp_path)
    for query_id in ["FEV-9", "IMDB-2", "IMDB-4", "CUAD-10", "REL-AGENT-3"]:
        named(query_id,
              check_two_gpus_ask_each_question_once_and_return_the_rows_of_one,
              tmp_path, query_id)

    # an anti-join keeps the anchors and pairs of one GPU
    one, two = (_run(tmp_path, gpus, lambda session: session.sql(ANTI_JOIN))
                for gpus in (1, 2))
    assert two[0] == one[0] and sorted(two[3]) == sorted(one[3])


@pytest.mark.xfail(strict=True, reason=(
    "on two GPUs the stages run a classification of a join's pairs apart "
    "from its join, and the executor classifies pairs only in the join's "
    "chain, on the anchors' KV"))
def test_a_classification_of_joined_rows_on_two_gpus_returns_the_rows_of_one(
        tmp_path):
    standin_sets(tmp_path)
    check_two_gpus_ask_each_question_once_and_return_the_rows_of_one(
        tmp_path, "IMDB-15")


def test_a_limit_on_two_gpus_returns_as_many_rows_each_a_row_without_it(tmp_path):
    standin_sets(tmp_path)
    sql = "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"
    every = _run(tmp_path, 1, lambda session: session.sql(sql))[0]
    for gpus in (1, 2):
        rows = _run(tmp_path, gpus, lambda session: session.sql(f"{sql} LIMIT 3"))[0]
        assert len(rows) == 3 and set(rows) <= set(every)


def _planned(tmp_path, query_id):
    """Return a QUAIL-B query's plan on two GPUs and its explain text."""
    standin_sets(tmp_path)
    with quail.Session(
        EngineConfig(gpus=2, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER,
    ) as session:
        register_tables(session, tmp_path)
        query = queries(session)[query_id][1]()
        return query.plan(), query.explain()


def test_a_plan_on_two_gpus_hashes_anchors_broadcasts_partners_and_explains_its_stages(
        tmp_path):
    # IMDB-4: the reviews' filters run with their join; aspects go whole
    stages = _planned(tmp_path, "IMDB-4")[0].settings["stages"]
    assert [(stage["key"], stage["nodes"],
             sorted({(spec["node_id"], spec["hashed"]) for spec in stage["inputs"]}))
            for stage in stages] == [
        ("r", ["ai_filter:r", "ai_filter:r:0", "ai_join:r"],
         [("scan:a", False), ("scan:r", True)])]

    # FEV-9: each partner's filter is a stage; the barrier's stage hashes
    # c2 again and reads the join answers whole
    text = _planned(tmp_path, "FEV-9")[1]
    assert [line for line in text.splitlines()
            if line.startswith("  Stage ")] == FEV9_STAGES_TEXT


@pytest.fixture
def contracts():
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=4),
        tokenizer=TOKENIZER,
    ) as session:
        session.register("contracts", quail.DocumentProvider.from_table(
            pa.table({"id": range(len(TEXTS)), "text": TEXTS}), id_col="id"))
        yield session


def test_several_gpus_ask_copies_of_a_text_once_and_extract_from_the_survivors(
        contracts):
    query = contracts.sql(
        "SELECT c.id FROM contracts c "
        "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text))")
    oracle = Oracle(filters=lambda alias, document, position: True)
    result = execute_query(query, physical_executor=oracle_executor(oracle))
    # the two "No date" contracts are one text, asked on one GPU
    assert sorted(document for _, document in oracle.asked) == [0, 1, 3]
    assert result.to_rows() == [(0,), (1,), (2,), (3,)]
    assert len(result.report["memory_workers"]) == 4

    # an extraction after a test on its answer reads the survivors
    query = contracts.sql(
        "SELECT c.id, AI.EXTRACT(c.text, {'ends': 'When does it end?'}) AS lease "
        "FROM contracts c "
        "WHERE AI.EXTRACT(c.text, {'ends': 'When does it end?'})['ends'] IS NOT NULL",
        order="as_written")
    oracle = Oracle(answers=[None if text == "No date" else text for text in TEXTS])
    result = execute_query(query, physical_executor=oracle_executor(oracle))
    assert result.collect().column("c.id").to_pylist() == [1, 3]
