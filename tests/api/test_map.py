"""AI.MAP: the operator surface, its prompt, and its plan."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, question_text
from helpers.planning import column_tests, operators_of, placements
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.common.expressions import CompileError
from quail.physical import AiMap

REVIEWS = pa.table({"id": [1, 2, 3],
                    "body": ["good film", "bad film", "long dull film"]})
SUMMARY = "Summarize the review in one sentence."


@pytest.fixture()
def session():
    value = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                          tokenizer=TOKENIZER)
    value.register("reviews", DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


def _maps(plan):
    return [node for node in plan.graph.nodes if isinstance(node, AiMap)]


def test_a_map_writes_one_answer_per_document(session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary FROM reviews r")
    rows = finish(query, Oracle()).collect().to_pylist()
    assert rows == [{"r.id": 1, "summary": "summary 0"},
                    {"r.id": 2, "summary": "summary 1"},
                    {"r.id": 3, "summary": "summary 2"}]
    (node,) = _maps(query.plan())
    assert node.spec.max_answer_tokens == 32
    # the prompt is the document, then the instruction and the answer cue
    head, tail = node.spec.prompt_token_parts
    turn = session.model.turn
    assert question_text(head) == turn[0] + quail.SHARED_PRE
    # the tail opens with the separator after the document
    assert question_text(tail).endswith(SUMMARY + turn[1] + "ANSWER:")
    # without a limit a map writes up to 256 tokens
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}') AS summary FROM reviews r")
    (node,) = _maps(query.plan())
    assert node.spec.max_answer_tokens == 256
    # an unnamed map returns each answer under its SQL name
    for function in ("AI.MAP", "ai_map"):
        query = session.sql(
            f"SELECT r.id, {function}(r.body, '{SUMMARY}') FROM reviews r")
        oracle = Oracle(maps=lambda _spec, document: f"summary {document}")
        table = finish(query, oracle).collect()
        rows = zip(table.column(0).to_pylist(), table.column(1).to_pylist())
        assert sorted(rows) == [
            (1, "summary 0"), (2, "summary 1"), (3, "summary 2")], function
        assert "map" in table.column_names[1].lower(), function
        assert not table.column_names[1].startswith("__quail_"), function


def test_a_map_shares_the_pipeline_of_the_filter_or_classification_before_it(
        session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 64) AS summary FROM reviews r "
        f"WHERE AI_FILTER(PROMPT('Is this review negative?\\n\\n{{0}}', r.body)) "
        f"ORDER BY r.id LIMIT 1")
    plan = query.plan()
    (node,) = _maps(plan)
    (filter_id,) = [n.node_id for n in plan.graph.nodes if n.node_id != node.node_id
                    and n.type_name == "quail.ai_filter"]
    assert [member.node_id for member in plan.graph.pipelines()[
        filter_id]] == [filter_id, node.node_id]
    oracle = Oracle(filters=lambda alias, document, position: document != 0)
    assert finish(query, oracle).collect().to_pylist() == [
        {"r.id": 2, "summary": "summary 1"}]
    query = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('What is the tone?\\n\\n{0}', r.body), "
        f"ARRAY['positive', 'negative']) AS tone, "
        f"AI.MAP(r.body, '{SUMMARY}', 64) AS summary FROM reviews r")
    plan = query.plan()
    (node,) = _maps(plan)
    (classify,) = [n for n in plan.graph.nodes if n.type_name == "quail.ai_classify"]
    assert [member.node_id for member in plan.graph.pipelines()[
        classify.node_id]] == [classify.node_id, node.node_id]


def test_a_malformed_map_is_refused(session):
    for sql, message in (
            (f"SELECT r.id FROM reviews r WHERE AI_MAP(r.body, '{SUMMARY}')",
             "not supported"),
            (f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 0) AS s FROM reviews r",
             "positive integer")):
        with pytest.raises(CompileError, match=message):
            session.sql(sql).plan()


def test_the_planner_may_map_only_the_documents_a_join_keeps(session):
    session.register("films", DocumentProvider.from_table(
        pa.table({"id": [7, 8], "plot": ["a heist", "a romance"]}), id_col="id"))
    query = session.sql(
        f"SELECT r.id, f.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary "
        "FROM reviews r JOIN films f ON "
        "AI_FILTER(PROMPT('Is review {0} about film {1}?', r.body, f.plot))")
    # as written the map runs below the join, on every review; the
    # placement rule also offers it above the join, like a classification
    written, lifted = placements(query.operators)
    assert [column.above_joins for column in written.found.maps] == [False]
    assert lifted.rules == ("should_lift_column_operators",)
    assert [column.above_joins for column in lifted.found.maps] == [True]


def test_a_plain_test_keeps_only_matching_map_answers(session):
    call = "AI.MAP(r.body, 'Name the topic.', 16)"
    select = f"SELECT r.id, {call} AS topic FROM reviews r"
    for form, dialect, sql in (
            ("call", "bq", f"{select} WHERE {call} = 'topic 1'"),
            ("cte", "snowflake", f"WITH m AS ({select}) SELECT m.id, m.topic FROM m "
                                 "WHERE m.topic = 'topic 1'")):
        query = session.sql(sql, dialect=dialect)
        plan = query.plan()
        (node,) = _maps(plan)
        (test,) = [node for node in plan.nodes if node.type_name == "quail.filter"]
        assert test.inputs[0].source.node_id == node.node_id, form
        assert [member.node_id
                for member in plan.graph.pipelines()[node.node_id]] == [
            node.node_id, test.node_id], form
        rows = finish(query, Oracle()).collect().to_pylist()
        assert [list(row.values()) for row in rows] == [[2, "topic 1"]], form
    # a map only tests read is computed once
    query = session.sql(f"SELECT r.id FROM reviews r WHERE LENGTH({call}) > 3 "
                        f"AND {call} LIKE '% 1'")
    assert len(_maps(query.plan())) == 1
    assert finish(query, Oracle(maps=lambda spec, document: f"topic {document}"))\
        .collect().to_pylist() == [{"r.id": 2}]


COLUMN_OPERATORS = [
    ("label", "AI.CLASSIFY(PROMPT('Topic? {0}', r.body), ARRAY['a', 'b'])", ""),
    ("extract", "AI.EXTRACT(r.body, {'topic': 'Topic?'})", "['topic']"),
    ("map", "AI.MAP(r.body, 'Topic?', 16)", ""),
    ("features", "AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'})",
     ""),
]


def test_every_column_operator_supports_a_plain_result_test(session):
    for kind, call, field in COLUMN_OPERATORS:
        model = "qwen3.5-2b-sae-bf16" if kind == "features" else "qwen3-4b-fp8"
        with quail.Session(EngineConfig(model=model, device="h100-sxm"),
                           tokenizer=TOKENIZER) as value:
            value.register("reviews", session.catalog.get("reviews"))
            query = value.sql(f"SELECT r.id, {call} AS value FROM reviews r "
                              f"WHERE HINT({call}{field} IS NOT NULL, "
                              "{'selectivity': 0.4})")
            (test,) = column_tests(operators_of(query))
            # the test follows the column operator whose answer it reads
            assert test.call.kind == kind
            assert test.selectivity == 0.4, kind
            plan = query.plan()
            (condition,) = [
                node for node in plan.nodes if node.type_name == "quail.filter"]
            producer = next(node for node in plan.nodes
                            if node.node_id == condition.inputs[0].source.node_id)
            assert producer.spec.name == test.name, kind
