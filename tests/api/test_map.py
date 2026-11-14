"""AI.MAP: the operator surface, its prompt, and its plan.

Each transcript test holds its expected transcript: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.oracle import question_text
from helpers.planning import column_tests, operators_of, placements
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.common.expressions import CompileError
from quail.physical import AiMap

REVIEWS = pa.table({"id": ["r1", "r2", "r3"],
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


def _answer(kind, prompt, rows):
    """The model's answers: a map writes its prompt's first word and the row."""
    if kind == "filter":
        return rows != ("r1",)
    if kind == "label":
        return "negative"
    return f"{prompt.split()[0].lower()} {rows[0]}"


def _transcript(*queries, columns=(), dialects=None) -> str:
    """The transcripts of `queries`; those in `columns` show column types."""
    dialects = dialects or {}
    with Transcripts({"reviews": REVIEWS}, _answer) as runs:
        return "\n".join(runs.query(sql, dialect=dialects.get(sql, "snowflake"),
                                     columns=sql in columns, ordered="LIMIT" in sql)
                         for sql in queries)


def test_a_map_writes_one_answer_per_document():
    unnamed = [f"SELECT r.id, {function}(r.body, '{SUMMARY}') FROM reviews r"
               for function in ("AI.MAP", "ai_map")]
    assert_transcript(_transcript(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary FROM reviews r",
        # an unnamed map returns each answer under its SQL name
        *unnamed, columns=unnamed), """\
--- SELECT r.id, AI.MAP(r.body, 'Summarize the review in one sentence.', 32) AS summary
    FROM reviews r ---
plan
  summary = ai_map(r.body, 'Summarize the review in one sentence.', Int64(32))
asked 3 of 3
  'Summarize the review in one sentence.': r1 'summarize r1', r2 'summarize r2', r3
      'summarize r3'
rows 3
  r1, 'summarize r1'
  r2, 'summarize r2'
  r3, 'summarize r3'

--- SELECT r.id, AI.MAP(r.body, 'Summarize the review in one sentence.') FROM reviews r
    ---
plan
  AI.MAP(r.body,'Summarize the review in one sentence.') = ai_map(r.body, 'Summarize the
      review in one sentence.')
asked 3 of 3
  'Summarize the review in one sentence.': r1 'summarize r1', r2 'summarize r2', r3
      'summarize r3'
columns r.id string, AI.MAP(r.body,Utf8("Summarize the review in one sentence.")) string
rows 3
  r1, 'summarize r1'
  r2, 'summarize r2'
  r3, 'summarize r3'

--- SELECT r.id, ai_map(r.body, 'Summarize the review in one sentence.') FROM reviews r
    ---
plan
  ai_map(r.body,'Summarize the review in one sentence.') = ai_map(r.body, 'Summarize the
      review in one sentence.')
asked 3 of 3
  'Summarize the review in one sentence.': r1 'summarize r1', r2 'summarize r2', r3
      'summarize r3'
columns r.id string, ai_map(r.body,Utf8("Summarize the review in one sentence.")) string
rows 3
  r1, 'summarize r1'
  r2, 'summarize r2'
  r3, 'summarize r3'
""")


def test_a_map_writes_up_to_its_answer_token_limit_after_the_document(session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary FROM reviews r")
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


def test_a_map_writes_only_the_documents_the_filter_before_it_keeps():
    assert_transcript(_transcript(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 64) AS summary FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Negative? {0}', r.body)) ORDER BY r.id LIMIT 1"), """\
--- SELECT r.id, AI.MAP(r.body, 'Summarize the review in one sentence.', 64) AS summary
    FROM reviews r WHERE AI_FILTER(PROMPT('Negative? {0}', r.body)) ORDER BY r.id LIMIT
    1 ---
plan
  summary = ai_map(r.body, 'Summarize the review in one sentence.', Int64(64))
  __quail_answer_1 = ai_filter('Negative? {0}', r.body), filter: __quail_answer_1,
      returns: []
asked 5 of 5
  'Negative? {0}': r1 FALSE, r2 TRUE, r3 TRUE
  'Summarize the review in one sentence.': r2 'summarize r2', r3 'summarize r3'
rows 1, in order
  r2, 'summarize r2'
""")


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
             "an integer from 1 to 32768"),
            # a limit the planner would size its answer rows by
            (f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 4294967296) AS s "
             "FROM reviews r", "an integer from 1 to 32768")):
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


def test_a_plain_test_keeps_only_matching_map_answers():
    call = "AI.MAP(r.body, 'Name the topic.', 16)"
    select = f"SELECT r.id, {call} AS topic FROM reviews r"
    bigquery = f"{select} WHERE {call} = 'name r1'"
    assert_transcript(_transcript(
        bigquery,
        f"WITH m AS ({select}) SELECT m.id, m.topic FROM m "
        "WHERE m.topic = 'name r1'",
        # a map only tests read is computed once
        f"SELECT r.id FROM reviews r WHERE LENGTH({call}) > 3 "
        f"AND {call} LIKE '% r1'", dialects={bigquery: "bq"}), """\
--- SELECT r.id, AI.MAP(r.body, 'Name the topic.', 16) AS topic FROM reviews r WHERE
    AI.MAP(r.body, 'Name the topic.', 16) = 'name r1' ---
plan
  topic = ai_map(r.body, 'Name the topic.', Int64(16)), filter: topic = 'name r1'
asked 3 of 3
  'Name the topic.': r1 'name r1', r2 'name r2', r3 'name r3'
rows 1
  r1, 'name r1'

--- WITH m AS (SELECT r.id, AI.MAP(r.body, 'Name the topic.', 16) AS topic FROM reviews
    r) SELECT m.id, m.topic FROM m WHERE m.topic = 'name r1' ---
plan
  topic = ai_map(r.body, 'Name the topic.', Int64(16)), filter: topic = 'name r1'
asked 3 of 3
  'Name the topic.': r1 'name r1', r2 'name r2', r3 'name r3'
rows 1
  r1, 'name r1'

--- SELECT r.id FROM reviews r WHERE LENGTH(AI.MAP(r.body, 'Name the topic.', 16)) > 3
    AND AI.MAP(r.body, 'Name the topic.', 16) LIKE '% r1' ---
plan
  __quail_column_1 = ai_map(r.body, 'Name the topic.', Int64(16)), filter:
      character_length(__quail_column_1) > Int32(3) AND __quail_column_1 LIKE '% r1'
asked 3 of 3
  'Name the topic.': r1 'name r1', r2 'name r2', r3 'name r3'
rows 1
  r1
""")


def test_a_plain_test_of_a_map_follows_it_in_its_pipeline(session):
    call = "AI.MAP(r.body, 'Name the topic.', 16)"
    select = f"SELECT r.id, {call} AS topic FROM reviews r"
    for form, dialect, sql in (
            ("call", "bq", f"{select} WHERE {call} = 'topic 1'"),
            ("cte", "snowflake", f"WITH m AS ({select}) SELECT m.id, m.topic FROM m "
                                 "WHERE m.topic = 'topic 1'")):
        plan = session.sql(sql, dialect=dialect).plan()
        (node,) = _maps(plan)
        (test,) = [node for node in plan.nodes if node.type_name == "quail.filter"]
        assert test.inputs[0].source.node_id == node.node_id, form
        assert [member.node_id
                for member in plan.graph.pipelines()[node.node_id]] == [
            node.node_id, test.node_id], form


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
