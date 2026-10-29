"""AI.MAP: the operator surface, its prompt, and its plan."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish

import quail
from quail.catalog import DocumentProvider
from quail.logical import CompileError
from quail.physical import AiMap, decode_graph, encode_graph
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig

REVIEWS = pa.table({"id": [1, 2, 3],
                    "body": ["good film", "bad film", "long dull film"]})
SUMMARY = "Summarize the review in one sentence."
TOPIC = "Name the main topic in a few words."


@pytest.fixture()
def session():
    value = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                          tokenizer=str.split)
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
    assert list(head) == (turn[0] + quail.SHARED_PRE).split()
    assert list(tail) == (SUMMARY + turn[1] + "ANSWER:").split()


def test_the_builder_and_sql_plan_the_same_map(session):
    built = (session.docs("reviews").alias("r")
             .ai_map("r.body", SUMMARY, name="summary", max_answer_tokens=32)
             .select("r.id", "summary"))
    written = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary FROM reviews r")
    assert _maps(built.plan()) == _maps(written.plan())


def test_a_map_without_a_limit_writes_up_to_256_tokens(session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}') AS summary FROM reviews r")
    (node,) = _maps(query.plan())
    assert node.spec.max_answer_tokens == 256


def test_maps_over_one_table_share_a_pipeline_and_its_kv(session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 64) AS summary, "
        f"AI.MAP(r.body, '{TOPIC}', 64) AS topic FROM reviews r")
    plan = query.plan()
    first, second = _maps(plan)
    assert [member.node_id for member in build_pipelines(plan.graph)[
        first.node_id]] == [first.node_id, second.node_id]
    # the second map reads the document's KV the first one wrote
    assert second.spec.estimated_seconds < first.spec.estimated_seconds
    rows = finish(query, Oracle()).collect().to_pylist()
    assert rows[1] == {"r.id": 2, "summary": "summary 1", "topic": "topic 1"}
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph


def test_the_planner_reads_long_documents_once_for_their_maps(session):
    words = " ".join(f"w{i}" for i in range(2000))
    session.register("long", DocumentProvider.from_table(
        pa.table({"id": [1, 2], "body": [words, words + " end"]}), id_col="id"))
    one = session.sql(f"SELECT l.id, AI.MAP(l.body, '{SUMMARY}', 64) AS s "
                      "FROM long l")
    assert [node.attention for node in _maps(one.plan())] == ["unified"]
    two = session.sql(f"SELECT l.id, AI.MAP(l.body, '{SUMMARY}', 64) AS s, "
                      f"AI.MAP(l.body, '{TOPIC}', 64) AS t FROM long l")
    # two branches read each 2,000-token document; once is cheaper
    assert [node.attention for node in _maps(two.plan())] == ["tree", "tree"]


def test_a_map_after_a_filter_reads_only_its_survivors(session):
    query = session.sql(
        f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 64) AS summary FROM reviews r "
        f"WHERE AI_FILTER(PROMPT('Is this review negative?\\n\\n{{0}}', r.body)) "
        f"ORDER BY r.id LIMIT 1")
    plan = query.plan()
    (node,) = _maps(plan)
    (filter_id,) = [n.node_id for n in plan.graph.nodes if n.node_id != node.node_id
                    and n.type_name == "quail.ai_filter"]
    assert [member.node_id for member in build_pipelines(plan.graph)[
        filter_id]] == [filter_id, node.node_id]
    oracle = Oracle(filters=lambda alias, document, position: document != 0)
    assert finish(query, oracle).collect().to_pylist() == [
        {"r.id": 2, "summary": "summary 1"}]


def test_a_map_after_a_classification_shares_its_pipeline(session):
    query = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('What is the tone?\\n\\n{0}', r.body), "
        f"ARRAY['positive', 'negative']) AS tone, "
        f"AI.MAP(r.body, '{SUMMARY}', 64) AS summary FROM reviews r")
    plan = query.plan()
    (node,) = _maps(plan)
    (classify,) = [n for n in plan.graph.nodes if n.type_name == "quail.ai_classify"]
    assert [member.node_id for member in build_pipelines(plan.graph)[
        classify.node_id]] == [classify.node_id, node.node_id]


@pytest.mark.parametrize("sql, message", [
    (f"SELECT r.id FROM reviews r WHERE AI_MAP(r.body, '{SUMMARY}')",
     "not supported"),
    (f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}') FROM reviews r", "AS name"),
    ("SELECT r.id, AI.MAP(r.body, '') AS s FROM reviews r", "at least one word"),
    (f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 0) AS s FROM reviews r",
     "positive integer"),
    (f"SELECT r.id, AI.MAP(r.body, '{SUMMARY}', 'x') AS s FROM reviews r",
     "whole number"),
    ("SELECT r.id, AI.MAP(r.body, 12) AS s FROM reviews r", "instruction string"),
])
def test_a_malformed_map_is_refused(session, sql, message):
    with pytest.raises(CompileError, match=message):
        session.sql(sql).plan()


def test_the_planner_may_map_only_the_documents_a_join_keeps(session):
    from quail.logical import LogicalPlan
    from quail.planner.rules.tree_transforms import lift_maps

    session.register("films", DocumentProvider.from_table(
        pa.table({"id": [7, 8], "plot": ["a heist", "a romance"]}), id_col="id"))
    query = session.sql(
        f"SELECT r.id, f.id, AI.MAP(r.body, '{SUMMARY}', 32) AS summary "
        "FROM reviews r JOIN films f ON "
        "AI_FILTER(PROMPT('Is review {0} about film {1}?', r.body, f.plot))")
    root = query.logical.root
    # as written the map runs below the join, on every review; the
    # placement rule also offers it above the join, like a classification
    written = [type(node).__name__ for node in LogicalPlan(root).walk()]
    assert written.index("SemanticMap") < written.index("SemanticJoin")
    lifted = [type(node).__name__ for node in LogicalPlan(lift_maps(root)).walk()]
    assert lifted.index("SemanticJoin") < lifted.index("SemanticMap")
    LogicalPlan(lift_maps(root)).validate()


@pytest.mark.parametrize("form", ["call", "alias", "cte"])
@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
def test_a_plain_test_keeps_only_matching_map_answers(session, form, dialect):
    call = "AI.MAP(r.body, 'Name the topic.', 16)"
    select = f"SELECT r.id, {call} AS topic FROM reviews r"
    if form == "cte":
        sql = (f"WITH m AS ({select}) SELECT m.id, m.topic FROM m "
               "WHERE m.topic = 'topic 1'")
    else:
        operand = call if form == "call" else "topic"
        sql = f"{select} WHERE {operand} = 'topic 1'"
    query = session.sql(sql, dialect=dialect)
    plan = query.plan()
    (node,) = _maps(plan)
    (test,) = [node for node in plan.nodes if node.type_name == "quail.filter"]
    assert test.inputs[0].source.node_id == node.node_id
    assert [member.node_id for member in build_pipelines(plan.graph)[node.node_id]] == [
        node.node_id, test.node_id]
    rows = finish(query, Oracle()).collect().to_pylist()
    assert [list(row.values()) for row in rows] == [[2, "topic 1"]]


def test_a_map_only_used_by_a_test_is_computed_once(session):
    call = "AI.MAP(r.body, 'Name the topic.', 16)"
    query = session.sql(f"SELECT r.id FROM reviews r WHERE LENGTH({call}) > 3 "
                        f"AND {call} LIKE '% 1'")
    assert len(_maps(query.plan())) == 1
    assert finish(query, Oracle(maps=lambda spec, document: f"topic {document}"))\
        .collect().to_pylist() == [{"r.id": 2}]


@pytest.mark.parametrize("kind, call, operand", [
    ("label", "AI.CLASSIFY(PROMPT('Topic? {0}', r.body), ARRAY['a', 'b'])", "value"),
    ("extract", "AI.EXTRACT(r.body, {'topic': 'Topic?'})", "value.topic"),
    ("map", "AI.MAP(r.body, 'Topic?', 16)", "value"),
    ("features", "AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), pooling => 'last')",
     "value"),
])
def test_every_column_operator_supports_a_plain_result_test(session, kind, call,
                                                           operand):
    from quail.logical.nodes.column import SemanticColumn

    assert {node.call_kind for node in SemanticColumn.__subclasses__()} == {
        "label", "extract", "map", "features"}
    model = "qwen3.5-2b-sae-bf16" if kind == "features" else "qwen3-4b-fp8"
    with quail.Session(EngineConfig(model=model, device="h100-sxm"),
                       tokenizer=lambda text: list(text.encode())) as value:
        value.register("reviews", session.catalog.get("reviews"))
        query = value.sql(f"SELECT r.id, {call} AS value FROM reviews r "
                          f"WHERE HINT({operand} IS NOT NULL, {{'selectivity': 0.4}})")
        (test,) = query.logical.operators().column_filters["r"]
        assert test.call.kind == kind
        assert test.selectivity == 0.4
        plan = query.plan()
        (condition,) = [node for node in plan.nodes if node.type_name == "quail.filter"]
        producer = next(node for node in plan.nodes
                        if node.node_id == condition.inputs[0].source.node_id)
        assert producer.spec.name == test.name


def test_map_answer_selectivity_prices_the_following_column(session):
    query = session.sql(
        "SELECT r.id, AI.MAP(r.body, 'Name the topic.', 16) AS topic, "
        "AI.MAP(r.body, 'Summarize.', 32) AS summary FROM reviews r "
        "WHERE HINT(topic LIKE 'topic%', {'selectivity': 0.5}) "
        "AND HINT(topic = 'topic 1', {'selectivity': 0.25})")
    first, second = _maps(query.plan())
    assert (first.spec.name, second.spec.name) == ("topic", "summary")
    assert second.spec.expected_rows == 3 * 0.5 * 0.25
    tests = query.logical.operators().column_filters["r"]
    assert [test.position for test in tests] == [0, 1]
    assert finish(query, Oracle()).collect().to_pylist() == [
        {"r.id": 2, "topic": "topic 1", "summary": "summary 1"}]


def test_a_documents_extract_fields_and_maps_share_one_attention_choice(session):
    words = " ".join(f"w{i}" for i in range(2000))
    session.register("deeds", DocumentProvider.from_table(
        pa.table({"id": [1, 2], "body": [words, words + " end"]}), id_col="id"))
    query = session.sql(
        f"SELECT d.id, AI.MAP(d.body, '{SUMMARY}', 64) AS s, "
        "AI.EXTRACT(d.body, {'signer': 'Who signed?'}) AS deed FROM deeds d")
    nodes = [node for node in query.plan().graph.nodes
             if type(node).__name__ in ("AiExtract", "AiMap")]
    # an extraction field and a map are two branches reading each long
    # document; the planner reads it once for both
    assert len(nodes) == 2 and {node.attention for node in nodes} == {"tree"}
