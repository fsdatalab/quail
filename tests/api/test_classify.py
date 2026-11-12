"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results."""


import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import (
    letter_tokens,
)
from helpers.oracle import Oracle, finish

import quail
from quail.catalog import DocumentProvider
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    RegularPredicate,
)
from quail.logical.prompts import join_anchor_note
from quail.physical import (
    AiClassify,
    AiJoin,
    Filter,
    decode_graph,
    encode_graph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig, Refusal

# one token per word: refund=1, request=2, status=3, shipping=4


def _bytes(text):
    return list(text.encode())


@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": [7, 9], "body": ["where is my refund", "love it"],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    value.register("documents",
                   DocumentProvider.from_parquet(str(path), id_col="id"))
    yield value
    value.close()


TOPIC = ("AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
         "ARRAY['refund', 'shipping', 'praise']")
KIND = ("AI.CLASSIFY(PROMPT('What kind of {0}?', d.body), "
        "ARRAY['complaint', 'question']")
CALLS = {"topic": TOPIC, "kind": KIND}


def _label_test(name, accepted):
    """A WHERE test that the label `name` is in `accepted`, at selectivity 0.5."""
    labels = ", ".join(f"'{label}'" for label in accepted)
    return f"{CALLS[name]}, {{'selectivity': 0.5}}) IN ({labels})"


def _topic(session, *, accepted=("refund", "shipping")):
    where = f" WHERE {_label_test('topic', accepted)}" if accepted else ""
    return session.sql(f"SELECT d.id, {TOPIC}) AS topic FROM documents d{where}")


def _labels(labels):
    """An oracle labeling document i with labels[i]; a "kind" gets its first label."""
    return Oracle(labels=lambda spec, ids: (spec.labels[0] if spec.name == "kind"
                                            else labels[ids[0]]),
                  probabilities=True)


def test_classify_plans_filters_and_returns_labels(session):
    query = _topic(session)
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert classify.spec.labels == ("refund", "shipping", "praise")
    assert classify.spec.label_token_ids[0] == tuple(_bytes("refund"))
    assert test.predicate == RegularPredicate(
        ColumnRef("d", "documents", "topic"), "in", ("refund", "shipping"))
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph

    result = finish(query, _labels(["refund", "praise"]))
    rows = result.collect()
    assert rows.column("topic").to_pylist() == ["refund"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["refund", "praise"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True, False]
    assert result.report["node_metrics"]["ai-classify:0"]["suffix_tokens"] == 6

    # a projected label with no filter classifies every document
    plain = _topic(session, accepted=())
    result = finish(plain, _labels(["praise", "shipping"]))
    assert result.collect().column("topic").to_pylist() == [
        "praise", "shipping"]

    # a document whose decoded answer names no label leaves the run
    result = finish(plain, _labels(["praise", None]))
    assert result.collect().column("topic").to_pylist() == ["praise"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["praise"]
    assert result.report["node_metrics"]["ai-classify:0"]["output_rows"] == 1

    result = finish(query, _labels([None, "refund"]))
    assert result.collect().column("topic").to_pylist() == ["refund"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True]
    assert result.answer_tables["classifies"]["topic"].column("d").to_pylist() == [1]

    # with probabilities, a map of each label's probability beside the
    # label, from a rule that scores every label
    probable = session.sql(
        "SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
        "ARRAY['refund', 'shipping', 'praise'], {'probabilities': true}) "
        "AS topic FROM documents d")
    plan = probable.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.probabilities
    assert classify.spec.scoring != "trie_decode"
    assert plan.nodes[-1].columns == ("d.id", "topic", "topic_probabilities")
    rows = finish(probable, _labels(["praise", "shipping"])).collect()
    assert rows.column("topic_probabilities").to_pylist() == [
        [("refund", 0.1), ("shipping", 0.1), ("praise", 0.8)],
        [("refund", 0.1), ("shipping", 0.8), ("praise", 0.1)]]


def test_classify_probabilities_beside_an_extract(session):
    # an extract in the same query projects the label through the label
    # tables; the probability map must come back as a map there too
    alone = session.sql(
        f"SELECT d.id, {TOPIC}, {{'probabilities': true}}) AS topic "
        "FROM documents d")
    query = session.sql(
        f"SELECT d.id, {TOPIC}, {{'probabilities': true}}) AS topic, "
        "AI.EXTRACT(d.body, {'item': 'What is asked for?'}) AS asked "
        "FROM documents d")
    oracle = Oracle(labels=["praise", "shipping"], probabilities=True,
                    answers=lambda spec, document: (
                        "refund" if document == 0 else None))
    rows = finish(query, oracle).collect()
    expected = finish(alone, _labels(["praise", "shipping"])).collect()
    assert rows.column("topic").to_pylist() == ["praise", "shipping"]
    assert (rows.schema.field("topic_probabilities").type
            == expected.schema.field("topic_probabilities").type)
    assert (rows.column("topic_probabilities").to_pylist()
            == expected.column("topic_probabilities").to_pylist())
    assert rows.column("asked").to_pylist() == [{"item": "refund"},
                                                {"item": None}]


def _classify_then_filter(session, *, filters=1, later_filters=()):
    tests = [("topic", ("refund", "shipping"))]
    if filters > 1:
        tests.append(("topic", ("refund",)))
    tests.extend(later_filters)
    where = " AND ".join(_label_test(name, accepted) for name, accepted in tests)
    return session.sql(f"SELECT d.id, {TOPIC}) AS topic, {KIND}) AS kind "
                       f"FROM documents d WHERE {where}")


def test_sql_classifies_and_tests_labels(session):
    call = ("AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
            "ARRAY['refund', 'shipping', 'praise'])")
    query = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE {call} IN ('refund', 'shipping')")
    (test,) = [n for n in query.plan().nodes if isinstance(n, Filter)]
    assert test.predicate == RegularPredicate(
        ColumnRef("d", "documents", "topic"), "in", ("refund", "shipping"))
    # a filter on the label alone, with a selectivity option
    filtered = session.sql(
        f"SELECT d.id FROM documents d WHERE {call[:-1]}, "
        f"{{'selectivity': 0.5}}) IN ('praise')")
    (predicate,) = filtered.logical.operators().column_filters["d"]
    assert predicate.selectivity == 0.5
    assert predicate.condition.value == ("praise",)
    assert "d" not in filtered.logical.operators().filters
    # equality is membership in one label, on either side
    for form in (f"{call} = 'refund'", f"'refund' = {call}"):
        equal = session.sql(f"SELECT d.id FROM documents d WHERE {form}")
        (predicate,) = equal.logical.operators().column_filters["d"]
        assert predicate.condition.value == ("refund",)
    for bad in (f"SELECT d.id, {call} FROM documents d",
                f"SELECT d.id FROM documents d WHERE {call} IN (1)"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_sql_classifies_the_rows_a_join_keeps(session, tmp_path):
    path = tmp_path / "aspects.parquet"
    pq.write_table(pa.table({"id": [1, 2], "aspect": ["price", "size"]}),
                   path)
    session.register("aspects",
                     DocumentProvider.from_parquet(str(path), id_col="id"))
    stance = ("AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say about "
              "DOCUMENT {1}?', d.body, a.aspect), ARRAY['praise', 'complaint'])")
    query = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        f"JOIN aspects a ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', "
        f"d.body, a.aspect))")
    (call,) = [column.expression for column in query.logical.root.columns
               if isinstance(column, Alias)]
    assert call.kind == "label" and call.aliases() == ("d", "a")
    assert call.prompt.frame == join_anchor_note(0)
    assert call.prompt.tail.startswith("\n\nAnswer with exactly one")
    # the classification sits above the join of the two tables
    assert [type(node).__name__ for node in query.logical.walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticClassify", "Project"]
    assert query.logical.root.input.call is call
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert classify.spec.partner is not None
    assert classify.spec.anchor == join.anchor
    assert classify.inputs[0].source.node_id == join.node_id
    # More than one independent label consumer ends the join pipeline.
    other = stance.replace("What does DOCUMENT", "How does DOCUMENT")
    refused = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance, {other} AS other "
        "FROM documents d JOIN aspects a "
        "ON AI_FILTER(PROMPT('Does {0} mention {1}?', d.body, a.aspect))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "joined_classify_pipeline"
    assert "standalone pair classification" in refused.reasons[0]
    distributed = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=2),
        tokenizer=letter_tokens)
    for name in ("documents", "aspects"):
        distributed.register(name, session.catalog.get(name))
    # every worker runs the join and the classification as one pipeline
    spread = distributed.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        "JOIN aspects a ON AI_FILTER(PROMPT('Does {0} mention {1}?', "
        "d.body, a.aspect))").plan()
    assert not isinstance(spread, Refusal)
    (classify,) = [n for n in spread.nodes if isinstance(n, AiClassify)]
    assert classify.node_id in {member.node_id for member in build_pipelines(
        spread.graph)[classify.inputs[0].source.node_id]}
    distributed.close()
    # a joined row's label is not tested in WHERE
    with pytest.raises(CompileError, match="one table|filter on a label"):
        session.sql(
            f"SELECT d.id, a.id FROM documents d JOIN aspects a "
            f"ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', d.body, "
            f"a.aspect)) "
            f"WHERE {stance} = 'praise'")


def test_streamed_classifications_share_one_pipeline(session):
    query = _classify_then_filter(session)
    plan = query.plan()
    classifies = [n for n in plan.nodes if isinstance(n, AiClassify)]
    topic, kind = classifies
    assert [node.spec.name for node in classifies] == ["topic", "kind"]
    assert tuple(
        member.node_id for member in build_pipelines(plan.graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", kind.node_id)
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    assert kind.spec.estimated_seconds < topic.spec.estimated_seconds

    scorer = _labels(["refund", "praise"])
    result = finish(query, scorer)
    assert result.collect().to_pydict() == {
        "d.id": [7], "topic": ["refund"], "kind": ["complaint"]}
    labels = result.answer_tables["classifies"]
    assert labels["topic"].column("topic").to_pylist() == ["refund", "praise"]
    assert labels["kind"].select(["d", "kind"]).to_pydict() == {
        "d": [0], "kind": ["complaint"]}
    assert scorer.calls == [("topic", [[0], [1]]), ("kind", [[0]])]

    split = _classify_then_filter(session, filters=2)
    assert tuple(
        member.node_id for member in build_pipelines(
            split.plan().graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", "filter:d:1", kind.node_id)

    # a later test on a label runs right after its classification, so
    # the next classification reads fewer documents
    interleaved = _classify_then_filter(session, later_filters=(
        ("kind", ("complaint",)), ("topic", ("refund",))))
    assert tuple(
        member.node_id for member in build_pipelines(
            interleaved.plan().graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", "filter:d:2", kind.node_id, "filter:d:1")
    scorer = _labels(["shipping", "praise"])
    result = finish(interleaved, scorer)
    assert result.collect().num_rows == 0
    # the second test on topic drops the last document before kind
    assert scorer.calls == [("topic", [[0], [1]])]
def test_plain_label_tests_share_the_classification_used_by_membership(session):
    call = "AI.CLASSIFY(PROMPT('Topic? {0}', d.body), ARRAY['a', 'b'])"
    query = session.sql(
        f"SELECT d.id FROM documents d WHERE {call} IN ('a', 'b') "
        f"AND {call} < 'b' AND {call} IS NOT NULL")
    assert len(query.logical.operators().classifies) == 1
    (classify,) = [node for node in query.plan().nodes if isinstance(node, AiClassify)]
    tests = query.logical.operators().column_filters["d"]
    assert {test.name for test in tests} == {classify.spec.name}
    assert finish(query, _labels(["a", "b"])).collect().to_pylist() == [{"d.id": 7}]


