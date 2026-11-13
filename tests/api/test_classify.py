"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results.

The transcript tests' expected text is in transcripts/: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""


import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.planning import column_tests, operators_of, row_tests
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.common.expressions import Alias
from quail.physical import AiClassify, AiJoin, Filter
from quail.physical.plan import Refusal

DOCUMENTS = pa.table({"id": ["d0", "d1", "d2"],
                      "body": ["where is my refund", "love it", "hmm"]})
ASPECTS = pa.table({"id": ["a0", "a1"], "aspect": ["price", "size"]})

TOPIC = ("AI.CLASSIFY(PROMPT('Which topic? {0}', d.body), "
         "ARRAY['refund', 'shipping', 'praise']")
KIND = ("AI.CLASSIFY(PROMPT('Which kind? {0}', d.body), "
        "ARRAY['complaint', 'question']")
CALLS = {"topic": TOPIC, "kind": KIND}
STANCE = ("AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say about "
          "DOCUMENT {1}?', d.body, a.aspect), ARRAY['praise', 'complaint'])")
MENTIONS = "AI_FILTER(PROMPT('Does {0} mention {1}?', d.body, a.aspect))"
# the transcripts name a prompt by text its question holds whole
PAIR_STANCE = ("AI.CLASSIFY(PROMPT('Stance? {0} {1}', d.body, a.aspect), "
               "ARRAY['praise', 'complaint'])")
PAIR_MENTIONS = "AI_FILTER(PROMPT('Mentions? {0} {1}', d.body, a.aspect))"


@pytest.fixture()
def session():
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER)
    value.register("documents",
                   DocumentProvider.from_table(DOCUMENTS, id_col="id"))
    value.register("aspects", DocumentProvider.from_table(ASPECTS, id_col="id"))
    yield value
    value.close()


def _label_test(name, accepted):
    """A WHERE test that the label `name` is in `accepted`, at selectivity 0.5."""
    labels = ", ".join(f"'{label}'" for label in accepted)
    return f"{CALLS[name]}, {{'selectivity': 0.5}}) IN ({labels})"


def _classify_then_filter(*tests):
    where = " AND ".join(_label_test(name, accepted) for name, accepted in tests)
    return (f"SELECT d.id, {TOPIC}) AS topic, {KIND}) AS kind "
            f"FROM documents d WHERE {where}")


def _answers(topics=("refund", "praise", None)):
    """The model's answers: topic `topics[i]` for document i, and the rest.

    A topic of None names no label. Every kind is a complaint, every
    pair but d2's matches, and a stance is praise for d0 and a1 only.
    """
    def answer(kind, prompt, rows):
        if prompt.startswith("Mentions"):
            return "d2" not in rows
        if kind == "extract":
            return "refund" if rows[0] == "d0" else None
        if prompt.startswith("Which topic"):
            return topics[int(rows[0][1:])]
        if prompt.startswith("Which kind"):
            return "complaint"
        if prompt.startswith("Stance"):
            return "praise" if set(rows) == {"d0", "a1"} else "complaint"
        if prompt.startswith("Number"):
            return "1"
        if prompt.startswith("Topic?"):
            return "a" if rows[0] == "d0" else "b"
        raise AssertionError(prompt)
    return answer


def _transcript(*queries, columns=(), answers=None) -> str:
    """The transcripts of `queries`; those in `columns` show column types."""
    with Transcripts({"documents": DOCUMENTS, "aspects": ASPECTS},
                     answers or _answers()) as runs:
        return "\n".join(runs.query(sql, columns=sql in columns)
                         for sql in queries)


def test_a_classification_returns_and_tests_labels_and_drops_a_row_without_one():
    call = TOPIC + ")"
    unnamed = f"SELECT d.id, {call} FROM documents d"
    assert_transcript(_transcript(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE {_label_test('topic', ('refund', 'shipping'))}",
        f"SELECT d.id, {call} AS topic FROM documents d",
        # an unnamed classification returns its label under its SQL name
        unnamed,
        # equality is membership in one label, on either side
        f"SELECT d.id FROM documents d WHERE {call} = 'refund'",
        f"SELECT d.id FROM documents d WHERE 'refund' = {call}",
        # DataFusion owns SQL coercion: numeric labels compare to numbers
        "SELECT d.id FROM documents d WHERE "
        "AI.CLASSIFY(PROMPT('Number {0}', d.body), ['1', '2']) IN (1)",
        # plain tests of a label read the one classification its
        # membership test uses
        "SELECT d.id FROM documents d WHERE "
        "AI.CLASSIFY(PROMPT('Topic? {0}', d.body), ARRAY['a', 'b']) IN ('a', 'b') "
        "AND AI.CLASSIFY(PROMPT('Topic? {0}', d.body), ARRAY['a', 'b']) < 'b' "
        "AND AI.CLASSIFY(PROMPT('Topic? {0}', d.body), ARRAY['a', 'b']) "
        "IS NOT NULL",
        columns=(unnamed,)))


def test_a_classification_whose_probabilities_are_not_read_asks_for_none():
    asked = f"{TOPIC}, {{'probabilities': true}})"
    assert_transcript(_transcript(
        # DataFusion groups the labels above the region, which then
        # returns no probabilities; a test alone does not return them
        f"SELECT x.topic, COUNT(*) AS n FROM (SELECT d.id, {asked} AS topic "
        "FROM documents d) x GROUP BY x.topic",
        f"SELECT d.id FROM documents d WHERE {asked} IN ('praise')",
        answers=_answers(topics=("praise", "shipping", "praise"))))


def test_label_probabilities_come_back_beside_the_label_and_an_extraction(
        session):
    # the region's own rows carry the probabilities, which DataFusion's
    # rows, and so the transcripts, leave out
    asked = f"{TOPIC}, {{'probabilities': true}}) AS topic"
    alone = session.sql(f"SELECT d.id, {asked} FROM documents d")
    beside = session.sql(
        f"SELECT d.id, {asked}, "
        "AI.EXTRACT(d.body, {'item': 'What is asked for?'}) AS asked "
        "FROM documents d")
    oracle = Oracle(labels=["praise", "shipping", "praise"], probabilities=True,
                    answers=lambda spec, document: (
                        "refund" if document == 0 else None))
    expected = finish(alone, oracle).collect()
    assert expected.column_names == ["d.id", "topic", "topic_probabilities"]
    assert expected.column("topic_probabilities").to_pylist() == [
        [("refund", 0.1), ("shipping", 0.1), ("praise", 0.8)],
        [("refund", 0.1), ("shipping", 0.8), ("praise", 0.1)],
        [("refund", 0.1), ("shipping", 0.1), ("praise", 0.8)]]
    # an extraction projects the label through the label tables; the
    # probabilities come back as a map there too
    rows = finish(beside, oracle).collect()
    assert (rows.schema.field("topic_probabilities").type
            == expected.schema.field("topic_probabilities").type)
    assert (rows.column("topic_probabilities").to_pylist()
            == expected.column("topic_probabilities").to_pylist())
    assert rows.column("asked").to_pylist() == [
        {"item": "refund"}, {"item": None}, {"item": None}]


def test_a_classification_scores_each_label_and_decodes_greedily_unless_read(
        session):
    query = session.sql(
        f"SELECT d.id, {TOPIC}) AS topic FROM documents d "
        f"WHERE {_label_test('topic', ('refund', 'shipping'))}")
    (classify,) = [n for n in query.plan().nodes if isinstance(n, AiClassify)]
    (test,) = [n for n in query.plan().nodes if isinstance(n, Filter)]
    assert classify.spec.labels == ("refund", "shipping", "praise")
    assert classify.spec.label_token_ids[0] == tuple("refund".encode())
    assert str(test.predicate) == "d.topic IN ('refund', 'shipping')"
    (predicate,) = column_tests(operators_of(query))
    assert predicate.selectivity == 0.5
    assert operators_of(query).filters == ()

    asked = f"{TOPIC}, {{'probabilities': true}})"
    specs = {}
    for name, sql in (
            ("read", f"SELECT d.id, {asked} AS topic FROM documents d"),
            ("grouped", f"SELECT x.topic, COUNT(*) AS n FROM (SELECT d.id, "
                        f"{asked} AS topic FROM documents d) x GROUP BY x.topic"),
            ("tested", f"SELECT d.id FROM documents d WHERE {asked} IN ('praise')")):
        (classify,) = [n for n in session.sql(sql).plan().nodes
                       if isinstance(n, AiClassify)]
        specs[name] = classify.spec
    # with probabilities read, a rule that scores every label
    assert specs["read"].probabilities
    assert specs["read"].scoring != "trie_decode"
    for name in ("grouped", "tested"):
        # the label is decoded greedily, which the probabilities rule out,
        # at a lower estimate than scoring every label
        assert not specs[name].probabilities
        assert specs[name].scoring == "trie_decode"
        assert specs[name].estimated_seconds < specs["read"].estimated_seconds


def test_a_classification_labels_only_the_pairs_a_join_keeps():
    assert_transcript(_transcript(
        f"SELECT d.id, a.id, {PAIR_STANCE} AS stance FROM documents d "
        f"JOIN aspects a ON {PAIR_MENTIONS}",
        # DataFusion tests a joined row's label on the region's rows
        f"SELECT d.id, a.id FROM documents d JOIN aspects a ON {PAIR_MENTIONS} "
        f"WHERE {PAIR_STANCE} = 'praise'"))


def test_a_pair_classification_runs_above_its_join_in_one_pipeline(session):
    query = session.sql(
        f"SELECT d.id, a.id, {STANCE} AS stance FROM documents d "
        f"JOIN aspects a ON {MENTIONS}")
    found = operators_of(query)
    (call,) = [column.expression for column in found.projection.columns
               if isinstance(column, Alias)]
    assert call.kind == "label" and call.aliases() == ("d", "a")
    assert call.prompt.frame == "\n\n(The document above is DOCUMENT {0}.)"
    assert call.prompt.tail.startswith("\n\nAnswer with exactly one")
    # the classification sits above the join of the two tables
    (classified,) = found.classifies
    assert (classified.name, classified.above_joins) == ("stance", True)
    assert classified.call.prompt == call.prompt
    assert len(found.joins) == 1
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert classify.spec.partner is not None
    assert classify.spec.anchor == join.anchor
    assert classify.inputs[0].source.node_id == join.node_id
    # More than one independent label consumer ends the join pipeline.
    other = STANCE.replace("What does DOCUMENT", "How does DOCUMENT")
    refused = session.sql(
        f"SELECT d.id, a.id, {STANCE} AS stance, {other} AS other "
        f"FROM documents d JOIN aspects a ON {MENTIONS}").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "joined_classify_pipeline"
    assert "standalone pair classification" in refused.reasons[0]
    tested = session.sql(
        f"SELECT d.id, a.id FROM documents d JOIN aspects a ON {MENTIONS} "
        f"WHERE {STANCE} = 'praise'")
    assert row_tests(tested) == ["__quail_column_1 = praise"]
    distributed = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=2),
        tokenizer=TOKENIZER)
    for name in ("documents", "aspects"):
        distributed.register(name, session.catalog.get(name))
    # every worker runs the join and the classification as one pipeline
    spread = distributed.sql(
        f"SELECT d.id, a.id, {STANCE} AS stance FROM documents d "
        f"JOIN aspects a ON {MENTIONS}").plan()
    assert not isinstance(spread, Refusal)
    (classify,) = [n for n in spread.nodes if isinstance(n, AiClassify)]
    pipeline = spread.graph.pipelines()[classify.inputs[0].source.node_id]
    assert classify.node_id in {member.node_id for member in pipeline}
    distributed.close()


def test_a_label_test_drops_documents_before_the_next_classification_asks():
    assert_transcript(_transcript(
        _classify_then_filter(("topic", ("refund", "shipping"))),
        # the second test on topic drops d0 before kind asks
        _classify_then_filter(("topic", ("refund", "shipping")),
                              ("kind", ("complaint",)), ("topic", ("refund",))),
        answers=_answers(topics=("shipping", "praise", "praise"))))


def test_streamed_classifications_and_their_tests_share_one_pipeline(session):
    plan = session.sql(_classify_then_filter(("topic", ("refund", "shipping")))).plan()
    topic, kind = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert [node.spec.name for node in (topic, kind)] == ["topic", "kind"]

    def members(*tests):
        plan = session.sql(_classify_then_filter(*tests)).plan()
        return tuple(member.node_id
                     for member in plan.graph.pipelines()[topic.node_id])

    assert members(("topic", ("refund", "shipping"))) == (
        topic.node_id, "filter:d:0", kind.node_id)
    assert kind.spec.estimated_seconds < topic.spec.estimated_seconds
    assert members(("topic", ("refund", "shipping")), ("topic", ("refund",))) == (
        topic.node_id, "filter:d:0", "filter:d:1", kind.node_id)
    # a later test on a label runs right after its classification, so
    # the next classification reads fewer documents
    assert members(("topic", ("refund", "shipping")), ("kind", ("complaint",)),
                   ("topic", ("refund",))) == (
        topic.node_id, "filter:d:0", "filter:d:2", kind.node_id, "filter:d:1")
