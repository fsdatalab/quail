"""Model operators over copied documents are asked once per distinct text.

The planner records, per table with copies, the row columns its
documents are grouped by beside their tokens; the executor asks the key
documents and gives every copy their answers. Each query here runs with and without
that rule and must return the same rows.
"""

import numpy as np
import pyarrow as pa
import pytest
from helpers.baselines import _session as _request_session
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.planner.rules.physical import ShouldDeduplicate

BODIES = ["noisy room", "cold room", "noisy room", "slow lift", "noisy room",
          "cold room"]
# a copy of a review can sit in another hotel than its first
HOTELS = ["h1", "h2", "h2", "h1", "h3", "h1"]
NOTES = ["noise", "cold", "noise", "lift"]
NOTE_HOTELS = ["h1", "h2", "h2", "h3"]


def _session():
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{index}" for index in range(len(BODIES))], "body": BODIES,
        "hotel": HOTELS,
    }), id_col="id"))
    session.register("notes", quail.DocumentProvider.from_table(pa.table({
        "id": [f"n{index}" for index in range(len(NOTES))], "body": NOTES,
        "hotel": NOTE_HOTELS,
    }), id_col="id"))
    return session


def _planned(query):
    """The plan over the exact tokens; the first plan estimates lengths."""
    query.plan()
    query.wait_for_tokens()
    return query.plan()


def _short(session):
    return (session.docs("reviews").alias("r")
            .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body"))))


def test_the_planner_deduplicates_tables_with_copies_except_on_request_backends():
    with _session() as session:
        plan = _planned(_short(session).select("r.id"))
        assert plan.settings["deduplicate"] == {"r": ()}
        # a filter's stop key is read beside the tokens, so the key
        # document must match in it too
        query = _short(session).distinct().select("r.hotel")
        assert _planned(query).settings["deduplicate"] == {"r": ("hotel",)}
        assert "deduplicate: r (by text and hotel)" in query.explain()
        session.register("lifts", quail.DocumentProvider.from_table(pa.table({
            "id": ["l0", "l1"], "body": ["slow lift", "fast lift"]}), id_col="id"))
        plan = _planned(
            session.docs("lifts").alias("l")
            .ai_filter(quail.prompt("Is {0} short?", quail.col("l.body")))
            .select("l.id"))
        assert "deduplicate" not in plan.settings
    with _request_session("stock_vllm", reviews={
            "id": [f"r{index}" for index in range(len(BODIES))],
            "body": BODIES}) as session:
        assert "deduplicate" not in _planned(_short(session).select("r.id")).settings


TOPIC = {"noisy room": "noise", "cold room": "cold", "slow lift": "lift"}
KEY = ()


def _matches(node, assignment):
    return TOPIC[BODIES[assignment["r"]]] == NOTES[assignment["n"]]


def _sorted(table):
    return sorted(tuple(row.values()) for row in table.to_pylist())


def _run(build, oracle, rule=True, monkeypatch=None, session=_session):
    """The query's rows, its answer tables, its plan, and the oracle's calls."""
    with session() as session, monkeypatch.context() as patch:
        if not rule:
            patch.setattr(ShouldDeduplicate, "rewrite",
                          lambda self, graph, context: None)
        query = build(session)
        plan = _planned(query)
        result = finish(query, oracle)
        answers = {(kind, name): _sorted(table)
                   for kind, tables in result.answer_tables.items()
                   for name, table in tables.items()}
        return sorted(result.to_rows()), answers, plan


def _both(build, grouped_oracle, plain_oracle, monkeypatch, session=_session):
    """Run a query with the rule and without it; both give the same answers."""
    grouped = _run(build, grouped_oracle, True, monkeypatch, session)
    plain = _run(build, plain_oracle, False, monkeypatch, session)
    assert grouped[:2] == plain[:2]
    return grouped


class _Asked:
    """An oracle that records the documents and pairs it is asked about."""

    def __init__(self, pipelines=False):
        self.documents, self.pairs = [], []
        self.pipelines = pipelines

    def oracle(self, labels=None):
        def filters(alias, document, written_position):
            self.documents.append(document)
            if written_position == 0:
                return not BODIES[document].startswith("slow")
            return BODIES[document].startswith("noisy")

        def joins(node, assignment):
            self.pairs.append((assignment["r"], assignment["n"]))
            return _matches(node, assignment)

        return Oracle(filters=filters, joins=joins, labels=labels,
                      pipelines=self.pipelines)


@pytest.mark.parametrize("pipelines", [False, True])
def test_a_filter_over_copies_asks_once_per_text_and_returns_every_copy(
        monkeypatch, pipelines):
    grouped, plain = _Asked(pipelines), _Asked(pipelines)

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} about a room?", quail.col("r.body")),
                           selectivity=0.5)
                .select("r.id"))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert plan.settings["deduplicate"] == {"r": KEY}
    assert rows == [("r0",), ("r1",), ("r2",), ("r4",), ("r5",)]
    assert (grouped.documents, plain.documents) == ([0, 1, 3], list(range(6)))


def test_an_extraction_over_copies_answers_every_copy_from_one_reading(
        monkeypatch):
    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_extract("r.body", {"thing": "What is wrong?"}, name="found")
                .select("r.id", "found"))

    def oracle(read):
        def answers(spec, document):
            read.append(document)
            return BODIES[document].split()[0]
        return Oracle(answers=answers)

    grouped, plain = [], []
    rows, _, _ = _both(build, oracle(grouped), oracle(plain), monkeypatch)
    # each row carries the answer and its span in the document
    assert [row[:2] for row in rows] == [
        ("r0", {"thing": "noisy"}), ("r1", {"thing": "cold"}),
        ("r2", {"thing": "noisy"}), ("r3", {"thing": "slow"}),
        ("r4", {"thing": "noisy"}), ("r5", {"thing": "cold"})]
    assert rows[0][2] == {"thing": {"start": 0, "end": 5}}
    assert (grouped, plain) == ([0, 1, 3], list(range(6)))


@pytest.mark.parametrize("pipelines", [False, True])
def test_a_filter_with_a_stop_key_keeps_a_text_apart_under_each_key(
        monkeypatch, pipelines):
    # hotel h1 stops at "fine", so its "ok" is never asked; h2's "ok" is
    # a copy of it that must still be asked, as its own key document
    bodies = ["fine", "ok", "ok", "ok", "meh", "meh"]
    hotels = ["h1", "h1", "h2", "h2", "h3", "h3"]
    asked = {True: [], False: []}

    def session():
        session = quail.Session(CONFIG, tokenizer=fake_tok)
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": [f"r{index}" for index in range(len(bodies))],
            "body": bodies, "hotel": hotels}), id_col="id"))
        return session

    def oracle(rule):
        def filters(alias, document, written_position):
            asked[rule].append(document)
            return bodies[document] != "meh"
        return Oracle(filters=filters, pipelines=pipelines)

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body")))
                .distinct().select("r.hotel"))

    rows, answers, plan = _run(build, oracle(True), True, monkeypatch, session)
    plain_rows, plain_answers, _ = _run(build, oracle(False), False, monkeypatch,
                                        session)
    assert rows == plain_rows == [("h1",), ("h2",)]
    assert plan.settings["deduplicate"] == {"r": ("hotel",)}
    assert asked == {True: [0, 2, 4], False: [0, 2, 4, 5]}
    # a copy of an asked document gets its answer even where its stop key
    # group had stopped, so the plain run's answers are a subset
    (grouped_answers,), (plain,) = answers.values(), plain_answers.values()
    assert set(plain) < set(grouped_answers)
    assert (3, 0, True) in grouped_answers


@pytest.mark.parametrize("pipelines", [False, True])
def test_a_join_over_key_pairs_asks_each_pair_of_texts_once(monkeypatch,
                                                            pipelines):
    grouped, plain = _Asked(pipelines), _Asked(pipelines)

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} about a room?",
                                        quail.col("r.body")), selectivity=0.5)
                .join(session.docs("notes").alias("n"),
                      on=quail.col("r.hotel") == quail.col("n.hotel"))
                .ai_filter(quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                        quail.col("n.body")))
                .select("r.id", "n.id"))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert rows == [("r0", "n0"), ("r1", "n1"), ("r2", "n2")]
    assert plan.settings["deduplicate"] == {"n": KEY, "r": KEY}
    # the hotel pairs of the rooms: r0-n0, r1-n1, r1-n2, r2-n1, r2-n2,
    # r4-n3, r5-n0; their key documents' pairs: 0-0, 1-1, 1-0, 0-1, 0-3
    assert sorted(plain.pairs) == [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2),
                                   (4, 3), (5, 0)]
    assert sorted(grouped.pairs) == [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1)]


@pytest.mark.parametrize("semantics", ["exists", "anti"])
def test_a_join_of_every_pair_asks_each_pair_of_texts_once(monkeypatch,
                                                           semantics):
    grouped, plain = _Asked(), _Asked()

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_join(session.docs("notes").alias("n").where(
                             quail.col("n.hotel") != "h3"),
                         quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                      quail.col("n.body")),
                         anchor="r", semantics=semantics)
                .select("r.id"))

    rows, _, _ = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert rows == {
        "exists": [("r0",), ("r1",), ("r2",), ("r4",), ("r5",)],
        # the lift note is in hotel h3, so no note matches r3
        "anti": [("r3",)],
    }[semantics]
    assert set(grouped.pairs) <= {(0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 1)}
    assert len(grouped.pairs) < len(plain.pairs)


def test_a_self_join_asks_each_pair_of_texts_once(monkeypatch):
    grouped, plain = [], []

    def oracle(asked):
        def joins(node, assignment):
            asked.append((assignment["a"], assignment["b"]))
            return all("room" in BODIES[assignment[alias]] for alias in "ab")
        return Oracle(joins=joins)

    def build(session):
        return session.sql(
            "SELECT a.id, b.id FROM reviews a JOIN reviews b "
            "ON a.hotel = b.hotel "
            "AND AI_FILTER(PROMPT('Do {0} and {1} agree?', a.body, b.body))")

    rows, _, _ = _both(build, oracle(grouped), oracle(plain), monkeypatch)
    # same-hotel pairs in both orders, each review with itself too: the
    # rooms of h1 are r0 and r5, of h2 r1 and r2, of h3 r4
    assert rows == [("r0", "r0"), ("r0", "r5"), ("r1", "r1"), ("r1", "r2"),
                    ("r2", "r1"), ("r2", "r2"), ("r4", "r4"), ("r5", "r0"),
                    ("r5", "r5")]
    assert len(plain) == 14
    # the pairs of their key documents 0, 1, and 3, on either side
    assert sorted(grouped) == [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3),
                               (3, 0), (3, 1), (3, 3)]


def test_copies_with_different_carried_values_are_not_merged():
    from quail.execution.deduplicate import _reduced_rows
    from quail.execution.rows import Rows

    ids = np.array([0, 1, 2])
    keys = np.array([0, 1, 0])
    same = Rows({"r": ids}, pa.table({"label": ["a", "b", "a"]}))
    reduced = _reduced_rows(same, "r", ids, keys)
    assert reduced.ids["r"].to_pylist() == [0, 1]
    assert reduced.column("label").to_pylist() == ["a", "b"]
    # r2 copies r0's text but carries its own value, which a merged row
    # would overwrite with r0's
    different = Rows({"r": ids}, pa.table({"stars": [5, 3, 1]}))
    assert _reduced_rows(different, "r", ids, keys) is None


def _every_pair(session):
    return (session.docs("reviews").alias("r")
            .ai_join(session.docs("notes").alias("n"),
                     quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                  quail.col("n.body")), anchor="r")
            .select("r.id"))


@pytest.mark.parametrize("build", [
    lambda session: _short(session).select("r.id"), _every_pair])
def test_the_planner_prices_only_the_documents_it_asks(monkeypatch, build):
    from quail.planner.graph import pricing

    with _session() as session:
        deduplicated = _planned(build(session)).estimated_seconds
    monkeypatch.setattr(pricing, "key_documents", lambda context, alias: None)
    with _session() as session:
        every = _planned(build(session)).estimated_seconds
    # 3 of the 6 reviews are copies
    assert deduplicated < every
