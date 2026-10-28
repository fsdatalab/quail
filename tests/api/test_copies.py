"""Model operators over copied documents are asked once per distinct text.

Each query runs with deduplication and without it; both must return the
same rows and answers, and the deduplicated run asks only key documents.
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
# the first review of each text
KEYS = {0, 1, 3}
NOTES = ["noise", "cold", "noise", "lift"]
NOTE_HOTELS = ["h1", "h2", "h2", "h3"]
TOPIC = {"noisy room": "noise", "cold room": "cold", "slow lift": "lift"}


def _session(bodies=BODIES, hotels=HOTELS):
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    for name, prefix, texts, places in (("reviews", "r", bodies, hotels),
                                        ("notes", "n", NOTES, NOTE_HOTELS)):
        session.register(name, quail.DocumentProvider.from_table(pa.table({
            "id": [f"{prefix}{index}" for index in range(len(texts))],
            "body": texts, "hotel": places}), id_col="id"))
    return session


def _planned(query):
    """The plan over the exact tokens; the first plan estimates lengths."""
    query.plan()
    query.wait_for_tokens()
    return query.plan()


def _short(session):
    return (session.docs("reviews").alias("r")
            .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body")))
            .select("r.id"))


def _oracle(asked, pipelines=False, filters=None):
    """An oracle that records each document, or pair, it is asked about."""
    def filter_answer(alias, document, position):
        asked.append(document)
        return (filters or (lambda d: not BODIES[d].startswith("slow")))(document)

    def join_answer(node, assignment):
        asked.append(tuple(assignment.values()))
        if "n" in assignment:
            return TOPIC[BODIES[assignment["r"]]] == NOTES[assignment["n"]]
        return all("room" in BODIES[document] for document in assignment.values())

    def extract_answer(spec, document):
        asked.append(document)
        return BODIES[document].split()[0]

    return Oracle(filters=filter_answer, joins=join_answer,
                  answers=extract_answer, pipelines=pipelines)


def _run(build, deduplicate, monkeypatch, session=_session, **oracle):
    """The query's rows, its answer tables, its plan, and what it asked."""
    asked = []
    with session() as opened, monkeypatch.context() as patch:
        if not deduplicate:
            patch.setattr(ShouldDeduplicate, "rewrite", lambda *_: None)
        query = build(opened)
        plan = _planned(query)
        result = finish(query, _oracle(asked, **oracle))
        answers = {name: sorted(map(tuple, (row.values() for row in
                                            table.to_pylist())))
                   for tables in result.answer_tables.values()
                   for name, table in tables.items()}
        return sorted(result.to_rows()), answers, plan, asked


def _both(build, monkeypatch, **options):
    """Run with deduplication and without; both give the same rows and answers."""
    rows, answers, plan, asked = _run(build, True, monkeypatch, **options)
    plain_rows, plain_answers, _, plain_asked = _run(build, False, monkeypatch,
                                                     **options)
    assert (rows, answers) == (plain_rows, plain_answers)
    return rows, plan, asked, plain_asked


def test_the_planner_deduplicates_tables_with_copies_except_on_request_backends():
    with _session() as session:
        assert _planned(_short(session)).settings["deduplicate"] == {"r": ()}
        # a filter's stop key is read beside the tokens, so the key
        # document must match in it too
        query = (session.docs("reviews").alias("r")
                 .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body")))
                 .distinct().select("r.hotel"))
        assert _planned(query).settings["deduplicate"] == {"r": ("hotel",)}
        assert "deduplicate: r (by text and hotel)" in query.explain()
    with _session(["a", "b"], ["h1", "h2"]) as session:
        assert "deduplicate" not in _planned(_short(session)).settings
    with _request_session("stock_vllm", reviews={
            "id": [f"r{index}" for index in range(len(BODIES))],
            "body": BODIES}) as session:
        assert "deduplicate" not in _planned(_short(session)).settings


@pytest.mark.parametrize("pipelines", [False, True])
def test_a_filter_asks_once_per_text_and_returns_every_copy(monkeypatch,
                                                            pipelines):
    rows, _, asked, plain = _both(_short, monkeypatch, pipelines=pipelines)
    assert rows == [("r0",), ("r1",), ("r2",), ("r4",), ("r5",)]
    assert (asked, plain) == ([0, 1, 3], list(range(6)))


def test_an_extraction_answers_every_copy_from_one_reading(monkeypatch):
    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_extract("r.body", {"thing": "What is wrong?"}, name="found")
                .select("r.id", "found"))

    rows, _, asked, plain = _both(build, monkeypatch)
    assert [row[1]["thing"] for row in rows] == [
        body.split()[0] for body in BODIES]
    assert (asked, plain) == ([0, 1, 3], list(range(6)))


@pytest.mark.parametrize("pipelines", [False, True])
def test_a_stop_key_keeps_a_text_apart_under_each_key(monkeypatch, pipelines):
    # hotel h1 stops at "fine", so its "ok" is never asked; h2's "ok" is
    # a copy of it that must still be asked, as its own key document
    bodies = ["fine", "ok", "ok", "ok", "meh", "meh"]

    def session():
        return _session(bodies, ["h1", "h1", "h2", "h2", "h3", "h3"])

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body")))
                .distinct().select("r.hotel"))

    options = {"session": session, "pipelines": pipelines,
               "filters": lambda document: bodies[document] != "meh"}
    rows, _, plan, asked = _run(build, True, monkeypatch, **options)
    plain_rows, _, _, plain = _run(build, False, monkeypatch, **options)
    assert rows == plain_rows == [("h1",), ("h2",)]
    assert plan.settings["deduplicate"] == {"r": ("hotel",)}
    assert (asked, plain) == ([0, 2, 4], [0, 2, 4, 5])


def _every_pair(semantics):
    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_join(session.docs("notes").alias("n").where(
                             quail.col("n.hotel") != "h3"),
                         quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                      quail.col("n.body")),
                         anchor="r", semantics=semantics)
                .select("r.id"))
    return build


def _hotel_pairs(session):
    return (session.docs("reviews").alias("r")
            .join(session.docs("notes").alias("n"),
                  on=quail.col("r.hotel") == quail.col("n.hotel"))
            .ai_filter(quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                    quail.col("n.body")))
            .select("r.id", "n.id"))


def _self_join(session):
    return session.sql(
        "SELECT a.id, b.id FROM reviews a JOIN reviews b ON a.hotel = b.hotel "
        "AND AI_FILTER(PROMPT('Do {0} and {1} agree?', a.body, b.body))")


@pytest.mark.parametrize("build", [
    _every_pair("exists"), _every_pair("anti"), _hotel_pairs, _self_join],
    ids=["exists", "anti", "hotel_pairs", "self_join"])
def test_a_join_asks_each_pair_of_texts_once(monkeypatch, build):
    _, _, asked, plain = _both(build, monkeypatch)
    # every asked review is the first of its text, on either side of a
    # self-join; the notes have no copies
    assert {pair[0] for pair in asked} <= KEYS
    if build is _self_join:
        assert {pair[1] for pair in asked} <= KEYS
    assert len(set(asked)) < len(set(plain))


def test_copies_with_different_carried_values_are_not_merged():
    from quail.execution.deduplicate import _reduced_rows
    from quail.execution.rows import Rows

    ids, keys = np.array([0, 1, 2]), np.array([0, 1, 0])
    same = _reduced_rows(Rows({"r": ids}, pa.table({"label": ["a", "b", "a"]})),
                         "r", ids, keys)
    assert same.column("label").to_pylist() == ["a", "b"]
    # r2 copies r0's text but carries its own value, which a merged row
    # would overwrite with r0's
    different = Rows({"r": ids}, pa.table({"stars": [5, 3, 1]}))
    assert _reduced_rows(different, "r", ids, keys) is None


@pytest.mark.parametrize("build", [_short, _every_pair("exists")],
                         ids=["filter", "every_pair"])
def test_the_planner_prices_only_the_documents_it_asks(monkeypatch, build):
    from quail.planner.graph import pricing

    with _session() as session:
        deduplicated = _planned(build(session)).estimated_seconds
    monkeypatch.setattr(pricing, "key_documents", lambda context, alias: None)
    with _session() as session:
        every = _planned(build(session)).estimated_seconds
    assert deduplicated < every
