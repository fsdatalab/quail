"""Model operators over copied documents are asked once per distinct text.

The planner records, per table with copies, the value column holding
each document's key document; the executor asks the key documents and
gives every copy their answers. Each query here runs with and without
that rule and must return the same rows.
"""

import pyarrow as pa
from helpers.baselines import _session as _request_session
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.catalog.tokens import first_document_column
from quail.planner.rules.physical import ShouldAskOncePerText

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


def test_the_planner_names_a_key_column_for_a_table_with_copies():
    with _session() as session:
        plan = _planned(_short(session).select("r.id"))
        assert plan.settings["ask_once"] == {"r": first_document_column(())}
        # a filter's stop key is read beside the tokens, so the key
        # document must match in it too
        plan = _planned(_short(session).distinct().select("r.hotel"))
        assert plan.settings["ask_once"] == {
            "r": first_document_column(("hotel",))}
        session.register("lifts", quail.DocumentProvider.from_table(pa.table({
            "id": ["l0", "l1"], "body": ["slow lift", "fast lift"]}), id_col="id"))
        plan = _planned(
            session.docs("lifts").alias("l")
            .ai_filter(quail.prompt("Is {0} short?", quail.col("l.body")))
            .select("l.id"))
        assert "ask_once" not in plan.settings


def test_explain_names_the_tables_asked_once_per_text():
    with _session() as session:
        query = _short(session).distinct().select("r.hotel")
        _planned(query)
        assert "ask once per text: r (by text and hotel)" in query.explain()


def test_a_request_backend_asks_every_copy(monkeypatch):
    with _request_session("stock_vllm", reviews={
            "id": [f"r{index}" for index in range(len(BODIES))],
            "body": BODIES}) as session:
        plan = _planned(_short(session).select("r.id"))
        assert "ask_once" not in plan.settings


def test_the_rule_is_the_only_source_of_the_setting(monkeypatch):
    monkeypatch.setattr(ShouldAskOncePerText, "rewrite",
                        lambda self, graph, context: None)
    with _session() as session:
        plan = _planned(_short(session).select("r.id"))
        assert "ask_once" not in plan.settings
