"""A map or extraction stops an answer once its tests are decided.

should_stop_answers_early gives a map or extraction the region's exact
tests of its answer alone; the executor stops an answer once one is
decided not TRUE on the text so far, or, when nothing else reads the
answer, once every one is decided TRUE. The oracle writes each answer
one character a token. The transcripts, in transcripts/, show the plan,
each answer and where it stopped, the answer tokens written, and the
rows (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.execution.answer_stops import (
    AnswerStop,
    Stop,
    answer_stop,
    copied_completions,
)

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["March 3, 2020 was kind", "May 2021 rude", "March 9 kind",
             "Jan 1 rude", "March 30 kind", "Apr 2 rude"],
})
VERDICT = "LEFT(AI_MAP(r.body, 'Is it kind?'), 3) = 'yes'"
VENDORS = ["Acme", "Initech Systems", "Globex", "Zeta Corporation",
           "Acme", "Hooli"]


def _answer(kind, prompt, rows):
    """Judge even reviews kind, name a vendor, and extract the text's start."""
    index = int(rows[0][1:])
    if prompt == "Name the vendor.":
        return VENDORS[index]
    if kind == "map":
        return "yes, it is kind" if index % 2 == 0 else "no, it is not kind"
    return REVIEWS["body"][index].as_py().rsplit(" ", 1)[0]


def test_a_map_stops_each_answer_once_its_prefix_test_is_decided():
    with Transcripts({"reviews": REVIEWS}, _answer, written=True) as runs:
        assert_transcript("\n".join(runs.query(sql) for sql in (
            f"SELECT r.id FROM reviews r WHERE {VERDICT}",
            # the SELECT list reads each answer the test keeps whole
            f"SELECT r.id, AI_MAP(r.body, 'Is it kind?') AS v FROM reviews r "
            f"WHERE {VERDICT}")))


def test_an_extraction_stops_once_no_span_of_its_document_can_pass():
    # a span goes on only with its document's text: "M" of "May 2021
    # rude" or "March 9 kind" cannot become "March 3"
    call = "AI_EXTRACT(r.body, {'w': 'When?'})"
    with Transcripts({"reviews": REVIEWS}, _answer, written=True) as runs:
        assert_transcript(runs.query(
            f"SELECT r.id, {call} AS e FROM reviews r "
            f"WHERE LEFT({call}['w'], 7) = 'March 3'"))


def test_a_join_on_a_maps_answer_stops_each_answer_that_can_match_no_key(
        monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    vendors = pa.table({"id": ["v0", "v1"], "name": ["Acme", "Globex"]})
    # the join builds on the vendors, so their names are the region's
    # hint before the model writes an answer
    with Transcripts({"reviews": REVIEWS, "vendors": vendors}, _answer,
                     written=True) as runs:
        assert_transcript(runs.query(
            "SELECT r.id, v.id FROM vendors v JOIN reviews r "
            "ON v.name = AI.MAP(r.body, 'Name the vendor.', 16)"))


def test_explain_says_when_a_map_stops_an_answer():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    physical = session.sql(f"SELECT r.id FROM reviews r WHERE {VERDICT}").explain()
    session.close()
    assert ("stops an answer once left(r.__quail_column_1, 3) = 'yes' is not TRUE on "
            "its tokens so far, or once every one is TRUE") in physical


def test_an_expression_decides_a_prefix_test_on_a_partial_answer():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    cases = {
        "LIKE 'Mar%'": (("Jan", "not true"), ("Mar", "true"), ("", "undecided"),
                        ("Ma", "undecided")),
        # a test in one letter case reads the text so far in that case
        "ILIKE 'mar%'": (("JAN", "not true"), ("MAR", "true"), ("mA", "undecided")),
        "ILIKE '%kind%'": (("It was KIND", "true"), ("It was", "undecided")),
    }
    for written, decisions in cases.items():
        plan = session.sql("SELECT r.id FROM reviews r "
                           f"WHERE AI_MAP(r.body, 'When?') {written}").plan()
        (node,) = [node for node in plan.graph.nodes
                   if node.type_name == "quail.ai_map"]
        (test,) = node.stop_tests
        for prefix, decided in decisions:
            assert test.decide(node.spec.name, prefix) == decided, (written, prefix)
    session.close()


LAWS = ("Disputes go to the state courts. This Agreement is governed by the laws "
        "of the State of California.")


def _plan_test(sql):
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    plan = session.sql(sql).plan()
    session.close()
    (node,) = [node for node in plan.graph.nodes
               if node.type_name in ("quail.ai_map", "quail.ai_extract")]
    return answer_stop(node)


LAW_SQL = ("SELECT r.id, AI_EXTRACT(r.body, {'law': 'Which law?'}) AS e "
           "FROM reviews r "
           "WHERE AI_EXTRACT(r.body, {'law': 'Which law?'})['law'] "
           "= 'State of California'")


def test_an_extraction_goes_on_until_no_copied_span_can_pass():
    stop = _plan_test(LAW_SQL)
    for prefix in (" State", "State", "State of", "State of ", "State of Cal", " "):
        # the copy may still be at any place the document holds the prefix,
        # in any case, and its answer is that span stripped
        spans = [(start, start + len(prefix)) for start in range(len(LAWS))
                 if LAWS[start:start + len(prefix)].lower() == prefix.lower()]
        assert stop.decide_any(copied_completions(LAWS, spans)) is Stop.CONTINUE, prefix
    start = LAWS.index("State of")
    spans = [(LAWS.index("state courts"), LAWS.index(" courts")),
             (start, start + len("State"))]
    assert stop.decide_any(copied_completions(LAWS.replace(
        "California", "New York"), spans)) is Stop.REJECTED


def test_a_map_answer_is_decided_on_its_text_with_its_leading_space():
    stop = _plan_test("SELECT r.id FROM reviews r "
                      "WHERE AI_MAP(r.body, 'Is it kind?') = ' Yes'")
    for prefix, decided in ((" ", Stop.CONTINUE), (" Ye", Stop.CONTINUE),
                            ("Ye", Stop.REJECTED), (" No", Stop.REJECTED)):
        assert stop.decide(prefix) is decided, prefix


def test_an_answer_stops_only_when_the_hint_rejects_every_completion():
    class Hint:
        def rejects_answer(self, answer, prefix, **completion):
            return answer == "vendor" and not prefix.startswith("A")

    stop = AnswerStop("vendor", (), False, Hint())
    assert stop.decide_any([dict(prefix="Z")]) is Stop.REJECTED
    assert stop.decide_any([dict(prefix="Z"), dict(prefix="Ac")]) is Stop.CONTINUE
    assert stop.decide_any([dict(prefix="Ac")]) is Stop.CONTINUE
