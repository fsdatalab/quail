"""A map or extraction stops an answer once its tests are decided.

should_stop_answers_early gives a map or extraction the region's exact
tests of its answer alone; the executor stops an answer once one is
decided not TRUE on the text so far, or, when nothing else reads the
answer, once every one is decided TRUE. The oracle
writes each answer one character a token. Each test runs a query with
the rule on and off, checks that both return the same rows, and counts
the characters written.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry
from quail.execution.answer_stops import Stop, answer_stop, copied_completions

STOP_RULE = "should_stop_answers_early"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["March 3, 2020 was kind", "May 2021 rude", "March 9 kind",
             "Jan 1 rude", "March 30 kind", "Apr 2 rude"],
})
VERDICT = "LEFT(AI_MAP(r.body, 'Is it kind?'), 3) = 'yes'"


def _verdict(spec, document):
    return "yes, it is kind" if document % 2 == 0 else "no, it is not kind"


def _run(sql, *, physical=PHYSICAL_RULES, logical=LOGICAL_RULES, **truth):
    """Return the query's rows, sorted, and the oracle after the run."""
    session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=built_in_registry(
        logical_rules=logical, physical_rules=physical))
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    oracle = Oracle(**truth)
    try:
        result = finish(session.sql(sql), oracle).collect()
    finally:
        session.close()
    return sorted(tuple(row.values()) for row in result.to_pylist()), oracle


OFF = tuple(name for name in PHYSICAL_RULES if name != STOP_RULE)


def test_a_map_stops_each_answer_once_its_prefix_test_is_decided():
    sql = f"SELECT r.id FROM reviews r WHERE {VERDICT}"
    on_rows, on = _run(sql, maps=_verdict)
    off_rows, off = _run(sql, physical=OFF, maps=_verdict)
    assert on_rows == off_rows == [("r0",), ("r2",), ("r4",)]
    # a rejected answer stops at "n", an accepted one at "yes"
    assert (on.answer_tokens, off.answer_tokens) == (3 * 1 + 3 * 3, 3 * 18 + 3 * 15)
    assert on.stopped_early == 6


def test_a_map_the_select_list_reads_writes_each_answer_it_keeps_whole():
    sql = "SELECT r.id, AI_MAP(r.body, 'Is it kind?') AS v FROM reviews r " \
          "WHERE LEFT(AI_MAP(r.body, 'Is it kind?'), 3) = 'yes'"
    on_rows, on = _run(sql, maps=_verdict)
    off_rows, off = _run(sql, physical=OFF, maps=_verdict)
    assert on_rows == off_rows == [("r0", "yes, it is kind"), ("r2", "yes, it is kind"),
                                   ("r4", "yes, it is kind")]
    assert (on.answer_tokens, off.answer_tokens) == (3 * 1 + 3 * 15, 3 * 18 + 3 * 15)


def test_an_extraction_stops_once_no_span_of_its_document_can_pass():
    sql = ("SELECT r.id, AI_EXTRACT(r.body, {'w': 'When?'}) AS e FROM reviews r "
           "WHERE LEFT(AI_EXTRACT(r.body, {'w': 'When?'})['w'], 7) = 'March 3'")

    def when(spec, document):
        return str(REVIEWS["body"][document]).rsplit(" ", 1)[0]

    on_rows, on = _run(sql, answers=when)
    off_rows, off = _run(sql, physical=OFF, answers=when)
    assert on_rows == off_rows
    assert [row[0] for row in on_rows] == ["r0", "r4"]
    # a span goes on only with its document's text: "M" of "May 2021
    # rude" or "March 9 kind" cannot become "March 3", and the answers
    # the test may keep go on, as the SELECT list reads them
    assert on.stopped_early == 4
    assert (on.answer_tokens, off.answer_tokens) == (
        len("March 3, 2020 was") + 1 + 1 + 1 + len("March 30") + 1,
        len("March 3, 2020 was") + len("May 2021") + len("March 9")
        + len("Jan 1") + len("March 30") + len("Apr 2"))


def test_the_physical_plan_shows_when_a_map_stops_its_answers():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    physical = session.sql(f"SELECT r.id FROM reviews r WHERE {VERDICT}").explain()
    session.close()
    assert ("stops an answer once left(r.__column_r_0, 3) = 'yes' is not TRUE on "
            "its tokens so far, or once every one is TRUE") in physical


@pytest.mark.parametrize(("prefix", "decided"), [
    ("Jan", "not true"), ("Mar", "true"), ("", "undecided"), ("Ma", "undecided")])
def test_an_expression_decides_a_prefix_test_on_a_partial_answer(prefix, decided):
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    plan = session.sql("SELECT r.id FROM reviews r "
                       "WHERE AI_MAP(r.body, 'When?') LIKE 'Mar%'").plan()
    session.close()
    (node,) = [node for node in plan.graph.nodes if node.type_name == "quail.ai_map"]
    (test,) = node.stop_tests
    assert test.decide(node.spec.name, prefix) == decided


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


@pytest.mark.parametrize("prefix", [" State", "State", "State of", "State of ",
                                    "State of Cal", " "])
def test_an_extraction_goes_on_while_a_copied_span_can_still_pass(prefix):
    stop = _plan_test(LAW_SQL)
    # the copy may still be at any place the document holds the prefix,
    # in any case, and its answer is that span stripped
    spans = [(start, start + len(prefix)) for start in range(len(LAWS))
             if LAWS[start:start + len(prefix)].lower() == prefix.lower()]
    assert stop.decide_any(copied_completions(LAWS, spans)) is Stop.CONTINUE


def test_an_extraction_stops_once_no_copied_span_can_pass():
    stop = _plan_test(LAW_SQL)
    start = LAWS.index("State of")
    spans = [(LAWS.index("state courts"), LAWS.index(" courts")),
             (start, start + len("State"))]
    assert stop.decide_any(copied_completions(LAWS.replace(
        "California", "New York"), spans)) is Stop.REJECTED


@pytest.mark.parametrize(("prefix", "decided"), [
    (" ", Stop.CONTINUE), (" Ye", Stop.CONTINUE), ("Ye", Stop.REJECTED),
    (" No", Stop.REJECTED)])
def test_a_map_answer_is_decided_on_its_text_with_its_leading_space(prefix, decided):
    stop = _plan_test("SELECT r.id FROM reviews r "
                      "WHERE AI_MAP(r.body, 'Is it kind?') = ' Yes'")
    assert stop.decide(prefix) is decided
