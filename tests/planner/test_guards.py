"""A guarded AI call asks the model only where its guard is TRUE.

An AI operator's guard says which rows the model is asked about; every
other row's answer is NULL. derive makes the condition under which a
CASE reads an answer the guard, and guards an AI filter in an OR with
the OR's other terms; in an OR of several AI filters, each is asked only
where the terms before it are not TRUE. Each test's transcript shows the
plan, the questions the model answered, and the
rows (helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without
from quail.physical import AiClassify, AiFilter
from quail.physical.plan import Refusal

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
    "long": [True, False, None, True, False, True],
})
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"
LONG = "AI_FILTER(PROMPT('long: {0}', r.body))"
EXTRACT = "AI.EXTRACT(r.body, {'w': 'Which word?'})"


def _answers(kind=("r0", "r2", "r4"), long=("r2", "r3", "r4")):
    """The model's answers: which rows each AI.IF accepts, and the rest."""
    def answer(asked, prompt, rows):
        (row,) = rows
        if prompt.startswith("kind:"):
            return row in kind
        if prompt.startswith("long:"):
            return row in long
        if asked == "label":
            return "a" if row in ("r0", "r2", "r4") else "b"
        if asked == "map":
            return f"summary {row}"
        return f"b{row[1:]}"
    return answer


def _session(rules=LOGICAL_RULES) -> quail.Session:
    found = quail.Session(CONFIG, tokenizer=TOKENIZER,
                          registry=built_in_registry(logical_rules=rules))
    found.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return found


@pytest.fixture
def session():
    found = _session()
    yield found
    found.close()


def _transcript(*queries, **answers) -> str:
    with Transcripts({"reviews": REVIEWS}, _answers(**answers)) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_a_case_over_a_classification_labels_only_the_rows_it_reads():
    assert_transcript(_transcript(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r",
        f"WITH t AS (SELECT r.id, r.long, {TONE} AS tone FROM reviews r) "
        "SELECT t.id, CASE WHEN t.long THEN t.tone END AS tone FROM t"), """\
--- SELECT r.id, CASE WHEN r.long THEN AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a',
    'b']) END AS t FROM reviews r ---
plan
  __quail_call_1 = ai_classify('tone: {0}', r.body, make_array('a', 'b')), guard: r.long
      IS TRUE
asked 3 of 6
  'tone: {0}': r0 'a', r3 'b', r5 'b'
rows 6
  r0, 'a'
  r1, NULL
  r2, NULL
  r3, 'b'
  r4, NULL
  r5, 'b'

--- WITH t AS (SELECT r.id, r.long, AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])
    AS tone FROM reviews r) SELECT t.id, CASE WHEN t.long THEN t.tone END AS tone FROM t
    ---
plan
  tone = ai_classify('tone: {0}', r.body, make_array('a', 'b')), guard: r.long IS TRUE
asked 3 of 6
  'tone: {0}': r0 'a', r3 'b', r5 'b'
rows 6
  r0, 'a'
  r1, NULL
  r2, NULL
  r3, 'b'
  r4, NULL
  r5, 'b'
""")


def test_a_case_over_a_map_writes_only_the_answers_it_reads():
    assert_transcript(_transcript(
        "SELECT r.id, CASE WHEN r.id <> 'r0' AND r.long "
        "THEN AI_MAP(r.body, 'Summarize.') END AS s FROM reviews r"), """\
--- SELECT r.id, CASE WHEN r.id <> 'r0' AND r.long THEN AI_MAP(r.body, 'Summarize.') END
    AS s FROM reviews r ---
plan
  __quail_call_1 = ai_map(r.body, 'Summarize.'), guard: r.id != 'r0' IS TRUE AND r.long
      IS TRUE
asked 2 of 6
  'Summarize.': r3 'summary r3', r5 'summary r5'
rows 6
  r0, NULL
  r1, NULL
  r2, NULL
  r3, 'summary r3'
  r4, NULL
  r5, 'summary r5'
""")


def test_a_case_over_an_extraction_is_null_where_its_test_is_not_true():
    assert_transcript(_transcript(
        f"SELECT r.id, CASE WHEN r.long THEN {EXTRACT} END AS e FROM reviews r"), """\
--- SELECT r.id, CASE WHEN r.long THEN AI.EXTRACT(r.body, {'w': 'Which word?'}) END AS e
    FROM reviews r ---
plan
  __quail_call_1 = ai_extract(r.body, named_struct('w', 'Which word?')), guard: r.long
      IS TRUE, returns: [__quail_call_1]
asked 3 of 6
  'Which word?': r0 'b0', r3 'b3', r5 'b5'
rows 6
  r0, {'w': 'b0'}
  r1, NULL
  r2, NULL
  r3, {'w': 'b3'}
  r4, NULL
  r5, {'w': 'b5'}
""")


def test_a_case_with_an_else_guards_its_call_by_the_condition():
    assert_transcript(_transcript(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} ELSE 'none' END AS t "
        "FROM reviews r"), """\
--- SELECT r.id, CASE WHEN r.long THEN AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a',
    'b']) ELSE 'none' END AS t FROM reviews r ---
plan
  __quail_call_1 = ai_classify('tone: {0}', r.body, make_array('a', 'b')), guard: r.long
      IS TRUE
asked 3 of 6
  'tone: {0}': r0 'a', r3 'b', r5 'b'
rows 6
  r0, 'a'
  r1, 'none'
  r2, 'none'
  r3, 'b'
  r4, 'none'
  r5, 'b'
""")


def test_a_case_and_its_answer_read_together_ask_every_row():
    assert_transcript(_transcript(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t, "
        f"{TONE} AS u FROM reviews r"), """\
--- SELECT r.id, CASE WHEN r.long THEN AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a',
    'b']) END AS t, AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) AS u FROM
    reviews r ---
plan
  u = ai_classify('tone: {0}', r.body, make_array('a', 'b'))
asked 6 of 6
  'tone: {0}': r0 'a', r1 'b', r2 'a', r3 'b', r4 'a', r5 'b'
rows 6
  r0, 'a', 'a'
  r1, NULL, 'b'
  r2, NULL, 'a'
  r3, 'b', 'b'
  r4, NULL, 'a'
  r5, 'b', 'b'
""")


def test_an_or_of_a_column_and_an_ai_filter_asks_only_where_the_column_is_not_true():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}",
        f"SELECT r.id FROM reviews r WHERE {KIND} OR (r.long AND r.id <> 'r0')"), """\
--- SELECT r.id FROM reviews r WHERE r.long OR AI_FILTER(PROMPT('kind: {0}', r.body))
    ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), guard: r.long IS NOT TRUE, filter:
      r.long OR __quail_answer_1, returns: []
asked 3 of 6
  'kind: {0}': r1 FALSE, r2 TRUE, r4 TRUE
rows 5
  r0
  r2
  r3
  r4
  r5

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) OR (r.long
    AND r.id <> 'r0') ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), guard: r.long AND r.id != 'r0' IS
      NOT TRUE, filter: __quail_answer_1 OR r.long AND r.id != 'r0', returns: []
asked 4 of 6
  'kind: {0}': r0 TRUE, r1 FALSE, r2 TRUE, r4 TRUE
rows 5
  r0
  r2
  r3
  r4
  r5
""")


def test_an_ai_filter_in_an_or_is_asked_everywhere_when_its_answer_is_read():
    assert_transcript(_transcript(
        # the SELECT list returns the answer
        f"SELECT r.id, {KIND} AS k FROM reviews r WHERE r.long OR {KIND}",
        # another term of the WHERE reads it
        f"SELECT r.id FROM reviews r WHERE (r.long OR {KIND}) AND {KIND} IS NOT NULL"),
        """\
--- SELECT r.id, AI_FILTER(PROMPT('kind: {0}', r.body)) AS k FROM reviews r WHERE r.long
    OR AI_FILTER(PROMPT('kind: {0}', r.body)) ---
plan
  k = ai_filter('kind: {0}', r.body)
asked 6 of 6
  'kind: {0}': r0 TRUE, r1 FALSE, r2 TRUE, r3 FALSE, r4 TRUE, r5 FALSE
rows 5
  r0, TRUE
  r2, TRUE
  r3, FALSE
  r4, TRUE
  r5, FALSE

--- SELECT r.id FROM reviews r WHERE (r.long OR AI_FILTER(PROMPT('kind: {0}', r.body)))
    AND AI_FILTER(PROMPT('kind: {0}', r.body)) IS NOT NULL ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body)
asked 6 of 11
  'kind: {0}': r0 TRUE, r1 FALSE, r2 TRUE, r3 FALSE, r4 TRUE, r5 FALSE
rows 5
  r0
  r2
  r3
  r4
  r5
""")


def test_an_or_of_two_ai_filters_asks_the_second_only_where_the_first_is_not_true():
    often = "AI_FILTER(PROMPT('long: {0}', r.body), {'selectivity': 0.9})"
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE {KIND} OR {LONG}",
        # the planner runs the term that passes more rows first
        f"SELECT r.id FROM reviews r WHERE {KIND} OR {often}",
        kind=("r0", "r3"), long=("r2", "r3", "r4")), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) OR
    AI_FILTER(PROMPT('long: {0}', r.body)) ---
plan
  __quail_answer_2 = ai_filter('long: {0}', r.body), guard: __quail_answer_1 IS NOT
      TRUE, filter: __quail_answer_1 OR __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('kind: {0}', r.body)
asked 10 of 12
  'kind: {0}': r0 TRUE, r1 FALSE, r2 FALSE, r3 TRUE, r4 FALSE, r5 FALSE
  'long: {0}': r1 FALSE, r2 TRUE, r4 TRUE, r5 FALSE
rows 4
  r0
  r2
  r3
  r4

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) OR
    AI_FILTER(PROMPT('long: {0}', r.body), {'selectivity': 0.9}) ---
plan
  __quail_answer_2 = ai_filter('long: {0}', r.body, named_struct('selectivity',
      Float64(0.9))), guard: __quail_answer_1 IS NOT TRUE, filter: __quail_answer_1 OR
      __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('kind: {0}', r.body)
asked 9 of 12
  'long: {0}': r0 FALSE, r1 FALSE, r2 TRUE, r3 TRUE, r4 TRUE, r5 FALSE
  'kind: {0}': r0 TRUE, r1 FALSE, r5 FALSE
rows 4
  r0
  r2
  r3
  r4
""")


def test_an_or_of_a_column_and_two_ai_filters_asks_each_where_no_earlier_term_is_true():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE r.long OR {KIND} OR {LONG}",
        kind=("r4",), long=("r1",)), """\
--- SELECT r.id FROM reviews r WHERE r.long OR AI_FILTER(PROMPT('kind: {0}', r.body)) OR
    AI_FILTER(PROMPT('long: {0}', r.body)) ---
plan
  __quail_answer_2 = ai_filter('long: {0}', r.body), guard: r.long IS NOT TRUE AND
      __quail_answer_1 IS NOT TRUE, filter: r.long OR __quail_answer_1 OR
      __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('kind: {0}', r.body)
asked 5 of 12
  'long: {0}': r1 TRUE, r2 FALSE, r4 FALSE
  'kind: {0}': r2 FALSE, r4 TRUE
rows 5
  r0
  r1
  r3
  r4
  r5
""")


def test_explain_says_which_rows_an_or_asks(session):
    physical = session.sql(
        f"SELECT r.id FROM reviews r WHERE r.long OR {KIND} OR {LONG}").explain()
    assert "asks only where r.long" in session.sql(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r"
    ).explain()
    assert "keeps every row for the later terms of its OR" in physical
    assert "asks only the rows no earlier term of its OR passed" in physical
    assert "returns the rows a term of its OR passed" in physical
    assert "passes each row where r.long" in physical


def test_stock_vllm_refuses_a_guarded_classification_and_a_guarded_ai_filter():
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="stock_vllm",
                     device="h100-sxm"),
        tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    plans = {
        "guard_needs_quail_backend": session.sql(
            f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t "
            "FROM reviews r").plan(),
        "or_filter_needs_quail_backend": session.sql(
            f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}").plan(),
    }
    session.close()
    for constraint, plan in plans.items():
        assert isinstance(plan, Refusal), constraint
        assert plan.constraint == constraint



def test_an_or_ranks_as_one_filter_among_the_filters_beside_it(session):
    kind = "AI_FILTER(PROMPT('kind: {0}', r.body), {'selectivity': 0.2})"
    long = "AI_FILTER(PROMPT('long: {0}', r.body), {'selectivity': 0.6})"
    for third_passes, printed in (
            # the third filter drops half the rows, more than the OR's 32%: it
            # runs first, then the OR's term that passes more rows
            (0.5, ["kind:", "long:", "third:"]),
            # it drops a tenth of the rows, so the OR runs first, its terms
            # together
            (0.9, ["third:", "kind:", "long:"])):
        third = (f"AI_FILTER(PROMPT('third: {{0}}', r.body), "
                 f"{{'selectivity': {third_passes}}})")
        query = session.sql(
            f"SELECT r.id FROM reviews r WHERE ({kind} OR {long}) AND {third}")
        # explain() prints the last filter to run first
        physical = query.explain().split("physical:")[1]
        assert sorted(printed, key=physical.index) == printed, third_passes


def test_a_guarded_ai_call_is_priced_on_the_rows_its_guard_passes(session):
    def classified_rows(sql):
        plan = session.sql(sql).plan()
        (node,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
        return node.spec.expected_rows, plan.estimated_seconds

    guarded = classified_rows(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r")
    everything = classified_rows(f"SELECT r.id, {TONE} AS t FROM reviews r")
    # r.long passes the default 20% of rows
    assert guarded[0] == pytest.approx(0.2 * everything[0])
    assert guarded[1] < everything[1]

    sql = f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}"
    (guarded,) = [node for node in session.sql(sql).plan().nodes
                  if isinstance(node, AiFilter)]
    unguarded = _session(logical_rules_without("derive.guard"))
    (everything,) = [node for node in unguarded.sql(sql).plan().nodes
                     if isinstance(node, AiFilter)]
    unguarded.close()
    # r.long IS NOT TRUE passes 1 - 20% of rows
    assert guarded.expected_rows == pytest.approx(0.8 * everything.expected_rows)


