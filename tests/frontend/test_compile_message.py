"""Compile errors show the query's problem, not DataFusion's rule wrappers."""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.compile import compile_message
from quail.common.expressions import CompileError

WRAPPERS = [
    "{}",
    "Error during planning: {}",
    "lower_ai_calls\ncaused by\nError during planning: {}",
    "optimizer\ncaused by\nlower_ai_calls\ncaused by\nExternal error: "
    "Error during planning: CompileError: {}",
    "lower_ai_calls\n  caused by\n  Error during planning: {}",
    "Error during planning: External error: Error during planning: {}",
]

# reason, SQL, expected message
EXPLANATIONS = [
    ("Invalid function 'nope'.\nDid you mean 'now'?", "", "unknown function 'nope'"),
    ("Invalid function 'mod'.", "SELECT MOD(stars, 2) FROM reviews",
     "MOD is not a DataFusion function; write stars % 2"),
    ('column "r.stars" must appear in the GROUP BY clause', "",
     "r.stars is in SELECT but not in GROUP BY and not aggregated"),
    # the details on later lines of the underlying error stay
    ("A model call cannot read this input:\n  missing column: r.body", "",
     "A model call cannot read this input:\n  missing column: r.body"),
]

# predicate, expected message
INVALID_ARGUMENTS = [
    ("AI.IF(r.body)", "AI.IF needs a PROMPT as its first argument."),
    ("AI.SCORE(r.body) > 0.5", "AI.SCORE needs a PROMPT as its first argument."),
    ("AI.IF(PROMPT(1, r.body))",
     "PROMPT needs a string template as its first argument."),
    ("AI.IF(PROMPT('q: {0}', r.body), {'selectivity': 'bad'})",
     "The selectivity option must be a number."),
    ("AI.IF(PROMPT('q: {0}', r.body), 3)",
     "AI options must be an object such as {'selectivity': 0.3}."),
    ("AI.IF(PROMPT('q: {0}', r.body), {'typo': 1})",
     "Unknown option 'typo' must be replaced with one of ['anchor', 'selectivity']."),
    ("AI.CLASSIFY(r.body, ['a', 'b'], {'probabilities': 'yes'}) = 'a'",
     "The probabilities option must be true or false."),
]


def test_rule_wrappers_are_removed_and_the_reason_or_its_explanation_is_kept():
    reason = "AI.CLASSIFY needs at least one label."
    for wrapper in WRAPPERS:
        error = RuntimeError(wrapper.format(reason))
        assert compile_message(error, "") == reason, wrapper
    for reason, sql, expected in EXPLANATIONS:
        error = RuntimeError(
            f"lower_ai_calls\ncaused by\nError during planning: {reason}")
        assert compile_message(error, sql) == expected, reason
    # words that name an error kind stay when they are inside the reason
    reason = "unknown column 'Error during planning: body'"
    assert compile_message(RuntimeError(reason), "") == reason
    assert compile_message(RuntimeError(), "") == ""


def test_a_session_reports_an_invalid_query_in_one_plain_sentence():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        context = quail.SessionContext(session)
        context.register("reviews", quail.DocumentProvider.from_table(
            pa.table({"id": ["r0"], "body": ["text"]}), id_col="id"))
        for predicate, expected in INVALID_ARGUMENTS:
            with pytest.raises(CompileError) as error:
                session.sql(f"SELECT r.id FROM reviews r WHERE {predicate}").plan()
            assert str(error.value) == expected, predicate
        for method in ("sql", "region_plans"):
            with pytest.raises((RuntimeError, CompileError)) as error:
                getattr(context, method)(
                    "SELECT r.id FROM reviews r WHERE AI.IF(r.body)")
            assert str(error.value) == (
                "AI.IF needs a PROMPT as its first argument."), method
        # Physical EXPLAIN does not execute or derive model regions, so use an
        # invalid column reference that DataFusion rejects during SQL planning.
        with pytest.raises(RuntimeError) as error:
            context.explain("SELECT missing FROM reviews r")
        assert str(error.value) == (
            "No field named missing.\nValid fields are r.id, r.body.")
