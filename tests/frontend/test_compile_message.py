"""Compile errors show the query's problem, not DataFusion's rule wrappers."""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.compile import compile_message
from quail.common.expressions import CompileError


@pytest.mark.parametrize("wrapper", [
    "{}",
    "Error during planning: {}",
    "lower_ai_calls\ncaused by\nError during planning: {}",
    "optimizer\ncaused by\nlower_ai_calls\ncaused by\nExternal error: "
    "Error during planning: CompileError: {}",
    "lower_ai_calls\n  caused by\n  Error during planning: {}",
    "Error during planning: External error: Error during planning: {}",
])
def test_rule_wrappers_are_removed_without_losing_the_reason(wrapper):
    reason = "AI.CLASSIFY needs at least one label."
    assert compile_message(RuntimeError(wrapper.format(reason)), "") == reason


def test_multiline_details_of_the_underlying_error_are_kept():
    reason = "A model call cannot read this input:\n  missing column: r.body"
    error = RuntimeError(f"lower_ai_calls\ncaused by\nError during planning: {reason}")
    assert compile_message(error, "") == reason


def test_error_kind_words_inside_the_reason_are_not_removed():
    reason = "unknown column 'Error during planning: body'"
    assert compile_message(RuntimeError(reason), "") == reason


def test_an_empty_error_message_does_not_raise_another_error():
    assert compile_message(RuntimeError(), "") == ""


@pytest.mark.parametrize("reason, sql, expected", [
    ("Invalid function 'nope'.\nDid you mean 'now'?", "", "unknown function 'nope'"),
    ("Invalid function 'mod'.", "SELECT MOD(stars, 2) FROM reviews",
     "MOD is not a DataFusion function; write stars % 2"),
    ('column "r.stars" must appear in the GROUP BY clause', "",
     "r.stars is in SELECT but not in GROUP BY and not aggregated"),
])
def test_sql_explanations_are_preserved_after_removing_wrappers(reason, sql, expected):
    error = RuntimeError(f"lower_ai_calls\ncaused by\nError during planning: {reason}")
    assert compile_message(error, sql) == expected


@pytest.mark.parametrize("predicate, expected", [
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
])
def test_invalid_ai_arguments_report_one_plain_sentence(predicate, expected):
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            pa.table({"id": ["r0"], "body": ["text"]}), id_col="id"))
        with pytest.raises(CompileError) as error:
            session.sql(f"SELECT r.id FROM reviews r WHERE {predicate}").plan()
    assert str(error.value) == expected


@pytest.mark.parametrize("method", ["sql", "region_plans"])
def test_session_context_removes_rule_wrappers(method):
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        context = quail.SessionContext(session)
        context.register("reviews", quail.DocumentProvider.from_table(
            pa.table({"id": ["r0"], "body": ["text"]}), id_col="id"))
        with pytest.raises((RuntimeError, CompileError)) as error:
            getattr(context, method)(
                "SELECT r.id FROM reviews r WHERE AI.IF(r.body)")
    assert str(error.value) == "AI.IF needs a PROMPT as its first argument."


def test_session_context_explain_removes_schema_error_wrapper():
    # Physical EXPLAIN does not execute or derive model regions, so use an
    # invalid column reference that DataFusion rejects during SQL planning.
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        context = quail.SessionContext(session)
        context.register("reviews", quail.DocumentProvider.from_table(
            pa.table({"id": ["r0"], "body": ["text"]}), id_col="id"))
        with pytest.raises(RuntimeError) as error:
            context.explain("SELECT missing FROM reviews r")
    assert str(error.value) == (
        "No field named missing.\nValid fields are r.id, r.body.")
