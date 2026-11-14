"""A planner cache must describe the actual document values and row order."""

from contextlib import contextmanager

import pyarrow as pa
import pytest
from helpers.inputs import CountingProvider, RecordingRegions, snapshot_inputs
from helpers.sessions import CONFIG, TOKENIZER, encoder

import quail

REVIEWS = pa.table({
    # Put unused columns before required ones to exercise projected hint indices.
    "unused": [b"x" * 1024] * 4,
    "id": ["r0", "r1", "r2", "r3"],
    "body": ["First Review", "Second Review", "Third Review", "Fourth Review"],
    "other": ["One", "Two", "Three", "Four"],
    "__quail_expr_0": ["user zero", "user one", "user two", "user three"],
})


@contextmanager
def input_context():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    recorder = RecordingRegions()
    context = quail.SessionContext(session, run_query=recorder)
    provider = CountingProvider(quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    context.register("reviews", provider)
    try:
        yield session, context, recorder, provider
    finally:
        session.close()


def test_custom_engine_inspection_does_not_materialize_named_tables():
    from types import SimpleNamespace

    import _quail_native

    from quail.api.context import ipc_bytes
    from quail.specs import QWEN3_4B_FP8

    engine = SimpleNamespace(turn=tuple(QWEN3_4B_FP8.turn),
                             layout=QWEN3_4B_FP8.prompt_layout, rules=None)
    context = _quail_native.NativeSession(engine)

    def forbidden(_columns):
        pytest.fail("inspection without a native planner read a named table")

    context.register("reviews", ipc_bytes(REVIEWS.schema.empty_table()),
                     forbidden, "id", "reviews")
    (_, planned), = context.plans(document_query("r.body"))
    assert planned is None


def document_query(expression):
    return ("SELECT r.id FROM reviews r "
            f"WHERE AI_FILTER(PROMPT('q1: {{0}}', {expression}))")


def run_inputs(session, context, recorder, sql, mode):
    if mode == "plan":
        return snapshot_inputs(session.sql(sql))
    recorder.inputs.clear()
    context.sql(sql)
    assert len(recorder.inputs) == 1
    return recorder.inputs[0]


def test_computed_documents_do_not_reuse_another_expressions_tokens():
    for functions in (("LOWER", "UPPER"), ("UPPER", "LOWER")):
        for mode in ("plan", "run"):
            with input_context() as (session, context, recorder, _):
                for function in functions:
                    case = (functions, mode, function)
                    sql = document_query(f"{function}(r.body)")
                    actual = run_inputs(session, context, recorder, sql, mode)
                    with input_context() as (fresh, fresh_context, fresh_recorder, _):
                        expected = run_inputs(
                            fresh, fresh_context, fresh_recorder, sql, mode)
                    assert actual == expected, case
                    scan, = actual.values()
                    transform = str.lower if function == "LOWER" else str.upper
                    texts = [transform(text)
                             for text in REVIEWS.column("body").to_pylist()]
                    assert scan["tokens"] == [
                        encoder()(text) for text in texts], case
                    assert all(len(column) == len(scan["tokens"])
                               for column in scan["columns"].values()), case


def test_generated_expression_names_preserve_registered_user_columns():
    with input_context() as (_, context, recorder, _):
        result = context.sql(
            "SELECT r.id, r.__quail_expr_0 FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', LOWER(r.body))) ORDER BY r.id"
        )
        assert result.column(0).to_pylist() == REVIEWS.column("id").to_pylist()
        assert result.column(1).to_pylist() == (
            REVIEWS.column("__quail_expr_0").to_pylist())
        scan, = recorder.inputs[0].values()
        assert scan["tokens"] == [encoder()(text.lower())
                                  for text in REVIEWS.column("body").to_pylist()]


def test_named_tables_materialize_only_required_columns_once():
    for self_join in (False, True):
        with input_context() as (_, context, _, provider):
            sql = (
                "SELECT r.id, s.id FROM reviews r JOIN reviews s "
                "ON AI_FILTER(PROMPT('q1: {0} {1}', r.body, s.other))"
                if self_join else document_query("r.body")
            )
            # Repeated queries cover both token-store construction and cache hits.
            for _ in range(2):
                provider.requests.clear()
                result = context.sql(sql)
                assert result.num_rows == (16 if self_join else 4)
                expected = ("id", "body", "other") if self_join else ("id", "body")
                assert provider.requests == [expected], self_join
    # an unchanged document column keeps the token buffers of the first read
    with input_context() as (session, _, _, provider):
        sql = document_query("r.body")
        first = session.sql(sql).token_inputs()["r"].tokens.arrow_array
        provider.requests.clear()
        second = session.sql(sql).token_inputs()["r"].tokens.arrow_array
        assert first.to_pylist() == second.to_pylist()
        assert first.chunk(0).values.buffers()[1].address == (
            second.chunk(0).values.buffers()[1].address)
        assert provider.requests == [("id", "body")]


def _check_join_hint(projected, warm_cache):
    """Check a hinted read, then a whole read, of the reviews."""
    case = f"projected={projected}, warm_cache={warm_cache}"
    with input_context() as (_, context, recorder, provider):
        context.register("picked", quail.DocumentProvider.from_table(
            pa.table({"id": ["p0", "p1"], "review": ["r1", "r3"]}), id_col="id"))
        expression = "LOWER(r.body)" if projected else "r.body"
        whole = document_query(expression)
        if warm_cache:
            context.sql(whole)
        recorder.inputs.clear()
        hinted = (
            "SELECT r.id FROM picked p JOIN reviews r ON r.id = p.review "
            f"WHERE AI_FILTER(PROMPT('q1: {{0}}', {expression}))"
        )
        assert "ModelRegionExec: hint=[DynamicFilter" in context.explain(hinted), case
        provider.requests.clear()
        assert sorted(context.sql(hinted).column(0).to_pylist()) == ["r1", "r3"], case
        if not projected:
            assert provider.requests == [("id", "body")], case
        scan, = recorder.inputs[-1].values()
        ids = scan["columns"]["id"]
        texts = dict(zip(REVIEWS.column("id").to_pylist(),
                         REVIEWS.column("body").to_pylist()))
        expected = [texts[key].lower() if projected else texts[key] for key in ids]
        assert scan["tokens"] == [encoder()(text) for text in expected], case
        if projected:
            # Materialized inputs are pruned before tokenization, not just by
            # the Python executor, so this must test a genuinely smaller store.
            assert ids == ["r1", "r3"], case
        recorder.inputs.clear()
        assert context.sql(whole).num_rows == 4
        scan, = recorder.inputs[-1].values()
        assert scan["columns"]["id"] == REVIEWS.column("id").to_pylist(), case
        assert scan["tokens"] == [encoder()(text.lower() if projected else text)
                                  for text in REVIEWS.column("body").to_pylist()], case


def test_join_hints_keep_tokens_and_values_aligned():
    for projected in (False, True):
        for warm_cache in (False, True):
            _check_join_hint(projected, warm_cache)
