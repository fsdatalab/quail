"""GROUP BY, aggregates, and HAVING: the Aggregate node, its runtime, the plan."""

import json

import numpy as np
import pyarrow as pa
import pytest
from helpers.planning import _plan, node_kinds
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.execution.result import QueryResult
from quail.execution.runtimes.relational import AggregateRuntime, aggregate_table
from quail.explain import explain
from quail.frontend.builder import docs
from quail.frontend.expressions import avg, col, count, max_, prompt, sum_
from quail.logical import AggregateCall, Aggregation, CompileError, HavingTest
from quail.physical import Aggregate, Sort
from quail.physical.base import input_ports


def _node(**fields):
    return Aggregate(node_id="aggregate", inputs=input_ports(()), **fields)


@pytest.mark.parametrize("literal", [np.int64(1), np.float32(1.5)])
def test_numpy_having_literal_survives_planning_and_execution(literal):
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", DocumentProvider.from_table(pa.table({
            "id": [0, 1, 2], "text": ["doc"] * 3, "group": ["a", "a", "b"],
        }), id_col="id"))
        query = (session.docs("docs").alias("r")
                 .ai_filter(prompt("keep: {0}", col("r.text")))
                 .group_by("r.group").agg(n=count()).having(count() > literal)
                 .select("r.group", "n"))
        json.dumps(query.plan().to_envelope(session.registry.codecs))
        assert _run(query, make_executor({"r": {"keep:": [True] * 3}})
                    ).to_rows() == [("a", 2)]


@pytest.mark.parametrize("function", ["sum", "min", "max"])
@pytest.mark.parametrize("dtype,values,literal,expected", [
    (pa.int64(), [2**53, 2**53 + 1, None], float(2**53), [1]),
    (pa.uint64(), [0, 2**63, 2**64 - 1], 2**63, [2]),
    (pa.float64(), [float(2**53), float(2**53 + 2), None], 2**53 + 1, [1]),
])
def test_having_preserves_large_numeric_values(
        function, dtype, values, literal, expected):
    table = pa.table({"group": range(len(values)), "value": pa.array(values, dtype)})
    node = _node(keys=("group",), aggregates=(("answer", function, "value"),),
                 having=(("answer", ">", literal),), columns=("group", "answer"))
    assert aggregate_table(table, node).to_pydict() == {
        "group": expected, "answer": [values[index] for index in expected]}


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype,values", [
    (pa.int64(), [2**63 - 1, 1]),
    (pa.int64(), [-2**63, -1]),
    (pa.uint64(), [2**64 - 1, 1]),
])
def test_integer_sum_does_not_wrap(dtype, values, grouped):
    table = pa.table({"group": ["a"] * len(values),
                      "value": pa.array(values, dtype)})
    node = _node(keys=("group",) if grouped else (),
                 aggregates=(("total", "sum", "value"),), columns=("total",))
    with pytest.raises(pa.ArrowInvalid):
        aggregate_table(table, node)


@pytest.mark.parametrize("dtype,values,expected", [
    (pa.int64(), [2**63 - 1, 1, -1, None], 2**63 - 1),
    (pa.int64(), [-2**63, -1, 1, None], -2**63),
    (pa.uint64(), [2**64 - 1, 0, None], 2**64 - 1),
    (pa.int32(), [2**31 - 1, 1, None], 2**31),
])
def test_integer_sum_keeps_exact_totals_and_other_aggregates(dtype, values, expected):
    table = pa.table({"group": ["a"] * len(values) + ["b"],
                      "value": pa.array([*values, None], dtype)})
    node = _node(keys=("group",), aggregates=(
        ("total", "sum", "value"), ("again", "sum", "value"),
        ("low", "min", "value"), ("n", "count", "value")),
        columns=("group", "total", "again", "low", "n"))
    result = aggregate_table(table, node)
    present = [value for value in values if value is not None]
    assert result.to_pylist() == [
        {"group": "a", "total": expected, "again": expected,
         "low": min(present), "n": len(present)},
        {"group": "b", "total": None, "again": None, "low": None, "n": 0}]
    assert result.schema.field("total").type == (
        pa.uint64() if pa.types.is_unsigned_integer(dtype) else pa.int64())


@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
def test_sql_integer_sum_cannot_hide_overflow_in_having(dialect):
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", DocumentProvider.from_table(pa.table({
            "id": [0, 1], "text": ["doc", "doc"],
            "amount": [2**63 - 1, 1],
        }), id_col="id"))
        query = session.sql(
            "SELECT SUM(r.amount) AS total FROM docs r "
            "WHERE AI_FILTER(PROMPT('keep: {0}', r.text)) "
            "HAVING SUM(r.amount) > 0", dialect=dialect)
        with pytest.raises(pa.ArrowInvalid):
            _run(query, make_executor({"r": {"keep:": [True, True]}})).collect()


def test_aggregate_table_groups_counts_and_filters_groups():
    table = pa.table({
        "lang": pa.array(["en", "fr", "en", "de", "en"]).dictionary_encode(),
        "stars": [5, 2, None, 4, 3],
        "r.id": ["a", "b", "c", "d", "e"],
    })
    node = _node(keys=("lang",),
                 aggregates=(("n", "count", None), ("rated", "count", "stars"),
                             ("mean", "avg", "stars"), ("top", "max", "stars"),
                             ("ids", "count_distinct", "r.id"),
                             ("n2", "count", None)),
                 columns=("lang", "n", "rated", "mean", "top", "ids", "n2"))
    result = aggregate_table(table, node)
    assert result.to_pydict() == {
        "lang": ["en", "fr", "de"], "n": [3, 1, 1], "rated": [2, 1, 1],
        "mean": [4.0, 2.0, 4.0], "top": [5, 2, 4], "ids": [3, 1, 1],
        "n2": [3, 1, 1]}
    kept = aggregate_table(table, _node(
        keys=("lang",), aggregates=(("n", "count", None), ("s", "sum", "stars")),
        having=(("n", ">", 1), ("s", "<=", 8)), columns=("lang", "s")))
    assert kept.to_pydict() == {"lang": ["en"], "s": [8]}
    whole = aggregate_table(table, _node(
        aggregates=(("n", "count", None), ("mean", "avg", "stars")),
        columns=("mean", "n")))
    assert whole.to_pydict() == {"mean": [3.5], "n": [5]}
    result = AggregateRuntime().execute(
        _node(keys=("lang",), aggregates=(("n", "count", None),),
              columns=("n", "lang")),
        {"rows": QueryResult.from_table(table)}, None)
    assert result.outputs["rows"].to_rows() == [(3, "en"), (1, "fr"), (1, "de")]
    assert (result.metrics.input_rows, result.metrics.output_rows) == (5, 3)


@pytest.fixture()
def catalog():
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_table(pa.table({
        "id": ["x", "y"], "review": ["x", "y"], "lang": ["en", "fr"],
        "stars": [1, 2]}), id_col="id"))
    return cat


def test_group_by_ends_the_plan_in_an_aggregate(catalog):
    logical = (docs(catalog, "reviews").alias("r")
               .ai_filter(prompt("flag: {0}", col("r.review")), selectivity=0.5)
               .group_by("r.lang").agg(n=count(), s=avg("r.stars"))
               .having(count() > 1, max_("r.stars") < 5)
               .order_by(col("n").desc()).limit(2).select("r.lang", "n", "s"))
    root = logical.root
    assert [type(node).__name__ for node in logical.walk()][-3:] == [
        "Project", "Aggregate", "Result"]
    aggregate_node = root.input
    assert aggregate_node.children() == (logical.projection,)
    assert aggregate_node.with_expressions(
        aggregate_node.expressions()) == aggregate_node
    assert aggregate_node.with_children((logical.projection,)) == aggregate_node
    assert tuple(ref.column for ref in aggregate_node.output_schema()) == (
        "lang", "n", "s")
    assert [str(column) for column in root.input.aggregates] == [
        "n = count(*)", "s = avg(r.stars)", "__having_1 = max(r.stars)"]
    assert [str(test) for test in root.input.having] == [
        "n > 1", "__having_1 < 5"]
    assert root.result_columns() == ("r.lang", "n", "s")
    plan = _plan(logical, {"r": [400] * 100})
    assert node_kinds(plan)[-3:] == ["Project", "Aggregate", "Sort"]
    assert "filter_limit" not in plan.settings
    aggregate = next(node for node in plan.nodes if isinstance(node, Aggregate))
    assert aggregate.keys == ("r.lang",)
    assert aggregate.having == (("n", ">", 1), ("__having_1", "<", 5))
    assert aggregate.columns == ("r.lang", "n", "s")
    sort = next(node for node in plan.nodes if isinstance(node, Sort))
    assert (sort.keys, sort.columns, sort.fetch) == (
        (("n", True, True),), ("r.lang", "n", "s"), 2)
    text = explain(logical, plan)
    assert "Aggregate: group by r.lang; n = count(*), s = avg(r.stars)" in text
    assert "having n > 1 and __having_1 < 5" in text
    assert "Aggregate: r.lang, n, s" in text.split("physical:", 1)[1]

    with pytest.raises(CompileError, match="neither a GROUP BY key"):
        (docs(catalog, "reviews").alias("r")
         .ai_filter(prompt("flag: {0}", col("r.review")))
         .agg(n=count()).select("r.lang", "n"))
    with pytest.raises(CompileError, match="names a key or an aggregate"):
        (docs(catalog, "reviews").alias("r")
         .ai_filter(prompt("flag: {0}", col("r.review")))
         .group_by("r.lang").agg(n=count()).order_by("r.stars")
         .select("r.lang", "n"))
    with pytest.raises(CompileError, match="takes count"):
        docs(catalog, "reviews").alias("r").agg(n="count")
    Aggregation(("r.lang",), (AggregateCall("count", None, "n"),),
                ("r.lang", "n"), (HavingTest(AggregateCall("count", None, "n"),
                                             ">", 1),)).validate(("r.lang",))
    with pytest.raises(CompileError, match="does not compute"):
        Aggregation(("r.lang",), (AggregateCall("count", None, "n"),),
                    ("r.lang", "n"),
                    (HavingTest(AggregateCall("sum", "r.lang", "n"), ">", 1),)
                    ).validate(("r.lang",))


@pytest.fixture()
def session():
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
        "stars": [5, 2, 4, None, 3, 5],
        "lang": ["en", "en", "fr", "en", "en", "de"],
    }), id_col="id"))
    yield session
    session.close()


TRUTH = {"r": {"q:": [1, 1, 1, 0, 1, 1]}}


@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
@pytest.mark.parametrize("argument", ["stars", "r.stars"])
def test_aggregate_arguments_read_source_columns_before_output_aliases(
    session, dialect, argument,
):
    query = session.sql(f"""
        SELECT r.lang AS stars, COUNT({argument}) AS n,
            COUNT(DISTINCT {argument}) AS distinct_values,
            SUM({argument}) AS total, AVG({argument}) AS mean
        FROM reviews r WHERE AI_FILTER(PROMPT('q: {{0}}', r.review))
        GROUP BY r.lang ORDER BY r.lang
    """, dialect=dialect)
    result = _run(query, make_executor({"r": {"q:": [1] * 6}}))
    assert result.to_rows() == [
        ("de", 1, 1, 5, 5.0), ("en", 3, 3, 10, 10 / 3), ("fr", 1, 1, 4, 4.0)]


@pytest.mark.parametrize("dialect, counts", [
    ("snowflake", [1, 1, 1, 1, 1, 1]), ("bq", [1, 1, 4]),
])
def test_group_by_resolves_source_and_alias_collisions_by_dialect(
    session, dialect, counts,
):
    query = session.sql("""
        SELECT r.lang AS stars, COUNT(*) AS n
        FROM reviews r WHERE AI_FILTER(PROMPT('q: {0}', r.review))
        GROUP BY r.lang, stars ORDER BY n
    """, dialect=dialect)
    result = _run(query, make_executor({"r": {"q:": [1] * 6}})).collect()
    assert result.column("n").to_pylist() == counts


def test_group_by_runs_on_the_result(session):
    result = _run(session.sql("""
        SELECT r.lang, COUNT(*) AS n, AVG(r.stars) AS mean, SUM(r.stars) AS total
        FROM reviews r
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
        GROUP BY r.lang
        HAVING COUNT(*) >= 1 AND mean > 2.5
        ORDER BY n DESC, r.lang
    """), make_executor(TRUTH))
    assert result.columns == ["r.lang", "n", "mean", "total"]
    assert result.to_rows() == [("en", 3, 10 / 3, 10), ("de", 1, 5.0, 5),
                                ("fr", 1, 4.0, 4)]
    assert "Aggregate: r.lang, n, mean, total" in result.explain()

    whole = _run(session.sql("""
        SELECT COUNT(*) AS n, MAX(r.stars) AS top FROM reviews r
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
    """), make_executor(TRUTH))
    assert whole.to_rows() == [(5, 5)]

    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q: {0}", col("r.review")))
             .group_by("r.lang").agg(n=count(), total=sum_("r.stars"))
             .having(count() > 1).select("r.lang", "total"))
    assert _run(built, make_executor(TRUTH)).to_rows() == [("en", 10)]

    for sql, fragment in (
        ("SELECT r.lang, COUNT(*) FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review)) GROUP BY r.lang", "AS name"),
        ("SELECT r.lang, COUNT(*) AS n FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review))", "not in GROUP BY"),
        ("SELECT r.lang, COUNT(*) AS n FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review)) GROUP BY r.lang HAVING r.lang = 'en'",
         "tests an aggregate"),
        ("SELECT r.lang, COUNT(*) AS n FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review)) GROUP BY r.lang HAVING n > 'two'",
         "with a number"),
        ("SELECT r.lang, COUNT(*) AS n FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review)) GROUP BY r.lang ORDER BY r.stars",
         "names a key or an aggregate"),
        ("SELECT COUNT(*) + 1 AS n FROM reviews r WHERE AI_FILTER("
         "PROMPT('q: {0}', r.review))", "direct COUNT"),
    ):
        with pytest.raises(CompileError, match=fragment):
            session.sql(sql)


@pytest.mark.parametrize("dialect, expected", [
    ("snowflake", []), ("bq", [("en", 4)]),
])
def test_having_aggregate_resolves_source_and_alias_collisions_by_dialect(
    session, dialect, expected,
):
    query = session.sql("""
        SELECT r.lang AS stars, COUNT(*) AS n
        FROM reviews r WHERE AI_FILTER(PROMPT('q: {0}', r.review))
        GROUP BY r.lang HAVING COUNT(stars) > 3
    """, dialect=dialect)
    result = _run(query, make_executor({"r": {"q:": [1] * 6}}))
    assert result.to_rows() == expected


def test_aggregate_internal_columns_do_not_replace_user_values():
    table = pa.table({
        "__row": [0.9, 0.9],
        "count_all": ["a", "a"],
        "row_position": [2, 4],
    })
    node = _node(
        keys=("count_all",),
        aggregates=(("mean", "avg", "__row"), ("n", "count", None),
                    ("first", "min", "row_position")),
        columns=("count_all", "mean", "n", "first"))
    assert aggregate_table(table, node).to_pydict() == {
        "count_all": ["a"], "mean": [0.9], "n": [2], "first": [2]}


@pytest.mark.parametrize("answers, expected", [([1, 1, 1, 0, 1, 1], 5),
                                                ([0] * 6, 0)])
def test_count_star_without_value_columns(session, answers, expected):
    sql = session.sql("""
        SELECT COUNT(*) AS n FROM reviews r
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
    """)
    python = (session.docs("reviews").alias("r")
              .ai_filter(prompt("q: {0}", col("r.review")))
              .agg(n=count()).select("n"))
    for query in (sql, python):
        result = _run(query, make_executor({"r": {"q:": answers}}))
        assert result.to_rows() == [(expected,)]


def test_grouped_select_order_does_not_duplicate_input_columns(session):
    results = []
    for columns in ("COUNT(r.lang) AS n, r.lang", "r.lang, COUNT(r.lang) AS n"):
        query = session.sql(f"""
            SELECT {columns} FROM reviews r
            WHERE AI_FILTER(PROMPT('q: {{0}}', r.review))
            GROUP BY r.lang ORDER BY r.lang
        """)
        result = _run(query, make_executor(TRUTH)).collect()
        results.append(result.select(["r.lang", "n"]).to_pydict())
    assert results[0] == results[1] == {"r.lang": ["de", "en", "fr"],
                                      "n": [1, 3, 1]}
    with pytest.raises(CompileError, match="projection names must be unique"):
        session.sql("""
            SELECT r.id, r.id FROM reviews r
            WHERE AI_FILTER(PROMPT('q: {0}', r.review))
        """)


@pytest.mark.parametrize("key", [None, "category", "value"])
@pytest.mark.parametrize("keep", [False, True])
def test_aggregates_ignore_untyped_null_values_with_or_without_groups(key, keep):
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", DocumentProvider.from_table(pa.table({
            "id": [1, 2], "body": ["text", "more text"],
            "category": ["a", "b"], "value": [None, None],
        }), id_col="id"))
        selection = "" if key is None else f"r.{key}, "
        group = "" if key is None else f"GROUP BY r.{key} ORDER BY r.{key}"
        query = session.sql(f"""
            SELECT {selection} COUNT(DISTINCT r.value) AS distinct_values,
                COUNT(r.value) AS present, COUNT(*) AS total,
                SUM(r.value) AS summed, AVG(r.value) AS mean,
                MIN(r.value) AS minimum, MAX(r.value) AS maximum
            FROM docs r WHERE AI_FILTER(PROMPT('q: {{0}}', r.body)) {group}
        """)
        result = _run(query, make_executor({"r": {"q:": [keep, keep]}}))
        if key is None:
            expected = [(0, 0, 2 if keep else 0, None, None, None, None)]
        elif not keep:
            expected = []
        elif key == "value":
            expected = [(None, 0, 0, 2, None, None, None, None)]
        else:
            expected = [(value, 0, 0, 1, None, None, None, None)
                        for value in ("a", "b")]
        assert result.to_rows() == expected
        if key == "value":
            assert pa.types.is_null(result.schema.field("r.value").type)
        assert pa.types.is_null(result.schema.field("minimum").type)
        assert pa.types.is_null(result.schema.field("maximum").type)
