"""GROUP BY, aggregates, and HAVING: the Aggregate node, its runtime, the plan."""

import pyarrow as pa
import pytest
from test_planner import _plan, node_kinds, tok
from test_session import CONFIG, _run, fake_tok, make_executor

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.execution.result import QueryResult
from quail.execution.runner import AggregateRuntime, aggregate_table
from quail.explain import explain
from quail.frontend.builder import avg, col, count, docs, max_, prompt, sum_
from quail.logical import AggregateCall, Aggregation, CompileError, HavingTest
from quail.physical import Aggregate, Sort
from quail.physical.base import input_ports


def _node(**fields):
    return Aggregate(node_id="aggregate", inputs=input_ports(()), **fields)


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
    logical = (docs(catalog, "reviews", tok).alias("r")
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
        (docs(catalog, "reviews", tok).alias("r")
         .ai_filter(prompt("flag: {0}", col("r.review")))
         .agg(n=count()).select("r.lang", "n"))
    with pytest.raises(CompileError, match="names a key or an aggregate"):
        (docs(catalog, "reviews", tok).alias("r")
         .ai_filter(prompt("flag: {0}", col("r.review")))
         .group_by("r.lang").agg(n=count()).order_by("r.stars")
         .select("r.lang", "n"))
    with pytest.raises(CompileError, match="takes count"):
        docs(catalog, "reviews", tok).alias("r").agg(n="count")
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
