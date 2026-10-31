"""GROUP BY, aggregates, and HAVING: the Aggregate node, its runtime, the plan."""


import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail.execution.runtimes.relational import aggregate_table
from quail.frontend.expressions import col, count, prompt, sum_
from quail.logical import CompileError
from quail.physical import Aggregate
from quail.physical.base import input_ports


def _node(**fields):
    return Aggregate(node_id="aggregate", inputs=input_ports(()), **fields)


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
    ):
        with pytest.raises(CompileError, match=fragment):
            session.sql(sql)


@pytest.mark.parametrize("grouped", [True])
@pytest.mark.parametrize("dtype,values", [(pa.int64(), [2**63 - 1, 1])])
def test_an_integer_sum_raises_instead_of_wrapping(dtype, values, grouped):
    table = pa.table({"group": ["a"] * len(values), "value": pa.array(values, dtype)})
    node = _node(keys=("group",) if grouped else (),
                 aggregates=(("total", "sum", "value"),), columns=("total",))
    with pytest.raises(pa.ArrowInvalid):
        aggregate_table(table, node)
