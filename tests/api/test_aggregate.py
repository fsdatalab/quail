"""GROUP BY, aggregates, and HAVING run in DataFusion on a model region's rows."""


import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER, _run, make_executor

import quail
from quail.common.expressions import CompileError


@pytest.fixture()
def session():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
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
    # DataFusion groups the rows the model region returns
    explained = result.explain()
    assert "AggregateExec: mode=FinalPartitioned, gby=[lang@0 as lang]" in explained
    assert explained.index("ModelRegion") < explained.index("model region:")

    whole = _run(session.sql("""
        SELECT COUNT(*) AS n, MAX(r.stars) AS top FROM reviews r
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
    """), make_executor(TRUTH))
    assert whole.to_rows() == [(5, 5)]

    # HAVING may test an aggregate the query does not return
    having = session.sql("""
        SELECT r.lang, SUM(r.stars) AS total FROM reviews r
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
        GROUP BY r.lang HAVING COUNT(*) > 1
    """)
    assert _run(having, make_executor(TRUTH)).to_rows() == [("en", 10)]

    # DataFusion names an aggregate the query leaves unnamed
    unnamed = _run(session.sql(
        "SELECT r.lang, COUNT(*) FROM reviews r WHERE AI_FILTER("
        "PROMPT('q: {0}', r.review)) GROUP BY r.lang ORDER BY r.lang"),
        make_executor(TRUTH))
    assert unnamed.columns == ["r.lang", "count(*)"]
    with pytest.raises(CompileError, match="not in GROUP BY"):
        session.sql("SELECT r.lang, COUNT(*) AS n FROM reviews r WHERE AI_FILTER("
                    "PROMPT('q: {0}', r.review))")
