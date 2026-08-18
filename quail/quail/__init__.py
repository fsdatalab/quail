"""Quail: a declarative query engine for AI_FILTER and AI_JOIN.

    import quail

    sess = quail.Session(quail.EngineConfig(gpus=1))
    sess.register("reviews", quail.DocumentProvider.from_parquet(
        "imdb.parquet", id_col="id"))
    q = sess.sql(\"\"\"
        SELECT r.id FROM reviews r
        WHERE AI_FILTER(PROMPT('This review is negative: {0}',
                               r.review), {'selectivity': 0.3})
    \"\"\")
    print(q.explain())
    res = q.run()
"""

from quail.builder import col, prompt
from quail.catalog import DocumentProvider
from quail.planner.plan import EngineConfig
from quail.runtime import Query, RefusalError, Result, Session

__all__ = ["col", "prompt", "DocumentProvider", "EngineConfig",
           "Query", "RefusalError", "Result", "Session"]
