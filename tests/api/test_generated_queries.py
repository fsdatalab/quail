"""Generated AI queries return the rows plain SQL computes for them."""

import pytest
from helpers.generated import Queries, ai_sql, plain_sql, tables, truth
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry

pytest.importorskip("_quail_native")

NUM_QUERIES = 40
# Quail's logical rewrites, and the executor's ways to ask fewer rows
QUAIL_RULES = ("merge_repeated_ai_calls", "derive",
               "place_ai_nodes")
EARLY_STOPS = ("should_stop_answers_early", "should_deduplicate",
               "should_push_down_limit")


def _context(seed, optimized):
    registry = built_in_registry() if optimized else built_in_registry(
        logical_rules=tuple(rule for rule in LOGICAL_RULES
                            if rule not in QUAIL_RULES),
        physical_rules=tuple(rule for rule in PHYSICAL_RULES
                             if rule not in EARLY_STOPS))
    session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=registry)
    oracle = Oracle(truth=truth)
    oracle.catalog = session.catalog
    context = quail.SessionContext(
        session, physical_executor=oracle_executor(oracle))
    for name, table in tables(seed).items():
        context.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    return session, context


def _rows(table):
    rows = zip(*(column.to_pylist() for column in table.columns))
    return sorted(rows, key=repr)


@pytest.mark.parametrize("seed", [0, 1, 4, 7, 40])
@pytest.mark.parametrize("optimized", [False, True], ids=["plain", "optimized"])
def test_generated_queries_return_the_rows_plain_sql_computes(
        seed, optimized, monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN",
                       "true" if optimized else "false")
    session, context = _context(seed, optimized)
    queries = Queries(seed)
    try:
        for _ in range(NUM_QUERIES):
            sql = queries.query()
            expected = _rows(context.sql(plain_sql(sql)))
            assert _rows(context.sql(ai_sql(sql))) == expected, sql
    finally:
        session.close()
