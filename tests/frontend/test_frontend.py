"""The SQL front end: dialects, plan shapes, and named errors."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.planning import compile_sql

from quail.api.compile import SQLDialect
from quail.builtins import built_in_registry
from quail.catalog import Catalog, DocumentProvider
from quail.logical import (
    ColumnRef,
    CompileError,
    Join,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
    join_conditions,
    join_outer_input,
)
from quail.planner import native
from quail.planner.optimizer.context import PlanningContext


@pytest.fixture()
def catalog(tmp_path):
    cat = Catalog()
    for name, columns in (("reviews", ["id", "review"]),
                          ("products", ["asin", "description"]),
                          ("threads", ["id", "thread"])):
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table({c: ["x", "y"] for c in columns}), path)
        cat.register(name, DocumentProvider.from_parquet(path, id_col=columns[0]))
    return cat


FILTER_JOIN_SQL = """
    SELECT r.id, p.asin
    FROM reviews r
    JOIN products p
      ON AI_FILTER(PROMPT('Review {0} discusses product {1}',
                          r.review, p.description),
                   {'selectivity': 0.05})
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.review),
                    {'selectivity': 0.3})
"""

THREE_WAY_TEMPLATE = ("Review {0} praises the thread in {1} and the "
                      "thread recommends the product in {2}.")

THREE_WAY_ON = """
    SELECT a.id, b.id, p.asin
    FROM reviews a
    JOIN threads b
    JOIN products p
      ON AI_FILTER(PROMPT('%s', a.review, b.thread, p.description),
                   {'selectivity': 0.02, 'anchor': 'b'})
""" % THREE_WAY_TEMPLATE

THREE_WAY_WHERE = """
    SELECT a.id, b.id, p.asin
    FROM reviews a, threads b, products p
    WHERE AI_FILTER(PROMPT('%s', a.review, b.thread, p.description),
                    {'selectivity': 0.02, 'anchor': 'b'})
""" % THREE_WAY_TEMPLATE


def test_sql_dialects_and_join_spellings_produce_the_same_plan(catalog):
    plan = compile_sql(FILTER_JOIN_SQL, catalog)
    root = plan.root
    assert isinstance(root, Project)
    assert [(c.alias, c.column) for c in root.columns] == \
        [("r", "id"), ("p", "asin")]
    join = root.input
    assert isinstance(join, SemanticJoin)
    assert (join.semantics, join.selectivity, join.anchor) == ("full", 0.05, None)
    # no ON equality means every pair is a candidate
    pairs = join.input
    assert isinstance(pairs, Join) and pairs.on == ()
    filt = pairs.left
    assert isinstance(filt, SemanticFilter)
    assert filt.predicates[0].selectivity == 0.3
    assert (filt.input.provider, str(filt.input.column)) == ("reviews", "r.review")
    assert (pairs.right.provider, str(pairs.right.column)) == \
        ("products", "p.description")
    # the join template is the per-tuple question and is never canonicalized
    assert join.prompt.template == "Review {0} discusses product {1}"

    star = compile_sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('neg: {0}', r.review))", catalog)
    assert [(c.alias, c.column) for c in star.root.columns] == \
        [("r", "id"), ("r", "review")]

    snowflake = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('negative {0}', r.review))",
        catalog, dialect=SQLDialect.SNOWFLAKE)
    bq = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI.IF(PROMPT('negative {0}', r.review))",
        catalog, dialect="bq")
    assert SQLDialect.BQ.value == "bq"
    assert bq == snowflake

    # one prompt, three placeholders, all three documents in the same
    # model call - never a sequence of pairwise stages
    on_plan = compile_sql(THREE_WAY_ON, catalog)
    join = on_plan.root.input
    assert isinstance(join, SemanticJoin)
    assert (join.anchor, join.selectivity) == ("b", 0.02)
    outer_pairs = join.input
    assert isinstance(outer_pairs, Join) and outer_pairs.on == ()
    assert isinstance(outer_pairs.left, Join)
    assert isinstance(outer_pairs.left.left, Scan)
    assert isinstance(join_outer_input(join), Scan)
    assert [r.alias for r in join.prompt.args] == ["a", "b", "p"]
    assert on_plan == compile_sql(THREE_WAY_WHERE, catalog)


TWO_ONS = """
    SELECT a.id FROM reviews a
    JOIN threads b
      ON AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread),
                   {'selectivity': 0.5})
    JOIN products p
      ON AI_FILTER(PROMPT('y {0} {1}', b.thread, p.description),
                   {'selectivity': 0.2})
"""

TWO_WHERE = """
    SELECT a.id FROM reviews a, threads b, products p
    WHERE AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread),
                    {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('y {0} {1}', b.thread, p.description),
                    {'selectivity': 0.2})
"""

PAIR_JOIN_SQL = """
    SELECT r.id, p.asin
    FROM reviews r
    JOIN products p
      ON r.id = p.asin
     AND AI_FILTER(PROMPT('Review {0} discusses product {1}',
                          r.review, p.description),
                   {'selectivity': 0.05})
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.review))
"""


def test_join_predicates_semantics_and_equality_pruning(catalog):
    # a sequence: each multi-table AI_FILTER is its own pairwise join
    plan = compile_sql(TWO_ONS, catalog)
    outer = plan.root.input
    inner = join_outer_input(outer)
    assert isinstance(outer, SemanticJoin) and isinstance(inner, SemanticJoin)
    assert [r.alias for r in inner.prompt.args] == ["a", "b"]
    assert [r.alias for r in outer.prompt.args] == ["b", "p"]
    assert inner.selectivity == 0.5 and outer.selectivity == 0.2
    assert plan == compile_sql(TWO_WHERE, catalog)

    sql = """
        SELECT t.id FROM threads t
        WHERE AI_FILTER(PROMPT('good: {0}', t.thread))
          AND NOT EXISTS (SELECT 1 FROM products s
                          WHERE AI_FILTER(PROMPT('match {0} {1}',
                                                 t.thread,
                                                 s.description)))
    """
    plan = compile_sql(sql, catalog)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    # the gate applies to the outer table, so it always anchors there
    assert (join.semantics, join.anchor) == ("anti", "t")
    outer = join_outer_input(join)
    assert isinstance(outer, SemanticFilter)
    assert outer.predicates[0].prompt.template.endswith("good:")
    exists_sql = sql.replace("NOT EXISTS", "EXISTS")
    assert compile_sql(exists_sql, catalog).root.input.semantics == "exists"
    anchored_sql = exists_sql.replace(
        "s.description)))", "s.description), {'anchor': 's'}))")
    with pytest.raises(CompileError, match="outer table"):
        compile_sql(anchored_sql, catalog)

    # two predicates over the same two tables: a pair must answer TRUE
    # to both, and only the first carries the joined table into the tree
    plan = compile_sql(
        "SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
        "PROMPT('x {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, p.description))",
        catalog)
    outer = plan.root.input
    inner = outer.input
    assert isinstance(outer, SemanticJoin) and isinstance(inner, SemanticJoin)
    assert isinstance(inner.input, Join)
    assert [r.alias for r in inner.prompt.args] == ["r", "p"]
    assert [r.alias for r in outer.prompt.args] == ["r", "p"]

    # an ON equality prunes the pairs before the AI predicate sees them
    plan = compile_sql(PAIR_JOIN_SQL, catalog)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    pairs = join.input
    assert isinstance(pairs, Join)
    assert [str(condition) for condition in pairs.on] == ["r.id = p.asin"]
    assert join_conditions(join) == pairs.on
    assert pairs.on[0].left == ColumnRef("r", "reviews", "id")
    assert pairs.on[0].right == ColumnRef("p", "products", "asin")
    # the equality columns are loaded as values for the pair table
    candidate, = native.logical_alternatives(
        plan, built_in_registry(), PlanningContext(catalog=catalog))
    optimized, _ = candidate.plan, candidate.rules
    scans = {node.alias: node for node in optimized.walk()
             if isinstance(node, Scan)}
    assert "id" in scans["r"].columns and "asin" in scans["p"].columns
    where_plan = compile_sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON r.id = p.asin
        WHERE AI_FILTER(PROMPT('Review {0} discusses product {1}',
                               r.review, p.description),
                        {'selectivity': 0.05})
          AND AI_FILTER(PROMPT('This review is negative: {0}', r.review))
    """, catalog)
    assert where_plan == plan


def test_each_part_of_a_where_clause_goes_to_the_operator_that_tests_it(catalog):
    sql = FILTER_JOIN_SQL + " AND r.id <> 'z'"
    # the rule puts the column test in the Scan, the one-table AI.IF in a
    # SemanticFilter, and the two-table AI.IF in a SemanticJoin
    plan = compile_sql(sql, catalog)
    scans = {node.alias: node for node in plan.walk() if isinstance(node, Scan)}
    assert [str(p) for p in scans["r"].predicates] == ["r.id <> 'z'"]
    (gate,) = [node for node in plan.walk() if isinstance(node, SemanticFilter)]
    assert [p.selectivity for p in gate.predicates] == [0.3]
    (join,) = [node for node in plan.walk() if isinstance(node, SemanticJoin)]
    assert join.selectivity == 0.05
