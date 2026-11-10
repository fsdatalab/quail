"""The SQL front end: dialects, the operators a query derives, and named errors."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.oracle import question_text
from helpers.planning import compile_sql, placements, read_operators
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.compile import SQLDialect
from quail.catalog import Catalog, DocumentProvider
from quail.common.expressions import ColumnRef, CompileError
from quail.physical import AiFilter, AiJoin


def compile_operators(*args, **options):
    """Compile SQL and read its region's operators."""
    return read_operators(compile_sql(*args, **options))


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
    CROSS JOIN threads b
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
    found = compile_operators(FILTER_JOIN_SQL, catalog)
    assert [(c.alias, c.column) for c in found.projection.columns] == \
        [("r", "id"), ("p", "asin")]
    (join,) = found.joins
    assert (join.semantics, join.selectivity, join.anchor) == ("full", 0.05, None)
    # no ON equality means every pair is a candidate
    assert join.on == ()
    assert join.outer_aliases == ("r",)
    ((alias, (predicate,)),) = found.filters
    assert (alias, predicate.selectivity) == ("r", 0.3)
    assert found.filters_before_joins == (("r", (True,)),)
    scans = {scan.alias: scan for scan in found.scans}
    assert (scans["r"].provider, str(scans["r"].column)) == ("reviews", "r.review")
    assert (scans["p"].provider, str(scans["p"].column)) == \
        ("products", "p.description")
    # the join template is the per-tuple question and is never canonicalized
    assert join.predicate.prompt.template == "Review {0} discusses product {1}"

    star = compile_operators(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('neg: {0}', r.review))", catalog)
    assert [(c.alias, c.column) for c in star.projection.columns] == [
        ("r", "id"), ("r", "review")]

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
    on_found = compile_operators(THREE_WAY_ON, catalog)
    (join,) = on_found.joins
    assert (join.anchor, join.selectivity) == ("b", 0.02)
    assert join.on == ()
    assert join.outer_aliases == ("a",)
    assert [r.alias for r in join.predicate.prompt.args] == ["a", "b", "p"]
    assert compile_sql(THREE_WAY_ON, catalog) == compile_sql(THREE_WAY_WHERE, catalog)


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
    inner, outer = compile_operators(TWO_ONS, catalog).joins
    assert [r.alias for r in inner.predicate.prompt.args] == ["a", "b"]
    assert [r.alias for r in outer.predicate.prompt.args] == ["b", "p"]
    assert inner.selectivity == 0.5 and outer.selectivity == 0.2
    assert (inner.outer_aliases, outer.outer_aliases) == (("a",), ("a", "b"))
    assert compile_sql(TWO_ONS, catalog) == compile_sql(TWO_WHERE, catalog)

    sql = """
        SELECT t.id FROM threads t
        WHERE AI_FILTER(PROMPT('good: {0}', t.thread))
          AND NOT EXISTS (SELECT 1 FROM products s
                          WHERE AI_FILTER(PROMPT('match {0} {1}',
                                                 t.thread,
                                                 s.description)))
    """
    found = compile_operators(sql, catalog)
    (join,) = found.joins
    # the gate applies to the outer table, so it always anchors there
    assert (join.semantics, join.anchor) == ("anti", "t")
    assert join.outer_aliases == ("t",)
    ((alias, (predicate,)),) = found.filters
    assert alias == "t" and predicate.prompt.template.endswith("good:")
    exists_sql = sql.replace("NOT EXISTS", "EXISTS")
    assert compile_operators(exists_sql, catalog).joins[0].semantics == "exists"
    anchored_sql = exists_sql.replace(
        "s.description)))", "s.description), {'anchor': 's'}))")
    with pytest.raises(CompileError, match="outer table"):
        compile_operators(anchored_sql, catalog)

    # two predicates over the same two tables: a pair must answer TRUE
    # to both, and only the first carries the joined table into the tree
    inner, outer = compile_operators(
        "SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
        "PROMPT('x {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, p.description))",
        catalog).joins
    assert [r.alias for r in inner.predicate.prompt.args] == ["r", "p"]
    assert [r.alias for r in outer.predicate.prompt.args] == ["r", "p"]
    assert (inner.outer_aliases, outer.outer_aliases) == (("r",), ("r", "p"))

    # an ON equality prunes the pairs before the AI predicate sees them
    (join,) = compile_operators(PAIR_JOIN_SQL, catalog).joins
    assert [str(condition) for condition in join.on] == ["r.id = p.asin"]
    assert join.on[0].left == ColumnRef("r", "reviews", "id")
    assert join.on[0].right == ColumnRef("p", "products", "asin")
    # the equality columns are loaded as values for the pair table
    candidate, = placements(compile_sql(PAIR_JOIN_SQL, catalog))
    scans = {scan.alias: scan for scan in candidate.found.scans}
    assert "id" in scans["r"].columns and "asin" in scans["p"].columns
    written_in_where = compile_sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON r.id = p.asin
        WHERE AI_FILTER(PROMPT('Review {0} discusses product {1}',
                               r.review, p.description),
                        {'selectivity': 0.05})
          AND AI_FILTER(PROMPT('This review is negative: {0}', r.review))
    """, catalog)
    assert written_in_where == compile_sql(PAIR_JOIN_SQL, catalog)


def test_each_part_of_a_where_clause_goes_to_the_operator_that_tests_it(catalog):
    sql = FILTER_JOIN_SQL + " AND r.id <> 'z'"
    # DataFusion runs the column test before the region, the one-table
    # AI.IF is an AI filter, and the two-table AI.IF an AI join
    found = compile_operators(sql, catalog)
    scans = {scan.alias: scan for scan in found.scans}
    assert scans["r"].predicates == ()
    ((_, predicates),) = found.filters
    assert [p.selectivity for p in predicates] == [0.3]
    (join,) = found.joins
    assert join.selectivity == 0.05


def test_a_join_with_two_ai_predicates_in_its_on_plans_both_predicates(catalog):
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        for name, provider in catalog.providers.items():
            session.register(name, provider)
        query = session.sql("""
            SELECT r.id FROM reviews r JOIN products p
            ON AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
            AND AI_FILTER(PROMPT('y {0} {1}', r.review, p.description))
        """)
        questions = []
        for region in query.regions():
            for node in region.plan().nodes:
                if isinstance(node, AiFilter):
                    questions.append(question_text(node.question_token_ids))
                elif isinstance(node, AiJoin):
                    questions.append(question_text([
                        *node.frame_token_ids, *node.tail_token_ids]))
        assert len(questions) == 2
        assert any("x" in question.split() for question in questions)
        assert any("y" in question.split() for question in questions)
