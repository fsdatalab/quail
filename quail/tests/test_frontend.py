"""The SQL front end and the builder: same LogicalPlan, named errors."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quail.builder import col, docs, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.logical import (SHARED_PRE, CompileError, Project, Scan,
                           SemanticFilter, SemanticJoin)
from quail.sqlfront import compile_sql


def _parquet(path, columns):
    pq.write_table(
        pa.table({c: ["x", "y"] for c in columns}), str(path))
    return str(path)


@pytest.fixture()
def catalog(tmp_path):
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", ["id", "review"]), id_col="id"))
    cat.register("products", DocumentProvider.from_parquet(
        _parquet(tmp_path / "p.parquet", ["asin", "description"]),
        id_col="asin"))
    cat.register("threads", DocumentProvider.from_parquet(
        _parquet(tmp_path / "t.parquet", ["id", "thread"]), id_col="id"))
    return cat


def tok(text):
    return text.split()


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


def test_filter_join_shape(catalog):
    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    root = plan.root
    assert isinstance(root, Project)
    assert [(c.alias, c.column) for c in root.columns] == \
        [("r", "id"), ("p", "asin")]
    join = root.input
    assert isinstance(join, SemanticJoin)
    assert join.semantics == "full"
    assert join.selectivity == 0.05
    assert join.anchor is None
    filt = join.left
    assert isinstance(filt, SemanticFilter)
    assert filt.predicates[0].selectivity == 0.3
    assert isinstance(filt.input, Scan)
    assert (filt.input.provider, filt.input.column) == ("reviews",
                                                        "review")
    assert isinstance(join.right, Scan)
    assert (join.right.provider, join.right.column) == ("products",
                                                        "description")


def test_builder_equals_sql(catalog):
    sql_plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    built = (docs(catalog, "reviews", tok).alias("r")
             .ai_filter(prompt("This review is negative: {0}",
                               col("r.review")), selectivity=0.3)
             .ai_join(docs(catalog, "products", tok).alias("p"),
                      prompt("Review {0} discusses product {1}",
                             col("r.review"), col("p.description")),
                      selectivity=0.05)
             .select("r.id", "p.asin"))
    assert built == sql_plan
    assert built.to_dict() == sql_plan.to_dict()


def test_three_way_chain_with_anchors(catalog):
    sql = """
        SELECT a.id, b.id, p.asin
        FROM reviews a
        JOIN threads b
          ON AI_FILTER(PROMPT('Does review {0} praise this thread? {1}',
                              a.review, b.thread),
                       {'selectivity': 0.2, 'anchor': 'b'})
        JOIN products p
          ON AI_FILTER(PROMPT('Does thread {0} recommend product {1}?',
                              b.thread, p.description),
                       {'selectivity': 0.1, 'anchor': 'b'})
    """
    plan = compile_sql(sql, catalog, tok)
    stage2 = plan.root.input
    assert isinstance(stage2, SemanticJoin)
    assert stage2.anchor == "b"
    assert stage2.selectivity == 0.1
    stage1 = stage2.left
    assert isinstance(stage1, SemanticJoin)
    assert stage1.anchor == "b"
    assert stage1.selectivity == 0.2


def test_exists_and_anti(catalog):
    sql = """
        SELECT t.id FROM threads t
        WHERE AI_FILTER(PROMPT('good: {0}', t.thread))
          AND NOT EXISTS (SELECT 1 FROM products s
                          WHERE AI_FILTER(PROMPT('match {0} {1}',
                                                 t.thread,
                                                 s.description)))
    """
    plan = compile_sql(sql, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    assert join.semantics == "anti"
    built = (docs(catalog, "threads", tok).alias("t")
             .ai_filter(prompt("good: {0}", col("t.thread")))
             .ai_join(docs(catalog, "products", tok).alias("s"),
                      prompt("match {0} {1}", col("t.thread"),
                             col("s.description")),
                      semantics="anti")
             .select("t.id"))
    assert built == plan

    exists_sql = sql.replace("NOT EXISTS", "EXISTS")
    assert compile_sql(exists_sql, catalog,
                       tok).root.input.semantics == "exists"


def test_where_conjuncts_keep_written_order(catalog):
    sql = """
        SELECT r.id FROM reviews r
        WHERE AI_FILTER(PROMPT('first: {0}', r.review))
          AND AI_FILTER(PROMPT('second: {0}', r.review))
          AND AI_FILTER(PROMPT('third: {0}', r.review))
    """
    plan = compile_sql(sql, catalog, tok)
    templates = [p.prompt.template
                 for p in plan.root.input.predicates]
    # canonical layout: the engine preamble sits before the document
    # and the user's pre-document text is relocated after it
    assert templates == [SHARED_PRE + "{0}\n\nfirst:",
                         SHARED_PRE + "{0}\n\nsecond:",
                         SHARED_PRE + "{0}\n\nthird:"]


def test_canonicalize_template():
    from quail.logical import canonicalize_template, split_frame
    # no user text before the document: the preamble is prepended
    assert canonicalize_template("{0}\nQ: is it good?") == \
        SHARED_PRE + "{0}\nQ: is it good?"
    # user text before the document relocates after it, so the fixed
    # preamble is the only thing ahead of the document's KV
    assert canonicalize_template("negative review: {0}") == \
        SHARED_PRE + "{0}\n\nnegative review:"
    # two placeholders: only the first (the KV-owning document) moves
    # behind the preamble
    assert canonicalize_template("Does {0} match {1}?") == \
        SHARED_PRE + "{0}\n\nDoes match {1}?"
    # no placeholder: no document to own, unchanged
    assert canonicalize_template("no placeholders") == "no placeholders"
    # the relocated text is recorded as the frame
    assert split_frame("Does {0} match {1}?")[0] == "Does"
    assert split_frame("{0} then {1}")[0] == ""


def test_frame_split_and_counts(catalog):
    plan = compile_sql("""
        SELECT r.id FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('Judge the pair. {0} against {1}. Done.',
                              r.review, p.description),
                       {'selectivity': 0.5})
    """, catalog, tok)
    pred = plan.root.input.predicate
    assert pred.frame == "Judge the pair."
    # frame tokens counted with the separating blank line
    assert pred.frame_tokens == len(tok("\n\nJudge the pair."))
    # the canonical template carries the frame at the head of the mid
    assert pred.template == (SHARED_PRE + "{0}\n\nJudge the pair. "
                             "against {1}. Done.")


def test_prompt_split_and_counts(catalog):
    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    pred = plan.root.input.left.predicates[0]
    # the preamble is always the engine's; the user's pre-document
    # text ("This review is negative:") moves into the tail
    assert pred.prompt.preamble == SHARED_PRE
    assert pred.prompt.tail == "{0}\n\nThis review is negative:"
    assert pred.prompt.preamble_tokens == len(tok(SHARED_PRE))
    assert pred.prompt.tail_tokens == len(tok("This review is "
                                              "negative:"))


def test_star_projection(catalog):
    sql = ("SELECT * FROM reviews r WHERE "
           "AI_FILTER(PROMPT('neg: {0}', r.review))")
    plan = compile_sql(sql, catalog, tok)
    assert [(c.alias, c.column) for c in plan.root.columns] == \
        [("r", "id"), ("r", "review")]


REJECTED = [
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) GROUP BY r.id", "GROUP BY"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) ORDER BY r.id", "ORDER BY"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) LIMIT 5", "LIMIT"),
    ("SELECT DISTINCT r.id FROM reviews r WHERE AI_FILTER("
     "PROMPT('x {0}', r.review))", "DISTINCT"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) OR AI_FILTER(PROMPT('y {0}', r.review))",
     "disjunction"),
    ("SELECT r.id FROM reviews r WHERE AI_CLASSIFY(PROMPT('x {0}', "
     "r.review))", "AI_CLASSIFY"),
    ("SELECT r.id FROM reviews r JOIN products p ON r.id = p.asin",
     "AI_FILTER"),
    ("SELECT r.id FROM reviews r LEFT JOIN products p ON AI_FILTER("
     "PROMPT('x {0} {1}', r.review, p.description))", "plain JOIN"),
    ("SELECT r.id + 1 FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review))", "column selection"),
    ("SELECT r.id FROM reviews r", "no AI predicate"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review), {'selectivty': 0.3})", "unknown option"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0} {1}', "
     "r.review), {'selectivity': 0.3})", "placeholders"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.nope))", "not in provider"),
    ("SELECT r.id FROM nowhere r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review))", "unknown provider"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT("
     "'x {0} {1}', r.review, r.id))", "two columns"),
    ("SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
     "PROMPT('x {0} {1}', r.review, p.description), "
     "{'anchor': 'z'})", "anchor"),
    ("SELECT r.id FROM reviews r WHERE r.id IN (SELECT p.asin FROM "
     "products p)", "subquer"),
]


@pytest.mark.parametrize("sql,fragment", REJECTED)
def test_rejected_with_named_error(catalog, sql, fragment):
    with pytest.raises(CompileError) as e:
        compile_sql(sql, catalog, tok)
    assert fragment.lower() in str(e.value).lower()


def test_where_join_predicate_rejected(catalog):
    sql = ("SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
           "PROMPT('x {0} {1}', r.review, p.description)) "
           "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, "
           "p.description))")
    with pytest.raises(CompileError) as e:
        compile_sql(sql, catalog, tok)
    assert "exactly one provider" in str(e.value)


def test_builder_rejects_same_shapes(catalog):
    with pytest.raises(CompileError):
        docs(catalog, "nowhere")
    q = docs(catalog, "reviews", tok).alias("r")
    with pytest.raises(CompileError):
        q.ai_filter(prompt("x {0} {1}", col("r.review"), col("r.id")))
    with pytest.raises(CompileError):
        docs(catalog, "reviews", tok).select("id")   # no predicate
