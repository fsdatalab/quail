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
    filt = join.inputs[0]
    assert isinstance(filt, SemanticFilter)
    assert filt.predicates[0].selectivity == 0.3
    assert isinstance(filt.input, Scan)
    assert (filt.input.provider, filt.input.column) == ("reviews",
                                                        "review")
    assert isinstance(join.inputs[1], Scan)
    assert (join.inputs[1].provider, join.inputs[1].column) == \
        ("products", "description")
    # the join template is never canonicalized: it is the per-tuple
    # question, kept as written
    assert join.predicate.template == \
        "Review {0} discusses product {1}"


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


def test_three_way_is_one_cross_product_join(catalog):
    # one prompt, three placeholders, all three documents in the same
    # model call - never a chain of pairwise stages
    plan = compile_sql(THREE_WAY_ON, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    assert join.anchor == "b"
    assert join.selectivity == 0.02
    assert len(join.inputs) == 3
    assert isinstance(join.inputs[0], Scan)   # no inner join anywhere
    assert [r.alias for r in join.predicate.args] == ["a", "b", "p"]


def test_three_way_forms_compile_equal(catalog):
    # Snowflake style (bare JOINs, the predicate on the last ON),
    # BigQuery style (comma cross product, the predicate in WHERE),
    # and the builder all produce the same plan
    on_plan = compile_sql(THREE_WAY_ON, catalog, tok)
    where_plan = compile_sql(THREE_WAY_WHERE, catalog, tok)
    built = (docs(catalog, "reviews", tok).alias("a")
             .ai_join([docs(catalog, "threads", tok).alias("b"),
                       docs(catalog, "products", tok).alias("p")],
                      prompt(THREE_WAY_TEMPLATE, col("a.review"),
                             col("b.thread"), col("p.description")),
                      selectivity=0.02, anchor="b")
             .select("a.id", "b.id", "p.asin"))
    assert on_plan == where_plan == built


def test_join_prompt_keeps_markers_and_labels_blocks():
    from quail.logical import (bind_join_prompt, join_anchor_note,
                               join_label, render_join_question)
    from quail.logical import ColumnRef
    args = (ColumnRef("a", "reviews", "review"),
            ColumnRef("b", "threads", "thread"))
    p = bind_join_prompt("Does {0} praise {1}?", args, tok)
    assert p.template == "Does {0} praise {1}?"
    assert p.preamble == SHARED_PRE
    # the question is the template verbatim - markers kept, nothing
    # filled in; the blocks above carry the matching labels
    assert p.tail == ("\n\nEvaluate TRUE or FALSE for the following "
                      "question: Does {0} praise {1}?\nANSWER:")
    assert p.tail_tokens == len(tok(p.tail))
    assert join_label(1) == "\n\nDOCUMENT {1}:\n"
    assert "{0}" in join_anchor_note(0)
    assert [a for a, _, _ in p.labels] == ["a", "b"]
    assert p.labels[1][1] == len(tok(join_label(1)))
    assert p.labels[0][2] == len(tok(join_anchor_note(0)))
    assert render_join_question(p.template) == p.tail


def test_join_rejects_second_predicate_and_repeat_alias(catalog):
    two_ons = """
        SELECT a.id FROM reviews a
        JOIN threads b
          ON AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread))
        JOIN products p
          ON AI_FILTER(PROMPT('y {0} {1}', b.thread, p.description))
    """
    with pytest.raises(CompileError) as e:
        compile_sql(two_ons, catalog, tok)
    assert "one join predicate" in str(e.value)

    with pytest.raises(CompileError) as e:
        compile_sql("""
            SELECT a.id FROM reviews a
            JOIN threads b
              ON AI_FILTER(PROMPT('x {0} {1} {2}', a.review, b.thread,
                                  a.review))
        """, catalog, tok)
    assert "distinct table" in str(e.value)

    q = (docs(catalog, "reviews", tok).alias("a")
         .ai_join(docs(catalog, "threads", tok).alias("b"),
                  prompt("x {0} {1}", col("a.review"), col("b.thread"))))
    with pytest.raises(CompileError) as e:
        q.ai_join(docs(catalog, "products", tok).alias("p"),
                  prompt("y {0} {1}", col("a.review"),
                         col("p.description")))
    assert "one full ai_join" in str(e.value)


def test_join_predicate_must_cover_every_joined_table(catalog):
    with pytest.raises(CompileError) as e:
        compile_sql("""
            SELECT a.id FROM reviews a, threads b, products p
            WHERE AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread))
        """, catalog, tok)
    assert "every JOINed table" in str(e.value)

    with pytest.raises(CompileError) as e:
        compile_sql("""
            SELECT a.id FROM reviews a JOIN threads b
        """, catalog, tok)
    assert "no join predicate" in str(e.value)


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
    # the gate applies to the outer table, so it always anchors there
    assert join.anchor == "t"
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

    with pytest.raises(CompileError) as e:
        docs(catalog, "threads", tok).alias("t").ai_join(
            docs(catalog, "products", tok).alias("s"),
            prompt("match {0} {1}", col("t.thread"),
                   col("s.description")),
            semantics="exists", anchor="s")
    assert "outer table" in str(e.value)


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


def test_join_prompt_binding_counts(catalog):
    plan = compile_sql("""
        SELECT r.id FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('Judge the pair: {0} against {1}. Done.',
                              r.review, p.description),
                       {'selectivity': 0.5})
    """, catalog, tok)
    pred = plan.root.input.predicate
    # the whole template is the per-tuple question, verbatim - no
    # substitution, no text relocated into the anchor's kept KV
    assert pred.frame == "" and pred.frame_tokens == 0
    assert pred.preamble == SHARED_PRE
    assert pred.tail == ("\n\nEvaluate TRUE or FALSE for the following "
                         "question: Judge the pair: {0} against "
                         "{1}. Done.\nANSWER:")
    assert pred.tail_tokens == len(tok(pred.tail))


def test_prompt_split_and_counts(catalog):
    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    pred = plan.root.input.inputs[0].predicates[0]
    # the preamble is always the engine's; the user's pre-document
    # text ("This review is negative:") moves into the tail
    assert pred.prompt.preamble == SHARED_PRE
    assert pred.prompt.tail == ("{0}\n\nEvaluate TRUE or FALSE for the "
                                "following question: This review is "
                                "negative:\nANSWER:")
    assert pred.prompt.preamble_tokens == len(tok(SHARED_PRE))
    assert pred.prompt.tail_tokens == len(tok(
        "Evaluate TRUE or FALSE for the following question: "
        "This review is negative: ANSWER:"))


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
     "r.review)) LIMIT 10 OFFSET 5", "OFFSET"),
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


def test_second_join_predicate_in_where_rejected(catalog):
    # a multi-provider WHERE AI_FILTER is the join predicate (the
    # BigQuery form); with an ON predicate already given it is a
    # second one, and there is only ever one
    sql = ("SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
           "PROMPT('x {0} {1}', r.review, p.description)) "
           "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, "
           "p.description))")
    with pytest.raises(CompileError) as e:
        compile_sql(sql, catalog, tok)
    assert "one join predicate" in str(e.value)


def test_limit_parses_and_threads(catalog):
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER("
           "PROMPT('x: {0}', r.review)) LIMIT 5")
    plan = compile_sql(sql, catalog, tok)
    assert plan.root.limit == 5


def test_limit_zero_and_negative_rejected(catalog):
    for n in ("0", "-1"):
        with pytest.raises(CompileError) as e:
            compile_sql(f"SELECT r.id FROM reviews r WHERE AI_FILTER("
                        f"PROMPT('x: {{0}}', r.review)) LIMIT {n}",
                        catalog, tok)
        assert "positive integer" in str(e.value)


def test_limit_string_rejected(catalog):
    with pytest.raises(CompileError):
        compile_sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
                    "PROMPT('x: {0}', r.review)) LIMIT 'five'",
                    catalog, tok)


def test_no_limit_gives_none(catalog):
    plan = compile_sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER("
        "PROMPT('x: {0}', r.review))", catalog, tok)
    assert plan.root.limit is None


def test_builder_limit(catalog):
    plan = (docs(catalog, "reviews", tok).alias("r")
            .ai_filter(prompt("x: {0}", col("r.review")))
            .limit(3)
            .select("r.id"))
    assert plan.root.limit == 3


def test_builder_rejects_same_shapes(catalog):
    with pytest.raises(CompileError):
        docs(catalog, "nowhere")
    q = docs(catalog, "reviews", tok).alias("r")
    with pytest.raises(CompileError):
        q.ai_filter(prompt("x {0} {1}", col("r.review"), col("r.id")))
    with pytest.raises(CompileError):
        docs(catalog, "reviews", tok).select("id")   # no predicate
