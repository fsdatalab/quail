"""The SQL front end and the builder: same LogicalPlan, named errors."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import keep_even, same_key

from quail.catalog import Catalog, DocumentProvider
from quail.frontend.builder import col, docs, prompt
from quail.frontend.sql import SQLDialect, compile_sql
from quail.logical import (
    SHARED_PRE,
    Apply,
    ColumnRef,
    CompileError,
    Join,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
    join_applies,
    join_conditions,
    join_outer_input,
)
from quail.planner.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.planner.logical_rules import built_in_logical_rules


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


def test_sql_dialects_and_builder_produce_the_same_plan(catalog):
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
    # the prompt sits on a relational Join; no ON equality means
    # every pair is a candidate
    pairs = join.input
    assert isinstance(pairs, Join) and pairs.on == ()
    filt = pairs.left
    assert isinstance(filt, SemanticFilter)
    assert filt.predicates[0].selectivity == 0.3
    assert isinstance(filt.input, Scan)
    assert (filt.input.provider, filt.input.column) == ("reviews",
                                                        "review")
    assert isinstance(pairs.right, Scan)
    assert (pairs.right.provider, pairs.right.column) == \
        ("products", "description")
    # the join template is never canonicalized: it is the per-tuple
    # question, kept as written
    assert join.prompt.template == \
        "Review {0} discusses product {1}"

    star = compile_sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('neg: {0}', r.review))", catalog, tok)
    assert [(c.alias, c.column) for c in star.root.columns] == \
        [("r", "id"), ("r", "review")]

    snowflake = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('negative {0}', r.review))",
        catalog,
        tok,
        dialect=SQLDialect.SNOWFLAKE,
    )
    bq = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI.IF(PROMPT('negative {0}', r.review))",
        catalog,
        tok,
        dialect="bq",
    )

    assert SQLDialect.BQ.value == "bq"
    assert bq == snowflake

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

    # one prompt, three placeholders, all three documents in the same
    # model call - never a chain of pairwise stages
    plan = compile_sql(THREE_WAY_ON, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    assert join.anchor == "b"
    assert join.selectivity == 0.02
    # both new tables are cross joined under one predicate
    outer_pairs = join.input
    assert isinstance(outer_pairs, Join) and outer_pairs.on == ()
    assert isinstance(outer_pairs.left, Join)
    assert isinstance(outer_pairs.left.left, Scan)
    assert isinstance(join_outer_input(join), Scan)
    assert [r.alias for r in join.prompt.args] == ["a", "b", "p"]

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


def test_prompt_layout_and_predicate_order(catalog):
    from quail.logical import (
        ANSWER_CUE,
        ColumnRef,
        bind_join_prompt,
        join_label,
        render_join_frame,
        render_join_question,
    )
    from quail.logical.prompts import DOCUMENT_PRE
    args = (ColumnRef("a", "reviews", "review"),
            ColumnRef("b", "threads", "thread"))
    p = bind_join_prompt("Does {0} praise {1}?", args, tok)
    assert p.template == "Does {0} praise {1}?"
    assert p.preamble == SHARED_PRE
    # the static question is in the anchor frame; only the answer cue
    # is paid per tuple
    assert p.frame == ("\n\nYou are performing a data processing task. "
                       "Evaluate TRUE or FALSE for the following "
                       "question: Does {0} praise {1}?")
    assert p.tail == ANSWER_CUE
    assert p.tail_tokens == len(tok(p.tail))
    assert join_label(1) == "\n\nDOCUMENT {1}:\n"
    assert [a for a, _, _ in p.labels] == ["a", "b"]
    assert p.labels[1][1] == len(tok(join_label(1)))
    assert p.labels[0][2] == len(tok(render_join_frame(p.template, 0)))
    assert render_join_question(p.template) == p.frame

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
    assert templates == [DOCUMENT_PRE + "{0}\n\nfirst:",
                         DOCUMENT_PRE + "{0}\n\nsecond:",
                         DOCUMENT_PRE + "{0}\n\nthird:"]

    from quail.logical import canonicalize_template, split_frame
    # no user text before the document: the preamble is prepended
    assert canonicalize_template("{0}\nQ: is it good?") == \
        DOCUMENT_PRE + "{0}\nQ: is it good?"
    # user text before the document relocates after it, so the fixed
    # preamble is the only thing ahead of the document's KV
    assert canonicalize_template("negative review: {0}") == \
        DOCUMENT_PRE + "{0}\n\nnegative review:"
    # two placeholders: only the first (the KV-owning document) moves
    # behind the preamble
    assert canonicalize_template("Does {0} match {1}?") == \
        DOCUMENT_PRE + "{0}\n\nDoes match {1}?"
    # no placeholder: no document to own, unchanged
    assert canonicalize_template("no placeholders") == "no placeholders"
    # the relocated text is recorded as the frame
    assert split_frame("Does {0} match {1}?")[0] == "Does"
    assert split_frame("{0} then {1}")[0] == ""

    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    pred = join_outer_input(plan.root.input).predicates[0]
    # the preamble is always the engine's; the user's pre-document
    # text ("This review is negative:") moves into the tail
    assert pred.prompt.preamble == SHARED_PRE
    assert pred.prompt.tail == ("{0}\n\nYou are performing a data processing task. "
                                "Evaluate TRUE or FALSE for the "
                                "following question: This review is "
                                "negative:" + ANSWER_CUE)
    assert pred.prompt.preamble_tokens == len(tok(SHARED_PRE))
    assert pred.prompt.tail_tokens == len(tok(
        "You are performing a data processing task. "
        "Evaluate TRUE or FALSE for the following question: "
        "This review is negative:" + ANSWER_CUE))


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


def test_join_predicates_and_semantics(catalog):
    # a chain: each multi-table AI_FILTER is its own pairwise join
    plan = compile_sql(TWO_ONS, catalog, tok)
    outer = plan.root.input
    assert isinstance(outer, SemanticJoin)
    inner = join_outer_input(outer)
    assert isinstance(inner, SemanticJoin)
    assert [r.alias for r in inner.prompt.args] == ["a", "b"]
    assert [r.alias for r in outer.prompt.args] == ["b", "p"]
    assert inner.selectivity == 0.5 and outer.selectivity == 0.2
    # both SQL styles and the chained builder produce the same plan
    built = (docs(catalog, "reviews", tok).alias("a")
             .ai_join(docs(catalog, "threads", tok).alias("b"),
                      prompt("x {0} {1}", col("a.review"),
                             col("b.thread")), selectivity=0.5)
             .ai_join(docs(catalog, "products", tok).alias("p"),
                      prompt("y {0} {1}", col("b.thread"),
                             col("p.description")), selectivity=0.2)
             .select("a.id"))
    assert plan == compile_sql(TWO_WHERE, catalog, tok) == built

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

    # two predicates over the same two tables: two specs; a pair must
    # answer TRUE to both. Only the first spec carries the joined
    # table into the tree
    sql = ("SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
           "PROMPT('x {0} {1}', r.review, p.description)) "
           "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, "
           "p.description))")
    plan = compile_sql(sql, catalog, tok)
    outer = plan.root.input
    assert isinstance(outer, SemanticJoin)
    inner = outer.input                         # p already in the tree
    assert isinstance(inner, SemanticJoin)
    assert isinstance(inner.input, Join)
    assert [r.alias for r in inner.prompt.args] == ["r", "p"]
    assert [r.alias for r in outer.prompt.args] == ["r", "p"]


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


def test_join_on_equality_prunes_the_pairs(catalog):
    # an ordinary equality in ON chooses the pairs; the AI predicate is
    # asked of those pairs only
    plan = compile_sql(PAIR_JOIN_SQL, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    pairs = join.input
    assert isinstance(pairs, Join)
    assert [str(condition) for condition in pairs.on] == ["r.id = p.asin"]
    assert join_conditions(join) == pairs.on
    assert pairs.on[0].left == ColumnRef("r", "reviews", "id")
    assert pairs.on[0].right == ColumnRef("p", "products", "asin")
    # the equality columns are loaded as values for the pair table
    optimized, _ = apply_logical_rules(
        plan, built_in_logical_rules(), LogicalPlanningContext(catalog, None))
    scans = {node.alias: node for node in optimized.walk()
             if isinstance(node, Scan)}
    assert "id" in scans["r"].columns and "asin" in scans["p"].columns
    # the builder spells it join(on=...) then ai_filter over both tables
    built = (docs(catalog, "reviews", tok).alias("r")
             .ai_filter(prompt("This review is negative: {0}",
                               col("r.review")))
             .join(docs(catalog, "products", tok).alias("p"),
                   on=col("r.id") == col("p.asin"))
             .ai_filter(prompt("Review {0} discusses product {1}",
                               col("r.review"), col("p.description")),
                        selectivity=0.05)
             .select("r.id", "p.asin"))
    assert built == plan
    # BigQuery style: the equality in ON, the AI predicate in WHERE
    where_plan = compile_sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON r.id = p.asin
        WHERE AI_FILTER(PROMPT('Review {0} discusses product {1}',
                               r.review, p.description),
                        {'selectivity': 0.05})
          AND AI_FILTER(PROMPT('This review is negative: {0}', r.review))
    """, catalog, tok)
    assert where_plan == plan

    cases = [
        ("SELECT r.id FROM reviews r JOIN products p ON r.id < p.asin "
         "AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))",
         "column = column"),
        ("SELECT r.id FROM reviews r JOIN products p ON r.id = r.review "
         "AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))",
         "table already in the query"),
        ("SELECT r.id FROM reviews r JOIN threads t ON t.id = r.id "
         "JOIN products p ON AI_FILTER(PROMPT('x {0} {1}', r.review, "
         "p.description)) WHERE AI_FILTER(PROMPT('y {0} {1}', "
         "t.thread, p.description))", "names a table the AI predicate"),
    ]
    for sql, fragment in cases:
        with pytest.raises(CompileError, match=fragment):
            compile_sql(sql, catalog, tok)
    base = docs(catalog, "reviews", tok).alias("r")
    with pytest.raises(CompileError, match="no AI predicate"):
        base.join(docs(catalog, "products", tok).alias("p"),
                  on=col("r.id") == col("p.asin")).select("r.id")
    with pytest.raises(CompileError, match="must name the joined table"):
        (docs(catalog, "reviews", tok).alias("r")
         .ai_join(docs(catalog, "threads", tok).alias("t"),
                  prompt("x {0} {1}", col("r.review"), col("t.thread")))
         .join(docs(catalog, "products", tok).alias("p"))
         .ai_filter(prompt("x {0} {1}", col("r.review"), col("t.thread"))))
    with pytest.raises(CompileError, match="waiting for the ai_filter"):
        (docs(catalog, "reviews", tok).alias("r")
         .join(docs(catalog, "products", tok).alias("p"))
         .join(docs(catalog, "threads", tok).alias("t")))
    with pytest.raises(CompileError, match="must relate the joined table"):
        (docs(catalog, "reviews", tok).alias("r")
         .join(docs(catalog, "products", tok).alias("p"),
               on=col("r.id") == col("r.review")))


def test_invalid_queries_and_limits(catalog):
    cases = [
        ("""
            SELECT a.id FROM reviews a, threads b, products p
            WHERE AI_FILTER(PROMPT('x {0} {1}', b.thread,
                                   p.description))
        """, "connected graph"),
        ("""
            SELECT a.id FROM reviews a, threads b, products p
            WHERE AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread))
        """, "every JOINed table"),
        ("""
            SELECT a.id FROM reviews a JOIN threads b
        """, "no join predicate"),
        ("""
            SELECT a.id FROM reviews a
            JOIN threads b
              ON AI_FILTER(PROMPT('x {0} {1} {2}', a.review, b.thread,
                                  a.review))
        """, "distinct table"),
    ]
    for sql, fragment in cases:
        with pytest.raises(CompileError) as e:
            compile_sql(sql, catalog, tok)
        assert fragment in str(e.value)

    q = (docs(catalog, "reviews", tok).alias("a")
         .ai_join(docs(catalog, "threads", tok).alias("b"),
                  prompt("x {0} {1}", col("a.review"),
                         col("b.thread"))))
    with pytest.raises(CompileError, match="joined but not referenced"):
        q.ai_join(docs(catalog, "products", tok).alias("p"),
                  prompt("y {0} {1}", col("a.review"),
                         col("b.thread")))

    with pytest.raises(CompileError, match="at least one table"):
        docs(catalog, "reviews", tok).alias("a").ai_join(
            [docs(catalog, "threads", tok).alias("b"),
             docs(catalog, "products", tok).alias("p")],
            prompt("y {0} {1}", col("b.thread"),
                   col("p.description")))

    with pytest.raises(CompileError):
        docs(catalog, "reviews", tok).select("id")
    with pytest.raises(CompileError):
        docs(catalog, "nowhere")
    with pytest.raises(CompileError):
        docs(catalog, "reviews", tok).alias("r").ai_filter(
            prompt("x {0} {1}", col("r.review"), col("r.id")))

    for sql, fragment in REJECTED:
        with pytest.raises(CompileError) as e:
            compile_sql(sql, catalog, tok)
        assert fragment.lower() in str(e.value).lower()

    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER("
           "PROMPT('x: {0}', r.review)) LIMIT 5")
    plan = compile_sql(sql, catalog, tok)
    assert plan.root.limit == 5
    built = (docs(catalog, "reviews", tok).alias("r")
             .ai_filter(prompt("x: {0}", col("r.review")))
             .limit(3)
             .select("r.id"))
    assert built.root.limit == 3
    with pytest.raises(CompileError, match="positive integer"):
        compile_sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
                    "PROMPT('x: {0}', r.review)) LIMIT 0",
                    catalog, tok)
    with pytest.raises(CompileError):
        compile_sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
                    "PROMPT('x: {0}', r.review)) LIMIT 'five'",
                    catalog, tok)


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
     "no AI predicate"),
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


def test_builder_places_apply_nodes_in_the_logical_tree():
    cat = Catalog()
    cat.register("claims", DocumentProvider.from_table(pa.table({
        "id": ["c0", "c1"], "claim": ["a b", "c d"], "url": ["u", "v"]}),
        id_col="id"))
    cat.register("evidence", DocumentProvider.from_table(pa.table({
        "id": ["u", "v"], "text": ["e f", "g h"]}), id_col="id"))
    plan = (docs(cat, "claims", tok).alias("c")
            .ai_filter(prompt("about a person: {0}", col("c.claim")))
            .apply(keep_even, columns=[col("c.url")])
            .join(docs(cat, "evidence", tok).alias("e"))
            .apply(same_key, columns=[col("c.url"), col("e.id")],
                   kind="barrier")
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")))
            .select("c.id", "e.id"))
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    (pairs,) = join_applies(join)
    assert (pairs.function, pairs.kind, pairs.ids, pairs.written_pos) == (
        "same_key", "barrier", "pairs", 0)
    assert pairs.aliases == ("c", "e")
    assert isinstance(pairs.input, Join)
    chain = pairs.input.left
    assert isinstance(chain, Apply)
    assert (chain.function, chain.kind, chain.ids, chain.aliases) == (
        "keep_even", "per_batch", "drop", ("c",))
    assert isinstance(chain.input, SemanticFilter)
    assert [str(ref.column) for ref in chain.columns] == ["url"]

    base = docs(cat, "claims", tok).alias("c")
    with pytest.raises(CompileError, match="must work on one table"):
        base.apply(keep_even)
    with pytest.raises(CompileError, match="needs a name for a lambda"):
        base.apply(lambda tables: [], columns=[col("c.url")])
    with pytest.raises(CompileError, match="returning pairs follows join"):
        base.apply(keep_even, columns=[col("c.url")], ids="pairs")
    with pytest.raises(CompileError, match="already used by another"):
        (base.apply(keep_even, columns=[col("c.url")])
         .apply(same_key, columns=[col("c.url")], name="keep_even"))
    with pytest.raises(CompileError, match="ids must be 'pairs'"):
        (docs(cat, "claims", tok).alias("c")
         .join(docs(cat, "evidence", tok).alias("e"))
         .apply(same_key, columns=[col("c.url")], ids="preserve"))
