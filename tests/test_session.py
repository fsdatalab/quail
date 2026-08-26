"""End-to-end Session tests with a fake executor: gating, tuple assembly, projection, and report."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.planner.plan import EngineConfig


def _parquet(path, table):
    pq.write_table(pa.table(table), str(path))
    return str(path)


def fake_tok(text):
    return text.split()


@pytest.fixture()
def sess(tmp_path):
    s = quail.Session(EngineConfig(gpus=1), tokenizer=fake_tok)
    # reviews: longer documents (they anchor); products: short
    s.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", {
            "id": [f"r{i}" for i in range(6)],
            "review": [f"review {i} " + "pad " * 20 for i in range(6)],
        }), id_col="id"))
    s.register("products", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "p.parquet", {
            "asin": [f"p{i}" for i in range(4)],
            "description": [f"product {i}" for i in range(4)],
        }), id_col="asin"))
    return s


def make_executor(filter_truth, join_truth=None, seen=None):
    """Build a fake executor from filter and join truth tables. Optionally captures the payload in `seen`."""
    import itertools

    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(payload):
        if seen is not None:
            seen["payload"] = payload
        out = dict(filters={}, joins=[], wall_s=1.0, boot_s=0.5,
                   fresh_tokens=1234)
        survivors = {a: list(range(len(d)))
                     for a, d in payload["docs"].items()}
        for alias, qids in payload["filters"].items():
            rows = {}
            for d in range(len(payload["docs"][alias])):
                row = []
                for q in qids:
                    bit = filter_truth[alias][_match_key(alias, q)][d]
                    row.append(bit)
                    if not bit:
                        break
                rows[d] = row
            out["filters"][alias] = rows
            survivors[alias] = [d for d, r in rows.items()
                                if len(r) == len(qids) and all(r)]
        for j in payload["joins"]:
            anchors = list(survivors[j["anchor"]])
            tuples = [list(t) for t in itertools.product(
                *[survivors[p] for p in j["partners"]])]
            rule = join_truth[(j["anchor"], *j["partners"])]
            rows = {ai: [rule(a, *t) for t in tuples]
                    for ai, a in enumerate(anchors)}
            out["joins"].append(dict(rows=rows, anchor_index=anchors,
                                     partner_index=tuples))
            kept = {anchors[ai] for ai, r in rows.items() if any(r)}
            if j["semantics"] == "anti":
                survivors[j["anchor"]] = [a for a in anchors
                                          if a not in kept]
            else:
                survivors[j["anchor"]] = sorted(kept)
        return out

    return _exec


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_filter_query_rows_and_report(sess):
    truth = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
                         "q2:": [1, 0, 1, 1, 0, 1]}}
    q = sess.sql(FILTER_SQL)
    res = q.run(_execute=make_executor(truth))
    assert res.columns == ["r.id"]
    assert sorted(res.rows) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6,
                                                              abs=1e-3)
    # stage 2 only saw stage-1 survivors
    assert stages[1]["evaluated"] == 4
    assert res.report["wall_s"] == 1.0

    limited = sess.sql(FILTER_SQL + " LIMIT 1").run(
        _execute=make_executor(truth))
    assert len(limited.rows) == 1


def test_join_query_pairs(sess):
    sql = """
        SELECT r.id, p.asin FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('match {0} {1}', r.review,
                              p.description), {'selectivity': 0.25})
        WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
                        {'selectivity': 0.5})
    """
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0]}}
    join = {("r", "p"): lambda a, p: 1 if (a + p) % 4 == 0 else 0}
    res = sess.sql(sql).run(_execute=make_executor(truth, join))
    # survivors r0, r2, r4; pairs pass when (a+p) % 4 == 0
    expect = sorted(("r%d" % a, "p%d" % p)
                    for a in (0, 2, 4) for p in range(4)
                    if (a + p) % 4 == 0)
    assert sorted(res.rows) == expect
    jstage = [s for s in res.report["stages"] if s["op"] == "join"][0]
    assert jstage["tuples"] == 12
    assert jstage["provided_selectivity"] == 0.25


def test_anti_join_keeps_unmatched(sess):
    sql = """
        SELECT r.id FROM reviews r
        WHERE NOT EXISTS (SELECT 1 FROM products s
                          WHERE AI_FILTER(PROMPT('m {0} {1}', r.review,
                                                 s.description)))
    """
    join = {("r", "s"): lambda a, p: 1 if a < 3 else 0}
    res = sess.sql(sql).run(_execute=make_executor({}, join))
    assert sorted(res.rows) == [("r3",), ("r4",), ("r5",)]


def test_order_by_cost_reorders_payload(sess):
    sql = """
        SELECT r.id FROM reviews r
        WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
                        {'selectivity': 0.9})
          AND AI_FILTER(PROMPT('q2: {0}', r.review),
                        {'selectivity': 0.1})
    """
    truth = {"r": {"q1:": [1] * 6, "q2:": [1] * 6}}
    seen = {}
    sess.sql(sql).run(_execute=make_executor(truth, seen=seen))
    # by_cost (the default: every predicate has a selectivity) runs
    # the 0.1 filter first
    first_q = seen["payload"]["filters"]["r"][0]
    assert "q2:" in first_q
    seen2 = {}
    sess.sql(sql, order="as_written").run(
        _execute=make_executor(truth, seen=seen2))
    assert "q1:" in seen2["payload"]["filters"]["r"][0]


def test_payload_carries_filter_arena_writes(sess):
    # one stage: nothing reads the KV again - the planner turns
    # writes off
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0]}}
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER("
           "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})")
    seen = {}
    sess.sql(sql).run(_execute=make_executor(truth, seen=seen))
    assert seen["payload"]["filter_arena_writes"] == {"r": False}
    assert "arena_writes=False" in sess.sql(sql).explain()

    # a second stage re-reads survivors' KV
    truth2 = {"r": {"q1:": [1] * 6, "q2:": [1] * 6}}
    seen = {}
    sess.sql(FILTER_SQL).run(_execute=make_executor(truth2, seen=seen))
    assert seen["payload"]["filter_arena_writes"] == {"r": True}


def test_refusal_raises_on_run_prints_in_explain(sess, tmp_path):
    sess.register("huge", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "h.parquet", {
            "id": ["h0"],
            "body": ["w " * 150_000],
        }), id_col="id"))
    q = sess.sql("SELECT h.id FROM huge h WHERE AI_FILTER("
                 "PROMPT('q: {0}', h.body))")
    assert "refusal: suffix_over_chunk" in q.explain()
    with pytest.raises(quail.RefusalError) as e:
        q.run(_execute=make_executor({}))
    assert e.value.refusal.constraint == "suffix_over_chunk"
    with pytest.raises(quail.RefusalError):
        quail.Session(EngineConfig(model="qwen9-99b"),
                      tokenizer=fake_tok)


def test_pick_corpus_tokenizer_parity_guard():
    from quail.runtime.session import pick_corpus_tokenizer

    primary = str.split
    texts = ["a b c", "d e", "f"]
    tok, note = pick_corpus_tokenizer(primary, None, texts)
    assert tok is primary and "transformers" in note

    matching = lambda t: t.split()          # noqa: E731
    tok, note = pick_corpus_tokenizer(primary, matching, texts)
    assert tok is matching and "bpe-qwen" in note

    broken = lambda t: t.split()[:-1]       # noqa: E731
    tok, note = pick_corpus_tokenizer(primary, broken, texts)
    assert tok is primary and "failed parity" in note


def test_payload_carries_workers_and_shards(tmp_path):
    import quail
    from quail.planner.plan import EngineConfig
    s = quail.Session(EngineConfig(gpus=2), tokenizer=fake_tok)
    s.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", {
            "id": [f"r{i}" for i in range(6)],
            "review": [f"review {i} " + "pad " * (10 + i)
                       for i in range(6)],
        }), id_col="id"))
    truth = {"r": {"q1:": [1] * 6}}
    seen = {}
    s.sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
          "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})").run(
        _execute=make_executor(truth, seen=seen))
    p = seen["payload"]
    assert p["workers"] == 2
    assert len(p["shards"]["r"]) == 2
    covered = sorted(i for sh in p["shards"]["r"] for i in sh)
    assert covered == list(range(6))


def test_payload_carries_true_false_and_join_spec(sess):
    sql = """
        SELECT r.id, p.asin FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('Does {0} match {1}? Answer.', r.review,
                              p.description), {'selectivity': 0.5})
    """
    seen = {}
    join = {("r", "p"): lambda a, p: 0}
    query = sess.sql(sql)
    query.run(_execute=make_executor({}, join, seen=seen))
    pred = query.logical.root.input.predicate
    payload = seen["payload"]
    assert payload["true_ids"] and payload["false_ids"]
    # the engine preamble ships once, not inside any join segment
    from quail.logical import (SHARED_PRE, join_label,
                               render_join_frame)
    assert payload["pre_ids"] == fake_tok(SHARED_PRE)
    j = payload["joins"][0]
    assert "pre" not in j
    assert j["anchor"] == "r" and j["partners"] == ["p"]
    assert j["aliases"] == ["r", "p"]
    # anchor frames and block labels ship for EVERY table, so a
    # barrier-time anchor re-pick needs no re-tokenization; the round
    # builder (stage_for_anchor) picks the chosen anchor's complete
    # frame and the partners' labels. r is placeholder 0,
    # p is placeholder 1.
    assert j["frames"] == {
        "r": fake_tok(render_join_frame(pred.template, 0)),
        "p": fake_tok(render_join_frame(pred.template, 1)),
    }
    assert j["labels"] == {"r": fake_tok(join_label(0)),
                           "p": fake_tok(join_label(1))}
    assert j["tail"] == fake_tok("\nANSWER:")


def test_three_way_join_tuples_and_gate(sess, tmp_path):
    sess.register("tags", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "g.parquet", {
            "id": [f"g{i}" for i in range(3)],
            "tag": [f"tag {i}" for i in range(3)],
        }), id_col="id"))
    q = (sess.docs("reviews").alias("r")
         .ai_join([sess.docs("products").alias("p"),
                   sess.docs("tags").alias("g")],
                  quail.prompt("Do {0}, {1} and {2} agree?",
                               quail.col("r.review"),
                               quail.col("p.description"),
                               quail.col("g.tag")),
                  selectivity=0.1)
         .select("r.id", "p.asin", "g.id"))
    join = {("r", "p", "g"):
            lambda a, p, g: 1 if (a + p + g) % 5 == 0 else 0}
    res = q.run(_execute=make_executor({}, join))
    expect = sorted((f"r{a}", f"p{p}", f"g{g}")
                    for a in range(6) for p in range(4)
                    for g in range(3) if (a + p + g) % 5 == 0)
    assert sorted(res.rows) == expect
    jstage = [s for s in res.report["stages"] if s["op"] == "join"][0]
    assert jstage["tuples"] == 6 * 4 * 3
    assert jstage["partners"] == ["p", "g"]


def _register_tags(sess, tmp_path):
    sess.register("tags", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "g.parquet", {
            "id": [f"g{i}" for i in range(3)],
            "tag": [f"tag {i}" for i in range(3)],
        }), id_col="id"))


def _chain_query(sess, limit=None):
    """Build a two-join chain sharing table p with both anchors forced onto p."""
    q = (sess.docs("reviews").alias("r")
         .ai_join(sess.docs("products").alias("p"),
                  quail.prompt("m1 {0} {1}", quail.col("r.review"),
                               quail.col("p.description")),
                  selectivity=0.5, anchor="p")
         .ai_join(sess.docs("tags").alias("g"),
                  quail.prompt("m2 {0} {1}", quail.col("p.description"),
                               quail.col("g.tag")),
                  selectivity=0.5, anchor="p"))
    if limit is not None:
        q = q.limit(limit)
    return q.select("r.id", "p.asin", "g.id")


# p1 matches no review, so the gate drops it before stage 2
J1 = lambda p, r: 1 if p != 1 and (p + r) % 3 == 0 else 0  # noqa: E731
J2 = lambda p, g: 1 if (p + g) % 2 == 0 else 0             # noqa: E731

CHAIN_EXPECT = sorted(
    (f"r{r}", f"p{p}", f"g{g}")
    for p in range(4) for r in range(6) for g in range(3)
    if J1(p, r) and J2(p, g))


def test_two_join_chain_recombination(sess, tmp_path):
    _register_tags(sess, tmp_path)
    seen = {}
    join = {("p", "r"): J1, ("p", "g"): J2}
    res = _chain_query(sess).run(
        _execute=make_executor({}, join, seen=seen))
    # both stages anchored on the shared table, one per payload entry
    assert [j["anchor"] for j in seen["payload"]["joins"]] == \
        ["p", "p"]
    assert sorted(res.rows) == CHAIN_EXPECT
    # p1 was gated after stage 1: stage 2 evaluated 3 anchors, not 4
    jstages = [s for s in res.report["stages"] if s["op"] == "join"]
    assert jstages[0]["tuples"] == 4 * 6
    assert jstages[1]["tuples"] == 3 * 3


def test_gate_after_two_join_chain_filters_tuples(sess, tmp_path):
    # an anti gate on g, written after both joins: its casualties
    # must not appear in any output triple (survivor-set filtering
    # applies to every stage's members)
    _register_tags(sess, tmp_path)
    q = (sess.docs("reviews").alias("r")
         .ai_join(sess.docs("products").alias("p"),
                  quail.prompt("m1 {0} {1}", quail.col("r.review"),
                               quail.col("p.description")),
                  anchor="p")
         .ai_join(sess.docs("tags").alias("g"),
                  quail.prompt("m2 {0} {1}", quail.col("p.description"),
                               quail.col("g.tag")),
                  anchor="p")
         .ai_join(sess.docs("reviews").alias("x"),
                  quail.prompt("m3 {0} {1}", quail.col("g.tag"),
                               quail.col("x.review")),
                  semantics="anti")
         .select("r.id", "p.asin", "g.id"))
    join = {("p", "r"): J1, ("p", "g"): J2,
            ("g", "x"): lambda g, x: 1 if g == 0 else 0}
    res = q.run(_execute=make_executor({}, join))
    assert sorted(res.rows) == [t for t in CHAIN_EXPECT
                                if t[2] != "g0"]


def test_two_join_chain_limit_caps_final_triples(sess, tmp_path):
    _register_tags(sess, tmp_path)
    seen = {}
    join = {("p", "r"): J1, ("p", "g"): J2}
    res = _chain_query(sess, limit=3).run(
        _execute=make_executor({}, join, seen=seen))
    # no upstream cut: the cap applies to the final triples only
    assert seen["payload"]["limit"] is None
    assert len(res.rows) == 3
    assert set(res.rows) <= set(CHAIN_EXPECT)
    assert len(CHAIN_EXPECT) > 3


def test_limit_join_payload_carries_no_filter_limit(sess):
    # #39: with a join, the filter round must not stop at LIMIT
    # survivors - the payload ships limit=None and _assemble caps the
    # output rows
    sql = """
        SELECT r.id, p.asin FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('match {0} {1}', r.review,
                              p.description), {'selectivity': 0.5})
        WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
                        {'selectivity': 0.9})
        LIMIT 2
    """
    truth = {"r": {"q1:": [1, 1, 1, 1, 1, 0]}}
    # only r4 matches: a filter cut to the first 2 survivors would
    # leave zero matching pairs; the correct answer is 2 of r4's 4
    join = {("r", "p"): lambda a, p: 1 if a == 4 else 0}
    seen = {}
    q = sess.sql(sql)
    assert q.plan().limit == 2
    res = q.run(_execute=make_executor(truth, join, seen=seen))
    assert seen["payload"]["limit"] is None
    assert len(res.rows) == 2
    assert all(r == "r4" for r, _ in res.rows)


def test_chain_with_barrier_recombination(sess, tmp_path):
    # stage 1 forced onto r, stage 2 forced onto g: two groups with a
    # barrier between them. Recombination equi-joins the two pair
    # sets on p - the alias the stages share - even though neither
    # stage anchors on it.
    _register_tags(sess, tmp_path)
    q = (sess.docs("reviews").alias("r")
         .ai_join(sess.docs("products").alias("p"),
                  quail.prompt("m1 {0} {1}", quail.col("r.review"),
                               quail.col("p.description")),
                  selectivity=0.5, anchor="r")
         .ai_join(sess.docs("tags").alias("g"),
                  quail.prompt("m2 {0} {1}", quail.col("p.description"),
                               quail.col("g.tag")),
                  selectivity=0.5, anchor="g")
         .select("r.id", "p.asin", "g.id"))
    plan = q.plan()
    kinds = [n["op"] for n in plan.nodes]
    assert kinds.count("JoinGroup") == 2
    assert kinds.count("Barrier") == 1
    seen = {}
    join = {("r", "p"): lambda r, p: J1(p, r),
            ("g", "p"): lambda g, p: J2(p, g)}
    res = q.run(_execute=make_executor({}, join, seen=seen))
    assert [j["anchor"] for j in seen["payload"]["joins"]] == \
        ["r", "g"]
    # same pair semantics as the shared-anchor chain, same triples
    assert sorted(res.rows) == CHAIN_EXPECT


def test_gate_after_full_join_filters_partner_tuples(sess, tmp_path):
    # an anti gate on the join's partner table, written after the
    # join: its casualties must not appear in output tuples (order
    # changes cost, never results)
    q = (sess.docs("reviews").alias("r")
         .ai_join(sess.docs("products").alias("p"),
                  quail.prompt("m {0} {1}", quail.col("r.review"),
                               quail.col("p.description")),
                  selectivity=0.5)
         .ai_join(sess.docs("reviews").alias("x"),
                  quail.prompt("m {0} {1}", quail.col("p.description"),
                               quail.col("x.review")),
                  semantics="anti")
         .select("r.id", "p.asin"))
    join = {("r", "p"): lambda a, p: 1,
            ("p", "x"): lambda p, x: 1 if p == 1 else 0}
    res = q.run(_execute=make_executor({}, join))
    # p1 is matched by the anti gate and drops; every (r, p!=1) stays
    assert sorted(res.rows) == sorted(
        (f"r{a}", f"p{p}") for a in range(6) for p in (0, 2, 3))
