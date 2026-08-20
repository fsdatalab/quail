"""Session end to end with a fake executor in the worker seam: the
coordinator's gating, tuple assembly, replay check, projection, and
report - everything except the GPU."""

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
    """filter_truth: alias -> {question first-token -> [bit per doc]}.
    join_truth: (anchor alias, *partner aliases) ->
    f(anchor_idx, *partner_idxs) -> bit, one call per cross-product
    tuple - the worker contract. seen: dict to capture the payload
    for order assertions."""
    import itertools

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
                    bit = filter_truth[alias][q[0]][d]
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


def test_builder_run_matches_sql(sess):
    truth = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
                         "q2:": [1, 0, 1, 1, 0, 1]}}
    q = (sess.docs("reviews").alias("r")
         .ai_filter(quail.prompt("q1: {0}", quail.col("r.review")),
                    selectivity=0.5)
         .ai_filter(quail.prompt("q2: {0}", quail.col("r.review")),
                    selectivity=0.5)
         .select("r.id"))
    res = q.run(_execute=make_executor(truth))
    sql_res = sess.sql(FILTER_SQL, order="as_written").run(
        _execute=make_executor(truth))
    assert res.rows == sql_res.rows
    assert "Scan reviews as r" in q.explain()


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
    assert first_q[0] == "q2:"
    seen2 = {}
    sess.sql(sql, order="as_written").run(
        _execute=make_executor(truth, seen=seen2))
    assert seen2["payload"]["filters"]["r"][0][0] == "q1:"


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


def test_unknown_model_refused_at_session():
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


def test_store_payload_and_warm_tracking(sess):
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0]}}
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER("
           "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})")
    seen = {}
    res = sess.sql(sql).run(_execute=make_executor(truth, seen=seen))
    store = seen["payload"]["store"]
    # default config: 64 GB container minus 16 GB headroom
    assert store is not None
    assert store["capacity_bytes"] == pytest.approx(48e9)
    assert store["min_doc_tokens"] == 1     # everything fits
    assert "r" in store["hashes"] and len(store["hashes"]["r"]) == 16
    assert res.report["store"] is None      # fake executor has none

    # a fake executor that reports store activity marks the session
    # warm, so the NEXT plan chooses restore
    def with_store(payload):
        out = make_executor(truth)(payload)
        out["store"] = {"r": dict(stored_docs=6, restored_docs=0)}
        return out

    sess.sql(sql).run(_execute=with_store)
    q2 = sess.sql(sql)
    plan = q2.plan()
    scan = next(op for op in plan.operators if op["op"] == "DocScan")
    assert scan["access"] == "restore"


def test_store_disabled_when_no_cpu_memory(tmp_path):
    import quail
    from quail.planner.plan import EngineConfig
    s = quail.Session(EngineConfig(cpu_memory_gb=0), tokenizer=fake_tok)
    s.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", {
            "id": ["r0"], "review": ["w " * 30]}), id_col="id"))
    assert s.store_spec(["h"]) is None


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


def test_payload_carries_yes_no_and_join_spec(sess):
    sql = """
        SELECT r.id, p.asin FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('Does {0} match {1}? Answer.', r.review,
                              p.description), {'selectivity': 0.5})
    """
    seen = {}
    join = {("r", "p"): lambda a, p: 0}
    sess.sql(sql).run(_execute=make_executor({}, join, seen=seen))
    payload = seen["payload"]
    assert payload["yes_ids"] and payload["no_ids"]
    # the engine preamble ships once, not inside any join segment
    from quail.logical import (SHARED_PRE, join_anchor_note,
                               join_label, render_join_question)
    assert payload["pre_ids"] == fake_tok(SHARED_PRE)
    j = payload["joins"][0]
    assert "pre" not in j
    assert j["anchor"] == "r" and j["partners"] == ["p"]
    # the naming line goes into the anchor's kept KV once per anchor;
    # each partner's block label (its own placeholder marker) and the
    # question (the raw template, markers kept - never filled in)
    # ride per tuple. r is placeholder 0, p is placeholder 1.
    assert j["frame"] == fake_tok(join_anchor_note(0))
    assert j["labels"] == {"p": fake_tok(join_label(1))}
    assert j["tail"] == fake_tok("\n\nDoes {0} match {1}? Answer.")
    logical = sess.sql(sql).logical
    pred = logical.root.input.predicate
    assert j["tail"] == fake_tok(render_join_question(pred.template))


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
