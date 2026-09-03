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


def _run(query, execute):
    from quail.runtime.worker import execute_worker_query

    return execute_worker_query(query, physical_executor=execute)


def runtime_plan(request):
    from quail.extensions import built_in_registry
    from quail.physical import (
        AdaptiveJoinPlan,
        DocumentInput,
        PackedFilter,
        decode_graph,
    )

    graph = decode_graph(
        request.plan["graph"], built_in_registry().codecs
    )
    filter_nodes = {
        node.alias: node for node in graph.nodes
        if isinstance(node, PackedFilter)
    }
    adaptive = next(
        (node for node in graph.nodes
         if isinstance(node, AdaptiveJoinPlan)),
        None,
    )
    return {
        "graph": graph,
        "filter_nodes": filter_nodes,
        "filters": {
            alias: [list(question)
                    for question in node.question_token_ids]
            for alias, node in filter_nodes.items()
        },
        "joins": [] if adaptive is None else list(adaptive.join_specs),
        "shards": {
            node.alias: node.shards for node in graph.nodes
            if isinstance(node, DocumentInput)
        },
    }


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
    """Build a fake executor from filter and join truth tables."""
    import itertools

    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(request):
        from quail.execution import PhysicalResponse, export_physical_outputs
        from quail.physical import AdaptiveJoinPlan, DocumentInput, PackedFilter
        from quail.runtime.runner import NodeMetrics, NodeResult, RunResult

        runtime = runtime_plan(request)
        if seen is not None:
            seen["request"] = request
            seen.update(runtime)
        graph = runtime["graph"]
        docs = {
            node.alias: request.inputs[node.input_id].documents
            for node in graph.nodes if isinstance(node, DocumentInput)
        }
        node_results = {}
        survivors = {
            alias: list(range(len(table))) for alias, table in docs.items()
        }
        for alias, qids in runtime["filters"].items():
            rows = {}
            for d in range(len(docs[alias])):
                row = []
                for q in qids:
                    bit = filter_truth[alias][_match_key(alias, q)][d]
                    row.append(bit)
                    if not bit:
                        break
                rows[d] = row
            survivors[alias] = [d for d, r in rows.items()
                                if len(r) == len(qids) and all(r)]
            node = next(
                node for node in graph.nodes
                if isinstance(node, PackedFilter) and node.alias == alias
            )
            node_results[node.node_id] = NodeResult({
                f"ids:{alias}": survivors[alias],
                f"filter_answers:{alias}": rows,
            })
        join_outputs = {}
        for j in runtime["joins"]:
            anchors = list(survivors[j["anchor"]])
            tuples = [list(t) for t in itertools.product(
                *[survivors[p] for p in j["partners"]])]
            rule = join_truth[(j["anchor"], *j["partners"])]
            rows = {ai: [rule(a, *t) for t in tuples]
                    for ai, a in enumerate(anchors)}
            join_outputs[f"join_answers:{j['written_pos']}"] = dict(
                rows=rows,
                anchor_index=anchors,
                partner_index=tuples,
                anchor=j["anchor"],
                partners=j["partners"],
                semantics=j["semantics"],
                selectivity=j["selectivity"],
                written_pos=j["written_pos"],
            )
            kept = {anchors[ai] for ai, r in rows.items() if any(r)}
            if j["semantics"] == "anti":
                survivors[j["anchor"]] = [a for a in anchors
                                          if a not in kept]
            else:
                survivors[j["anchor"]] = sorted(kept)
        adaptive = next(
            (node for node in graph.nodes
             if isinstance(node, AdaptiveJoinPlan)),
            None,
        )
        if adaptive is not None:
            node_results[adaptive.node_id] = NodeResult({
                **{
                    f"ids:{alias}": survivors[alias]
                    for alias in adaptive.aliases
                },
                **join_outputs,
            })
        outputs = export_physical_outputs(
            graph, RunResult(None, node_results, NodeMetrics())
        )
        return PhysicalResponse(outputs, {
            "wall_s": 1.0,
            "boot_s": 0.5,
            "fresh_tokens": 1234,
        })

    return _exec


def test_remote_source_query_does_not_scan_or_tokenize_on_client():
    from quail.catalog import TableStatistics
    from quail.runtime.result import QueryResult

    class RemoteProvider:
        id_col = "id"
        columns = ("id", "body")

        def schema(self):
            return pa.schema({"id": pa.string(), "body": pa.string()})

        def content_identity(self):
            return "remote:test"

        def statistics(self):
            return TableStatistics(row_count=2)

        def scan(self, request):
            raise AssertionError("the client scanned a remote source")

        def remote_source(self):
            return {
                "type": "parquet",
                "paths": ["s3://bucket/docs.parquet"],
                "id_col": "id",
            }

    class RemoteCompute:
        def __init__(self):
            self.request = None

        def execute(self, request):
            self.request = request
            return QueryResult.from_table(
                pa.table({"d.id": ["a"]}),
                report={"wall_s": 1.0, "fresh_tokens": 5},
            )

        def close(self):
            pass

    compute = RemoteCompute()
    session = quail.Session(tokenizer=fake_tok, compute_provider=compute)
    session.register("docs", RemoteProvider())

    result = session.sql(
        "SELECT d.id FROM docs d WHERE "
        "AI_FILTER(PROMPT('question {0}', d.body))"
    ).run()

    assert result.to_rows() == [("a",)]
    assert compute.request.providers["docs"].remote_source()["paths"] == [
        "s3://bucket/docs.parquet"
    ]
    assert compute.request.logical_plan.root.type_name \
        == "quail.logical_project"


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_filter_query_rows_and_report(sess):
    truth = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
                         "q2:": [1, 0, 1, 1, 0, 1]}}
    q = sess.sql(FILTER_SQL)
    res = _run(q, make_executor(truth))
    assert res.columns == ["r.id"]
    assert sorted(res.to_rows()) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6,
                                                              abs=1e-3)
    # stage 2 only saw stage-1 survivors
    assert stages[1]["evaluated"] == 4
    assert res.report["wall_s"] == 1.0

    limited = _run(
        sess.sql(FILTER_SQL + " LIMIT 1"), make_executor(truth))
    assert limited.count() == 1


def test_observer_sees_the_complete_physical_graph(tmp_path):
    registry = quail.ExtensionRegistry.with_built_ins()
    registry.load_extension(
        "quail_ext_examples.plan_trace",
        local_python_sources=("quail_ext_examples",),
    )
    session = quail.Session(
        EngineConfig(gpus=1),
        tokenizer=fake_tok,
        registry=registry,
        compute_provider=object(),
    )
    session.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "reviews.parquet", {
            "id": ["r0", "r1"],
            "review": ["first", "second"],
        }),
        id_col="id",
    ))
    truth = {"r": {"q:": [1, 0]}}

    result = _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor(truth))

    nodes = result.report["observers"]["example.plan_trace"]["nodes"]
    assert [node["node_type"] for node in nodes] == [
        "quail.document_input",
        "quail.packed_filter",
        "quail.project",
        "quail.limit",
    ]


def test_sql_query_streams_and_collects_arrow(sess):
    truth = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
                   "q2:": [1, 0, 1, 1, 0, 1]}}
    query = sess.sql(FILTER_SQL)

    reader = _run(
        query, make_executor(truth)).execute_stream(batch_rows=1)
    batches = list(reader)
    assert isinstance(reader, pa.RecordBatchReader)
    assert [len(batch) for batch in batches] == [1, 1]

    table = _run(query, make_executor(truth)).collect(limit=1)
    assert isinstance(table, pa.Table)
    assert table.column("r.id").to_pylist() == ["r0"]


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
    res = _run(sess.sql(sql), make_executor(truth, join))
    # survivors r0, r2, r4; pairs pass when (a+p) % 4 == 0
    expect = sorted(("r%d" % a, "p%d" % p)
                    for a in (0, 2, 4) for p in range(4)
                    if (a + p) % 4 == 0)
    assert sorted(res.to_rows()) == expect
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
    res = _run(sess.sql(sql), make_executor({}, join))
    assert sorted(res.to_rows()) == [("r3",), ("r4",), ("r5",)]


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
    _run(sess.sql(sql), make_executor(truth, seen=seen))
    # by_cost (the default: every predicate has a selectivity) runs
    # the 0.1 filter first
    first_q = seen["filters"]["r"][0]
    assert "q2:" in first_q
    seen2 = {}
    _run(
        sess.sql(sql, order="as_written"),
        make_executor(truth, seen=seen2),
    )
    assert "q1:" in seen2["filters"]["r"][0]


def test_builder_can_request_cost_order(sess):
    def build(order=None):
        query = (sess.docs("reviews").alias("r")
                 .ai_filter(
                     quail.prompt("q1: {0}", quail.col("r.review")),
                     selectivity=0.9)
                 .ai_filter(
                     quail.prompt("q2: {0}", quail.col("r.review")),
                     selectivity=0.1))
        if order is None:
            return query.select("r.id")
        return query.select("r.id", order=order)

    assert build().plan().settings["order_rule"] == "as_written"
    assert build("by_cost").plan().settings["order_rule"] == "by_cost"


def test_payload_carries_filter_arena_writes(sess):
    # one stage: nothing reads the KV again - the planner turns
    # writes off
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0]}}
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER("
           "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})")
    seen = {}
    _run(sess.sql(sql), make_executor(truth, seen=seen))
    assert seen["filter_nodes"]["r"].arena_writes is False
    assert "arena_writes=False" in sess.sql(sql).explain()

    # a second stage re-reads survivors' KV
    truth2 = {"r": {"q1:": [1] * 6, "q2:": [1] * 6}}
    seen = {}
    _run(sess.sql(FILTER_SQL), make_executor(truth2, seen=seen))
    assert seen["filter_nodes"]["r"].arena_writes is True


def test_request_carries_file_backed_document_inputs(sess):
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0]}}
    seen = {}

    _run(sess.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER("
        "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})"
    ), make_executor(truth, seen=seen))

    request = seen["request"]
    source = request.inputs["r"]
    assert len(source) == 6
    assert list(source.documents[0])[:2] == ["review", "0"]


def test_projection_uses_the_tokenization_scan(tmp_path):
    from quail.catalog import TableStatistics

    table = pa.table({
        "id": ["a", "b"],
        "body": ["first document", "second document"],
        "unused": [1, 2],
    })

    class CountingProvider:
        id_col = "id"
        columns = tuple(table.column_names)

        def __init__(self):
            self.requests = []

        def schema(self):
            return table.schema

        def content_identity(self):
            return "counting-provider"

        def statistics(self):
            return TableStatistics(row_count=len(table))

        def scan(self, request):
            self.requests.append(request)
            selected = table.select(request.columns)
            return selected.to_reader(max_chunksize=1)

        def remote_source(self):
            return None

    provider = CountingProvider()
    session = quail.Session(tokenizer=fake_tok)
    session.register("docs", provider)
    truth = {"d": {"q:": [1, 0]}}

    result = _run(
        session.sql(
            "SELECT d.id FROM docs d WHERE "
            "AI_FILTER(PROMPT('q: {0}', d.body))"
        ),
        make_executor(truth),
    )

    assert result.collect().to_pydict() == {"d.id": ["a"]}
    assert [request.columns for request in provider.requests] == [
        ("body", "id")
    ]


def test_physical_request_has_one_plan_and_arrow_inputs(sess):
    from quail.runtime.worker import _validate_physical_request

    query = sess.sql(FILTER_SQL)
    request = query._prepare_physical()
    assert request.plan["backend"] == "quail"
    _validate_physical_request(request)
    assert set(request.inputs) == {"r"}
    runtime = runtime_plan(request)
    assert runtime["filters"]
    assert runtime["filter_nodes"]["r"].arena_writes is True
    assert runtime["shards"]


def test_session_uses_quail_backend_by_default(sess):
    assert sess.config.backend == "quail"
    assert sess.registry.backend(sess.config.backend).name == "quail"


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
        _run(q, make_executor({}))
    assert e.value.refusal.constraint == "suffix_over_chunk"
    with pytest.raises(quail.RefusalError):
        quail.Session(EngineConfig(model="qwen9-99b"),
                      tokenizer=fake_tok)
    with pytest.raises(quail.RefusalError) as unknown_backend:
        quail.Session(EngineConfig(backend="missing"),
                      tokenizer=fake_tok)
    assert unknown_backend.value.refusal.constraint == "unknown_backend"
    with pytest.raises(quail.RefusalError) as unsupported_gpus:
        quail.Session(EngineConfig(gpus=3), tokenizer=fake_tok)
    assert unsupported_gpus.value.refusal.constraint == \
        "unsupported_backend_configuration"


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
    _run(
        s.sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
              "PROMPT('q1: {0}', r.review), {'selectivity': 0.5})"),
        make_executor(truth, seen=seen),
    )
    request = seen["request"]
    assert request.gpu_count == 2
    assert len(seen["shards"]["r"]) == 2
    covered = sorted(i for sh in seen["shards"]["r"] for i in sh)
    assert covered == list(range(6))


def test_plan_carries_true_false_and_join_spec(sess):
    from quail.runtime.worker import _validate_physical_request

    sql = """
        SELECT r.id, p.asin FROM reviews r
        JOIN products p
          ON AI_FILTER(PROMPT('Does {0} match {1}? Answer.', r.review,
                              p.description), {'selectivity': 0.5})
    """
    seen = {}
    join = {("r", "p"): lambda a, p: 0}
    query = sess.sql(sql)
    _run(query, make_executor({}, join, seen=seen))
    pred = query.logical.root.input.predicate
    request = seen["request"]
    _validate_physical_request(request)
    settings = request.plan["settings"]
    assert settings["true_ids"] and settings["false_ids"]
    # the engine preamble ships once, not inside any join segment
    from quail.logical import (SHARED_PRE, join_label,
                               render_join_frame)
    assert settings["pre_ids"] == fake_tok(SHARED_PRE)
    j = seen["joins"][0]
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
    res = _run(q, make_executor({}, join))
    expect = sorted((f"r{a}", f"p{p}", f"g{g}")
                    for a in range(6) for p in range(4)
                    for g in range(3) if (a + p + g) % 5 == 0)
    assert sorted(res.to_rows()) == expect
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
    res = _run(
        _chain_query(sess), make_executor({}, join, seen=seen))
    # both stages anchored on the shared table, one per payload entry
    assert [j["anchor"] for j in seen["joins"]] == \
        ["p", "p"]
    assert sorted(res.to_rows()) == CHAIN_EXPECT
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
    res = _run(q, make_executor({}, join))
    assert sorted(res.to_rows()) == [t for t in CHAIN_EXPECT
                                     if t[2] != "g0"]


def test_two_join_chain_limit_caps_final_triples(sess, tmp_path):
    _register_tags(sess, tmp_path)
    seen = {}
    join = {("p", "r"): J1, ("p", "g"): J2}
    res = _run(
        _chain_query(sess, limit=3),
        make_executor({}, join, seen=seen),
    )
    # no upstream cut: the cap applies to the final triples only
    assert seen["request"].plan["settings"]["filter_limit"] is None
    assert res.count() == 3
    assert set(res.to_rows()) <= set(CHAIN_EXPECT)
    assert len(CHAIN_EXPECT) > 3


def test_limit_join_payload_carries_no_filter_limit(sess):
    # #39: with a join, the filter round must not stop at LIMIT
    # survivors - the request ships limit=None and Query.finish caps the
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
    assert q.plan().graph.node("limit").count == 2
    res = _run(q, make_executor(truth, join, seen=seen))
    assert seen["request"].plan["settings"]["filter_limit"] is None
    assert res.count() == 2
    assert all(r == "r4" for r, _ in res.to_rows())


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
    from quail.backends.quail import expected_join_nodes

    kinds = [type(n).__name__ for n in expected_join_nodes(plan)]
    assert kinds.count("AnchoredJoin") == 2
    assert kinds.count("Exchange") == 1
    seen = {}
    join = {("r", "p"): lambda r, p: J1(p, r),
            ("g", "p"): lambda g, p: J2(p, g)}
    res = _run(q, make_executor({}, join, seen=seen))
    assert [j["anchor"] for j in seen["joins"]] == \
        ["r", "g"]
    # same pair semantics as the shared-anchor chain, same triples
    assert sorted(res.to_rows()) == CHAIN_EXPECT


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
    res = _run(q, make_executor({}, join))
    # p1 is matched by the anti gate and drops; every (r, p!=1) stays
    assert sorted(res.to_rows()) == sorted(
        (f"r{a}", f"p{p}") for a in range(6) for p in (0, 2, 3))
