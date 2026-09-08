"""End-to-end Session tests with a fake executor.

Covers gating, tuple assembly, projection, and the report.
"""

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
    from quail.runtime.local import execute_worker_query

    return execute_worker_query(query, physical_executor=execute)


def runtime_plan(request):
    from quail.builtins import built_in_registry
    from quail.physical import (
        AnchoredJoin,
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
    return {
        "graph": graph,
        "filter_nodes": filter_nodes,
        "filters": {
            alias: [list(question)
                    for question in node.question_token_ids]
            for alias, node in filter_nodes.items()
        },
        "joins": [stage.runtime_spec() for node in graph.nodes
                  if isinstance(node, AnchoredJoin) for stage in node.stages],
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
    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(request):
        from test_quail_backend import graph_state

        from quail.backends.quail.graph import execute_single_graph
        from quail.execution import PhysicalResponse
        from quail.physical import DocumentInput, PackedFilter
        from quail.runtime.runner import NodeResult

        runtime = runtime_plan(request)
        if seen is not None:
            seen["request"] = request
            seen.update(runtime)
        graph = runtime["graph"]
        docs = {
            node.alias: request.inputs[node.input_id].documents
            for node in graph.nodes if isinstance(node, DocumentInput)
        }

        class FixedAnswers:
            def execute(self, node, inputs):
                if isinstance(node, PackedFilter):
                    rows = {}
                    for document in inputs["document_ids"]:
                        row = []
                        for question in node.question_token_ids:
                            bit = filter_truth[node.alias][
                                _match_key(node.alias, question)
                            ][document]
                            row.append(bit)
                            if not bit:
                                break
                        rows[document] = row
                    return NodeResult({
                        f"ids:{node.alias}": [d for d, row in rows.items()
                            if len(row) == len(node.stages) and all(row)],
                        f"filter_answers:{node.alias}": rows,
                    })
                anchors = list(inputs["anchor_ids"])
                outputs = {}
                for stage in node.stages:
                    partners = inputs["partner_indices"][stage.written_pos]
                    rule = join_truth[(stage.anchor, *stage.partners)]
                    rows = {i: [rule(a, *pair) for pair in partners]
                            for i, a in enumerate(anchors)}
                    outputs[f"join_answers:{stage.written_pos}"] = {
                        "rows": rows, "anchor_index": anchors,
                        "partner_index": partners, "anchor": stage.anchor,
                        "partners": list(stage.partners),
                        "semantics": stage.semantics,
                        "selectivity": stage.selectivity,
                        "written_pos": stage.written_pos,
                    }
                passing = {anchors[i] for i, row in rows.items() if any(row)}
                outputs[f"ids:{node.anchor}"] = [a for a in anchors
                    if (a not in passing if stage.semantics == "anti"
                        else a in passing)]
                return NodeResult(outputs)

        report = execute_single_graph(graph_state(FixedAnswers(), docs), {
            "filter_limit": None, "pre_ids": [],
        }, graph)
        outputs = report.pop("_outputs")
        return PhysicalResponse(outputs, {
            **report, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234,
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


class NodeTypes:
    """Observer that records the node types it saw."""

    name = "test.node_types"

    def __init__(self):
        self.types = []

    def after_node(self, node, result) -> None:
        self.types.append(node.type_name)

    def report(self) -> dict:
        return {"types": list(self.types)}


def _observed_result(tmp_path, registry):
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
    return _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor(truth))


def test_observer_sees_the_complete_physical_graph(tmp_path):
    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)

    result = _observed_result(tmp_path, registry)

    assert result.observer(NodeTypes)["types"] == [
        "quail.document_input",
        "quail.packed_filter",
        "quail.project",
        "quail.limit",
    ]
    assert result.observer(NodeTypes) is result.observer("test.node_types")
    with pytest.raises(KeyError):
        result.observer("example.missing")


def test_executed_plan_survives_the_report_round_trip(tmp_path):
    from quail.runtime.result import QueryResult

    registry = quail.ExtensionRegistry.with_built_ins()
    result = _observed_result(tmp_path, registry)
    # Saved reports can be loaded separately from their result tables.
    table, report = result.collect(), dict(result.report)

    restored = QueryResult.from_table(table, report=report)
    assert restored.plan is None
    restored.attach_executed_plan(registry.codecs)

    assert [node.node_id for node in restored.plan.topological_nodes()] == [
        node.node_id for node in result.plan.topological_nodes()]
    assert restored.node_metrics == result.node_metrics
    assert restored.explain() == result.explain()


def test_query_explain_uses_optimized_projection(sess):
    query = sess.sql("SELECT r.id FROM reviews r WHERE "
                     "AI_FILTER(PROMPT('q: {0}', r.review))")
    text = query.explain()
    assert "Scan reviews as r [review, id]" in text
    assert "admission_tokens=" not in text
    assert "admission_tokens=" in query.explain(verbose=True)
