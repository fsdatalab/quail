"""End-to-end Session tests with a fake executor."""

import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.sessions import CONFIG, TOKENIZER, TRUTH, _run, make_executor

import quail
from quail.api import execute as execution
from quail.api.config import EngineConfig
from quail.execution.result import (
    QueryResult,
    build_result_declaration,
    document_index_table,
)


@pytest.fixture()
def sess(tmp_path):
    s = quail.Session(CONFIG, tokenizer=TOKENIZER)
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
    }), tmp_path / "r.parquet")
    s.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"))
    yield s
    s.close()


def test_session_requires_model_and_device_and_loads_the_models_tokenizer(
        monkeypatch):
    from helpers.sessions import TOKENIZER

    with pytest.raises(TypeError, match="'model' and 'device'"):
        EngineConfig()
    with pytest.raises(TypeError, match="'device'"):
        EngineConfig(model="qwen3-4b-fp8")
    with pytest.raises(TypeError, match="required positional argument: 'config'"):
        quail.Session()
    assert (CONFIG.gpus, CONFIG.backend) == (1, "quail")

    sources = []

    def resolve(name, revision, **kwargs):
        sources.append(f"{name}@{revision}")
        return TOKENIZER

    monkeypatch.setattr("quail.api.session.resolve_model_path", resolve)
    with quail.Session(CONFIG) as session:
        token_ids = session.tokenizer("hello world")
        text = session.tokenizer.decode(token_ids)
    assert sources == ["Qwen/Qwen3-4B-FP8@" + session.model.revision]
    assert token_ids and all(type(token_id) is int for token_id in token_ids)
    assert text == b"hello world"


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_explain_analyze_shows_measured_rows_beside_estimates(
        sess, monkeypatch):
    query = sess.sql(FILTER_SQL + " LIMIT 1")
    execution_calls = []

    def executor(request):
        execution_calls.append(request)
        return make_executor(TRUTH)(request)

    original = execution.execute_query
    # Query.run binds execute_query at import, so patch it there
    monkeypatch.setattr(
        "quail.api.query.execute_query",
        lambda q, physical_executor=executor, plan=None: original(
            q, physical_executor=physical_executor, plan=plan))
    planned = query.explain()
    text = query.explain(analyze=True)

    assert len(execution_calls) == 1
    assert "   rows   " not in planned and "run:" not in planned
    header = next(line for line in text.splitlines() if "est. rows" in line)
    assert header.split() == ["est.", "rows", "rows", "est.", "pass", "pass",
                              "est.", "time", "time", "fresh", "tokens"]
    assert re.search(
        r"AiFilter: r\s+3\s+4\s+[\d.]+ ms\s+(?:<1|[\d.]+) ms\s+60$",
        text, re.M)
    # Both predicates have equal estimated cost and pass four of six rows.
    first = re.search(r"(?:1st: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                      r"\{0\}\\n\\nq\1:'\)\s+6\s+6\s+50%\s+66\.7%$",
                      text, re.M)
    second = re.search(r"(?:2nd: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                       r"\{0\}\\n\\nq\1:'\)\s+3\s+4\s+50%\s+50%$",
                       text, re.M)
    assert first and second and first[1] != second[1]
    assert re.search(r"Scan reviews as r\s+6\s+6\s+\S+ ms$", text, re.M)
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", text, re.M)
    assert "run:" in text
    assert "query time   1 s (model startup excluded)" in text
    assert "startup      500 ms" in text
    assert "throughput   6 documents/second over 6 input documents" in text
    assert "tokens       1,234 fresh" in text
    assert "GPU cost     $0.0011 per query (1 GPU at $3.9492/hour" in text

    # the executed result renders the same measured table on its own
    result = _run(sess.sql(FILTER_SQL + " LIMIT 1"), make_executor(TRUTH))
    explained = result.explain()
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", explained, re.M)
    assert re.search(rf"(?:1st: )?predicate {first[1]}\s+6\s+6\s+50%\s+66\.7%$",
                     explained, re.M)
    stages = [s for s in result.report["stages"] if s["op"] == "filter"]
    assert [s["written_position"] for s in stages] == [int(first[1]) - 1,
                                                int(second[1]) - 1]


def test_query_rows_observers_saved_reports_and_streamed_results(sess):
    res = _run(sess.sql(FILTER_SQL), make_executor(TRUTH))
    assert res.columns == ["r.id"]
    assert sorted(res.to_rows()) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6, abs=1e-3)
    assert stages[1]["evaluated"] == 4, "stage 2 only saw stage-1 survivors"
    assert res.report["wall_s"] == res.report["model_wall_s"] == 1.0
    assert res.report["finish_s"] >= 0
    assert res.report["input_ready_s"] >= res.report["planning_s"]
    assert "gpu_s" not in res.report

    def timed(request):
        response = make_executor(TRUTH)(request)
        response.metrics["gpu_s"] = 0.75
        return response

    limited = _run(sess.sql(FILTER_SQL + " LIMIT 1"), timed)
    assert limited.count() == 1
    assert limited.report["gpu_s"] == 0.75

    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)
    session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=registry)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "review": ["first", "second"]}), id_col="id"))
    result = _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor({"r": {"q:": [1, 0]}}))
    assert result.observer(NodeTypes)["types"] == [
        "quail.scan", "quail.ai_filter"]
    assert result.observer(NodeTypes) is result.observer("test.node_types")
    with pytest.raises(KeyError):
        result.observer("example.missing")

    # saved reports can be loaded separately from their result tables
    restored = QueryResult.from_table(result.collect(), report=dict(result.report))
    assert restored.plan is None
    restored.attach_executed_plan()
    assert [node.node_id for node in restored.plan.topological_nodes()] == [
        node.node_id for node in result.plan.topological_nodes()]
    assert restored.node_metrics == result.node_metrics
    assert restored.explain() == result.explain()
    # a node's extra counters are saved beside the standard fields and
    # return to its extension metrics when the report is loaded
    report = dict(result.report)
    report["node_metrics"] = {
        node_id: {**metrics, "borrowed_prefix_tokens": 3}
        for node_id, metrics in report["node_metrics"].items()}
    counted = QueryResult.from_table(result.collect(), report=report)
    counted.attach_executed_plan()
    assert all(metrics.extension == {"borrowed_prefix_tokens": 3}
               for metrics in counted.node_metrics.values())
    assert {node_id: metrics.wall_s
            for node_id, metrics in counted.node_metrics.items()} == {
        node_id: metrics.wall_s
        for node_id, metrics in result.node_metrics.items()}

    # results are assembled from survivor indices and join tables
    true_join_tables = [
        document_index_table({"r1": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1, 1], "a2": [0, 0, 1]}, "join_answers"),
    ]
    survivors = {alias: pa.array(rows, type=pa.int32()) for alias, rows in
                 {"r1": [0, 1], "r2": [0, 1], "a1": [0], "a2": [0, 1]}.items()}
    declaration, _ = build_result_declaration(true_join_tables, survivors, "r1")
    assert declaration.to_table().num_rows == 6

    relation = document_index_table({"r": list(range(5))}, "filter_survivors")
    assert relation.schema.metadata[b"quail.kind"] == b"filter_survivors"
    result = QueryResult.from_table(
        pa.table({"r.id": [f"r{i}" for i in range(5)]}))

    reader = result.execute_stream(batch_rows=2)
    batches = list(reader)
    assert isinstance(reader, pa.RecordBatchReader)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.Table.from_batches(batches).column("r.id").to_pylist() == [
        "r0", "r1", "r2", "r3", "r4"]
    assert result.count() == 5
    assert result.collect(limit=3).column("r.id").to_pylist() == ["r0", "r1", "r2"]
    assert not hasattr(result, "rows")


def test_order_by_offset_and_distinct_run_in_datafusion_on_the_regions_rows():
    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)
    session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=registry)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": [f"r{i}" for i in range(6)],
                  "review": [f"review {i} " + "pad " * 20 for i in range(6)],
                  "stars": [3, 1, 4, 5, None, 2]}), id_col="id"))
    # survivors of both predicates are r0 and r3; sorted by a column the
    # query does not return, so the first survivor to finish is not first
    result = _run(session.sql(FILTER_SQL + " ORDER BY r.stars DESC LIMIT 1"),
                  make_executor(TRUTH))
    assert result.to_rows() == [("r3",)]
    assert result.count() == 1
    assert result.observer(NodeTypes)["types"] == [
        "quail.scan", "quail.ai_filter", "quail.ai_filter"]
    assert "SortExec: TopK(fetch=1), expr=[stars@1 DESC]" in result.explain()

    result = _run(session.sql(
        "SELECT r.id, r.stars FROM reviews r WHERE AI_FILTER(PROMPT("
        "'q1: {0}', r.review)) ORDER BY r.stars NULLS FIRST OFFSET 1"
    ), make_executor(TRUTH))
    assert result.to_rows() == [("r1", 1), ("r0", 3), ("r3", 5)]

    result = _run(session.sql(
        "SELECT DISTINCT r.stars FROM reviews r WHERE AI_FILTER(PROMPT("
        "'q2: {0}', r.review)) ORDER BY r.stars LIMIT 2"
    ), make_executor({"r": {"q2:": [1, 1, 1, 1, 1, 1]}}))
    assert result.to_rows() == [(1,), (2,)]

    result = _run(session.sql(
        "SELECT DISTINCT r.stars FROM reviews r WHERE AI_FILTER(PROMPT("
        "'q2: {0}', r.review)) ORDER BY r.stars DESC NULLS LAST LIMIT 2 OFFSET 1"
    ), make_executor({"r": {"q2:": [1, 1, 1, 1, 1, 1]}}))
    assert result.to_rows() == [(4,), (3,)]
    session.close()


class NodeTypes:
    """Observer that records the node types it saw."""

    name = "test.node_types"

    def __init__(self):
        self.types = []

    def after_node(self, node, result) -> None:
        self.types.append(node.type_name)

    def report(self) -> dict:
        return {"types": list(self.types)}
