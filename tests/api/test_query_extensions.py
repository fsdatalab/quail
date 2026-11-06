"""Query execution with custom providers and extensions."""

import pyarrow as pa
import pyarrow.parquet as pq
from helpers.sessions import TOKENIZER, encoder
from modal._serialization import deserialize, serialize

import quail
from quail.api import execute as runtime
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.execution.result import filter_answer_table
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import AiFilter, Scan, decode_graph


def test_query_scans_provider_and_preserves_extension_objects(
    tmp_path, monkeypatch
):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": ["a", "b"],
        "body": ["first document", "second document"],
        "unused": [1, 2],
    }), path)
    session = quail.Session(
        quail.EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                           device="h100-sxm"),
        tokenizer=TOKENIZER,
    )
    calls = []

    def execute(physical, registry):
        assert registry is session.registry
        calls.append("execute")
        assert "extensions" not in physical.plan
        assert physical.plan["device"] == session.config.device
        graph = decode_graph(physical.plan["graph"], registry.codecs)
        source = next(node for node in graph.nodes if isinstance(node, Scan))
        documents = physical.inputs[source.input_id].documents
        assert [list(document) for document in documents] == [
            encoder()("first document"), encoder()("second document")]
        filter_node = next(node for node in graph.nodes if isinstance(node, AiFilter))
        run = RunResult(None, {filter_node.node_id: NodeResult({
            "rows:d": [0],
            "filter_answers:d": filter_answer_table(
                "d", [0, 1], [filter_node.written_position] * 2, [True, False]),
        })}, NodeMetrics())
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {"wall_s": 1.0, "boot_s": 0.0, "fresh_tokens": 4},
        )

    monkeypatch.setattr(runtime, "gpu_problem", lambda: None)
    monkeypatch.setattr(runtime, "_execute_physical", execute)
    monkeypatch.setattr(runtime, "_prepare_backend", lambda plan, registry: None)

    session.register(
        "documents", quail.DocumentProvider.from_parquet(path, id_col="id")
    )
    response = session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('keep {0}', d.body))"
    ).run()
    materialized = quail.QueryResult.from_table(response.collect(), response.report)
    materialized.plan = response.plan
    materialized.node_metrics = response.node_metrics
    received = deserialize(serialize(materialized), None)

    assert response.collect().to_pydict() == {"d.id": ["a"]}
    assert response.report["fresh_tokens"] == 4
    assert calls == ["execute"]
    assert received.collect().equals(response.collect())
    assert received.plan == response.plan
    assert received.node_metrics == response.node_metrics
    session.close()
