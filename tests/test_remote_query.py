"""Remote source planning and execution tests."""

from dataclasses import replace
from threading import Lock

import pyarrow as pa
import pyarrow.parquet as pq
from modal._serialization import deserialize, serialize

import quail
from quail.execution import PhysicalResponse
from quail.planner.plan import EngineConfig


def _tokens(text):
    return text.split()


def test_worker_reads_tokenizes_plans_and_projects_remote_source(
    tmp_path, monkeypatch
):
    from quail.execution import export_physical_outputs
    from quail.physical import DocumentInput, PackedFilter, decode_graph
    from quail.planning import SupportResult
    from quail.runtime import local as local_runtime
    from quail.runtime import worker
    from quail.runtime.runner import NodeMetrics, NodeResult, RunResult
    from quail.runtime.session import Session
    from quail.specs import H100_SXM

    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": ["a", "b"],
        "body": ["first document", "second document"],
        "unused": [1, 2],
    }), path)
    local = quail.Session(tokenizer=_tokens, compute_provider=object())
    local.register(
        "docs",
        quail.DocumentProvider.from_parquet(str(path), id_col="id"),
    )
    query = local.sql(
        "SELECT d.id FROM docs d WHERE "
        "AI_FILTER(PROMPT('keep {0}', d.body))"
    )
    local.registry.register_device(replace(H100_SXM, name="test-h100"))
    monkeypatch.setattr(local.registry.backend("quail"), "supports",
                        lambda model, device, count: SupportResult.accept())
    request = {
        "logical_plan": query.logical,
        "sources": {"docs": {"remote": {
            "type": "test.source",
            "paths": [str(path)],
            "id_col": "id",
        }}},
        "config": EngineConfig(
            gpus=1, model="qwen3-4b-fp8", backend="quail", device="test-h100"
        ),
        "order": None,
        "registry": local.registry,
    }

    monkeypatch.setattr(Session, "tokenizer", property(lambda self: _tokens))
    monkeypatch.setattr(Session, "_fast_tokenizer", lambda self: None)
    calls = []

    def initialize(registry):
        assert registry is local.registry
        calls.append("initialize")
        lock = Lock()

        def open_source(source):
            with lock:
                calls.append("source")
            return quail.DocumentProvider.from_parquet(source["paths"], id_col="id")

        class CheckPlanning:
            name = "test.planning"

            def rewrite(self, graph, context):
                with lock:
                    calls.append("plan")
                return None

        registry.register_source_reader(open_source, source_type="test.source")
        registry.register_physical_rule(CheckPlanning())

    def execute(physical, registry):
        assert registry is local.registry
        assert calls == ["initialize", "source", "plan"]
        calls.append("execute")
        assert "extensions" not in physical.plan
        assert physical.plan["device"] == "test-h100"
        graph = decode_graph(
            physical.plan["graph"], registry.codecs
        )
        source = next(
            node for node in graph.nodes if isinstance(node, DocumentInput)
        )
        documents = physical.inputs[source.input_id].documents
        assert [list(document) for document in documents] == [
            ["first", "document"],
            ["second", "document"],
        ]
        filter_node = next(
            node for node in graph.nodes if isinstance(node, PackedFilter)
        )
        run = RunResult(
            None,
            {
                filter_node.node_id: NodeResult({
                    "ids:d": [0],
                    "filter_answers:d": {0: [True], 1: [False]},
                })
            },
            NodeMetrics(),
        )
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {"wall_s": 1.0, "boot_s": 0.0, "fresh_tokens": 4},
        )

    monkeypatch.setattr(local_runtime, "_execute_physical", execute)
    monkeypatch.setattr(local_runtime, "_prepare_backend",
                        lambda plan, registry: None)

    response = worker._execute_logical_query(request, 1, initialize)
    received = deserialize(serialize(response), None)

    assert response.collect().to_pydict() == {"d.id": ["a"]}
    assert response.report["fresh_tokens"] == 4
    assert calls == ["initialize", "source", "plan", "execute"]
    assert received.collect().equals(response.collect())
    assert received.plan == response.plan
    assert received.node_metrics == response.node_metrics
