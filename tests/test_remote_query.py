"""Remote source planning and execution tests."""

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.execution import PhysicalResponse
from quail.extensions import ExtensionManifest
from quail.planner.plan import EngineConfig


def _tokens(text):
    return text.split()


def test_worker_reads_tokenizes_plans_and_projects_remote_source(
    tmp_path, monkeypatch
):
    from quail.execution import export_physical_outputs
    from quail.builtins import built_in_registry
    from quail.physical import DocumentInput, PackedFilter, decode_graph
    from quail.runtime import local as local_runtime
    from quail.runtime import worker
    from quail.runtime.runner import NodeMetrics, NodeResult, RunResult
    from quail.runtime.session import Session

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
    request = {
        "logical_plan": query.logical,
        "sources": {"docs": {"remote": {
            "type": "parquet",
            "paths": [str(path)],
            "id_col": "id",
        }}},
        "config": EngineConfig(
            gpus=1, model="qwen3-4b-fp8", backend="quail"
        ),
        "device": "h100-sxm",
        "order": None,
        "extensions": ExtensionManifest().to_value(),
    }

    monkeypatch.setattr(Session, "tokenizer", property(lambda self: _tokens))
    monkeypatch.setattr(Session, "_fast_tokenizer", lambda self: None)

    def execute(physical):
        graph = decode_graph(
            physical.plan["graph"], built_in_registry().codecs
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

    response = worker._execute_logical_query(request, 1)

    assert response.collect().to_pydict() == {"d.id": ["a"]}
    assert response.report["fresh_tokens"] == 4
