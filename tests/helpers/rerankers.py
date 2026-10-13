"""A fake reranker and a runner that finishes a query through it."""

from quail.api.execute import execute_query
from quail.backends.quail.reranker import RerankerBatch, RerankerModelExecution
from quail.execution.metrics import scalar_node_metrics
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.execution.types import PhysicalResponse, export_physical_outputs


class _RowReranker:
    def __init__(self):
        self.calls = 0

    def score(self, spec, rows, documents):
        self.calls += 1
        scores = [(int(sum(row)) + 1) / 10 for row in rows]
        return RerankerBatch(scores, fresh_tokens=len(rows), cached_tokens=0)


def _run_graph(session, query, request, reranker):
    graph = compute_subgraph(query.plan().graph)
    model = RerankerModelExecution.__new__(RerankerModelExecution)
    model.reranker = reranker
    model.documents = {
        node.alias: request.inputs[node.input_id].documents
        for node in query.plan().nodes if node.type_name == "quail.scan"
    }
    run = GenericRunner().run(
        graph,
        ExecutionContext(
            runtimes=session.registry.runtimes,
            model_execution=model,
            sources={
                **{alias: range(2) for alias in model.documents},
                **request.relations,
            },
        ),
    )
    return graph, model, run


def _finish(query, session, reranker):
    def execute(request):
        graph, _, run = _run_graph(session, query, request, reranker)
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {"backend": "quail", "wall_s": 0.1, "fresh_tokens": 12,
             "cached_tokens": 3, "node_metrics": scalar_node_metrics(run.nodes)},
        )

    return execute_query(query, physical_executor=execute)
