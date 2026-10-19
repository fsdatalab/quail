"""A runner that executes a query's compute graph on a reranker."""

from quail.backends.quail.reranker import RerankerModelExecution
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph


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
