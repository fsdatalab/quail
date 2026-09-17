"""AI.SCORE parsing, planning, execution, and score semantics."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.execution.execute import execute_query
from quail.execution.reranker import (
    Qwen3VllmReranker,
    RerankerBatch,
    RerankerModelExecution,
    ScoreFilterRuntime,
    compare_score,
    normalized_yes_score,
)
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.frontend.sql import compile_sql
from quail.logical import CompileError, ScoreExpression, SemanticJoin
from quail.physical import AiScore, ScoreFilter, decode_graph, encode_graph
from quail.planner.plan import EngineConfig


def _tokens(text):
    return text.split()


def _provider(path, columns, *, id_col):
    pq.write_table(pa.table(columns), path)
    return DocumentProvider.from_parquet(str(path), id_col=id_col)


@pytest.fixture()
def catalog(tmp_path):
    value = Catalog()
    value.register("documents", _provider(
        tmp_path / "documents.parquet",
        {"id": [1, 2], "body": ["refund please", "all good"]},
        id_col="id",
    ))
    value.register("queries", _provider(
        tmp_path / "queries.parquet",
        {"id": [1, 2], "text": ["refund", "shipping"]},
        id_col="id",
    ))
    return value


def _session(catalog, *, model="qwen3-reranker-0.6b-bf16", gpus=1):
    session = quail.Session(
        EngineConfig(model=model, device="h100-sxm", gpus=gpus),
        tokenizer=_tokens,
    )
    for name in ("documents", "queries"):
        session.register(name, catalog.get(name))
    return session


def test_score_projection_is_a_named_numeric_expression(catalog):
    plan = compile_sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d",
        catalog,
        _tokens,
    )
    score = plan.root.columns[1]
    assert isinstance(score, ScoreExpression)
    assert score.name == "score"
    assert score.prompt.args[0].alias == "d"


def test_score_join_comparison_uses_two_relations(catalog):
    plan = compile_sql(
        "SELECT q.id, d.id FROM queries q JOIN documents d ON "
        "AI.SCORE(PROMPT('Is {1} relevant to {0}', q.text, d.body)) "
        "> 0.8",
        catalog,
        _tokens,
    )
    node = plan.root.input
    assert isinstance(node, SemanticJoin)
    assert node.comparison == ">"
    assert node.threshold == 0.8
    assert tuple(ref.alias for ref in node.predicate.args) == ("q", "d")


@pytest.mark.parametrize("sql", [
    "SELECT AI.SCORE(PROMPT('refund {0}', d.body)) FROM documents d",
    "SELECT d.id FROM documents d "
    "WHERE AI.SCORE(PROMPT('refund {0}', d.body))",
    "SELECT d.id FROM documents d "
    "WHERE AI.SCORE(PROMPT('refund {0}', d.body)) = 1",
])
def test_score_rejects_unnamed_or_uncompared_calls(catalog, sql):
    with pytest.raises(CompileError, match="AI.SCORE"):
        compile_sql(sql, catalog, _tokens)


def test_yes_score_normalization_and_comparisons():
    assert normalized_yes_score(0.0) == 0.5
    assert normalized_yes_score(1000.0) == 1.0
    assert normalized_yes_score(-1000.0) == 0.0
    assert compare_score(0.7, ">=", 0.7)
    assert compare_score(0.7, "<=", 0.7)
    assert compare_score(0.7, ">", 0.6)
    assert compare_score(0.7, "<", 0.8)


def test_vllm_adapter_subtracts_cached_tokens(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm",
        SimpleNamespace(PoolingParams=lambda **_kwargs: object()),
    )
    outputs = [
        SimpleNamespace(
            outputs=SimpleNamespace(score=0.0),
            prompt_token_ids=range(10),
            num_cached_tokens=4,
        ),
        SimpleNamespace(
            outputs=SimpleNamespace(score=1.0),
            prompt_token_ids=range(8),
            num_cached_tokens=3,
        ),
    ]
    llm = SimpleNamespace(score=lambda *_args, **_kwargs: outputs)
    batch = Qwen3VllmReranker(llm).score(["a", "b"], ["c", "d"])
    assert batch.fresh_tokens == 11
    assert batch.cached_tokens == 7


def test_projection_only_plan_does_not_filter(catalog):
    session = _session(catalog, gpus=2)
    query = session.sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d"
    )
    physical = query.plan()
    scores = [
        node for node in physical.nodes if isinstance(node, AiScore)
    ]
    assert len(scores) == 1
    assert not any(
        isinstance(node, ScoreFilter) for node in physical.nodes
    )
    assert scores[0].spec.name == "score"
    assert scores[0].spec.expected_inputs == 2
    assert scores[0].spec.estimated_seconds > 0
    assert physical.workers == 2
    assert physical.settings["data_parallel_copies"] == 2
    session.close()


def test_projection_and_comparison_reuse_one_model_node(catalog):
    session = _session(catalog)
    query = session.sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d "
        "WHERE AI.SCORE("
        "PROMPT('Requests a refund: {0}', d.body)"
        ") >= 0.75"
    )
    physical = query.plan()
    assert sum(isinstance(node, AiScore) for node in physical.nodes) == 1
    score_filter = next(
        node for node in physical.nodes if isinstance(node, ScoreFilter)
    )
    assert score_filter.score_name == "score"
    assert score_filter.comparison == ">="
    assert score_filter.threshold == 0.75
    session.close()


def test_pair_projection_plans_one_cross_product_score(catalog):
    session = _session(catalog, model="qwen3-reranker-4b-bf16")
    query = session.sql(
        "SELECT q.id, d.id, "
        "AI.SCORE(PROMPT("
        "'Is {1} relevant to {0}', q.text, d.body"
        ")) AS score "
        "FROM queries q CROSS JOIN documents d"
    )
    physical = query.plan()
    score = next(
        node for node in physical.nodes if isinstance(node, AiScore)
    )
    assert score.spec.aliases == ("q", "d")
    assert score.spec.expected_inputs == 4
    assert score.spec.pair_fraction == 1.0
    assert not any(
        isinstance(node, ScoreFilter) for node in physical.nodes
    )
    session.close()


def test_score_physical_nodes_round_trip(catalog):
    session = _session(catalog)
    query = session.sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d "
        "WHERE AI.SCORE("
        "PROMPT('Requests a refund: {0}', d.body)"
        ") >= 0.7"
    )
    graph = query.plan().graph
    decoded = decode_graph(
        encode_graph(graph, session.registry.codecs),
        session.registry.codecs,
    )
    assert decoded == graph
    session.close()


class _FakeReranker:
    def __init__(self, scores):
        self.scores = tuple(scores)

    def score(self, queries, documents):
        assert len(queries) == len(documents) == len(self.scores)
        return RerankerBatch(self.scores, fresh_tokens=12, cached_tokens=3)


def test_score_execution_then_filter_retains_float64_column(catalog):
    session = _session(catalog)
    query = session.sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d "
        "WHERE AI.SCORE("
        "PROMPT('Requests a refund: {0}', d.body)"
        ") >= 0.7"
    )
    request = query._prepare_physical()
    graph = compute_subgraph(query.plan().graph)
    model = RerankerModelExecution.__new__(RerankerModelExecution)
    model.reranker = _FakeReranker((0.9, 0.1))
    model.columns = request.column_tables()
    run = GenericRunner().run(
        graph,
        ExecutionContext(
            runtimes=session.registry.runtimes,
            model_execution=model,
            sources={
                "d": range(2),
                **request.relations,
            },
        ),
    )
    score_node = next(
        node for node in graph.nodes if isinstance(node, AiScore)
    )
    filter_node = next(
        node for node in graph.nodes if isinstance(node, ScoreFilter)
    )
    score_table = run.nodes[score_node.node_id].outputs["scores"]
    filtered = run.nodes[filter_node.node_id].outputs["scores"]
    answers = run.nodes[filter_node.node_id].outputs["filter_answers:d"]
    assert score_table.schema.field("score").type == pa.float64()
    assert filtered.column("score").to_pylist() == [0.9]
    assert answers.column("answer").to_pylist() == [True, False]
    assert run.metrics.evaluated_documents == 2
    session.close()


def test_score_query_finishes_with_projected_score(catalog):
    session = _session(catalog)
    query = session.sql(
        "SELECT d.id, "
        "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
        "FROM documents d"
    )

    def execute(request):
        graph = compute_subgraph(query.plan().graph)
        model = RerankerModelExecution.__new__(RerankerModelExecution)
        model.reranker = _FakeReranker((0.9, 0.1))
        model.columns = request.column_tables()
        run = GenericRunner().run(
            graph,
            ExecutionContext(
                runtimes=session.registry.runtimes,
                model_execution=model,
                sources={"d": range(2), **request.relations},
            ),
        )
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {
                "backend": "quail",
                "wall_s": 0.1,
                "fresh_tokens": 12,
                "cached_tokens": 3,
                "node_metrics": scalar_node_metrics(run.nodes),
            },
        )

    result = execute_query(query, physical_executor=execute)
    assert result.collect().to_pylist() == [
        {"d.id": 1, "score": 0.9},
        {"d.id": 2, "score": 0.1},
    ]
    assert result.schema.field("score").type == pa.float64()
    assert result.report["estimated_seconds"] > 0
    session.close()


def test_score_filter_runtime_handles_pair_answers():
    node = ScoreFilter(
        node_id="filter",
        score_name="score",
        aliases=("q", "d"),
        comparison=">",
        threshold=0.5,
        written_pos=0,
    )
    table = pa.table({
        "q": pa.array([0, 0, 1, 1], pa.int32()),
        "d": pa.array([0, 1, 0, 1], pa.int32()),
        "score": pa.array([0.9, 0.1, 0.2, 0.8], pa.float64()),
    })
    result = ScoreFilterRuntime().execute(
        node, {"input:0": table}, ExecutionContext(runtimes={})
    )
    answers = result.outputs["join_answers:0"]
    assert answers.column("answer").to_pylist() == [
        True, False, False, True
    ]
    assert result.outputs["scores"].num_rows == 2
