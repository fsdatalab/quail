"""AI.SCORE parsing, planning, execution, and score semantics."""


import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens
from helpers.oracle import Oracle, finish

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.frontend.sql import compile_sql
from quail.logical import ColumnRef, RegularPredicate, SemanticJoin
from quail.physical import (
    AiClassify,
    AiScore,
    Filter,
    decode_graph,
    encode_graph,
)
from quail.planner.plan import EngineConfig

PROJECTION_SQL = (
    "SELECT d.id, "
    "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
    "FROM documents d"
)


QUERY_TEXTS = [
    ("Does {0} ask for a refund?", "Does this document ask for a refund?"),
    ("Is {0} a refund request? Does {0} ask for money?",
     "Is this document a refund request? Does this document ask for money?"),
    ("Refund request?\n\n{0}", "Refund request?"),
    ("{0}\n\nRefund request?", "Refund request?"),
]


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


def _session(
    catalog, *, model="qwen3-reranker-0.6b-bf16", gpus=1, tokenizer=_tokens
):
    session = quail.Session(
        EngineConfig(model=model, device="h100-sxm", gpus=gpus),
        tokenizer=tokenizer,
    )
    for name in ("documents", "queries"):
        session.register(name, catalog.get(name))
    return session


def test_score_plan_shape_query_text_and_round_trip(catalog):
    session = _session(catalog)
    query = session.sql(
        f"{PROJECTION_SQL} "
        "WHERE AI.SCORE("
        "PROMPT('Requests a refund: {0}', d.body)"
        ") >= 0.7"
    )
    physical = query.plan()
    assert sum(isinstance(node, AiScore) for node in physical.nodes) == 1
    score_filter = next(
        node for node in physical.nodes if isinstance(node, Filter)
    )
    assert score_filter.predicate == RegularPredicate(
        ColumnRef("d", "", "score"), ">=", 0.7)
    graph = physical.graph
    decoded = decode_graph(
        encode_graph(graph, session.registry.codecs),
        session.registry.codecs,
    )
    assert decoded == graph

    for template, query_text in QUERY_TEXTS:
        query = session.sql(
            f"SELECT d.id, AI.SCORE(PROMPT('{template}', d.body)) AS s "
            "FROM documents d"
        )
        score = next(node for node in query.plan().nodes if isinstance(node, AiScore))
        assert score.spec.query_template == query_text, template
    session.close()

    plan = compile_sql(
        "SELECT q.id, d.id FROM queries q JOIN documents d ON "
        "0.8 > AI.SCORE(PROMPT('Is {1} relevant to {0}', q.text, d.body)) "
        "WHERE 0.5 <= AI.SCORE(PROMPT('Refund? {0}', d.body))",
        catalog,
    )
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    assert (join.predicate.comparison, join.predicate.value) == ("<", 0.8)
    assert join.predicate.expression.aliases() == ("q", "d")
    (predicate,) = plan.operators().filters["d"]
    expression = predicate.expression
    assert (expression.comparison, expression.value) == (">=", 0.5)


def test_scores_and_labels_keep_their_types_and_prior_columns(catalog):
    def mixed():
        return Oracle(labels=lambda spec, ids: spec.labels[ids[0] % 2])

    session = _session(catalog, model="qwen3-4b-fp8", tokenizer=letter_tokens)
    score = "AI.SCORE(PROMPT('Refund? {0}', d.body))"
    label = "AI.CLASSIFY(PROMPT('Topic {0}', d.body), ARRAY['refund','praise'])"
    for columns in (f"{score} AS s, {label} AS topic",
                    f"{label} AS topic, {score} AS s"):
        for predicate, expected in (
                ("", [(1, "refund"), (2, "praise")]),
                (f"WHERE {score} >= 0.15", [(2, "praise")]),
                (f"WHERE {label} IN ('praise') AND {score} >= 0.15",
                 [(2, "praise")])):
            query = session.sql(
                f"SELECT d.id, {columns} FROM documents d {predicate}")
            physical = query.plan()
            classifications = [node for node in physical.nodes
                               if isinstance(node, AiClassify)]
            assert len(classifications) == 1
            assert classifications[0].spec.labels == ("refund", "praise")
            assert sum(type(node) is AiScore for node in physical.nodes) == 1
            if "IN" in predicate:
                assert any(isinstance(node, Filter)
                           and node.predicate.value == ("praise",)
                           for node in physical.nodes)
            decoded = decode_graph(
                encode_graph(physical.graph, session.registry.codecs),
                session.registry.codecs)
            assert decoded == physical.graph
            result = finish(query, mixed())
            rows = result.collect()
            assert list(zip(rows.column("d.id").to_pylist(),
                            rows.column("topic").to_pylist())) == expected
            assert rows.column("s").to_pylist() == pytest.approx(
                [document / 10 for document, _ in expected])
            assert rows.schema.field("topic").type == pa.string()
            assert rows.schema.field("s").type == pa.float64()
    # A pair score retains a one-document classification from its input.
    query = session.sql(
        f"SELECT q.id, d.id, {label} AS topic, "
        "AI.SCORE(PROMPT('Relevance {0} {1}', q.text, d.body)) AS relevance "
        "FROM queries q CROSS JOIN documents d")
    assert finish(query, mixed()).collect().column(
        "topic").to_pylist() == ["refund", "praise", "refund", "praise"]
    session.close()
