"""AI.SCORE parsing, planning, execution, and score semantics.

The transcript test's expected text is in transcripts/: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""


import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.planning import compile_sql, read_operators
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.catalog import Catalog, DocumentProvider
from quail.physical import AiScore, Filter

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
    catalog, *, model="qwen3-reranker-0.6b-bf16", gpus=1, tokenizer=TOKENIZER
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
    assert str(score_filter.predicate) == "d.score >= 0.7"

    for template, query_text in QUERY_TEXTS:
        query = session.sql(
            f"SELECT d.id, AI.SCORE(PROMPT('{template}', d.body)) AS s "
            "FROM documents d"
        )
        score = next(node for node in query.plan().nodes if isinstance(node, AiScore))
        assert score.spec.query_template == query_text, template
    session.close()

    found = read_operators(compile_sql(
        "SELECT q.id, d.id FROM queries q JOIN documents d ON "
        "0.8 > AI.SCORE(PROMPT('Is {1} relevant to {0}', q.text, d.body)) "
        "WHERE 0.5 <= AI.SCORE(PROMPT('Refund? {0}', d.body))",
        catalog,
    ))
    (join,) = found.joins
    # the test reads the score as its one column, swapped to its left
    assert join.predicate.test.text == "{0} < 0.8"
    assert [arg.alias for arg in join.predicate.call["prompt"].args] == ["q", "d"]
    ((alias, (predicate,)),) = found.filters
    assert alias == "d"
    expression = predicate.expression
    assert expression.test.text == "{0} >= 0.5"


def test_scores_and_labels_keep_their_types_and_the_columns_before_them():
    score = "AI.SCORE(PROMPT('Refund? {0}', d.body))"
    label = "AI.CLASSIFY(PROMPT('Topic {0}', d.body), ARRAY['refund','praise'])"
    queries = [f"SELECT d.id, {columns} FROM documents d {predicate}"
               for columns in (f"{score} AS s, {label} AS topic",
                               f"{label} AS topic, {score} AS s")
               for predicate in ("", f"WHERE {score} >= 0.15",
                                 f"WHERE {label} IN ('praise') AND {score} >= 0.15")]
    # a pair score keeps a one-document classification from its input
    queries.append(
        f"SELECT q.id, d.id, {label} AS topic, "
        "AI.SCORE(PROMPT('Relevance {0} {1}', q.text, d.body)) AS relevance "
        "FROM queries q CROSS JOIN documents d")

    def answer(kind, prompt, rows):
        if kind == "label":
            return "refund" if rows == ("d1",) else "praise"
        return sum(int(row[1:]) for row in rows) / 10

    tables = {"documents": pa.table({"id": ["d1", "d2"],
                                     "body": ["refund please", "all good"]}),
              "queries": pa.table({"id": ["q1", "q2"],
                                   "text": ["refund", "shipping"]})}
    with Transcripts(tables, answer) as runs:
        assert_transcript("\n".join(runs.query(sql, columns=True)
                                     for sql in queries))


def test_a_set_attention_score_plans_a_document_longer_than_one_page(tmp_path):
    catalog = Catalog()
    catalog.register("documents", _provider(
        tmp_path / "documents.parquet",
        {"id": [1, 2], "body": [" ".join(["word"] * 2000), "all good"]},
        id_col="id",
    ))
    catalog.register("queries", _provider(
        tmp_path / "queries.parquet", {"id": [1], "text": ["refund"]},
        id_col="id",
    ))
    session = _session(catalog, model="d1-2b-bf16")

    physical = session.sql(PROJECTION_SQL).plan()

    assert sum(isinstance(node, AiScore) for node in physical.nodes) == 1
    session.close()
