"""AI.EXTRACT: the operator surface, its prompt, and its plan."""


import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens
from helpers.oracle import Oracle, finish, oracle_executor

import quail
from quail.catalog import DocumentProvider
from quail.physical import (
    AiExtract,
    Filter,
    decode_graph,
    encode_graph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig, Refusal

QUESTION = "When does the lease end?"
LEASE = "Lease ends on March 3, 2027.\nRent is $2,000."

@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "contracts.parquet"
    pq.write_table(pa.table({
        "id": [1, 2], "text": [LEASE, "No dates here."],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    value.register("contracts",
                   DocumentProvider.from_parquet(str(path), id_col="id"))
    yield value
    value.close()


@pytest.mark.parametrize("filtered", [True])
def test_score_and_extract_return_both_columns(session, monkeypatch, filtered):
    oracle = Oracle(answers=lambda spec, document: (
        "March 3, 2027" if document == 0 else None))
    query = session.sql(
        "SELECT c.id, AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) AS score, "
        f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease FROM contracts c "
        + ("WHERE AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) > 0.15"
           if filtered else ""))
    result = finish(query, oracle)
    rows = result.collect()
    assert rows.column("score").type == pa.float64()
    expected = [
        {"c.id": 1, "score": 0.1, "lease": {"ends": "March 3, 2027"},
         "lease_span": {"ends": {"start": 14, "end": 27}}},
        {"c.id": 2, "score": 0.2, "lease": {"ends": None},
         "lease_span": {"ends": None}},
    ]
    assert rows.to_pylist() == (expected[1:] if filtered else expected)


STRUCT_SQL = ("SELECT c.id, AI.EXTRACT(c.text, {'ends': '" + QUESTION
              + "', 'rent': 'What is the rent?'}) AS lease FROM contracts c")


def _struct_answers():
    answers = {"lease.ends": ["March 3, 2027", None],
               "lease.rent": ["$2,000", None]}
    return Oracle(answers=lambda spec, document: answers[spec.name][document])


@pytest.mark.parametrize("suffix", ["", " ORDER BY c.id DESC LIMIT 1"])
def test_extract_returns_answer_and_span_structs(
        session, monkeypatch, suffix):
    result = finish(session.sql(STRUCT_SQL + suffix), _struct_answers())
    rows = result.collect()
    assert rows.column_names == ["c.id", "lease", "lease_span"]
    span = pa.struct([("start", pa.int32()), ("end", pa.int32())])
    assert rows.schema.field("lease").type == pa.struct(
        [("ends", pa.string()), ("rent", pa.string())])
    assert rows.schema.field("lease_span").type == pa.struct(
        [("ends", span), ("rent", span)])
    expected = [
        {"c.id": 1, "lease": {"ends": "March 3, 2027", "rent": "$2,000"},
         "lease_span": {"ends": {"start": 14, "end": 27},
                        "rent": {"start": 37, "end": 43}}},
        {"c.id": 2, "lease": {"ends": None, "rent": None},
         "lease_span": {"ends": None, "rent": None}},
    ]
    assert rows.to_pylist() == (expected[1:] if "DESC" in suffix else expected)


def _answers_executor(answers):
    """A physical executor whose extractions answer answers[document]."""
    return oracle_executor(Oracle(answers=answers), fresh_tokens=12)


def test_filters_on_answers_keep_the_documents_whose_answer_passes(session):
    from quail.api.execute import execute_query
    from quail.logical import Filter as LogicalFilter

    sql = session.sql
    built = (session.docs("contracts").alias("c")
             .ai_extract("c.text", {"ends": QUESTION, "rent": "How much?"},
                         name="lease")
             .where(quail.col("lease.ends").is_not_null().hint(selectivity=0.5)
                    & quail.col("lease.ends").ilike("%march%").hint(selectivity=0.5))
             .filter((quail.col("lease_span.ends").field("start") < 20)
                     .hint(selectivity=0.5))
             .where((quail.col("lease.rent").length() < 20).hint(selectivity=0.5))
             .select("c.id", "lease"))
    tests = " AND ".join(f"HINT({test}, {{'selectivity': 0.5}})" for test in (
        "lease.ends IS NOT NULL", "lease.ends ILIKE '%march%'",
        "lease_span.ends.start < 20", "LENGTH(lease.rent) < 20"))
    written = sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}', "
        f"'rent': 'How much?'}}) AS lease FROM contracts c WHERE {tests}")
    assert written.logical == built.logical
    # each test follows its field's extraction; the span and length
    # tests use the same Filter node
    kinds = [type(node).__name__ for node in built.logical.walk()]
    assert kinds == ["Scan", "SemanticExtract", "Filter", "Filter", "Filter",
                     "SemanticExtract", "Filter", "Project"]
    tests = [node for node in built.logical.walk()
             if isinstance(node, LogicalFilter)]
    assert [str(node.condition) for node in tests] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "length(c.lease.rent) < 20"]
    assert [node.selectivity for node in tests] == [0.5, 0.5, 0.5, 0.5]
    operators = built.logical.operators()
    assert [(test.name, test.position)
            for test in operators.extract_filters["c"]] == [
        ("lease.ends", 0), ("lease.ends", 1), ("lease.ends", 2),
        ("lease.rent", 3)]
    plan = built.plan()
    assert not isinstance(plan, Refusal)
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    filters = [n for n in plan.nodes if isinstance(n, Filter)]
    assert [str(node.predicate) for node in filters] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "length(c.lease.rent) < 20"]
    assert filters[0].inputs[0].source.node_id == extracts[0].node_id
    assert extracts[1].inputs[0].source.node_id == filters[2].node_id
    # the three tests on the first field thin the second extraction,
    # which runs in the same pipeline over the resident KV
    assert extracts[1].spec.expected_rows == pytest.approx(2 * 0.5 ** 3)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    pipelines = build_pipelines(plan.graph)
    assert [member.node_id for member in pipelines[extracts[0].node_id]] == [
        extracts[0].node_id, filters[0].node_id, filters[1].node_id,
        filters[2].node_id, extracts[1].node_id, filters[3].node_id]
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    assert "Filter: c.lease.ends ilike '%march%'" in built.explain()

    # one document answers; its answer passes the tests
    result = execute_query(built, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: None}))
    rows = result.collect()
    assert rows.column("c.id").to_pylist() == [1]
    assert rows.column("lease").to_pylist() == [
        {"ends": "March 3, 2027", "rent": "March 3, 2027"}]
    answers = result.answer_tables["extracts"]["lease.ends"]
    assert answers.column("lease.ends").to_pylist() == ["March 3, 2027", None]
    # the second question was asked of the survivor only
    assert result.answer_tables["extracts"]["lease.rent"].num_rows == 1
    passed = result.answer_tables["filters"]
    assert passed[("c", 0)].column("answer").to_pylist() == [True, False]
    assert passed[("c", 2)].column("answer").to_pylist() == [True]
    # a negated text test drops the null answer as SQL does
    negated = (session.docs("contracts").alias("c")
               .ai_extract("c.text", {"ends": QUESTION}, name="lease")
               .where(~quail.col("lease.ends").like("March%"))
               .select("c.id", "lease"))
    assert str(negated.logical.root.input.condition) == (
        "NOT (c.lease.ends like 'March%')")
    result = execute_query(negated, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: None}))
    assert result.collect().num_rows == 0


def test_not_and_or_of_answers_run_after_one_extraction_or_on_result_rows(session):
    from quail.api.execute import execute_query
    from quail.logical import Filter as LogicalFilter

    head = (f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}', "
            f"'rent': 'How much?'}}) AS lease FROM contracts c WHERE ")

    def placed(query):
        return [(type(node.input).__name__, str(node.condition))
                for node in query.logical.walk() if isinstance(node, LogicalFilter)]

    def kept(query):
        result = execute_query(query, physical_executor=_answers_executor(
            {0: "March 3, 2027", 1: None}))
        return result.collect().column("c.id").to_pylist()

    built = (session.docs("contracts").alias("c")
             .ai_extract("c.text", {"ends": QUESTION, "rent": "How much?"},
                         name="lease")
             .where((quail.col("lease.ends") == "x")
                    | quail.col("lease.rent").is_not_null())
             .select("c.id", "lease"))
    written = session.sql(head + "lease.ends = 'x' OR lease.rent IS NOT NULL")
    assert built.logical == written.logical
    # one extraction's answer and span: the Filter follows the extraction
    one = session.sql(head + "NOT (lease.ends IS NULL OR lease_span.ends.start < 0)")
    assert placed(one) == [
        ("SemanticExtract", "c.lease.ends IS NOT NULL"),
        ("Filter", "NOT (c.lease.ends_span.start < 0)")]
    one = session.sql(head + "lease.ends IS NULL OR lease_span.ends.start >= 0")
    assert placed(one) == [
        ("SemanticExtract", "c.lease.ends IS NULL OR c.lease.ends_span.start >= 0")]
    assert kept(one) == [1, 2]
    # an answer and a source column, or two extractions: the result rows
    mixed = session.sql(head + "NOT (lease.ends IS NULL AND c.id = 2)")
    assert placed(mixed) == [
        ("Project", "lease.ends IS NOT NULL OR NOT (c.id = 2)")]
    assert kept(mixed) == [1]
    fields = session.sql(head + "lease.ends = 'x' OR lease.rent IS NOT NULL")
    assert placed(fields) == [("Project", "lease.ends = 'x' OR lease.rent IS NOT NULL")]
    assert kept(fields) == [1]
