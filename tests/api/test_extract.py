"""AI.EXTRACT: the operator surface, its prompt, and its plan."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers import baselines as request_tests
from helpers.oracle import Oracle, finish, oracle_executor, question_text
from helpers.planning import column_tests, operators_of
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.physical import (
    AiExtract,
    Filter,
    RequestExecution,
    RequestExtractSpec,
)
from quail.physical.plan import Refusal

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
        tokenizer=TOKENIZER)
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
        f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease, "
        f"AI.EXTRACT_SPAN(c.text, {{'ends': '{QUESTION}'}}) AS lease_span "
        "FROM contracts c "
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


class _PhraseClient(request_tests._Client):
    """Writes each document's answer from the text its prompt tokens spell."""

    def __init__(self):
        self.params = []

    def decode_params(self, max_tokens, stop=None):
        self.params.append((max_tokens, stop))
        return ("decode", max_tokens)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert sampling_params == ("decode", self.params[-1][0]) or all(
            params[0] == "decode" for params in sampling_params)
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            text = question_text(ids)
            if "March" in text:
                answer, tokens = "March 3, 2027", [1, 2, 3, 4]
            elif "five" in text:
                answer, tokens = "FIVE   years", [1, 2]
            elif "absent" in text:
                answer, tokens = "a phrase the document lacks", [1, 2, 3]
            else:
                # an empty phrase: the model ends its turn at once
                answer, tokens = "", [1]
            outputs.append(SimpleNamespace(
                prompt_token_ids=ids, num_cached_tokens=0,
                outputs=[SimpleNamespace(text=answer, token_ids=tokens)]))
        return outputs


class _TextPhraseClient(_PhraseClient):
    """Takes text prompts, as vLLM does, and answers from their words."""

    accepts_text = True

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts = list(prompts)
        return super().generate(
            [{"prompt_token_ids": request_tests._tokens(prompt)}
             for prompt in prompts], sampling_params, use_tqdm)


TEXTS = [LEASE, "No dates here.", "Term: five years\nmore text",
         "the absent one"]


def test_request_backend_decodes_and_locates_one_answer_per_document():
    limit = 7
    documents = {"c": [request_tests._tokens(text) for text in TEXTS]}
    specs = (
        RequestExtractSpec(max_answer_tokens=limit,
                           alias="c", output="ends", question=QUESTION,
                           tail_token_ids=(90,)),
    )
    node = RequestExecution(
        node_id="request-model", backend_name="stock_vllm", aliases=("c",),
        preamble_token_ids=(3,), extracts=specs)
    client = _PhraseClient()
    execution = request_tests._execution(
        documents, client=client, document_texts={"c": TEXTS})
    result = execution.execute(node, {"input:0": [0, 1, 2, 3]})
    assert client.params == [(limit, None)]
    answers = result.outputs["extract_answers:ends"].to_pydict()
    assert answers == {
        "c": [0, 1, 2, 3],
        "ends": ["March 3, 2027", None, "five years", None],
        "ends_span": [{"start": 14, "end": 27}, None, {"start": 6, "end": 16},
                      None]}
    # an extraction keeps every document
    assert result.outputs["rows:c"].ids["c"].to_pylist() == [0, 1, 2, 3]
    step, = result.metrics.extension["steps"]
    assert (step["kind"], step["output"], step["num_in"], step["num_out"]) == (
        "extract", "ends", 4, 2)
    assert (step["requests"], step["unanswered"], step["unaligned"]) == (4, 1, 1)
    assert step["generated_tokens"] == 4 + 1 + 2 + 3
    # the preamble, document, and tail of each prompt, and each
    # answer's tokens but the last, fed back
    assert step["fresh_tokens"] == sum(
        2 + len(ids) for ids in documents["c"]) + (3 + 0 + 1 + 2)
    prompt_tokens = sum(2 + len(ids) for ids in documents["c"])
    assert step["prompt_tokens"] == prompt_tokens
    assert result.metrics.extension["prompt_tokens"] == prompt_tokens
    assert result.metrics.evaluated_documents == 4
    # an engine that takes text gets the preamble, the document, and
    # the tail's text, tokenized by the engine itself
    texted = _TextPhraseClient()
    node = node.replace(preamble_text="DOCUMENT:\n",
                   extracts=(specs[0].replace(tail_text="\nQ?\nPhrase:"),))
    result = request_tests._execution(
        documents, client=texted, document_texts={"c": TEXTS}).execute(node, {})
    assert texted.prompts == ["DOCUMENT:\n" + text + "\nQ?\nPhrase:"
                              for text in TEXTS]
    assert result.outputs["extract_answers:ends"].to_pydict()["ends"] == [
        "March 3, 2027", None, "five years", None]


class _FieldsClient(_PhraseClient):
    """Runs the fields of each document together, as the vLLM client does."""

    def __init__(self):
        super().__init__()
        self.runs = []

    def run_extract_fields(self, sampling_params, body_ids, tail_ids, *,
                           answer_tokens, body_texts=None, tail_texts=None):
        self.runs.append((tail_ids, answer_tokens))
        return {"wall": 1.5, "outputs": [
            self.generate([{"prompt_token_ids": body + tail}
                           for body in body_ids], sampling_params)
            for tail in tail_ids]}


def test_request_backend_runs_the_fields_of_a_document_together():
    documents = {"c": [request_tests._tokens(text) for text in TEXTS]}
    specs = tuple(
        RequestExtractSpec(alias="c", output=output, question=QUESTION,
                           tail_token_ids=(tail,), max_answer_tokens=limit)
        for output, tail, limit in (("ends", 90, 7), ("term", 91, 9)))
    node = RequestExecution(
        node_id="request-model", backend_name="pipelined_vllm", aliases=("c",),
        preamble_token_ids=(3,), extracts=specs)
    client = _FieldsClient()
    result = request_tests._execution(
        documents, client=client, document_texts={"c": TEXTS},
        answer_submission="pipelined").execute(node, {"input:0": [0, 1, 2, 3]})
    # one run decodes both fields, up to the longer answer limit
    assert client.runs == [([[90], [91]], 9)]
    expected = ["March 3, 2027", None, "five years", None]
    for output in ("ends", "term"):
        answers = result.outputs[f"extract_answers:{output}"].to_pydict()
        assert answers[output] == expected
    steps = result.metrics.extension["steps"]
    assert [(step["output"], step["requests"], step["wall_s"])
            for step in steps] == [("ends", 4, 1.5), ("term", 4, 1.5)]


def test_stock_vllm_plans_and_runs_an_extraction(monkeypatch):
    session = request_tests._session(
        "stock_vllm", contracts={"id": ["a", "b", "c", "d"], "text": TEXTS})
    fields = f"c.text, {{'ends': '{QUESTION}', 'term': 'How long is the term?'}}"
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT({fields}) AS lease, "
        f"AI.EXTRACT_SPAN({fields}) AS lease_span FROM contracts c")
    boot = {"client": _PhraseClient(), "sampling_params": object(),
            "capacity": request_tests.CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids, *, device, resources: (
            boot, {"kind": "cold", "boot_s": 0.1}))
    result = request_tests._run_stock_vllm(session, query)
    rows = result.collect()
    assert rows.column_names == ["c.id", "lease", "lease_span"]
    assert rows.column("c.id").to_pylist() == ["a", "b", "c", "d"]
    # the fake client answers from the document, whatever the question
    answers = ["March 3, 2027", None, "five years", None]
    assert rows.column("lease").to_pylist() == [
        {"ends": answer, "term": answer} for answer in answers]
    spans = [{"start": 14, "end": 27}, None, {"start": 6, "end": 16}, None]
    assert rows.column("lease_span").to_pylist() == [
        {"ends": span, "term": span} for span in spans]
    answers = result.answer_tables["extracts"]["lease.ends"]
    assert answers.column_names == ["c", "lease.ends", "lease.ends_span"]
    assert answers.num_rows == 4
    steps = result.report["backend_metrics"]["steps"]
    assert [(step["kind"], step["requests"]) for step in steps] == [
        ("extract", 4), ("extract", 4)]
    session.close()


def test_stock_vllm_extracts_every_row_a_top_k_reads(monkeypatch):
    session = request_tests._session(
        "stock_vllm", contracts={"id": ["a", "b", "c", "d"], "text": TEXTS})
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease "
        "FROM contracts c ORDER BY c.id DESC LIMIT 2")
    boot = {"client": _PhraseClient(), "sampling_params": object(),
            "capacity": request_tests.CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids, *, device, resources: (
            boot, {"kind": "cold", "boot_s": 0.1}))
    # the region extracts every row; DataFusion sorts them and keeps two
    rows = request_tests._run_stock_vllm(session, query).collect()
    assert rows.column_names == ["c.id", "lease"]
    leases = dict(zip(rows.column("c.id").to_pylist(),
                      rows.column("lease").to_pylist(), strict=True))
    assert len(leases) == 4
    assert (leases["d"], leases["c"]) == ({"ends": None}, {"ends": "five years"})
    assert "SortExec: TopK(fetch=2)" in query.explain()
    session.close()


STRUCT_FIELDS = ("c.text, {'ends': '" + QUESTION
                 + "', 'rent': 'What is the rent?'}")
STRUCT_SQL = (f"SELECT c.id, AI.EXTRACT({STRUCT_FIELDS}) AS lease, "
              f"AI.EXTRACT_SPAN({STRUCT_FIELDS}) AS lease_span FROM contracts c")


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

    fields = f"c.text, {{'ends': '{QUESTION}', 'rent': 'How much?'}}"
    lease, span = f"AI.EXTRACT({fields})", f"AI.EXTRACT_SPAN({fields})"
    tests = " AND ".join(f"HINT({test}, {{'selectivity': 0.5}})" for test in (
        f"{lease}['ends'] IS NOT NULL", f"{lease}['ends'] ILIKE '%march%'",
        f"{span}['ends']['start'] < 20", f"LENGTH({lease}['rent']) < 20"))
    query = session.sql(
        f"SELECT c.id, {lease} AS lease FROM contracts c WHERE {tests}")
    # each test follows its field's extraction
    found = operators_of(query)
    assert [column.name for column in found.extracts] == ["lease.ends", "lease.rent"]
    tests = column_tests(found)
    assert [str(test.condition) for test in tests] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "character_length(c.lease.rent) < 20"]
    assert [test.selectivity for test in tests] == [0.5, 0.5, 0.5, 0.5]
    # the first three tests read the first field, the last the second
    assert [test.name for test in tests] == ["lease.ends"] * 3 + ["lease.rent"]
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    filters = [n for n in plan.nodes if isinstance(n, Filter)]
    assert [str(node.predicate) for node in filters] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "character_length(c.lease.rent) < 20"]
    assert filters[0].inputs[0].source.node_id == extracts[0].node_id
    assert extracts[1].inputs[0].source.node_id == filters[2].node_id
    # the three tests on the first field thin the second extraction,
    # which runs in the same pipeline over the resident KV
    assert extracts[1].spec.expected_rows == pytest.approx(2 * 0.5 ** 3)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    pipelines = plan.graph.pipelines()
    assert [member.node_id for member in pipelines[extracts[0].node_id]] == [
        extracts[0].node_id, filters[0].node_id, filters[1].node_id,
        filters[2].node_id, extracts[1].node_id, filters[3].node_id]
    # the second extraction reads the KV the first left resident
    residents = plan.graph.resident_inputs()
    assert extracts[1].node_id in residents and extracts[0].node_id not in residents
    assert "Filter: c.lease.ends ilike '%march%'" in query.explain()

    # one document answers; its answer passes the tests
    result = execute_query(query, physical_executor=_answers_executor(
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
    lease = f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}})"
    negated = session.sql(
        f"SELECT c.id, {lease} AS lease "
        f"FROM contracts c WHERE NOT ({lease}['ends'] LIKE 'March%')")
    (test,) = column_tests(operators_of(negated))
    assert str(test.condition) == "NOT (c.lease.ends like 'March%')"
    result = execute_query(negated, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: None}))
    assert result.collect().num_rows == 0


def test_a_datafusion_function_arrow_compute_lacks_tests_an_answer(session):
    from quail.api.execute import execute_query

    # SPLIT_PART is DataFusion's; Arrow compute has no such function
    lease = f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}})"
    query = session.sql(
        f"SELECT c.id, {lease} AS lease "
        f"FROM contracts c WHERE SPLIT_PART({lease}['ends'], ' ', 1) = 'March'")
    (test,) = column_tests(operators_of(query))
    assert str(test.condition) == "split_part(c.lease.ends, ' ', 1) = 'March'"
    result = execute_query(query, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: "June 1, 2030"}))
    assert result.collect().column("c.id").to_pylist() == [1]

def test_not_and_or_of_answers_run_after_the_extractions_they_read(session):
    from quail.api.execute import execute_query

    fields = f"c.text, {{'ends': '{QUESTION}', 'rent': 'How much?'}}"
    lease, span = f"AI.EXTRACT({fields})", f"AI.EXTRACT_SPAN({fields})"
    head = f"SELECT c.id, {lease} AS lease FROM contracts c WHERE "
    ends, rent = f"{lease}['ends']", f"{lease}['rent']"
    start = f"{span}['ends']['start']"

    def placed(query):
        found = operators_of(query)
        return [("extraction", str(test.condition)) for test in column_tests(found)]

    def kept(query):
        result = execute_query(query, physical_executor=_answers_executor(
            {0: "March 3, 2027", 1: None}))
        return result.collect().column("c.id").to_pylist()

    # one extraction's answer and span: the Filter follows the extraction
    one = session.sql(head + f"NOT ({ends} IS NULL OR {start} < 0)")
    assert placed(one) == [
        ("extraction", "c.lease.ends IS NOT NULL"),
        ("extraction", "NOT (c.lease.ends_span.start < 0)")]
    one = session.sql(head + f"{ends} IS NULL OR {start} >= 0")
    assert placed(one) == [
        ("extraction", "c.lease.ends IS NULL OR c.lease.ends_span.start >= 0")]
    assert kept(one) == [1, 2]
    # an answer and a source column, or two extractions: after the last
    # extraction read, each other column read by document ID
    mixed = session.sql(head + f"NOT ({ends} IS NULL AND c.id = 2)")
    # DataFusion computes the source column's test, c.id = 2
    assert placed(mixed) == [
        ("extraction", "c.lease.ends IS NOT NULL OR NOT (c.__quail_expr_0)")]
    assert kept(mixed) == [1]
    fields = session.sql(head + f"{ends} = 'x' OR {rent} IS NOT NULL")
    assert placed(fields) == [
        ("extraction", "c.lease.ends = 'x' OR c.lease.rent IS NOT NULL")]
    (test,) = [node for node in fields.plan().nodes if isinstance(node, Filter)]
    assert [port.source.node_id for port in test.inputs] == [
        "ai-extract:1", "ai-extract:0"]
    assert kept(fields) == [1]


@pytest.mark.parametrize("label", ["lease", "deed"])
@pytest.mark.parametrize("answer", [None, "March 3, 2027"])
def test_an_or_combines_a_label_and_an_extracted_answer(session, label, answer):
    query = session.sql(
        "SELECT c.id AS contract_id, "
        "AI.CLASSIFY(c.text, ['lease', 'deed']) AS kind, "
        f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease "
        "FROM contracts c WHERE AI.CLASSIFY(c.text, ['lease', 'deed']) = 'lease' "
        f"OR AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}})['ends'] IS NULL "
        "ORDER BY c.id")
    oracle = Oracle(labels=lambda *_: label, answers=lambda *_: answer)
    rows = finish(query, oracle).collect()
    expected = [
        {"contract_id": identity, "kind": label, "lease": {"ends": answer}}
        for identity in (1, 2)
    ] if label == "lease" or answer is None else []
    assert rows.to_pylist() == expected


def test_an_extracted_tables_token_store_holds_each_tokens_characters(session):
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease "
        "FROM contracts c")
    query.plan()
    (scan,) = query.token_inputs().values()
    assert scan.tokens.has_offsets
    view = scan.tokens[0]
    assert view.character_offsets is not None
    assert len(view.character_offsets) == len(view)


@pytest.mark.parametrize("with_other_extraction", [False, True])
def test_a_span_column_can_run_without_a_matching_extraction(
        session, with_other_extraction):
    other = ("AI.EXTRACT(c.text, {'ends': 'When?'}) AS lease, "
             if with_other_extraction else "")
    query = session.sql(
        f"SELECT c.id AS contract_id, {other}"
        "AI.EXTRACT_SPAN(c.text, {'ends': 'Why?'}) AS lease_span "
        "FROM contracts c ORDER BY c.id")
    questions = set()

    def answer(spec, document):
        questions.add(spec.question)
        text = str(oracle.texts[spec.alias][document])
        if "March" not in text:
            return None
        return "March 3, 2027" if spec.question == "When?" else "Rent"

    oracle = Oracle(answers=answer)
    rows = finish(query, oracle).collect()
    start = LEASE.index("Rent")
    expected = [
        {"contract_id": 1,
         "lease_span": {"ends": {"start": start, "end": start + len("Rent")}}},
        {"contract_id": 2, "lease_span": {"ends": None}},
    ]
    if with_other_extraction:
        expected[0]["lease"] = {"ends": "March 3, 2027"}
        expected[1]["lease"] = {"ends": None}
    assert rows.to_pylist() == expected
    assert questions == ({"When?", "Why?"} if with_other_extraction else {"Why?"})
