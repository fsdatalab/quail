"""AI.EXTRACT: the operator surface, its prompt, and its plan.

The transcript tests' expected text is in transcripts/: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from helpers import baselines as request_tests
from helpers.oracle import Oracle, finish, question_text
from helpers.planning import column_tests, operators_of
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

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


TEXTS = [LEASE, "No dates here.", "Term: five years\nmore text",
         "the absent one"]


def test_a_request_backend_decodes_and_locates_each_fields_answer_per_document():
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

    # pipelined submission runs the fields of a document together
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


def test_stock_vllm_extracts_every_row_and_datafusion_keeps_the_top_k(monkeypatch):
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
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease "
        "FROM contracts c ORDER BY c.id DESC LIMIT 2")
    # the region extracts every row; DataFusion sorts them and keeps two
    rows = request_tests._run_stock_vllm(session, query).collect()
    assert rows.column_names == ["c.id", "lease"]
    leases = dict(zip(rows.column("c.id").to_pylist(),
                      rows.column("lease").to_pylist(), strict=True))
    assert len(leases) == 4
    assert (leases["d"], leases["c"]) == ({"ends": None}, {"ends": "five years"})
    assert "SortExec: TopK(fetch=2)" in query.explain()
    session.close()


CONTRACTS = pa.table({
    "id": ["c1", "c2", "c3", "c4"],
    "text": [LEASE, "No dates here.", "Deed ends on March 3, 2027.",
             "A deed with no date."]})
LEASE_ENDS = f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}})"
FIELDS = f"c.text, {{'ends': '{QUESTION}', 'rent': 'How much?'}}"
SELECT_LEASE = f"SELECT c.id, AI.EXTRACT({FIELDS}) AS lease FROM contracts c WHERE "
ENDS = f"AI.EXTRACT({FIELDS})['ends']"
START = f"AI.EXTRACT_SPAN({FIELDS})['ends']['start']"


@pytest.fixture()
def session():
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER)
    value.register("contracts",
                   DocumentProvider.from_table(CONTRACTS, id_col="id"))
    yield value
    value.close()


def _answer(kind, prompt, rows):
    """The model's answers.

    The lease ends on March 3, 2027 in c1 and c3, whose rent is $2,000 in
    c1; asked when it ends, c2 says June 1, 2030, which its text does
    not hold, so the executor returns no answer. c1 and c2 are leases;
    c1 gives "Rent" as the reason. The score of c<i> is i / 8, which
    a float32, as the executor reads a score, holds exactly.
    """
    (row,) = rows
    if kind == "label":
        return "lease" if row in ("c1", "c2") else "deed"
    if kind == "score":
        return int(row[1:]) / 8
    if prompt in (QUESTION, "When?"):
        return "March 3, 2027" if row in ("c1", "c3") else None
    if prompt in ("How much?", "What is the rent?"):
        return "$2,000" if row == "c1" else None
    if prompt == "When does it end?":
        return {"c1": "March 3, 2027", "c2": "June 1, 2030"}.get(row)
    if prompt == "Why?":
        return "Rent" if row == "c1" else None
    raise AssertionError(prompt)


def _transcript(*queries, columns=(), ordered=()) -> str:
    """The transcripts of `queries`.

    Those in `columns` show column types, and those in `ordered` their
    rows in order.
    """
    with Transcripts({"contracts": CONTRACTS}, _answer) as runs:
        return "\n".join(runs.query(sql, columns=sql in columns,
                                     ordered=sql in ordered)
                         for sql in queries)


def test_an_extraction_returns_each_fields_answer_and_span():
    fields = f"c.text, {{'ends': '{QUESTION}', 'rent': 'What is the rent?'}}"
    both = (f"SELECT c.id, AI.EXTRACT({fields}) AS lease, "
            f"AI.EXTRACT_SPAN({fields}) AS lease_span FROM contracts c")
    top = f"{both} ORDER BY c.id DESC LIMIT 1"
    score = "AI.SCORE(PROMPT('Is this a lease? {0}', c.text))"
    # a score beside the extraction, and a test on it, keep both columns
    scored = (f"SELECT c.id, {score} AS score, {LEASE_ENDS} AS lease, "
              f"AI.EXTRACT_SPAN(c.text, {{'ends': '{QUESTION}'}}) AS lease_span "
              f"FROM contracts c WHERE {score} > 0.25")
    assert_transcript(_transcript(both, top, scored, columns=(both, top, scored),
                                  ordered=(top,)))


def test_an_extraction_of_no_rows_returns_no_rows(session):
    query = session.sql(f"SELECT c.id, {LEASE_ENDS} AS lease FROM contracts c "
                        "WHERE c.id = 'none'")
    assert finish(query, Oracle()).collect().num_rows == 0


def test_a_span_column_runs_with_or_without_a_matching_extraction():
    assert_transcript(_transcript(*(
        f"SELECT c.id AS contract_id, {other}"
        "AI.EXTRACT_SPAN(c.text, {'ends': 'Why?'}) AS lease_span FROM contracts c"
        for other in ("", "AI.EXTRACT(c.text, {'ends': 'When?'}) AS lease, "))))


def test_tests_of_answers_keep_the_documents_whose_answers_pass():
    lease, span = f"AI.EXTRACT({FIELDS})", f"AI.EXTRACT_SPAN({FIELDS})"
    tests = " AND ".join(f"HINT({test}, {{'selectivity': 0.5}})" for test in (
        f"{lease}['ends'] IS NOT NULL", f"{lease}['ends'] ILIKE '%march%'",
        f"{span}['ends']['start'] < 20", f"LENGTH({lease}['rent']) < 20"))
    assert_transcript(_transcript(
        # the second field is asked only of the documents the tests of the
        # first keep
        SELECT_LEASE + tests,
        # a negated text test drops the null answer as SQL does
        f"SELECT c.id, {LEASE_ENDS} AS lease FROM contracts c "
        f"WHERE NOT ({LEASE_ENDS}['ends'] LIKE 'March%')",
        # SPLIT_PART is DataFusion's; Arrow compute has no such function
        "SELECT c.id FROM contracts c WHERE SPLIT_PART(AI.EXTRACT(c.text, "
        "{'ends': 'When does it end?'})['ends'], ' ', 1) = 'March'"))


def test_not_and_or_of_answers_labels_and_columns_keep_the_rows_sql_would():
    kind = "AI.CLASSIFY(c.text, ['lease', 'deed'])"
    rent = f"AI.EXTRACT({FIELDS})['rent']"
    assert_transcript(_transcript(
        # one extraction's answer and span
        SELECT_LEASE + f"NOT ({ENDS} IS NULL OR {START} < 0)",
        SELECT_LEASE + f"{ENDS} IS NULL OR {START} >= 0",
        # an answer and a source column
        SELECT_LEASE + f"NOT ({ENDS} IS NULL AND c.id = 'c2')",
        # two extractions
        SELECT_LEASE + f"{ENDS} = 'x' OR {rent} IS NOT NULL",
        # a label and an extracted answer
        f"SELECT c.id AS contract_id, {kind} AS kind, {LEASE_ENDS} AS lease "
        f"FROM contracts c WHERE {kind} = 'lease' "
        f"OR {LEASE_ENDS}['ends'] IS NULL"))


def test_tests_of_answers_follow_the_extraction_they_read(session):
    lease, span = f"AI.EXTRACT({FIELDS})", f"AI.EXTRACT_SPAN({FIELDS})"
    tests = " AND ".join(f"HINT({test}, {{'selectivity': 0.5}})" for test in (
        f"{lease}['ends'] IS NOT NULL", f"{lease}['ends'] ILIKE '%march%'",
        f"{span}['ends']['start'] < 20", f"LENGTH({lease}['rent']) < 20"))
    query = session.sql(SELECT_LEASE + tests)
    found = operators_of(query)
    assert [column.name for column in found.extracts] == ["lease.ends", "lease.rent"]
    tests = column_tests(found)
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
    assert extracts[1].spec.expected_rows == pytest.approx(
        CONTRACTS.num_rows * 0.5 ** 3)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    pipelines = plan.graph.pipelines()
    assert [member.node_id for member in pipelines[extracts[0].node_id]] == [
        extracts[0].node_id, filters[0].node_id, filters[1].node_id,
        filters[2].node_id, extracts[1].node_id, filters[3].node_id]
    # the second extraction reads the KV the first left resident
    residents = plan.graph.resident_inputs()
    assert extracts[1].node_id in residents and extracts[0].node_id not in residents
    assert "Filter: c.lease.ends ilike '%march%'" in query.explain()
    # a test of two extractions reads both
    (test,) = [node for node in session.sql(
        SELECT_LEASE + f"{ENDS} = 'x' OR AI.EXTRACT({FIELDS})['rent'] IS NOT NULL"
    ).plan().nodes if isinstance(node, Filter)]
    assert [port.source.node_id for port in test.inputs] == [
        "ai-extract:1", "ai-extract:0"]
