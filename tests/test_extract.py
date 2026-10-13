"""AI.EXTRACT: the operator surface, its prompt, and its plan."""

from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_request_backends as request_tests
import test_score as score_tests
from fakes import letter_tokens

import quail
from quail.backends.quail.reranker import RerankerModelExecution
from quail.baselines import (
    RequestExecution,
    RequestExtractSpec,
)
from quail.catalog import DocumentProvider
from quail.execution.result import extract_answer_table
from quail.execution.runner import NodeResult
from quail.execution.spans import line_span, locate_span, read_phrase
from quail.logical import (
    CompileError,
    extracted_above_joins,
)
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    ExtractSpec,
    Filter,
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


def _ends(session, trim=True):
    return (session.docs("contracts").alias("c")
            .ai_extract("c.text", QUESTION, name="ends", trim=trim)
            .select("c.id", "ends"))


def test_extract_prefix_sharing_is_planned_from_exact_document_tokens():
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens,
    ) as session:
        shared = "The following agreement is made between the parties. " * 4
        session.register("contracts", DocumentProvider.from_table(pa.table({
            "id": [1, 2, 3],
            "text": [shared + LEASE, shared + "No dates here.", "Unrelated."],
        }), id_col="id"))
        query = _ends(session)
        query.plan()
        query.wait_for_tokens()
        plan = query.plan()
        (node,) = [node for node in plan.nodes if isinstance(node, AiExtract)]
        assert node.spec.share_prefixes


def test_sql_and_builder_extract_the_same_answer_columns(session):
    built = _ends(session)
    for dialect in ('snowflake', 'bq'):
        query = session.sql(
            f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') AS ends "
            f"FROM contracts c", dialect=dialect)
        assert query.logical == built.logical
    loose = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}', {{'trim': false}}) "
        f"AS ends FROM contracts c")
    assert loose.logical == _ends(session, trim=False).logical


def test_planning_reuses_bound_extraction_tokens(session, monkeypatch):
    query = _ends(session)
    tokenizer = session.tokenizer

    def reuse_prompt(text):
        assert QUESTION not in text, "planning must reuse the bound extraction prompt"
        return tokenizer(text)

    monkeypatch.setattr(session, "_tok", reuse_prompt)
    assert not isinstance(query.plan(), Refusal)


def test_extract_follows_the_filters_and_labels_of_its_table(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
             .ai_classify(quail.prompt("{0}", quail.col("c.text")),
                          ["lease", "deed"], name="kind")
             .label_in("kind", ["lease"], selectivity=0.5)
             .ai_extract("c.text", QUESTION, name="ends")
             .select("c.id", "kind", "ends"))
    plan = query.plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert node.inputs[0].source.node_id == test.node_id
    # half the documents pass the label filter
    assert node.spec.expected_inputs == pytest.approx(2 * 0.2 * 0.5)
    assert plan.nodes[-1].columns == ("c.id", "kind", "ends", "ends_span")


def test_extractions_of_a_table_form_one_pipeline_and_are_priced_as_resident(
        session):
    query = (session.docs("contracts").alias("c")
             .ai_extract("c.text", QUESTION, name="ends")
             .ai_extract("c.text", "Who signs?", name="signer")
             .ai_extract("c.text", "Where?", name="place")
             .select("c.id", "ends", "signer", "place"))
    plan = query.plan()
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    pipelines = build_pipelines(plan.graph)
    # the three run as one pipeline: the later ones read the KV the
    # first leaves and are priced without reading the document
    assert tuple(member.node_id for member in pipelines[extracts[0].node_id]) == tuple(
        extract.node_id for extract in extracts)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    assert extracts[2].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    different = replace(extracts[1], spec=replace(
        extracts[1].spec, draft_tokens=2))
    graph = replace(plan.graph, nodes=tuple(
        different if node.node_id == different.node_id else node
        for node in plan.nodes))
    separated = build_pipelines(graph)
    assert tuple(
        member.node_id for member in separated[extracts[0].node_id]) == (
            extracts[0].node_id,)
    assert tuple(
        member.node_id for member in separated[extracts[1].node_id]) == (
            extracts[1].node_id,)
    # a plain extraction shows the document under the query's shared
    # head, as the filter does, so it continues the filter's pipeline
    # and asks its question of the KV the filter wrote
    filtered = (session.docs("contracts").alias("c")
                .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
                .ai_extract("c.text", QUESTION, name="ends")
                .select("c.id", "ends"))
    plan = filtered.plan()
    (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (ai_filter,) = [n for n in plan.nodes if isinstance(n, AiFilter)]
    assert extract.spec.shares_head
    assert list(extract.spec.prompt_token_parts[0]) == plan.settings["pre_ids"]
    pipelines = build_pipelines(plan.graph)
    assert tuple(member.node_id for member in pipelines[ai_filter.node_id]) == (
        ai_filter.node_id, extract.node_id)
    # a classification of the table writes the documents under the
    # same head, so the extraction continues its pipeline too
    classified = (session.docs("contracts").alias("c")
                  .ai_classify(quail.prompt("Which kind is {0}?", quail.col("c.text")),
                               ["lease", "license"], name="kind")
                  .ai_extract("c.text", QUESTION, name="ends")
                  .select("c.id", "kind", "ends"))
    plan = classified.plan()
    (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert (tuple(classify.spec.prompt_token_parts[0])
            == tuple(extract.spec.prompt_token_parts[0]))
    pipelines = build_pipelines(plan.graph)
    assert tuple(member.node_id for member in pipelines[classify.node_id]) == (
        classify.node_id, extract.node_id)


def _candidates(session, query):
    """Return the Quail backend's candidates for a planned query."""
    from quail.planner.physical_optimizer import ModelRegion, PlanningContext

    query.plan()
    context = PlanningContext(
        model=session.model, device=session.device, gpu_count=1,
        document_tokens=query._doc_tokens, backend="quail",
        tokenizer=session.tokenizer)
    return session.registry.backend("quail").plan(
        ModelRegion(query.logical), context)


def test_extract_of_a_joined_table_is_offered_after_the_join(session, tmp_path):
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.table({"id": [1], "ask": ["end dates"]}), path)
    session.register("questions",
                     DocumentProvider.from_parquet(str(path), id_col="id"))
    query = session.sql(
        f"SELECT c.id, q.id, AI.EXTRACT(c.text, '{QUESTION}') AS ends "
        f"FROM contracts c JOIN questions q "
        f"ON AI_FILTER(PROMPT('Does {{0}} answer {{1}}?', c.text, q.ask))")
    assert extracted_above_joins(query.logical.root) == frozenset()
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    written, lifted = _candidates(session, query)
    # the backend also offers the extraction after the join, over the
    # documents the join matched: a fifth of them by default, so the
    # lifted plan is cheaper and wins
    assert written.logical_plan.root == query.logical.root
    assert extracted_above_joins(lifted.logical_plan.root) == {"c"}
    assert lifted.estimated_seconds < written.estimated_seconds
    (join,) = [n for n in written.plan.nodes if isinstance(n, AiJoin)]
    assert ("ai-extract:0", "ids:c") in [
        (port.source.node_id, port.source.port) for port in join.inputs]
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    assert node.spec.expected_inputs == pytest.approx(2 * 0.2)
    (before,) = [n for n in written.plan.nodes if isinstance(n, AiExtract)]
    assert before.spec.expected_inputs == 2
    assert node.spec.estimated_seconds < before.spec.estimated_seconds
    assert plan.nodes[-1].columns == ("c.id", "q.id", "ends", "ends_span")


def test_extract_errors_and_refusals(session):
    sql = session.sql
    for bad, message in (
            (f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') FROM contracts c",
             "AS name"),
            (f"SELECT c.id FROM contracts c WHERE AI.EXTRACT(c.text, "
             f"'{QUESTION}') = 'x'", "not a predicate"),
            (f"SELECT c.id, AI.EXTRACT('{QUESTION}', c.text) AS x "
             f"FROM contracts c", "document column"),
            ("SELECT c.id, AI.EXTRACT(c.text, c.id) AS x FROM contracts c",
             "question as a string"),
            ("SELECT c.id, AI.EXTRACT(c.text) AS x FROM contracts c",
             "a question string"),
            (f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}', {{'trim': 1}}) "
             f"AS x FROM contracts c", "true or false"),
            (f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}', {{'lines': 2}}) "
             f"AS x FROM contracts c", "unknown AI.EXTRACT option"),
            (f"SELECT c.id, UPPER(AI.EXTRACT(c.text, '{QUESTION}')) AS x "
             f"FROM contracts c", "direct expressions"),
            ("SELECT c.id, AI.EXTRACT(c.text, '') AS x FROM contracts c",
             "at least one word")):
        with pytest.raises(CompileError, match=message):
            sql(bad)
    builder = session.docs("contracts").alias("c")
    with pytest.raises(CompileError, match="without a dot"):
        builder.ai_extract("c.text", QUESTION, name="c.ends")
    with pytest.raises(CompileError, match="already used"):
        builder.ai_extract("c.text", QUESTION, name="c")
    with pytest.raises(CompileError, match="already used"):
        (builder.ai_extract("c.text", QUESTION, name="ends")
         .ai_classify(quail.prompt("{0}", quail.col("c.text")), ["a", "b"],
                      name="ends_span"))
    # the speed of light estimate prices an extraction as one question
    # per document, read once at the cue
    estimate = quail.speed_of_light_estimate(_ends(session), lambda *_: True)
    (stage,) = estimate.filter_stages
    assert (stage["operator"], stage["alias"], stage["name"]) == (
        "AI.EXTRACT", "c", "ends")
    assert (stage["evaluated"], stage["passed"]) == (2, 2)
    assert estimate.extraction_evaluations == 2
    assert estimate.seconds > 0 and "extraction" in estimate.assumptions()
    assert estimate.as_dict()["extraction_evaluations"] == 2
    # SGLang returns no decoded text, so it does not run it
    sglang = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                        backend="pipelined_sglang"),
                           tokenizer=letter_tokens)
    sglang.register("contracts", session.catalog.get("contracts"))
    refused = _ends(sglang).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "extract_needs_quail_backend"
    sglang.close()
    # a reranker gives no next-token probabilities over the document,
    # nor does a decision model or a diffusion model with a canvas
    reranker = quail.Session(
        EngineConfig(model="qwen3-reranker-0.6b-bf16", device="h100-sxm"),
        tokenizer=letter_tokens)
    reranker.register("contracts", session.catalog.get("contracts"))
    refused = _ends(reranker).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "reranker_only_scores"
    reranker.close()


def test_span_helpers_read_locate_and_widen_an_answer():
    assert read_phrase('March 3, 2027" and more') == "March 3, 2027"
    assert read_phrase("March 3, 2027") == "March 3, 2027"
    assert read_phrase(" five years\"") == "five years"
    for empty in ("", '"', "none", "None.", ' none"'):
        assert read_phrase(empty) is None
    assert locate_span(LEASE, "March 3, 2027") == (14, 27)
    # case and runs of whitespace are forgiven; the first occurrence wins
    assert locate_span(LEASE, "RENT   is") == (29, 36)
    assert locate_span("a b a b", "a b") == (0, 3)
    assert locate_span(LEASE, "April") is None
    assert locate_span(LEASE, "   ") is None
    assert line_span(LEASE, (14, 27)) == (0, 28)
    assert line_span(LEASE, (29, 36)) == (29, len(LEASE))
    assert line_span("one\n\nthree", (4, 4)) == (4, 4)


@pytest.mark.parametrize("filtered", [False, True])
def test_score_and_extract_return_both_columns(session, monkeypatch, filtered):
    execute = RerankerModelExecution.execute

    def mixed(self, node, inputs):
        if not isinstance(node, AiExtract):
            return execute(self, node, inputs)
        (source,) = inputs.values()
        ids = source.column("c").to_pylist()
        table = extract_answer_table(
            node.spec.alias, node.spec.name, node.spec.span_name,
            ids, ["March 3, 2027" if i == 0 else None for i in ids],
            [(14, 27) if i == 0 else None for i in ids])
        return NodeResult({"rows": table, "ids:c": ids})

    monkeypatch.setattr(RerankerModelExecution, "execute", mixed)
    query = session.sql(
        "SELECT c.id, AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) AS score, "
        f"AI.EXTRACT(c.text, '{QUESTION}') AS ends FROM contracts c "
        + ("WHERE AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) > 0.15"
           if filtered else ""))
    result = score_tests._finish(query, session, score_tests._RowReranker())
    rows = result.collect()
    assert rows.column("score").type == pa.float64()
    expected = [
        {"c.id": 1, "score": 0.1, "ends": "March 3, 2027",
         "ends_span": {"start": 14, "end": 27}},
        {"c.id": 2, "score": 0.2, "ends": None, "ends_span": None},
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
        assert sampling_params == ("decode", self.params[-1][0])
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            spelled = bytes(token - 1 for token in ids if 1 <= token <= 256)
            text = spelled.decode("utf-8", errors="ignore")
            if "March" in text:
                answer, tokens = "March 3, 2027", [1, 2, 3, 4]
            elif "five" in text:
                answer, tokens = "FIVE   years", [1, 2]
            elif "absent" in text:
                answer, tokens = "a phrase the document lacks", [1, 2, 3]
            else:
                answer, tokens = "none", [1]
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
        RequestExtractSpec(max_answer_tokens=limit,
                           alias="c", output="lines", question=QUESTION,
                           tail_token_ids=(90,), trim=False),
    )
    node = RequestExecution(
        node_id="request-model", backend_name="stock_vllm", aliases=("c",),
        preamble_token_ids=(3,), extracts=specs)
    client = _PhraseClient()
    execution = request_tests._execution(
        documents, client=client, document_texts={"c": TEXTS})
    result = execution.execute(node, {"input:0": [0, 1, 2, 3]})
    assert client.params == [(limit, ['"'])] * 2
    trimmed = result.outputs["extract_answers:ends"].to_pydict()
    assert trimmed == {
        "c": [0, 1, 2, 3],
        "ends": ["March 3, 2027", None, "five years",
                 "a phrase the document lacks"],
        "ends_span": [{"start": 14, "end": 27}, None, {"start": 6, "end": 16},
                      None]}
    whole = result.outputs["extract_answers:lines"].to_pydict()
    assert whole["lines"] == ["Lease ends on March 3, 2027.", None,
                              "Term: five years", "a phrase the document lacks"]
    assert whole["lines_span"] == [{"start": 0, "end": 28}, None,
                                   {"start": 0, "end": 16}, None]
    # an extraction keeps every document
    assert result.outputs["ids:c"] == [0, 1, 2, 3]
    step, _ = result.metrics.extension["steps"]
    assert (step["kind"], step["output"], step["n_in"], step["n_out"]) == (
        "extract", "ends", 4, 3)
    assert (step["requests"], step["unanswered"], step["unaligned"]) == (4, 1, 1)
    assert step["generated_tokens"] == 4 + 1 + 2 + 3
    # the preamble, document, and tail of each prompt, and each
    # answer's tokens but the last, fed back
    assert step["fresh_tokens"] == sum(
        2 + len(ids) for ids in documents["c"]) + (3 + 0 + 1 + 2)
    prompt_tokens = sum(2 + len(ids) for ids in documents["c"])
    assert step["prompt_tokens"] == prompt_tokens
    assert result.metrics.extension["prompt_tokens"] == 2 * prompt_tokens
    assert result.metrics.evaluated_documents == 8
    # an engine that takes text gets the preamble, the document, and
    # the tail's text, tokenized by the engine itself
    texted = _TextPhraseClient()
    node = replace(node, preamble_text="DOCUMENT:\n",
                   extracts=(replace(specs[0], tail_text="\nQ?\nPhrase: \""),))
    result = request_tests._execution(
        documents, client=texted, document_texts={"c": TEXTS}).execute(node, {})
    assert texted.prompts == ["DOCUMENT:\n" + text + "\nQ?\nPhrase: \""
                              for text in TEXTS]
    assert result.outputs["extract_answers:ends"].to_pydict()["ends"] == [
        "March 3, 2027", None, "five years", "a phrase the document lacks"]


def test_stock_vllm_plans_and_runs_an_extraction(monkeypatch):
    session = request_tests._session(
        "stock_vllm", contracts={"id": ["a", "b", "c", "d"], "text": TEXTS})
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') AS ends "
        f"FROM contracts c")
    boot = {"client": _PhraseClient(), "sampling_params": object(),
            "capacity": request_tests.CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids: (boot, {"kind": "cold", "boot_s": 0.1}))
    result = request_tests._run_stock_vllm(session, query)
    rows = result.collect()
    assert rows.column_names == ["c.id", "ends", "ends_span"]
    assert rows.column("c.id").to_pylist() == ["a", "b", "c", "d"]
    assert rows.column("ends").to_pylist() == [
        "March 3, 2027", None, "five years", "a phrase the document lacks"]
    assert rows.column("ends_span").to_pylist() == [
        {"start": 14, "end": 27}, None, {"start": 6, "end": 16}, None]
    answers = result.answer_tables["extracts"]["ends"]
    assert answers.column_names == ["c", "ends", "ends_span"]
    assert answers.num_rows == 4
    (step,) = result.report["backend_metrics"]["steps"]
    assert step["kind"] == "extract" and step["requests"] == 4
    session.close()


def test_extraction_settings_survive_plan_serialization(session):
    query = _ends(session)
    node = next(node for node in query.plan().nodes if isinstance(node, AiExtract))
    settings = dict(max_answer_tokens=7, draft_tokens=2, top_tokens=32,
                    first_window_chars=16, window_chars=96)
    spec = replace(node.spec, **settings)
    assert ExtractSpec.from_mapping(spec.to_dict()) == spec
    legacy = spec.to_dict()
    for name in settings:
        legacy.pop(name)
    assert ExtractSpec.from_mapping(legacy) == node.spec
    request = RequestExtractSpec(alias="c", output="ends", question=QUESTION,
                                 tail_token_ids=(1, 2), max_answer_tokens=7)
    assert RequestExtractSpec.from_mapping(request.to_dict()) == request
    legacy = request.to_dict()
    legacy.pop("max_answer_tokens")
    assert RequestExtractSpec.from_mapping(legacy).max_answer_tokens == 64
