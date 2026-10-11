"""AI.EXTRACT: the operator surface, its prompt, and its plan."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import letter_tokens

import quail
from quail.catalog import DocumentProvider
from quail.execution.execute import execute_query
from quail.logical import (
    SHARED_PRE,
    Alias,
    ColumnRef,
    CompileError,
    LogicalPlan,
    ModelCall,
    Scan,
    SemanticExtract,
    bind_extract_prompt,
    extracted_above_joins,
)
from quail.logical.prompts import (
    EXTRACT_LINES_CUE,
    EXTRACT_PHRASE_CUE,
    NUMBERED_DOCUMENT_LABEL,
)
from quail.physical import (
    AiExtract,
    AiFilter,
    AiJoin,
    ExtractSpec,
    Filter,
    GraphValidationError,
    decode_graph,
    encode_graph,
)
from quail.planner.logical_rules import push_down_projection
from quail.planner.plan import EngineConfig, Refusal
from quail.planner.validation import extraction_refusal
from quail.specs import (
    DECISION_2_KAI_0_6B_BF16,
    DIFFUSION_GEMMA_26B_FP8,
    QWEN3_RERANKER_0_6B_BF16,
)

QUESTION = "When does the lease end?"
LEASE = "Lease ends on March 3, 2027.\nRent is $2,000."

# the text after the document, as measured on SQuAD and CUAD
AFTER_DOCUMENT_TEXT = (
    "\n\nQuestion: When does the lease end?\n\n"
    "Copy the words that answer the question from the document. Copy them "
    "exactly. Use the fewest words that answer it. If the document does not "
    "answer the question, write none.\n"
    "Answer in this format:\n"
    'Phrase: "<copied words>"')
AFTER_NUMBERED_DOCUMENT_TEXT = AFTER_DOCUMENT_TEXT.replace(
    "Answer in this format:\n", "Answer in this format:\n"
    "Lines: <first line>-<last line>\n")


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


def test_extract_prompt_puts_every_question_dependent_text_after_the_document():
    ref = ColumnRef("c", "contracts", "text")
    turn = ("<user>", "<assistant>")
    prompt = bind_extract_prompt((ref,), QUESTION, letter_tokens, turn)
    assert prompt.template == "{0}" and prompt.args == (ref,)
    assert prompt.preamble == "<user>" + SHARED_PRE
    assert prompt.tail == (
        "{0}" + AFTER_DOCUMENT_TEXT + "<assistant>" + EXTRACT_PHRASE_CUE)
    assert prompt.frame == "" and prompt.frame_tokens == 0
    assert prompt.preamble_tokens == len(prompt.preamble_token_ids) == len(
        prompt.preamble.encode())
    assert prompt.tail_tokens == len(prompt.tail_token_ids) == len(
        prompt.tail[3:].encode())
    numbered = bind_extract_prompt((ref,), QUESTION, letter_tokens, turn,
                                   numbered=True)
    assert numbered.preamble == "<user>" + NUMBERED_DOCUMENT_LABEL
    assert numbered.tail == (
        "{0}" + AFTER_NUMBERED_DOCUMENT_TEXT + "<assistant>" + EXTRACT_LINES_CUE)
    # without a tokenizer the counts stay for the planner to fill
    untokenized = bind_extract_prompt((ref,), QUESTION)
    assert untokenized.preamble_tokens is None
    assert untokenized.tail_token_ids == ()
    with pytest.raises(CompileError, match="one document column"):
        bind_extract_prompt((ref, ref), QUESTION)
    with pytest.raises(CompileError, match="at least one word"):
        bind_extract_prompt((ref,), "  ")


def test_sql_and_builder_extract_the_same_answer_columns(session):
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') AS ends "
        f"FROM contracts c")
    built = _ends(session)
    assert query.logical == built.logical
    logical = built.logical
    assert [type(node).__name__ for node in logical.walk()] == [
        "Scan", "SemanticExtract", "Project"]
    (node,) = logical.operators().extracts
    assert isinstance(node, SemanticExtract)
    assert (node.name, node.alias, node.span_name) == ("ends", "c", "ends_span")
    call = node.call
    assert call.kind == "extract" and call.question == QUESTION and call.trim
    assert call.aliases() == ("c",)
    assert logical.root.columns[1] == Alias(call, "ends")
    assert node.output_schema()[-2:] == (
        ColumnRef("c", "contracts", "ends"), ColumnRef("c", "contracts", "ends_span"))
    # the answer columns are not read below the extraction; the scan
    # keeps the id column and, as for every projected call, the text
    pushed = LogicalPlan(push_down_projection(logical.root))
    (scan,) = [node for node in pushed.walk() if isinstance(node, Scan)]
    assert scan.columns == ("id", "text")
    # trim is an option of the call
    loose = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}', {{'trim': false}}) "
        f"AS ends FROM contracts c")
    assert loose.logical == _ends(session, trim=False).logical
    (node,) = loose.logical.operators().extracts
    assert not node.call.trim
    # the BigQuery dialect names the call the same way
    assert session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') AS ends "
        f"FROM contracts c", dialect="bq").logical == logical
    text = built.explain()
    assert "SemanticExtract: ends, ends_span" in text
    assert f"question={QUESTION!r} trim=on" in text
    assert "AiExtract: ends over c" in text
    assert "not priced" in text
    assert "Project: c.id, ends" in text


def test_extract_plans_one_physical_node_with_both_prompts(session):
    query = _ends(session)
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    assert [type(node).__name__ for node in plan.nodes] == [
        "Scan", "AiExtract", "Project"]
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    spec = node.spec
    assert isinstance(spec, ExtractSpec)
    assert (spec.name, spec.alias, spec.span_name) == ("ends", "c", "ends_span")
    assert spec.question == QUESTION and spec.trim
    assert spec.arguments == (("c", "text"),)
    assert spec.expected_inputs == 2
    assert spec.estimated_seconds == 0.0
    turn = session.model.turn
    head, tail = spec.prompt_token_parts
    assert list(head) == letter_tokens(turn[0] + SHARED_PRE)
    assert list(tail) == letter_tokens(
        AFTER_DOCUMENT_TEXT + turn[1] + EXTRACT_PHRASE_CUE)
    head, tail = spec.numbered_token_parts
    assert list(head) == letter_tokens(turn[0] + NUMBERED_DOCUMENT_LABEL)
    assert list(tail) == letter_tokens(
        AFTER_NUMBERED_DOCUMENT_TEXT + turn[1] + EXTRACT_LINES_CUE)
    assert [port.name for port in node.outputs] == ["scores", "ids:c"]
    assert node.outputs[0].schema == ("c", "ends", "ends_span")
    assert node.runtime_key == "quail.ai_extract"
    assert plan.nodes[-1].columns == ("c.id", "ends", "ends_span")
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    # the executor for the node is not registered yet, so running the
    # plan is refused by name rather than failing inside the engine
    with pytest.raises(GraphValidationError, match="quail.ai_extract"):
        execute_query(query, physical_executor=lambda request: None)


def test_extract_follows_the_filters_and_labels_of_its_table(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
             .ai_classify(quail.prompt("{0}", quail.col("c.text")),
                          ["lease", "deed"], name="kind")
             .label_in("kind", ["lease"], selectivity=0.5)
             .ai_extract("c.text", QUESTION, name="ends")
             .select("c.id", "kind", "ends"))
    assert [type(node).__name__ for node in query.logical.walk()] == [
        "Scan", "SemanticFilter", "SemanticClassify", "Filter",
        "SemanticExtract", "Project"]
    plan = query.plan()
    assert [type(node).__name__ for node in plan.nodes] == [
        "Scan", "AiFilter", "AiClassify", "Filter", "AiExtract", "Project"]
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert node.inputs[0].source.node_id == test.node_id
    # half the documents pass the label filter
    assert node.spec.expected_inputs == pytest.approx(2 * 0.2 * 0.5)
    assert plan.nodes[-1].columns == ("c.id", "kind", "ends", "ends_span")


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
    # the extraction is written on its table, below the join
    assert [type(node).__name__ for node in query.logical.walk()] == [
        "Scan", "SemanticExtract", "Scan", "Join", "SemanticJoin", "Project"]
    assert extracted_above_joins(query.logical.root) == frozenset()
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    written, lifted = _candidates(session, query)
    # the backend also offers the extraction after the join, over the
    # documents the join matched; the two cost the same until the
    # extraction is priced, so the plan as written wins
    assert written.logical_plan.root == query.logical.root
    assert extracted_above_joins(lifted.logical_plan.root) == {"c"}
    assert [type(node).__name__
            for node in LogicalPlan(lifted.logical_plan.root).walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticExtract", "Project"]
    assert written.estimated_seconds == lifted.estimated_seconds
    assert [node.node_id for node in plan.nodes] == [
        "scan:c", "scan:q", "ai-extract:0", "ai_join:q", "project"]
    (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert ("ai-extract:0", "ids:c") in [
        (port.source.node_id, port.source.port) for port in join.inputs]
    assert [node.node_id for node in lifted.plan.nodes] == [
        "scan:c", "scan:q", "ai_join:q", "ai-extract:0", "project"]
    (node,) = [n for n in lifted.plan.nodes if isinstance(n, AiExtract)]
    # a fifth of the pairs match by default, one partner each
    assert node.spec.expected_inputs == pytest.approx(2 * 0.2)
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
    call = ModelCall(bind_extract_prompt(
        (ColumnRef("c", "contracts", "text"),), QUESTION), "extract",
        question=QUESTION)
    call.validate()
    with pytest.raises(CompileError, match="only an AI.EXTRACT call"):
        ModelCall(call.prompt, "boolean", question=QUESTION).validate()
    with pytest.raises(CompileError, match="only an AI.EXTRACT call"):
        ModelCall(call.prompt, "label", labels=("a", "b"), trim=False).validate()
    with pytest.raises(CompileError, match="only an AI.CLASSIFY call"):
        ModelCall(call.prompt, "extract", question=QUESTION,
                  labels=("a",)).validate()

    # the speed of light estimate does not price an extraction yet
    with pytest.raises(NotImplementedError, match="semantic_extract"):
        quail.speed_of_light_estimate(_ends(session), lambda *_: True)
    # neither request backend runs it
    vllm = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                      backend="stock_vllm"),
                         tokenizer=letter_tokens)
    vllm.register("contracts", session.catalog.get("contracts"))
    refused = _ends(vllm).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "extract_needs_quail_backend"
    vllm.close()
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
    for model in (DECISION_2_KAI_0_6B_BF16, DIFFUSION_GEMMA_26B_FP8,
                  QWEN3_RERANKER_0_6B_BF16):
        refused = extraction_refusal(
            SimpleNamespace(model=model, tokenizer=letter_tokens))
        assert refused.constraint == "unsupported_extract_query"
        assert "does not give" in refused.reasons[0]
    assert extraction_refusal(
        SimpleNamespace(model=session.model, tokenizer=letter_tokens)) is None
    assert "tokenizer" in extraction_refusal(
        SimpleNamespace(model=session.model, tokenizer=None)).reasons[0]
    # an extraction and a score do not mix
    mixed = (session.docs("contracts").alias("c")
             .ai_score(quail.prompt("Is {0} a lease?", quail.col("c.text")),
                       name="lease")
             .ai_extract("c.text", QUESTION, name="ends")
             .select("c.id", "lease", "ends"))
    refused = mixed.plan()
    assert isinstance(refused, Refusal)
    assert "AI.SCORE" in refused.reasons[0]
    # an extraction beside an AI.IF filter over the same column shares
    # the document's tokens
    filtered = (session.docs("contracts").alias("c")
                .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
                .ai_extract("c.text", QUESTION, name="ends")
                .select("c.id", "ends"))
    nodes = filtered.plan().nodes
    assert isinstance(nodes[1], AiFilter) and isinstance(nodes[2], AiExtract)
