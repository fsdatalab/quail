"""CPU tests for the vLLM backend interfaces."""

import sys
from types import SimpleNamespace

import pytest
from helpers.baselines import (
    CAPACITY,
    _Client,
    _execution,
    _run_stock_vllm,
    _session,
)

from quail.baselines.nodes import RequestExecution
from quail.baselines.vllm import stock_vllm_backend
from quail.builtins import built_in_registry
from quail.physical import decode_graph
from quail.specs import (
    B200,
    H100_SXM,
    QWEN3_4B_FP8,
)

DOCS = {"id": ["a", "b"], "body": ["one", "two"]}


MATCHED_NOTES_SQL = """
    FROM documents d
    JOIN notes n
      ON d.key = n.key AND AI_FILTER(PROMPT('{0} matches {1}', d.body, n.text))
"""


def _filter_query(session):
    return session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('useful {0}', d.body))")


def test_request_backends_plan_validate_and_execute(monkeypatch):
    registry = built_in_registry()
    assert set(registry.backends) == {
        "quail", "stock_vllm", "pipelined_vllm", "shuffled_vllm", "ngram_vllm",
        "dumb_vllm"}

    for backend_name in ("stock_vllm", "pipelined_vllm"):
        session = _session(backend_name, documents=DOCS)
        plan = _filter_query(session).plan()
        request_node = next(
            node for node in plan.nodes if isinstance(node, RequestExecution))
        assert plan.backend == request_node.backend == backend_name
        assert request_node.filters[0].alias == "d"
        plan.graph.validate_backend(backend_name)
        envelope = plan.to_envelope(registry.codecs)
        decoded = decode_graph(envelope["graph"], registry.codecs)
        assert decoded.node("request-model") == request_node
        session.close()

    support = stock_vllm_backend().supports(QWEN3_4B_FP8, H100_SXM, 2)
    assert not support.supported
    assert "one model copy on one GPU" in support.reason

    boot = {"client": _Client(), "sampling_params": object(), "capacity": CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids, *, device, resources: (
            boot, {"kind": "cold", "boot_s": 0.1}))
    session = _session(
        "stock_vllm",
        documents={**DOCS, "key": ["k1", "k2"]},
        notes={"id": ["x", "y", "z"], "text": ["p", "q", "r"],
               "key": ["k2", "k1", "k3"]},
    )
    result = _run_stock_vllm(session, _filter_query(session))
    assert result.report["backend"] == "stock_vllm"
    assert result.report["backend_metrics"]["requests"] == 2
    assert result.count() == 0
    query = session.sql(f"SELECT d.id, n.id {MATCHED_NOTES_SQL}")
    result = _run_stock_vllm(session, query)
    assert result.report["backend_metrics"]["requests"] == 2
    joins = result.answer_tables["joins"][0].to_pydict()
    assert sorted(zip(*joins.values())) == [(0, 1, False), (1, 0, False)]

    # a classification of the join's pairs: its label table names both
    # aliases before the label column; the join kept no pair here
    boot["client"] = _JoinedClient()
    query = session.sql(
        "SELECT d.id, n.id, AI.CLASSIFY(PROMPT('{0} about {1}', d.body, n.text), "
        f"['a', 'b']) AS kind {MATCHED_NOTES_SQL}")
    result = _run_stock_vllm(session, query)
    labels = result.answer_tables["classifies"]["kind"]
    assert labels.column_names == ["d", "n", "kind"]
    assert labels.num_rows == 0 and result.count() == 0
    session.close()


class _JoinedClient(_Client):
    """Joins by token, and decodes a joined row's label from its partner."""

    def decode_params(self, max_tokens):
        return ("decode", max_tokens)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = super().generate(prompts, sampling_params, use_tqdm)
        if sampling_params != ("decode", 3):
            return outputs
        self.prompts = [tuple(prompt["prompt_token_ids"]) for prompt in prompts]
        for output in outputs:
            output.outputs[0].text = " a" if 20 in output.prompt_token_ids else " b"
            output.outputs[0].token_ids = [1, 2]
        return outputs


class _Adapter:
    """An engine adapter whose boot fails until told otherwise."""

    kind = "fake"

    def __init__(self):
        self.live = []
        self.fail = True

    def boot(self, spec, allowed_ids, *, device, resources):
        self.device = device
        client = SimpleNamespace(close=lambda: self.live.remove(client))
        self.live.append(client)
        resources.callback(client.close)
        if self.fail:
            raise RuntimeError("warmup failed")
        return {"client": client}, {"kind": "cold"}


def test_a_request_backend_owns_its_engine_from_boot_to_release(monkeypatch):
    monkeypatch.setattr("quail.backends.resources._release_vllm_parallel_state",
                        lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    adapter = _Adapter()
    backend = stock_vllm_backend().__class__("test", adapter, "pipelined")
    context = SimpleNamespace(
        registry=built_in_registry(), runtime_state={},
        request=SimpleNamespace(plan={
            "model": "qwen3-4b-fp8", "device": "b200",
            "settings": {"true_ids": [1], "false_ids": [2]}}))
    with pytest.raises(RuntimeError, match="warmup failed"):
        backend.prepare_request(context)
    assert adapter.live == [] and context.runtime_state == {}
    assert adapter.device is B200

    adapter.fail = False
    backend.prepare_request(context)
    engine = context.runtime_state["engine"]
    assert len(adapter.live) == 1
    assert engine.prepared_boot["kind"] == "cold"
    monkeypatch.setattr("quail.baselines.request.execute_request_graph",
                        lambda context, backend, settings, boot: boot["kind"])
    assert backend.execute_request(context) == "cold"
    assert backend.execute_request(context) == "warm"
    backend.release_request(context.runtime_state)
    assert adapter.live == [] and context.runtime_state == {}
class _MapClient:
    """Answers each prompt with its last two tokens and limit, by call or step loop."""

    def __init__(self):
        self.pending = {}

    def decode_params(self, max_tokens, stop=None):
        return max_tokens

    @staticmethod
    def _output(ids, limit, request_id=None):
        return SimpleNamespace(
            request_id=request_id, finished=True, prompt_token_ids=ids,
            num_cached_tokens=0,
            outputs=[SimpleNamespace(token_ids=[1],
                                     text=f"{ids[-2]}:{ids[-1]}/{limit}")])

    def generate(self, prompts, params, use_tqdm=False):
        return [self._output(prompt["prompt_token_ids"], limit)
                for prompt, limit in zip(prompts, params)]

    def add_request(self, request_id, prompt, limit):
        self.pending[request_id] = (prompt["prompt_token_ids"], limit)

    def step(self):
        outputs = [self._output(ids, limit, request_id)
                   for request_id, (ids, limit) in self.pending.items()]
        self.pending.clear()
        return outputs

    def run_extract_fields(self, sampling_params, body_ids, tail_ids, *,
                           answer_tokens, body_texts=None, tail_texts=None):
        from quail.baselines.request_scheduling import run_extract_fields

        return run_extract_fields(self, sampling_params, body_ids, tail_ids,
                                  CAPACITY["kv_cache_size_tokens"],
                                  answer_tokens=answer_tokens)


@pytest.mark.parametrize("with_maps", [False, True])
@pytest.mark.parametrize("submission", ["operator-at-a-time", "shuffled"])
def test_extracts_and_maps_of_a_document_go_in_one_generate_call(submission,
                                                                 with_maps):
    from quail.baselines.nodes import RequestExtractSpec, RequestMapSpec

    class Client(_MapClient):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, prompts, params, use_tqdm=False):
            self.calls.append(len(prompts))
            return super().generate(prompts, params, use_tqdm)

    client = Client()
    maps = (RequestMapSpec(alias="d", output="plot", tail_token_ids=(90,),
                           max_answer_tokens=1024),) if with_maps else ()
    node = RequestExecution(
        node_id="request-model", backend_name="stock_vllm", aliases=("d",),
        extracts=tuple(RequestExtractSpec(alias="d", output=output,
                                          tail_token_ids=(tail,),
                                          max_answer_tokens=16)
                       for output, tail in (("kind", 80), ("size", 81))),
        maps=maps)
    result = _execution(
        {"d": [[10], [11], [12]]}, client=client,
        document_texts={"d": ["10:80/16 10:81/16", "it says 11:80", "three"]},
        answer_submission=submission).execute(node, {"input:0": [0, 1, 2]})
    assert client.calls == [9 if with_maps else 6]
    if with_maps:
        assert result.outputs["map_answers:plot"].to_pydict()["plot"] == [
            "10:90/1024", "11:90/1024", "12:90/1024"]
    # each answer is cut to the longest prefix its document holds
    assert result.outputs["extract_answers:kind"].to_pydict()["kind"] == [
        "10:80/16", "11:80", None]
    assert result.outputs["extract_answers:size"].to_pydict()["size"] == [
        "10:81/16", "11:", None]


@pytest.mark.parametrize("submission", ["operator-at-a-time", "shuffled", "pipelined"])
def test_maps_write_each_answer_under_its_own_limit(submission):
    from quail.baselines.nodes import RequestMapSpec

    specs = tuple(RequestMapSpec(alias="d", output=output, tail_token_ids=(tail,),
                                 max_answer_tokens=limit)
                  for output, tail, limit in (("plot", 90, 1024), ("topic", 91, 32)))
    node = RequestExecution(node_id="request-model", backend_name="stock_vllm",
                            aliases=("d",), maps=specs)
    result = _execution(
        {"d": [[10], [11], [12]]}, client=_MapClient(),
        document_texts={"d": ["one", "two", "three"]},
        answer_submission=submission).execute(node, {"input:0": [0, 1, 2]})
    assert result.outputs["map_answers:plot"].to_pydict()["plot"] == [
        "10:90/1024", "11:90/1024", "12:90/1024"]
    assert result.outputs["map_answers:topic"].to_pydict()["topic"] == [
        "10:91/32", "11:91/32", "12:91/32"]
    steps = result.metrics.extension["steps"]
    assert {step["submission"] for step in steps} == {submission}




def test_ngram_vllm_drafts_from_the_prompt():
    from quail.baselines.vllm import NgramVLLMEngine, VLLMEngine, ngram_vllm_backend

    backend = ngram_vllm_backend()
    assert backend.answer_submission == "operator-at-a-time"
    assert backend.engine_name != VLLMEngine.kind
    kwargs = NgramVLLMEngine().llm_kwargs(QWEN3_4B_FP8, H100_SXM)
    assert kwargs["speculative_config"] == {
        "method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
    assert "speculative_config" not in VLLMEngine().llm_kwargs(QWEN3_4B_FP8, H100_SXM)
