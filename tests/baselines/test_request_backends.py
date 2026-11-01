"""CPU tests for vLLM and SGLang backend interfaces."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from helpers.baselines import (
    CAPACITY,
    _Client,
    _run_stock_vllm,
    _session,
)

from quail.baselines.nodes import RequestExecution
from quail.baselines.sglang import SGLangClient, pipelined_sglang_backend
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
        "quail", "stock_vllm", "pipelined_vllm", "dumb_vllm", "pipelined_sglang",
    }

    for backend_name in ("stock_vllm", "pipelined_vllm", "pipelined_sglang"):
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

    for backend in (stock_vllm_backend(), pipelined_sglang_backend()):
        support = backend.supports(QWEN3_4B_FP8, H100_SXM, 2)
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


def test_sglang_submission_and_cancellation():
    class Engine:
        def __init__(self):
            self.calls = []

        def generate(self, *, input_ids, sampling_params):
            self.calls.append(input_ids)
            return [
                {"output_ids": [prompt[0] % 2], "meta_info": {"cached_tokens": 1}}
                for prompt in input_ids
            ]

    engine = Engine()
    client = SGLangClient(engine, {})
    prompts = [{"prompt_token_ids": [index, 9]} for index in range(17_000)]
    result = client.generate(prompts, {})
    assert engine.calls == [[prompt["prompt_token_ids"] for prompt in prompts]]
    assert [output.outputs[0].token_ids[0] for output in result] == [
        index % 2 for index in range(17_000)]
    assert all(output.num_cached_tokens == 1 for output in result)
    assert client.generate([], {}) == []
    assert len(engine.calls) == 1

    async def run():
        started = asyncio.Event()
        submitted = []
        aborted = []

        async def generate(**kwargs):
            submitted.append(kwargs["rid"])
            started.set()
            await asyncio.Event().wait()

        engine = SimpleNamespace(
            async_generate=generate,
            tokenizer_manager=SimpleNamespace(abort_request=aborted.append),
        )
        task = asyncio.create_task(SGLangClient(engine, {})._generate_one([1, 2], {}))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert aborted == submitted
        assert len(aborted) == 1

    asyncio.run(run())


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
