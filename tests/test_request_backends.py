"""CPU tests for vLLM and SGLang backend interfaces."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

import quail
from quail.backends import pipelined_sglang_backend, stock_vllm_backend
from quail.backends.base import BackendExecutionContext, GpuContext
from quail.backends.request import RequestModelExecution
from quail.backends.sglang import SGLangClient
from quail.builtins import built_in_registry
from quail.physical import (
    RequestExecution,
    RequestFilterSpec,
    RequestJoinSpec,
    decode_graph,
)
from quail.planner.plan import EngineConfig
from quail.specs import H100_SXM, QWEN3_4B_FP8


def _tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _table():
    return quail.DocumentProvider.from_table(
        pa.table({
            "id": ["a", "b"],
            "body": ["one", "two"],
        }),
        id_col="id",
    )


def test_request_backends_plan_validate_and_execute(monkeypatch):
    registry = built_in_registry()
    assert set(registry.backends) == {
        "quail",
        "stock_vllm",
        "pipelined_vllm",
        "dumb_vllm",
        "pipelined_sglang",
    }

    for backend_name in (
        "stock_vllm",
        "pipelined_vllm",
        "pipelined_sglang",
    ):
        session = quail.Session(
            EngineConfig(
                gpus=1,
                model="qwen3-4b-fp8",
                backend=backend_name,
                device="h100-sxm",
            ),
            tokenizer=_tokens,
        )
        session.register("docs", _table())
        query = (
            session.docs("docs")
            .alias("d")
            .ai_filter(
                quail.prompt("useful {0}", quail.col("d.body"))
            )
            .select("d.id")
        )
        plan = query.plan()
        request_node = next(
            node for node in plan.nodes
            if isinstance(node, RequestExecution)
        )

        assert plan.backend == backend_name
        assert request_node.backend == backend_name
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

    with monkeypatch.context() as patch:
        session = quail.Session(
            EngineConfig(
                gpus=1,
                model="qwen3-4b-fp8",
                backend="stock_vllm",
                device="h100-sxm",
            ),
            tokenizer=_tokens,
        )
        session.register("docs", _table())
        query = (
            session.docs("docs")
            .alias("d")
            .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
            .select("d.id")
        )
        plan = query.plan()
        request = query._prepare_physical()
        client = _Client()

        patch.setattr(
            "quail.backends.vllm.VLLMEngine.boot",
            lambda self, spec, allowed_ids: (
                {
                    "client": client,
                    "sampling_params": object(),
                    "capacity": {
                        "kv_cache_size_tokens": 1_000,
                        "block_size": 1,
                        "max_num_seqs": 16,
                    },
                },
                {"kind": "cold", "boot_s": 0.1},
            ),
        )
        backend = session.registry.backend("stock_vllm")
        response = backend.execute_request(BackendExecutionContext(
            request=request,
            graph=plan.graph,
            registry=session.registry,
            gpu_count=1,
            runtime_state={},
        ))
        result = query.finish(response)

        assert result.report["backend"] == "stock_vllm"
        assert result.report["backend_metrics"]["requests"] == 2
        assert result.count() == 0
        session.close()

        session = quail.Session(
            EngineConfig(
                gpus=1,
                model="qwen3-4b-fp8",
                backend="stock_vllm",
                device="h100-sxm",
            ),
            tokenizer=_tokens,
        )
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": ["a", "b"], "body": ["one", "two"], "key": ["k1", "k2"],
        }), id_col="id"))
        session.register("notes", quail.DocumentProvider.from_table(pa.table({
            "id": ["x", "y", "z"], "text": ["p", "q", "r"],
            "key": ["k2", "k1", "k3"],
        }), id_col="id"))
        query = (
            session.docs("docs").alias("d")
            .join(session.docs("notes").alias("n"),
                  on=[quail.col("d.key") == quail.col("n.key")])
            .ai_filter(quail.prompt("{0} matches {1}", quail.col("d.body"),
                                    quail.col("n.text")))
            .select("d.id", "n.id")
        )
        plan = query.plan()
        request = query._prepare_physical()
        backend = session.registry.backend("stock_vllm")
        response = backend.execute_request(BackendExecutionContext(
            request=request, graph=plan.graph, registry=session.registry,
            gpu_count=1, runtime_state={}))
        result = query.finish(response)
        assert result.report["backend_metrics"]["requests"] == 2
        assert sorted(zip(*result.answer_tables["joins"][0].to_pydict().values())
                      ) == [(0, 1, False), (1, 0, False)]
        session.close()


class _Output:
    def __init__(self, prompt, answer, cached=0):
        self.prompt_token_ids = prompt
        self.num_cached_tokens = cached
        self.outputs = [SimpleNamespace(token_ids=[1 if answer else 2])]


class _Client:
    def generate(self, prompts, sampling_params, use_tqdm=False):
        del sampling_params, use_tqdm
        return [
            _Output(
                prompt["prompt_token_ids"],
                10 in prompt["prompt_token_ids"],
            )
            for prompt in prompts
        ]

    def reset_prefix_cache(self):
        return True


def _execution(documents, **settings):
    return RequestModelExecution(GpuContext(
        gpu_index=0,
        gpu_count=1,
        model=QWEN3_4B_FP8,
        device=H100_SXM,
        query_settings={
            "client": _Client(),
            "sampling_params": object(),
            "documents": documents,
            "true_ids": [1],
            "capacity": {
                "kv_cache_size_tokens": 1_000,
                "block_size": 1,
                "max_num_seqs": 16,
            },
            "filter_submission": "operator-at-a-time",
            "join_submission": "anchor-major",
            **settings,
        },
    ))


def test_suffix_major_falls_back_when_the_anchors_exceed_kv():
    class RecordingClient(_Client):
        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.prompts = [tuple(prompt["prompt_token_ids"]) for prompt in prompts]
            return super().generate(prompts, sampling_params, use_tqdm=use_tqdm)

    node = RequestExecution(
        node_id="request-model",
        backend_name="pipelined_sglang",
        aliases=("r", "p"),
        preamble_token_ids=(3,),
        joins=(RequestJoinSpec(
            written_pos=0, aliases=("r", "p"), outer_aliases=("r",),
            anchor="r", semantics="full", selectivity=0.5,
            label_token_ids=(("r", (40,)), ("p", (41,))),
            frame_token_ids=(("r", (30,)), ("p", (31,))),
            tail_token_ids=(50,),
        ),),
    )
    documents = {"r": [[10], [11]], "p": [[20], [21]]}
    # two anchor prefixes of three tokens each: six tokens held at once
    orders = {}
    for kv_tokens in (5, 6):
        execution = _execution(
            documents, join_submission="suffix-major",
            capacity={"kv_cache_size_tokens": kv_tokens, "block_size": 1,
                      "max_num_seqs": 16})
        execution.client = RecordingClient()
        execution.execute(node, {})
        orders[kv_tokens] = [(prompt[1], prompt[4])
                             for prompt in execution.client.prompts]
    assert orders[6] == [(10, 20), (11, 20), (10, 21), (11, 21)]
    assert orders[5] == [(10, 20), (10, 21), (11, 20), (11, 21)]


def test_filter_and_join_answer_relations():
    node = RequestExecution(
        node_id="request-model",
        backend_name="stock_vllm",
        aliases=("d",),
        preamble_token_ids=(3,),
        filters=(RequestFilterSpec(
            alias="d",
            written_positions=(0,),
            question_token_ids=((90,),),
        ),),
    )
    result = _execution({"d": [[10], [11]]}).execute(node, {})

    table = result.outputs["filter_answers:d"]
    assert table.to_pydict() == {
        "d": [0, 1],
        "predicate": [0, 0],
        "answer": [True, False],
    }
    assert result.outputs["ids:d"] == [0]
    assert result.metrics.evaluated_documents == 2

    class NativeIntClient(_Client):
        def generate(self, prompts, sampling_params, use_tqdm=False):
            assert all(
                isinstance(token, int)
                for prompt in prompts
                for token in prompt["prompt_token_ids"]
            )
            return super().generate(
                prompts, sampling_params, use_tqdm=use_tqdm
            )

    execution = _execution({"d": [[np.int32(10)]]})
    execution.client = NativeIntClient()
    node = RequestExecution(
        node_id="request-model",
        backend_name="stock_vllm",
        aliases=("d",),
        preamble_token_ids=(3,),
        filters=(RequestFilterSpec(
            alias="d",
            written_positions=(0,),
            question_token_ids=((90,),),
        ),),
    )

    result = execution.execute(node, {})

    assert result.outputs["ids:d"] == [0]

    node = RequestExecution(
        node_id="request-model",
        backend_name="stock_vllm",
        aliases=("r", "p"),
        preamble_token_ids=(3,),
        joins=(RequestJoinSpec(
            written_pos=0,
            aliases=("r", "p"),
            outer_aliases=("r",),
            anchor="r",
            semantics="full",
            selectivity=0.5,
            label_token_ids=(("r", (40,)), ("p", (41,))),
            frame_token_ids=(("r", (30,)), ("p", (31,))),
            tail_token_ids=(50,),
        ),),
    )
    result = _execution({
        "r": [[10], [11]],
        "p": [[20], [21]],
    }).execute(node, {})

    table = result.outputs["join_answers:0"]
    assert table.to_pydict() == {
        "r": [0, 0, 1, 1],
        "p": [0, 1, 0, 1],
        "answer": [True, True, False, False],
    }
    assert result.outputs["ids:r"] == [0]
    assert result.outputs["ids:p"] == [0, 1]
    assert result.metrics.evaluated_document_pairs == 4


def test_cached_token_accounting():
    from quail.backends.request_scheduling import run_filter_chain_async

    class Output:
        def __init__(self, prompt, cached):
            self.prompt_token_ids = prompt
            self.num_cached_tokens = cached
            self.outputs = [SimpleNamespace(token_ids=[1])]

    async def generate(prompt, sampling_params):
        cached = {(5, 8): 0, (6, 8): 6, (5, 9): 2, (6, 9): 6}
        return Output(prompt, cached[(prompt[0], prompt[-1])])

    result = asyncio.run(run_filter_chain_async(
        generate, object(), [[5, 5, 5, 5], [6, 6, 6, 6]],
        [[8, 8], [9, 9]], 100, true_ids={1}, block_size=1,
    ))

    # four requests of six tokens; the fake engine served 14 of them
    assert result["prompt_tokens"] == 24
    assert result["cached_tokens"] == 14


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
    assert len(engine.calls) == 1
    assert engine.calls[0] == [prompt["prompt_token_ids"] for prompt in prompts]
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


def test_vllm_engine_settings_follow_the_model():
    from quail.backends.vllm import VLLMEngine, sampling_kwargs
    from quail.specs import DIFFUSION_GEMMA_26B_FP8, QWEN3_4B_FP8

    engine = VLLMEngine()
    assert engine.llm_kwargs(QWEN3_4B_FP8)["max_num_batched_tokens"] == 25_305
    assert "diffusion_config" not in engine.llm_kwargs(QWEN3_4B_FP8)
    # the mixture-of-experts model batches its own chunk cap; its
    # one-row canvas commits after one denoising step, and stays under
    # vLLM's eight-sequence cap trigger
    gemma = engine.llm_kwargs(DIFFUSION_GEMMA_26B_FP8)
    assert gemma["max_num_batched_tokens"] == 65_536
    assert gemma["diffusion_config"] == {"canvas_length": 1,
                                         "max_denoising_steps": 1}
    assert gemma["max_num_seqs"] == 127
    assert gemma["max_logprobs"] == 500
    wide = engine.llm_kwargs(replace(DIFFUSION_GEMMA_26B_FP8, canvas_tokens=256))
    assert wide["diffusion_config"] == {"canvas_length": 256}
    assert wide["max_num_seqs"] == engine.llm_kwargs(QWEN3_4B_FP8)["max_num_seqs"]
    assert sampling_kwargs([1, 2]) == {
        "temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
        "allowed_token_ids": [1, 2]}
    # the diffusion sampler rejects everything but the length; a
    # one-row canvas returns its logprobs, a longer one free text
    assert sampling_kwargs([1, 2], canvas_tokens=1) == {
        "max_tokens": 1, "logprobs": 500}
    assert sampling_kwargs([1, 2], canvas_tokens=256) == {"max_tokens": 16}


def test_true_bit_reads_the_answer_word_from_free_text():
    from types import SimpleNamespace

    from quail.backends.request_scheduling import true_bit

    def output(token_ids, text):
        return SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids,
                                                        text=text)])

    assert true_bit(output([7], "TRUE"), {7}) == 1
    assert true_bit(output([8], "FALSE"), {7}) == 0
    assert true_bit(output([1, 2], "The answer is TRUE."), {7}) == 1
    assert true_bit(output([1, 2], "**FALSE** because"), {7}) == 0
    assert true_bit(output([1, 2], "TRUEISH FALSE"), {7}) == 0
    assert true_bit(output([], ""), {7}) == 0


def test_true_bit_ranks_the_answer_words_in_the_logprobs():
    from types import SimpleNamespace

    from quail.backends.request_scheduling import true_bit

    def entry(logprob, decoded=None):
        return SimpleNamespace(logprob=logprob, decoded_token=decoded)

    def output(logprobs, token_ids=(1,), text=""):
        return SimpleNamespace(outputs=[SimpleNamespace(
            token_ids=list(token_ids), text=text, logprobs=logprobs)])

    # the sampled token is end-of-turn; TRUE outranks FALSE below it
    assert true_bit(output([{1: entry(-0.1, "<eos>"), 7: entry(-2.0, " TRUE"),
                             8: entry(-3.5, " FALSE")}]), {7}) == 1
    assert true_bit(output([{1: entry(-0.1, "<eos>"), 7: entry(-4.0, " TRUE"),
                             8: entry(-3.5, "FALSE")}]), {7}) == 0
    # a decoded TRUE counts even when its id is not a known true id
    assert true_bit(output([{1: entry(-0.1, "<eos>"), 9: entry(-1.0, "TRUE")}]),
                    {7}) == 1
    # neither word present: fall back to the text, then to FALSE
    assert true_bit(output([{1: entry(-0.1, "<eos>")}], text="TRUE"), {7}) == 1
    assert true_bit(output([{1: entry(-0.1, "<eos>")}]), {7}) == 0
    assert true_bit(output(None), {7}) == 0
