"""CPU tests for vLLM and SGLang backend interfaces."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa

import quail
from quail.backends.base import BackendExecutionContext
from quail.backends import pipelined_sglang_backend, stock_vllm_backend
from quail.backends.base import GpuContext
from quail.backends.request import RequestModelExecution
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


def test_built_in_request_backends_plan_their_own_model_node():
    registry = built_in_registry()
    assert set(registry.backends) == {
        "quail",
        "stock_vllm",
        "pipelined_vllm",
        "pipelined_sglang",
    }

    for backend_name in (
        "stock_vllm",
        "pipelined_vllm",
        "pipelined_sglang",
    ):
        session = quail.Session(
            EngineConfig(backend=backend_name),
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


def test_request_backends_reject_multiple_gpus():
    for backend in (stock_vllm_backend(), pipelined_sglang_backend()):
        support = backend.supports(QWEN3_4B_FP8, H100_SXM, 2)
        assert not support.supported
        assert "one model copy on one GPU" in support.reason


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


def _execution(documents):
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
            "filter_submission": "stage-major",
            "join_submission": "anchor-major",
        },
    ))


def test_request_model_execution_returns_filter_answer_relation():
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


def test_request_model_execution_converts_numpy_token_ids():
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


def test_request_model_execution_returns_join_answer_relation():
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
    assert result.metrics.regret_tokens > 0


def test_vllm_backend_executes_a_physical_request(monkeypatch):
    session = quail.Session(
        EngineConfig(backend="stock_vllm"),
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

    monkeypatch.setattr(
        "quail.backends.vllm.VLLMEngine.boot",
        lambda self, model_name, allowed_ids: (
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
