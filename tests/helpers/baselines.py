"""Sessions and a fake vLLM client for the request backend tests."""

from types import SimpleNamespace

import pyarrow as pa

import quail
from quail.backends.base import BackendExecutionContext, GpuContext
from quail.baselines.execution import RequestModelExecution
from quail.planner.plan import EngineConfig
from quail.specs import H100_SXM, QWEN3_4B_FP8

CAPACITY = {"kv_cache_size_tokens": 1_000, "block_size": 1, "max_num_seqs": 16}


def _tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _session(backend, **tables):
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=_tokens,
    )
    for name, columns in tables.items():
        session.register(name, quail.DocumentProvider.from_table(
            pa.table(columns), id_col="id"))
    return session


def _run_stock_vllm(session, query):
    plan = query.plan()
    response = session.registry.backend("stock_vllm").execute_request(
        BackendExecutionContext(
            request=query._prepare_physical(), graph=plan.graph,
            registry=session.registry, gpus=1, runtime_state={}))
    return query.finish(response)


class _Client:
    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert all(isinstance(token, int)
                   for prompt in prompts for token in prompt["prompt_token_ids"])
        return [SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"], num_cached_tokens=0,
            outputs=[SimpleNamespace(
                token_ids=[1 if 10 in prompt["prompt_token_ids"] else 2])],
        ) for prompt in prompts]

    def reset_prefix_cache(self):
        return True


def _execution(documents, *, model=QWEN3_4B_FP8, **settings):
    return RequestModelExecution(GpuContext(
        gpu_index=0, gpus=1, model=model, device=H100_SXM, query_settings={
            "client": _Client(), "sampling_params": object(),
            "documents": documents, "true_ids": [1], "capacity": CAPACITY,
            "filter_submission": "operator-at-a-time",
            "join_submission": "anchor-major",
            "map_submission": "operator-at-a-time",
            **settings,
        },
    ))
