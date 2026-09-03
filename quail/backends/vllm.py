"""vLLM model backends."""

from __future__ import annotations

import time
from dataclasses import dataclass

from quail.backends.request import (
    RequestModelExecution,
    execute_request_graph,
    plan_request_backend,
)
from quail.planning import SupportResult


MAX_BATCHED_TOKENS = 25_305
MAX_SEQUENCES = 4_096
GPU_MEMORY_UTILIZATION = 0.91
CUDA_GRAPH_CAPTURE_SIZE = 8_192


def _capacity(llm) -> dict:
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    tokens = getattr(cache, "kv_cache_size_tokens", None)
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    if tokens is None and blocks is not None and block_size is not None:
        tokens = blocks * block_size
    if tokens is None or block_size is None:
        raise RuntimeError("vLLM did not report its KV capacity")
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None if blocks is None else int(blocks),
        "block_size": int(block_size),
        "max_num_seqs": int(config.scheduler_config.max_num_seqs),
        "kv_cache_dtype": str(cache.cache_dtype),
    }


def _boot(model_name: str, allowed_ids: list[int]) -> tuple[dict, dict]:
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=model_name,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_num_seqs=MAX_SEQUENCES,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,
        disable_log_stats=True,
        compilation_config={
            "cudagraph_capture_sizes": [CUDA_GRAPH_CAPTURE_SIZE]
        },
    )
    boot_s = time.perf_counter() - started
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed_ids,
    )
    llm.generate(
        [{"prompt_token_ids": allowed_ids}],
        sampling_params,
        use_tqdm=False,
    )
    return (
        {
            "client": llm,
            "sampling_params": sampling_params,
            "capacity": _capacity(llm),
        },
        {
            "kind": "cold",
            "llm_init_s": round(boot_s, 2),
            "weight_load_s": None,
            "kv_profile_s": None,
            "boot_s": round(boot_s, 2),
        },
    )


@dataclass(frozen=True)
class VLLMBackend:
    """Plan and run one vLLM submission strategy."""

    name: str
    filter_submission: str
    join_submission: str = "anchor-major"
    runtime_package: str = "vllm==0.26.0"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if model.name not in {"qwen3-4b-fp8", "qwen3-32b-fp8"}:
            return SupportResult.reject(
                f"vLLM does not support model {model.name!r}"
            )
        if device.name != "h100-sxm":
            return SupportResult.reject(
                f"vLLM does not support device {device.name!r}"
            )
        if gpu_count != 1:
            return SupportResult.reject(
                "the vLLM request backends use one model copy on one GPU"
            )
        return SupportResult.accept()

    def plan(self, region, context):
        return plan_request_backend(
            region,
            context,
            backend_name=self.name,
            filter_submission=self.filter_submission,
            join_submission=self.join_submission,
        )

    def start(self, context):
        return RequestModelExecution(context)

    def execute_request(self, context):
        envelope = context.request.plan
        model = context.registry.model(envelope["model"])
        state_key = ("request-engine", "vllm", model.name)
        engine_state = context.runtime_state.get(state_key)
        if engine_state is None:
            allowed_ids = sorted(set(
                envelope["settings"]["true_ids"]
            ) | set(envelope["settings"]["false_ids"]))
            engine_state, boot = _boot(model.hf_name, allowed_ids)
            context.runtime_state[state_key] = engine_state
        else:
            boot = {
                "kind": "warm",
                "llm_init_s": 0.0,
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": 0.0,
            }
        return execute_request_graph(
            context,
            self,
            engine_state,
            boot,
        )


def stock_vllm_backend() -> VLLMBackend:
    """Return stock vLLM with separate requests per filter stage."""
    return VLLMBackend(
        name="stock_vllm",
        filter_submission="stage-major",
    )


def pipelined_vllm_backend() -> VLLMBackend:
    """Return vLLM with per document filter pipelining."""
    return VLLMBackend(
        name="pipelined_vllm",
        filter_submission="pipelined",
    )
