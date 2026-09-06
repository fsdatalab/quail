"""vLLM engine adapter and the two vLLM request backends."""

from __future__ import annotations

import time

from quail.backends.request import RequestBackend
from quail.backends.request_scheduling import (
    MAX_BATCHED_TOKENS,
    MAX_SEQUENCES,
    run_filter_chain,
)

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


class VLLMClient:
    """The request operations the backends need from one vLLM LLM."""

    def __init__(self, llm, capacity: dict):
        self.llm = llm
        self.capacity = capacity

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()

    def run_filter_chain(self, sampling_params, body_ids, question_ids,
                         true_ids, *, tag="q"):
        """Pipeline filter stages through the engine's step loop."""
        return run_filter_chain(
            self.llm.llm_engine,
            sampling_params,
            body_ids,
            question_ids,
            self.capacity["kv_cache_size_tokens"],
            tag=tag,
            true_ids=true_ids,
            block_size=self.capacity["block_size"],
            max_num_seqs=self.capacity["max_num_seqs"],
        )


class VLLMEngine:
    """Boot vLLM for a request backend."""

    kind = "vllm"
    label = "vLLM"
    runtime_package = "vllm==0.26.0"

    def boot(self, model_name: str, allowed_ids: list[int]) -> tuple[dict, dict]:
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
        capacity = _capacity(llm)
        client = VLLMClient(llm, capacity)
        client.generate(
            [{"prompt_token_ids": allowed_ids}], sampling_params
        )
        return (
            {
                "client": client,
                "sampling_params": sampling_params,
                "capacity": capacity,
            },
            {
                "kind": "cold",
                "llm_init_s": round(boot_s, 2),
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": round(boot_s, 2),
            },
        )


def stock_vllm_backend() -> RequestBackend:
    """Return stock vLLM with operator-at-a-time filter execution."""
    return RequestBackend(
        name="stock_vllm",
        engine=VLLMEngine(),
        filter_submission="operator-at-a-time",
    )


def pipelined_vllm_backend() -> RequestBackend:
    """Return vLLM with per document filter pipelining."""
    return RequestBackend(
        name="pipelined_vllm",
        engine=VLLMEngine(),
        filter_submission="pipelined",
    )
