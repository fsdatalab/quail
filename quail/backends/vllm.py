"""vLLM engine adapter and the vLLM request backends."""

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
# vLLM caps a diffusion model at 8 sequences per step whenever the
# setting is 128 or more, sized for its 256-row canvas; one row needs
# no cap, so stay just under the trigger.
DIFFUSION_SEQUENCES = 127
# top logprobs read at the canvas row; the answer words rank within
# the top 500 on 511 of 512 probed reviews and within 1,718 on all
# (/results/ablations/diffusion_gemma_readout_probe_k5000.json)
DIFFUSION_LOGPROBS = 500
# generated tokens a longer canvas gets; the answer word is read
# from the text
DIFFUSION_TEXT_TOKENS = 16


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
    scheduler = config.scheduler_config
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None if blocks is None else int(blocks),
        "block_size": int(block_size),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_model_len": int(config.model_config.max_model_len),
        "gpu_memory_utilization": float(cache.gpu_memory_utilization),
        "kv_cache_dtype": str(cache.cache_dtype),
    }


def sampling_kwargs(allowed_ids: list[int], canvas_tokens: int = 0) -> dict:
    """SamplingParams arguments for one greedy answer token.

    A diffusion model's sampler takes no temperature, min_tokens, or
    allowed_token_ids. With a one-row canvas it commits that row after
    one pass and returns its top logprobs, which the reader ranks TRUE
    against FALSE; a longer canvas writes free text, and the reader
    finds the answer word.
    """
    if canvas_tokens == 1:
        return {"max_tokens": 1, "logprobs": DIFFUSION_LOGPROBS}
    if canvas_tokens:
        return {"max_tokens": DIFFUSION_TEXT_TOKENS}
    return {"temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
            "allowed_token_ids": allowed_ids}


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
    """Boot vLLM for a request backend, with Quail's tuned engine settings."""

    kind = "vllm"
    label = "vLLM"
    runtime_package = "vllm==0.26.0"

    def llm_kwargs(self, spec) -> dict:
        """Return the LLM constructor arguments beyond the model name.

        A spec with its own chunk cap (a mixture-of-experts model)
        batches at least that many tokens per step. A diffusion model
        gets the spec's canvas; a one-row canvas takes one denoising
        step, so a request is one prefill pass plus one row.
        """
        batched = max(MAX_BATCHED_TOKENS, spec.chunk_cap_tokens)
        sequences = MAX_SEQUENCES
        kwargs = {
            "max_num_batched_tokens": batched,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "compilation_config": {
                "cudagraph_capture_sizes": [CUDA_GRAPH_CAPTURE_SIZE]
            },
        }
        if spec.canvas_tokens:
            diffusion = {"canvas_length": spec.canvas_tokens}
            if spec.canvas_tokens == 1:
                diffusion["max_denoising_steps"] = 1
                sequences = DIFFUSION_SEQUENCES
                # vLLM refuses a request for more logprobs than this
                kwargs["max_logprobs"] = DIFFUSION_LOGPROBS
            kwargs["diffusion_config"] = diffusion
        kwargs["max_num_seqs"] = sequences
        return kwargs

    def boot(self, spec, allowed_ids: list[int]) -> tuple[dict, dict]:
        """Load the spec's model and return the engine state and boot record."""
        from vllm import LLM, SamplingParams

        started = time.perf_counter()
        llm = LLM(model=spec.hf_name, **self.llm_kwargs(spec))
        boot_s = time.perf_counter() - started
        sampling_params = SamplingParams(
            **sampling_kwargs(allowed_ids, spec.canvas_tokens))
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


class DefaultVLLMEngine(VLLMEngine):
    """Boot vLLM with every engine setting at its default.

    Only the model name is passed, as `vllm serve Qwen/Qwen3-4B-FP8`
    would. The engine kind differs from `VLLMEngine` so this engine
    never shares a loaded model with the tuned vLLM backends.
    """

    kind = "dumb_vllm"
    label = "vLLM with default settings"

    def llm_kwargs(self, spec) -> dict:
        return {}


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


def dumb_vllm_backend() -> RequestBackend:
    """Return vLLM at its default settings, operator-at-a-time filters."""
    return RequestBackend(
        name="dumb_vllm",
        engine=DefaultVLLMEngine(),
        filter_submission="operator-at-a-time",
    )
