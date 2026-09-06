"""SGLang engine adapter and the pipelined SGLang request backend."""

from __future__ import annotations

import time

from quail.backends.request import RequestBackend
from quail.backends.request_scheduling import (
    MAX_BATCHED_TOKENS,
    MAX_SEQUENCES,
    run_filter_chain_waves,
)

PAGE_SIZE = 16
CHUNKED_PREFILL_TOKENS = (
    MAX_BATCHED_TOKENS // PAGE_SIZE
) * PAGE_SIZE
MEM_FRACTION_STATIC = 0.76
TRUE_FALSE_LOGIT_BIAS = 1_000.0
# One generate() call per request keeps the driver process busy with
# one asyncio task each; a slice bounds that so the Modal health
# heartbeat thread keeps running.
SUBMIT_SLICE = 16_384


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _RequestOutput:
    """The vLLM output shape the shared scheduling code reads."""

    def __init__(self, prompt_token_ids, token_ids, cached_tokens):
        self.outputs = [_Completion(token_ids)]
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = cached_tokens


class SGLangClient:
    """The request operations the backends need from one SGLang engine."""

    def __init__(self, engine, capacity):
        self.engine = engine
        self.capacity = capacity
        # Half the KV pool bounds a join tile: in-flight suffixes and
        # the previous tile's leftovers share the pool with the tile's
        # anchors.
        self.join_tile_budget_tokens = (
            capacity["kv_cache_size_tokens"] // 2
        )

    def generate(self, prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        outputs = []
        for start in range(0, len(prompts), SUBMIT_SLICE):
            outputs.extend(self._generate_slice(
                prompts[start:start + SUBMIT_SLICE],
                sampling_params,
            ))
        return outputs

    def _generate_slice(self, prompts, sampling_params):
        input_ids = [prompt["prompt_token_ids"] for prompt in prompts]
        raw = self.engine.generate(
            input_ids=input_ids,
            sampling_params=dict(sampling_params),
        )
        if isinstance(raw, dict):
            raw = [raw]
        if len(raw) != len(input_ids):
            raise RuntimeError(
                f"SGLang returned {len(raw)} outputs for "
                f"{len(input_ids)} requests"
            )
        outputs = []
        for prompt, result in zip(input_ids, raw):
            metadata = result["meta_info"]
            outputs.append(_RequestOutput(
                prompt,
                list(result.get("output_ids") or []),
                int(metadata.get("cached_tokens") or 0),
            ))
        return outputs

    def run_filter_chain(self, sampling_params, body_ids, question_ids,
                         true_ids, *, tag="q"):
        """Pipeline filter stages in waves of blocking generate calls."""
        del tag
        # the measured SGLang runs sized admission without rounding to
        # pages, so the cap stays unrounded here
        return run_filter_chain_waves(
            self,
            sampling_params,
            body_ids,
            question_ids,
            self.capacity["kv_cache_size_tokens"],
            true_ids=true_ids,
            block_size=1,
            max_num_seqs=MAX_SEQUENCES,
        )

    def reset_prefix_cache(self):
        result = self.engine.flush_cache()
        if isinstance(result, bool):
            return result
        return bool(getattr(result, "success", True))


def _capacity(engine) -> dict:
    info = engine.get_server_info()
    tokens = info.get("max_total_num_tokens")
    page_size = info.get("page_size")
    if tokens is None or page_size is None:
        raise RuntimeError("SGLang did not report its KV capacity")
    running = info.get("max_running_requests")
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None,
        "block_size": int(page_size),
        "max_num_seqs": None if running is None else int(running),
        "kv_cache_dtype": str(info.get("kv_cache_dtype")),
    }


class SGLangEngine:
    """Boot SGLang for a request backend."""

    kind = "sglang"
    label = "SGLang"
    runtime_package = "sglang==0.5.18"

    def boot(self, model_name: str, allowed_ids: list[int]) -> tuple[dict, dict]:
        import sglang as sgl

        started = time.perf_counter()
        engine = sgl.Engine(
            model_path=model_name,
            mem_fraction_static=MEM_FRACTION_STATIC,
            max_running_requests=MAX_SEQUENCES,
            chunked_prefill_size=CHUNKED_PREFILL_TOKENS,
            max_prefill_tokens=CHUNKED_PREFILL_TOKENS,
            page_size=PAGE_SIZE,
            skip_tokenizer_init=True,
            disable_radix_cache=False,
            disable_prefill_cuda_graph=True,
            log_level="warning",
        )
        boot_s = time.perf_counter() - started
        capacity = _capacity(engine)
        client = SGLangClient(engine, capacity)
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "logit_bias": {
                str(token): TRUE_FALSE_LOGIT_BIAS for token in allowed_ids
            },
        }
        client.generate(
            [{"prompt_token_ids": allowed_ids}],
            sampling_params,
        )
        return (
            {
                "client": client,
                "sampling_params": sampling_params,
                "capacity": capacity,
                "close": getattr(engine, "shutdown", None),
            },
            {
                "kind": "cold",
                "llm_init_s": round(boot_s, 2),
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": round(boot_s, 2),
            },
        )


def pipelined_sglang_backend() -> RequestBackend:
    """Return SGLang with filter waves and suffix major tiled joins."""
    return RequestBackend(
        name="pipelined_sglang",
        engine=SGLangEngine(),
        filter_submission="pipelined",
        join_submission="suffix-major-tiled",
    )
