"""SGLang model backend."""

from __future__ import annotations

import time

from quail.backends.request import (
    RequestModelExecution,
    execute_request_graph,
    plan_request_backend,
)
from quail.planning import SupportResult


MAX_SEQUENCES = 4_096
MAX_BATCHED_TOKENS = 25_305
PAGE_SIZE = 16
CHUNKED_PREFILL_TOKENS = (
    MAX_BATCHED_TOKENS // PAGE_SIZE
) * PAGE_SIZE
MEM_FRACTION_STATIC = 0.76
TRUE_FALSE_LOGIT_BIAS = 1_000.0
SUBMIT_SLICE = 16_384


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _RequestOutput:
    def __init__(self, prompt_token_ids, token_ids, cached_tokens):
        self.outputs = [_Completion(token_ids)]
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = cached_tokens


class SGLangClient:
    """Expose the request operations used by the SGLang backend."""

    def __init__(self, engine, capacity):
        self.engine = engine
        self.block_size = capacity["block_size"]
        self.filter_budget_tokens = capacity["kv_cache_size_tokens"]
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

    def run_pipelined_filter_chain(
        self,
        sampling_params,
        body_ids,
        question_ids,
        true_ids,
        *,
        tag="q",
    ):
        """Advance each live document by one filter stage per wave."""
        del tag
        longest_tail = max(len(question) for question in question_ids)
        request_sizes = [
            len(body) + longest_tail + 1 for body in body_ids
        ]
        mean_request = sum(request_sizes) // max(1, len(request_sizes))
        document_cap = min(
            max(1, self.filter_budget_tokens // mean_request),
            MAX_SEQUENCES,
        )
        answers = {}
        survivors = []
        requests = prompt_tokens = cached_tokens = 0
        active = []
        next_document = 0
        started = time.perf_counter()
        while active or next_document < len(body_ids):
            while (
                next_document < len(body_ids)
                and len(active) < document_cap
            ):
                active.append((next_document, 0))
                next_document += 1
            prompts = [
                {
                    "prompt_token_ids": (
                        body_ids[document] + question_ids[stage]
                    )
                }
                for document, stage in active
            ]
            outputs = self.generate(prompts, sampling_params)
            advanced = []
            for (document, stage), output in zip(active, outputs):
                requests += 1
                prompt_tokens += len(output.prompt_token_ids)
                cached_tokens += output.num_cached_tokens
                tokens = output.outputs[0].token_ids
                answer = int(bool(
                    tokens and int(tokens[0]) in true_ids
                ))
                answers[(document, stage + 1)] = answer
                if answer and stage + 1 < len(question_ids):
                    advanced.append((document, stage + 1))
                elif answer:
                    survivors.append(document)
            active = advanced
        return {
            "wall": time.perf_counter() - started,
            "survivors": sorted(survivors),
            "answers": answers,
            "requests": requests,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "doc_cap": document_cap,
            "budget_tokens": self.filter_budget_tokens,
            "block_size": self.block_size,
            "max_num_seqs": MAX_SEQUENCES,
        }

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


def _boot(model_name: str, allowed_ids: list[int]) -> tuple[dict, dict]:
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


class SGLangBackend:
    """Plan and run pipelined SGLang requests."""

    name = "pipelined_sglang"
    runtime_package = "sglang==0.5.18"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if model.name not in {"qwen3-4b-fp8", "qwen3-32b-fp8"}:
            return SupportResult.reject(
                f"SGLang does not support model {model.name!r}"
            )
        if device.name != "h100-sxm":
            return SupportResult.reject(
                f"SGLang does not support device {device.name!r}"
            )
        if gpu_count != 1:
            return SupportResult.reject(
                "the SGLang backend uses one model copy on one GPU"
            )
        return SupportResult.accept()

    def plan(self, region, context):
        return plan_request_backend(
            region,
            context,
            backend_name=self.name,
            filter_submission="pipelined",
            join_submission="suffix-major-tiled",
        )

    def start(self, context):
        return RequestModelExecution(context)

    def execute_request(self, context):
        envelope = context.request.plan
        model = context.registry.model(envelope["model"])
        state_key = ("request-engine", "sglang", model.name)
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
