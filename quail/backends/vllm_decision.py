"""Decision 2.0 checkpoints on vLLM's pooling runner.

The converted Qwen3 backbone runs as a vLLM pooling model. Its pooler
keeps each request's trailing rows across prefill chunks and, when the
prompt finishes, returns the decision head's fp32 option scores. The
backend reads an answer from the scores as the Quail readout does.
"""

from __future__ import annotations

import math
import time

ARCHITECTURE = "Qwen3DecisionModel"
TASK = "token_classify"
OFFSETS_KEY = "decision_offsets"


def register() -> None:
    """Register the pooling model class with vLLM's model registry."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, "quail.backends.vllm_decision_model:Qwen3DecisionModel")


def pooling_params(offsets, *, read_cache: bool = True):
    """Build the parameters of one request whose head reads rows at offsets.

    Args:
        offsets: Distances before the last row: one per option, then 0.
        read_cache: Whether the request may start from prefix-cached KV.
            vLLM skips the cache for token pooling unless told otherwise.
    """
    from vllm import PoolingParams

    return PoolingParams(task=TASK, skip_reading_prefix_cache=not read_cache,
                         extra_kwargs={OFFSETS_KEY: [int(o) for o in offsets]})


def pool_decision_rows(torch, head, chunks, finished, params, states) -> list:
    """Score each finished request's options from its trailing rows.

    Each request keeps its last rows across prefill chunks. When its
    prompt finishes, the head scores the rows at the request's offsets
    (one per option, then the last row). A request whose rows came from
    the prefix cache gets NaN scores, so the caller reruns it.

    Args:
        torch: The torch module.
        head: The model's DecisionHead.
        chunks: Per request, the hidden rows this step computed.
        finished: Per request, whether its prompt is complete.
        params: Per request, its PoolingParams.
        states: Per request, vLLM's PoolingStates.

    Returns:
        Per request, float32 option scores, or None while it prefills.
    """
    out = [None] * len(chunks)
    groups = {}
    for index, (chunk, param, state, done) in enumerate(zip(
            chunks, params, states, finished)):
        offsets = (param.extra_kwargs or {}).get(OFFSETS_KEY)
        if offsets is None:    # vLLM's profiling and warmup requests
            if done:
                out[index] = chunk[-1, :1].float()
            continue
        need = max(offsets) + 1
        cache = state.hidden_states_cache
        # the runner reuses the hidden-state buffer on the next step
        cache.append(chunk[-need:].clone())
        if len(cache) > 1:
            cache[:] = [torch.cat(cache)[-need:]]
        if not done:
            continue
        rows = cache[0]
        state.clean()
        if rows.shape[0] < need:
            out[index] = torch.full((len(offsets) - 1,), float("nan"),
                                    device=rows.device)
            continue
        last = rows.shape[0] - 1
        groups.setdefault(tuple(offsets), []).append(
            (index, rows[[last - offset for offset in offsets]]))
    for members in groups.values():
        picked = torch.stack([rows for _, rows in members])
        scores = head.scores(picked[:, :-1], picked[:, -1])
        for (index, _), row in zip(members, scores):
            out[index] = row
    return out


def missing_rows(output) -> bool:
    """Whether the request's trailing rows came from the prefix cache."""
    return bool(output.outputs.data.isnan().any())


def decision_bit(output) -> int:
    """Read a yes/no answer: 1 when Yes scores above No."""
    scores = output.outputs.data
    return int(scores[1] > scores[0])


def choice_index(output) -> int:
    """Read the index of the highest-scoring option; ties go to the first."""
    return int(output.outputs.data.argmax())


class _RetryingEngine:
    """An LLMEngine view that reruns a request missing its trailing rows.

    A rerun skips the prefix cache and is returned under the original
    request id.
    """

    def __init__(self, client):
        self.client = client
        self.engine = client.llm.llm_engine
        self.prompts = {}
        self.retries = {}

    def add_request(self, request_id, prompt, params):
        self.prompts[request_id] = (prompt, params)
        self.engine.add_request(request_id, prompt, params)

    def step(self):
        ready = []
        for output in self.engine.step():
            if not output.finished:
                continue
            request_id = self.retries.pop(output.request_id, output.request_id)
            prompt, params = self.prompts.pop(request_id)
            if missing_rows(output):
                if params.skip_reading_prefix_cache:
                    raise RuntimeError(f"request {request_id} has no trailing rows")
                retry = f"{request_id}-recompute"
                self.client.recomputed += 1
                self.retries[retry] = request_id
                self.prompts[request_id] = (prompt, _uncached(params))
                self.engine.add_request(retry, prompt, self.prompts[request_id][1])
                continue
            output.request_id = request_id
            ready.append(output)
        return ready


def _uncached(params):
    clone = params.clone()
    clone.skip_reading_prefix_cache = True
    return clone


class VLLMDecisionClient:
    """The request operations of a decision model on one vLLM LLM.

    Prompts are token ids: the layout tokenizes each segment apart, so
    rendering text and tokenizing it again would change the prompt.
    """

    accepts_text = False

    def __init__(self, llm, capacity: dict):
        self.llm = llm
        self.capacity = capacity
        self.recomputed = 0
        self._counter = 0

    def generate(self, prompts, params, use_tqdm=False):
        """Pool each prompt; return the outputs in prompt order."""
        del use_tqdm
        engine = _RetryingEngine(self)
        ids = []
        for prompt in prompts:
            self._counter += 1
            ids.append(f"d{self._counter}")
            engine.add_request(ids[-1], prompt, params)
        done = {}
        while len(done) < len(ids):
            for output in engine.step():
                done[output.request_id] = output
        return [done[request_id] for request_id in ids]

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()

    def run_filter_chain(self, params, body_ids, question_ids, read_answer, *,
                         tag="q", body_texts=None, question_texts=None):
        """Pipeline filter stages through the engine's step loop."""
        from quail.backends.request_scheduling import run_filter_chain

        if body_texts is not None or question_texts is not None:
            raise ValueError("decision prompts are submitted as token ids")
        return run_filter_chain(
            _RetryingEngine(self), params, body_ids, question_ids,
            self.capacity["kv_cache_size_tokens"], tag=tag,
            read_answer=read_answer, block_size=self.capacity["block_size"],
            max_num_seqs=self.capacity["max_num_seqs"])


def boot_decision(spec, llm_kwargs: dict) -> tuple[dict, dict]:
    """Load a decision checkpoint on the pooling runner.

    Each query sets its own pooling parameters from its plan. Prompts
    arrive as token ids, so vLLM's own tokenizer is never used.

    Args:
        spec: The decision model's ModelSpec.
        llm_kwargs: LLM constructor arguments beyond the model.

    Returns:
        The engine state and the boot record.
    """
    from vllm import LLM

    from quail.backends.quail.executor.model import checkpoint_path
    from quail.backends.vllm import _capacity

    register()
    path = checkpoint_path(spec.hf_name, spec.revision)
    llm_kwargs = {key: value for key, value in llm_kwargs.items()
                  if key != "tokenizer_mode"}
    started = time.perf_counter()
    llm = LLM(model=path, runner="pooling",
              hf_overrides={"architectures": [ARCHITECTURE]}, **llm_kwargs)
    boot_s = time.perf_counter() - started
    config = llm.llm_engine.vllm_config
    capacity = _capacity(llm)
    capacity.update(
        enable_prefix_caching=config.cache_config.enable_prefix_caching,
        enable_chunked_prefill=config.scheduler_config.enable_chunked_prefill,
        runner="pooling", architecture=ARCHITECTURE)
    if not capacity["enable_prefix_caching"]:
        raise RuntimeError("vLLM turned prefix caching off for the decision model")
    client = VLLMDecisionClient(llm, capacity)
    params = pooling_params((1, 0))
    warm = client.generate([{"prompt_token_ids": [0, 0]}], params)
    if not all(math.isfinite(float(v)) for v in warm[0].outputs.data):
        raise RuntimeError("the decision pooler returned no scores")
    return (
        {"client": client, "sampling_params": params, "capacity": capacity},
        {"kind": "cold", "llm_init_s": round(boot_s, 2), "weight_load_s": None,
         "kv_profile_s": None, "boot_s": round(boot_s, 2)},
    )
