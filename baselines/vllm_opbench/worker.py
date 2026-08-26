"""vLLM-opbench worker — offline batched-inference vLLM on Modal (H100).

The worker attaches to the existing "quail-milestone1" Modal app and
uses the same app lifecycle as the other benchmark cells.

Ported from SQPE's vllm_worker.py (/Users/adhariya/SQPE, a separate
benchmarking project), with two deliberate departures:

  - Prompts are raw prompt_token_ids (matching quail's own engine and
    baselines/stock.py), not chat-template text via llm.chat().
    SQPE's chat-template-based shared-prefix padding (the vLLM issue
    #40696 block-boundary workaround) is NOT ported - it's a text/
    chat-template-specific mechanism, and doesn't carry over cleanly
    to raw-token prompts. Prefix-cache reuse here relies on vLLM's own
    cache plus request ordering (anchor-major for joins, same as
    stock.py's run_join_grouped), not explicit block padding. This is
    a known simplification versus SQPE's original - full parity would
    mean re-deriving the padding math in token space, not text space.
  - The answer is a constrained TRUE/FALSE token id (allowed_token_ids
    + max_tokens=1), not free-text "true"/"false" parsing.

Prometheus metric snapshot/diff, the gauge-polling timeseries, the
KV-cache regret oracle, and the nsys hookup are otherwise a straight
port - all of that is generic vLLM-introspection code with no
SQPE-specific coupling.
"""

import time

import modal

from .config import (
    APP_NAME,
    DISABLE_LOG_STATS,
    ENABLE_CHUNKED_PREFILL,
    GPU_MEMORY_UTILIZATION,
    LONG_PREFILL_TOKEN_THRESHOLD,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS_BY_GPU,
    MODEL_NAMES,
    NSYS_OUTPUT_DIR,
    NSYS_VOLUME_NAME,
    PREFIX_CACHE_BLOCK_SIZE,
    SCHEDULER_RESERVE_FULL_ISL,
    TENSOR_PARALLEL_SIZE,
)
from . import gpu_profiling

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04",
                              add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.26.0", "transformers>=5.2.0",
                    "huggingface_hub[hf_transfer]")
    .apt_install("wget", "gnupg")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/devtools/repos/"
        "ubuntu2204/amd64/nvidia.pub -O /tmp/nvidia.pub && gpg --dearmor "
        "-o /usr/share/keyrings/nvidia-devtools.gpg /tmp/nvidia.pub",
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] "
        "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/"
        "amd64/ /' > /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-get update && apt-get install -y nsight-systems-cli || "
        "echo 'nsys install failed -- gpu_profiling.py will no-op if "
        "PROFILE_GPU is set'")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
         # FlashInfer reserves another 1.5 GiB after vLLM sizes the KV cache.
         "VLLM_USE_FLASHINFER_SAMPLER": "0",
         # Without this, PyTorch's CUDA caching allocator needs an exact
         # contiguous block and can fail even when enough total free
         # memory exists, just fragmented - the allocator OOM warnings
         # seen at large batch sizes (60k+ pending requests) regardless
         # of model size. stock_join_imdb.py's baseline already sets
         # this; vllm_opbench's image was missing it.
         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # add_local_* must be last: Modal requires it after every build step.
    # config.py/gpu_profiling.py are submodules of the baselines package
    # here (unlike SQPE's original flat repo layout), so the whole
    # package needs mounting, not the two files by their bare names -
    # matching how quail/runtime/worker.py mounts "quail" whole.
    .add_local_python_source("baselines")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache",
                                      create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
nsys_traces_vol = modal.Volume.from_name(NSYS_VOLUME_NAME,
                                         create_if_missing=True)


# --------------------------------------------------- Prometheus snapshot/diff

def snapshot_vllm_metrics(llm) -> dict:
    """Normalize vLLM's typed metric objects into
    {sample_name: [{labels, value}]}. Captures every object
    llm.get_metrics() returns, including Info/config-style metrics
    (e.g. vllm:cache_config_info) via the generic else branch, so
    nothing get_metrics() exposes is dropped."""
    from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

    snapshot = {}

    def add(sample_name, labels, value):
        snapshot.setdefault(sample_name, []).append(
            {"labels": dict(labels), "value": value})

    for metric in llm.get_metrics():
        if isinstance(metric, (Counter, Gauge)):
            add(metric.name, metric.labels, metric.value)
        elif isinstance(metric, Vector):
            for i, v in enumerate(metric.values):
                add(metric.name, {**metric.labels, "index": str(i)}, v)
        elif isinstance(metric, Histogram):
            add(f"{metric.name}_count", metric.labels, metric.count)
            add(f"{metric.name}_sum", metric.labels, metric.sum)
            for le, count in metric.buckets.items():
                add(f"{metric.name}_bucket", {**metric.labels, "le": str(le)},
                    count)
        else:
            add(metric.name, getattr(metric, "labels", {}) or {},
                getattr(metric, "value", 1))
    return snapshot


def snapshot_gauge_metrics(llm) -> dict:
    """Gauge-only snapshot: the one metric type whose useful signal is
    an instantaneous, mid-batch value (queue depth, KV occupancy),
    invisible to a before/after diff since the engine reads idle at
    both of those instants."""
    from vllm.v1.metrics.reader import Gauge

    snapshot = {}
    for metric in llm.get_metrics():
        if isinstance(metric, Gauge):
            snapshot.setdefault(metric.name, []).append(
                {"labels": dict(metric.labels), "value": metric.value})
    return snapshot


def diff_metrics(before: dict, after: dict) -> dict:
    value_only = {"vllm:num_requests_running", "vllm:kv_cache_usage_perc",
                  "vllm:cache_config_info"}
    result = {}
    for name, after_samples in after.items():
        before_samples = {tuple(sorted(s["labels"].items())): s["value"]
                          for s in before.get(name, [])}
        entries = []
        for s in after_samples:
            key = tuple(sorted(s["labels"].items()))
            before_val = before_samples.get(key, 0)
            val = s["value"]
            if (name in value_only or not isinstance(val, (int, float))
                    or isinstance(val, bool)):
                entries.append({"labels": s["labels"], "value": val})
            else:
                entries.append({"labels": s["labels"],
                                "delta": val - before_val})
        result[name] = entries
    return result


def _get_ts(m, *names):
    for n in names:
        v = getattr(m, n, None)
        if v is not None:
            return v
    return None


def _to_mono(ts, epoch_to_mono):
    if ts is None:
        return None
    if ts > 1e6:
        return ts - epoch_to_mono
    return ts


def _get_resolved_block_size(llm) -> int:
    """The actual resolved KV-cache block size from vLLM's own
    vllm:cache_config_info metric, never the requested value."""
    for metric in llm.get_metrics():
        if metric.name == "vllm:cache_config_info":
            bs = metric.labels.get("block_size")
            if bs is not None:
                return int(bs)
    raise RuntimeError(
        "Could not resolve block_size from vllm:cache_config_info metric")


# ------------------------------------------------------ KV-cache regret oracle

def _chain_hashes(token_ids: list[int], block_size: int) -> list:
    """One rolling hash per whole block boundary in token_ids - the
    same chain-hash structure vLLM's own prefix cache uses to key a
    block on everything that precedes it, not just its own contents."""
    chain = []
    h = 0
    for i in range(0, len(token_ids) - len(token_ids) % block_size, block_size):
        h = hash((h, tuple(token_ids[i:i + block_size])))
        chain.append(h)
    return chain


def _compute_oracle_regret_from_token_ids(all_token_ids: list[list[int]],
                                          block_size: int) -> dict:
    """How many prefix-cache blocks WOULD hit under an infinite,
    never-evicting cache, vs. how many vLLM's real (finite, arrival-
    order) cache actually hit. Takes the REAL token id sequences
    straight from each output's own o.prompt_token_ids, not a
    re-tokenization - o.prompt_token_ids is the literal sequence the
    engine already processed, no reconstruction risk.

    With an infinite cache, processing order doesn't change the total
    hit count - each unique block-hash-chain is computed once (a
    miss) regardless of which request reaches it first, so this only
    needs one pass in arrival order, not a search over orderings."""
    seen: set = set()
    per_request = []
    total_hit_blocks = 0
    total_query_blocks = 0

    for token_ids in all_token_ids:
        chain = _chain_hashes(token_ids, block_size)
        hits = 0
        still_matching = True
        for h in chain:
            if still_matching and h in seen:
                hits += 1
            else:
                still_matching = False
                seen.add(h)
        per_request.append({"n_tokens": len(token_ids),
                            "oracle_query_blocks": len(chain),
                            "oracle_hit_blocks": hits})
        total_hit_blocks += hits
        total_query_blocks += len(chain)

    total_tokens = sum(len(ids) for ids in all_token_ids)
    return {
        "block_size": block_size,
        "oracle_hit_blocks": total_hit_blocks,
        "oracle_query_blocks": total_query_blocks,
        "oracle_hit_rate": ((total_hit_blocks * block_size) / total_tokens
                            if total_tokens else None),
        "oracle_hit_tokens": total_hit_blocks * block_size,
        "oracle_query_tokens": total_tokens,
        "per_request": per_request,
    }


# ------------------------------------------------------------------- LLM build

def _build_llm(model_name: str, quantization: str, enable_prefix_caching: bool,
               max_num_seqs: int, block_size: int = PREFIX_CACHE_BLOCK_SIZE,
               max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS):
    from vllm import LLM
    kwargs = dict(
        model=model_name,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        long_prefill_token_threshold=LONG_PREFILL_TOKEN_THRESHOLD,
        scheduler_reserve_full_isl=SCHEDULER_RESERVE_FULL_ISL,
        disable_log_stats=DISABLE_LOG_STATS,
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        # matches quail's own POOL_FRACTION (budgets.py) - the baseline
        # must get the analytically equivalent memory budget, not vLLM's
        # own default (0.9)
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    if quantization == "fp8":
        # weight quantization only. KV cache dtype is a separate,
        # decoupled choice in quail's own engine, which always uses
        # bf16 KV regardless of weight dtype (commit f7bc656, "Always
        # use bf16 KV; drop the planner dtype argmin and q_kv") - so
        # the baseline must NOT tie kv_cache_dtype to this flag too,
        # or it's comparing against a KV dtype quail's engine doesn't
        # actually use.
        kwargs["quantization"] = "fp8"
    return LLM(**kwargs)


def _warmup_impl(llm, true_ids: list[int], false_ids: list[int]) -> dict:
    from vllm import SamplingParams
    allowed = sorted(set(true_ids) | set(false_ids))
    llm.generate(
        [{"prompt_token_ids": allowed}],
        SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=allowed),
        use_tqdm=False)
    return {"warmed": True}


def _reset_impl(llm) -> dict:
    return {"reset": bool(llm.reset_prefix_cache())}


def _poll_metrics_loop(llm, t0, stop_event, poll_interval_s, timeseries):
    """Background thread, gauge-only metrics every poll_interval_s -
    the only way queue depth / KV occupancy get seen mid-batch (the
    before/after snapshot sees the engine idle at both instants)."""
    while not stop_event.is_set():
        try:
            snap = snapshot_gauge_metrics(llm)
        except Exception as e:
            snap = {"_poll_error": str(e)}
        timeseries.append({"t_offset_s": time.perf_counter() - t0,
                           "metrics": snap})
        stop_event.wait(poll_interval_s)


def _generate_batch_impl(llm, gpu, quantization, max_num_seqs,
                         prompt_token_ids: list[list[int]],
                         true_ids: list[int], false_ids: list[int],
                         max_tokens: int,
                         poll_interval_s: float = 0.05,
                         do_profile: bool = False,
                         profile_name: str | None = None,
                         block_size: int | None = None) -> dict:
    from vllm import SamplingParams
    import threading

    true_set, false_set = set(true_ids), set(false_ids)
    allowed_token_ids = sorted(true_set | false_set)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens,
                        allowed_token_ids=allowed_token_ids)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    metrics_before = snapshot_vllm_metrics(llm)

    timeseries: list[dict] = []
    stop_event = threading.Event()
    t0 = time.perf_counter()
    poll_thread = threading.Thread(target=_poll_metrics_loop,
                                   args=(llm, t0, stop_event, poll_interval_s,
                                         timeseries),
                                   daemon=True)
    poll_thread.start()

    trace_path = None
    with gpu_profiling.profiled_batch(do_profile, profile_name) as prof:
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        if prof.captured:
            trace_path = gpu_profiling.current_report_path()

    stop_event.set()
    poll_thread.join(timeout=max(0.5, poll_interval_s * 5))
    t1 = time.perf_counter()
    timeseries.append({"t_offset_s": t1 - t0,
                       "metrics": snapshot_gauge_metrics(llm)})

    metrics_after = snapshot_vllm_metrics(llm)
    vllm_metrics_delta = diff_metrics(metrics_before, metrics_after)
    epoch_to_mono = time.time() - time.monotonic()

    per_request = []
    real_prompt_token_ids: list[list[int]] = []
    for o in outputs:
        m = o.metrics
        real_prompt_token_ids.append(list(o.prompt_token_ids))
        arrival = _to_mono(_get_ts(m, "arrival_time", "arrival_ts"),
                          epoch_to_mono) if m else None
        scheduled = _to_mono(_get_ts(m, "first_scheduled_time",
                                    "scheduled_time", "scheduled_ts",
                                    "queued_ts"), epoch_to_mono) if m else None
        first_tok = _to_mono(_get_ts(m, "first_token_time", "first_token_ts"),
                             epoch_to_mono) if m else None
        finished = _to_mono(_get_ts(m, "finished_time", "finished_ts",
                                    "last_token_time", "last_token_ts"),
                            epoch_to_mono) if m else None
        toks = o.outputs[0].token_ids
        tok0 = int(toks[0]) if toks else None
        answer = (1 if tok0 in true_set else
                  0 if tok0 in false_set else None)
        per_request.append({
            "prompt_tokens": len(o.prompt_token_ids),
            "output_tokens": len(o.outputs[0].token_ids),
            "answer": answer,
            "arrival_time": arrival, "first_scheduled_time": scheduled,
            "first_token_time": first_tok, "finished_time": finished,
            "time_in_queue": (scheduled - arrival
                              if scheduled is not None and arrival is not None
                              else None),
            "ttft": (first_tok - arrival
                     if first_tok is not None and arrival is not None
                     else None),
            "e2e_latency": (finished - arrival
                            if finished is not None and arrival is not None
                            else None),
            "prefill_time": (first_tok - scheduled
                             if first_tok is not None and scheduled is not None
                             else None),
            "decode_time": (finished - first_tok
                            if finished is not None and first_tok is not None
                            else None),
        })

    # The resolved block size can't change after the engine is built, so
    # a caller that already has it (WorkerH100 caches it at boot) should
    # pass it through instead of paying a fresh get_metrics() walk here
    # on every batch.
    oracle_block_size = (block_size if block_size is not None
                         else _get_resolved_block_size(llm))
    oracle_regret = _compute_oracle_regret_from_token_ids(
        real_prompt_token_ids, oracle_block_size)

    return {
        "wall_time_s": t1 - t0,
        "n_prompts": len(prompt_token_ids),
        "gpu": gpu, "quantization": quantization, "max_num_seqs": max_num_seqs,
        "per_request": per_request,
        "vllm_metrics": vllm_metrics_delta,
        "timeseries": timeseries,
        "oracle_regret": oracle_regret,
        "trace_path": f"{trace_path}.nsys-rep" if trace_path else None,
    }


# --------------------------------------------------------------------- worker

@app.cls(
    image=vllm_image,
    gpu="H100!",
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache_vol,
             "/root/.cache/vllm": vllm_cache_vol,
             NSYS_OUTPUT_DIR: nsys_traces_vol},
)
class WorkerH100:
    GPU = "H100!"
    model: str = modal.parameter(default="qwen3-4b")   # key into MODEL_NAMES
    quantization: str = modal.parameter(default="fp8")
    enable_prefix_caching: bool = modal.parameter(default=True)
    block_size: int = modal.parameter(default=PREFIX_CACHE_BLOCK_SIZE)
    max_num_batched_tokens: int = modal.parameter(
        default=MAX_NUM_BATCHED_TOKENS)

    @modal.enter()
    def load(self):
        gpu_profiling.maybe_reexec_under_nsys()
        model_name = MODEL_NAMES[self.model]
        print(f"Loading {model_name} | gpu={self.GPU} | "
              f"quant={self.quantization} | "
              f"prefix_caching={self.enable_prefix_caching} | "
              f"block_size={self.block_size} | "
              f"max_num_batched_tokens={self.max_num_batched_tokens} | "
              f"under_nsys={gpu_profiling.under_nsys()}", flush=True)
        max_num_seqs = MAX_NUM_SEQS_BY_GPU.get(self.GPU, 256)
        self.llm = _build_llm(model_name, self.quantization,
                              self.enable_prefix_caching, max_num_seqs,
                              self.block_size, self.max_num_batched_tokens)
        self.max_num_seqs = max_num_seqs
        self._resolved_block_size = _get_resolved_block_size(self.llm)
        print(f"  [cache-layout] resolved_block_size="
              f"{self._resolved_block_size}", flush=True)

    @modal.method()
    def warmup(self, true_ids: list[int], false_ids: list[int]):
        return _warmup_impl(self.llm, true_ids, false_ids)

    @modal.method()
    def reset_prefix_cache(self):
        return _reset_impl(self.llm)

    @modal.method()
    def generate_batch(self, prompt_token_ids: list[list[int]],
                       true_ids: list[int], false_ids: list[int],
                       max_tokens: int = 1, poll_interval_s: float = 0.05,
                       do_profile: bool = False,
                       profile_name: str | None = None):
        return _generate_batch_impl(
            self.llm, self.GPU, self.quantization, self.max_num_seqs,
            prompt_token_ids, true_ids, false_ids, max_tokens,
            poll_interval_s, do_profile, profile_name,
            block_size=self._resolved_block_size)

    @modal.method()
    def generate_join_batch(self, prefixes: list[list[int]],
                            suffixes: list[list[int]],
                            true_ids: list[int], false_ids: list[int],
                            max_tokens: int = 1,
                            poll_interval_s: float = 0.05,
                            do_profile: bool = False,
                            profile_name: str | None = None):
        # Building the full cross product in the CPU orchestrator makes
        # Modal serialize hundreds of millions of repeated token ids.
        prompt_token_ids = [prefix + suffix
                            for prefix in prefixes for suffix in suffixes]
        return _generate_batch_impl(
            self.llm, self.GPU, self.quantization, self.max_num_seqs,
            prompt_token_ids, true_ids, false_ids, max_tokens,
            poll_interval_s, do_profile, profile_name,
            block_size=self._resolved_block_size)
