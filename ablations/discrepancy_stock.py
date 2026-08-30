"""Stock vLLM's side of the IMDB-3 / BIO-2 discrepancy: GPU
timeline, KV regret, and cache churn, measured on stock itself.

The companion cell (`ablations/discrepancy_timeline.py`) measured
Quail. This cell measures stock vLLM the same way, reusing the
benchmark baseline's own code for everything that defines the
measured system - prompts, boot flags, sampling, submission order,
cache resets - so the numbers compare directly against the recorded
2026-08-27 family runs. No engine or baseline file changes.

What it records:

- Per-request `num_cached_tokens` from vLLM's own outputs, for every
  filter and join request, bucketed by the request's position within
  its anchor group (pair 0 computes the anchor prefix; pairs 1+ can
  hit it).
- KV regret per request: the tokens of this prompt whose KV an
  earlier request in the same query already computed (the longest
  common token prefix, rounded down to vLLM's 16-token cache block),
  minus the tokens vLLM actually served from cache. Summed, this is
  the same infinite-KV regret the Quail cell measures.
- Churn evidence: stock exposes no eviction counter, so the record
  is the observable consequence - how much of the would-hit prefix
  the cache still held when the join asked for it.
- torch.profiler windows through vLLM's own profiler hooks
  (VLLM_TORCH_PROFILER_DIR + llm.start_profile), one window per
  regime: the IMDB-3 filter, the IMDB-3 join, and the BIO-2 join at
  steady state. Windows run in a second pass over a re-reset cache,
  so the primary pass's walls carry no profiler overhead.

BIO-2 runs the first `bio2_reports` anchor reports (default 60,
about 3 minutes) rather than all 500: the recorded run already
supplies the full wall, and the per-pair pattern repeats per report.

Predictions, stated before the run:
- IMDB-3 filter: full batches, GPU mostly busy, near-zero cached
  tokens (a single-stage scan has nothing to reuse).
- IMDB-3 join: pair-0 cached tokens near zero for every review (the
  1.76M-token filter scan flushed the 479,248-token cache), pairs
  1-11 cached near the full anchor prefix. Regret about 1.35M
  tokens, slightly larger than Quail's 1,220,547.
- BIO-2 join: pair-0 misses only on each report's first appearance
  (first computation, not regret); regret about 0. GPU busy fraction
  well below Quail's 99%, because a step can hold at most about 117
  resident 4,066-token sequences (479,248 KV tokens), so steps stay
  small.

Run:

    uv run modal run ablations/discrepancy_stock.py::run_smoke
    uv run modal run ablations/discrepancy_stock.py::run_queries

Outputs on the quail-results volume:

    /results/ablations/discrepancy_stock_imdb3.json
    /results/ablations/discrepancy_stock_bio2.json
    /results/ablations/discrepancy_traces/stock_kineto/  (raw traces)
"""

import glob
import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

KINETO_DIR = "/results/ablations/discrepancy_traces/stock_kineto"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "transformers>=5.2.0",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets",
    )
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "VLLM_TORCH_PROFILER_DIR": KINETO_DIR})
    .add_local_python_source("quail", "baselines")
)

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=98304,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

CACHE_BLOCK = 16


def _write(result, name):
    print(json.dumps(result, indent=2, default=str)[:4000], flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()


def _lcp(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _block_floor(n):
    return (n // CACHE_BLOCK) * CACHE_BLOCK


def _boot(model, sf):
    """The benchmark baseline's boot, verbatim settings."""
    from baselines.stock_boot import time_llm_boot
    from baselines.stock_vllm.run import DATA_DIR, _vllm_filter_capacity
    from quail.bench.quailb import build_sets
    from quail.executor.loop import true_false_ids
    from quail.specs import MODELS
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    os.makedirs(KINETO_DIR, exist_ok=True)
    build_sets(DATA_DIR, sf)
    results_vol.commit()
    hf_name = MODELS[model].hf_name
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    true, false = true_false_ids(tokenizer)
    allowed = sorted(true | false)
    llm, boot = time_llm_boot(
        model=hf_name,
        max_num_batched_tokens=25_305,
        max_num_seqs=4096,
        gpu_memory_utilization=0.91,
        enable_prefix_caching=True,
        disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=allowed)
    capacity = _vllm_filter_capacity(llm)
    llm.generate([{"prompt_token_ids": allowed}], sp, use_tqdm=False)
    print(f"[stock] boot {boot} capacity {capacity}", flush=True)
    return llm, sp, set(true), tokenizer, capacity


def _reset(llm):
    if llm.reset_prefix_cache() is False:
        raise RuntimeError("vLLM refused to reset its prefix cache")


def _generate(llm, sp, prompts):
    """One generate call; walls, answers, per-request cached tokens."""
    t0 = time.time()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    return dict(
        wall_s=round(wall, 3),
        prompt_tokens=sum(len(o.prompt_token_ids) for o in outputs),
        cached=[int(getattr(o, "num_cached_tokens", 0) or 0)
                for o in outputs],
        first_token=[int(o.outputs[0].token_ids[0])
                     if o.outputs[0].token_ids else -1
                     for o in outputs])


def _profiled_slice(llm, sp, prompts, label):
    """Profile one generate call through vLLM's profiler hooks.

    Returns the trace files the window produced (vLLM names them
    itself inside VLLM_TORCH_PROFILER_DIR).
    """
    before = set(glob.glob(f"{KINETO_DIR}/*"))
    llm.start_profile()
    t0 = time.time()
    llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    llm.stop_profile()
    time.sleep(2)   # the engine process finishes writing the trace
    files = sorted(set(glob.glob(f"{KINETO_DIR}/*")) - before)
    results_vol.commit()
    print(f"[stock] window {label}: {wall:.1f}s -> {files}",
          flush=True)
    return dict(label=label, wall_s=round(wall, 3),
                requests=len(prompts), files=files)


def _filter_step(llm, sp, true_set, template, texts, tokenizer):
    from baselines.stock_vllm.run import _filter_prompts

    prompts = _filter_prompts(template, texts, tokenizer)
    out = _generate(llm, sp, prompts)
    survivors = [i for i, t in enumerate(out["first_token"])
                 if t in true_set]
    return prompts, out, survivors


def _join_inputs(template, left_texts, right_texts, tokenizer):
    from baselines.stock import build_join_grouped_inputs
    from baselines.stock_vllm.run import _select_join_anchor
    from quail.logical import ColumnRef, bind_join_prompt

    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)
    args = (ColumnRef("left", "left", "document"),
            ColumnRef("right", "right", "document"))
    bound = bind_join_prompt(template, args, tok)
    documents = ([tok(t) for t in left_texts],
                 [tok(t) for t in right_texts])
    anchor, _ = _select_join_anchor(documents)
    prefixes, suffixes, members = build_join_grouped_inputs(
        bound, documents, anchor, tok)
    return prefixes, suffixes


def _pair_prompts(prefixes, suffixes):
    return [{"prompt_token_ids": p + s}
            for p in prefixes for s in suffixes]


def _join_accounting(prefixes, suffixes, cached, seen_prefix_lens):
    """Regret and cache-pattern buckets for one anchor-major join.

    seen_prefix_lens: per anchor, the tokens of this join prefix an
    earlier request already computed (0 when the anchor is new to
    the query). Pair 0 of an anchor can hit only that; pairs 1+ can
    hit the whole prefix pair 0 computed.
    """
    k = len(suffixes)
    buckets = dict(
        pair0=dict(n=0, cached=0, would_hit=0, regret=0),
        rest=dict(n=0, cached=0, would_hit=0, regret=0))
    for a, prefix in enumerate(prefixes):
        for j in range(k):
            got = cached[a * k + j]
            if j == 0:
                would = _block_floor(seen_prefix_lens[a])
                b = buckets["pair0"]
            else:
                would = _block_floor(len(prefix))
                b = buckets["rest"]
            b["n"] += 1
            b["cached"] += got
            b["would_hit"] += would
            b["regret"] += max(0, would - got)
    return buckets


def _measure_imdb3(llm, sp, true_set, tokenizer, sf, profiled):
    from baselines.stock_vllm.run import (
        DATA_DIR,
        _load_alias_data,
        define_all_queries,
    )

    q = define_all_queries()["IMDB-3"]
    data = _load_alias_data(DATA_DIR, sf, q["aliases"])
    (_, f_alias, templates), (_, j_template, left, right) = q["steps"]
    texts = data[f_alias][1]

    _reset(llm)
    result = dict(query="IMDB-3", sf=sf, n_docs=len(texts))

    f_prompts, f_out, survivors = _filter_step(
        llm, sp, true_set, templates[0], texts, tokenizer)
    result["filter"] = dict(
        wall_s=f_out["wall_s"], requests=len(f_prompts),
        prompt_tokens=f_out["prompt_tokens"],
        cached_tokens=sum(f_out["cached"]),
        survivors=len(survivors))

    left_texts = [data[left][1][i] for i in survivors]
    prefixes, suffixes = _join_inputs(
        j_template, left_texts, data[right][1], tokenizer)
    # what an earlier request computed of each join prefix: the
    # longest common token prefix with that review's filter prompt
    seen = [_lcp(prefixes[s], f_prompts[orig]["prompt_token_ids"])
            for s, orig in enumerate(survivors)]
    j_out = _generate(llm, sp, _pair_prompts(prefixes, suffixes))
    buckets = _join_accounting(prefixes, suffixes, j_out["cached"],
                               seen)
    result["join"] = dict(
        wall_s=j_out["wall_s"],
        requests=len(prefixes) * len(suffixes),
        prompt_tokens=j_out["prompt_tokens"],
        cached_tokens=sum(j_out["cached"]),
        buckets=buckets,
        mean_seen_prefix=round(sum(seen) / len(seen), 1))
    result["regret_tokens"] = (buckets["pair0"]["regret"]
                               + buckets["rest"]["regret"])

    if profiled:
        result["windows"] = []
        _reset(llm)
        n = len(texts)
        warm, win = f_prompts[:n // 3], f_prompts[n // 3:n // 3 + 600]
        llm.generate(warm, sp, use_tqdm=False)
        result["windows"].append(_profiled_slice(
            llm, sp, win, "stock_imdb3_filter"))
        llm.generate(f_prompts[n // 3 + 600:], sp, use_tqdm=False)
        pair = _pair_prompts(prefixes, suffixes)
        llm.generate(pair[:1800], sp, use_tqdm=False)
        result["windows"].append(_profiled_slice(
            llm, sp, pair[1800:3600], "stock_imdb3_join"))
    return result


def _measure_bio2(llm, sp, true_set, tokenizer, sf, profiled,
                  bio2_reports):
    from baselines.stock_vllm.run import (
        DATA_DIR,
        _load_alias_data,
        define_all_queries,
    )

    q = define_all_queries()["BIO-2"]
    data = _load_alias_data(DATA_DIR, sf, q["aliases"])
    (_, template, left, right) = q["steps"][0]
    prefixes, suffixes = _join_inputs(
        template, data[left][1], data[right][1], tokenizer)
    part = prefixes[:bio2_reports]

    _reset(llm)
    out = _generate(llm, sp, _pair_prompts(part, suffixes))
    # no earlier operator: nothing of a report's prefix was computed
    # before its own pair 0, so seen is 0 for every report
    buckets = _join_accounting(part, suffixes, out["cached"],
                               [0] * len(part))
    result = dict(
        query="BIO-2", sf=sf,
        reports_measured=len(part), reports_total=len(prefixes),
        pairs=len(part) * len(suffixes),
        wall_s=out["wall_s"],
        ms_per_pair=round(1e3 * out["wall_s"]
                          / (len(part) * len(suffixes)), 3),
        prompt_tokens=out["prompt_tokens"],
        cached_tokens=sum(out["cached"]),
        buckets=buckets,
        regret_tokens=(buckets["pair0"]["regret"]
                       + buckets["rest"]["regret"]))

    if profiled:
        _reset(llm)
        pair = _pair_prompts(part, suffixes)
        k = len(suffixes)
        llm.generate(pair[:6 * k], sp, use_tqdm=False)
        result["windows"] = [_profiled_slice(
            llm, sp, pair[6 * k:12 * k], "stock_bio2_join")]
    return result


@app.function(timeout=7200, **GPU_KW)
def measure(model: str = "qwen3-4b-fp8", sf: float = 0.1,
            bio2_reports: int = 60, profiled: bool = True) -> str:
    llm, sp, true_set, tokenizer, capacity = _boot(model, sf)
    summary = dict(cell="discrepancy_stock", model=model, sf=sf,
                   capacity=capacity, queries={})
    suffix = "" if sf == 0.1 else f"_sf{sf}"

    r3 = _measure_imdb3(llm, sp, true_set, tokenizer, sf, profiled)
    r3["capacity"] = capacity
    _write(r3, f"discrepancy_stock_imdb3{suffix}")
    summary["queries"]["IMDB-3"] = dict(
        filter_wall_s=r3["filter"]["wall_s"],
        join_wall_s=r3["join"]["wall_s"],
        regret_tokens=r3["regret_tokens"],
        pair0_cached=r3["join"]["buckets"]["pair0"]["cached"])

    rb = _measure_bio2(llm, sp, true_set, tokenizer, sf, profiled,
                       bio2_reports)
    rb["capacity"] = capacity
    _write(rb, f"discrepancy_stock_bio2{suffix}")
    summary["queries"]["BIO-2"] = dict(
        wall_s=rb["wall_s"], ms_per_pair=rb["ms_per_pair"],
        regret_tokens=rb["regret_tokens"])
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def run_queries(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                bio2_reports: int = 60):
    handle = measure.spawn(model, sf, bio2_reports, True)
    print(f"stock measure fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_smoke(model: str = "qwen3-4b-fp8"):
    """The full harness on the sf 0.01 tables before the measured
    run."""
    handle = measure.spawn(model, 0.01, 10, True)
    print(f"stock smoke fc: {handle.object_id}", flush=True)
    print(handle.get())
