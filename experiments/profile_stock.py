"""Run any QuailB query on stock vLLM under instrumentation: GPU
profiler windows, per-request cached tokens, and KV regret.

The companion cell (`experiments/profile_quail.py`) measures Quail.
This cell measures stock vLLM the same way, reusing the benchmark
baseline's own code for everything that defines the measured system
- prompts, boot flags, sampling, submission order, anchor choice,
cache resets - and mirroring the baseline's own step loop
(`baselines/stock_vllm/run.py::run_query`) so any query's filter
chains and joins gate the same way. No engine or baseline file
changes.

What it records, per stage of each query:

- Per-request `num_cached_tokens` from vLLM's own outputs. For a
  join, requests are bucketed by position within the anchor group
  (pair 0 computes the anchor prefix; pairs 1+ can hit it).
- KV regret per request: the tokens of this prompt whose KV an
  earlier request in the same query already computed (the longest
  common token prefix with any earlier prompt that carried the same
  document, rounded down to vLLM's 16-token cache block), minus the
  tokens vLLM actually served from cache. Summed, this is the same
  infinite-KV regret the Quail cell measures.
- torch.profiler windows through vLLM's own profiler hooks, one per
  stage, placed after a warmup slice so the window sees the stage's
  steady state. Windows run in a second pass over a re-reset cache
  that replays every stage in order (so cache pressure from earlier
  stages is faithful); the primary pass's walls carry no profiler
  overhead.

`--max-join-anchors N` measures only the first N anchor groups of
each join (0 = all). Use it for joins whose full cross product is
slow on stock; later stages then gate on the measured anchors only,
so use it for single-join queries or accept the truncation. The
recorded BIO-2 run used 60 of its 500 reports.

Run (`--queries` is a comma-separated list of QuailB ids):

    uv run modal run experiments/profile_stock.py::run_smoke --queries IMDB-3,BIO-2
    uv run modal run experiments/profile_stock.py::run --queries IMDB-3,BIO-2 --out-prefix myrun

Outputs on the quail-results volume (pick an --out-prefix that does
not overwrite files a report already cites):

    /results/ablations/<prefix>_stock_<queryslug>.json
    /results/ablations/<prefix>_traces/stock_kineto/  (raw traces)

The recorded 2026-08-30 runs used prefix "discrepancy" through this
cell's predecessor (experiments/discrepancy_stock.py, which hardcoded
IMDB-3 and BIO-2); predictions and results live in
reports/2026-08-30-imdb3-bio2-discrepancies.md.
"""

import glob
import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

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
    .env({"VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HUB_ENABLE_HF_TRANSFER": "1"})
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
# requests per profiler window, rounded to whole anchor groups
WINDOW_REQUESTS = 1200


def _slug(qid):
    return qid.lower().replace("-", "")


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


def _boot(model, sf, kineto_dir):
    """The benchmark baseline's boot, verbatim settings."""
    from baselines.stock_boot import time_llm_boot
    from baselines.stock_vllm.run import DATA_DIR, _vllm_filter_capacity
    from quail.bench.quailb import build_sets
    from quail.executor.loop import true_false_ids
    from quail.specs import MODELS
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    os.makedirs(kineto_dir, exist_ok=True)
    # some vLLM paths read the env, the engine reads the config;
    # both must name the same directory
    os.environ["VLLM_TORCH_PROFILER_DIR"] = kineto_dir
    build_sets(DATA_DIR, sf)
    results_vol.commit()
    hf_name = MODELS[model].hf_name
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    true, false = true_false_ids(tokenizer)
    allowed = sorted(true | false)
    # vLLM 0.26 enables its torch.profiler hooks through the engine's
    # profiler config; recording still only happens between
    # start_profile and stop_profile
    try:
        from vllm.config import ProfilerConfig
        prof_cfg = ProfilerConfig(profiler="torch",
                                  torch_profiler_dir=kineto_dir)
    except ImportError:
        prof_cfg = {"profiler": "torch",
                    "torch_profiler_dir": kineto_dir}
    # same pinned revision as the engine, so boots stay comparable
    revision = MODELS[model].revision or None
    llm, boot = time_llm_boot(
        model=hf_name,
        revision=revision,
        tokenizer_revision=revision,
        max_num_batched_tokens=25_305,
        max_num_seqs=4096,
        gpu_memory_utilization=0.91,
        enable_prefix_caching=True,
        disable_log_stats=True,
        profiler_config=prof_cfg)
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


def _profiled_slice(llm, sp, prompts, label, kineto_dir):
    """Profile one generate call through vLLM's profiler hooks.

    Returns the trace files the window produced (vLLM names them
    itself inside the kineto directory).
    """
    before = set(glob.glob(f"{kineto_dir}/*"))
    llm.start_profile()
    t0 = time.time()
    llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    llm.stop_profile()
    time.sleep(2)   # the engine process finishes writing the trace
    files = sorted(set(glob.glob(f"{kineto_dir}/*")) - before)
    results_vol.commit()
    print(f"[stock] window {label}: {wall:.1f}s -> {files}",
          flush=True)
    return dict(label=label, wall_s=round(wall, 3),
                requests=len(prompts), files=files)


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
    return prefixes, suffixes, members, anchor


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


def _measure_query(llm, sp, true_set, tokenizer, sf, qid,
                   max_join_anchors):
    """The primary pass of one query: the baseline's own step loop
    with per-request cache capture and regret bookkeeping.

    Returns (result, plan): plan is the ordered request lists the
    profiled pass replays.
    """
    from baselines.stock_vllm.run import (
        DATA_DIR,
        _filter_prompts,
        _load_alias_data,
        define_all_queries,
    )

    q = define_all_queries()[qid]
    data = _load_alias_data(DATA_DIR, sf, q["aliases"])
    live = {a: list(range(len(d[1]))) for a, d in data.items()}
    # (alias, global row) -> earlier prompts that computed this
    # document's KV as a prompt prefix; the source of would-hit
    prior = {a: {} for a in data}

    _reset(llm)
    result = dict(query=qid, sf=sf, stages=[], regret_tokens=0)
    plan = []

    for n, step in enumerate(q["steps"]):
        if step[0] == "filter":
            _, alias, templates = step
            texts_all = data[alias][1]
            for t_i, template in enumerate(templates):
                rows = list(live[alias])
                prompts = _filter_prompts(
                    template, [texts_all[r] for r in rows], tokenizer)
                out = _generate(llm, sp, prompts)
                passed = [i for i, t in enumerate(out["first_token"])
                          if t in true_set]
                live[alias] = [rows[i] for i in passed]
                for i in passed:
                    prior[alias].setdefault(rows[i], []).append(
                        prompts[i]["prompt_token_ids"])
                result["stages"].append(dict(
                    kind="filter", step=n, stage=t_i, alias=alias,
                    wall_s=out["wall_s"], requests=len(prompts),
                    prompt_tokens=out["prompt_tokens"],
                    cached_tokens=sum(out["cached"]),
                    survivors=len(passed)))
                plan.append(("wave", prompts,
                             f"stock_{_slug(qid)}_s{n}_filter{t_i}"))
                print(f"[stock] {qid} filter({alias} stage {t_i}): "
                      f"{len(rows)}->{len(passed)} "
                      f"wall={out['wall_s']}s", flush=True)

        elif step[0] == "join":
            _, template, left_a, right_a, *_unused = step
            left_rows, right_rows = live[left_a], live[right_a]
            if not left_rows or not right_rows:
                result["stages"].append(dict(
                    kind="join", step=n, left=left_a, right=right_a,
                    pairs=0, note="a side is empty"))
                continue
            prefixes, suffixes, members, anchor = _join_inputs(
                template, [data[left_a][1][r] for r in left_rows],
                [data[right_a][1][r] for r in right_rows], tokenizer)
            anchor_a = left_a if anchor == 0 else right_a
            member_a = right_a if anchor == 0 else left_a
            anchor_rows = left_rows if anchor == 0 else right_rows
            member_rows = right_rows if anchor == 0 else left_rows
            total_anchors = len(prefixes)
            if max_join_anchors:
                prefixes = prefixes[:max_join_anchors]
            seen = [max((_lcp(prefixes[a], p)
                         for p in prior[anchor_a].get(anchor_rows[a],
                                                      [])),
                        default=0)
                    for a in range(len(prefixes))]
            pair = _pair_prompts(prefixes, suffixes)
            out = _generate(llm, sp, pair)
            buckets = _join_accounting(prefixes, suffixes,
                                       out["cached"], seen)
            regret = (buckets["pair0"]["regret"]
                      + buckets["rest"]["regret"])
            result["regret_tokens"] += regret

            # gate live sets exactly as the baseline's run_query does
            k = len(suffixes)
            surv_anchor, surv_member = set(), set()
            for a in range(len(prefixes)):
                for j, member in enumerate(members):
                    if out["first_token"][a * k + j] in true_set:
                        surv_anchor.add(a)
                        surv_member.add(member[0])
            live[anchor_a] = [anchor_rows[a]
                              for a in sorted(surv_anchor)]
            live[member_a] = [member_rows[m]
                              for m in sorted(surv_member)]
            for a in surv_anchor:
                prior[anchor_a].setdefault(
                    anchor_rows[a], []).append(prefixes[a])

            result["stages"].append(dict(
                kind="join", step=n, left=left_a, right=right_a,
                anchor=anchor_a,
                anchors_measured=len(prefixes),
                anchors_total=total_anchors,
                pairs=len(pair), wall_s=out["wall_s"],
                ms_per_pair=round(1e3 * out["wall_s"] / len(pair), 3),
                prompt_tokens=out["prompt_tokens"],
                cached_tokens=sum(out["cached"]),
                buckets=buckets, regret_tokens=regret,
                mean_seen_prefix=round(sum(seen) / len(seen), 1)))
            plan.append(("groups", prefixes, suffixes,
                         f"stock_{_slug(qid)}_s{n}_join"))
            print(f"[stock] {qid} join({left_a}x{right_a}, "
                  f"{len(prefixes)}/{total_anchors} anchors): "
                  f"wall={out['wall_s']}s buckets={buckets}",
                  flush=True)
    return result, plan


def _replay_profiled(llm, sp, plan, kineto_dir):
    """The profiled pass: replay every stage in order on a re-reset
    cache, profiling a steady-state window inside each. The last
    stage stops after its window; nothing downstream needs its
    tail."""
    windows = []
    _reset(llm)
    for i, entry in enumerate(plan):
        last = i == len(plan) - 1
        if entry[0] == "wave":
            _, prompts, label = entry
            n = len(prompts)
            warm = min(n - 1, max(1, n // 3))
            win = max(1, min(WINDOW_REQUESTS, n - warm))
        else:
            _, prefixes, suffixes, label = entry
            k = len(suffixes)
            groups = -(-WINDOW_REQUESTS // k)
            n_groups = len(prefixes)
            warm_g = min(n_groups - 1, max(1, min(n_groups // 3,
                                                  groups)))
            win_g = max(1, min(n_groups - warm_g, groups))
            prompts = _pair_prompts(prefixes, suffixes)
            warm, win = warm_g * k, win_g * k
        if warm:
            llm.generate(prompts[:warm], sp, use_tqdm=False)
        windows.append(_profiled_slice(
            llm, sp, prompts[warm:warm + win], label, kineto_dir))
        rest = prompts[warm + win:]
        if rest and not last:
            llm.generate(rest, sp, use_tqdm=False)
    return windows


@app.function(timeout=7200, **GPU_KW)
def measure(model: str = "qwen3-4b-fp8", sf: float = 0.1,
            queries: tuple = (), max_join_anchors: int = 0,
            profiled: bool = True,
            out_prefix: str = "profile") -> str:
    if not queries:
        raise ValueError("pass at least one QuailB query id")
    from baselines.stock_vllm.run import define_all_queries
    known = define_all_queries()
    missing = [q for q in queries if q not in known]
    if missing:
        raise KeyError(f"unknown QuailB queries {missing}; "
                       f"known: {sorted(known)}")
    kineto_dir = f"/results/ablations/{out_prefix}_traces/stock_kineto"
    llm, sp, true_set, tokenizer, capacity = _boot(model, sf,
                                                   kineto_dir)
    summary = dict(cell="profile_stock", model=model, sf=sf,
                   capacity=capacity, queries={})
    suffix = "" if sf == 0.1 else f"_sf{sf}"
    for qid in queries:
        result, plan = _measure_query(llm, sp, true_set, tokenizer,
                                      sf, qid, max_join_anchors)
        result["capacity"] = capacity
        if profiled:
            result["windows"] = _replay_profiled(llm, sp, plan,
                                                 kineto_dir)
        _write(result, f"{out_prefix}_stock_{_slug(qid)}{suffix}")
        summary["queries"][qid] = dict(
            regret_tokens=result["regret_tokens"],
            stages=[dict(kind=s["kind"], wall_s=s.get("wall_s"),
                         cached_tokens=s.get("cached_tokens"))
                    for s in result["stages"]])
    return json.dumps(summary, indent=2)


# ---------------------------------------------------------- entrypoints

def _parse_queries(queries):
    out = tuple(q.strip() for q in queries.split(",") if q.strip())
    if not out:
        raise ValueError("--queries must name at least one QuailB id")
    return out


@app.local_entrypoint()
def run(queries: str, model: str = "qwen3-4b-fp8", sf: float = 0.1,
        max_join_anchors: int = 0, out_prefix: str = "profile"):
    handle = measure.spawn(model, sf, _parse_queries(queries),
                           max_join_anchors, True, out_prefix)
    print(f"profile_stock fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_smoke(queries: str, model: str = "qwen3-4b-fp8",
              max_join_anchors: int = 10,
              out_prefix: str = "profile"):
    """The full harness on the sf 0.01 tables before the measured
    run."""
    handle = measure.spawn(model, 0.01, _parse_queries(queries),
                           max_join_anchors, True, out_prefix)
    print(f"profile_stock smoke fc: {handle.object_id}", flush=True)
    print(handle.get())
