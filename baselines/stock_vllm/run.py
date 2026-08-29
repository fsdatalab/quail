"""Stock vLLM baselines for the default QUAIL-B queries.

Runs on H100s via Modal. ``stock_vllm`` submits one filter stage at a
time. ``pipelined_vllm`` submits the next filter for each document as
soon as that document passes its current filter. Both configurations
use the same full cross product join implementation.

Single container (default):
    uv run modal run -m baselines.stock_vllm.run::main

Parallel across N containers:
    uv run modal run -m baselines.stock_vllm.run::main --containers 4

    Each container boots its own LLM and runs a subset of queries.
    Results are merged and saved to the quail-results volume.

Paired comparison, one container per query set:
    uv run modal run -m baselines.stock_vllm.run::paired_main

    Each container boots one LLM and runs both stock_vllm and
    pipelined_vllm for every query in its set. The execution order
    alternates by query.
"""

import json
import time
import uuid
from pathlib import Path

import modal

from quail.bench.quailb import QUERY_ORDER

app = modal.App("quail-milestone1")

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub[hf_transfer]",
                 "transformers>=5.2.0", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "baselines")
)

hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

DATA_DIR = "/results/quailb_data"

MODELS = {
    "qwen3-4b-fp8": "Qwen/Qwen3-4B-FP8",
    "qwen3-32b-fp8": "Qwen/Qwen3-32B-FP8",
}

BASELINES = ("stock_vllm", "pipelined_vllm")
BASELINE_ORDERS = ("alternating-by-query", "method-major")


def _baseline_configuration(name):
    if name == "stock_vllm":
        return "stage-major"
    if name == "pipelined_vllm":
        return "pipelined"
    raise ValueError(
        f"unknown baseline {name!r}; expected stock_vllm or "
        "pipelined_vllm")


def _paired_baseline_order(rep: int, query_position: int) -> tuple[str, str]:
    """Alternate which configuration runs first."""

    if (rep + query_position) % 2:
        return tuple(reversed(BASELINES))
    return BASELINES


def _baseline_schedule(ids, baselines, rep, method_order):
    """Return the ordered baseline and query pairs for one repetition."""
    if method_order not in BASELINE_ORDERS:
        raise ValueError(
            f"method_order must be one of {BASELINE_ORDERS}; "
            f"got {method_order!r}")
    if method_order == "method-major":
        return [
            (baseline, query_id)
            for baseline in baselines
            for query_id in ids
        ]
    return [
        (baseline, query_id)
        for query_position, query_id in enumerate(ids)
        for baseline in (
            _paired_baseline_order(rep, query_position)
            if len(baselines) == 2 else baselines)
    ]


def _vllm_filter_capacity(llm):
    """Read the post-startup KV capacity from vLLM."""
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    capacity = getattr(cache, "kv_cache_size_tokens", None)
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    if capacity is None and blocks is not None and block_size is not None:
        capacity = blocks * block_size
    if capacity is None or block_size is None:
        raise RuntimeError("vLLM did not report its KV capacity")
    return dict(
        kv_cache_size_tokens=int(capacity),
        num_gpu_blocks=(None if blocks is None else int(blocks)),
        block_size=int(block_size),
        max_num_seqs=int(config.scheduler_config.max_num_seqs),
        kv_cache_dtype=str(cache.cache_dtype),
    )

def _split_queries(ids, n):
    """Split query IDs into n roughly equal chunks."""
    k, m = divmod(len(ids), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(ids[start:start + size])
        start += size
    return [c for c in chunks if c]


def _split_query_sets(ids):
    """Put each QUAIL-B query set in its own chunk."""
    from quail.bench.quailb import split_query_families

    return [list(group) for group in split_query_families(ids)]


def _query_set_name(ids):
    """Return the ground truth workload name for one query-set chunk."""
    from quail.bench.quailb import query_family_name

    return query_family_name(ids)


def _paired_ground_truth_workload(requested, ids):
    if requested == "auto":
        return _query_set_name(ids)
    return requested


def define_all_queries():
    """QUAIL-B query definitions used by the vLLM baselines.

    Each query is a dict with:
        aliases: {alias: (table_name, text_col)}
        steps: list of step tuples, each either
            ("filter", alias, [template, ...])
            ("join", template, left_alias, right_alias)
    """
    from quail.bench.quailb import (
        F1, F4, F5, F7, F8, F9, F11, F12, F13,
        LEP1, LEP2, LEP3, LEP4, LEP5, LEPS1,
        DISCUSS_ASPECT, ASPECT_SENTIMENT,
        REACTION, REACTION_SEVERE,
        SUPPORT, REFUTE, LEPJOIN,
    )

    rv = ("reviews", "body")
    asp = ("aspects", "aspect")
    rp = ("reports", "report")
    tm = ("terms", "term")
    cl = ("claims", "claim")
    ev = ("evidence", "text")
    dc = ("citation_contexts", "destination_context")
    pt = ("citation_passages", "passage_text")

    Q = {}

    # ---- IMDB ----
    Q["IMDB-1"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1])])
    Q["IMDB-2"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a")])
    Q["IMDB-3"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1]),
                              ("join", DISCUSS_ASPECT, "r", "a")])
    Q["IMDB-4"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4]),
                              ("join", DISCUSS_ASPECT, "r", "a")])
    Q["IMDB-5"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4, F5]),
                              ("join", DISCUSS_ASPECT, "r", "a")])
    Q["IMDB-6"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4])])
    Q["IMDB-7"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4, F5])])
    Q["IMDB-8"] = dict(aliases={"r": rv, "a": asp, "a2": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a"),
                              ("join", ASPECT_SENTIMENT, "r", "a2")])
    Q["IMDB-9"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("join", DISCUSS_ASPECT, "r1", "a1"),
               ("join", DISCUSS_ASPECT, "r2", "a1"),
               ("join", ASPECT_SENTIMENT, "r2", "a2")])
    Q["IMDB-10"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("filter", "r1", [F1]),
               ("join", DISCUSS_ASPECT, "r1", "a1"),
               ("join", DISCUSS_ASPECT, "r2", "a1"),
               ("join", ASPECT_SENTIMENT, "r2", "a2")])

    # ---- BioDEX ----
    Q["BIO-1"] = dict(aliases={"r": rp},
                      steps=[("filter", "r", [F7])])
    Q["BIO-2"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("join", REACTION, "r", "m")])
    Q["BIO-3"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7]),
                             ("join", REACTION, "r", "m")])
    Q["BIO-4"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8]),
                             ("join", REACTION, "r", "m")])
    Q["BIO-5"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8, F9]),
                             ("join", REACTION, "r", "m")])
    Q["BIO-6"] = dict(aliases={"r": rp, "m": tm, "m2": tm},
                      steps=[("join", REACTION, "r", "m"),
                             ("join", REACTION_SEVERE, "r", "m2")])
    Q["BIO-7"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("join", REACTION, "r1", "m1"),
               ("join", REACTION_SEVERE, "r2", "m1"),
               ("join", REACTION, "r2", "m2")])
    Q["BIO-8"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("filter", "r1", [F7]),
               ("join", REACTION, "r1", "m1"),
               ("join", REACTION_SEVERE, "r2", "m1"),
               ("join", REACTION, "r2", "m2")])

    # ---- FEVER ----
    Q["FEV-1"] = dict(aliases={"c": cl},
                      steps=[("filter", "c", [F11])])
    Q["FEV-2"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("join", SUPPORT, "c", "e")])
    Q["FEV-3"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("join", SUPPORT, "c", "e")])
    Q["FEV-4"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("join", SUPPORT, "c", "e")])
    Q["FEV-5"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e")])
    Q["FEV-6"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e")])
    Q["FEV-7"] = dict(aliases={"c": cl, "e": ev, "e2": ev},
                      steps=[("join", SUPPORT, "c", "e"),
                             ("join", REFUTE, "c", "e2")])
    Q["FEV-8"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("join", SUPPORT, "c1", "e1"),
               ("join", REFUTE, "c2", "e1"),
               ("join", SUPPORT, "c2", "e2")])
    Q["FEV-9"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("filter", "c1", [F11]),
               ("join", SUPPORT, "c1", "e1"),
               ("join", REFUTE, "c2", "e1"),
               ("join", SUPPORT, "c2", "e2")])

    # ---- LePaRD ----
    Q["LEP-1"] = dict(aliases={"d": dc},
                      steps=[("filter", "d", [LEP1])])
    Q["LEP-2"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("join", LEPJOIN, "d", "s")])
    Q["LEP-3"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1]),
                             ("join", LEPJOIN, "d", "s")])
    Q["LEP-4"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("join", LEPJOIN, "d", "s")])
    Q["LEP-5"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2, LEP3]),
                             ("join", LEPJOIN, "d", "s")])
    Q["LEP-6"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d",
                              [LEP1, LEP2, LEP3, LEP4, LEP5]),
                             ("join", LEPJOIN, "d", "s")])
    Q["LEP-7"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("filter", "s", [LEPS1]),
                             ("join", LEPJOIN, "d", "s")])
    Q["LEP-8"] = dict(aliases={"d": dc},
                      steps=[("filter", "d",
                              [LEP1, LEP2, LEP3, LEP4, LEP5])])

    return Q


def _read_table(data_dir, sf, name, text_col):
    import pyarrow.parquet as pq
    path = Path(data_dir) / f"sf{sf}" / f"{name}.parquet"
    t = pq.read_table(path)
    return t.column("id").to_pylist(), t.column(text_col).to_pylist()


def _load_alias_data(data_dir, sf, aliases):
    cache = {}
    result = {}
    for alias, (table, text_col) in aliases.items():
        key = (table, text_col)
        if key not in cache:
            cache[key] = _read_table(data_dir, sf, table, text_col)
        result[alias] = cache[key]
    return result


def _run_filter_stage(llm, sp, true_set, template, texts, tokenizer):
    """Run one filter template over a list of texts.

    Returns the surviving indices and one answer for every input text.
    """
    prompts = _filter_prompts(template, texts, tokenizer)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    answers = [bool(o.outputs[0].token_ids
                    and int(o.outputs[0].token_ids[0]) in true_set)
               for o in outputs]
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    cached_tokens = sum(
        getattr(o, "num_cached_tokens", 0) or 0 for o in outputs)
    return dict(
        survivors=[i for i, answer in enumerate(answers) if answer],
        answers=answers,
        requests=len(outputs),
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )


def _filter_prompts(template, texts, tokenizer):
    from quail.logical import (
        ColumnRef,
        bind_prompt,
        render_filter_prompt_ids,
    )

    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)
    ref = ColumnRef("document", "document", "document")
    bound = bind_prompt(template, (ref,), tok)
    return [
        {"prompt_token_ids": render_filter_prompt_ids(
            bound, tok(text), tok)}
        for text in texts
    ]


def _filter_chain_inputs(templates, texts, tokenizer):
    """Split canonical filter prompts into a shared body and stage tails."""
    from quail.logical import ColumnRef, bind_prompt

    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)

    ref = ColumnRef("document", "document", "document")
    bound = [bind_prompt(template, (ref,), tok) for template in templates]
    preambles = [tok(prompt.preamble) for prompt in bound]
    if any(preamble != preambles[0] for preamble in preambles[1:]):
        raise ValueError("all filters in a chain must share one preamble")

    body_ids = [preambles[0] + tok(text) for text in texts]
    question_ids = []
    for prompt in bound:
        if not prompt.tail.startswith("{0}"):
            raise ValueError("a filter prompt must start its tail with {0}")
        question_ids.append(tok(prompt.tail.replace("{0}", "", 1)))
    return body_ids, question_ids


def _run_stage_major_filter_chain(llm, sp, true_set, templates, texts,
                                  tokenizer):
    active = list(range(len(texts)))
    answers = {}
    stages = []
    requests = 0
    prompt_tokens = 0
    cached_tokens = 0
    t0 = time.time()

    for stage, template in enumerate(templates, 1):
        evaluated = list(active)
        if evaluated:
            result = _run_filter_stage(
                llm, sp, true_set, template,
                [texts[index] for index in evaluated], tokenizer)
            for position, answer in enumerate(result["answers"]):
                answers[(evaluated[position], stage)] = int(answer)
            active = [evaluated[position]
                      for position in result["survivors"]]
            requests += result["requests"]
            prompt_tokens += result["prompt_tokens"]
            cached_tokens += result["cached_tokens"]
        stages.append(dict(stage=stage, n_in=len(evaluated),
                           n_out=len(active)))

    return dict(
        wall=time.time() - t0,
        survivors=active,
        answers=answers,
        stages=stages,
        requests=requests,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        fresh_tokens=prompt_tokens - cached_tokens,
        doc_cap=None,
    )


def _run_pipelined_filter_chain(llm, sp, true_set, templates, texts,
                                tokenizer, capacity, tag):
    from baselines.stock import run_filter_chain

    if not texts:
        return dict(
            wall=0.0,
            survivors=[],
            answers={},
            stages=[dict(stage=stage, n_in=0, n_out=0)
                    for stage in range(1, len(templates) + 1)],
            requests=0,
            prompt_tokens=0,
            cached_tokens=0,
            fresh_tokens=0,
            doc_cap=0,
        )

    body_ids, question_ids = _filter_chain_inputs(
        templates, texts, tokenizer)
    result = run_filter_chain(
        llm.llm_engine, sp, body_ids, question_ids,
        capacity["kv_cache_size_tokens"], tag=tag, true_ids=true_set,
        block_size=capacity["block_size"],
        max_num_seqs=capacity["max_num_seqs"])
    stages = []
    for stage in range(1, len(templates) + 1):
        evaluated = [index for index in range(len(texts))
                     if (index, stage) in result["answers"]]
        passed = sum(result["answers"][(index, stage)]
                     for index in evaluated)
        stages.append(dict(stage=stage, n_in=len(evaluated),
                           n_out=passed))
    return dict(
        **result,
        stages=stages,
        fresh_tokens=result["prompt_tokens"] - result["cached_tokens"],
    )


def _run_filter_chain(llm, sp, true_set, templates, texts, tokenizer,
                      submission, capacity, tag):
    if submission == "stage-major":
        return _run_stage_major_filter_chain(
            llm, sp, true_set, templates, texts, tokenizer)
    if submission == "pipelined":
        return _run_pipelined_filter_chain(
            llm, sp, true_set, templates, texts, tokenizer,
            capacity, tag)
    raise ValueError(f"unknown filter submission {submission!r}")


def _select_join_anchor(documents):
    """Choose the side with the larger mean surviving document length."""
    means = [sum(map(len, side)) / len(side) for side in documents]
    return (0 if means[0] >= means[1] else 1), means


def _run_join(llm, sp, true_set, template, left_texts, right_texts,
              tokenizer):
    """Run one join as a full cross product.

    Returns the model result, counts, survivors, and true local pairs.
    """
    from baselines.stock import build_join_grouped_inputs, run_join_grouped
    from quail.logical import ColumnRef, bind_join_prompt

    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)
    args = (ColumnRef("left", "left", "document"),
            ColumnRef("right", "right", "document"))
    bound = bind_join_prompt(template, args, tok)
    documents = ([tok(t) for t in left_texts],
                 [tok(t) for t in right_texts])
    anchor, mean_document_tokens = _select_join_anchor(documents)
    prefixes, suffixes, members = build_join_grouped_inputs(
        bound, documents, anchor, tok)
    n_pairs = len(prefixes) * len(suffixes)

    result = run_join_grouped(llm, sp, prefixes, suffixes, true_set)

    surviving_left, surviving_right = set(), set()
    pairs = []
    answers = []
    true_pairs = []
    n_true = 0
    idx = 0
    for anc_idx in range(len(prefixes)):
        for member in members:
            pair = ((anc_idx, member[0]) if anchor == 0
                    else (member[0], anc_idx))
            answer = result["answers"][idx] == 1
            pairs.append(pair)
            answers.append(answer)
            if answer:
                if anchor == 0:
                    surviving_left.add(anc_idx)
                    surviving_right.add(member[0])
                else:
                    surviving_right.add(anc_idx)
                    surviving_left.add(member[0])
                true_pairs.append(pair)
                n_true += 1
            idx += 1

    return (result, n_pairs, surviving_left, surviving_right,
            n_true, true_pairs, pairs, answers, anchor,
            mean_document_tokens)


def run_query(llm, sp, true_set, tokenizer, qid, query_def,
              data_dir, sf, evaluator=None, quail_query=None,
              filter_submission="stage-major", filter_capacity=None):
    """Execute one multi-step query. Returns a summary dict."""
    alias_data = _load_alias_data(data_dir, sf, query_def["aliases"])
    live = {a: list(range(len(data[1])))
            for a, data in alias_data.items()}

    step_results = []
    filter_answer_records = []
    join_answer_records = []
    true_join_tables = []
    step_n = 0

    for step in query_def["steps"]:
        if step[0] == "filter":
            _, alias, templates = step
            _, texts_all = alias_data[alias]
            live_idx = list(live[alias])
            live_texts = [texts_all[index] for index in live_idx]
            if filter_submission == "pipelined" and filter_capacity is None:
                raise ValueError("pipelined filters need the vLLM KV capacity")
            result = _run_filter_chain(
                llm, sp, true_set, templates, live_texts, tokenizer,
                filter_submission, filter_capacity,
                tag=f"{qid}-{step_n}-{alias}")
            new_live = [live_idx[index] for index in result["survivors"]]

            for stage, template in enumerate(templates, 1):
                evaluated = sorted(
                    index for index in range(len(live_idx))
                    if (index, stage) in result["answers"])
                answers = [bool(result["answers"][(index, stage)])
                           for index in evaluated]
                filter_answer_records.append((
                    alias,
                    template,
                    [live_idx[index] for index in evaluated],
                    answers,
                ))

            step_results.append(dict(
                kind="filter_chain", step=step_n, alias=alias,
                submission=filter_submission,
                n_stages=len(templates), n_in=len(live_idx),
                n_out=len(new_live), wall_s=result["wall"],
                stages=result["stages"], requests=result["requests"],
                fresh_tokens=result["fresh_tokens"],
                prompt_tokens=result["prompt_tokens"],
                cached_tokens=result["cached_tokens"],
                doc_cap=result["doc_cap"]))
            print(f"  filter({alias}, {filter_submission}): "
                  f"{len(live_idx)}->{len(new_live)} "
                  f"wall={result['wall']:.2f}s", flush=True)
            live[alias] = new_live
            step_n += 1

        elif step[0] == "join":
            _, template, left_a, right_a, *_unused_anchor = step
            _, left_texts_all = alias_data[left_a]
            _, right_texts_all = alias_data[right_a]
            left_live = live[left_a]
            right_live = live[right_a]
            left_texts = [left_texts_all[i] for i in left_live]
            right_texts = [right_texts_all[i] for i in right_live]
            nl, nr = len(left_texts), len(right_texts)

            if nl == 0 or nr == 0:
                from quail.runtime.result import document_index_table

                true_table = document_index_table(
                    {left_a: [], right_a: []}, "join_answers")
                true_join_tables.append(true_table)
                join_answer_records.append(
                    (template, left_a, right_a, [], [], true_table))
                step_results.append(dict(
                    kind="join", step=step_n, left=left_a,
                    right=right_a, n_left=nl, n_right=nr,
                    n_pairs=0, n_true=0, wall_s=0))
                step_n += 1
                continue

            print(f"  join({left_a}x{right_a}): {nl}x{nr}="
                  f"{nl * nr} pairs...", flush=True)

            t0 = time.time()
            (result, n_pairs, surv_l, surv_r,
             n_true, true_pairs, pairs, answers, anchor,
             mean_document_tokens) = _run_join(
                llm, sp, true_set, template, left_texts,
                right_texts, tokenizer)
            wall = time.time() - t0

            from quail.runtime.result import document_index_table

            true_table = document_index_table({
                left_a: [left_live[left] for left, _ in true_pairs],
                right_a: [right_live[right] for _, right in true_pairs],
            }, "join_answers")
            true_join_tables.append(true_table)
            global_pairs = [
                (left_live[left], right_live[right])
                for left, right in pairs
            ]
            join_answer_records.append(
                (template, left_a, right_a, global_pairs, answers,
                 true_table))

            live[left_a] = sorted(left_live[i] for i in surv_l)
            live[right_a] = sorted(right_live[i] for i in surv_r)

            step_results.append(dict(
                kind="join", step=step_n, left=left_a,
                right=right_a, anchor=anchor,
                anchor_alias=(left_a if anchor == 0 else right_a),
                mean_document_tokens={
                    left_a: mean_document_tokens[0],
                    right_a: mean_document_tokens[1],
                },
                n_left=nl, n_right=nr,
                n_pairs=n_pairs, n_true=n_true,
                wall_s=wall,
                fresh_tokens=result["fresh_tokens"],
                prompt_tokens=result["prompt_tokens"],
                cached_tokens=result["cached_tokens"]))
            print(f"  join({left_a}x{right_a}): "
                  f"{n_true}/{n_pairs} TRUE "
                  f"anchor={left_a if anchor == 0 else right_a} "
                  f"wall={wall:.2f}s "
                  f"fresh={result['fresh_tokens']}", flush=True)
            step_n += 1

    import pyarrow as pa

    from quail.runtime.result import (
        build_result_declaration,
        count_rows,
    )

    survivor_arrays = {
        alias: pa.array(indices, type=pa.int32())
        for alias, indices in live.items()
    }
    result_declaration, result_schema = build_result_declaration(
        true_join_tables,
        survivor_arrays,
        next(iter(query_def["aliases"])),
    )
    total_wall = sum(s["wall_s"] for s in step_results)
    row_count = count_rows(result_declaration)
    entry = dict(query=qid, steps=step_results,
                total_wall_s=total_wall, rows=row_count,
                result_schema=str(result_schema),
                rows_materialized=False)
    if evaluator is not None:
        if quail_query is None:
            raise ValueError("accuracy scoring requires the Quail query")
        from quail.logical import ColumnRef, bind_prompt
        from quail.planner.decide import _collect
        from quail.runtime.result import answer_table

        _scans, logical_filters, logical_joins = _collect(
            quail_query.logical)
        filter_records = {}
        for alias, template, indices, answers in filter_answer_records:
            ref = ColumnRef(alias, query_def["aliases"][alias][0],
                            query_def["aliases"][alias][1])
            key = evaluator.ground_truth.key_for_template(
                bind_prompt(template, (ref,)).template)
            filter_records[(alias, key)] = answer_table(
                {alias: indices}, answers, "filter_answers")
        filter_tables = {}
        for alias, predicates in logical_filters.items():
            for written_pos, predicate in enumerate(predicates):
                key = evaluator.ground_truth.key_for_template(
                    predicate.prompt.template)
                filter_tables[(alias, written_pos)] = \
                    filter_records[(alias, key)]

        join_records = {}
        for (template, left_alias, right_alias, pairs, answers,
             true_table) in join_answer_records:
            key = evaluator.ground_truth.key_for_template(template)
            join_records[key] = (
                answer_table(
                    {left_alias: [left for left, _ in pairs],
                     right_alias: [right for _, right in pairs]},
                    answers, "join_answers"),
                true_table,
            )
        join_tables = {}
        true_tables = {}
        for written_pos, join in enumerate(logical_joins):
            key = evaluator.ground_truth.key_for_template(
                join.predicate.template)
            join_tables[written_pos], true_tables[written_pos] = \
                join_records[key]

        class EvaluationResult:
            answer_tables = {"filters": filter_tables,
                             "joins": join_tables}
            survivor_indices = survivor_arrays
            true_join_tables = true_tables

            @staticmethod
            def count():
                return row_count

        entry["accuracy"] = evaluator.evaluate(
            quail_query, EvaluationResult())
    return entry


@app.function(timeout=600, image=image,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def ensure_data(sf: float):
    """Build QUAIL-B datasets on the volume before parallel fan-out."""
    from quail.bench.quailb import build_sets
    build_sets(DATA_DIR, sf)


def _run_query_batches(model: str, sf: float, query_ids_csv: str,
                       reps: int, ground_truth_workload: str,
                       prediction: str, baselines: tuple[str, ...],
                       paired_run_id: str = "", lf: int = 1,
                       method_order: str = "alternating-by-query") -> dict:
    """Boot one LLM and run one or both baseline configurations.

    Args:
        model: Key into MODELS dict.
        sf: Scale factor for QUAIL-B data.
        query_ids_csv: Comma-separated query IDs to run.
            Empty string means all default queries.
        reps: Number of repetitions.

    Returns:
        JSON string with boot info and per-query results.
    """
    from baselines.stock_boot import time_llm_boot
    from quail.bench.quailb import (
        DATA_SEED,
        SOURCE_REVISIONS,
        build_sets,
        queries as quail_queries,
        register_sets,
    )
    from quail.executor.loop import true_false_ids
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    if not baselines:
        raise ValueError("at least one baseline is required")
    for baseline in baselines:
        _baseline_configuration(baseline)
    if method_order not in BASELINE_ORDERS:
        raise ValueError(
            f"method_order must be one of {BASELINE_ORDERS}; "
            f"got {method_order!r}")

    hf_name = MODELS[model]
    data_path = build_sets(DATA_DIR, sf, lf)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    true, false = true_false_ids(tokenizer)
    allowed = sorted(true | false)

    llm, boot = time_llm_boot(
        model=hf_name,
        max_num_batched_tokens=25_305,
        max_num_seqs=4096,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=allowed)
    filter_capacity = _vllm_filter_capacity(llm)
    run_name = paired_run_id or baselines[0]
    print(f"[{run_name}] boot: {boot}", flush=True)
    print(f"[{run_name}] KV capacity: {filter_capacity}", flush=True)

    llm.generate([{"prompt_token_ids": allowed}], sp, use_tqdm=False)

    queries = define_all_queries()
    if query_ids_csv:
        ids = [q.strip() for q in query_ids_csv.split(",")]
        for qid in ids:
            if qid not in queries:
                raise ValueError(
                    f"unknown query {qid!r}; available: "
                    f"{sorted(queries)}")
    else:
        ids = [qid for qid in QUERY_ORDER if qid in queries]

    evaluator = None
    truth = None
    evaluation_queries = {}
    if ground_truth_workload:
        import quail
        from quail.bench.evaluate import (
            BenchmarkEvaluator,
            LocalVolumeFiles,
            corpus_identity,
            load_ground_truth_workload,
            read_corpus,
        )
        from quail.planner.plan import EngineConfig

        corpus_rows = read_corpus(data_path)
        corpus = corpus_identity(
            corpus_rows, sf, DATA_SEED, SOURCE_REVISIONS)
        truth = load_ground_truth_workload(
            LocalVolumeFiles("/results"),
            scale_factor=sf,
            corpus_id=corpus["corpus_id"],
            corpus_full_hash=corpus["corpus_full_hash"],
            workload=ground_truth_workload,
        )
        evaluator = BenchmarkEvaluator(truth, corpus_rows)
        session = quail.Session(
            EngineConfig(gpus=1, model=model),
            tokenizer=lambda text: tokenizer.encode(
                text, add_special_tokens=False),
        )
        register_sets(session, data_path)
        evaluation_queries = quail_queries(session)

    all_results = {baseline: [] for baseline in baselines}
    execution_order = []
    for rep in range(reps):
        rep_results = {baseline: [] for baseline in baselines}
        schedule = _baseline_schedule(
            ids, baselines, rep, method_order)
        for baseline, qid in schedule:
            reset = llm.reset_prefix_cache()
            if reset is False:
                raise RuntimeError(
                    "vLLM refused to reset its prefix cache before "
                    f"{baseline} {qid}")
            filter_submission = _baseline_configuration(baseline)
            print(f"\n[{baseline}] rep={rep} {qid}", flush=True)
            try:
                entry = run_query(
                    llm, sp, true, tokenizer,
                    qid, queries[qid], DATA_DIR, sf,
                    evaluator=evaluator,
                    quail_query=(evaluation_queries[qid][1]()
                                 if evaluator else None),
                    filter_submission=filter_submission,
                    filter_capacity=filter_capacity)
            except Exception as e:                      # noqa: BLE001
                entry = dict(query=qid,
                             error=f"{type(e).__name__}: {e}")
                print(f"  ERROR: {e}", flush=True)
            rep_results[baseline].append(entry)
        for baseline in baselines:
            all_results[baseline].append(rep_results[baseline])
        execution_order.append([
            {"baseline": baseline, "query": qid}
            for baseline, qid in schedule
        ])

    reports = {}
    for baseline in baselines:
        filter_submission = _baseline_configuration(baseline)
        submission = (
            "separate generate() call per filter stage, full cross "
            "product per join"
            if filter_submission == "stage-major"
            else "pipelined per-document filter chain, full cross "
                 "product per join"
        )
        report = dict(
            baseline=baseline, model=model, hf_name=hf_name, sf=sf, lf=lf,
            boot=boot, reps=reps,
            prediction=prediction,
            ground_truth_workload=(ground_truth_workload or None),
            ground_truth=(None if truth is None else {
                "collection_id": truth.collection_id,
                "corpus_id": truth.corpus_id,
                "reference_model": truth.reference_model,
            }),
            query_ids=ids,
            filter_submission=filter_submission,
            filter_capacity=filter_capacity,
            submission=submission,
            checkpoint="pre-quantized FP8",
            max_num_seqs=4096, max_num_batched_tokens=25_305,
            gpu_memory_utilization=0.92,
            enable_prefix_caching=True,
            results=all_results[baseline])
        if len(baselines) == 2:
            report["paired_run"] = {
                "id": paired_run_id,
                "same_model_process": True,
                "prefix_cache_reset_before_each_configuration": True,
                "method_order": method_order,
                "execution_order": execution_order,
            }
        reports[baseline] = report
    return {"reports": reports, "execution_order": execution_order}


@app.function(timeout=7200, **GPU_KW)
def run_query_batch(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                    query_ids_csv: str = "", reps: int = 1,
                    ground_truth_workload: str = "",
                    prediction: str = "",
                    baseline: str = "stock_vllm") -> str:
    """Boot one LLM and run one baseline configuration."""

    result = _run_query_batches(
        model, sf, query_ids_csv, reps, ground_truth_workload,
        prediction, (baseline,))
    return json.dumps(result["reports"][baseline])


@app.function(timeout=21600, **GPU_KW)
def run_paired_query_batch(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                           query_ids_csv: str = "", reps: int = 1,
                           ground_truth_workload: str = "",
                           prediction: str = "",
                           paired_run_id: str = "") -> str:
    """Boot one LLM and run both configurations in that process."""

    result = _run_query_batches(
        model, sf, query_ids_csv, reps, ground_truth_workload,
        prediction, BASELINES, paired_run_id)
    return json.dumps(result)


@app.function(timeout=300, image=image,
              volumes={"/results": results_vol})
def save_report(report_json: str, label: str, baseline: str) -> str:
    """Save a merged report to the quail-results volume."""
    _baseline_configuration(baseline)
    out_dir = Path("/results") / baseline / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        f.write(report_json)
    results_vol.commit()
    return str(out_path)


def _merge_reports(batch_reports, n_containers):
    """Merge results from parallel containers into one report."""
    base = batch_reports[0]
    reps = base["reps"]
    boots = [r["boot"] for r in batch_reports]
    all_query_ids = []
    for r in batch_reports:
        all_query_ids.extend(r["query_ids"])

    merged_results = []
    for rep_idx in range(reps):
        rep_queries = []
        for report in batch_reports:
            rep_queries.extend(report["results"][rep_idx])
        merged_results.append(rep_queries)

    return dict(
        baseline=base["baseline"],
        model=base["model"],
        hf_name=base["hf_name"],
        sf=base["sf"],
        lf=base.get("lf", 1),
        boots=boots,
        reps=reps,
        containers=n_containers,
        query_ids=all_query_ids,
        filter_submission=base["filter_submission"],
        filter_capacity=[report["filter_capacity"]
                         for report in batch_reports],
        submission=base["submission"],
        checkpoint=base["checkpoint"],
        max_num_seqs=base["max_num_seqs"],
        max_num_batched_tokens=base["max_num_batched_tokens"],
        gpu_memory_utilization=base["gpu_memory_utilization"],
        enable_prefix_caching=base["enable_prefix_caching"],
        prediction=base.get("prediction"),
        ground_truth_workload=base.get("ground_truth_workload"),
        ground_truth=base.get("ground_truth"),
        results=merged_results,
    )


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", sf: float = 0.1,
         query: str = "", reps: int = 1,
         containers: int = 1, by_query_set: bool = False,
         ground_truth_workload: str = "", prediction: str = "",
         baseline: str = "stock_vllm"):
    _baseline_configuration(baseline)
    if query:
        ids = [q.strip() for q in query.split(",")]
    else:
        ids = list(QUERY_ORDER)

    label = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if prediction:
        print(f"PREDICTION: {prediction}", flush=True)

    if containers <= 1 and not by_query_set:
        fc = run_query_batch.spawn(
            model=model, sf=sf,
            query_ids_csv=",".join(ids), reps=reps,
            ground_truth_workload=ground_truth_workload,
            prediction=prediction, baseline=baseline)
        print(f"function call id: {fc.object_id}")
        result_json = fc.get()
        report = json.loads(result_json)
    else:
        print(f"[{baseline}] building data (sf={sf}) before "
              f"fan-out...")
        ensure_data.remote(sf)
        chunks = (_split_query_sets(ids) if by_query_set
                  else _split_queries(ids, containers))
        print(f"[{baseline}] data ready, spawning {len(chunks)} "
              f"containers for {len(ids)} queries")
        handles = []
        for i, chunk in enumerate(chunks):
            fc = run_query_batch.spawn(
                model=model, sf=sf,
                query_ids_csv=",".join(chunk), reps=reps,
                ground_truth_workload=ground_truth_workload,
                prediction=prediction, baseline=baseline)
            print(f"  container {i}: {fc.object_id} "
                  f"({len(chunk)} queries: "
                  f"{chunk[0]}..{chunk[-1]})")
            handles.append(fc)

        batch_reports = []
        for i, h in enumerate(handles):
            print(f"  waiting for container {i}...")
            batch_reports.append(json.loads(h.get()))
            print(f"  container {i} done")

        report = _merge_reports(batch_reports, len(chunks))

    out_path = save_report.remote(
        json.dumps(report, indent=2), label, baseline)
    print(f"\n[{baseline}] saved {out_path}")
    print(f"  {len(ids)} queries, {reps} reps, "
          f"{report.get('containers', 1)} container(s)")
    print(json.dumps(report))


@app.local_entrypoint()
def paired_main(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                query: str = "", reps: int = 1,
                ground_truth_workload: str = "",
                prediction: str = ""):
    """Run both baselines in one model process per query set.

    Set ``--ground-truth-workload auto`` to load the matching ground
    truth collection independently in each query-set container.
    """

    if reps < 1:
        raise ValueError("reps must be at least one")
    if query:
        ids = [query_id.strip() for query_id in query.split(",")]
    else:
        ids = list(QUERY_ORDER)

    label = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if prediction:
        print(f"PREDICTION: {prediction}", flush=True)

    print(f"[paired] building data (sf={sf}) before fan-out...")
    ensure_data.remote(sf)
    chunks = _split_query_sets(ids)
    print(f"[paired] data ready, spawning {len(chunks)} containers "
          f"for {len(ids)} queries")

    handles = []
    for chunk in chunks:
        workload = _paired_ground_truth_workload(
            ground_truth_workload, chunk)
        fc = run_paired_query_batch.spawn(
            model=model, sf=sf,
            query_ids_csv=",".join(chunk), reps=reps,
            ground_truth_workload=workload,
            prediction=prediction, paired_run_id=label)
        query_set = _query_set_name(chunk)
        print(f"  {query_set}: {fc.object_id} "
              f"({len(chunk)} queries: {chunk[0]}..{chunk[-1]})")
        handles.append((query_set, chunk, workload, fc))

    batches = []
    for query_set, chunk, workload, handle in handles:
        print(f"  waiting for {query_set}...")
        batches.append((
            query_set,
            chunk,
            workload,
            handle.object_id,
            json.loads(handle.get()),
        ))
        print(f"  {query_set} done")

    pairing = {
        "id": label,
        "same_model_process": True,
        "one_container_per_query_set": True,
        "prefix_cache_reset_before_each_configuration": True,
        "order_rule": (
            "stock first when rep plus query position is even; "
            "pipelined first otherwise"),
        "query_sets": [
            {
                "query_set": query_set,
                "query_ids": chunk,
                "function_call_id": function_call_id,
                "execution_order": output["execution_order"],
            }
            for (query_set, chunk, _workload, function_call_id,
                 output) in batches
        ],
    }

    reports = {}
    for baseline in BASELINES:
        report = _merge_reports(
            [output["reports"][baseline]
             for _query_set, _chunk, _workload, _function_call_id,
             output in batches],
            len(batches))
        report["paired_run"] = pairing
        report["ground_truth_workload"] = {
            query_set: (workload or None)
            for query_set, _chunk, workload, _function_call_id,
            _output in batches
        }
        report["ground_truth"] = {
            query_set: output["reports"][baseline].get("ground_truth")
            for query_set, _chunk, _workload, _function_call_id,
            output in batches
        }
        reports[baseline] = report

    paths = {}
    for baseline in BASELINES:
        paths[baseline] = save_report.remote(
            json.dumps(reports[baseline], indent=2), label, baseline)
        print(f"[{baseline}] saved {paths[baseline]}")

    print(f"[paired] {len(ids)} queries, {reps} reps, "
          f"{len(batches)} query-set containers")
    print(json.dumps({"paired_run_id": label, "paths": paths,
                      "function_call_ids": {
                          query_set: function_call_id
                          for query_set, _chunk, _workload,
                          function_call_id, _output in batches
                      }}))
