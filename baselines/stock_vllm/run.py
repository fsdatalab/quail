"""Stock vLLM baseline for all 35 QUAIL-B queries.

Runs on a single H100 via Modal. Each filter stage is a separate
llm.generate() call. Each join runs the full cross product via
run_join_grouped(). Multi-step queries feed survivors from one step
to the next.

    uv run modal run -m baselines.stock_vllm.run::main
"""

import json
import time
from pathlib import Path

import modal

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
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

QUERY_ORDER = [
    "IMDB-1", "IMDB-2", "IMDB-3", "IMDB-4", "IMDB-5",
    "IMDB-6", "IMDB-7", "IMDB-8", "IMDB-9", "IMDB-10",
    "BIO-1", "BIO-2", "BIO-3", "BIO-4", "BIO-5",
    "BIO-6", "BIO-7", "BIO-8",
    "FEV-1", "FEV-2", "FEV-3", "FEV-4", "FEV-5", "FEV-6",
    "FEV-7", "FEV-8", "FEV-9",
    "LEP-1", "LEP-2", "LEP-3", "LEP-4", "LEP-5", "LEP-6",
    "LEP-7", "LEP-8",
]


def define_all_queries():
    """All 35 QUAIL-B queries.

    Each query is a dict with:
        aliases: {alias: (table_name, text_col)}
        steps: list of step tuples, each either
            ("filter", alias, [template, ...])
            ("join", template, left_alias, right_alias, anchor)
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
    dc = ("citations", "destination_context")
    pt = ("citations", "passage_text")

    Q = {}

    # ---- IMDB ----
    Q["IMDB-1"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1])])
    Q["IMDB-2"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-3"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-4"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-5"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4, F5]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-6"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4])])
    Q["IMDB-7"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4, F5])])
    Q["IMDB-8"] = dict(aliases={"r": rv, "a": asp, "a2": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a", 0),
                              ("join", ASPECT_SENTIMENT, "r", "a2", 0)])
    Q["IMDB-9"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("join", DISCUSS_ASPECT, "r1", "a1", 0),
               ("join", DISCUSS_ASPECT, "r2", "a1", 0),
               ("join", ASPECT_SENTIMENT, "r2", "a2", 0)])
    Q["IMDB-10"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("filter", "r1", [F1]),
               ("join", DISCUSS_ASPECT, "r1", "a1", 0),
               ("join", DISCUSS_ASPECT, "r2", "a1", 0),
               ("join", ASPECT_SENTIMENT, "r2", "a2", 0)])

    # ---- BioDEX ----
    Q["BIO-1"] = dict(aliases={"r": rp},
                      steps=[("filter", "r", [F7])])
    Q["BIO-2"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("join", REACTION, "r", "m", 0)])
    Q["BIO-3"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7]),
                             ("join", REACTION, "r", "m", 0)])
    Q["BIO-4"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8]),
                             ("join", REACTION, "r", "m", 0)])
    Q["BIO-5"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8, F9]),
                             ("join", REACTION, "r", "m", 0)])
    Q["BIO-6"] = dict(aliases={"r": rp, "m": tm, "m2": tm},
                      steps=[("join", REACTION, "r", "m", 0),
                             ("join", REACTION_SEVERE, "r", "m2", 0)])
    Q["BIO-7"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("join", REACTION, "r1", "m1", 0),
               ("join", REACTION_SEVERE, "r2", "m1", 0),
               ("join", REACTION, "r2", "m2", 0)])
    Q["BIO-8"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("filter", "r1", [F7]),
               ("join", REACTION, "r1", "m1", 0),
               ("join", REACTION_SEVERE, "r2", "m1", 0),
               ("join", REACTION, "r2", "m2", 0)])

    # ---- FEVER ----
    Q["FEV-1"] = dict(aliases={"c": cl},
                      steps=[("filter", "c", [F11])])
    Q["FEV-2"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("join", SUPPORT, "c", "e", 1)])
    Q["FEV-3"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-4"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-5"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-6"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-7"] = dict(aliases={"c": cl, "e": ev, "e2": ev},
                      steps=[("join", SUPPORT, "c", "e", 1),
                             ("join", REFUTE, "c", "e2", 1)])
    Q["FEV-8"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("join", SUPPORT, "c1", "e1", 1),
               ("join", REFUTE, "c2", "e1", 1),
               ("join", SUPPORT, "c2", "e2", 1)])
    Q["FEV-9"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("filter", "c1", [F11]),
               ("join", SUPPORT, "c1", "e1", 1),
               ("join", REFUTE, "c2", "e1", 1),
               ("join", SUPPORT, "c2", "e2", 1)])

    # ---- LePaRD ----
    Q["LEP-1"] = dict(aliases={"d": dc},
                      steps=[("filter", "d", [LEP1])])
    Q["LEP-2"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-3"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-4"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-5"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2, LEP3]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-6"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d",
                              [LEP1, LEP2, LEP3, LEP4, LEP5]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-7"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("filter", "s", [LEPS1]),
                             ("join", LEPJOIN, "d", "s", 0)])
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

    Returns list of indices (into texts) that answered TRUE.
    """
    prompts = [{"prompt_token_ids":
                tokenizer.encode(template.format(t),
                                 add_special_tokens=False)}
               for t in texts]
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    return [i for i, o in enumerate(outputs)
            if o.outputs[0].token_ids
            and int(o.outputs[0].token_ids[0]) in true_set]


def _run_join(llm, sp, true_set, template, left_texts, right_texts,
              anchor, tokenizer):
    """Run one join as a full cross product.

    Returns (result_dict, n_pairs, surviving_left_set,
    surviving_right_set, n_true).
    """
    from baselines.stock import build_join_grouped_inputs, run_join_grouped
    from quail.logical import ColumnRef, bind_join_prompt

    tok = lambda text: tokenizer.encode(text, add_special_tokens=False)
    args = (ColumnRef("left", "left", "document"),
            ColumnRef("right", "right", "document"))
    bound = bind_join_prompt(template, args, tok)
    documents = ([tok(t) for t in left_texts],
                 [tok(t) for t in right_texts])
    prefixes, suffixes, members = build_join_grouped_inputs(
        bound, documents, anchor, tok)
    n_pairs = len(prefixes) * len(suffixes)

    result = run_join_grouped(llm, sp, prefixes, suffixes, true_set)

    surviving_left, surviving_right = set(), set()
    n_true = 0
    idx = 0
    for anc_idx in range(len(prefixes)):
        for member in members:
            if result["answers"][idx] == 1:
                if anchor == 0:
                    surviving_left.add(anc_idx)
                    surviving_right.add(member[0])
                else:
                    surviving_right.add(anc_idx)
                    surviving_left.add(member[0])
                n_true += 1
            idx += 1

    return result, n_pairs, surviving_left, surviving_right, n_true


def run_query(llm, sp, true_set, tokenizer, qid, query_def,
              data_dir, sf):
    """Execute one multi-step query. Returns a summary dict."""
    alias_data = _load_alias_data(data_dir, sf, query_def["aliases"])
    live = {a: list(range(len(data[1])))
            for a, data in alias_data.items()}

    step_results = []
    step_n = 0

    for step in query_def["steps"]:
        if step[0] == "filter":
            _, alias, templates = step
            _, texts_all = alias_data[alias]

            for tmpl in templates:
                live_idx = live[alias]
                n_in = len(live_idx)
                if n_in == 0:
                    step_results.append(dict(
                        kind="filter", step=step_n, alias=alias,
                        n_in=0, n_out=0, wall_s=0))
                    step_n += 1
                    continue

                live_texts = [texts_all[i] for i in live_idx]
                t0 = time.time()
                survivors = _run_filter_stage(
                    llm, sp, true_set, tmpl, live_texts, tokenizer)
                wall = time.time() - t0
                new_live = [live_idx[i] for i in survivors]

                step_results.append(dict(
                    kind="filter", step=step_n, alias=alias,
                    n_in=n_in, n_out=len(new_live), wall_s=wall))
                print(f"  filter({alias}): {n_in}->{len(new_live)} "
                      f"wall={wall:.2f}s", flush=True)
                live[alias] = new_live
                step_n += 1

        elif step[0] == "join":
            _, template, left_a, right_a, anchor = step
            _, left_texts_all = alias_data[left_a]
            _, right_texts_all = alias_data[right_a]
            left_live = live[left_a]
            right_live = live[right_a]
            left_texts = [left_texts_all[i] for i in left_live]
            right_texts = [right_texts_all[i] for i in right_live]
            nl, nr = len(left_texts), len(right_texts)

            if nl == 0 or nr == 0:
                step_results.append(dict(
                    kind="join", step=step_n, left=left_a,
                    right=right_a, n_left=nl, n_right=nr,
                    n_pairs=0, n_true=0, wall_s=0))
                step_n += 1
                continue

            print(f"  join({left_a}x{right_a}): {nl}x{nr}="
                  f"{nl * nr} pairs...", flush=True)

            t0 = time.time()
            result, n_pairs, surv_l, surv_r, n_true = _run_join(
                llm, sp, true_set, template, left_texts,
                right_texts, anchor, tokenizer)
            wall = time.time() - t0

            live[left_a] = sorted(left_live[i] for i in surv_l)
            live[right_a] = sorted(right_live[i] for i in surv_r)

            step_results.append(dict(
                kind="join", step=step_n, left=left_a,
                right=right_a, anchor=anchor,
                n_left=nl, n_right=nr,
                n_pairs=n_pairs, n_true=n_true,
                wall_s=wall,
                fresh_tokens=result["fresh_tokens"],
                prompt_tokens=result["prompt_tokens"],
                cached_tokens=result["cached_tokens"]))
            print(f"  join({left_a}x{right_a}): "
                  f"{n_true}/{n_pairs} TRUE "
                  f"wall={wall:.2f}s "
                  f"fresh={result['fresh_tokens']}", flush=True)
            step_n += 1

    total_wall = sum(s["wall_s"] for s in step_results)
    return dict(query=qid, steps=step_results,
                total_wall_s=total_wall)


@app.function(timeout=7200, **GPU_KW)
def run_stock_baseline(model: str = "qwen3-4b", sf: float = 0.1,
                       query_id: str = "",
                       reps: int = 1) -> str:
    """Run all 35 QUAIL-B queries on stock vLLM."""
    from baselines.stock_boot import time_llm_boot
    from quail.bench.quailb import build_sets
    from quail.executor.loop import true_false_ids
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    hf_name = MODELS[model]
    build_sets(DATA_DIR, sf)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    true, false = true_false_ids(tokenizer)
    allowed = sorted(true | false)

    llm, boot = time_llm_boot(
        model=hf_name,
        max_num_batched_tokens=25_305,
        max_num_seqs=4096,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        disable_log_stats=True,
        quantization="fp8")
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=allowed)
    print(f"[stock_vllm] boot: {boot}", flush=True)

    llm.generate([{"prompt_token_ids": allowed}], sp, use_tqdm=False)

    queries = define_all_queries()
    if query_id:
        ids = [q.strip() for q in query_id.split(",")]
        for qid in ids:
            if qid not in queries:
                raise ValueError(
                    f"unknown query {qid!r}; available: "
                    f"{sorted(queries)}")
    else:
        ids = [qid for qid in QUERY_ORDER if qid in queries]

    out_dir = Path("/results/stock_vllm") / time.strftime(
        "%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for rep in range(reps):
        print(f"\n[stock_vllm] === rep {rep} ===", flush=True)
        if rep > 0:
            llm.reset_prefix_cache()
        rep_results = []
        for qid in ids:
            print(f"\n[stock_vllm] {qid}", flush=True)
            try:
                entry = run_query(llm, sp, true, tokenizer,
                                  qid, queries[qid], DATA_DIR, sf)
            except Exception as e:                          # noqa: BLE001
                entry = dict(query=qid,
                             error=f"{type(e).__name__}: {e}")
                print(f"  ERROR: {e}", flush=True)
            rep_results.append(entry)
        all_results.append(rep_results)

    report = dict(model=model, hf_name=hf_name, sf=sf,
                  boot=boot, reps=reps,
                  submission="separate generate() per filter stage, "
                             "full cross product per join",
                  max_num_seqs=4096, max_num_batched_tokens=25_305,
                  gpu_memory_utilization=0.92,
                  enable_prefix_caching=True,
                  results=all_results)

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()

    print(f"\n[stock_vllm] saved {out_path} "
          f"({len(ids)} queries x {reps} reps)", flush=True)
    return json.dumps(report)


@app.local_entrypoint()
def main(model: str = "qwen3-4b", sf: float = 0.1,
         query: str = "", reps: int = 1):
    fc = run_stock_baseline.spawn(
        model=model, sf=sf, query_id=query, reps=reps)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
