"""Accuracy validation of every attention path against stock vLLM
(issue #24, part 1 - the most important part).

The comparison: the same token streams answered by two systems.

- Stock side: standard vLLM serving (v1 LLM engine, bf16 KV, prefix
  caching on, the committed client's batch settings), one request per
  (document, stage) for filters and one request per pair for joins,
  TRUE/FALSE constrained to one token at temperature 0. Every
  (document, stage) runs unconditionally, so stock defines a complete
  answer function; per-answer TRUE-FALSE logprob margins ride along.
  A second pass in shuffled submission order measures stock's own
  run-to-run flip rate - continuous batching changes batch
  composition, which changes kernel reduction order, which flips
  answers whose margins sit near zero. That self-flip band is the
  yardstick for judging Quail-vs-stock disagreements.
- Quail side: the packed executor's real loops (run_filter, run_join)
  under each attention path, on the same token id lists. A final
  round runs the production sequence (filters on FILTER_ATTENTION,
  then joins on JOIN_ATTENTION, same arena) to check the mode switch
  between rounds changes nothing.

The corpus (built identically in both containers from seeded IMDB):

- 1,000 documents with planted [FLAGS] lines in TRUE/FALSE form,
  including 12 long documents (8 reviews concatenated, hundreds of
  16-token pages), 25 very short documents (~30 tokens), and 5
  documents whose body is only the flags line.
- 5 filter stages: four flag questions (selectivity 0.9/0.9/0.9/0.8,
  planted ground truth) and one natural sentiment question (no
  planted truth), all in the engine's real framing (SHARED_PRE +
  document, then "Evaluate TRUE or FALSE...": logical.py's
  render_filter_question).
- A join: 24 anchors (6 reviews + [KEYS] X=k line) x 48 partners
  (1 review + [KEY] X=k line, 8 of them carrying an X no anchor
  has), planted key-equality truth, rendered exactly as the engine
  renders joins (naming line in kept KV, labeled partner block,
  verbatim template question).

Predictions, stated before the run (house rule):

- Flag-question accuracy is high for both systems (the TRUE/FALSE
  framing matches the readout; the old 27%-wrong figure came from
  YES/NO-planted flags read through TRUE/FALSE ids).
- unified vs stock disagrees least: the unified path is bit-identical
  to a contiguous causal FA3 call (results/attention_parity.json),
  so its differences from stock are kernel-stack differences (fp8
  GEMMs, fused norms), not attention-path differences.
- Every path's disagreement rate lands within a small multiple of
  stock's own pass-to-pass flip rate, and disagreements concentrate
  at |margin| near zero. Disagreements at decisive margins (say
  |margin| > 1.0) are the red flag; the target for those is zero.

Run from the quail/ directory (tee per house rule):

    uv run modal run ablations/accuracy_vs_stock.py::run_all \
        2>&1 | tee results/accuracy_vs_stock.log
"""

import json
import os

import modal

from split_reference import set_path

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "baselines",
                             "split_reference")
    .add_local_dir("tests/gpu", remote_path="/root/gpu_tests")
)

# House rule: never create new Modal app names.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

# the committed stock client's engine settings (ablation A1: bf16 KV)
STOCK_STEP_TOKENS = 25_305
STOCK_MAX_SEQS = 2648

FLAG_SELECTIVITY = (0.9, 0.9, 0.9, 0.8)
N_LONG = 12      # 8 reviews concatenated: hundreds of arena pages
N_SHORT = 25     # ~30 tokens: suffix comparable to the document
N_MINIMAL = 5    # body is only the flags line
CORPUS_SEED = 20260821

JOIN_ANCHORS = 24
JOIN_PARTNERS = 48
# word keys: the first accuracy run used X = 0..11 and the 4B model
# answered TRUE at chance (11-12% key accuracy both systems, margins
# collapsed near zero, so every numeric wobble flipped answers);
# word values make the comparison a task the model can actually do,
# which makes margins - and disagreements - meaningful
JOIN_KEY_WORDS = ("red", "blue", "green", "gold", "silver", "black",
                  "white", "purple", "orange", "brown", "pink",
                  "teal")
JOIN_ORPHANS = 8     # the last partners carry an X no anchor has
JOIN_TEMPLATE = ("Does the [KEY] X value in {1} equal the [KEYS] X "
                 "value in {0}?")

# The five filter stages: (kind, flag column, question text). The
# corpus builders and combine() both read this one definition, so the
# truth combine grades against is the truth the GPU cells planted.
FILTER_STAGES = (
    ("flag", 0, "Does the [FLAGS] line above show FLAG_1=TRUE?"),
    ("flag", 1, "Does the [FLAGS] line above show FLAG_2=TRUE?"),
    ("flag", 2, "Does the [FLAGS] line above show FLAG_3=TRUE?"),
    ("natural", None,
     "Is the overall sentiment of the review above positive?"),
    ("flag", 3, "Does the [FLAGS] line above show FLAG_4=TRUE?"),
)


def _draw_flags(n_docs):
    """The seeded flag matrix: the planted truth, drawn identically
    by the GPU corpus builders and the CPU combine cell."""
    import numpy as np
    rng = np.random.default_rng(CORPUS_SEED)
    return (rng.random((n_docs, len(FLAG_SELECTIVITY)))
            < np.array(FLAG_SELECTIVITY)[None, :]).astype(int)


def _join_key_truth():
    """(a_keys, p_keys, truth): the planted key assignment and the
    pair truth it implies, shared by build_join_corpus and combine."""
    a_keys = [JOIN_KEY_WORDS[i % len(JOIN_KEY_WORDS)]
              for i in range(JOIN_ANCHORS)]
    p_keys = [JOIN_KEY_WORDS[p % len(JOIN_KEY_WORDS)]
              if p < JOIN_PARTNERS - JOIN_ORPHANS else f"nomatch{p}"
              for p in range(JOIN_PARTNERS)]
    truth = [[int(a_keys[a] == p_keys[p])
              for p in range(JOIN_PARTNERS)]
             for a in range(JOIN_ANCHORS)]
    return a_keys, p_keys, truth


def _write(result, name):
    print(json.dumps(result, indent=2)[:4000], flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    return json.dumps(result)


# ------------------------------------------------------------ corpus

def build_filter_corpus(tokenizer, n_docs):
    """(body_ids, q_ids, flags, kinds). flags[d][j] is the planted
    truth for flag stages, None columns for natural stages. kinds[d]
    labels the document's edge class."""
    import sys
    sys.path.insert(0, "/root/gpu_tests")
    from corpus import build_pool, flags_line

    from quail.logical import SHARED_PRE, render_filter_question

    flags = _draw_flags(n_docs)
    pool = build_pool(min(10_000, n_docs + 8 * N_LONG))
    extra = pool[n_docs:]

    bodies, kinds = [], []
    for d in range(n_docs):
        if d < N_LONG:
            body = "\n\n".join([pool[d]] + extra[d * 7:(d + 1) * 7])
            kinds.append("long")
        elif d < N_LONG + N_SHORT:
            body = pool[d][:120]
            kinds.append("short")
        elif d < N_LONG + N_SHORT + N_MINIMAL:
            body = ""
            kinds.append("minimal")
        else:
            body = pool[d]
            kinds.append("plain")
        bodies.append(SHARED_PRE + body + flags_line(flags[d]))

    body_ids = tokenizer(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tokenizer(render_filter_question("\n\n" + text),
                       add_special_tokens=False)["input_ids"]
             for _, _, text in FILTER_STAGES]
    stage_truth = [flags[:, col] if kind == "flag" else None
                   for kind, col, _ in FILTER_STAGES]
    return body_ids, q_ids, stage_truth, kinds


def build_join_corpus(tokenizer):
    """(anchor_ids, frame_ids, suffix_ids, truth). anchor_ids carry
    SHARED_PRE + document (the engine-owned prefix); each suffix is
    label + partner document + rendered question; truth[a][p] is the
    planted key equality."""
    import sys
    sys.path.insert(0, "/root/gpu_tests")
    from corpus import build_pool

    from quail.logical import (SHARED_PRE, join_anchor_note, join_label,
                               render_join_question)

    pool = build_pool(2_000)
    used = iter(range(200, 2_000))
    a_keys, p_keys, truth = _join_key_truth()

    anchors = []
    for i in range(JOIN_ANCHORS):
        body = "\n\n".join(pool[next(used)] for _ in range(6))
        anchors.append(f"{SHARED_PRE}{body}\n\n[KEYS] X={a_keys[i]}")

    partners = []
    for p in range(JOIN_PARTNERS):
        partners.append(f"{pool[next(used)]}\n\n[KEY] X={p_keys[p]}")

    tok = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    anchor_ids = [tok(a) for a in anchors]
    frame_ids = tok(join_anchor_note(0))
    label = tok(join_label(1))
    tail = tok(render_join_question(JOIN_TEMPLATE))
    suffix_ids = [label + tok(p) + tail for p in partners]
    return anchor_ids, frame_ids, suffix_ids, truth


# --------------------------------------------------------- stock side

@app.function(timeout=5400, **GPU_KW)
def stock_side(n_docs: int = 1000,
               model: str = "qwen3-4b-fp8") -> str:
    """Every (document, stage) and every join pair through standard
    vLLM serving, twice for filters (natural and shuffled order)."""
    import time

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from quail.executor.loop import true_false_ids
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    body_ids, q_ids, _, kinds = build_filter_corpus(tokenizer, n_docs)
    anchor_ids, frame_ids, suffix_ids, _ = build_join_corpus(tokenizer)
    t_ids, f_ids = true_false_ids(tokenizer)
    allowed = sorted(t_ids | f_ids)

    llm = LLM(model=spec.hf_name, kv_cache_dtype="auto",
              max_num_batched_tokens=STOCK_STEP_TOKENS,
              max_num_seqs=STOCK_MAX_SEQS,
              gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=allowed,
                              logprobs=len(allowed))

    def answer_of(out):
        tok0 = int(out.outputs[0].token_ids[0])
        bit = 1 if tok0 in t_ids else 0
        lps = out.outputs[0].logprobs[0]
        def side_max(ids):
            best = None
            for i in ids:
                if i in lps:
                    lp = getattr(lps[i], "logprob", lps[i])
                    best = lp if best is None else max(best, lp)
            return best
        t_best, f_best = side_max(t_ids), side_max(f_ids)
        margin = (0.0 if t_best is None or f_best is None
                  else t_best - f_best)
        return bit, round(float(margin), 4)

    def run_prompts(prompts, tag):
        t0 = time.time()
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in prompts], sampling,
            use_tqdm=False)
        wall = round(time.time() - t0, 2)
        pairs = [answer_of(o) for o in outs]
        print(f"[stock_side] {tag}: {len(prompts)} requests "
              f"in {wall}s", flush=True)
        return [b for b, _ in pairs], [m for _, m in pairs], wall

    def run_shuffled(prompts, seed, tag):
        """The pass-to-pass control: the same prompts in a shuffled
        submission order, answers unscrambled back to prompt order."""
        order = list(range(len(prompts)))
        np.random.default_rng(seed).shuffle(order)
        bits_s, margins_s, wall = run_prompts(
            [prompts[i] for i in order], tag)
        bits = [0] * len(order)
        margins = [0.0] * len(order)
        for pos, idx in enumerate(order):
            bits[idx] = bits_s[pos]
            margins[idx] = margins_s[pos]
        return bits, margins, wall

    n_stages = len(q_ids)
    filter_prompts = [body_ids[d] + q_ids[j]
                      for d in range(n_docs) for j in range(n_stages)]
    bits1, margins1, wall1 = run_prompts(filter_prompts, "filters/1")
    bits2, margins2, wall2 = run_shuffled(filter_prompts, 7,
                                          "filters/2-shuffled")

    join_prompts = [a + frame_ids + s
                    for a in anchor_ids for s in suffix_ids]
    jbits, jmargins, jwall = run_prompts(join_prompts, "join")
    jbits2, jmargins2, jwall2 = run_shuffled(join_prompts, 11,
                                             "join/2-shuffled")

    report = dict(
        cell="stock_side", model=spec.name, n_docs=n_docs,
        n_stages=n_stages, kinds=kinds,
        config=dict(kv="auto (bf16)", prefix_caching=True,
                    step_tokens=STOCK_STEP_TOKENS,
                    max_num_seqs=STOCK_MAX_SEQS,
                    sampler="temperature 0, 1 token, TRUE/FALSE ids"),
        walls=dict(filters1=wall1, filters2=wall2, join=jwall,
                   join2=jwall2),
        filter_bits=bits1, filter_margins=margins1,
        filter_bits_rep=bits2, filter_margins_rep=margins2,
        join_bits=jbits, join_margins=jmargins,
        join_bits_rep=jbits2, join_margins_rep=jmargins2)
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(report, f"accuracy_stock_raw{tag}")


# --------------------------------------------------------- quail side

@app.function(timeout=5400, **GPU_KW)
def quail_side(n_docs: int = 1000,
               model: str = "qwen3-4b-fp8") -> str:
    """The packed executor's real loops under each attention path on
    the same token streams, plus the production two-round sequence."""
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.executor.arena import KVArena
    from quail.executor.attention import (FILTER_ATTENTION,
                                          JOIN_ATTENTION, Pipeline)
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     run_join)
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    body_ids, q_ids, _, kinds = build_filter_corpus(tokenizer, n_docs)
    anchor_ids, frame_ids, suffix_ids, _ = build_join_corpus(tokenizer)

    model_mod = load_model(spec.hf_name)
    chunk = budgets.chunk_budget(spec, H100_SXM)
    arena_tok = budgets.arena_tokens(spec, H100_SXM, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model_mod, arena, kernels="quail",
                        attention_mode="merge_quant")
    answerer = Answerer(torch, F, model_mod, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    budget = min(chunk, pipeline.max_chunk_tokens)

    filters, joins, walls = {}, {}, {}
    with torch.inference_mode():
        for mode in ("split", "merge_quant", "unified"):
            set_path(pipeline, mode)
            t0 = time.perf_counter()
            answers, _, _ = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                budget, arena_writes=True)
            torch.cuda.synchronize()
            walls[f"filter_{mode}"] = round(
                time.perf_counter() - t0, 2)
            filters[mode] = {int(d): row for d, row in answers.items()}
            print(f"[quail_side] filter {mode}: "
                  f"{sum(len(r) for r in answers.values())} answers "
                  f"in {walls[f'filter_{mode}']}s", flush=True)

        for mode in ("split", "merge_quant"):
            set_path(pipeline, mode)
            t0 = time.perf_counter()
            ans, _, _ = run_join(
                torch, arena, pipeline, async_ans, anchor_ids,
                [suffix_ids], budget, stage_frames=[frame_ids])
            torch.cuda.synchronize()
            walls[f"join_{mode}"] = round(time.perf_counter() - t0, 2)
            joins[mode] = {int(a): row for a, row in ans[0].items()}
            print(f"[quail_side] join {mode} in "
                  f"{walls[f'join_{mode}']}s", flush=True)

        # the production sequence: the worker's mode switch between
        # rounds, on one arena - must reproduce the isolated runs
        set_path(pipeline, FILTER_ATTENTION)
        prod_f, _, _ = run_filter(torch, arena, pipeline, async_ans,
                                  body_ids, q_ids, budget,
                                  arena_writes=True)
        set_path(pipeline, JOIN_ATTENTION)
        prod_j, _, _ = run_join(torch, arena, pipeline, async_ans,
                                anchor_ids, [suffix_ids], budget,
                                stage_frames=[frame_ids])
        torch.cuda.synchronize()
    mode_switch_clean = (
        {int(d): r for d, r in prod_f.items()} == filters[
            FILTER_ATTENTION]
        and {int(a): r for a, r in prod_j[0].items()} == joins[
            JOIN_ATTENTION])

    report = dict(
        cell="quail_side", model=spec.name, n_docs=n_docs,
        kinds=kinds, walls=walls,
        assignment=dict(filters=FILTER_ATTENTION, joins=JOIN_ATTENTION),
        mode_switch_clean=bool(mode_switch_clean),
        filters=filters, joins=joins)
    tag = "" if spec.name == "qwen3-4b-fp8" else "_32b"
    return _write(report, f"accuracy_quail_raw{tag}")


# --------------------------------------------------------- comparison

@app.function(timeout=1800, image=image,
              volumes={"/results": results_vol})
def combine(n_docs: int = 1000,
            model: str = "qwen3-4b-fp8") -> str:
    """Read both raw answer sets from the results volume and build
    the comparison report (CPU only)."""
    tag = "" if model == "qwen3-4b-fp8" else "_32b"
    results_vol.reload()
    with open(f"/results/ablations/accuracy_stock_raw{tag}.json") as f:
        stock = json.load(f)
    with open(f"/results/ablations/accuracy_quail_raw{tag}.json") as f:
        quail = json.load(f)
    assert stock["n_docs"] == quail["n_docs"] == n_docs

    # the corpus truth is rebuilt from the same module-level
    # definitions the GPU cells planted (CPU, no tokenizer needed)
    flags = _draw_flags(n_docs)
    stage_kind = [kind for kind, _, _ in FILTER_STAGES]
    stage_col = [col for _, col, _ in FILTER_STAGES]
    n_stages = len(FILTER_STAGES)
    kinds = stock["kinds"]

    sbits = stock["filter_bits"]
    smargins = stock["filter_margins"]

    def s_at(d, j):
        return sbits[d * n_stages + j]

    def flip_control(bits, bits_rep, margins):
        """Pass-to-pass flips and the margins they happened at."""
        flips = [i for i, (a, b) in enumerate(zip(bits, bits_rep))
                 if a != b]
        fm = sorted(abs(margins[i]) for i in flips)
        return dict(
            flips=len(flips), of=len(bits),
            rate=round(len(flips) / len(bits), 5),
            flip_margin_max=(fm[-1] if fm else 0.0),
            flip_margin_p50=(fm[len(fm) // 2] if fm else 0.0))

    # stock's own pass-to-pass control
    control = flip_control(sbits, stock["filter_bits_rep"], smargins)

    # stock accuracy on the planted flags (all documents, pass 1)
    stock_acc = {}
    for j in range(n_stages):
        if stage_kind[j] != "flag":
            continue
        col = stage_col[j]
        good = sum(1 for d in range(n_docs)
                   if s_at(d, j) == int(flags[d][col]))
        stock_acc[f"stage_{j}"] = round(good / n_docs, 4)

    def compare_filters(mode):
        q = {int(d): row for d, row in quail["filters"][mode].items()}
        n_comp = disag = 0
        by_stage = [0] * n_stages
        by_kind = {}
        dis_margins = []
        flag_good = flag_total = 0
        for d, row in q.items():
            for j, bit in enumerate(row):
                n_comp += 1
                if stage_kind[j] == "flag":
                    flag_total += 1
                    flag_good += int(bit == int(flags[d][stage_col[j]]))
                if bit != s_at(d, j):
                    disag += 1
                    by_stage[j] += 1
                    by_kind[kinds[d]] = by_kind.get(kinds[d], 0) + 1
                    dis_margins.append(abs(smargins[d * n_stages + j]))
        dis_margins.sort()
        # stock's survivors under the same gated chain
        s_surv = set()
        for d in range(n_docs):
            if all(s_at(d, j) for j in range(n_stages)):
                s_surv.add(d)
        q_surv = {d for d, row in q.items()
                  if len(row) == n_stages and all(row)}
        return dict(
            compared=n_comp, disagreements=disag,
            rate=round(disag / max(n_comp, 1), 5),
            by_stage=by_stage, by_kind=by_kind,
            disagreement_margin_max=(dis_margins[-1]
                                     if dis_margins else 0.0),
            disagreement_margin_p50=(dis_margins[len(dis_margins) // 2]
                                     if dis_margins else 0.0),
            decisive_disagreements=sum(1 for m in dis_margins
                                       if m > 1.0),
            flag_accuracy=round(flag_good / max(flag_total, 1), 4),
            survivors=len(q_surv),
            survivors_stock=len(s_surv),
            survivor_overlap=len(q_surv & s_surv))

    # join truth from the same key assignment the corpus planted
    _, _, jtruth = _join_key_truth()
    jbits = stock["join_bits"]
    jmargins = stock["join_margins"]

    def margin_overview(margins):
        m = sorted(abs(x) for x in margins)
        n = len(m)
        return dict(p25=m[n // 4], p50=m[n // 2], p75=m[3 * n // 4],
                    under_1=round(sum(1 for x in m if x < 1.0) / n, 4))

    join_control = flip_control(jbits, stock["join_bits_rep"],
                                jmargins)

    def compare_join(mode):
        q = {int(a): row for a, row in quail["joins"][mode].items()}
        disag = good_q = good_s = total = 0
        dis_margins = []
        for a in range(JOIN_ANCHORS):
            for p in range(JOIN_PARTNERS):
                s_bit = jbits[a * JOIN_PARTNERS + p]
                q_bit = q[a][p]
                total += 1
                good_s += int(s_bit == jtruth[a][p])
                good_q += int(q_bit == jtruth[a][p])
                if q_bit != s_bit:
                    disag += 1
                    dis_margins.append(
                        abs(jmargins[a * JOIN_PARTNERS + p]))
        dis_margins.sort()
        return dict(
            compared=total, disagreements=disag,
            rate=round(disag / total, 5),
            disagreement_margin_max=(dis_margins[-1]
                                     if dis_margins else 0.0),
            decisive_disagreements=sum(1 for m in dis_margins
                                       if m > 1.0),
            key_accuracy_quail=round(good_q / total, 4),
            key_accuracy_stock=round(good_s / total, 4))

    report = dict(
        cell="accuracy_vs_stock", model=model, n_docs=n_docs,
        stock_config=stock["config"],
        stock_self_control=control,
        stock_join_control=join_control,
        stock_flag_accuracy=stock_acc,
        stock_filter_margins=margin_overview(smargins),
        stock_join_margins=margin_overview(jmargins),
        mode_switch_clean=quail["mode_switch_clean"],
        walls=dict(stock=stock["walls"], quail=quail["walls"]),
        filters={m: compare_filters(m)
                 for m in ("split", "merge_quant", "unified")},
        join={m: compare_join(m)
              for m in ("split", "merge_quant")})
    return _write(report, f"accuracy_vs_stock{tag}")


@app.local_entrypoint()
def run_all(n_docs: int = 1000, model: str = "qwen3-4b-fp8"):
    sh = stock_side.spawn(n_docs, model)
    qh = quail_side.spawn(n_docs, model)
    print(f"stock fc: {sh.object_id}", flush=True)
    print(f"quail fc: {qh.object_id}", flush=True)
    sh.get()
    qh.get()
    print(combine.remote(n_docs, model))


@app.local_entrypoint()
def run_combine(n_docs: int = 1000, model: str = "qwen3-4b-fp8"):
    print(combine.remote(n_docs, model))


@app.local_entrypoint()
def run_quail_only(n_docs: int = 1000, model: str = "qwen3-4b-fp8"):
    """Re-run the Quail side against an already-written stock raw
    file, then combine."""
    qh = quail_side.spawn(n_docs, model)
    print(f"quail fc: {qh.object_id}", flush=True)
    qh.get()
    print(combine.remote(n_docs, model))
