"""Packing sweep: per-chunk us/token on the packings real benchmark
queries produce (issue #25).

The cost model prices every fresh token at t = a + a2*h and assumes
the mix inside a chunk does not matter. This cell measures that claim
on the packings the engine actually builds, by running four QUAIL-B
queries exactly as the benchmark's cold pass runs them (no store;
single-stage filters on the fast path with arena writes off; joins
write anchor KV) and timing every chunk with CUDA events:

  BIO-1   filter F7 over 200 real BioDEX reports - long documents,
          833..15k tokens mixed inside one chunk
  IMDB-1  filter F1 over 5,000 real IMDB reviews - short documents,
          ~300 pieces per chunk
  BIO-2   join REACTION, 200 reports x ~614 terms - 1-2 long anchor
          prefixes + ~1,200 short suffixes per chunk
  IMDB-2  join DISCUSS_ASPECT, 5,000 reviews x 12 aspects - ~90 short
          anchors + ~1,050 suffixes per chunk

Every chunk is one measured point: its composition (the executor's
trace) and its GPU milliseconds. Raw per-chunk records go to the
quail-results volume as /results/ablations/packing_sweep_<model>_<tag>.json;
only aggregated summaries are committed to results/.

Predictions, stated before the run (house rule), by replaying the
packers over the same corpora with the CURRENT constants (4B: the old
exploration's a=8.26us/tok, a2=4.93e-10; 32B: the 08-20 sweep's
a=56.6us/tok, a2=1.53e-9 - both predate the attention-path and
bug-fix rounds, which is what issue #25 is about):

  4B:  BIO-1 11.0, IMDB-1 8.5, BIO-2 10.4, IMDB-2 8.5 us/token
  32B: BIO-1 65.2, IMDB-1 57.3, BIO-2 63.3, IMDB-2 57.3 us/token

The old benchmark's join queries measured ~15 us/token wall against a
~10 prediction, so the join queries are expected to land above the
filter fit; the per-chunk GPU times say whether the gap is on the GPU
or host-side.

Run from the quail/ directory (tee per house rule; keep the fc- ids):

    uv run modal run ablations/packing_sweep.py --model qwen3-4b-fp8 \
        2>&1 | tee results/packing_sweep_4b.log
    uv run modal run ablations/packing_sweep.py --model qwen3-32b-fp8 \
        2>&1 | tee results/packing_sweep_32b.log
    # container variation: 3 concurrent filter-only containers
    uv run modal run ablations/packing_sweep.py --model qwen3-4b-fp8 \
        --queries BIO-1,IMDB-1 --reps 3 \
        2>&1 | tee results/packing_reps_4b.log
"""

import json
import os

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# Same image and pins as tests/gpu/milestone1.py: vllm==0.26.0 is
# measurement hygiene (same kernels, same loader), not an API
# dependency.
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

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

QUERY_IDS = ("BIO-1", "IMDB-1", "BIO-2", "IMDB-2")


def _queries(tok):
    """The four QUAIL-B workloads as executor inputs, built exactly
    the way the session builds them (session._payload / _join_spec):
    filter bodies are [engine preamble + document], the question is
    the bound prompt's tail with placeholders stripped; join suffixes
    are [partner label + partner document + rendered question], the
    anchor naming line rides as the stage frame."""
    import re

    from quail.bench.quailb import (ASPECTS, DISCUSS_ASPECT, F1, F7,
                                    REACTION, _biodex_rows, _imdb_pool,
                                    _vocab_table)
    from quail.logical import (SHARED_PRE, ColumnRef, bind_join_prompt,
                               bind_prompt, join_anchor_note,
                               join_label, render_join_question)

    bio = _biodex_rows(200)                       # sf=0.1 counts
    reports = [t for t, _ in bio]
    terms = _vocab_table(bio, 1, cap=2_560)
    reviews = _imdb_pool()[:5_000]

    pre = tok(SHARED_PRE)

    def filter_q(template, alias, col, texts):
        p = bind_prompt(template, (ColumnRef(alias, alias, col),), tok)
        qids = tok(re.sub(r"\{\d+\}", "", p.tail))
        return dict(kind="filter", qids=qids,
                    bodies=[pre + tok(t) for t in texts])

    def join_q(template, cols, anchor_texts, partner_texts):
        args = tuple(ColumnRef(a, a, c) for a, c in cols)
        bind_join_prompt(template, args, tok)   # placeholder check
        label, tail = tok(join_label(1)), tok(
            render_join_question(template))
        return dict(kind="join",
                    frame=tok(join_anchor_note(0)),
                    prefixes=[pre + tok(t) for t in anchor_texts],
                    suffixes=[label + tok(t) + tail
                              for t in partner_texts])

    return {
        "BIO-1": filter_q(F7, "r", "report", reports),
        "IMDB-1": filter_q(F1, "r", "body", reviews),
        "BIO-2": join_q(REACTION, (("r", "report"), ("m", "term")),
                        reports, terms),
        "IMDB-2": join_q(DISCUSS_ASPECT, (("r", "body"), ("a", "aspect")),
                         reviews, ASPECTS),
    }


def _run_one(torch, arena, pipeline, async_ans, budget, q):
    """One query, chunk-instrumented. Returns (summary, chunks)."""
    import time

    from quail.executor.attention import (FILTER_ATTENTION,
                                          JOIN_ATTENTION)
    from quail.executor.loop import run_filter, run_join

    trace = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        if q["kind"] == "filter":
            pipeline.attention_mode = FILTER_ATTENTION
            # single stage, no store: the cold pass runs the fast path
            _, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, q["bodies"],
                [q["qids"]], budget, trace=trace, arena_writes=False)
        else:
            pipeline.attention_mode = JOIN_ATTENTION
            _, spans, tokens = run_join(
                torch, arena, pipeline, async_ans, q["prefixes"],
                [q["suffixes"]], budget, stage_frames=[q["frame"]],
                trace=trace)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    assert len(spans) == len(trace), (len(spans), len(trace))
    chunks = [dict(tokens=tr["tokens"],
                   gpu_ms=round(e0.elapsed_time(e1), 3),
                   pieces=tr["pieces"])
              for (_, e0, e1), tr in zip(spans, trace)]
    gpu_s = sum(c["gpu_ms"] for c in chunks) / 1e3
    summary = dict(kind=q["kind"], chunks=len(chunks),
                   fresh_tokens=tokens, wall_s=round(wall, 2),
                   gpu_s=round(gpu_s, 2),
                   us_per_token_wall=round(wall / tokens * 1e6, 3),
                   us_per_token_gpu=round(gpu_s / tokens * 1e6, 3))
    return summary, chunks


@app.function(**GPU_KW, timeout=3600)
def sweep(model_name: str, only=None, tag: str = "full") -> str:
    import socket
    import time

    from quail.executor.loop import run_join, warm_kernels
    from quail.planner.calibrate import _boot, resolve_pair

    spec, device = resolve_pair(model_name, "h100-sxm")
    t0 = time.perf_counter()
    (torch, tokenizer, pipeline, arena, async_ans,
     budget) = _boot(spec, device)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    queries = _queries(tok)
    ids = [i for i in QUERY_IDS if only is None or i in only]
    # warm both kernel paths outside the measured walls
    filler = tok("The document discusses a clinical finding. ")
    doc = (filler * ((512 // len(filler)) + 1))[:512]
    qsuf = tok("\n\nAnswer TRUE or FALSE.\nANSWER:")
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, [doc] * 64,
                     [qsuf], budget)
        if any(queries[i]["kind"] == "join" for i in ids):
            from quail.executor.attention import JOIN_ATTENTION
            pipeline.attention_mode = JOIN_ATTENTION
            run_join(torch, arena, pipeline, async_ans, [doc] * 4,
                     [[qsuf] * 8], budget)
    torch.cuda.synchronize()
    boot_s = round(time.perf_counter() - t0, 2)

    record = dict(model=model_name, device="h100-sxm", tag=tag,
                  budget=budget, boot_s=boot_s,
                  host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), queries={})
    summaries = {}
    for qid in ids:
        q = queries[qid]
        summary, chunks = _run_one(torch, arena, pipeline, async_ans,
                                   budget, q)
        lengths = (dict(body_tokens=[len(b) for b in q["bodies"]],
                        q_tokens=len(q["qids"]))
                   if q["kind"] == "filter" else
                   dict(prefix_tokens=[len(p) for p in q["prefixes"]],
                        suffix_tokens=[len(s) for s in q["suffixes"]],
                        frame_tokens=len(q["frame"])))
        record["queries"][qid] = dict(summary=summary, chunks=chunks,
                                      **lengths)
        summaries[qid] = summary
        print(f"[packing_sweep] {qid} {summary}", flush=True)

    os.makedirs("/results/ablations", exist_ok=True)
    path = f"/results/ablations/packing_sweep_{model_name}_{tag}.json"
    with open(path, "w") as f:
        json.dump(record, f)
    results_vol.commit()
    kernel_cache.commit()
    print(f"[packing_sweep] raw record: {path}", flush=True)
    return json.dumps(dict(model=model_name, tag=tag, boot_s=boot_s,
                           host=record["host"], raw=path,
                           queries=summaries))


@app.local_entrypoint()
def run(model: str = "qwen3-4b-fp8", queries: str = "",
        reps: int = 1, tag: str = ""):
    only = [q.strip() for q in queries.split(",") if q.strip()] or None
    base = tag or ("full" if reps == 1 else "rep")
    calls = [sweep.spawn(model, only,
                         base if reps == 1 else f"{base}{i}")
             for i in range(reps)]
    for c in calls:
        # house rule: keep the fc- id in the tee file; results can be
        # re-pulled with modal.FunctionCall.from_id(id).get()
        print(f"[packing_sweep] fc={c.object_id}", flush=True)
    for c in calls:
        print(c.get(), flush=True)
