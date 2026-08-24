"""Measure a, a2, c, p for a (model, device) pair.

The four-constant cost model at the chunk level:

    gpu_seconds = a * T  +  a2 * S  +  c  +  p * suffixes

Pins (a, a2, c) from filter chunks at varying document lengths, and p
from join chunks with synthetic anchors and suffixes. Each chunk is
one data point; the fit is ordinary least squares over all chunks.

Nothing here talks to Modal. The GPU entry that calls measure() is
quail/runtime/calibrate.py, attached to the quail-engine app.

    uv run modal run quail/runtime/calibrate.py --model qwen3-4b-fp8 \
        --device h100-sxm --commit 2>&1 | tee results/calibrate.log
"""

from quail.planner import budgets
from quail.planner.calibration import (Calibration, fit_cost_model,
                                       load_calibration, make_record)
from quail.specs import DEVICES, MODELS, DeviceSpec, ModelSpec

FILTER_LENGTHS = (256, 1024, 4096, 8192)
TOKENS_PER_POINT = 1_500_000
JOIN_N_ANCHORS = 32
JOIN_ANCHOR_LENGTH = 2048
JOIN_N_SUFFIXES = 256
JOIN_SUFFIX_LENGTH = 128

_FILLER = "The document discusses a clinical finding. "
_QUESTION = ("\n\nDoes the document mention a finding? "
             "Answer TRUE or FALSE.\nANSWER:")
_FRAME = "[partner 1] "


def resolve_pair(model: str, device: str) -> tuple[ModelSpec, DeviceSpec]:
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r}; known: {sorted(MODELS)}")
    if device not in DEVICES:
        raise ValueError(
            f"unknown device {device!r}; known: {sorted(DEVICES)}")
    return MODELS[model], DEVICES[device]


def _token_ids(tokenizer, n: int) -> list[int]:
    chunk = tokenizer(_FILLER, add_special_tokens=False)["input_ids"]
    reps = -(-n // len(chunk))
    return (chunk * reps)[:n]


def _boot(spec: ModelSpec, device: DeviceSpec):
    """Model, pipeline, arena, answerers, sized by the plan arithmetic."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import Answerer, AsyncAnswers
    from quail.executor.model import load_model

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = load_model(spec.hf_name)
    chunk = budgets.chunk_budget(spec, device)
    arena_tok = budgets.arena_tokens(spec, device, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena,
                        attention_mode=FILTER_ATTENTION)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)
    return (torch, tokenizer, pipeline, arena, async_ans, exec_budget)


def _probe_channels(torch) -> dict:
    """2 GiB timed copies, pinned and unpinned, both directions."""
    import time

    channels = {}
    buf_bytes = 2 << 30
    dev_buf = torch.empty(buf_bytes, dtype=torch.uint8, device="cuda")
    for pinned in (True, False):
        host = torch.empty(buf_bytes, dtype=torch.uint8,
                           pin_memory=pinned)
        name = "pinned" if pinned else "unpinned"
        for tag, src, dst in ((f"{name}_h2d", host, dev_buf),
                              (f"{name}_d2h", dev_buf, host)):
            dst.copy_(src)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize()
            channels[tag] = round(
                3 * buf_bytes / (time.perf_counter() - t0), 0)
        del host
    del dev_buf
    return channels


def _filter_chunk_points(spans, trace):
    """Per-chunk (T, S, suffixes, gpu_s) from a single-stage filter
    run's trace. Each piece is one document+question at stage 0;
    S = sum of n^2 per piece (causal attention, /2 in a2)."""
    points = []
    for (_, e0, e1), tr in zip(spans, trace):
        T = S = 0.0
        for doc, stage, fresh in tr["pieces"]:
            T += fresh
            S += fresh * fresh
        points.append(dict(T=T, S=S, suffixes=0,
                           gpu_s=e0.elapsed_time(e1) / 1e3))
    return points


def _join_chunk_points(spans, trace, prefix_lengths, suffix_lengths,
                       frame_length):
    """Per-chunk (T, S, suffixes, gpu_s) from a join run's trace.
    S includes both causal self-attention (n^2 per segment) and
    cross-read pairs (each suffix token attends to every anchor +
    frame token)."""
    points = []
    for (_, e0, e1), tr in zip(spans, trace):
        T = S = 0.0
        n_suf = 0
        for a, start, end, carried in tr["pieces"]:
            h0 = prefix_lengths[a]
            if carried:
                T += h0
                S += h0 * h0
            if start == 0 and end > start:
                T += frame_length
                S += frame_length * frame_length + frame_length * h0
            for k in range(start, end):
                n = suffix_lengths[k]
                T += n
                S += n * n + n * (h0 + frame_length)
            n_suf += end - start
        points.append(dict(T=T, S=S, suffixes=n_suf,
                           gpu_s=e0.elapsed_time(e1) / 1e3))
    return points


def measure(model: ModelSpec, device: DeviceSpec,
            tokens_per_point: int = TOKENS_PER_POINT,
            filter_lengths: tuple[int, ...] = FILTER_LENGTHS,
            loaded: Calibration | None = None) -> dict:
    """Run filter and join sweeps, fit (a, a2, c, p) from per-chunk
    GPU time, and probe the host copy channels.

    Needs a GPU and the executor. Call from the Modal entry, not from
    the coordinator.
    """
    import time

    from quail.executor.attention import (FILTER_ATTENTION,
                                          JOIN_ATTENTION)
    from quail.executor.loop import run_filter, run_join, warm_kernels

    loaded = loaded or load_calibration(model, device)
    (torch, tokenizer, pipeline, arena, async_ans,
     exec_budget) = _boot(model, device)
    q_ids = tokenizer(_QUESTION, add_special_tokens=False)["input_ids"]
    stream = _token_ids(tokenizer,
                        max(filter_lengths) + tokens_per_point)

    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans,
                     [stream[:512]] * 64, [q_ids], exec_budget)
        pipeline.attention_mode = JOIN_ATTENTION
        run_join(torch, arena, pipeline, async_ans,
                 [stream[:512]] * 4, [[q_ids] * 8], exec_budget)
        pipeline.attention_mode = FILTER_ATTENTION
    torch.cuda.synchronize()

    all_points = []
    rows = []

    # --- filter sweep: chunks at varying document lengths ---
    for h in filter_lengths:
        n_docs = max(1, min(max(8, tokens_per_point // h),
                         len(stream) // h))
        body_ids = [stream[i * h:(i + 1) * h] for i in range(n_docs)]
        trace = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            _, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids,
                [q_ids], exec_budget, trace=trace, arena_writes=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        pts = _filter_chunk_points(spans, trace)
        gpu_s = sum(p["gpu_s"] for p in pts)
        all_points.extend(pts)
        rows.append(dict(kind="filter", doc_tokens=h, n_docs=n_docs,
                         n_chunks=len(pts), fresh_tokens=tokens,
                         wall_s=round(wall, 2), gpu_s=round(gpu_s, 2),
                         us_per_token_gpu=round(gpu_s / tokens * 1e6,
                                                3)))
        print(f"[calibrate] {rows[-1]}", flush=True)

    # --- join sweep: synthetic anchors + suffixes ---
    pipeline.attention_mode = JOIN_ATTENTION
    anchor_ids = [_token_ids(tokenizer, JOIN_ANCHOR_LENGTH)
                  ] * JOIN_N_ANCHORS
    prefix_lengths = [len(a) for a in anchor_ids]
    suffix_body = _token_ids(tokenizer, JOIN_SUFFIX_LENGTH)
    suffix_ids = [suffix_body + q_ids] * JOIN_N_SUFFIXES
    suffix_lengths = [len(s) for s in suffix_ids]
    frame_ids = tokenizer(_FRAME, add_special_tokens=False)["input_ids"]
    frame_length = len(frame_ids)
    trace = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        _, spans, tokens = run_join(
            torch, arena, pipeline, async_ans, anchor_ids,
            [suffix_ids], exec_budget, stage_frames=[frame_ids],
            trace=trace)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    pts = _join_chunk_points(spans, trace, prefix_lengths,
                             suffix_lengths, frame_length)
    gpu_s = sum(p["gpu_s"] for p in pts)
    all_points.extend(pts)
    n_dispatches = JOIN_N_ANCHORS * JOIN_N_SUFFIXES
    rows.append(dict(kind="join", n_anchors=JOIN_N_ANCHORS,
                     anchor_length=JOIN_ANCHOR_LENGTH,
                     n_suffixes=JOIN_N_SUFFIXES,
                     suffix_length=JOIN_SUFFIX_LENGTH,
                     n_dispatches=n_dispatches,
                     n_chunks=len(pts), fresh_tokens=tokens,
                     wall_s=round(wall, 2), gpu_s=round(gpu_s, 2),
                     us_per_token_gpu=round(gpu_s / tokens * 1e6, 3)))
    print(f"[calibrate] {rows[-1]}", flush=True)
    pipeline.attention_mode = FILTER_ATTENTION

    # drop the first chunk: potential kernel-compile outlier
    if len(all_points) > 4:
        all_points = all_points[1:]

    a, a2, c, p = fit_cost_model(all_points)
    a2 = max(a2, 0.0)
    c = max(c, 0.0)
    p = max(p, 0.0)

    channels = _probe_channels(torch)
    return make_record(model, device, a, a2, c, p,
                       rows, channels, loaded)
