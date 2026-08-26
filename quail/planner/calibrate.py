"""Measure calibration constants (a, a2) for a (model, device) pair
by sweeping document lengths through the packed filter loop.
"""

from quail.planner import budgets
from quail.planner.calibration import (Calibration, fit_affine,
                                       load_calibration, make_record)
from quail.specs import DEVICES, MODELS, DeviceSpec, ModelSpec

LENGTHS = (256, 1024, 4096, 8192)
TOKENS_PER_POINT = 1_500_000

# Any short TRUE/FALSE suffix works; the answers are not graded.
_QUESTION = ("\n\nDoes the document mention a finding? "
             "Answer TRUE or FALSE.\nANSWER:")
_FILLER = "The document discusses a clinical finding. "


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
    """Load the model and build the pipeline, arena, and answerers."""
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
    """Measure host-device copy bandwidth, pinned and unpinned."""
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


def measure(model: ModelSpec, device: DeviceSpec,
            tokens_per_point: int = TOKENS_PER_POINT,
            lengths: tuple[int, ...] = LENGTHS,
            loaded: Calibration | None = None) -> dict:
    """Sweep document lengths through the packed filter and fit a, a2.

    Runs on GPU. `loaded` supplies the previous calibration constants
    so this function does not need to read the JSON files itself.
    """
    import time

    from quail.executor.loop import run_filter, warm_kernels

    loaded = loaded or load_calibration(model, device)
    spec, _ = resolve_pair(model, device)
    (torch, tokenizer, pipeline, arena, async_ans,
     exec_budget) = _boot(model, device)
    q_ids = [tokenizer(_QUESTION, add_special_tokens=False)["input_ids"]]
    stream = _token_ids(tokenizer, max(lengths) + tokens_per_point)
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, exec_budget,
                     model_name=spec.hf_name)
    torch.cuda.synchronize()

    points, rows = [], []
    for h in lengths:
        n_docs = max(1, min(max(8, tokens_per_point // h),
                         len(stream) // h))
        body_ids = [stream[i * h:(i + 1) * h] for i in range(n_docs)]
        t0 = time.perf_counter()
        with torch.inference_mode():
            # single stage, no store: matches the fast path
            _, spans, tokens = run_filter(
                torch, arena, pipeline, async_ans, body_ids, q_ids,
                exec_budget, arena_writes=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        gpu_s = sum(e0.elapsed_time(e1) for _, e0, e1 in spans) / 1e3
        t_wall = wall / tokens
        points.append((h, t_wall))
        rows.append(dict(doc_tokens=h, n_docs=n_docs,
                         fresh_tokens=tokens, wall_s=round(wall, 2),
                         gpu_s=round(gpu_s, 2),
                         us_per_token_wall=round(t_wall * 1e6, 3),
                         us_per_token_gpu=round(gpu_s / tokens * 1e6,
                                                3)))
        print(f"[calibrate] {rows[-1]}", flush=True)

    a, a2 = fit_affine(points)
    a2 = max(a2, 0.0)
    channels = _probe_channels(torch)
    return make_record(model, device, a, a2, rows, channels, loaded,
                       lengths, tokens_per_point)
