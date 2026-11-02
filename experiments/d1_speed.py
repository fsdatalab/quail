"""IMDB-11 on d1-2B: Liquid's compiled runtime against Quail, and Quail's profile.

    uv run modal run experiments/d1_speed.py 2>&1 | tee "$log"

Pass --only liquid or --only quail to run one cell.

Both cells run on one H100 SXM over the 4,995 IMDB-11 reviews at scale
factor 0.1 and its question. The Liquid cell times
``model.system_one_batch`` after ``model.compile()``, at its default
token budget and at Quail's chunk budget. The Quail cell times
``classify_passes``, the code IMDB-11's classification runs, then
profiles one more run with the torch profiler and prints the CUDA time
by kernel and by kind. Both print one JSON line starting with
``RESULT``; the profiler trace is saved on the results volume.
"""

import json
import os
import subprocess
import sys
import time

import modal

from quail.deploy.images import gpu_image

REFERENCE_PACKAGES = "/opt/d1-reference"
image = gpu_image(
    ("experiments", "/root/experiments"),
    commands=(f"/usr/local/bin/python3 -m pip install --quiet --target "
              f"{REFERENCE_PACKAGES} transformers==5.19.0",))
app = modal.App("quail-milestone1")
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
REVIEWS = "/results/quailb_data/sf0.1/reviews.parquet"
TRACE_DIR = "/results/profiles/d1_imdb11"
QUESTION = ("Judge strictly from the review above which category best "
            "describes the overall sentiment it expresses about the movie.")
LABELS = ["positive", "negative", "neutral", "mixed"]
# Quail's chunk budget for d1-2B on an H100 (explain(): chunk budget)
QUAIL_CHUNK = 139_810


def _reviews() -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(REVIEWS, columns=["body"]).column("body").to_pylist()


def liquid_main(out_path: str) -> None:
    """Time Liquid's compiled runtime; runs with transformers 5.19 on the path."""
    import torch
    from d1_spec import HF_NAME, REVISION  # written next to this file by the cell
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_NAME, revision=REVISION,
                                      trust_remote_code=True,
                                      dtype=torch.bfloat16).cuda().eval()
    reviews = _reviews()
    questions = {"sentiment": {"type": "choice", "instructions": QUESTION,
                               "criteria": {label: None for label in LABELS}}}
    requests = [(review, questions) for review in reviews]
    started = time.perf_counter()
    model.compile()
    engine = model.engine
    rows = []
    for budget in (engine.token_budget, QUAIL_CHUNK):
        engine.token_budget = budget
        # compile and warm on every batch shape once
        model.system_one_batch(requests)
        torch.cuda.synchronize()
        times = []
        for _ in range(2):
            begin = time.perf_counter()
            done = model.system_one_batch(requests)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
        tokens = sum(out["usage"]["input_tokens"] for out in done)
        best = min(times)
        rows.append(dict(token_budget=budget, seconds=best, runs=times,
                         decisions_per_second=len(reviews) / best,
                         input_tokens=tokens, tokens_per_second=tokens / best))
    with open(out_path, "w") as f:
        json.dump(dict(reviews=len(reviews), setup_s=time.perf_counter() - started,
                       budgets=rows), f)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def liquid_runtime() -> dict:
    """Liquid's d1-2B runtime over the IMDB-11 reviews, compiled."""
    from quail.specs import D1_2B_BF16

    folder = "/root/experiments"
    with open(f"{folder}/d1_spec.py", "w") as f:
        f.write(f"HF_NAME = {D1_2B_BF16.hf_name!r}\n"
                f"REVISION = {D1_2B_BF16.revision!r}\n")
    out = "/tmp/liquid.json"
    paths = [REFERENCE_PACKAGES, folder, "/root", os.environ.get("PYTHONPATH", "")]
    environment = dict(os.environ, PYTHONPATH=":".join(filter(None, paths)))
    subprocess.run([sys.executable, f"{folder}/d1_speed.py", out], check=True,
                   env=environment)
    with open(out) as f:
        result = json.load(f)
    print("RESULT liquid " + json.dumps(result), flush=True)
    return result


def _quail_state():
    """Quail's d1-2B model and the execution state classify_passes reads."""
    from types import SimpleNamespace

    import torch

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.set_passes import StateTails
    from quail.catalog.tokens import GigatokenEncoder
    from quail.cost import budgets
    from quail.planner.prompt_lowering import answer_ids
    from quail.specs import D1_2B_BF16, H100_SXM
    from quail.specs.files import resolve_model_path

    spec, device = D1_2B_BF16, H100_SXM
    encoder = GigatokenEncoder(resolve_model_path(spec.hf_name, spec.revision))
    true_ids, false_ids = answer_ids(spec, encoder)
    model = load_model(spec.hf_name, revision=spec.revision,
                       answer_token_ids=list(true_ids) + list(false_ids))
    chunk = budgets.chunk_budget(spec, device)
    full, sliding = budgets.arena_pages(spec, device, chunk)
    arena = KVArena(num_layers=spec.layers, num_pages=full,
                    page_tokens=device.page_tokens,
                    num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                    dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    num_sliding_pages=sliding)
    state = SimpleNamespace(
        torch=torch, chunk_tokens=chunk, gpu_timing=False,
        loaded_model=SimpleNamespace(
            model=model, arena=arena, pipeline=build_pipeline(spec, model, arena),
            model_spec=spec, state_tails=StateTails(encoder), input_staging=None))
    return state, encoder


def _classify_spec(encoder):
    """IMDB-11's classification as a set_choice spec over the reviews."""
    from types import SimpleNamespace

    from quail.logical import ColumnRef
    from quail.logical.prompts import bind_classify_prompt
    from quail.planner.prompt_lowering import lower_prompt, option_spellings
    from quail.specs import D1_2B_BF16

    spec = D1_2B_BF16
    prompt = bind_classify_prompt(
        QUESTION + "\n\n{0}", (ColumnRef(alias="r", provider="reviews",
                                         column="body"),),
        LABELS, (), turn=spec.turn, layout=spec.prompt_layout)
    lowered = lower_prompt(prompt, encoder)
    return SimpleNamespace(
        aliases=("r",), labels=tuple(LABELS), probabilities=False,
        prompt_token_parts=(tuple(lowered.preamble_token_ids),
                            tuple(lowered.tail_token_ids)),
        state_tail_tokens=len(encoder(lowered.prompt.tail_segments[0])),
        question_header_token_ids=tuple(encoder("QUESTION:\n")),
        label_token_ids=option_spellings(lowered.prompt.letters, encoder))


KINDS = (  # first match wins: substrings of kernel names, lower case
    ("attention", ("flash", "fmha", "attn")),
    ("matrix multiply", ("gemm", "cutlass", "sm90_xmma", "nvjet", "ampere_",
                         "cublas")),
    ("convolution", ("conv",)),
    ("copy", ("memcpy", "memset", "copy", "cat", "index", "gather", "scatter")),
    ("elementwise and norms", ("elementwise", "vectorized", "reduce", "norm",
                               "softmax", "triton")),
)


def _kind(name: str) -> str:
    lower = name.lower()
    return next((kind for kind, marks in KINDS if any(m in lower for m in marks)),
                "other")


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def quail_profile() -> dict:
    """Quail's IMDB-11 classification passes: timed, then profiled once."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from quail.backends.quail.executor.operators.set_classify import (
        classify_passes,
    )

    state, encoder = _quail_state()
    spec = _classify_spec(encoder)
    documents = {"r": [encoder(review) for review in _reviews()]}
    rows = list(range(len(documents["r"])))
    classify_passes(state, [spec], rows, documents)    # warm
    times = []
    for _ in range(2):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        (labels,) = classify_passes(state, [spec], rows, documents)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    tokens = labels.tokens
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        classify_passes(state, [spec], rows, documents)
        torch.cuda.synchronize()
    os.makedirs(TRACE_DIR, exist_ok=True)
    trace = f"{TRACE_DIR}/quail_classify_passes.json.gz"
    prof.export_chrome_trace(trace)
    kernels = [event for event in prof.key_averages()
               if event.device_type.name == "CUDA" or event.self_device_time_total]
    by_kind, by_name = {}, {}
    for event in kernels:
        microseconds = event.self_device_time_total
        if not microseconds:
            continue
        by_name[event.key] = by_name.get(event.key, 0) + microseconds
        kind = _kind(event.key)
        by_kind[kind] = by_kind.get(kind, 0) + microseconds
    total = sum(by_kind.values())
    best = min(times)
    result = dict(
        reviews=len(rows), seconds=best, runs=times,
        decisions_per_second=len(rows) / best, fresh_tokens=tokens,
        tokens_per_second=tokens / best, cuda_seconds=total / 1e6,
        by_kind={kind: round(us / total, 4) for kind, us in
                 sorted(by_kind.items(), key=lambda item: -item[1])},
        top_kernels=[(name[:120], round(us / 1e6, 3)) for name, us in
                     sorted(by_name.items(), key=lambda item: -item[1])[:30]],
        trace=trace)
    modal.Volume.from_name("quail-results").commit()
    print("RESULT quail " + json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(only: str = ""):
    cells = {"liquid": liquid_runtime, "quail": quail_profile}
    calls = {name: cell.spawn() for name, cell in cells.items()
             if not only or name == only}
    for name, call in calls.items():
        print(f"function call id ({name}): {call.object_id}", flush=True)
    for name, call in calls.items():
        result = call.get()
        print(f"RESULT {name} " + json.dumps(result), flush=True)


if __name__ == "__main__":
    liquid_main(sys.argv[1])
