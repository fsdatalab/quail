"""IMDB-11 on d1-2B: Liquid's compiled runtime against Quail, and Quail's profile.

    uv run modal run experiments/d1_speed.py 2>&1 | tee "$log"

Pass --only liquid or --only quail to run one cell.

Both cells run on one H100 SXM over the 4,995 IMDB-11 reviews at scale
factor 0.1 and its question. The Liquid cell times
``model.system_one_batch`` after ``model.compile()``, at its default
token budget and at Quail's chunk budget. The Quail cell times
``run_pass_group`` for IMDB-11's classification alone, and one
mixed pass group (a yes/no, a choice, and a read in each review's
pass), then profiles one more run of each with the torch profiler. It
prints the CUDA time by kernel and by kind, and the host share: the
part of wall time no kernel ran. Both print one JSON line starting with
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
    """Quail's d1-2B model and the execution state its set passes read."""
    from types import SimpleNamespace

    import torch

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows
    from quail.common import budgets
    from quail.common.tokens import GigatokenEncoder
    from quail.physical.lowering import answer_ids
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
        answer_rows=AnswerRows(torch, torch.nn.functional, model, true_ids,
                               false_ids),
        loaded_model=SimpleNamespace(
            model=model, arena=arena, pipeline=build_pipeline(spec, model, arena),
            model_spec=spec, input_staging=None))
    return state, encoder


def _classify_spec(encoder):
    """IMDB-11's classification as a ClassifySpec over the reviews."""
    from quail.common.prompts import bind_classify_prompt
    from quail.logical import ColumnRef
    from quail.physical import ClassifySpec
    from quail.physical.lowering import lower_prompt, option_spellings
    from quail.specs import D1_2B_BF16

    spec = D1_2B_BF16
    prompt = bind_classify_prompt(
        QUESTION + "\n\n{0}", (ColumnRef(alias="r", provider="reviews",
                                         column="body"),),
        LABELS, (), turn=spec.turn, layout=spec.prompt_layout)
    lowered = lower_prompt(prompt, encoder)
    return ClassifySpec(
        name="label", aliases=("r",), query_template=QUESTION, arguments=(),
        expected_rows=0.0, estimated_seconds=0.0,
        labels=tuple(LABELS), probabilities=False,
        prompt_token_parts=(tuple(lowered.preamble_token_ids),
                            tuple(lowered.tail_token_ids)),
        state_tail_tokens=len(encoder(lowered.prompt.tail_segments[0])),
        question_header_token_ids=tuple(encoder("QUESTION:\n")),
        label_token_ids=option_spellings(lowered.prompt.letters, encoder))


def _mixed_questions(encoder, choice):
    """A yes/no, IMDB-11's choice, and a read, as one pass group's SetQuestions."""
    from quail.backends.quail.executor.pass_groups import SetQuestion, node_question
    from quail.common.prompts import PROMPT_LAYOUTS, bind_extract_prompt
    from quail.logical import ColumnRef, bind_prompt
    from quail.physical import AiClassify
    from quail.physical.lowering import (
        lower_prompt,
        read_slot_offsets,
        set_question_parts,
    )
    from quail.specs import D1_2B_BF16

    spec = D1_2B_BF16
    lowered = lower_prompt(bind_prompt(
        "Does the review praise the acting? {0}", ("body",), turn=spec.turn,
        layout=spec.prompt_layout), encoder)
    parts = set_question_parts(lowered, spec, encoder)
    tail, cut = tuple(lowered.tail_token_ids), parts["state_tail_tokens"]
    yes_no = SetQuestion("yes_no", tuple(lowered.preamble_token_ids), tail[:cut],
                         tail[cut:], parts["question_header_token_ids"])
    lowered = lower_prompt(bind_extract_prompt(
        (ColumnRef(alias="r", provider="reviews", column="body"),),
        "The actor the reviewer singles out", spec.turn, spec.prompt_layout),
        encoder)
    segments = lowered.prompt.tail_segments
    tail, cut = tuple(lowered.tail_token_ids), len(encoder(segments[0]))
    read = SetQuestion(
        "read", tuple(lowered.preamble_token_ids), tail[:cut], tail[cut:],
        tuple(encoder("QUESTION:\n")), slots=read_slot_offsets(
            segments[-1], PROMPT_LAYOUTS[spec.prompt_layout].read_slots, encoder))
    return [yes_no, node_question(AiClassify(node_id="label", spec=choice)), read]


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


def _profile(name, run, documents, tokens_of):
    """Time run() twice after a warm run, then profile it once."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    run()
    times = []
    for _ in range(2):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        out = run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    os.makedirs(TRACE_DIR, exist_ok=True)
    trace = f"{TRACE_DIR}/quail_{name}.json.gz"
    prof.export_chrome_trace(trace)
    by_kind, by_name = {}, {}
    for event in prof.key_averages():
        # a CPU op's device time repeats its kernels' time
        microseconds = (event.self_device_time_total
                        if event.device_type.name == "CUDA" else 0)
        if not microseconds:
            continue
        by_name[event.key] = by_name.get(event.key, 0) + microseconds
        kind = _kind(event.key)
        by_kind[kind] = by_kind.get(kind, 0) + microseconds
    total = sum(by_kind.values())
    best = min(times)
    tokens = tokens_of(out)
    return dict(
        reviews=documents, seconds=best, runs=times,
        decisions_per_second=documents / best, fresh_tokens=tokens,
        tokens_per_second=tokens / best, cuda_seconds=total / 1e6,
        host_share=max(0.0, 1 - total / 1e6 / best),
        by_kind={kind: round(us / total, 4) for kind, us in
                 sorted(by_kind.items(), key=lambda item: -item[1])},
        top_kernels=[(name[:120], round(us / 1e6, 3)) for name, us in
                     sorted(by_name.items(), key=lambda item: -item[1])[:20]],
        trace=trace)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def quail_profile() -> dict:
    """IMDB-11's classification, then one mixed pass group: timed and profiled.

    The host share is 1 minus CUDA kernel seconds over wall seconds: the
    part of the run no kernel ran, an upper bound on host overhead.
    """
    from dataclasses import replace

    from quail.backends.quail.executor.pass_groups import run_pass_group
    from quail.common.document_ends import document_ends
    from quail.common.tokens import DocumentPrefixes

    state, encoder = _quail_state()
    spec = _classify_spec(encoder)
    documents = {"r": [encoder(review) for review in _reviews()]}
    rows = list(range(len(documents["r"])))
    # the plan's joins of the reviews' endings with the state tail
    first = _mixed_questions(encoder, spec)[0]
    ends = document_ends(sorted({document[-1] for document in documents["r"]}
                                | set(first.head[-1:])), first.state_tail, encoder)
    questions = [replace(question, ends=ends)
                 for question in _mixed_questions(encoder, spec)]
    prefixes = DocumentPrefixes(first.head, documents["r"], rows)
    result = dict(
        classify=_profile(
            "classify", lambda: run_pass_group(state, questions[1:2], prefixes),
            len(rows), lambda group: group.tokens),
        mixed=_profile(
            "mixed", lambda: run_pass_group(state, questions, prefixes),
            len(rows), lambda group: group.tokens))
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
