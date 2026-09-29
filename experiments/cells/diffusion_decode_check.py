"""Decode DiffusionGemma's own AGENT-4 outcome answers with vLLM.

The prompt is the one Quail scores: the outcome classification's
prompt head, the trace, and its tail with the category list and the
answer cue, in DiffusionGemma's chat turn. vLLM decodes the answer on
a 16-row canvas with its default denoising steps; the answer text is
matched to a label as the stock vLLM baseline matches it.

    uv run modal run --detach -m experiments.cells.diffusion_decode_check \
        --prediction "State the expected result before starting." \
        2>&1 | tee <scratchpad>/diffusion-decode-check.log

Writes /results/ablations/diffusion-gemma-agent4-decode.parquet, one
row per trace (id, decoded text, matched label or null), and a JSON
summary beside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

OUT_DIR = Path("/results/ablations")
ROWS_PATH = OUT_DIR / "diffusion-gemma-agent4-decode.parquet"
SUMMARY_PATH = OUT_DIR / "diffusion-gemma-agent4-decode.json"
CANVAS_ROWS = 16
# under vLLM's 128-sequence setting that caps a diffusion model at 8
MAX_SEQUENCES = 127


def outcome_prompt(limit: int):
    """The outcome classification's labels and the first traces' prompt ids."""
    import quail
    from quail import EngineConfig
    from quail.bench.quailb import _build
    from quail.bench.substrait import read_plan
    from quail.physical import AiClassify
    from quail.specs import DIFFUSION_GEMMA_26B_FP8
    from quail_b.data import load_table
    from quail_b.queries import get_query

    table = load_table("agent_traces", scale_factor=0.1)
    session = quail.Session(EngineConfig(model=DIFFUSION_GEMMA_26B_FP8.name,
                                         device="h100-sxm"))
    session.register("agent_traces",
                     quail.DocumentProvider.from_table(table, id_col="id"))
    plan = _build(session, read_plan(get_query("AGENT-4").plan)).plan()
    outcome = next(node for node in plan.nodes
                   if isinstance(node, AiClassify)).spec
    head, tail = outcome.prompt_token_parts
    encode = session.tokenizer
    ids = table.column("id").to_pylist()[:limit]
    traces = table.column("trace").to_pylist()[:limit]
    prompts = [list(head) + encode(trace) + list(tail) for trace in traces]
    session.close()
    return outcome.labels, ids, prompts


@app.function(image=image, gpu="H100!", memory=98304, timeout=2 * 3600,
              volumes=volumes)
def decode(limit: int, prediction: str) -> dict:
    """Decode the first ``limit`` traces' outcome answers and save them."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams

    from quail.backends.request import match_label
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    labels, ids, prompts = outcome_prompt(limit)
    started = time.perf_counter()
    llm = LLM(model=DIFFUSION_GEMMA_26B_FP8.hf_name,
              diffusion_config={"canvas_length": CANVAS_ROWS},
              max_num_seqs=MAX_SEQUENCES, enable_prefix_caching=True,
              gpu_memory_utilization=0.91, disable_log_stats=True)
    boot_s = time.perf_counter() - started
    started = time.perf_counter()
    outputs = llm.generate([{"prompt_token_ids": ids_} for ids_ in prompts],
                           SamplingParams(max_tokens=CANVAS_ROWS),
                           use_tqdm=False)
    decode_s = time.perf_counter() - started
    texts = [output.outputs[0].text or "" for output in outputs]
    matched = [match_label(text, labels) for text in texts]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "id": pa.array([str(i) for i in ids]),
        "text": pa.array(texts),
        "label": pa.array(matched, pa.string()),
    }), ROWS_PATH)
    counts = {}
    for label in matched:
        counts[str(label)] = counts.get(str(label), 0) + 1
    summary = {
        "prediction": prediction,
        "model": DIFFUSION_GEMMA_26B_FP8.hf_name,
        "canvas_rows": CANVAS_ROWS,
        "traces": len(ids),
        "labels": list(labels),
        "label_counts": counts,
        "prompt_tokens_mean": sum(map(len, prompts)) / max(1, len(prompts)),
        "boot_s": round(boot_s, 1),
        "decode_s": round(decode_s, 1),
        "rows": str(ROWS_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(prediction: str = "", limit: int = 500):
    """Spawn the decode and print its function call id."""
    if not prediction:
        raise SystemExit("state the prediction with --prediction")
    call = decode.spawn(limit, prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"rows: {ROWS_PATH}", flush=True)
