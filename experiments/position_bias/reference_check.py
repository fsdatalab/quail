"""Check whether the post's top-k reference is Qwen3-32B's pointwise top 25.

Scores the 225 SALES-5 calls with AI.SCORE and QUAIL-B's SALES_COMMITS
prompt on Qwen3-32B and on Qwen3-4B, in one container, and saves both
score lists. If the post's reference is Qwen3-32B's pointwise top 25,
Qwen3-4B's pointwise top 25 overlaps it on about 18 of 25 calls (0.72).

    uv run modal run -m experiments.position_bias.reference_check \
        2>&1 | tee results/position_bias/reference_check.log

Record: /results/position_bias/reference_check.json on the quail-results
volume.
"""

import json
from pathlib import Path

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}
MODELS = ("qwen3-32b-fp8", "qwen3-4b-fp8")
K = 25


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def score_calls() -> dict:
    """Score every SALES-5 call on each model; return id -> score per model."""
    from contextlib import ExitStack

    import pyarrow.compute as pc

    import quail
    from quail_b.data import load_table
    from quail_b.prompts import SALES_COMMITS

    calls = load_table("sales_calls", scale_factor=0.1)
    calls = calls.filter(pc.equal(calls["deal_stage"], "Negotiation"))
    template = SALES_COMMITS.replace("'", "''")
    scores = {}
    for model in MODELS:
        with ExitStack() as stack:
            session = stack.enter_context(quail.Session(
                quail.EngineConfig(model=model, device="h100-sxm")))
            session.register("calls", quail.DocumentProvider.from_table(
                calls.select(["id", "transcript"]), id_col="id"))
            result = session.sql(
                f"SELECT c.id, AI.SCORE(PROMPT('{template}', c.transcript)) "
                f"AS score FROM calls c").collect()
        scores[model] = dict(zip(map(str, result.column(0).to_pylist()),
                                 result.column(1).to_pylist()))
        print(f"[{model}] scored {len(scores[model])} calls", flush=True)
    path = Path("/results/position_bias/reference_check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores))
    results.commit()
    return scores


def top(scores: dict, k: int) -> set:
    """The k highest-scoring ids, ties broken by id."""
    return set(sorted(scores, key=lambda key: (-scores[key], key))[:k])


@app.local_entrypoint()
def main():
    """Score the calls, then print the top-25 overlap of Qwen3-4B with Qwen3-32B."""
    call = score_calls.spawn()
    print(f"score_calls: {call.object_id}", flush=True)
    scores = call.get()
    reference = top(scores["qwen3-32b-fp8"], K)
    candidate = top(scores["qwen3-4b-fp8"], K)
    print(f"overlap@{K} of Qwen3-4B pointwise with Qwen3-32B pointwise: "
          f"{len(reference & candidate)}/{K} = {len(reference & candidate) / K:.2f} "
          f"(post: 0.72)", flush=True)
