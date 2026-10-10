"""Boot Qwen3.8 27B on stock vLLM and run one planted filter query.

Qwen3.8 27B is registered for the stock vLLM backend only. This cell
checks that the pinned vLLM loads the FP8 checkpoint through that
backend and that the TRUE/FALSE answers come back: a two-predicate
AI_FILTER over 200 synthetic documents, each ending in a [FLAGS] line
that states the answers.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/qwen3_8_boot.py \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/qwen3_8-boot.log

Writes /results/ablations/qwen3_8_27b_boot_<run id>.json: the boot
record, the planner's report, the elapsed time, and the agreement with
the planted answers.
"""

import json
import time
import uuid

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

MODEL = "qwen3.8-27b-fp8"
BACKEND = "stock_vllm"
SEED = 20260818
DOCUMENTS = 200
RATES = (0.6, 0.5)
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")
FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
            "then FLAG_9 has value FALSE.\nInstruction: output only the value "
            "of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}=")


def _write_corpus(path):
    """Write the planted documents and return their flags."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(SEED)
    flags = rng.random((DOCUMENTS, len(RATES))) < np.array(RATES)
    bodies = []
    for row in flags:
        line = " ".join(f"FLAG_{j + 1}={'TRUE' if flag else 'FALSE'}"
                        for j, flag in enumerate(row))
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({"id": [f"d{i}" for i in range(DOCUMENTS)],
                             "body": bodies}), path)
    return flags


@app.function(image=image, gpu="H100!", memory=98304, timeout=3 * 3600,
              volumes=volumes)
def boot(run_id: str) -> dict:
    """Boot the model, run the filter, and save the summary."""
    import tempfile
    from pathlib import Path

    import quail
    from quail.planner.plan import EngineConfig

    tmp = tempfile.mkdtemp()
    flags = _write_corpus(f"{tmp}/docs.parquet")
    session = quail.Session(EngineConfig(
        model=MODEL, backend=BACKEND, device="h100-sxm", gpus=1))
    session.register("docs", quail.DocumentProvider.from_parquet(
        f"{tmp}/docs.parquet", id_col="id"))
    q1 = FILTER_Q.replace("{j}", "1")
    q2 = FILTER_Q.replace("{j}", "2")
    query = session.sql(f"""
        SELECT d.id FROM docs d
        WHERE AI_FILTER(PROMPT('{{0}}{q1}', d.body), {{'selectivity': 0.6}})
          AND AI_FILTER(PROMPT('{{0}}{q2}', d.body), {{'selectivity': 0.5}})
    """)
    print(query.explain(), flush=True)
    started = time.perf_counter()
    result = query.run()
    elapsed = time.perf_counter() - started
    planted = {f"d{i}" for i in range(DOCUMENTS) if flags[i].all()}
    got = {row[0] for row in result.to_rows()}
    summary = {
        "run_id": run_id,
        "model": MODEL,
        "backend": BACKEND,
        "documents": DOCUMENTS,
        "planted_survivors": len(planted),
        "rows": len(got),
        "agree_with_planted": len(got & planted),
        "query_seconds_with_boot": elapsed,
        "report": result.report,
    }
    session.close()
    path = Path(f"/results/ablations/qwen3_8_27b_boot_{run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    results_vol.commit()
    summary["path"] = str(path)
    return summary


@app.local_entrypoint()
def main(prediction: str):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    call = boot.spawn(run_id)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2, default=str), flush=True)
