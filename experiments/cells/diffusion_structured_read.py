"""Read AGENT-4's labels from DiffusionGemma the way vLLM PR 57250 does.

vLLM 0.30.1rc0 serves the model with a 64-row canvas, and the pull
request's example server, `structured_server.py` at that tag, runs in
front of it unchanged. Each trace is one request to the example server.
Its schema asks the outcome, then the failure mode only when the outcome
is "not resolved", as AGENT-4 does. The server relabels the options A to
H, seeds the canvas with the answer template, leaves each answer slot as
noise, runs one denoising step, reads the letters' probabilities at the
slot, and averages up to four noise draws when the first read is unsure.

    uv run modal run --detach -m experiments.cells.diffusion_structured_read \
        --prediction "State the expected result before starting." \
        2>&1 | tee <scratchpad>/diffusion-structured-read.log

Writes /results/ablations/diffusion-gemma-agent4-structured.parquet, one
row per trace, and a JSON summary beside it that compares the labels with
the Qwen3 32B reference labels, Quail's canvas run, and vLLM's full
decode of Quail's prompt.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from quail.bench.images import UV_VERSION, _cuda_base

# v0.30.1rc0, the first release that holds PR 57250
VLLM_COMMIT = "153242a314153637999eb6ebe8dd830e63433bb6"
VLLM_WHEEL = (f"https://wheels.vllm.ai/{VLLM_COMMIT}/"
              "vllm-0.30.1rc0-cp38-abi3-manylinux_2_28_x86_64.whl")
SERVER_URL = (f"https://raw.githubusercontent.com/vllm-project/vllm/"
              f"{VLLM_COMMIT}/examples/features/structured_diffusion/"
              "structured_server.py")
SERVER_PATH = "/root/structured_server.py"

app = modal.App("quail-milestone1")
image = (_cuda_base()
         .uv_sync(groups=["dev"], uv_version=UV_VERSION,
                  extra_options="--no-install-package vllm")
         .uv_pip_install(VLLM_WHEEL, uv_version=UV_VERSION)
         .run_commands(
             "python -c \"import urllib.request; urllib.request.urlretrieve("
             f"'{SERVER_URL}', '{SERVER_PATH}')\"")
         .add_local_python_source("quail"))
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

OUT_DIR = Path("/results/ablations")
ROWS_PATH = OUT_DIR / "diffusion-gemma-agent4-structured.parquet"
SUMMARY_PATH = OUT_DIR / "diffusion-gemma-agent4-structured.json"
# Quail's canvas run of AGENT-4 at sf 0.1 and vLLM's full decode of it
QUAIL_RUN = Path("/results/benchmarks/quailb/20260929T071645Z-23970920"
                 "/quail/agent/AGENT-4")
DECODES = {"outcome": OUT_DIR / "diffusion-gemma-agent4-decode.parquet",
           "failure_mode": OUT_DIR / "diffusion-gemma-agent4-failure.parquet"}
KEYS = {"outcome": "quailb.agent.trace.outcome",
        "failure_mode": "quailb.agent.trace.failure_mode"}
VLLM_PORT = 8010
SERVER_PORT = 8011
CANVAS = 64
# the example server reads up to 16 question groups at once
CLIENTS = 32
# vLLM sizes an fp32 logits buffer of sequences x canvas rows x the
# 262,144-token vocabulary at startup: 127 sequences need 7.9 GiB and
# run out of memory beside the model and KV, 32 need 2 GiB
MAX_SEQUENCES = 32


def schema() -> dict:
    """AGENT-4's two questions in the example server's schema."""
    from quail_b.predicates import PREDICATE_BY_KEY, _descriptions

    questions = []
    for name, key in KEYS.items():
        spec = PREDICATE_BY_KEY[key]
        question = {
            "id": name, "type": "choice",
            "instructions": spec.template.replace("{0}", "").strip(),
            "options": [{"name": label, "description": description or None}
                        for label, description
                        in zip(spec.labels, _descriptions(spec))],
        }
        if name == "failure_mode":
            question["ask_if"] = {"outcome": ["not resolved"]}
        questions.append(question)
    return {"questions": questions}


def wait_for(url: str, process, seconds: int) -> None:
    """Poll ``url`` until it answers, or raise if ``process`` exits."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{url} process exited with {process.returncode}")
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(url)


def ask(body: dict) -> dict:
    """POST one structured request and return its answer set."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    return json.loads(reply["choices"][0]["message"]["content"])


def agreement(mine: dict, other: dict) -> dict:
    """Matching labels over the ids both label."""
    shared = [key for key in mine if key in other]
    same = sum(mine[key] == other[key] for key in shared)
    return {"evaluated": len(shared), "correct": same,
            "agreement": round(same / len(shared), 4) if shared else None}


def labels_of(table, id_column: str, label_column: str) -> dict:
    """Id -> label, for rows with a label."""
    return {str(row_id): label for row_id, label in zip(
        table.column(id_column).to_pylist(),
        table.column(label_column).to_pylist()) if label is not None}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3 * 3600,
              volumes=volumes)
def read(prediction: str, limit: int) -> dict:
    """Serve the model, read every trace's labels, and compare them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail_b
    from quail.specs import DIFFUSION_GEMMA_26B_FP8
    from quail_b.data import load_table
    from quail_b.predicates import PREDICATE_BY_KEY

    model = DIFFUSION_GEMMA_26B_FP8.hf_name
    table = load_table("agent_traces", scale_factor=0.1)
    ids = [str(i) for i in table.column("id").to_pylist()][:limit]
    traces = table.column("trace").to_pylist()[:limit]
    started = time.perf_counter()
    vllm = subprocess.Popen([
        "vllm", "serve", model, "--served-model-name", "dgemma",
        "--diffusion-config", json.dumps({"canvas_length": CANVAS}),
        "--max-logprobs", "32", "--enable-prefix-caching",
        "--max-num-seqs", str(MAX_SEQUENCES), "--port", str(VLLM_PORT)])
    wait_for(f"http://127.0.0.1:{VLLM_PORT}/health", vllm, 3600)
    server = subprocess.Popen([
        "python", SERVER_PATH, "--upstream", f"http://127.0.0.1:{VLLM_PORT}",
        "--model", "dgemma", "--tokenizer", model, "--canvas", str(CANVAS),
        "--port", str(SERVER_PORT)])
    time.sleep(20)
    if server.poll() is not None:
        raise RuntimeError(f"structured server exited with {server.returncode}")
    boot_s = time.perf_counter() - started
    system = json.dumps(schema())

    def one(index: int) -> dict:
        body = {"model": "dgemma", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"trace": traces[index]})}]}
        begun = time.perf_counter()
        try:
            answers = ask(body)
        except Exception as error:  # recorded per trace
            return {"id": ids[index], "error": str(error)[:300]}
        row = {"id": ids[index], "ms": (time.perf_counter() - begun) * 1e3}
        diagnostics = answers["diagnostics"]
        for name in KEYS:
            answer = answers["answers"].get(name)
            row[name] = None if answer is None else answer["choice"]
            row[f"{name}_confidence"] = (None if answer is None
                                         else answer["confidence"])
        row["reads"] = diagnostics["timing"]["reads"]
        return row

    started = time.perf_counter()
    with ThreadPoolExecutor(CLIENTS) as pool:
        rows = list(pool.map(one, range(len(ids))))
    read_s = time.perf_counter() - started
    server.terminate()
    vllm.terminate()

    errors = [row for row in rows if "error" in row]
    columns = ["id", "outcome", "outcome_confidence", "failure_mode",
               "failure_mode_confidence", "reads", "ms"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: [row.get(column) for row in rows]
                             for column in columns}), ROWS_PATH)

    suite = quail_b.load_benchmark(["AGENT-4"], scale_factor=0.1)
    truth = suite.ground_truth
    comparisons = {}
    for index, name in enumerate(KEYS):
        mine = {row["id"]: row[name] for row in rows
                if row.get(name) is not None}
        reference = truth.predicates[truth.key_for_template(
            PREDICATE_BY_KEY[KEYS[name]].template)].table
        quail_labels = labels_of(
            pq.read_table(QUAIL_RUN / f"classifications-{index}.parquet"),
            "t", "label")
        decoded = labels_of(pq.read_table(DECODES[name]), "id", "label")
        counts = {}
        for label in mine.values():
            counts[label] = counts.get(label, 0) + 1
        comparisons[name] = {
            "labeled": len(mine),
            "label_counts": counts,
            "reference": agreement(mine, labels_of(reference, "left_id",
                                                   "label")),
            "quail_canvas": agreement(mine, quail_labels),
            "vllm_decode": agreement(mine, decoded),
            "quail_canvas_against_reference": agreement(
                quail_labels, labels_of(reference, "left_id", "label")),
        }
    summary = {
        "prediction": prediction,
        "model": model,
        "vllm": "0.30.1rc0",
        "vllm_commit": VLLM_COMMIT,
        "canvas": CANVAS,
        "traces": len(ids),
        "errors": len(errors),
        "first_errors": [row["error"] for row in errors[:3]],
        "reads_mean": round(sum(row.get("reads", 0) for row in rows)
                            / max(1, len(rows) - len(errors)), 3),
        "boot_s": round(boot_s, 1),
        "read_s": round(read_s, 1),
        "comparisons": comparisons,
        "rows": str(ROWS_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(prediction: str = "", limit: int = 100_000):
    """Spawn the read and print its function call id."""
    if not prediction:
        raise SystemExit("state the prediction with --prediction")
    call = read.spawn(prediction, limit)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"rows: {ROWS_PATH}", flush=True)
