r"""Smoke test for extensions registered inside a Modal GPU function.

Registers the observer in experiments/cells/row_trace.py, runs one
filter query in process, and checks that
the observer's report, the executed plan, and the per node metrics
come back. Run from the repository root as a module:

    uv run modal run -m experiments.cells.extension_smoke 2>&1 \
        | tee results/extension_smoke.log

Prints the query's explain and the cost ledger totals.
"""

import json
import tempfile
import uuid
from pathlib import Path

import modal

import quail
from experiments.cells.row_trace import RowTrace
from experiments.cells.session_smoke import FILTER_Q, make_filter_parquet
from quail.planner.plan import EngineConfig
from quail_ext_examples import cost_ledger

app = modal.App("quail-milestone1")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow",
                 "sqlglot>=27.0", "bpe-qwen>=0.1.5", "datasets>=5.0.1")
    .env({
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
    })
    .add_local_python_source("quail", "quail_b", "experiments", "quail_ext_examples")
)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/results": results_vol,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": kernel_cache,
}


@app.function(
    image=image,
    gpu="H100!", memory=98304, volumes=volumes, timeout=1200,
)
def run_query():
    tmp = tempfile.mkdtemp()
    flags = make_filter_parquet(f"{tmp}/docs.parquet", n_docs=40)
    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        RowTrace)
    session = quail.Session(
        EngineConfig(gpus=1), registry=registry,
    )
    session.register("docs", quail.DocumentProvider.from_parquet(
        f"{tmp}/docs.parquet", id_col="id"))
    q1_text = FILTER_Q.replace("{j}", "1")
    query = session.sql(f"""
        SELECT d.id FROM docs d
        WHERE AI_FILTER(PROMPT('{{0}}{q1_text}', d.body),
                        {{'selectivity': 0.6}})
    """)
    result = query.run()
    destination = Path("/results/extension-smoke") / f"{uuid.uuid4().hex}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.report["result_volume_path"] = str(destination)
    destination.write_text(json.dumps(result.report))
    results_vol.commit()

    planted = sorted(f"d{i}" for i in range(len(flags)) if flags[i][0])
    got = sorted(row[0] for row in result.to_rows())
    print("rows:", len(got), "planted:", len(planted),
          "agree:", len(set(got) & set(planted)), flush=True)
    print("explain:\n" + result.explain(), flush=True)
    print("observer:", json.dumps(result.observer(RowTrace)), flush=True)
    print("ledger totals:", json.dumps(cost_ledger.charge(result)["totals"]),
          flush=True)
    print("node_metrics:", json.dumps({
        node_id: metrics.wall_s
        for node_id, metrics in result.node_metrics.items()}), flush=True)
    session.close()
    kernel_cache.commit()


@app.local_entrypoint()
def main():
    call = run_query.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
