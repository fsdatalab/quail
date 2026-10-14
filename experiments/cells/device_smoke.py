"""Smoke Qwen3 4B filter and join on H100, L40S, and B200.

Prediction: both attention paths complete on every device. Filter ids
and join pairs match the H100 run. L40S uses FA2 and Triton; B200 uses
FA4 and DeepGEMM; H100 stays FA3 and DeepGEMM. No Refusal.

    mkdir -p results
    log="results/$(date -u +%Y%m%dT%H%M%SZ)-device-smoke.log"
    uv run modal run --detach experiments/cells/device_smoke.py \
      --devices h100-sxm,l40s,b200 \
      2>&1 | tee "$log"

Results go to /results/ablations/device_smoke/<stamp>/.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quail.bench.images import gpu_image
from quail.specs import DEVICES, modal_gpu_type

PREDICTION_TEXT = (
    "H100, L40S, and B200 each complete unified and tree. Filter and "
    "join answers match H100. L40S: FA2, no DeepGEMM. B200: FA4, "
    "DeepGEMM. H100: FA3, DeepGEMM."
)
FILTER_PROMPT = (
    "\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
    "then FLAG_9 has value FALSE.\nInstruction: output only the value "
    "of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}="
)
FILLER = (
    "The projector hummed while the reel changed and nobody in "
    "the back row noticed the splice. ")
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")
SEED = 20260818
PATHS = ("unified", "tree")

app = modal.App("quail-milestone1")
image = gpu_image()
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {
    "/results": results_vol,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}


def _write_tables(tmp: str) -> None:
    """Write the planted filter and join tables."""
    rng = np.random.default_rng(SEED)
    flags = rng.random((40, 2)) < np.array((0.6, 0.5))
    bodies = []
    for i in range(40):
        line = " ".join(
            f"FLAG_{j + 1}={'TRUE' if flags[i, j] else 'FALSE'}"
            for j in range(2))
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(40)],
        "body": bodies,
    }), f"{tmp}/docs.parquet")
    keys = len(COLORS)
    reports = [
        FILLER * 30 + f"\n\nThe dominant color in this scene is "
                      f"{COLORS[i % keys]}."
        for i in range(keys)]
    cands = [f"The candidate color is {COLORS[j]}." for j in range(keys)]
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(keys)],
        "report": reports,
    }), f"{tmp}/reports.parquet")
    pq.write_table(pa.table({
        "id": [f"c{j}" for j in range(keys)],
        "body": cands,
    }), f"{tmp}/cands.parquet")


def _node_attentions(plan) -> list[str]:
    """Return attention values on filter and join nodes."""
    from quail.physical import AiFilter, AiJoin

    return [node.attention for node in plan.graph.nodes
            if isinstance(node, (AiFilter, AiJoin))]


@app.function(
    image=image, gpu=modal_gpu_type("h100-sxm"), memory=98304, timeout=3600,
    volumes=volumes)
def smoke(device: str, run_dir: str) -> dict:
    """Run filter and join on both attention paths. Return the answers."""
    import torch

    import quail
    from quail.backends.quail.executor.attention import (
        deep_gemm_supported,
        flash_attention_version,
    )
    from quail.planner.plan import EngineConfig, Refusal

    capability = tuple(torch.cuda.get_device_capability(0))
    tmp = tempfile.mkdtemp()
    _write_tables(tmp)
    paths = {}
    for attention in PATHS:
        session = quail.Session(EngineConfig(
            model="qwen3-4b-fp8", device=device, gpus=1,
            backend="quail", attention=attention))
        session.register("docs", quail.DocumentProvider.from_parquet(
            f"{tmp}/docs.parquet", id_col="id"))
        session.register("reports", quail.DocumentProvider.from_parquet(
            f"{tmp}/reports.parquet", id_col="id"))
        session.register("cands", quail.DocumentProvider.from_parquet(
            f"{tmp}/cands.parquet", id_col="id"))
        q1 = FILTER_PROMPT.replace("{j}", "1")
        q2 = FILTER_PROMPT.replace("{j}", "2")
        filter_query = session.sql(f"""
            SELECT d.id FROM docs d
            WHERE AI_FILTER(PROMPT('{{0}}{q1}', d.body),
                            {{'selectivity': 0.6}})
              AND AI_FILTER(PROMPT('{{0}}{q2}', d.body),
                            {{'selectivity': 0.5}})
        """)
        filter_plan = filter_query.plan()
        if isinstance(filter_plan, Refusal):
            session.close()
            raise RuntimeError(f"{device} {attention} filter refused: "
                               f"{filter_plan}")
        filter_result = filter_query.run()
        join_query = (
            session.docs("reports").alias("r")
            .ai_join(
                session.docs("cands").alias("c"),
                quail.prompt(
                    "Judge strictly from {0} whether it says its "
                    "dominant color is the color named in {1}. "
                    "Answer TRUE if it does, FALSE otherwise."
                    "\nANSWER=",
                    quail.col("r.report"), quail.col("c.body")),
                selectivity=1 / 6)
            .select("r.id", "c.id"))
        join_plan = join_query.plan()
        if isinstance(join_plan, Refusal):
            session.close()
            raise RuntimeError(f"{device} {attention} join refused: "
                               f"{join_plan}")
        join_result = join_query.run()
        paths[attention] = {
            "forced": attention,
            "filter_attentions": _node_attentions(filter_plan),
            "join_attentions": _node_attentions(join_plan),
            "filter_ids": sorted(row[0] for row in filter_result.to_rows()),
            "join_pairs": sorted(list(row) for row in join_result.to_rows()),
            "filter_wall_s": filter_result.report.get("wall_s"),
            "join_wall_s": join_result.report.get("wall_s"),
            "filter_boot_s": filter_result.report.get("boot_s"),
        }
        session.close()
    record = {
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "flash_attention_version": flash_attention_version(capability),
        "deep_gemm_supported": deep_gemm_supported(capability),
        "prediction": PREDICTION_TEXT,
        "paths": paths,
    }
    path = Path(run_dir) / f"{device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    results_vol.commit()
    print(json.dumps(record, indent=2, default=str), flush=True)
    record["volume_path"] = str(path)
    return record


def _same_answers(left: dict, right: dict) -> bool:
    """True if filter ids and join pairs match."""
    return (left["filter_ids"] == right["filter_ids"]
            and left["join_pairs"] == right["join_pairs"])


@app.local_entrypoint()
def main(devices: str = "h100-sxm,l40s,b200"):
    """Submit one smoke per device and compare answers to H100."""
    selected = [item.strip() for item in devices.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DEVICES]
    if unknown:
        raise ValueError(f"unknown devices: {unknown}")
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"/results/ablations/device_smoke/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    calls = {
        device: smoke.with_options(gpu=modal_gpu_type(device)).spawn(
            device, run_dir)
        for device in selected
    }
    for device, call in calls.items():
        print(f"function call id: {call.object_id} ({device})", flush=True)
    records = {device: call.get() for device, call in calls.items()}
    reference = records.get("h100-sxm")
    if reference is None:
        print("no H100 run; skip answer compare", flush=True)
        return
    for device, record in records.items():
        for attention in PATHS:
            match = _same_answers(
                record["paths"][attention], reference["paths"][attention])
            print(
                f"{device} {attention}: "
                f"{'match H100' if match else 'MISMATCH H100'} "
                f"filter={record['paths'][attention]['filter_ids']!s} "
                f"pairs={len(record['paths'][attention]['join_pairs'])} "
                f"plan={record['paths'][attention]['filter_attentions']}/"
                f"{record['paths'][attention]['join_attentions']}",
                flush=True)
        if device != "h100-sxm":
            cross = all(
                _same_answers(record["paths"][attention],
                              reference["paths"][attention])
                for attention in PATHS)
            print(f"{device} vs H100: "
                  f"{'match' if cross else 'MISMATCH'}", flush=True)
