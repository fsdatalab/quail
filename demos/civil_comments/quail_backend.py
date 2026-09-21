"""Run the Civil Comments comparison query with Quail.

The query joins comments to 30 semantic fields and filters for toxicity.
The package initializer shares the sample, prompts, and scoring code with
the Jev backend.

Run one or two H100s from the repository root:

    uv run modal run --detach demos/civil_comments/quail_backend.py \
      --limit 10000 --gpus 1 \
      2>&1 | tee /tmp/civil-comments-quail.log

Results are saved under ``/demos/civil-comments/quail/<run-id>`` on the
``quail-results`` Modal volume. ``--limit 0`` uses all comments.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

import modal
import numpy as np
import pyarrow.parquet as pq

from demos.civil_comments import (
    DATASET,
    DATASET_REVISION,
    FILTER_PROMPT,
    JOIN_PROMPT,
    accuracy_summary,
    fields_table,
    load_comments,
    requested_input_tokens,
)
from quail.bench.images import gpu_image

MODEL = "diffusion-gemma-26b-a4b-fp8"
DEVICE = "h100-sxm"
RESULTS_MOUNT = Path("/results")
RESULTS_VOLUME_DIR = Path("demos/civil-comments/quail")

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
image = gpu_image(("demos", "/root/demos")).add_local_python_source("demos")


def result_volume_path(directory: Path) -> str:
    """Return a path relative to the Modal volume root."""
    return f"/{directory.relative_to(RESULTS_MOUNT)}"


def build_sql() -> str:
    """Return the shared Civil Comments query."""
    return f"""
    SELECT c.comment_id, f.field
    FROM comments c
    JOIN fields f
      ON AI_FILTER(PROMPT('{JOIN_PROMPT}', c.text, f.statement))
    WHERE AI_FILTER(PROMPT('{FILTER_PROMPT}', c.text))
"""


def json_ready(value):
    """Convert numpy scalars for JSON output."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(directory: Path, limit: int | None, gpus: int) -> dict:
    """Run and save one Quail configuration."""
    import quail
    from quail.specs import H100_USD_PER_HOUR

    comments = load_comments(limit)
    ids = comments["comment_id"].to_pylist()
    sql = build_sql()
    (directory / "inputs.json").write_text(
        json.dumps(
            {
                "backend": "quail",
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "model": MODEL,
                "device": DEVICE,
                "gpus": gpus,
                "limit": limit,
                "comments": len(ids),
                "sql": sql,
            },
            indent=2,
        )
    )
    pq.write_table(
        comments.select(["comment_id", "text"]),
        directory / "comments.parquet",
    )
    with quail.Session(
        config=quail.EngineConfig(
            model=MODEL,
            device=DEVICE,
            gpus=gpus,
            gpu_timing=True,
        )
    ) as session:
        session.register(
            "comments",
            quail.DocumentProvider.from_table(comments, id_col="comment_id"),
        )
        session.register(
            "fields",
            quail.DocumentProvider.from_table(fields_table(), id_col="field"),
        )
        query = session.sql(sql)
        explanation = query.explain()
        print(explanation, flush=True)
        (directory / "plan.txt").write_text(explanation)
        result = query.run()
        table = result.collect()
        report = dict(result.report)

    answers = result.answer_tables["filters"][("c", 0)]
    pq.write_table(answers, directory / "filter_answers.parquet")
    toxic_found = {
        ids[int(index)]
        for index, yes in zip(
            answers["c"].to_pylist(),
            answers["answer"].to_pylist(),
        )
        if yes
    }
    input_tokens = requested_input_tokens(
        comments,
        filter_ids=set(ids),
        join_ids=toxic_found,
    )
    pq.write_table(table, directory / "retained.parquet")
    pairs_found = {
        (row["c.comment_id"], row["f.field"]) for row in table.to_pylist()
    }
    accuracy = accuracy_summary(comments, toxic_found, pairs_found)
    wall_s = report["wall_s"]
    boot_s = report.get("boot_s", 0.0)
    evaluated_pairs = sum(
        stage["tuples"]
        for stage in report.get("stages", ())
        if stage.get("op") == "join"
    )
    summary = {
        **report,
        "backend": "quail",
        "model": MODEL,
        "device": DEVICE,
        "gpus": gpus,
        "comments": len(ids),
        "limit": limit,
        "input_tokens": input_tokens,
        "input_tokens_per_second": input_tokens / wall_s,
        "evaluated_pairs": evaluated_pairs,
        "gpu_cost_usd": wall_s * gpus * H100_USD_PER_HOUR / 3600,
        "gpu_startup_cost_usd": (
            boot_s * gpus * H100_USD_PER_HOUR / 3600
        ),
        "accuracy": accuracy,
        "field_counts": dict(Counter(field for _, field in pairs_found)),
        "result_volume_path": result_volume_path(directory),
    }
    summary = json_ready(summary)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "backend": "quail",
                "gpus": gpus,
                "wall_s": wall_s,
                "input_tokens": input_tokens,
                "input_tokens_per_second": input_tokens / wall_s,
                "gpu_cost_usd": summary["gpu_cost_usd"],
                "filter": accuracy["filter"],
                "join": accuracy["join"],
                "result_volume_path": result_volume_path(directory),
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


@app.function(
    image=image,
    gpu="H100!",
    timeout=86_400,
    memory=98_304,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
    },
)
def run(limit: int, gpus: int) -> dict:
    """Run Quail on Modal."""
    results_volume.reload()
    hf_cache.reload()
    directory = (
        RESULTS_MOUNT / RESULTS_VOLUME_DIR / uuid.uuid4().hex
    )
    directory.mkdir(parents=True)
    summary = evaluate(directory, None if limit == 0 else limit, gpus)
    results_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def main(limit: int = 10_000, gpus: int = 1):
    """Run the Quail configuration. ``--limit 0`` uses every comment."""
    if limit < 0 or gpus not in (1, 2, 4, 8):
        raise ValueError("limit must be >= 0; gpus must be 1, 2, 4, or 8")
    gpu = "H100!" if gpus == 1 else f"H100!:{gpus}"
    call = run.with_options(gpu=gpu).spawn(limit, gpus)
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
