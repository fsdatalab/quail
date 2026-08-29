"""Check Arrow token transport through one filter and one join on H100!.

Prediction: the worker decodes Arrow IPC without materializing full Python
token lists, executes the filter and join, and returns a result without an
out-of-memory error.

Run from the repository root:

    uv run python tests/gpu/arrow_token_transport.py 2>&1 \
      | tee results/arrow_token_transport.log

The worker writes its measured run summary to the quail-results volume at
the result_volume_path printed by this script.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.planner.decide import _collect
from quail.planner.plan import EngineConfig, Refusal


def _write_inputs(directory: str) -> tuple[str, str]:
    reports = str(Path(directory) / "reports.parquet")
    candidates = str(Path(directory) / "candidates.parquet")
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2", "r3"],
        "body": [
            "[KEEP=TRUE] The named color is blue. " * 64,
            "[KEEP=FALSE] The named color is red. " * 64,
            "[KEEP=TRUE] The named color is green. " * 64,
            "[KEEP=FALSE] The named color is yellow. " * 64,
        ],
    }), reports)
    pq.write_table(pa.table({
        "id": ["c0", "c1", "c2", "c3"],
        "body": [
            "The candidate color is blue.",
            "The candidate color is red.",
            "The candidate color is green.",
            "The candidate color is yellow.",
        ],
    }), candidates)
    return reports, candidates


def main() -> None:
    directory = tempfile.mkdtemp()
    reports, candidates = _write_inputs(directory)
    print("prepared Arrow inputs", flush=True)
    session = quail.Session(EngineConfig(gpus=1))
    session.register("reports", quail.DocumentProvider.from_parquet(
        reports, id_col="id"))
    session.register("candidates", quail.DocumentProvider.from_parquet(
        candidates, id_col="id"))

    query = session.sql("""
        SELECT r.id, c.id FROM reports r
        JOIN candidates c
          ON AI_FILTER(PROMPT(
              'Judge whether the color named in {0} is the candidate color named in {1}. Output only TRUE or FALSE.',
              r.body, c.body), {'selectivity': 0.25})
        WHERE AI_FILTER(PROMPT(
            'Read the KEEP marker in {0}. Output only its TRUE or FALSE value.',
            r.body), {'selectivity': 0.5})
    """)
    print("compiled SQL", flush=True)
    plan = query.plan()
    print("planned query", flush=True)
    if isinstance(plan, Refusal):
        raise RuntimeError(str(plan))
    scans, filters, joins = _collect(query.logical)
    payload = query._payload(plan, scans, filters, joins)
    print("built Arrow IPC payload", flush=True)
    if not all(isinstance(value, bytes)
               for value in payload["docs"].values()):
        raise AssertionError("document tokens are not Arrow IPC bytes")

    worker = session.worker()
    print("started Modal app", flush=True)
    call = worker.execute.spawn(payload)
    print(f"function call id: {call.object_id}", flush=True)
    output = call.get()
    print(f"result volume path: {output['result_volume_path']}", flush=True)
    print(f"fresh tokens: {output['fresh_tokens']}", flush=True)
    print(f"filter rows: {len(output['filters']['r'])}", flush=True)
    print(f"join stages: {len(output['joins'])}", flush=True)
    session.close()


if __name__ == "__main__":
    main()
