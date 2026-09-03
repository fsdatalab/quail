"""Check Modal Function transport through one filter and one join on H100!.

Prediction: one warm Modal Function container completes two sequential filter
and join queries. Both calls return the same final Arrow rows, and neither call
crashes during PyTorch cleanup.

The remote source check predicts that the client sends only a logical plan and
Hugging Face source description. The Modal worker should read and tokenize all
five source rows, then return one projected row because the query has LIMIT 1.

Run from the repository root:

    uv run python tests/gpu/modal_function_transport.py 2>&1 \
      | tee results/modal_function_transport.log

    uv run python tests/gpu/modal_function_transport.py remote 2>&1 \
      | tee results/remote_source_transport.log

The worker writes its measured run summary to the quail-results volume at
the result_volume_path printed by this script.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.planner.plan import EngineConfig


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
    with quail.Session(EngineConfig(gpus=1)) as session:
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
        expected = None
        for index in range(2):
            result = query.run()
            rows = result.to_rows()
            if expected is None:
                expected = rows
            elif rows != expected:
                raise AssertionError("sequential calls returned different rows")
            print(f"finished Modal Function request {index + 1}", flush=True)
            print(
                f"result volume path: {result.report['result_volume_path']}",
                flush=True,
            )
            print(
                f"fresh tokens: {result.report['fresh_tokens']}",
                flush=True,
            )
            print(f"result rows: {len(rows)}", flush=True)


def remote_source_main() -> None:
    """Check remote reading, tokenization, planning, and projection."""
    with quail.Session(EngineConfig(gpus=1)) as session:
        session.register(
            "reviews",
            quail.DocumentProvider.from_hf(
                "lhoestq/demo1",
                id_col="id",
                split="test",
            ),
        )
        query = session.sql("""
            SELECT r.id FROM reviews r
            WHERE AI_FILTER(PROMPT(
                'Does {0} describe a positive software experience?',
                r.review), {'selectivity': 0.5})
            LIMIT 1
        """)
        request = query._request()
        if request.providers["reviews"].remote_source() is None:
            raise AssertionError("Hugging Face source was not remote")
        print(
            f"remote source descriptions: {len(request.providers)}",
            flush=True,
        )
        result = query.run()
        if result.count() != 1:
            raise AssertionError(
                f"remote query returned {result.count()} rows, expected 1"
            )
        print(f"remote result rows: {result.count()}", flush=True)
        print(
            f"worker total seconds: {result.report['worker_total_s']}",
            flush=True,
        )
        print(
            f"result volume path: {result.report['result_volume_path']}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["remote"]:
        remote_source_main()
    else:
        main()
