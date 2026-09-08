"""Run one filter over four documents on the GPU in this process.

Run with: uv run python demos/local_gpu_smoke.py 2>&1 | tee local_gpu_smoke.log
For RTX, add --device rtx-pro-6000-blackwell-server. Use --gpus to select
1, 2, 4, or 8 model copies on the host.
"""

import argparse

import pyarrow as pa

import quail
from quail.specs import DEVICES

REPORTS = pa.table({
    "report_id": ["r1", "r2", "r3", "r4"],
    "report": [
        "A 34-year-old woman reported nausea after starting the medication.",
        "A 71-year-old man reported dizziness after taking the medication.",
        "The patient reported a rash, but the report does not state their sex.",
        "A 52-year-old woman had no side effects during the trial.",
    ],
})

SQL = """
    SELECT r.report_id
    FROM reports AS r
    WHERE AI.IF(
        PROMPT('Does this report describe a female patient?\\n\\n{0}', r.report),
        {'selectivity': 0.5}
    )
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=sorted(DEVICES), default="h100-sxm")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()
    config = quail.EngineConfig(device=args.device, gpus=args.gpus)
    with quail.Session(config) as session:
        print("compute provider:", type(session.compute_provider).__name__)
        session.register(
            "reports",
            quail.DocumentProvider.from_table(REPORTS, id_col="report_id"),
        )
        result = session.sql(SQL, dialect="bq").run()
        print("rows:", result.to_rows())
        for key in ("backend", "boot_kind", "boot_s", "wall_s",
                    "fresh_tokens", "worker_total_s"):
            print(f"{key}: {result.report.get(key)}")


if __name__ == "__main__":
    main()
