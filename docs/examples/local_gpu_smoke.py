"""Run one filter query on the GPU in this process, without Modal.

Uses the default compute provider, so the model loads onto the GPU
visible to this process. Prints the matching rows and the parts of the
report that show where the time went.

    uv run python docs/examples/local_gpu_smoke.py 2>&1 | tee local_gpu_smoke.log
"""

import pyarrow as pa

import quail

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
    with quail.Session() as session:
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
