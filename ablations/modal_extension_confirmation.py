"""Confirm that an external physical node runs through Modal.

Run from the repository root and tee every line:

    uv run python ablations/modal_extension_confirmation.py \
      --prediction "The extension node will count all four documents." \
      2>&1 | tee results/modal-extension-confirmation.log

The Modal worker writes its result record under `/results/runs/` on the
`quail-results` volume and returns that volume path in the query report.
"""

import argparse
import json

import pyarrow as pa

import quail


def main(prediction: str) -> None:
    """Run one filter with an extension node in its remote graph."""
    print(f"prediction: {prediction}", flush=True)
    registry = quail.ExtensionRegistry.with_built_ins()
    registry.load_extension(
        "quail_ext_examples.count_documents",
        local_python_sources=("quail_ext_examples",),
    )
    with quail.Session(registry=registry) as session:
        session.register("docs", quail.DocumentProvider.from_table(
            pa.table({
                "id": ["d0", "d1", "d2", "d3"],
                "body": [
                    "A unit test fails after a parser change.",
                    "The weather is clear today.",
                    "A type check fails in the query planner.",
                    "The package release completed successfully.",
                ],
            }),
            id_col="id",
            identity="modal-extension-confirmation",
        ))
        query = session.sql("""
            SELECT d.id
            FROM docs d
            WHERE AI_FILTER(PROMPT(
              'Does document {0} describe a software failure?', d.body),
              {'selectivity': 0.5})
        """)
        plan = query.plan()
        result = query.run()
        report = result.report
        count_node = next(
            node for node in plan.nodes
            if node.type_name == "example.count_documents.v1"
        )
        measured_count = report["node_metrics"][
            count_node.node_id
        ]["evaluated_documents"]
        output = {
            "prediction": prediction,
            "extension_modules": list(registry.extension_modules),
            "extension_node": count_node.node_id,
            "measured_documents": measured_count,
            "input_documents": 4,
            "query_rows": result.count(),
            "query_wall_s": report["wall_s"],
            "result_volume_path": report["result_volume_path"],
        }
        print(json.dumps(output, indent=2), flush=True)
        if measured_count != 4:
            raise RuntimeError(
                f"extension counted {measured_count} documents, expected 4"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    args = parser.parse_args()
    main(args.prediction)
