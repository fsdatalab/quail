"""Smoke test for extensions crossing to the Modal worker.

Registers the observer in experiments/cells/row_trace.py, runs one
filter query through the default ModalComputeProvider, and checks that
the observer's report, the executed plan, and the per node metrics
come back. Run from the repository root as a module:

    uv run python -m experiments.cells.extension_smoke 2>&1 \\
        | tee results/extension_smoke.log

Prints the query's explain_analyze and the cost ledger totals.
"""

import json
import tempfile

import quail
from quail.planner.plan import EngineConfig
from quail_ext_examples import cost_ledger

from experiments.cells.row_trace import RowTrace
from experiments.cells.session_smoke import FILTER_Q, make_filter_parquet


def main():
    tmp = tempfile.mkdtemp()
    flags = make_filter_parquet(f"{tmp}/docs.parquet", n_docs=40)
    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        RowTrace)
    session = quail.Session(
        EngineConfig(gpus=1), registry=registry,
        compute_provider=quail.ModalComputeProvider(
            local_python_sources=("experiments",)),
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

    planted = sorted(f"d{i}" for i in range(len(flags)) if flags[i][0])
    got = sorted(row[0] for row in result.to_rows())
    print("rows:", len(got), "planted:", len(planted),
          "agree:", len(set(got) & set(planted)), flush=True)
    print("explain_analyze:\n" + result.explain_analyze(), flush=True)
    print("observer:", json.dumps(result.observer(RowTrace)), flush=True)
    print("ledger totals:", json.dumps(cost_ledger.charge(result)["totals"]),
          flush=True)
    print("node_metrics:", json.dumps({
        node_id: metrics.wall_s
        for node_id, metrics in result.node_metrics.items()}), flush=True)
    session.close()


if __name__ == "__main__":
    main()
