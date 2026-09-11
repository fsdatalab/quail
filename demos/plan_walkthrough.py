"""Plan, edit, and run a pair join on the CPU, with no GPU or Modal.

Run from the repository root:

    .venv/bin/python demos/plan_walkthrough.py

Three claims and three evidence passages live in memory. The tokenizer
is a stand-in that counts bytes, and the model answers come from the
test fakes, so every second printed here is tiny. The plan text, node
ids, estimates, edits, and reports have the same shape as on a GPU.
"""

import sys
from pathlib import Path

import pyarrow as pa

# The fake model and arena live with the tests, and the package is not
# installed in the virtual environment.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from test_fixed_join_plan import FixedFeverAnswers  # noqa: E402
from test_quail_backend import graph_state  # noqa: E402

import quail  # noqa: E402
from quail import DocumentProvider, EngineConfig, col, prompt  # noqa: E402
from quail.backends.quail.graph import execute_single_graph  # noqa: E402
from quail.execution import PhysicalResponse  # noqa: E402
from quail.physical import Barrier, Scan, decode_graph  # noqa: E402
from quail.planner.decide import explain  # noqa: E402
from quail.planner.plan import PlanEditError  # noqa: E402
from quail.runtime.execute import execute_query  # noqa: E402

PERSON = "Is the claim in DOCUMENT {0} about a person?"
SUPPORT = ("Does the passage in DOCUMENT {1} support the claim in "
           "DOCUMENT {0}?")

SQL = f"""
SELECT c.id, e.id
FROM claims c
JOIN evidence e
  ON c.evidence_wiki_url = e.id
 AND AI_FILTER(PROMPT('{SUPPORT}', c.claim, e.text))
WHERE AI_FILTER(PROMPT('{PERSON}', c.claim))
"""


def tables():
    """Three claims, two of them on page e0, and three evidence passages."""
    claims = pa.table({
        "id": ["c0", "c1", "c2"],
        "claim": [f"{i} " + "word " * 20 for i in range(3)],
        "evidence_wiki_url": ["e0", "e0", "e2"],
    })
    evidence = pa.table({
        "id": ["e0", "e1", "e2"],
        "text": [f"{i} " + "word " * 300 for i in range(3)],
    })
    return claims, evidence


def fake_executor(session, seen):
    """A physical executor that answers from the FEVER test fake."""

    def execute(request):
        graph = decode_graph(request.plan["graph"], session.registry.codecs)
        seen.append([node.node_id for node in graph.nodes])
        docs = {node.alias: request.inputs[node.input_id].documents
                for node in graph.nodes if isinstance(node, Scan)}
        state = graph_state(None, docs)
        state["model_execution"] = FixedFeverAnswers(state, 10, False)
        state["pairs"] = request.pair_tables()
        state["columns"] = request.column_tables()
        state["functions"] = session.registry.functions
        report = execute_single_graph(state, request.plan["settings"], graph)
        return PhysicalResponse(report.pop("_outputs"), report)

    return execute


def same_page(tables):
    """Pair each claim with the evidence row its wiki url names."""
    claims, evidence = tables["c"], tables["e"]
    return claims.join(evidence, keys=["evidence_wiki_url"],
                       right_keys=["id"], join_type="inner").select(["c", "e"])


def main():
    """Print the plans, edit one, and run all of them under the fakes."""
    claims, evidence = tables()
    with quail.Session(EngineConfig(),
                       tokenizer=lambda text: list(text.encode())) as session:
        session.register("claims",
                         DocumentProvider.from_table(claims, id_col="id"))
        session.register("evidence",
                         DocumentProvider.from_table(evidence, id_col="id"))

        print("=== SQL ===" + SQL)
        query = session.sql(SQL)
        print("=== query.explain() ===")
        print(query.explain())
        plan = query.plan()
        print(f"\nplan.estimated_seconds = {plan.estimated_seconds:.4f}")
        print("node ids:", [node.node_id for node in plan.nodes])
        for node_id, entry in plan.estimates.items():
            if entry:
                print(f"  {node_id}: {entry}")

        print("\n=== insert a Barrier between ai_filter:c and ai_join:c ===")
        edited = plan.insert(
            Barrier(node_id="barrier:c", next_anchor="c", aliases=("c",)),
            between=("ai_filter:c", "ai_join:c"))
        print("pin_survivors before:",
              plan.graph.node("ai_filter:c").pin_survivors,
              "after:", edited.graph.node("ai_filter:c").pin_survivors)
        print(f"edited.estimated_seconds = {edited.estimated_seconds:.4f}")
        print(explain(query.logical, edited))
        print("remove gives back the same plan:",
              edited.remove("barrier:c") == plan)

        print("\n=== a refused edit ===")
        try:
            plan.remove("ai_filter:c")
        except PlanEditError as error:
            print("PlanEditError:", error)

        print("\n=== run the planner's plan and the edited plan ===")
        seen = []
        execute = fake_executor(session, seen)
        rows = execute_query(query, physical_executor=execute).collect()
        print("planner's plan rows:", rows.to_pylist())
        edited_rows = execute_query(session.sql(SQL), physical_executor=execute,
                                    plan=edited).collect()
        print("edited plan rows:   ", edited_rows.to_pylist())
        print("nodes executed:", seen[0])
        print("nodes executed:", seen[1])

        print("\n=== the builder form: a Python function pairs the rows ===")
        paired = (session.docs("claims").alias("c")
                  .ai_filter(prompt(PERSON, col("c.claim")))
                  .join(session.docs("evidence").alias("e"))
                  .apply(same_page, columns=[col("c.evidence_wiki_url"),
                                             col("e.id")])
                  .ai_filter(prompt(SUPPORT, col("c.claim"), col("e.text")))
                  .select("c.id", "e.id"))
        print(paired.explain())
        print("node ids:", [node.node_id for node in paired.plan().nodes])
        apply_rows = execute_query(paired, physical_executor=execute).collect()
        print("apply rows:", apply_rows.to_pylist())
        print("nodes executed:", seen[2])


if __name__ == "__main__":
    main()
