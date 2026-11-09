"""Plan and edit a pair join on the CPU, with no GPU or Modal.

Run from the repository root:

    uv run python demos/plan_walkthrough.py

Three claims and three evidence passages live in memory. The tokenizer
is the small one the tests use, so nothing is downloaded. The script
does not execute inference.
"""

from pathlib import Path

import pyarrow as pa

import quail
from quail import EngineConfig
from quail.explain import explain
from quail.physical import Barrier
from quail.physical.plan import PlanEditError

# a tokenizer.json directory
TOKENIZER = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tokenizer"

PERSON = "Is the claim in DOCUMENT {0} about a person?"
SUPPORT = ("Does the passage in DOCUMENT {1} support the claim in "
           "DOCUMENT {0}?")

SQL = f"""
SELECT c.id, e.id
FROM claims c
JOIN evidence e
  ON c.evidence_wiki_url = e.id
 AND AI.IF(PROMPT('{SUPPORT}', c.claim, e.text))
WHERE AI.IF(PROMPT('{PERSON}', c.claim))
"""


def register_demo_data(session):
    """Register the three claims and evidence passages used by the demo."""
    for name, column, prefix, length in (
        ("claims", "claim", "c", 20),
        ("evidence", "text", "e", 300),
    ):
        table = pa.table({
            "id": [f"{prefix}{index}" for index in range(3)],
            column: [f"{index} " + "word " * length for index in range(3)],
        })
        if name == "claims":
            table = table.append_column(
                "evidence_wiki_url", pa.array(["e0", "e0", "e2"]))
        session.register(
            name, quail.DocumentProvider.from_table(table, id_col="id"))


def main():
    """Print a SQL query's plan, then edit it."""
    config = EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    with quail.Session(config, tokenizer=TOKENIZER) as session:
        register_demo_data(session)

        print("=== SQL ===" + SQL)
        query = session.sql(SQL, dialect="bq")
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
        print("keep_kv before:", plan.graph.node("ai_filter:c").keep_kv,
              "after:", edited.graph.node("ai_filter:c").keep_kv)
        print(f"edited.estimated_seconds = {edited.estimated_seconds:.4f}")
        print(explain(query.logical, query.operators, edited))
        print("remove gives back the same plan:",
              edited.remove("barrier:c") == plan)

        print("\n=== a refused edit ===")
        try:
            plan.remove("ai_filter:c")
        except PlanEditError as error:
            print("PlanEditError:", error)


if __name__ == "__main__":
    main()
