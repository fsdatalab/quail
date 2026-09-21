"""Plan and edit a pair join on the CPU, with no GPU or Modal.

Run from the repository root:

    uv run python demos/plan_walkthrough.py

Three claims and three evidence passages live in memory. The tokenizer
is a stand-in that counts bytes. The script does not execute inference.
"""

import pyarrow as pa

import quail
from quail import EngineConfig, col, prompt
from quail.physical import Barrier
from quail.planner.decide import explain
from quail.planner.plan import PlanEditError

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


def same_page(tables):
    """Pair each claim with the evidence row its wiki URL names."""
    return tables["c"].join(
        tables["e"],
        keys=["evidence_wiki_url"],
        right_keys=["id"],
        join_type="inner",
    ).select(["c", "e"])


def main():
    """Print and edit equivalent SQL and builder plans."""
    config = EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    with quail.Session(
        config, tokenizer=lambda text: list(text.encode())
    ) as session:
        register_demo_data(session)

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


if __name__ == "__main__":
    main()
