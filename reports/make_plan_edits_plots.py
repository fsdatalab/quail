r"""Plan IMDB-3 and FEV-9 on the CPU, edit them, and plot the estimates.

No GPU: the planner prices each node from the corpus token counts, so
the estimates and the recompute column come straight from the plan.
Pull the sf=0.1 corpus off the volume, then run this script on it:

    W=/tmp/quail-plan-edits; mkdir -p "$W/data"
    uv run modal volume get quail-results quailb_data/sf0.1 "$W/data/" --force
    uv run --with matplotlib --with transformers python \
      reports/make_plan_edits_plots.py "$W"

Writes plots/plan_edits.png: per node estimated seconds for each
query's plan and for the same plan with a Barrier inserted on its
pinned edge, with the expected recompute of the barrier variant. It
also prints the table the report quotes. Percentages and differences
are derived here.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import quail
from quail.bench.quailb import queries, register_sets
from quail.physical import AiFilter, AiJoin, Barrier
from quail.planner.plan import EngineConfig

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, ORANGE  # noqa: E402

QUERIES = ("IMDB-3", "FEV-9")


def edited_plan(plan):
    """The plan with a Barrier on its first pinned edge."""
    chain = next(node for node in plan.nodes
                 if isinstance(node, AiFilter) and node.pin_survivors)
    join = next(node for node in plan.nodes
                if isinstance(node, AiJoin) and node.anchor == chain.alias)
    return chain, plan.insert(
        Barrier(node_id=f"barrier:{chain.alias}", next_anchor=chain.alias,
                aliases=(chain.alias,)),
        between=(chain.node_id, join.node_id))


def main(workdir):
    rows = {}
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", gpus=1)) as session:
        register_sets(session, Path(workdir) / "data" / "sf0.1")
        defs = queries(session)
        for query_id in QUERIES:
            query = defs[query_id][1]()
            plan = query.plan()
            chain, edited = edited_plan(plan)
            rows[query_id] = dict(plan=plan, edited=edited, chain=chain.node_id)
            print(f"{query_id}: plan estimate {plan.estimated_seconds:.3f} s; "
                  f"edited {edited.estimated_seconds:.3f} s")
            for label, current in (("planned", plan), ("barrier", edited)):
                for node_id, entry in current.estimates.items():
                    if entry:
                        print(f"  {label} {node_id}: " + ", ".join(
                            f"{key}={value:,.3f}" if isinstance(value, float)
                            else f"{key}={value:,}"
                            for key, value in entry.items()))
                print(f"  {label} pinned: "
                      f"{[n.node_id for n in current.nodes
                           if isinstance(n, AiFilter) and n.pin_survivors]}")

    fig, axes = plt.subplots(1, len(QUERIES), figsize=(13, 4.4))
    for axis, query_id in zip(axes, QUERIES):
        plan, edited = rows[query_id]["plan"], rows[query_id]["edited"]
        node_ids = [node.node_id for node in plan.nodes
                    if "seconds" in plan.estimates.get(node.node_id, {})]
        x = np.arange(len(node_ids))
        planned = [plan.estimates[n]["seconds"] for n in node_ids]
        barrier = [edited.estimates[n]["seconds"] for n in node_ids]
        recompute = [edited.estimates[n].get("release_recompute_seconds", 0.0)
                     for n in node_ids]
        axis.bar(x - 0.2, planned, width=0.38, color=BLUE, label="as planned")
        axis.bar(x + 0.2, barrier, width=0.38, color=GRAY,
                 label="with a Barrier on the pinned edge")
        axis.bar(x + 0.2, recompute, width=0.38, bottom=barrier, color=ORANGE,
                 label="expected recompute behind the Barrier")
        for index, (a, b, c) in enumerate(zip(planned, barrier, recompute)):
            axis.annotate(f"{a:.2f}", (index - 0.2, a), ha="center",
                          va="bottom", xytext=(0, 3),
                          textcoords="offset points", fontsize=8)
            axis.annotate(f"{b + c:.2f}", (index + 0.2, b + c), ha="center",
                          va="bottom", xytext=(0, 3),
                          textcoords="offset points", fontsize=8)
        axis.set_xticks(x)
        axis.set_xticklabels(node_ids, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel("estimated seconds")
        axis.set_title(f"{query_id}: {plan.estimated_seconds:.2f} s as planned, "
                       f"{edited.estimated_seconds:.2f} s with the Barrier")
        axis.set_ylim(0, max(max(planned), max(b + c for b, c in zip(
            barrier, recompute))) * 1.3)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Per-node estimates as planned and after inserting a Barrier "
                 "on the pinned filter-to-join edge (sf=0.1, Qwen3 4B fp8)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "plan_edits.png", dpi=300)
    print(f"wrote {OUT / 'plan_edits.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
