"""Plots for the QUAIL-B speed-of-light table.

Reads the answers make_sol_quailb.py wrote. They live on the
quail-results volume, so pull them into a workdir first:

    W=<workdir>
    modal volume get quail-results /sol/sol_quailb_sf0.1.json $W/
    uv run --with matplotlib python reports/make_sol_quailb_plots.py $W
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, DARK, ORANGE  # noqa

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "plots"
plt.style.use(Path(__file__).parent / "quail.mplstyle")

W = Path(sys.argv[1])
D = json.loads((W / "sol_quailb_sf0.1.json").read_text())
Q = D["queries"]
ORDER = list(Q)
MODELS = ("qwen3-4b-fp8", "qwen3-32b-fp8")
ATTENTION_ORDER = [query_id for query_id in ORDER
                   if all(len(Q[query_id]["models"][model]["join_stages"])
                          <= 1 for model in MODELS)]


def sol(q, model):
    return Q[q]["models"][model]["sol_s"]


def plot_sol_per_query():
    """Both models on one log axis: the suite spans four orders of
    magnitude, so a linear axis would show only BIO-2."""
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = range(len(ORDER))
    w = 0.4
    a = [sol(q, "qwen3-4b-fp8") for q in ORDER]
    b = [sol(q, "qwen3-32b-fp8") for q in ORDER]
    ax.bar([i - w / 2 for i in x], a, w, color=BLUE, label="Qwen3-4B-fp8")
    ax.bar([i + w / 2 for i in x], b, w, color=ORANGE, label="Qwen3-32B-fp8")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va * 1.12, f"{va:.3g}", ha="center", fontsize=6,
                color=BLUE, rotation=90)
        ax.text(i + w / 2, vb * 1.12, f"{vb:.3g}", ha="center", fontsize=6,
                color=ORANGE, rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("speed of light, seconds on one H100! request "
                  "(log scale, 4 orders of magnitude)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=7)
    ax.set_ylim(0.015, 600)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_per_query.png", dpi=150)


# Each context length, the document set it comes from, and where to
# put its label. The three middle clusters are ~20pt apart on a log
# axis, so their labels sit at staggered heights to clear each other.
CONTEXTS = {
    11.4: ("FEVER", 6.0),
    233.1: ("LePaRD", 11.0),
    298.8: ("IMDB", 18.0),
    370.2: ("FEVER", 25.0),
    4146.0: ("BioDEX", 44.0),
}


def plot_attention_share():
    """Attention's share of the compute, against the context each new
    token attends over.

    A token attending over a 4,146-token report scores 4,146 pairs; a
    token attending over an 11-token claim scores 11. So the longer
    the context, the more of the work is attention. Each model uses
    the context selected by its own plan and runtime anchor choice.
    A line connects the two model results for one query.

    Multi join queries are omitted because they can use more than one
    anchor context. Points are not labelled by query because many
    queries share the same context length. The report's table gives
    the per-query numbers.
    """
    def share(q, model):
        m = Q[q]["models"][model]
        return 100 * m["t_attention"] / m["t_compute"]

    x_by_model = {
        model: [Q[q]["models"][model]["held_mean_doc_tokens"]
                for q in ATTENTION_ORDER]
        for model in MODELS}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for index, query_id in enumerate(ATTENTION_ORDER):
        ax.plot([x_by_model[model][index] for model in MODELS],
                [share(query_id, model) for model in MODELS],
                color=DARK, lw=0.7, alpha=0.25)
    for model, colour, label in (("qwen3-4b-fp8", BLUE, "Qwen3-4B-fp8"),
                                 ("qwen3-32b-fp8", ORANGE,
                                  "Qwen3-32B-fp8")):
        ax.scatter(x_by_model[model],
                   [share(q, model) for q in ATTENTION_ORDER],
                   s=34, color=colour, label=label, zorder=3)

    for ctx, (name, ty) in CONTEXTS.items():
        matching = [
            (query_id, model)
            for query_id in ATTENTION_ORDER
            for model in MODELS
            if abs(Q[query_id]["models"][model]["held_mean_doc_tokens"]
                   - ctx) < 0.05]
        top = max(share(query_id, model) for query_id, model in matching)
        n = len({query_id for query_id, _ in matching})
        ax.vlines(ctx, top + 0.8, ty - 1.6, color=DARK, lw=0.6,
                  alpha=0.35)
        ax.text(ctx, ty, f"{name}\n{ctx:,.0f} tokens, "
                         f"{n} quer{'y' if n == 1 else 'ies'}",
                ha="center", va="bottom", fontsize=7.5, color=DARK)

    ax.set_xscale("log")
    ax.set_xlim(7, 9000)
    ax.set_ylim(-2, 50)
    ax.set_xlabel("mean context each new token attends over, tokens "
                  "(log scale)")
    ax.set_ylabel("attention share of compute, percent")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


OUT.mkdir(exist_ok=True)
for old in ("sol_quailb_measured.png", "sol_quailb_kv_reuse.png"):
    (OUT / old).unlink(missing_ok=True)
plot_sol_per_query()
plot_attention_share()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
