"""QUAIL-B benchmark anatomy: each query as a vertical plan tree.

Queries are laid out side by side, grouped by family.  The tree nodes
are AI_FILTER (blue), AI_JOIN (orange), and table scan (gray), each
shaded by selectivity.

    uv run --with matplotlib python tools/make_anatomy_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F11,
    F12,
    F13,
    LEP1,
    LEP2,
    LEP3,
    LEP4,
    LEP5,
    LEPJOIN,
    LEPS1,
    REACTION,
    REFUTE,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
)
from quail_b.queries import (
    FILTER_SELECTIVITY_ESTIMATES,
    JOIN_SELECTIVITY_ESTIMATES,
    QUERIES,
    QUERY_FAMILY_WORKLOADS,
)
from quail_b.substrait import _inspect_plan

# ---- colors (self-contained, no external style dependency) ----

BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
SCAN_COLOR = "#BBBBBB"

TEMPLATE_LABELS = {
    F1: "F1", F4: "F4", F5: "F5",
    SERIOUS_ADVERSE_EVENT: "serious",
    AGENT_RECOVERED: "recov.", AGENT_IMPLEMENTED_FIX: "impl.",
    F11: "F11", F12: "F12", F13: "F13",
    LEP1: "LEP1", LEP2: "LEP2", LEP3: "LEP3",
    LEP4: "LEP4", LEP5: "LEP5", LEPS1: "LEPS1",
    DISCUSS_ASPECT: "discuss", ASPECT_SENTIMENT: "sentim.",
    REACTION: "react.",
    SUPPORT: "support", REFUTE: "refute",
    LEPJOIN: "cites",
}


def _shade(base_hex, selectivity):
    """Blend a base color between light (selectivity 0) and dark (1)."""
    r, g, b = mcolors.to_rgb(base_hex)
    t = selectivity
    lr, lg, lb = r + 0.7 * (1 - r), g + 0.7 * (1 - g), b + 0.7 * (1 - b)
    dr, dg, db = r * 0.8, g * 0.8, b * 0.8
    return (lr + t * (dr - lr), lg + t * (dg - lg), lb + t * (db - lb))


def _text_color(bg):
    if isinstance(bg, str):
        bg = mcolors.to_rgb(bg)
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "white" if lum < 0.55 else DARK


# ---- tree nodes ----


class Scan:
    def __init__(self, table, alias):
        self.table = table
        self.alias = alias


class Filter:
    def __init__(self, label, template, child):
        self.label = label
        self.template = template
        self.child = child


class Join:
    def __init__(self, label, template, left, right):
        self.label = label
        self.template = template
        self.left = left
        self.right = right


def _build_tree(spec):
    """Build a plan tree from one QuerySpec."""
    details = _inspect_plan(spec.plan)
    subtrees = {}
    for relation in details.relations:
        node = Scan(relation.table, relation.alias)
        for filter_spec in details.filters:
            if filter_spec.relation == relation.alias:
                node = Filter(
                    TEMPLATE_LABELS[filter_spec.prompt],
                    filter_spec.prompt,
                    node,
                )
        subtrees[relation.alias] = node

    base_alias = details.relations[0].alias
    current = subtrees[base_alias]
    joined = {base_alias}
    for join in details.joins:
        (added,) = set(join.relations) - joined
        current = Join(
            TEMPLATE_LABELS[join.prompt], join.prompt,
            current, subtrees[added])
        joined.add(added)
    return current


# ---- layout engine (root at top, leaves at bottom) ----

BW = 0.85
BH = 0.50
VG = 0.18
HG = 0.12


def _size(node):
    if isinstance(node, Scan):
        return BW, BH
    if isinstance(node, Filter):
        cw, ch = _size(node.child)
        return cw, ch + BH + VG
    if isinstance(node, Join):
        lw, lh = _size(node.left)
        rw, rh = _size(node.right)
        return lw + HG + rw, max(lh, rh) + BH + VG
    return 0, 0


def _place(node, x, top_y):
    if isinstance(node, Scan):
        cx = x + BW / 2
        return [(node, cx, top_y)], x, x + BW

    if isinstance(node, Filter):
        child_items, cl, cr = _place(node.child, x, top_y + BH + VG)
        cx = (cl + cr) / 2
        return [(node, cx, top_y)] + child_items, cl, cr

    if isinstance(node, Join):
        lw, _ = _size(node.left)
        child_top = top_y + BH + VG
        left_items, ll, lr = _place(node.left, x, child_top)
        right_items, rl, rr = _place(node.right, x + lw + HG, child_top)
        cx = (ll + lr + rl + rr) / 4
        return [(node, cx, top_y)] + left_items + right_items, ll, rr

    return [], x, x


def _get_children(node):
    if isinstance(node, Filter):
        return [node.child]
    if isinstance(node, Join):
        return [node.left, node.right]
    return []


def _draw_tree(ax, tree, ox, oy):
    items, _, _ = _place(tree, 0, 0)
    tw, th = _size(tree)
    pos_map = {id(n): (cx, cy) for n, cx, cy in items}

    for node, cx, cy in items:
        sx = ox + cx - BW / 2
        sy = oy - cy - BH

        if isinstance(node, Scan):
            color = SCAN_COLOR
            label = node.alias
            fontsize = 6.5
        elif isinstance(node, Filter):
            sel = FILTER_SELECTIVITY_ESTIMATES.get(node.template)
            color = _shade(BLUE, sel) if sel is not None else BLUE
            label = node.label
            if sel is not None:
                label += f"\n{sel:.0%}" if sel >= 0.01 else "\n<1%"
            fontsize = 6.0
        elif isinstance(node, Join):
            sel = JOIN_SELECTIVITY_ESTIMATES.get(node.template)
            color = _shade(ORANGE, sel) if sel is not None else ORANGE
            label = node.label
            if sel is not None:
                label += f"\n{sel:.0%}" if sel >= 0.01 else "\n<1%"
            fontsize = 6.0
        else:
            continue

        rect = mpatches.FancyBboxPatch(
            (sx, sy), BW, BH,
            boxstyle="round,pad=0.015", facecolor=color,
            edgecolor="white", linewidth=0.6)
        ax.add_patch(rect)
        ax.text(ox + cx, sy + BH / 2, label,
                ha="center", va="center", fontsize=fontsize,
                color=_text_color(color), fontweight="bold",
                fontfamily="monospace")

        for child in _get_children(node):
            ccx, ccy = pos_map[id(child)]
            ax.plot([ox + cx, ox + ccx],
                    [sy, oy - ccy - BH],
                    color="#CCCCCC", linewidth=0.5, zorder=0)

    return tw, th


# ---- build the figure ----

def main():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.facecolor": "white", "axes.grid": False,
        "figure.facecolor": "white", "figure.autolayout": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": DARK,
    })

    families = []
    for spec in QUERIES:
        prefix = spec.id.split("-", 1)[0]
        if not families or families[-1][0] != prefix:
            families.append((prefix, []))
        families[-1][1].append(spec)

    query_gap = 0.3
    domain_y_gap = 1.3

    family_metas = []
    for prefix, specs in families:
        trees = [(_build_tree(s), s.id) for s in specs]
        dh = max(_size(t)[1] for t, _ in trees)
        dw = sum(_size(t)[0] + query_gap for t, _ in trees) - query_gap
        name = QUERY_FAMILY_WORKLOADS.get(prefix, prefix)
        family_metas.append((name, trees, dw, dh))

    max_w = max(m[2] for m in family_metas)
    total_h = (sum(m[3] for m in family_metas)
               + (len(family_metas) - 1) * domain_y_gap)

    n_queries = len(QUERIES)
    n_predicates = len(set(TEMPLATE_LABELS.keys())
                       & (set(FILTER_SELECTIVITY_ESTIMATES.keys())
                          | set(JOIN_SELECTIVITY_ESTIMATES.keys())
                          | {
                              filter_spec.prompt
                              for spec in QUERIES
                              for filter_spec in spec._info.filters
                          }
                          | {j.prompt for s in QUERIES
                             for j in s._info.joins}))
    n_families = len(family_metas)
    n_tables = len({
        relation.table
        for spec in QUERIES
        for relation in spec._info.relations
    })

    scale = 0.65
    fig, ax = plt.subplots(
        figsize=(max_w * scale + 1, total_h * scale + 2))

    y_offset = 0
    for name, trees, dw, dh in family_metas:
        ax.text(-0.15, -y_offset + BH + 0.55, name,
                ha="left", va="bottom", fontsize=10, fontweight="bold",
                color=DARK)

        x_cursor = 0
        for tree, qid in trees:
            tw, _ = _size(tree)
            _draw_tree(ax, tree, x_cursor, -y_offset)
            ax.text(x_cursor + tw / 2, -y_offset + BH + 0.08, qid,
                    ha="center", va="bottom", fontsize=5.5,
                    color="#999999", fontfamily="monospace")
            x_cursor += tw + query_gap

        y_offset += dh + domain_y_gap

    legend_patches = [
        mpatches.Patch(color=SCAN_COLOR, label="table scan"),
        mpatches.Patch(color=BLUE, label="AI_FILTER"),
        mpatches.Patch(color=ORANGE, label="AI_JOIN"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))

    ax.set_title(
        f"QUAIL-B: {n_queries} queries, {n_predicates} predicates, "
        f"{n_families} families, {n_tables} tables (sf = 0.1)",
        fontsize=12, fontweight="bold", color=DARK, pad=16)

    ax.set_xlim(-0.5, max_w + 1.5)
    ax.set_ylim(-y_offset + domain_y_gap - 0.5, BH + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    base = Path(__file__).resolve().parent.parent / "figures"
    base.mkdir(parents=True, exist_ok=True)
    pdf = base / "quailb_anatomy.pdf"
    fig.savefig(pdf)
    png = base / "quailb_anatomy.png"
    fig.savefig(png, dpi=300)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
