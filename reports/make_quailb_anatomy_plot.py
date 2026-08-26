"""QUAIL-B benchmark anatomy: each query as a vertical plan tree,
queries laid out side by side grouped by domain.

    uv run --with matplotlib python reports/make_quailb_anatomy_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

plt.style.use(Path(__file__).parent / "quail.mplstyle")

from plot_colors import BLUE, ORANGE, DARK, TEAL, GREEN, RED

FILTER_BASE = BLUE
JOIN_BASE = ORANGE
SCAN_COLOR = "#BBBBBB"


def _shade(base_hex, selectivity):
    r, g, b = mcolors.to_rgb(base_hex)
    t = selectivity
    lr = r + 0.7 * (1 - r)
    lg = g + 0.7 * (1 - g)
    lb = b + 0.7 * (1 - b)
    dr, dg, db = r * 0.8, g * 0.8, b * 0.8
    return (lr + t * (dr - lr), lg + t * (dg - lg), lb + t * (db - lb))


def _text_color(bg):
    if isinstance(bg, str):
        bg = mcolors.to_rgb(bg)
    return "white" if 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2] < 0.55 else DARK


SELECTIVITIES = {
    "F1": 0.775, "F4": 0.231, "F5": 0.568,
    "discuss": 0.019, "sentim.": 0.068,
    "F7": 0.595, "F8": 0.680, "F9": 0.630,
    "react.": 0.001, "severe": 0.091,
    "F11": 0.590, "F12": 0.180, "F13": 0.649,
    "support": 0.006, "refute": 0.035,
    "LEP1": 0.020, "LEP2": 0.515, "LEP3": 0.050,
    "LEP4": 0.065, "LEP5": 0.030,
    "LEPS1": 1.000, "cites": 0.241,
}


class Scan:
    def __init__(self, table, alias=None):
        self.table = table
        self.alias = alias or table

class Filter:
    def __init__(self, pred, child):
        self.pred = pred
        self.child = child

class Join:
    def __init__(self, pred, left, right):
        self.pred = pred
        self.left = left
        self.right = right


def _fchain(preds, table, alias=None):
    node = Scan(table, alias)
    for p in preds:
        node = Filter(p, node)
    return node


# ---- query definitions ----

QUERIES = []

def Q(qid, domain, tree):
    QUERIES.append((qid, domain, tree))

# IMDB
Q("IMDB-1", "IMDB", _fchain(["F1"], "reviews", "r"))
Q("IMDB-6", "IMDB", _fchain(["F1", "F4"], "reviews", "r"))
Q("IMDB-7", "IMDB", _fchain(["F1", "F4", "F5"], "reviews", "r"))
Q("IMDB-2", "IMDB", Join("discuss",
    Scan("reviews", "r"), Scan("aspects", "a")))
Q("IMDB-3", "IMDB", Join("discuss",
    _fchain(["F1"], "reviews", "r"), Scan("aspects", "a")))
Q("IMDB-4", "IMDB", Join("discuss",
    _fchain(["F1", "F4"], "reviews", "r"), Scan("aspects", "a")))
Q("IMDB-5", "IMDB", Join("discuss",
    _fchain(["F1", "F4", "F5"], "reviews", "r"), Scan("aspects", "a")))
Q("IMDB-8", "IMDB", Join("sentim.",
    Join("discuss", Scan("reviews", "r"), Scan("aspects", "a1")),
    Scan("aspects", "a2")))
Q("IMDB-9", "IMDB", Join("sentim.",
    Join("discuss",
         Join("discuss", Scan("reviews", "r1"), Scan("aspects", "a1")),
         Scan("reviews", "r2")),
    Scan("aspects", "a2")))
Q("IMDB-10", "IMDB", Join("sentim.",
    Join("discuss",
         Join("discuss",
              Filter("F1", Scan("reviews", "r1")),
              Scan("aspects", "a1")),
         Scan("reviews", "r2")),
    Scan("aspects", "a2")))

# BioDEX
Q("BIO-1", "BioDEX", _fchain(["F7"], "reports", "r"))
Q("BIO-2", "BioDEX", Join("react.",
    Scan("reports", "r"), Scan("terms", "m")))
Q("BIO-3", "BioDEX", Join("react.",
    _fchain(["F7"], "reports", "r"), Scan("terms", "m")))
Q("BIO-4", "BioDEX", Join("react.",
    _fchain(["F7", "F8"], "reports", "r"), Scan("terms", "m")))
Q("BIO-5", "BioDEX", Join("react.",
    _fchain(["F7", "F8", "F9"], "reports", "r"), Scan("terms", "m")))
Q("BIO-6", "BioDEX", Join("severe",
    Join("react.", Scan("reports", "r"), Scan("terms", "m1")),
    Scan("terms", "m2")))
Q("BIO-7", "BioDEX", Join("react.",
    Join("severe",
         Join("react.", Scan("reports", "r1"), Scan("terms", "m1")),
         Scan("reports", "r2")),
    Scan("terms", "m2")))
Q("BIO-8", "BioDEX", Join("react.",
    Join("severe",
         Join("react.",
              Filter("F7", Scan("reports", "r1")),
              Scan("terms", "m1")),
         Scan("reports", "r2")),
    Scan("terms", "m2")))

# FEVER
Q("FEV-1", "FEVER", _fchain(["F11"], "claims", "c"))
Q("FEV-2", "FEVER", Join("support",
    Scan("claims", "c"), Scan("evidence", "e")))
Q("FEV-3", "FEVER", Join("support",
    _fchain(["F11"], "claims", "c"), Scan("evidence", "e")))
Q("FEV-4", "FEVER", Join("support",
    _fchain(["F11", "F12"], "claims", "c"), Scan("evidence", "e")))
Q("FEV-5", "FEVER", Join("support",
    _fchain(["F11"], "claims", "c"),
    _fchain(["F13"], "evidence", "e")))
Q("FEV-6", "FEVER", Join("support",
    _fchain(["F11", "F12"], "claims", "c"),
    _fchain(["F13"], "evidence", "e")))
Q("FEV-7", "FEVER", Join("refute",
    Join("support", Scan("claims", "c"), Scan("evidence", "e1")),
    Scan("evidence", "e2")))
Q("FEV-8", "FEVER", Join("support",
    Join("refute",
         Join("support", Scan("claims", "c1"), Scan("evidence", "e1")),
         Scan("claims", "c2")),
    Scan("evidence", "e2")))
Q("FEV-9", "FEVER", Join("support",
    Join("refute",
         Join("support",
              Filter("F11", Scan("claims", "c1")),
              Scan("evidence", "e1")),
         Scan("claims", "c2")),
    Scan("evidence", "e2")))

# LePaRD
Q("LEP-1", "LePaRD", _fchain(["LEP1"], "citations", "d"))
Q("LEP-2", "LePaRD", Join("cites",
    Scan("citations", "d"), Scan("citations", "s")))
Q("LEP-3", "LePaRD", Join("cites",
    _fchain(["LEP1"], "citations", "d"), Scan("citations", "s")))
Q("LEP-4", "LePaRD", Join("cites",
    _fchain(["LEP1", "LEP2"], "citations", "d"), Scan("citations", "s")))
Q("LEP-5", "LePaRD", Join("cites",
    _fchain(["LEP1", "LEP2", "LEP3"], "citations", "d"),
    Scan("citations", "s")))
Q("LEP-6", "LePaRD", Join("cites",
    _fchain(["LEP1", "LEP2", "LEP3", "LEP4", "LEP5"], "citations", "d"),
    Scan("citations", "s")))
Q("LEP-8", "LePaRD",
    _fchain(["LEP1", "LEP2", "LEP3", "LEP4", "LEP5"], "citations", "d"))
Q("LEP-7", "LePaRD", Join("cites",
    _fchain(["LEP1", "LEP2"], "citations", "d"),
    _fchain(["LEPS1"], "citations", "s")))


# ---- layout engine (top-down: root at top, leaves at bottom) ----

BW = 0.85
BH = 0.50
VG = 0.18
HG = 0.12


def _size(node):
    """Return (width, height) of the subtree."""
    if isinstance(node, Scan):
        return BW, BH
    if isinstance(node, Filter):
        cw, ch = _size(node.child)
        return cw, ch + BH + VG
    if isinstance(node, Join):
        lw, lh = _size(node.left)
        rw, rh = _size(node.right)
        w = lw + HG + rw
        h = max(lh, rh) + BH + VG
        return w, h
    return 0, 0


def _place(node, x, top_y):
    """Place nodes top-down. Root at (x_center, top_y), children below.
    Returns list of (node, cx, cy) and (left_edge_x, right_edge_x)."""
    if isinstance(node, Scan):
        cx = x + BW / 2
        return [(node, cx, top_y)], x, x + BW

    if isinstance(node, Filter):
        cw, ch = _size(node.child)
        child_items, cl, cr = _place(node.child, x, top_y + BH + VG)
        cx = (cl + cr) / 2
        items = [(node, cx, top_y)] + child_items
        return items, cl, cr

    if isinstance(node, Join):
        lw, lh = _size(node.left)
        rw, rh = _size(node.right)
        child_top = top_y + BH + VG
        left_items, ll, lr = _place(node.left, x, child_top)
        right_items, rl, rr = _place(node.right, x + lw + HG, child_top)
        cx = (ll + lr + rl + rr) / 4
        items = [(node, cx, top_y)] + left_items + right_items
        return items, ll, rr

    return [], x, x


def _get_children(node):
    if isinstance(node, Filter):
        return [node.child]
    if isinstance(node, Join):
        return [node.left, node.right]
    return []


def _draw_tree(ax, tree, ox, oy):
    """Draw one query plan tree. Root at top, leaves at bottom.
    oy is the TOP of the tree (y increases downward in data coords,
    but we'll flip by using negative y so matplotlib shows root on top)."""
    items, lx, rx = _place(tree, 0, 0)
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
            sel = SELECTIVITIES.get(node.pred)
            color = _shade(FILTER_BASE, sel) if sel is not None else FILTER_BASE
            label = node.pred
            if sel is not None:
                label += f"\n{sel:.0%}" if sel >= 0.01 else "\n<1%"
            fontsize = 6.0
        elif isinstance(node, Join):
            sel = SELECTIVITIES.get(node.pred)
            color = _shade(JOIN_BASE, sel) if sel is not None else JOIN_BASE
            label = node.pred
            if sel is not None:
                label += f"\n{sel:.0%}" if sel >= 0.01 else "\n<1%"
            fontsize = 6.0

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


# ---- main figure: 2x2 grid, one domain per cell ----

domains = []
for qid, domain, tree in QUERIES:
    if not domains or domains[-1][0] != domain:
        domains.append((domain, []))
    domains[-1][1].append((qid, tree))

QUERY_GAP = 0.3
DOMAIN_Y_GAP = 1.3

domain_metas = []
for domain, qs in domains:
    dh = max(_size(t)[1] for _, t in qs)
    dw = sum(_size(t)[0] + QUERY_GAP for _, t in qs) - QUERY_GAP
    domain_metas.append((domain, qs, dw, dh))

max_w = max(m[2] for m in domain_metas)
total_h = sum(m[3] for m in domain_metas) + (len(domain_metas) - 1) * DOMAIN_Y_GAP

scale = 0.65
fig, ax = plt.subplots(figsize=(max_w * scale + 1, total_h * scale + 2))

y_offset = 0
for domain, qs, dw, dh in domain_metas:
    ax.text(-0.15, -y_offset + BH + 0.55, domain,
            ha="left", va="bottom", fontsize=10, fontweight="bold",
            color=DARK)

    x_cursor = 0
    for qi, (qid, tree) in enumerate(qs):
        tw, th = _size(tree)
        _draw_tree(ax, tree, x_cursor, -y_offset)

        ax.text(x_cursor + tw / 2, -y_offset + BH + 0.08, qid,
                ha="center", va="bottom", fontsize=5.5,
                color="#999999", fontfamily="monospace")

        x_cursor += tw + QUERY_GAP

    y_offset += dh + DOMAIN_Y_GAP

legend_patches = [
    mpatches.Patch(color=SCAN_COLOR, label="table scan"),
    mpatches.Patch(color=FILTER_BASE, label="AI_FILTER"),
    mpatches.Patch(color=JOIN_BASE, label="AI_JOIN"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
          frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))

ax.set_title(
    "QUAIL-B: 35 queries, 22 predicates, 4 domains, 7 tables (sf = 0.1)",
    fontsize=12, fontweight="bold", color=DARK, pad=16)

ax.set_xlim(-0.5, max_w + 1.5)
ax.set_ylim(-y_offset + DOMAIN_Y_GAP - 0.5, BH + 1.2)
ax.set_aspect("equal")
ax.axis("off")

out = Path(__file__).parent / "plots" / "quailb_anatomy.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
print(f"wrote {out}")
