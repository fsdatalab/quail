"""The end-to-end check: predicted makespan against measured walls.

predict_makespan prices a query from the calibrated constants. This
module prices the measured filter query the same way and reports the
error against the walls the query actually took.

Two predictions per row:

    fleet      shipped constants only: the PHI-derived rate and the
               shipped c0. What the planner predicts with no
               knowledge of the container.
    container  the same token count at the rate the c0 anchor's probe
               measured in the container that produced the walls,
               plus that arm's measured c0 median. Separates the
               model's structural error from host speed.

The rewind arm's token count is the model's own: the corpus once,
plus question tails thinned by survival. The stock arm's read
multiplier is an input taken from the measurement - the estimator
does not predict a prefix cache's eviction behavior - so the stock
row checks only the fleet constants, and no container prediction is
made for it: its measured c0 came from the same cells by the same
subtraction, and the comparison would be circular.

Run:
    python -m quail.plan.validate \
        --anchor results/engine/c0_anchor.json \
        --banked results/engine/filter_cells.json \
        --out results/engine/makespan_check.json
"""

import json

from ..configs import DEVICES, MODELS
from .cost import ENGINE_OVERHEAD_S, read_rate, t_quest


def question_tokens(model, device, n_docs, n_filters, selectivity):
    """The question tokens the planned arm prefills, from the same
    arithmetic t_quest prices: seconds at R, converted back to
    tokens."""
    return (t_quest(model, device, n_docs, 1, n_filters,
                    selectivity=selectivity)
            * read_rate(model, device))


def check(anchor, banked=None, model_name="Qwen3-4B-FP8",
          device_name="H100-SXM-80GB"):
    model = MODELS[model_name]
    device = DEVICES[device_name]
    rate_fleet = read_rate(model, device)
    rows = []
    sources = [("anchor", anchor)]
    if banked is not None:
        sources.append(("banked", banked))
    for src, rep in sources:
        corpus = rep["corpus_tokens"]
        quest = question_tokens(model, device, rep["n_docs"],
                                rep["n_filters"], rep["selectivity"])
        probe = rep.get("probe")
        for arm in ("rewind", "stock"):
            cells = [c for c in rep["cells"] if c["arm"] == arm]
            if not cells:
                continue
            walls = [c["wall"] for c in cells]
            wall_mean = sum(walls) / len(walls)
            reads_mean = sum(c["reads"] for c in cells) / len(cells)
            if arm == "rewind":
                tokens = corpus + quest
                basis = "model: corpus once plus thinned question tails"
            else:
                tokens = reads_mean * corpus
                basis = "measured reads (the estimator does not " \
                        "predict prefix-cache eviction)"
            pred_fleet = tokens / rate_fleet + ENGINE_OVERHEAD_S
            row = dict(
                source=src, arm=arm, walls=walls,
                wall_mean_s=round(wall_mean, 3),
                reads_measured=round(reads_mean, 3),
                tokens_predicted=round(tokens),
                reads_basis=basis,
                fleet=dict(rate_tok_s=round(rate_fleet, 1),
                           c0_s=ENGINE_OVERHEAD_S,
                           predicted_s=round(pred_fleet, 3),
                           error=round((pred_fleet - wall_mean)
                                       / wall_mean, 4)))
            if probe and arm == "rewind":
                c0s = sorted(c["c0_s"] for c in cells if "c0_s" in c)
                if c0s:
                    c0_med = c0s[len(c0s) // 2]
                    pred = tokens / probe["rate_tok_s"] + c0_med
                    row["container"] = dict(
                        rate_tok_s=probe["rate_tok_s"], c0_s=c0_med,
                        predicted_s=round(pred, 3),
                        error=round((pred - wall_mean) / wall_mean, 4))
            rows.append(row)
    return dict(model=model.name, device=device.name, rows=rows)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--banked")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.anchor) as f:
        anchor = json.load(f)
    banked = None
    if args.banked:
        with open(args.banked) as f:
            banked = json.load(f)
    out = check(anchor, banked)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")
    for r in out["rows"]:
        line = (f"{r['source']:>7} {r['arm']:<7} wall "
                f"{r['wall_mean_s']:7.2f}s  fleet pred "
                f"{r['fleet']['predicted_s']:7.2f}s "
                f"({r['fleet']['error']:+.1%})")
        if "container" in r:
            line += (f"  container pred "
                     f"{r['container']['predicted_s']:7.2f}s "
                     f"({r['container']['error']:+.1%})")
        print(line)
    return out


if __name__ == "__main__":
    main()
