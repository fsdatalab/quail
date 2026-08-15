"""The end-to-end check: predicted makespan against measured walls.

The estimator prices the measured filter query from the calibrated
constants, and this module reports the error against the walls the
query actually took. Since the estimator was rewired to measured
constants, a prediction here is: computed tokens at the sustained
per-token cost (1/STEP_TOKEN_S, the batch-sweep rate), later stages'
re-reads of resident context at the cached-read price, the per-step
fixed cost amortized at the boot's step budget, and the measured
per-query residue. The quadratic prefill surcharge is omitted - the
banked reports do not carry the corpus's squared lengths - and the
omission is noted in the output (about half a second low at this
corpus's document lengths).

Each prediction comes in two selectivity variants:

    designed    the pass rates the workload planted (0.9 ... 0.8).
                What a planner would predict before running.
    effective   the pass rates the model's verdicts actually
                produced, read off the report's answered-stage
                count. Prices the query that actually ran.

The rewind arm's computed tokens are the model's own (corpus once
plus thinned question tails); the stock arm's read multiplier is an
input taken from the measurement, because the estimator does not
predict a prefix cache's eviction behavior. For walls from the c0
anchor run, the rewind row adds a container variant: the same token
counts at the rate the probe measured in that container, plus that
arm's measured c0.

Run:
    python -m quail.plan.validate \
        --anchor results/engine/c0_anchor.json \
        --banked results/engine/filter_cells.json \
        --out results/engine/makespan_check.json
"""

import json

from ..configs import DEVICES, MODELS
from .cost import (ENGINE_OVERHEAD_S, STEP_FIXED_S, quest_read_tokens,
                   quest_token_count, read_seconds_per_token,
                   token_seconds)

QUESTION_TOKENS = 46
PREAMBLE_TOKENS = 33


def price(model, device, computed_tokens, read_tokens, step_tokens,
          c0=ENGINE_OVERHEAD_S, s_per_token=None):
    """Seconds for a query computing `computed_tokens` fresh tokens
    and re-reading `read_tokens` resident ones. s_per_token overrides
    the fleet per-token cost (the container variant prices at the
    probed rate instead)."""
    tok_s = s_per_token if s_per_token is not None \
        else token_seconds(model, device)
    return (computed_tokens * tok_s
            + read_tokens * read_seconds_per_token(model, device)
            + STEP_FIXED_S * computed_tokens / max(1, step_tokens)
            + c0)


def _quest_counts(n_docs, n_filters, mean_doc, kept_preamble,
                  selectivity=None, stages_per_doc=None):
    """(computed question tokens, re-read resident tokens), from
    either designed selectivities or a measured stages-per-document
    count. Only the survival-weighted stage total enters either
    formula, so the measured count substitutes exactly."""
    if stages_per_doc is None:
        q = quest_token_count(n_docs, 1, n_filters, QUESTION_TOKENS,
                              PREAMBLE_TOKENS, selectivity)
        r = quest_read_tokens(n_docs, 1, n_filters, mean_doc,
                              PREAMBLE_TOKENS, selectivity,
                              kept_preamble=kept_preamble)
        return q, r
    tail = max(1, QUESTION_TOKENS - PREAMBLE_TOKENS)
    later = max(0.0, stages_per_doc - 1.0)
    ctx = mean_doc + (PREAMBLE_TOKENS if kept_preamble else 0)
    return (n_docs * (QUESTION_TOKENS + later * tail),
            n_docs * later * ctx)


def check(anchor, banked=None, model_name="Qwen3-4B-FP8",
          device_name="H100-SXM-80GB"):
    model = MODELS[model_name]
    device = DEVICES[device_name]
    rows = []
    sources = [("anchor", anchor)]
    if banked is not None:
        sources.append(("banked", banked))
    for src, rep in sources:
        corpus = rep["corpus_tokens"]
        n_docs = rep["n_docs"]
        z = rep["n_filters"]
        sel = rep["selectivity"]
        step_tokens = rep.get("step_tokens", 25_305)
        mean_doc = corpus / n_docs
        probe = rep.get("probe")
        for arm in ("rewind", "stock"):
            cells = [c for c in rep["cells"] if c["arm"] == arm]
            if not cells:
                continue
            walls = [c["wall"] for c in cells]
            wall_mean = sum(walls) / len(walls)
            reads_mean = sum(c["reads"] for c in cells) / len(cells)
            stages_per_doc = (sum(c["answered"] for c in cells)
                              / len(cells) / n_docs)
            kept = arm == "rewind"

            variants = {}
            for name, kwargs in (
                    ("designed", dict(selectivity=sel)),
                    ("effective", dict(stages_per_doc=stages_per_doc))):
                q, r = _quest_counts(n_docs, z, mean_doc, kept, **kwargs)
                if arm == "rewind":
                    computed = corpus + q
                    basis = "model: corpus once plus thinned tails"
                else:
                    computed = reads_mean * corpus
                    basis = ("measured reads (the estimator does not "
                             "predict prefix-cache eviction)")
                pred = price(model, device, computed, r, step_tokens)
                variants[name] = dict(
                    computed_tokens=round(computed),
                    read_tokens=round(r),
                    predicted_s=round(pred, 3),
                    error=round((pred - wall_mean) / wall_mean, 4))
                if name == "effective" and probe and arm == "rewind":
                    c0s = sorted(c["c0_s"] for c in cells if "c0_s" in c)
                    if c0s:
                        pc = price(model, device, computed, r,
                                   step_tokens,
                                   c0=c0s[len(c0s) // 2],
                                   s_per_token=1.0
                                   / probe["rate_tok_s"])
                        variants["container_effective"] = dict(
                            predicted_s=round(pc, 3),
                            error=round((pc - wall_mean) / wall_mean,
                                        4))
            rows.append(dict(
                source=src, arm=arm, walls=walls,
                wall_mean_s=round(wall_mean, 3),
                reads_measured=round(reads_mean, 3),
                stages_per_doc=round(stages_per_doc, 4),
                reads_basis=basis, **variants))
    return dict(model=model.name, device=device.name,
                note="quadratic prefill surcharge omitted: reports "
                     "carry no squared-length sum (~0.5 s low here)",
                rows=rows)


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
                f"{r['wall_mean_s']:7.2f}s  designed "
                f"{r['designed']['predicted_s']:7.2f}s "
                f"({r['designed']['error']:+.1%})  effective "
                f"{r['effective']['predicted_s']:7.2f}s "
                f"({r['effective']['error']:+.1%})")
        if "container_effective" in r:
            ce = r["container_effective"]
            line += (f"  container {ce['predicted_s']:7.2f}s "
                     f"({ce['error']:+.1%})")
        print(line)
    return out


if __name__ == "__main__":
    main()
