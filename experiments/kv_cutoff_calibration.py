"""Pick the redo cutoff that keeps rounded-KV answers within an error target.

Reads kv_plane_margin result files for DiffusionGemma and pools every
item that has the plane_a_rne_keep configuration. An item is answered
from rounded KV when |margin| is at least the cutoff and redone with
bf16 KV otherwise; a redone item always matches bf16. The error rate is
the share of all items whose accepted answer differs from bf16.

For each cutoff, from the largest to the smallest, the script computes
a one-sided Clopper-Pearson upper limit on the error rate. It stops at
the first cutoff whose limit exceeds the target, so the chosen cutoff
keeps its confidence level even though many cutoffs are tried.

Pull the files, then run from the repository root:

    W=$(mktemp -d)
    for f in confirm cal1 cal2 cal3 cal4; do
      uv run modal volume get quail-results \
        /ablations/kv_plane_margin_dgemma26b_$f.json $W/
    done
    uv run python experiments/kv_cutoff_calibration.py $W
"""

import json
import math
import sys
from pathlib import Path

VARIANT = "plane_a_rne_keep"
CUTOFFS = [x / 4 for x in range(48, -1, -1)]


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1)
                    - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(total, 1.0)


def upper_limit(k: int, n: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper limit on a rate with k of n."""
    if k >= n:
        return 1.0
    alpha = 1.0 - confidence
    low, high = k / n, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        if binomial_cdf(k, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def load_items(workdir: Path) -> list[dict]:
    items = []
    for path in sorted(workdir.glob("kv_plane_margin_dgemma26b*.json")):
        records = json.loads(path.read_text())["records"]
        usable = [r for r in records if VARIANT in r["margins"]]
        print(f"{path.name}: {len(usable)} of {len(records)} items usable")
        items.extend(usable)
    return items


def calibrate(items, target, confidence):
    """Return rows for each cutoff tried and the chosen cutoff."""
    n = len(items)
    rows, chosen = [], None
    for cutoff in CUTOFFS:
        redone = sum(abs(r["margins"][VARIANT]) < cutoff for r in items)
        wrong = sum(
            abs(r["margins"][VARIANT]) >= cutoff
            and (r["margins"][VARIANT] > 0) != (r["margins"]["bf16"] > 0)
            for r in items)
        limit = upper_limit(wrong, n, confidence)
        rows.append((cutoff, redone, wrong, limit))
        if limit > target:
            break
        chosen = (cutoff, redone, wrong, limit)
    return rows, chosen


def main(workdir: str, target: float = 0.001, confidence: float = 0.95) -> None:
    items = load_items(Path(workdir))
    n = len(items)
    print(f"items: {n}; target error rate {target}; confidence {confidence}")
    rows, chosen = calibrate(items, target, confidence)
    for cutoff, redone, wrong, limit in rows:
        print(f"cutoff {cutoff:5.2f}: redone {redone:5d} ({100 * redone / n:5.1f}%),"
              f" wrong {wrong:3d}, upper limit {1000 * limit:.2f} per 1,000")
    if chosen is None:
        print("no cutoff meets the target with these items")
        return
    cutoff, redone, wrong, limit = chosen
    print(f"chosen cutoff {cutoff}: redo {100 * redone / n:.1f}% of items; "
          f"error rate under {1000 * limit:.2f} per 1,000 at {confidence:.0%}")
    for table in sorted({r["table"] for r in items}):
        part = [r for r in items if r["table"] == table]
        redo = sum(abs(r["margins"][VARIANT]) < cutoff for r in part)
        print(f"  {table}: {len(part)} items, {100 * redo / len(part):.1f}% redone")


if __name__ == "__main__":
    main(sys.argv[1])
