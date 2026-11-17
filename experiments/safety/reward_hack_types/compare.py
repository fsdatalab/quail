"""Compare the six codebooks without a reference.

For each run: the intents and targets, how many steps the agents did not
read fall under "None of these" or get a low-confidence intent, and the
intent and target counts. Across runs: intent and target NMI
(normalized mutual information, 0 for unrelated codes, 1 for the same
codes) for every pair, within each method and between methods.

    uv run --no-sync python -m experiments.safety.reward_hack_types.compare WORK CODES

CODES holds <run>/codes.parquet for each run, copied from
/results/reward_hack_types/<name>/ on the quail-results volume:

    modal volume get quail-results reward_hack_types/<name> CODES
"""

import argparse
import collections
import itertools
import json
import math
import os

import pyarrow.parquet as pq

RUNS = ("sae_a", "sae_b", "text_a", "text_b", "random_a", "random_b")


def nmi(left: list, right: list) -> float:
    """Return the normalized mutual information of two labelings of the same items."""
    total = len(left)
    count_l, count_r = collections.Counter(left), collections.Counter(right)
    joint = collections.Counter(zip(left, right))
    mutual = sum(
        v / total * math.log(v * total / (count_l[a] * count_r[b]))
        for (a, b), v in joint.items()
    )

    def entropy(count):
        return -sum(v / total * math.log(v / total) for v in count.values())

    spread = entropy(count_l) + entropy(count_r)
    return 2 * mutual / spread if spread else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="folder with each run's codebook and meta.json")
    parser.add_argument("codes", help="folder with <run>/codes.parquet")
    args = parser.parse_args()
    codes = {}
    for run in RUNS:
        with open(os.path.join(args.work, run, "book", "codebook.json")) as source:
            book = json.load(source)
        with open(os.path.join(args.work, run, "meta.json")) as source:
            seen = set(json.load(source)["seen"])
        rows = {
            r["s.id"]: r
            for r in pq.read_table(
                os.path.join(args.codes, run, "codes.parquet")
            ).to_pylist()
        }
        codes[run] = rows
        unseen = [r for i, r in rows.items() if i not in seen]
        low = sum(max(dict(r["intent_probabilities"]).values()) < 0.6 for r in unseen)
        print(
            f"\n== {run}: {len(book['intents'])} intents, "
            f"{len(book['targets'])} targets; "
            f"read {len(seen)}, unread {len(unseen)} =="
        )
        for field in ("intent", "target"):
            none = sum(r[field].lower().startswith("none") for r in unseen)
            print(f"  unread under 'None of these' ({field}): {none}")
        print(f"  unread with a low-confidence intent (under 0.6): {low}")
        for field in ("intent", "target"):
            print(
                f"  {field}s:",
                dict(
                    collections.Counter(r[field] for r in rows.values()).most_common()
                ),
            )
    ids = sorted(codes[RUNS[0]])
    for field in ("intent", "target"):
        pairs = {
            (a, b): nmi(
                [codes[a][i][field] for i in ids], [codes[b][i][field] for i in ids]
            )
            for a, b in itertools.combinations(RUNS, 2)
        }
        print(f"\n{field} NMI")
        for (a, b), value in pairs.items():
            print(f"  {a} vs {b}: {value:.2f}")
        method = lambda run: run.rsplit("_", 1)[0]  # noqa: E731
        for name in ("sae", "text", "random"):
            print(f"  within {name}: {pairs[(name + '_a', name + '_b')]:.2f}")
        between = [v for (a, b), v in pairs.items() if method(a) != method(b)]
        print(
            f"  between methods: mean {sum(between) / len(between):.2f}, "
            f"range {min(between):.2f} to {max(between):.2f}"
        )


if __name__ == "__main__":
    main()
