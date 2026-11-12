"""Quickselect with a simulated comparator whose only flaw is position bias.

For each ordered pair the comparator, with probability `position_rate`,
says the first record wins regardless of content. Otherwise it reads
the labels: between a relevant and an irrelevant record it picks the
relevant one with probability `content_accuracy`, and between two
records with the same label it picks either at random.

Under this comparator a pair of one relevant and one irrelevant
record lands in the same slot in both orders with probability
`position_rate`, when content accuracy is 1. That is the swap test's
same-slot rate on mixed pairs, so a model's measured rate places it
on the curve.

    uv run python -m experiments.position_bias.simulate
"""

import json
import random
from pathlib import Path

from experiments.position_bias.quickselect import precision, quickselect

# (records, relevant records, k) of each task at sf=0.1
TASK_SHAPES = {"sales": (985, 33, 25), "support": (900, 38, 25),
               "legal": (500, 15, 15)}
POSITION_RATES = tuple(round(0.05 * step, 2) for step in range(19))
CONTENT_ACCURACIES = (1.0, 0.9)
NUM_SEEDS = 40
OUTPUT = Path("results/position_bias/simulation.json")


def simulated_comparator(labels, position_rate, content_accuracy, seed):
    """Return a comparator that answers each ordered pair independently."""
    rng = random.Random(seed)

    def compare(pairs):
        answers = []
        for first, second in pairs:
            if rng.random() < position_rate:
                answers.append(1.0)
            elif labels[first] != labels[second]:
                correct = rng.random() < content_accuracy
                answers.append(1.0 if labels[first] == correct else 0.0)
            else:
                answers.append(float(rng.random() < 0.5))
        return answers

    return compare


def simulate_task(num_records, num_relevant, k):
    """Mean top-k precision per (content accuracy, position rate, rule)."""
    ids = [str(index) for index in range(num_records)]
    labels = {key: int(key) < num_relevant for key in ids}
    rows = []
    for content_accuracy in CONTENT_ACCURACIES:
        for position_rate in POSITION_RATES:
            for rule in ("raw", "both_orders"):
                values = []
                for seed in range(NUM_SEEDS):
                    compare = simulated_comparator(labels, position_rate,
                                                   content_accuracy, seed)
                    top = quickselect(ids, k, compare, rule=rule,
                                      seed=seed).top
                    values.append(precision(top, labels))
                rows.append({"content_accuracy": content_accuracy,
                             "position_rate": position_rate, "rule": rule,
                             "precision": sum(values) / len(values)})
    return rows


def main():
    """Simulate every task shape and save the curves."""
    curves = {name: simulate_task(*shape) for name, shape in TASK_SHAPES.items()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(curves, indent=1))
    for name, rows in curves.items():
        line = [f"{row['position_rate']:.2f}:{row['precision']:.2f}"
                for row in rows
                if row["rule"] == "raw" and row["content_accuracy"] == 1.0]
        print(name, " ".join(line))


if __name__ == "__main__":
    main()
