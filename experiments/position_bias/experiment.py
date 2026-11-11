"""One model's runs on one task: pointwise, swap test, and quickselect.

A scorer takes a prompt template with one `{0}` placeholder and a list
of documents, and returns P(TRUE) for each document. Quail's AI.SCORE
and the Jev API are both scorers. A pair is scored as one document
holding RECORD 1 and RECORD 2, so every model reads a pair the same
way.
"""

import math
import time
from collections.abc import Callable, Sequence

from experiments.position_bias.quickselect import RULES, precision, quickselect
from experiments.position_bias.tasks import TaskData, pair_document, sample_swap_pairs

Scorer = Callable[[str, Sequence[str]], list[float]]

NUM_SWAP_PAIRS = 2000
SEEDS = (0, 1, 2)


class PairCache:
    """Score ordered pairs once each, in batches.

    Attributes:
        probabilities: (first id, second id) -> P(first wins).
        seconds: Time spent in the scorer.
    """

    def __init__(self, data: TaskData, scorer: Scorer):
        self.data = data
        self.scorer = scorer
        self.probabilities: dict[tuple[str, str], float] = {}
        self.seconds = 0.0

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return P(first wins) for each ordered pair."""
        missing = list(dict.fromkeys(
            pair for pair in pairs if pair not in self.probabilities))
        if missing:
            texts = self.data.texts
            documents = [pair_document(texts[first], texts[second])
                         for first, second in missing]
            start = time.perf_counter()
            scores = self.scorer(self.data.task.pair_template, documents)
            self.seconds += time.perf_counter() - start
            self.probabilities.update(zip(missing, scores))
        return [self.probabilities[pair] for pair in pairs]


def top_by_score(scores: dict[str, float], k: int) -> list[str]:
    """The k highest-scoring ids, ties broken by id."""
    return sorted(scores, key=lambda key: (-scores[key], key))[:k]


def swap_summary(rows: list[dict]) -> dict:
    """Summarize swap-test rows into flip rates and the first-slot bias.

    Each row holds `p_forward` = P(first wins) for (a, b) and
    `p_backward` for (b, a). A pair keeps one position when the model
    picks the same slot in both orders. The bias b is the mean of
    p_forward + p_backward - 1: 0 for no pull, positive toward the
    first slot.
    """
    def stats(subset):
        if not subset:
            return {"num_pairs": 0}
        same_slot = [(r["p_forward"] > 0.5) == (r["p_backward"] > 0.5)
                     for r in subset]
        first_both = [r["p_forward"] > 0.5 and r["p_backward"] > 0.5
                      for r in subset]
        sums = [r["p_forward"] + r["p_backward"] - 1 for r in subset]
        mean = sum(sums) / len(sums)
        variance = sum((s - mean) ** 2 for s in sums) / max(1, len(sums) - 1)
        return {
            "num_pairs": len(subset),
            "same_slot_rate": sum(same_slot) / len(subset),
            "first_slot_both_rate": sum(first_both) / len(subset),
            "bias": mean,
            "bias_stderr": math.sqrt(variance / len(subset)),
        }

    mixed = [r for r in rows if r["label_a"] != r["label_b"]]
    tied = [r for r in rows if r["label_a"] == r["label_b"]]
    summary = {"all": stats(rows), "mixed": stats(mixed), "tied": stats(tied)}
    # mixed pairs: how often the relevant record wins, in each order
    correct = []
    for r in mixed:
        a_wins_forward = r["p_forward"] > 0.5
        a_wins_backward = r["p_backward"] <= 0.5
        correct += [a_wins_forward == r["label_a"], a_wins_backward == r["label_a"]]
    summary["mixed"]["accuracy"] = sum(correct) / len(correct) if correct else None
    return summary


def run_task(data: TaskData, scorer: Scorer, *, seeds: Sequence[int] = SEEDS,
             num_swap_pairs: int = NUM_SWAP_PAIRS, log=print) -> dict:
    """Run pointwise top-k, the swap test, and quickselect under each rule.

    Returns:
        A JSON-ready record of every probability and selection.
    """
    task = data.task
    record = {"task": task.name, "k": task.k, "num_records": len(data.ids),
              "num_relevant": sum(data.labels.values())}

    start = time.perf_counter()
    scores = dict(zip(data.ids, scorer(task.pointwise_template,
                                       [data.texts[key] for key in data.ids])))
    top = top_by_score(scores, task.k)
    record["pointwise"] = {"seconds": time.perf_counter() - start,
                           "scores": scores, "top": top,
                           "precision": precision(top, data.labels)}
    log(f"{task.name}: pointwise precision {record['pointwise']['precision']:.3f}")

    cache = PairCache(data, scorer)
    pairs = sample_swap_pairs(data, num_swap_pairs)
    forward = cache(pairs)
    backward = cache([(b, a) for a, b in pairs])
    rows = [{"a": a, "b": b, "label_a": data.labels[a], "label_b": data.labels[b],
             "p_forward": p, "p_backward": q}
            for (a, b), p, q in zip(pairs, forward, backward)]
    summary = swap_summary(rows)
    record["swap"] = {"seconds": cache.seconds, "rows": rows, "summary": summary}
    bias = summary["all"]["bias"]
    log(f"{task.name}: swap same-slot rate on mixed pairs "
        f"{summary['mixed']['same_slot_rate']:.3f}, bias {bias:+.3f}")

    record["quickselect"] = []
    for seed in seeds:
        for rule in RULES:
            before = cache.seconds
            selection = quickselect(data.ids, task.k, cache, rule=rule,
                                    bias=bias, seed=seed)
            entry = {"seed": seed, "rule": rule, "top": selection.top,
                     "precision": precision(selection.top, data.labels),
                     "num_rounds": selection.num_rounds,
                     "num_comparisons": selection.num_comparisons,
                     "truncated": selection.truncated,
                     "new_seconds": cache.seconds - before,
                     "rounds": selection.rounds}
            record["quickselect"].append(entry)
            log(f"{task.name}: quickselect {rule} seed {seed} precision "
                f"{entry['precision']:.3f} after {selection.num_comparisons} "
                f"comparisons")
    record["pair_seconds"] = cache.seconds
    record["num_scored_pairs"] = len(cache.probabilities)
    return record
