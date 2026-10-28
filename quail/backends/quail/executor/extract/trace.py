"""Opt-in extraction choices for comparing decoder paths on identical inputs."""

import json
import math
import os

import numpy as np

TRACE_WIDTH = 7


def trace_path():
    """Return the JSONL destination, or None when tracing is disabled."""
    return os.environ.get("QUAIL_EXTRACT_TRACE") or None


def choice_values(scores, allowed, best, stop, top):
    """Return best/runner-up tokens, margin, and allowed/stop/top log probabilities."""
    allowed = np.unique(allowed)
    allowed = allowed[allowed >= 0]
    best = -1 if best is None or not np.isfinite(scores[best]) else int(best)
    others = allowed[allowed != best]
    runner = int(others[np.argmax(scores[others])]) if len(others) else -1
    if runner >= 0 and not np.isfinite(scores[runner]):
        runner = -1
    best_lp = float(scores[best]) if best >= 0 else -float("inf")
    runner_lp = float(scores[runner]) if runner >= 0 else -float("inf")
    return np.array(
        [best, runner, best_lp - runner_lp, best_lp, stop, top, scores[top]],
        dtype=np.float32,
    )


def write_choices(path, documents, question, indices, chosen, values):
    """Append accepted choices, including stopping choices, to a JSONL file."""

    def number(value):
        return float(value) if math.isfinite(value) else None

    records = []
    for document, index, token, row in zip(documents, indices, chosen, values):
        records.append(
            json.dumps(
                {
                    "document": int(document.key[1]),
                    "question": int(question),
                    "index": int(index),
                    "chosen": int(token),
                    "allowed": int(row[0]),
                    "runner_up": int(row[1]),
                    "margin": number(row[2]),
                    "allowed_lp": number(row[3]),
                    "stop_lp": number(row[4]),
                    "top": int(row[5]),
                    "top_lp": number(row[6]),
                },
                separators=(",", ":"),
            )
        )
    if records:
        with open(path, "a", encoding="utf-8") as output:
            output.write("\n".join(records) + "\n")
