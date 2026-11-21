"""Guaranteed escalation thresholds and SoL cost for a distilled IMDB-1 filter.

Simulates the distilled filter's calibration step offline, from the
student scores the learning-curve cell saved for 5,000 held-out reviews.
Each split draws `num_calibration` of them as the teacher-labeled
calibration set; the rest stand for the rows the plan answers. On the
calibration set the simulator picks two thresholds: rows scored below
t_lo are answered FALSE, above t_hi TRUE, and the rows between go to
the teacher. The thresholds hold the precision and recall of the
answered rows against the teacher's TRUE set at 1 - epsilon each, with
probability at least 1 - delta, under three Clopper-Pearson bounds at
delta / 3 each:

    recall: the fraction of teacher-TRUE rows scored below t_lo;
    precision: the fraction of teacher-FALSE rows scored above t_hi,
        against epsilon / (1 - epsilon) * (1 - epsilon) * pi / (1 - pi)
        = epsilon * pi / (1 - pi), where pi, the TRUE fraction, is at
        its lower bound; the (1 - epsilon) is the recall bound on the
        TRUE rows answered.

The bounds treat the remaining rows as draws from the calibration
rows' distribution; coverage, the fraction of splits whose remaining
rows meet both targets, checks that on the data.

Cost is SoL prefill seconds from the model and device specs, per row
at each row's own prompt length: the teacher on the training and
calibration labels and on the escalated rows, the student on every
other row and on the calibration rows, and LoRA training at a forward
and a backward pass of the student per training pass. The student uses
the decision-2.0-kai-0.6b-bf16 spec, the Qwen3 0.6B backbone.

Inputs, pulled from the results volume into one directory:

    W=/tmp/distill_guarantee
    R=distillation/learning_curves/20261120T021520Z-imdb1-sf05
    T=distillation/answer_trace/20261119T221615Z-db4b2ebe
    modal volume get quail-results $R/records.json $W/
    modal volume get quail-results $R/probes.json $W/
    modal volume get quail-results $T/answer_trace/imdb/IMDB-1 $W/trace
    modal volume get quail-results \
        $T/quail/imdb/IMDB-1/prompt_pieces.json $W/

    uv run --with scipy python experiments/distill_guarantee.py $W
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cells.distill_learning_curves import (
    ACCUMULATE,
    EPOCHS,
    MAX_STEPS,
    MICRO_DOCUMENTS,
    MIN_STEPS,
    TRAINED,
)

EPSILON = 0.05
DELTA = 0.1
NUM_CALIBRATION = (200, 500, 1000)
NUM_SPLITS = 200
SETUPS = ("margin", "bits", "frozen_answer_probe", "zero_shot")
TEACHER = "qwen3.8-27b-fp8"
STUDENT = "decision-2.0-kai-0.6b-bf16"
DEVICE = "h100-sxm"
TABLE_ROWS = (5_000, 24_893, 100_000, 1_000_000)


def _upper(count, total, level):
    """Return the Clopper-Pearson upper bound of a rate at one-sided `level`."""
    from scipy.stats import beta
    count = np.asarray(count, dtype=float)
    bound = beta.ppf(1 - level, count + 1, np.maximum(total - count, 1e-12))
    return np.where(count >= total, 1.0, bound)


def _lower(count, total, level):
    """Return the Clopper-Pearson lower bound of a rate at one-sided `level`."""
    from scipy.stats import beta
    if count == 0:
        return 0.0
    return float(beta.ppf(level, count, total - count + 1))


def guaranteed_thresholds(scores, positive):
    """Choose (t_lo, t_hi) on calibration rows under the three bounds.

    Args:
        scores: The student's scores of the calibration rows.
        positive: Whether the teacher answered TRUE for each row.

    Returns:
        (t_lo, t_hi), with t_lo <= t_hi.
    """
    level = DELTA / 3
    true_scores = np.sort(scores[positive])
    false_scores = np.sort(scores[~positive])[::-1]
    num_true, num_false = len(true_scores), len(false_scores)
    # t_lo = true_scores[k] answers the k lower TRUE rows FALSE
    allowed = np.flatnonzero(_upper(np.arange(num_true), num_true, level)
                             <= EPSILON)
    t_lo = true_scores[allowed.max()] if len(allowed) else -np.inf
    share = _lower(num_true, num_true + num_false, level)
    limit = EPSILON * share / (1 - share) if share < 1 else np.inf
    # t_hi = false_scores[j] answers the j higher FALSE rows TRUE
    allowed = np.flatnonzero(_upper(np.arange(num_false), num_false, level)
                             <= limit)
    t_hi = false_scores[allowed.max()] if len(allowed) else np.inf
    return min(t_lo, t_hi), t_hi


def best_thresholds(scores, positive):
    """Choose the narrowest band meeting both targets on these rows themselves."""
    true_scores = np.sort(scores[positive])
    false_scores = np.sort(scores[~positive])[::-1]
    t_lo = true_scores[int(EPSILON * len(true_scores))]
    kept = len(true_scores) - int(EPSILON * len(true_scores))
    wrong = int(EPSILON / (1 - EPSILON) * kept)
    t_hi = false_scores[wrong] if wrong < len(false_scores) else -np.inf
    return min(t_lo, t_hi), t_hi


def answered(scores, positive, t_lo, t_hi):
    """Return precision, recall, and escalated fraction of the cascade."""
    escalated = (scores >= t_lo) & (scores <= t_hi)
    output = (scores > t_hi) | (escalated & positive)
    both = int((output & positive).sum())
    precision = both / max(1, int(output.sum()))
    recall = both / max(1, int(positive.sum()))
    return precision, recall, float(escalated.mean())


def _prefill_costs():
    from quail.common.decode import DecodeCost
    from quail.specs import DEVICES, MODELS
    device = DEVICES[DEVICE]
    teacher = DecodeCost.from_specs(MODELS[TEACHER], device).recompute_cost
    student = DecodeCost.from_specs(MODELS[STUDENT], device).recompute_cost
    return teacher.prefill_seconds, student.prefill_seconds, device.usd_per_hour


def _lengths(directory):
    """Return teacher and student prompt lengths of every traced review."""
    from transformers import AutoTokenizer

    import experiments.cells.distill_learning_curves as cell

    cell.TRACE_DIR = str(directory / "trace")
    cell.PROMPT_PIECES = str(directory / "prompt_pieces.json")
    data = cell._load()
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        cell.TEACHER, revision=cell.TEACHER_REVISION)
    student_tokenizer = AutoTokenizer.from_pretrained(
        cell.STUDENT, revision=cell.STUDENT_REVISION)
    texts = [student_tokenizer.decode(prompt) for prompt in data["prompts"]]
    teacher_lengths = np.array([len(teacher_tokenizer.encode(
        text, add_special_tokens=False)) for text in texts])
    student_lengths = np.array([len(prompt) for prompt in data["prompts"]])
    return data, teacher_lengths, student_lengths


def _costs(record, num_calibration, escalated, per_row):
    """Return SoL seconds of the distilled plan and of the teacher alone."""
    teacher_row, student_row = per_row
    labels = record["num_labels"] + num_calibration
    if record["setup"] in TRAINED:
        per_step = MICRO_DOCUMENTS * ACCUMULATE
        steps = int(np.clip(EPOCHS * record["num_labels"] // per_step,
                            MIN_STEPS, MAX_STEPS))
        training = 2 * student_row * steps * per_step
    else:
        training = student_row * record["num_labels"]
    out = {}
    for rows in TABLE_ROWS:
        rest = max(0, rows - labels)
        distilled = (teacher_row * labels + training
                     + student_row * (rest + num_calibration)
                     + teacher_row * escalated * rest)
        out[str(rows)] = {"distilled_seconds": distilled,
                          "teacher_seconds": teacher_row * rows,
                          "speedup": teacher_row * rows / distilled}
    saved = teacher_row * (1 - escalated) - student_row
    fixed = training + student_row * num_calibration
    out["break_even_rows"] = (labels + fixed / saved) if saved > 0 else None
    return out


def main():
    """Simulate every saved run at every calibration size; print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data, teacher_lengths, student_lengths = _lengths(args.directory)
    teacher_seconds, student_seconds, usd_per_hour = _prefill_costs()
    per_row = (float(np.mean([teacher_seconds(t) for t in teacher_lengths])),
               float(np.mean([student_seconds(t) for t in student_lengths])))
    positive = data["margins"][data["test"]] > 0
    runs = (json.loads((args.directory / "records.json").read_text())["records"]
            + json.loads((args.directory / "probes.json").read_text())["records"])
    rng = np.random.default_rng(0)
    splits = {num: [rng.permutation(len(positive)) for _ in range(NUM_SPLITS)]
              for num in NUM_CALIBRATION}
    summary = []
    for record in runs:
        if record["setup"] not in SETUPS:
            continue
        scores = np.asarray(record["scores"], dtype=float)
        for num_calibration, orders in splits.items():
            met, escalated, best, precisions, recalls = [], [], [], [], []
            for order in orders:
                calibration, rest = order[:num_calibration], order[num_calibration:]
                t_lo, t_hi = guaranteed_thresholds(scores[calibration],
                                                   positive[calibration])
                precision, recall, fraction = answered(
                    scores[rest], positive[rest], t_lo, t_hi)
                met.append(precision >= 1 - EPSILON and recall >= 1 - EPSILON)
                escalated.append(fraction)
                precisions.append(precision)
                recalls.append(recall)
                b_lo, b_hi = best_thresholds(scores[rest], positive[rest])
                best.append(answered(scores[rest], positive[rest], b_lo, b_hi)[2])
            summary.append({
                "setup": record["setup"], "num_labels": record["num_labels"],
                "seed": record["seed"], "num_calibration": num_calibration,
                "coverage": float(np.mean(met)),
                "escalated": float(np.mean(escalated)),
                "best_escalated": float(np.mean(best)),
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                **_costs(record, num_calibration, float(np.mean(escalated)),
                         per_row)})
    result = {"epsilon": EPSILON, "delta": DELTA, "num_splits": NUM_SPLITS,
              "teacher": TEACHER, "student": STUDENT, "device": DEVICE,
              "usd_per_hour": usd_per_hour,
              "teacher_row_seconds": per_row[0], "student_row_seconds": per_row[1],
              "mean_teacher_tokens": float(teacher_lengths.mean()),
              "mean_student_tokens": float(student_lengths.mean()),
              "summary": summary}
    text = json.dumps(result, indent=1)
    if args.out is not None:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
