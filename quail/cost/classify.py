"""Classification costs: expected work priced like a filter's work.

The estimate sums each document's work (quail/cost/work.py) over every
document, weights each label's greedy decode rounds by its share of
the labels, and prices each dependent step with ideal batch packing.
Each nonempty step reads the weights. KV capacity and admission order
are not simulated.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from quail.cost.batches import batch_work, cached_profile
from quail.cost.sol import unrounded_seconds
from quail.cost.work import Work, ask, stream
from quail.labels import (
    DECODE_SCORING,
    LETTERS_SCORING,
    TREE_SCORING,
    decode_round_tokens,
    fed_nodes,
    read_nodes,
)


def suffix_lengths(scoring: str, labels, canvas_rows: int = 0,
                   draws: int = 1, starts=()) -> list[int]:
    """Estimate the request lengths for a classification scoring rule.

    Args:
        scoring: Label scoring rule: letters, trie_tree, or trie_decode.
        labels: Token sequence for each label.
        canvas_rows: Number of diffusion answer canvas rows.
        draws: Maximum number of diffusion draws per document.
        starts: Per label, which tokens begin a new word.

    Returns:
        Request lengths in tokens, including the answer cue. The packed
        trie feeds the read nodes and their ancestors. Greedy decoding
        feeds one run per round, the longest run of any label in that
        round, for as many rounds as the label with the most choices.

    Raises:
        ValueError: The scoring rule is unknown.
    """
    if scoring == LETTERS_SCORING:
        return [1 + canvas_rows] * (draws if canvas_rows else 1)
    if scoring == TREE_SCORING:
        return [len(fed_nodes(read_nodes(labels, starts)))]
    if scoring == DECODE_SCORING:
        fed = decode_round_tokens(labels)
        return [max(counts[round_] for counts in fed if len(counts) > round_)
                for round_ in range(max(map(len, fed)))]
    raise ValueError(f"unknown label scoring rule {scoring!r}")


@dataclass(frozen=True)
class ClassifyCost:
    """Estimated classification time, work, and token counts.

    Attributes:
        seconds: Component roofline time of the expected work at ideal packing.
        passes: Expected weight reads across the packed steps.
        work: Expected token, attention, and KV work.
        suffix_tokens: Expected suffix tokens streamed after the documents.
        rounds: Maximum number of rounds needed by any label.
    """

    seconds: float
    passes: float
    work: Work
    suffix_tokens: float
    rounds: int


def _first_work(prefix, shared, tail: int, resident: bool, window: int) -> Work:
    """Sum every document's prefix-and-tail work: scan, or ask on resident KV."""
    if resident:
        return ask(prefix, np.full_like(prefix, tail), window=window).total()
    fresh = prefix - shared + tail
    return ask(shared, fresh, window=window).total()


def _stream_work(context, suffixes, window: int, canvas_rows: int) -> Work:
    """Sum every document's stream of the suffixes after its context tokens."""
    # Every document receives the same suffixes in this forward pass.
    work = stream(context, [np.full_like(context, s) for s in suffixes],
                  window=window).total()
    return _canvas_work(work, canvas_rows)


def _canvas_work(work: Work, canvas_rows: int) -> Work:
    """Include the second attention read of a multi-row answer canvas."""
    if canvas_rows > 1:
        work += Work(pairs=work.pairs, kv_read=work.kv_read,
                     sliding_pairs=work.sliding_pairs,
                     sliding_kv_read=work.sliding_kv_read)
    return work


def _profile(prefix, shared, frame_tokens, suffixes, resident, window,
             canvas_rows, runs, answer_rows):
    """Summarize token and KV work by dependent step."""
    n = len(prefix)
    if runs is None:
        work = (_first_work(prefix, shared, frame_tokens, resident, window)
                + _stream_work(prefix + frame_tokens, suffixes, window, canvas_rows))
        rows = canvas_rows or sum(suffixes)
        rows = rows if answer_rows is None else answer_rows
        return [(work, rows * n, 1.0)], float(sum(suffixes)), 1
    steps = []
    suffix_tokens = 0.0
    for fed, count in Counter(tuple(fed) for fed in runs).items():
        share = count / len(runs)
        context = prefix + frame_tokens
        for index, tokens in enumerate(fed):
            own = (_first_work(prefix, shared, frame_tokens + tokens, resident, window)
                   if index == 0 else _canvas_work(
                       ask(context, np.full_like(prefix, tokens),
                           window=window).total(),
                       canvas_rows))
            if index == len(steps):
                steps.append((Work(), 0.0, 0.0))
            old, rows, probability = steps[index]
            steps[index] = (old + own * share, rows + n * share, probability + share)
            context = context + tokens
        suffix_tokens += sum(fed) * share
    return steps, suffix_tokens, len(steps)


def estimate_suffixes(head_tokens: int, frame_tokens: int, suffixes, *,
                    live: float, lengths, shared, chunk: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0, runs=None,
                    answer_rows: float | None = None,
                    head_rows: int = 0, memo=None) -> ClassifyCost:
    """Estimate expected classification work and its roofline time.

    Args:
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        suffixes: Request lengths in tokens, shared by all documents, in
            one round.
        live: Expected number of documents to classify.
        lengths: Document lengths in tokens.
        shared: Shared document prefix lengths, or an empty sequence.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether document KV is already available.
        canvas_rows: Number of answer canvas rows.
        runs: For a greedy decode, the tokens each label feeds in each
            of its rounds, with each label equally likely; it replaces
            suffixes. A round reads one row.
        answer_rows: Rows the readout processes per document. None reads
            the canvas rows or every request token in one round, and one
            row per decode round.
        head_rows: Output-head rows each answer row multiplies; 0 is the
            whole vocabulary. A decision model ignores it.
        memo: Query memo for document-length-dependent step summaries.

    Returns:
        Expected work over the documents, scaled to live, priced at
        ideal packing.
    """
    n = len(lengths)
    if live <= 0 or n == 0 or not (suffixes or runs):
        return ClassifyCost(0.0, 0.0, Work(), 0.0, 0)
    key = ("classify_steps", id(lengths), id(shared), head_tokens, frame_tokens,
           tuple(suffixes), resident, model.sliding_window, canvas_rows,
           None if runs is None else tuple(map(tuple, runs)), answer_rows)

    def build():
        prefix = np.asarray(getattr(lengths, "arrow_array", lengths), dtype=float)
        prefix = prefix + head_tokens
        borrowed = np.asarray(shared if len(shared) else np.zeros(n), dtype=float)
        borrowed = np.where(borrowed > 0, borrowed + head_tokens, 0.0)
        return _profile(prefix, borrowed, frame_tokens, suffixes, resident,
                        model.sliding_window, canvas_rows, runs, answer_rows)

    steps, suffix_tokens, rounds = cached_profile(memo, key, (lengths, shared), build)
    total = batch_work(steps, live, n, chunk, model, head_rows)
    passes = sum(count for _, count in total.batches)
    seconds = unrounded_seconds(total, model, device, chunk)
    return ClassifyCost(seconds, passes, total, suffix_tokens * live, rounds)


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int,
             model, device, resident: bool = False, draws: int = 1,
             starts=(), memo=None) -> ClassifyCost:
    """Estimate the cost of one classification scoring rule.

    Args:
        scoring: Label scoring rule name.
        live: Expected number of documents to classify.
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        labels: Token sequence for each label.
        lengths: Document lengths in tokens.
        shared: Shared document prefix lengths.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether document KV is already available.
        draws: Maximum number of diffusion draws per document.
        starts: Per label, which tokens begin a new word.
        memo: Query memo for document-length-dependent step summaries.

    Returns:
        Estimated execution time, work, and token counts.

    Raises:
        ValueError: The scoring rule is unknown.
    """
    canvas = model.answer_canvas
    canvas_rows = canvas.rows if canvas is not None else 0
    suffixes = suffix_lengths(scoring, labels, canvas_rows, draws, starts)
    runs = answer_rows = None
    if scoring == DECODE_SCORING:
        # a document stops decoding at the choice that decides its label
        runs = decode_round_tokens(labels)
    elif scoring == TREE_SCORING:
        answer_rows = float(len(read_nodes(labels, starts)))
    # one-token labels are all read at the cue row from their target
    # rows of the head; longer labels normalize over the vocabulary
    targets = {token for ids in labels for token in ids}
    return estimate_suffixes(
        head_tokens, frame_tokens, suffixes, live=live, lengths=lengths,
        shared=shared, chunk=chunk, model=model, device=device,
        resident=resident,
        head_rows=len(targets) if all(len(ids) == 1 for ids in labels) else 0,
        canvas_rows=(canvas_rows if scoring == LETTERS_SCORING
                     else model.canvas_tokens),
        runs=runs, answer_rows=answer_rows, memo=memo)
