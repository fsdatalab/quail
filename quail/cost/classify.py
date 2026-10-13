"""Classification costs: expected work priced like a filter's work.

The estimate sums each document's work (quail/cost/work.py) over every
document, weights each label's greedy decode rounds by its share of
the labels, and prices the total with the component roofline at ideal
packing: forward passes are tokens divided by the chunk, as in
quail/cost/sol.py. KV capacity, the two-pass round delay, and admission
order do not enter, just as they do not enter filter and join costs.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work
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


def _readout(model, rows: float, passes: float, head_rows: int) -> CostComponent:
    """Price the answer readout: the decision head, or output-head rows.

    A decision model passes each option row and the prompt's last row
    through its fp32 head: four projections of the hidden size to the
    head width per row, and the head's weights once per pass. Any other
    model multiplies each answer row by head_rows rows of the bf16
    output head, the whole vocabulary when head_rows is 0, read once per
    pass.
    """
    if model.role == "decision":
        width, hidden = model.decision_head_dim, model.hidden
        return CostComponent(
            name="readout", flops=4.0 * width * hidden * rows,
            bytes_moved=4.0 * (4 * width * hidden + 2 * width + 4 * hidden) * passes,
            precision="fp32")
    head = 2.0 * model.hidden * (head_rows or model.vocab)
    return CostComponent(name="readout", flops=head * rows,
                         bytes_moved=head * passes, precision="bf16")


def _seconds(work: Work, rows: float, passes: float, model, device,
             head_rows: int) -> float:
    """Price work, answer rows, and forward passes with the component roofline."""
    components = dense_decoder_components(work, model, passes=passes)
    components += (_readout(model, rows, passes, head_rows),)
    return sum(component.seconds
               for component in component_latencies(components, device))


@dataclass(frozen=True)
class ClassifyCost:
    """Estimated classification time, work, and token counts.

    Attributes:
        seconds: Component roofline time of the expected work at ideal packing.
        passes: Expected fresh tokens divided by the chunk.
        work: Expected token, attention, and KV work.
        suffix_tokens: Expected suffix tokens streamed after the documents.
        rounds: Maximum number of rounds needed by any label.
    """

    seconds: float
    passes: float
    work: Work
    suffix_tokens: float
    rounds: int


def _triangle(n, window: int = 0):
    """Vectorized causal pair count, within the window when one is set."""
    n = np.asarray(n, dtype=float)
    full = n * (n + 1) / 2
    if not window:
        return full
    return np.where(n > window, window * n - window * (window - 1) / 2, full)


def _total(tokens, pairs, written, read, sliding_pairs, sliding_read) -> Work:
    """Sum per-document work arrays into one Work record."""
    return Work(float(np.sum(tokens)), float(np.sum(pairs)), float(np.sum(written)),
                float(np.sum(read)), float(np.sum(sliding_pairs)),
                float(np.sum(sliding_read)))


def _first_work(prefix, shared, tail: int, resident: bool, window: int) -> Work:
    """Sum every document's prefix-and-tail work: scan, or ask on resident KV."""
    if resident:
        # ask(prefix, tail)
        suffix = np.full_like(prefix, tail)
        return _total(
            suffix, tail * prefix + _triangle(tail), suffix, prefix,
            _triangle(prefix + tail, window) - _triangle(prefix, window)
            if window else 0.0,
            np.minimum(prefix, window - 1) if window else 0.0)
    # scan(prefix, tail) without a shared prefix, ask(shared, rest) with one
    fresh = prefix - shared + tail
    return _total(
        fresh, fresh * shared + _triangle(fresh), fresh, shared,
        _triangle(prefix + tail, window) - _triangle(shared, window)
        if window else 0.0,
        np.minimum(shared, window - 1) if window else 0.0)


def _stream_work(context, suffixes, window: int, canvas_rows: int) -> Work:
    """Sum every document's stream of the suffixes after its context tokens."""
    total = sum(suffixes)
    pairs = sum(s * context + _triangle(s) for s in suffixes)
    sliding_pairs = (sum(_triangle(context + s, window) - _triangle(context, window)
                         for s in suffixes) if window else 0.0)
    work = _total(
        np.full_like(context, total), pairs, np.full_like(context, total),
        context, sliding_pairs,
        np.minimum(context, window - 1) if window else 0.0)
    if canvas_rows > 1:
        # a canvas longer than one row reads the document a second time
        work += Work(pairs=work.pairs, kv_read=work.kv_read,
                     sliding_pairs=work.sliding_pairs,
                     sliding_kv_read=work.sliding_kv_read)
    return work


def _rounds_work(prefix, shared, frame_tokens: int, runs, resident: bool,
                 window: int, canvas_rows: int) -> Work:
    """Sum every document's work over one label's decode rounds.

    The first run is packed with the frame into the prefix's request;
    each later run streams after everything fed before it.
    """
    first_tail = frame_tokens + runs[0]
    work = _first_work(prefix, shared, first_tail, resident, window)
    context = prefix + first_tail
    for fed in runs[1:]:
        work += _stream_work(context, [fed], window, canvas_rows)
        context = context + fed
    return work


def estimate_chains(head_tokens: int, frame_tokens: int, chains, *,
                    live: float, lengths, shared, chunk: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0, runs=None,
                    answer_rows: float | None = None,
                    head_rows: int = 0) -> ClassifyCost:
    """Estimate expected classification work and its roofline time.

    Args:
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        chains: Request lengths in tokens, shared by all documents, in
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
            chains. A round reads one row.
        answer_rows: Rows the readout processes per document. None reads
            the canvas rows or every request token in one round, and one
            row per decode round.
        head_rows: Output-head rows each answer row multiplies; 0 is the
            whole vocabulary. A decision model ignores it.

    Returns:
        Expected work over the documents, scaled to live, priced at
        ideal packing.
    """
    n = len(lengths)
    if live <= 0 or n == 0 or not (chains or runs):
        return ClassifyCost(0.0, 0.0, Work(), 0.0, 0)
    window = model.sliding_window
    prefix = np.asarray(lengths, dtype=float) + head_tokens
    shared = np.asarray(shared if len(shared) else np.zeros(n), dtype=float)
    # a borrowed prefix includes the prompt head the documents share
    shared = np.where(shared > 0, shared + head_tokens, 0.0)
    if runs is not None:
        work = Work()
        suffix_tokens = rows = 0.0
        for fed, count in Counter(tuple(fed) for fed in runs).items():
            share = count / len(runs)
            work += _rounds_work(prefix, shared, frame_tokens, fed, resident,
                                 window, canvas_rows) * share
            suffix_tokens += sum(fed) * share
            rows += len(fed) * share
        rounds = max(map(len, runs))
    else:
        work = (_first_work(prefix, shared, frame_tokens, resident, window)
                + _stream_work(prefix + frame_tokens, chains, window, canvas_rows))
        suffix_tokens = float(sum(chains))
        rows = float(canvas_rows or sum(chains))
        rounds = 1
    if answer_rows is not None:
        rows = answer_rows
    total = work * (live / n)
    passes = total.tokens / chunk
    seconds = _seconds(total, rows * live, passes, model, device, head_rows)
    return ClassifyCost(seconds, passes, total, suffix_tokens * live, rounds)


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int,
             model, device, resident: bool = False, draws: int = 1,
             starts=()) -> ClassifyCost:
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

    Returns:
        Estimated execution time, work, and token counts.

    Raises:
        ValueError: The scoring rule is unknown.
    """
    canvas = model.answer_canvas
    canvas_rows = canvas.rows if canvas is not None else 0
    chains = suffix_lengths(scoring, labels, canvas_rows, draws, starts)
    runs = answer_rows = None
    if scoring == DECODE_SCORING:
        # a document stops decoding at the choice that decides its label
        runs = decode_round_tokens(labels)
    elif scoring == TREE_SCORING:
        answer_rows = float(len(read_nodes(labels, starts)))
    # one-token labels are all read at the cue row from their target
    # rows of the head; longer labels normalize over the vocabulary
    targets = {token for ids in labels for token in ids}
    return estimate_chains(
        head_tokens, frame_tokens, chains, live=live, lengths=lengths,
        shared=shared, chunk=chunk, model=model, device=device,
        resident=resident,
        head_rows=len(targets) if all(len(ids) == 1 for ids in labels) else 0,
        canvas_rows=(canvas_rows if scoring == LETTERS_SCORING
                     else model.canvas_tokens),
        runs=runs, answer_rows=answer_rows)
