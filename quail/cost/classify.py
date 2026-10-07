"""Classification costs: expected work priced like a filter's work.

The estimate sums each document's work (quail/cost/work.py) over every
document, weights each greedy decode round by the share of labels that
reach it, and prices the total with the component roofline at ideal
packing: forward passes are tokens divided by the chunk, as in
quail/cost/sol.py. KV capacity, the two-pass round delay, and admission
order do not enter, just as they do not enter filter and join costs.
"""

from dataclasses import dataclass

import numpy as np

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work
from quail.labels import (
    DECODE_SCORING,
    LETTERS_SCORING,
    TREE_SCORING,
    label_trie,
)


def suffix_lengths(scoring: str, labels, canvas_rows: int = 0,
                   draws: int = 1) -> list[int]:
    """Estimate the request lengths for a classification scoring rule.

    Args:
        scoring: Label scoring rule: letters, trie_tree, or trie_decode.
        labels: Token sequence for each label.
        canvas_rows: Number of diffusion answer canvas rows.
        draws: Maximum number of diffusion draws per document.

    Returns:
        Request lengths in tokens, including the answer cue. Greedy decoding
        feeds one token per round, for as many rounds as the longest label.

    Raises:
        ValueError: The scoring rule is unknown.
    """
    if scoring == LETTERS_SCORING:
        return [1 + canvas_rows] * (draws if canvas_rows else 1)
    if scoring == TREE_SCORING:
        return [len(label_trie(labels))]
    if scoring == DECODE_SCORING:
        return [1] * max(len(ids) for ids in labels)
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


def estimate_chains(head_tokens: int, frame_tokens: int, chains, *,
                    live: float, lengths, shared, chunk: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0,
                    one_per_round: bool = False, depths=None,
                    answer_rows: float | None = None,
                    head_rows: int = 0) -> ClassifyCost:
    """Estimate expected classification work and its roofline time.

    Args:
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        chains: Request lengths in tokens, shared by all documents.
        live: Expected number of documents to classify.
        lengths: Document lengths in tokens.
        shared: Shared document prefix lengths, or an empty sequence.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether document KV is already available.
        canvas_rows: Number of answer canvas rows.
        one_per_round: Whether each request runs in a separate decoder round,
            packed with the frame into the prefix's request in the first.
        depths: Label lengths in rounds, with each label equally likely.
            None runs every round for every document.
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
    if live <= 0 or n == 0 or not chains:
        return ClassifyCost(0.0, 0.0, Work(), 0.0, 0)
    window = model.sliding_window
    prefix = np.asarray(lengths, dtype=float) + head_tokens
    shared = np.asarray(shared if len(shared) else np.zeros(n), dtype=float)
    # a borrowed prefix includes the prompt head the documents share
    shared = np.where(shared > 0, shared + head_tokens, 0.0)
    rounds = len(chains) if one_per_round else 1
    reach = np.ones(rounds)
    if one_per_round and depths is not None:
        reach = np.array([sum(depth > round_ for depth in depths) / len(depths)
                          for round_ in range(rounds)])
    first_tail = frame_tokens + (chains[0] if one_per_round else 0)
    work = _first_work(prefix, shared, first_tail, resident, window)
    context = prefix + first_tail
    if one_per_round:
        for round_ in range(1, rounds):
            work += _stream_work(context, [chains[round_]], window,
                                 canvas_rows) * float(reach[round_])
            context = context + chains[round_]
    else:
        work += _stream_work(context, chains, window, canvas_rows)
    scale = live / n
    total = work * scale
    if one_per_round:
        suffix_tokens = float(np.dot(reach, chains))
        rows = float(reach.sum())
    else:
        suffix_tokens = float(sum(chains))
        rows = float(canvas_rows or sum(chains))
    if answer_rows is not None:
        rows = answer_rows
    passes = total.tokens / chunk
    seconds = _seconds(total, rows * live, passes, model, device, head_rows)
    return ClassifyCost(seconds, passes, total, suffix_tokens * live, rounds)


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int,
             model, device, resident: bool = False, draws: int = 1) -> ClassifyCost:
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

    Returns:
        Estimated execution time, work, and token counts.

    Raises:
        ValueError: The scoring rule is unknown.
    """
    canvas = model.answer_canvas
    canvas_rows = canvas.rows if canvas is not None else 0
    chains = suffix_lengths(scoring, labels, canvas_rows, draws)
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
        one_per_round=scoring == DECODE_SCORING,
        # a document stops decoding at its label's last token
        depths=([len(ids) for ids in labels] if scoring == DECODE_SCORING
                else None))
