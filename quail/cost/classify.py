"""Analytical classification costs from document lengths and label probabilities.

Every label is equally likely, independently of document length. Work is
averaged over the document-length sample before scaling to the input count.
Token and KV capacity determine analytical pass counts and an overlap fraction;
no batches, request queues, or per-document execution states are constructed.
"""

import math
from dataclasses import dataclass

import numpy as np

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work, ask, scan, stream
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


def _seconds(work: Work, rows: float, passes: float, model, device) -> float:
    """Price expected work and forward passes with the component roofline."""
    components = dense_decoder_components(work, model, passes=passes)
    components += (CostComponent(
        name="readout", flops=2.0 * model.hidden * model.vocab * rows,
        # The bf16 output head is read once per nonempty forward pass.
        bytes_moved=2.0 * model.hidden * model.vocab * passes,
        precision="bf16"),)
    return sum(component.seconds
               for component in component_latencies(components, device))


@dataclass(frozen=True)
class ClassifyCost:
    """Estimated classification time, work, and token counts.

    Attributes:
        seconds: Component roofline time with estimated prefill/decode overlap.
        passes: Estimated forward passes, including the final decode work.
        work: Expected token, attention, and KV work.
        suffix_tokens: Expected suffix tokens streamed after the documents.
        rounds: Maximum number of rounds needed by any label.
    """

    seconds: float
    passes: float
    work: Work
    suffix_tokens: float
    rounds: int


SAMPLE_DOCUMENTS = 1000


def sample_documents(lengths, shared) -> list[tuple[int, int]]:
    """Sample document and shared-prefix lengths across the sorted distribution."""
    shared = shared or (0,) * len(lengths)
    ordered = sorted(zip(lengths, shared))
    count = min(len(ordered), SAMPLE_DOCUMENTS)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in picks]


def _suffix_work(prefix, chains, canvas_rows, window):
    """Count suffix work, including a canvas's second attention read."""
    suffixes = stream(prefix, chains, window=window)
    if canvas_rows > 1:
        suffixes += Work(
            pairs=suffixes.pairs, kv_read=suffixes.kv_read,
            sliding_pairs=suffixes.sliding_pairs,
            sliding_kv_read=suffixes.sliding_kv_read)
    return suffixes


def _mean_work(documents, head, frame, chains, model, resident,
               canvas_rows, one_per_round) -> tuple[list[Work], float]:
    """Return per-document work by label depth and mean KV reservation."""
    window = model.sliding_window
    rounds = len(chains) if one_per_round else 1
    work = [Work() for _ in range(rounds)]
    reservation = 0.0
    for length, shared in documents:
        prefix = head + length
        shared_prefix = head + shared if shared else 0
        reservation += prefix + frame + (
            sum(chains) if one_per_round else max(chains))
        previous = 0
        for round_ in range(rounds):
            suffixes = [chains[round_]] if one_per_round else chains
            first_tail = frame
            if round_ == 0 and one_per_round:
                # Decode packs the frame and cue into the prefix's causal request.
                first_tail += sum(suffixes)
            else:
                work[round_] += _suffix_work(prefix + frame + previous, suffixes,
                                             canvas_rows, window)
            if round_ == 0:
                if resident:
                    work[0] += ask(prefix, first_tail, window=window)
                elif shared:
                    work[0] += ask(shared_prefix, prefix - shared_prefix + first_tail,
                                   window=window)
                else:
                    work[0] += scan(prefix, first_tail, window=window)
            previous += sum(suffixes)
    count = len(documents)
    return [value * (1 / count) for value in work], reservation / count


def _nonempty(probability, count):
    """Return the probability of at least one independent continuing document."""
    if count <= 0:
        return 0.0
    return (1.0 if probability == 1 else
            -math.expm1(count * math.log1p(-probability)))


def _aggregate_cost(work, probabilities, reservation, count, chunk, capacity,
                    rows, model, device) -> tuple[float, float]:
    """Price aggregate work with KV-limited overlap and a single terminal drain.

    Mean request size estimates documents per prefill pass. Spare KV divided
    by the expected KV of unfinished documents estimates the overlap fraction.
    The executor reads answers after launching another chunk, so two passes
    separate successive decode rounds when admission can continue.

    At decode depth r, estimate the remaining documents as 2*r times the
    admissions per pass, capped by the estimated resident population. Their
    work is removed from the aggregate and priced separately, once per query.
    Both occupancy and the final population are estimates, not a schedule.
    """
    slots = max(1, capacity // reservation)
    width = min(count, slots, max(1, chunk // max(1, work[0].tokens)))
    prefill_passes = count / width
    pending = 2 * width * sum(probabilities[1:])
    overlap = min(1.0, max(0.0, (slots - width) / pending)) if pending else 0.0
    residents = min(count, slots, width + pending)
    mixed_work = work[0] * count
    mixed_rows = rows * count
    separate_seconds = _seconds(mixed_work, mixed_rows, prefill_passes, model, device)
    separate_passes = prefill_passes
    tail_seconds = tail_passes = 0.0
    for depth, (value, probability) in enumerate(zip(work[1:], probabilities[1:]), 1):
        tail_count = min(residents, 2 * depth * width)
        remaining = count - tail_count
        bulk_work = value * (probability * remaining)
        bulk_rows = probability * remaining
        bulk_passes = max(
            remaining / width * _nonempty(probability, width), bulk_work.tokens / chunk)
        separate_seconds += _seconds(bulk_work, bulk_rows, bulk_passes, model, device)
        separate_passes += bulk_passes
        mixed_work += bulk_work
        mixed_rows += bulk_rows
        tail_work = value * (probability * tail_count)
        passes = max(_nonempty(probability, tail_count), tail_work.tokens / chunk)
        tail_seconds += _seconds(tail_work, probability * tail_count, passes,
                                 model, device)
        tail_passes += passes
    mixed_passes = max(prefill_passes, mixed_work.tokens / chunk)
    mixed_seconds = _seconds(mixed_work, mixed_rows, mixed_passes, model, device)
    return (
        overlap * mixed_seconds + (1 - overlap) * separate_seconds + tail_seconds,
        overlap * mixed_passes + (1 - overlap) * separate_passes + tail_passes,
    )


def estimate_chains(head_tokens: int, frame_tokens: int, chains, *,
                    live: float, lengths, shared, chunk: int, capacity: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0,
                    one_per_round: bool = False, depths=None) -> ClassifyCost:
    """Estimate aggregate classification work and roofline time.

    Args:
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        chains: Request lengths in tokens, shared by all documents.
        live: Expected number of documents to classify.
        lengths: Document lengths in tokens.
        shared: Shared document prefix lengths.
        chunk: Maximum tokens per forward pass.
        capacity: KV arena capacity in tokens.
        model: Model specification.
        device: Device specification.
        resident: Whether document KV is already available.
        canvas_rows: Number of answer canvas rows.
        one_per_round: Whether each request runs in a separate decoder round.
        depths: Label lengths in rounds, with each label equally likely.
            None runs every round for every document.

    Returns:
        Expected work and time, with one terminal drain for the whole query.
    """
    if live <= 0 or not len(lengths) or not chains:
        return ClassifyCost(0.0, 0.0, Work(), 0.0, 0)
    documents = sample_documents(lengths, shared)
    work, reservation = _mean_work(documents, head_tokens, frame_tokens, chains,
                                   model, resident, canvas_rows, one_per_round)
    probabilities = [1.0] * len(work)
    if one_per_round and depths is not None:
        probabilities = [sum(depth > round_ for depth in depths) / len(depths)
                         for round_ in range(len(work))]
    total = Work()
    for value, probability in zip(work, probabilities):
        total += value * (live * probability)
    rows = 1 if one_per_round else canvas_rows or sum(chains)
    seconds, passes = _aggregate_cost(
        work, probabilities, reservation, max(1.0, live), chunk, capacity,
        rows, model, device)
    suffix_tokens = live * (sum(length * probability for length, probability
                               in zip(chains, probabilities))
                            if one_per_round else sum(chains))
    return ClassifyCost(seconds * min(1.0, live), passes * min(1.0, live),
                        total, suffix_tokens, len(work))


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int, capacity: int,
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
        capacity: KV arena capacity in tokens.
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
    return estimate_chains(
        head_tokens, frame_tokens, chains, live=live, lengths=lengths,
        shared=shared, chunk=chunk, capacity=capacity, model=model,
        device=device, resident=resident,
        canvas_rows=(canvas_rows if scoring == LETTERS_SCORING
                     else model.canvas_tokens),
        one_per_round=scoring == DECODE_SCORING,
        depths=([len(ids) for ids in labels] if scoring == DECODE_SCORING
                else None))
