"""Analytical classification costs from document lengths and label probabilities.

Fixed batches obey the fresh-token and KV budgets. Every label is equally
likely, independently of document length and other documents. Each round
prices expected work and weight reads with the shared component roofline.
Batches do not admit new documents as earlier documents finish decoding.
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
        seconds: Summed component roofline time across batches and rounds.
        passes: Expected number of nonempty forward passes.
        work: Expected token, attention, and KV work.
        suffix_tokens: Expected suffix tokens streamed after the documents.
        rounds: Maximum number of rounds needed by any label.
    """

    seconds: float
    passes: float
    work: Work
    suffix_tokens: float
    rounds: int

    def scaled(self, factor: float) -> "ClassifyCost":
        """Scale the estimate by the expected document-count factor."""
        return ClassifyCost(self.seconds * factor, self.passes * factor,
                            self.work * factor, self.suffix_tokens * factor,
                            self.rounds)


SAMPLE_DOCUMENTS = 1000


def sample_documents(
        lengths, shared, live: float) -> tuple[list[tuple[int, int]], float]:
    """Sample documents across the table's sorted length distribution.

    Args:
        lengths: Document lengths in tokens.
        shared: Shared prefix length per document, or an empty sequence.
        live: Expected number of documents reaching the classification.

    Returns:
        Sampled (length, shared prefix length) pairs and the number of expected
        documents represented by each sample, preserving fractional counts.
        The sample contains at most SAMPLE_DOCUMENTS entries.
    """
    if live <= 0 or not len(lengths):
        return [], 1.0
    shared = shared or (0,) * len(lengths)
    ordered = sorted(zip(lengths, shared))
    count = min(len(ordered), math.ceil(live), SAMPLE_DOCUMENTS)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in picks], live / count


def _batches(documents, head, frame, chains, chunk, capacity, resident,
             one_per_round):
    """Group whole first requests within the fresh-token and KV limits."""
    first = chains[0] if one_per_round else sum(chains)
    extra = frame + (sum(chains) if one_per_round else max(chains))
    batch = []
    tokens = held = 0
    for length, shared in documents:
        prefix = head + length
        shared_prefix = head + shared if shared else 0
        fresh = frame + first + (0 if resident else prefix - shared_prefix)
        reservation = prefix + extra
        if batch and (tokens + fresh > chunk or held + reservation > capacity):
            yield batch
            batch = []
            tokens = held = 0
        batch.append((prefix, shared_prefix))
        tokens += fresh
        held += reservation
    if batch:
        yield batch


def _suffix_work(prefix, chains, canvas_rows, window):
    """Count suffix work, including a canvas's second attention read."""
    suffixes = stream(prefix, chains, window=window)
    if canvas_rows > 1:
        suffixes += Work(
            pairs=suffixes.pairs, kv_read=suffixes.kv_read,
            sliding_pairs=suffixes.sliding_pairs,
            sliding_kv_read=suffixes.sliding_kv_read)
    return suffixes


def _batch_cost(batch, frame, chains, probabilities, model, device,
                resident, canvas_rows, one_per_round) -> ClassifyCost:
    """Price a fixed batch using each round's expected work and weight reads.

    A round runs if at least one document still needs it. For b documents
    with independent continuation probability p, that probability is
    1 - (1 - p)**b. The roofline prices expected work, an approximation to
    averaging the roofline time over every possible set of surviving rows.
    """
    window = model.sliding_window
    count = len(batch)
    total = Work()
    seconds = passes = suffix_tokens = 0.0
    previous = 0
    for round_, probability in enumerate(probabilities):
        suffixes = [chains[round_]] if one_per_round else chains
        work = Work()
        for prefix, shared in batch:
            work += _suffix_work(prefix + frame + previous, suffixes,
                                 canvas_rows, window)
            if round_ == 0:
                if resident:
                    work += ask(prefix, frame, window=window)
                elif shared:
                    work += ask(shared, prefix - shared + frame, window=window)
                else:
                    work += scan(prefix + frame, 0, window=window)
        work *= probability
        nonempty = (1.0 if probability == 1 else
                    -math.expm1(count * math.log1p(-probability)))
        rows = 1 if one_per_round else canvas_rows or sum(chains)
        seconds += _seconds(work, count * rows * probability, nonempty,
                            model, device)
        passes += nonempty
        total += work
        suffix_tokens += count * sum(suffixes) * probability
        previous += sum(suffixes)
    return ClassifyCost(seconds, passes, total, suffix_tokens, len(probabilities))


def estimate_chains(head_tokens: int, frame_tokens: int, chains, *,
                    live: float, lengths, shared, chunk: int, capacity: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0,
                    one_per_round: bool = False, depths=None) -> ClassifyCost:
    """Estimate classification work and roofline time over fixed batches.

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
        Expected cost scaled to the expected document count.
    """
    documents, weight = sample_documents(lengths, shared, live)
    if not documents or not chains:
        return ClassifyCost(0.0, 0.0, Work(), 0.0, 0)
    rounds = len(chains) if one_per_round else 1
    probabilities = [1.0] * rounds
    if one_per_round and depths is not None:
        probabilities = [sum(depth > round_ for depth in depths) / len(depths)
                         for round_ in range(rounds)]
    total = Work()
    seconds = passes = suffix_tokens = 0.0
    for batch in _batches(documents, head_tokens, frame_tokens, chains, chunk,
                          capacity, resident, one_per_round):
        cost = _batch_cost(batch, frame_tokens, chains, probabilities,
                           model, device, resident, canvas_rows, one_per_round)
        seconds += cost.seconds
        passes += cost.passes
        total += cost.work
        suffix_tokens += cost.suffix_tokens
    return ClassifyCost(seconds, passes, total, suffix_tokens, rounds).scaled(weight)


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
