"""Estimate classification work and time by replaying sampled requests on the CPU.

The cost helpers take prepared token counts and model and device specs.
Prompt preparation, scoring rule selection, and plan construction belong
in the planner.
"""

import math
from collections import deque
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
    """Estimate the request lengths for a classification scoring method.

    Args:
        scoring: Method name: letters, trie_tree, or trie_decode.
        labels: Token sequence for each category.
        canvas_rows: Number of diffusion answer canvas rows.
        draws: Maximum number of diffusion draws per document.

    Returns:
        Request lengths in tokens, including the answer cue. Greedy decoding
        feeds one token per round, for as many rounds as the longest label.

    Raises:
        ValueError: The scoring method is unknown.
    """
    if scoring == LETTERS_SCORING:
        return [1 + canvas_rows] * (draws if canvas_rows else 1)
    if scoring == TREE_SCORING:
        return [len(label_trie(labels))]
    if scoring == DECODE_SCORING:
        return [1] * max(len(ids) for ids in labels)
    raise ValueError(f"unknown label scoring rule {scoring!r}")


def readout_component(rows: float, model) -> CostComponent:
    """Estimate computation and memory reads for the full output head."""
    return CostComponent(
        name="readout", flops=2.0 * model.hidden * model.vocab * rows,
        # the bf16 head streams from memory once per chunk that reads
        bytes_moved=2.0 * model.hidden * model.vocab if rows else 0.0,
        precision="bf16")


def chunk_seconds(work: Work, rows: float, model, device) -> float:
    """Estimate one forward pass in seconds using the GPU roofline model."""
    components = dense_decoder_components(work, model, passes=1.0)
    components += (readout_component(rows, model),)
    return sum(component.seconds
               for component in component_latencies(components, device))


@dataclass(frozen=True)
class Simulated:
    """Estimated time, work, and token counts from a scheduler replay.

    Attributes:
        seconds: The summed roofline time of every chunk.
        passes: Forward chunks launched.
        work: The token, attention, and KV work of every chunk.
        suffix_tokens: Suffix tokens streamed after the documents.
        rounds: The most rounds any document ran.
    """

    seconds: float
    passes: int
    work: Work
    suffix_tokens: float
    rounds: int

    def scaled(self, factor: float) -> "Simulated":
        """Scale estimated time, work, and passes by the document-count factor."""
        if factor == 1:
            return self
        return Simulated(self.seconds * factor,
                         int(math.ceil(self.passes * factor)),
                         self.work * factor, self.suffix_tokens * factor,
                         self.rounds)


# The documents a replay prices: past this many, an even sample over
# the lengths is replayed and scaled, which moves the estimate of
# AGENT-4 at sf 1.0 (17,711 traces) by under 0.3%
SAMPLE_DOCUMENTS = 1000


def simulate(prefixes, frame: int, chains, chunk: int, capacity: int,
             model, device, resident: bool = False, canvas_rows: int = 0,
             one_per_round: bool = False, shared=None) -> Simulated:
    """Replay classification scheduling on the CPU and estimate execution time.

    Requests are kept whole within chunks. Documents reserve KV until their
    last round finishes. Waiting rounds run before new documents. Each
    chunk's estimate includes one read of the model weights and the output
    head work for its answer rows.

    Args:
        prefixes: Prompt head and document lengths, in tokens, per document.
        frame: Number of tokens written after each document before requests.
        chains: Request lengths in tokens for each document.
        chunk: Maximum tokens per forward pass.
        capacity: KV arena capacity in tokens.
        model: Model specification.
        device: Device specification.
        resident: Whether document prefixes are already in KV.
        canvas_rows: Number of answer canvas rows. Zero uses suffix rows
            for the readout.
        one_per_round: Whether to send one chain per round and read its last
            row, instead of sending all chains in one round. Each round's
            tokens stay in KV, so later rounds read them as context.
        shared: Shared prefix length per document. None means no sharing.

    Returns:
        Simulated execution time, forward-pass count, work, and suffix tokens.
    """
    window = model.sliding_window
    n = len(prefixes)
    shared = [0] * n if shared is None else shared
    if one_per_round:
        longest = max(map(sum, chains), default=0)
    else:
        longest = max((length for document in chains for length in document),
                      default=0)
    extra = frame + longest

    def round_chains(document, round_):
        if one_per_round:
            return [chains[document][round_]]
        return chains[document]

    def round_count(document):
        return len(chains[document]) if one_per_round else 1

    def first_tokens(document):
        tokens = sum(round_chains(document, 0)) + frame
        if not resident:
            tokens += prefixes[document] - shared[document]
        return tokens

    def suffix_work(document, round_):
        prefix = prefixes[document]
        if one_per_round:
            prefix += sum(chains[document][:round_])
        # a canvas longer than one row reads the document a second
        # time, in the non-causal call every canvas row runs
        suffixes = stream(prefix + frame, round_chains(document, round_),
                          window=window)
        if canvas_rows > 1:
            suffixes = suffixes + Work(
                pairs=suffixes.pairs, kv_read=suffixes.kv_read,
                sliding_pairs=suffixes.sliding_pairs,
                sliding_kv_read=suffixes.sliding_kv_read)
        return suffixes

    def first_work(document):
        prefix = prefixes[document]
        suffixes = suffix_work(document, 0)
        if resident:
            return ask(prefix, frame, window=window) + suffixes
        if shared[document]:
            return ask(shared[document], prefix - shared[document] + frame,
                       window=window) + suffixes
        return scan(prefix + frame, 0, window=window) + suffixes

    def read_rows(document, round_):
        if one_per_round:
            return 1
        return canvas_rows or sum(chains[document])

    next_document = 0
    seconds = 0.0
    passes = 0
    total = Work()
    suffix_tokens = 0.0
    most_rounds = 0
    held = 0
    waiting = deque()    # (first chunk it may enter, document, round)
    while next_document < n or waiting:
        room = chunk
        work = Work()
        rows = 0.0
        launched = []
        while waiting and waiting[0][0] <= passes:
            _, document, round_ = waiting[0]
            tokens = sum(round_chains(document, round_))
            if tokens > room and room != chunk:
                break
            waiting.popleft()
            room -= tokens
            work += suffix_work(document, round_)
            rows += read_rows(document, round_)
            suffix_tokens += tokens
            launched.append((document, round_))
        while (next_document < n
               and (held + prefixes[next_document] + extra <= capacity
                    or not held)
               and (first_tokens(next_document) <= room or room == chunk)):
            document = next_document
            next_document += 1
            held += prefixes[document] + extra
            room -= first_tokens(document)
            work += first_work(document)
            rows += read_rows(document, 0)
            suffix_tokens += sum(round_chains(document, 0))
            launched.append((document, 0))
        if not launched:
            # the waiting rounds' answers are read before anything packs
            waiting = deque((min(ready, passes), document, round_)
                            for ready, document, round_ in waiting)
            continue
        seconds += chunk_seconds(work, rows, model, device)
        for document, round_ in launched:
            most_rounds = max(most_rounds, round_ + 1)
            if round_ + 1 < round_count(document):
                waiting.append((passes + 2, document, round_ + 1))
            else:
                held -= prefixes[document] + extra
        passes += 1
        total += work
    return Simulated(seconds, passes, total, suffix_tokens, most_rounds)


def sample_documents(
        lengths, shared, live: float) -> tuple[list[tuple[int, int]], float]:
    """Sample documents across the table's sorted length distribution.

    Args:
        lengths: Document lengths in tokens.
        shared: Shared prefix length per document, or an empty sequence.
        live: Expected number of documents reaching the classification.

    Returns:
        A tuple of sampled (length, shared prefix length) pairs and the
        number of expected documents represented by each sample. The sample
        contains at most SAMPLE_DOCUMENTS entries.
    """
    shared = shared or (0,) * len(lengths)
    ordered = sorted(zip(lengths, shared))
    if not ordered:
        return [], 1.0
    expected = max(1, int(round(live)))
    count = min(len(ordered), expected, SAMPLE_DOCUMENTS)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in picks], max(1.0, expected / count)


def estimate_chains(head_tokens: int, frame_tokens: int, chains, *,
                    live: float, lengths, shared, chunk: int, capacity: int,
                    model, device, resident: bool = False,
                    canvas_rows: int = 0,
                    one_per_round: bool = False, depths=None) -> Simulated:
    """Estimate classification cost from a sample of document lengths.

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
        depths: Optional rounds per document, given to the sampled
            documents in turn. None runs every round for every document.

    Returns:
        Simulated cost scaled to the expected document count.
    """
    documents, weight = sample_documents(lengths, shared, live)
    prefixes = [head_tokens + length for length, _ in documents]
    # a borrowed prefix includes the prompt head the documents share
    shared_prefixes = [head_tokens + tokens if tokens else 0
                       for _, tokens in documents]
    rounds = ([chains] * len(prefixes) if depths is None
              else [chains[:depths[i % len(depths)]]
                    for i in range(len(prefixes))])
    return simulate(prefixes, frame_tokens, rounds,
                    chunk, capacity, model, device, resident=resident,
                    canvas_rows=canvas_rows, one_per_round=one_per_round,
                    shared=shared_prefixes).scaled(weight)


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int, capacity: int,
             model, device, resident: bool = False, draws: int = 1) -> Simulated:
    """Estimate the cost of one classification scoring method.

    Args:
        scoring: Scoring method name.
        live: Expected number of documents to classify.
        head_tokens: Number of prompt tokens before each document.
        frame_tokens: Number of prompt tokens written after each document.
        labels: Token sequence for each category.
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
        ValueError: The scoring method is unknown.
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
        # a document stops decoding at its label's last token
        depths=([len(ids) for ids in labels] if scoring == DECODE_SCORING
                else None))
