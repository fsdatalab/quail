"""Classification work and time from sampled CPU scheduler replay.

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
    """Tokens of each suffix a document streams under one scoring rule.

    Every suffix starts with the answer cue's last token. Under
    ``letters`` the one suffix is the cue alone, whose row scores every
    letter, followed on a canvas model by the ``canvas_rows`` rows of
    the seeded canvas whose first row is read. Under ``trie_tree`` the
    one suffix holds every trie node once; every row is read. Under
    ``trie_decode`` a document sends one chain per round, the cue and
    the tokens decoded so far, for as many rounds as the mean label
    length rounded up, and reads each chain's last row; how many
    rounds it needs is decided as it runs.

    Raises:
        ValueError: The rule is not one of LABEL_SCORINGS.
    """
    if scoring == LETTERS_SCORING:
        return [1 + canvas_rows] * (draws if canvas_rows else 1)
    if scoring == TREE_SCORING:
        return [len(label_trie(labels))]
    if scoring == DECODE_SCORING:
        rounds = math.ceil(sum(len(ids) for ids in labels) / len(labels))
        return [1 + depth for depth in range(rounds)]
    raise ValueError(f"unknown label scoring rule {scoring!r}")


def readout_component(rows: float, model) -> CostComponent:
    """The label readout: every read row through the whole output head."""
    return CostComponent(
        name="readout", flops=2.0 * model.hidden * model.vocab * rows,
        # the bf16 head streams from memory once per chunk that reads
        bytes_moved=2.0 * model.hidden * model.vocab if rows else 0.0,
        precision="bf16")


def chunk_seconds(work: Work, rows: float, model, device) -> float:
    """Price one forward chunk at the roofline: weights stream once."""
    components = dense_decoder_components(work, model, passes=1.0)
    components += (readout_component(rows, model),)
    return sum(component.seconds
               for component in component_latencies(components, device))


@dataclass(frozen=True)
class Simulated:
    """What a scoring rule costs on a table, from the replayed scheduler.

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
        """The simulation of ``factor`` times as many documents alike."""
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
    """Replay the stage scheduler on the CPU and price each chunk.

    Documents are admitted in order while their reservation, the
    prefix plus the frame and longest chain, fits the arena, and a
    chunk takes documents up to the chunk budget; a document's request
    is atomic. A document sends its chains once, or with
    ``one_per_round`` one of its chains per round, in order, reading
    only that chain's last row. A round's answers are read while the
    next chunk runs, so the document's next round enters the chunk
    after that, or the next chunk when nothing else is ready; waiting
    rounds enter a chunk before fresh documents. A document leaves the
    arena once its last round's chunk has run. Each chunk costs its
    roofline time with the weights streamed once, plus the readout of
    every row it reads. Every document is priced at every round; a
    decode that ends sooner takes fewer.

    Args:
        prefixes: Per document, the tokens before the frame (the prompt
            head and the document).
        frame: The frame tokens written once per document.
        chains: Per document, the lengths of the chains it streams.
        chunk: The chunk budget in tokens.
        capacity: The arena's tokens.
        model: The ModelSpec.
        device: The DeviceSpec.
        resident: Whether the prefixes are in KV already.
        canvas_rows: The rows of the answer canvas a document's suffix
            packs, which read the document a second time and are the
            rows read; 0 for a rule that reads its suffixes' rows.
        one_per_round: Each round sends one of the document's chains,
            in order, and reads its last row.
        shared: Per document, the leading prefix tokens an earlier
            document already computed, which it borrows from KV
            instead of computing; None when no prefix is shared.
    """
    window = model.sliding_window
    n = len(prefixes)
    shared = [0] * n if shared is None else shared
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
    """A sample of the documents expected to reach a classification.

    Takes ``live`` documents, at most SAMPLE_DOCUMENTS, spread evenly
    over the documents sorted by length, so the sample keeps the
    table's length distribution.

    Returns:
        Per sampled document, its length and its shared prefix
        tokens; and how many expected documents each stands for.
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
                    one_per_round: bool = False) -> Simulated:
    """Sample a table, replay prepared suffix chains, and scale their cost.

    The planner supplies the prompt head, frame, and suffix lengths.
    Shared document prefixes include the prompt head when replayed.
    """
    documents, weight = sample_documents(lengths, shared, live)
    prefixes = [head_tokens + length for length, _ in documents]
    # a borrowed prefix includes the prompt head the documents share
    shared_prefixes = [head_tokens + tokens if tokens else 0
                       for _, tokens in documents]
    return simulate(prefixes, frame_tokens, [chains] * len(prefixes),
                    chunk, capacity, model, device, resident=resident,
                    canvas_rows=canvas_rows, one_per_round=one_per_round,
                    shared=shared_prefixes).scaled(weight)


def estimate(scoring: str, live: float, head_tokens: int, frame_tokens: int,
             labels, *, lengths, shared, chunk: int, capacity: int,
             model, device, resident: bool = False, draws: int = 1) -> Simulated:
    """Replay one scoring rule over the expected documents and price it."""
    canvas = model.answer_canvas
    canvas_rows = canvas.rows if canvas is not None else 0
    chains = suffix_lengths(scoring, labels, canvas_rows, draws)
    return estimate_chains(
        head_tokens, frame_tokens, chains, live=live, lengths=lengths,
        shared=shared, chunk=chunk, capacity=capacity, model=model,
        device=device, resident=resident,
        canvas_rows=(canvas_rows if scoring == LETTERS_SCORING
                     else model.canvas_tokens),
        one_per_round=scoring == DECODE_SCORING)
