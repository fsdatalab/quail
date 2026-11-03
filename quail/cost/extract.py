"""Extraction costs: expected work priced like a classification's work.

The estimate sums each document's work over every document, scales it
to the expected rows, and prices the total with the component roofline
at ideal packing, as quail/cost/classify.py does. The steps follow the
executor (quail/backends/quail/executor/extract/): the prompt with
the document, then the answer, one chosen token and a draft of the
document per round, every row read. The answer length is a cost
assumption: answers are taken as geometric with the assumed mean, and
a document's rounds are the expected number under that curve. Drafts
are taken as accepted while the answer continues. The second and later
extractions of a pipeline reuse the document's KV. Each answer round
reads the document, question, and previously accepted answer tokens
from KV.
"""

from dataclasses import dataclass, fields, replace

import numpy as np

from quail.cost.budgets import parallel_copies
from quail.cost.draft_policy import DraftPolicy, draft_lengths
from quail.cost.sol import unrounded_seconds
from quail.cost.work import Work, ask, scan, triangle

# a larger table is priced from this many evenly spaced documents, so
# planning time stays flat as the table grows
SAMPLE_DOCUMENTS = 1024


@dataclass(frozen=True)
class ExtractAssumptions:
    """The answer-length curve the extraction cost estimate assumes.

    Answers are geometric with mean answer_tokens: the chance an answer
    has more than j tokens after its first is (1 - 1/answer_tokens)**j.
    The executor's adaptive draft policy starts from the same curve.
    Actual answer lengths are not available during planning.
    """

    answer_tokens: int = 16

    def __post_init__(self):
        if type(self.answer_tokens) is not int or self.answer_tokens < 0:
            raise ValueError("answer_tokens must be a nonnegative integer")

    def survival(self, tokens: int) -> float:
        """Return the chance an answer has more than tokens tokens after its first."""
        if self.answer_tokens == 0:
            return 0.0
        return float(self._keep() ** tokens)

    def rounds(self, max_answer_tokens: int, draft_tokens) -> float | np.ndarray:
        """Return the expected rounds an answer takes, limited by the answer limit.

        The cue chooses the first token. Round r, counting from zero,
        runs when the answer has more than r (1 + draft_tokens) tokens
        after it; at the limit no stop is read, so the limit minus one
        token bounds the rounds. The sum over rounds is geometric.

        Args:
            max_answer_tokens: The answer limit.
            draft_tokens: Draft tokens per round, or an array of them.

        Returns:
            The expected rounds, an array when draft_tokens is one.
        """
        per_round = 1 + np.asarray(draft_tokens)
        most = np.maximum(0, -(-(max_answer_tokens - 1) // per_round))
        keep = self._keep()
        if self.answer_tokens == 0:
            rounds = np.zeros(np.shape(per_round))
        elif keep == 0.0:
            rounds = (most > 0).astype(float)
        else:
            log_q = per_round * np.log(keep)
            rounds = np.expm1(most * log_q) / np.expm1(log_q)
        return float(rounds) if np.ndim(rounds) == 0 else rounds

    def _keep(self) -> float:
        return 1.0 - 1.0 / max(1.0, float(self.answer_tokens))


@dataclass(frozen=True)
class ExtractCost:
    """Estimated extraction time, work, and token counts.

    Attributes:
        seconds: Component roofline time of the expected work at ideal packing.
        passes: Expected fresh tokens divided by the chunk.
        work: Expected token, attention, and KV work.
        suffix_tokens: Expected tokens fed after the documents.
        rows: Expected rows the readout processes.
    """

    seconds: float
    passes: float
    work: Work
    suffix_tokens: float
    rows: float


def document_work(length, head_tokens: int, tail_tokens: int, *,
                  window: int = 0, resident: bool = False,
                  shared=0, max_answer_tokens: int, draft_tokens: int,
                  draft_policy: str = "adaptive", round_cost_per_token: float = 0.0,
                  assumptions: ExtractAssumptions | None = None,
                  recurrent: bool = False,
                  ) -> tuple[Work, np.ndarray, np.ndarray]:
    """Return each document's work, its fed tokens, and its readout rows.

    Every count is an array with one entry per document. The expected
    rounds and their decode work are closed-form sums, so the cost does
    not grow with the answer limit.

    Args:
        length: The document's tokens, or an array of document lengths.
        head_tokens: Prompt tokens before the document.
        tail_tokens: Prompt tokens after the document, through the cue.
        window: The model's sliding window, 0 for none.
        resident: Whether the document's KV is already available, as
            it is for the second and later extractions of a pipeline.
        shared: Prefix tokens borrowed from an earlier document, one per
            document; 0 borrows nothing.
        max_answer_tokens: Answer limit from the extraction plan.
        draft_tokens: Source tokens proposed per round from the extraction
            plan; the cap under the adaptive policy.
        draft_policy: "fixed" or "adaptive", from the extraction plan.
        round_cost_per_token: One document token's KV read per round, in
            fed tokens, from the extraction plan.
        assumptions: Expected answer length, separate from execution limits.
        recurrent: Whether rounds replay accepted inputs to advance a checkpoint.
    """
    assumptions = ExtractAssumptions() if assumptions is None else assumptions
    length = np.asarray(length, dtype=np.float64)
    shape = length.shape
    prefix = head_tokens + length
    context = prefix + tail_tokens
    if draft_policy == "adaptive":
        # the policy's first-round choice for each document, with one stream
        policy = DraftPolicy(draft_tokens, max_answer_tokens, assumptions.answer_tokens)
        ones = np.ones(context.size, dtype=np.int64)
        draft_tokens = draft_lengths(
            policy.cap, policy.max_answer_tokens, policy.curve(), ones, ones,
            round_cost_per_token * context.reshape(-1), policy.trials,
            policy.accepts).reshape(shape)
    borrowed = np.zeros(shape) if resident else np.broadcast_to(
        np.asarray(shared, dtype=np.float64), shape)
    if resident:
        work = _per_document(ask(prefix, tail_tokens, window=window), shape)
    else:
        # a borrowed prefix includes the prompt head the documents share
        work = _where(
            borrowed > 0,
            ask(borrowed + head_tokens, context - borrowed - head_tokens,
                window=window),
            scan(prefix, tail_tokens, window=window), shape)
    if recurrent:
        work = replace(work, state_read=1.0 + (borrowed > 0),
                       state_written=np.full(shape, 1.0 + (not resident)))
    # each round appends the chosen token and the draft, and reads
    # every row; the last row read decides the closing quote
    round_tokens = 1 + np.broadcast_to(draft_tokens, shape)
    rounds = np.broadcast_to(assumptions.rounds(max_answer_tokens, draft_tokens),
                             shape)
    whole = np.floor(rounds)
    part = rounds - whole
    fed = tail_tokens + rounds * round_tokens
    if recurrent:
        # round 0 asks for one round's tokens; each later round replays
        # the previous round's tokens with its own to advance the state
        first = ask(context, round_tokens, window=window)
        work += replace(first, state_read=1.0, state_written=0.0) * np.minimum(
            1.0, rounds)
        later = np.maximum(whole - 1, 0)
        replays = _series(context, round_tokens, 2, later, window)
        work += replace(replays, state_read=2 * later, state_written=later)
        last = ask(context + later * round_tokens, 2 * round_tokens, window=window)
        partial = part * (whole >= 1)
        work += replace(last, state_read=2.0, state_written=1.0) * partial
        fed = fed + round_tokens * (later + partial)
    else:
        steps = _series(context, round_tokens, 1, whole, window)
        work += replace(steps, state_written=steps.state_read)
        last = ask(context + whole * round_tokens, round_tokens, window=window)
        work += replace(last, state_written=last.state_read) * part
    rows = 1.0 + rounds * round_tokens
    return work, fed, rows


def _series(context, step, multiple: int, count, window: int) -> Work:
    """Return the sum of ask(context + j step, multiple step) over j < count.

    Each document has its own context, step, and count of asks.
    """
    suffix = multiple * step
    kv_read = count * context + step * count * (count - 1) / 2
    sliding_pairs = sliding_kv_read = 0.0
    if window:
        # the per-ask window gains telescope: the sum keeps the last
        # multiple windows and drops the first multiple
        sliding_pairs = sum(
            triangle(context + (count + i) * step, window)
            - triangle(context + i * step, window) for i in range(multiple))
        # asks before the context reaches window - 1 read their whole context
        below = np.clip(np.ceil((window - 1 - context) / step), 0, count)
        sliding_kv_read = (below * context + step * below * (below - 1) / 2
                           + (count - below) * (window - 1))
    return Work(tokens=count * suffix, pairs=suffix * kv_read
                + count * triangle(suffix), kv_written=count * suffix,
                kv_read=kv_read, sliding_pairs=sliding_pairs,
                sliding_kv_read=sliding_kv_read, state_read=count)


def _per_document(work: Work, shape) -> Work:
    """Return work with every count broadcast to one entry per document."""
    return Work(**{field.name: np.broadcast_to(
        np.asarray(getattr(work, field.name), dtype=np.float64), shape)
        for field in fields(Work)})


def _where(mask, chosen: Work, other: Work, shape) -> Work:
    """Return chosen's counts where mask holds and other's elsewhere."""
    chosen, other = _per_document(chosen, shape), _per_document(other, shape)
    return Work(**{field.name: np.where(mask, getattr(chosen, field.name),
                                        getattr(other, field.name))
                   for field in fields(Work)})


def table_work(lengths, head_tokens: int, tail_tokens: int, *, window: int,
               resident: bool, shared, max_answer_tokens: int,
               draft_tokens: int, draft_policy: str,
               round_cost_per_token: float, assumptions: ExtractAssumptions,
               recurrent: bool = False, memo=None) -> tuple[Work, float, float]:
    """Return the work, fed tokens, and readout rows of every document.

    A planner passes its memo to compute the sum once per table and
    prompt; the key holds the length and shared columns by identity,
    and the memo keeps them alive so another column cannot reuse an
    identity. A table of more than SAMPLE_DOCUMENTS documents sums
    evenly spaced documents and scales the sum to the table.
    """
    key = ("extract_work", id(lengths), id(shared), head_tokens, tail_tokens,
           window, resident, max_answer_tokens, draft_tokens, draft_policy,
           round_cost_per_token, assumptions, recurrent)
    if memo is not None and key in memo:
        return memo[key][2]
    n = len(lengths)
    indices = (np.arange(n) if n <= SAMPLE_DOCUMENTS else
               np.arange(SAMPLE_DOCUMENTS) * (n - 1) // (SAMPLE_DOCUMENTS - 1))
    own, own_fed, own_rows = document_work(
        np.asarray(lengths)[indices], head_tokens, tail_tokens, window=window,
        resident=resident,
        shared=0 if shared is None else np.asarray(shared)[indices],
        max_answer_tokens=max_answer_tokens, draft_tokens=draft_tokens,
        draft_policy=draft_policy, round_cost_per_token=round_cost_per_token,
        assumptions=assumptions, recurrent=recurrent)
    shape = np.shape(indices)
    work = _per_document(own, shape).total()
    fed, rows = float(np.sum(own_fed)), float(np.sum(own_rows))
    if n > SAMPLE_DOCUMENTS:
        scale = n / SAMPLE_DOCUMENTS
        work, fed, rows = work * scale, fed * scale, rows * scale
    if memo is not None:
        memo[key] = (lengths, shared, (work, fed, rows))
    return work, fed, rows


def estimate(expected_rows: float, lengths, head_tokens: int, tail_tokens: int, *,
             chunk: int, model, device, resident: bool = False,
             shared=None, max_answer_tokens: int, draft_tokens: int,
             draft_policy: str = "adaptive", round_cost_per_token: float = 0.0,
             gpus: int = 1, assumptions: ExtractAssumptions | None = None,
             memo=None) -> ExtractCost:
    """Estimate the cost of one extraction over a table.

    Args:
        expected_rows: Expected number of documents to extract from.
        lengths: Document lengths in tokens.
        head_tokens: Prompt tokens before the document.
        tail_tokens: Prompt tokens after the document, through the cue.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether the documents' KV is already available.
        shared: Per-document borrowed prefix lengths, or None.
        max_answer_tokens: Answer limit from the extraction plan.
        draft_tokens: Source tokens proposed per round from the extraction
            plan; the cap under the adaptive policy.
        draft_policy: "fixed" or "adaptive", from the extraction plan.
        round_cost_per_token: One document token's KV read per round, in
            fed tokens, from the extraction plan.
        gpus: Model copies sharing the work, assuming balanced document costs.
        assumptions: Expected answer length, separate from execution limits.
        memo: Where table_work keeps the per-table sum, or None.

    Returns:
        Estimated execution time, work, and token counts.
    """
    n = len(lengths)
    if expected_rows <= 0 or n == 0:
        return ExtractCost(0.0, 0.0, Work(), 0.0, 0.0)
    work, fed, rows = table_work(
        lengths, head_tokens, tail_tokens, window=model.sliding_window,
        resident=resident, shared=shared, max_answer_tokens=max_answer_tokens,
        draft_tokens=draft_tokens, draft_policy=draft_policy,
        round_cost_per_token=round_cost_per_token,
        assumptions=ExtractAssumptions() if assumptions is None else assumptions,
        recurrent=bool(model.state_layers), memo=memo)
    scale = expected_rows / n
    total = work * scale
    copies = parallel_copies(gpus, min(expected_rows, n))
    seconds = unrounded_seconds(total * (1 / copies), model, device, chunk,
                                rows=rows * scale / copies)
    passes = total.tokens / chunk
    return ExtractCost(seconds, passes, total, fed * scale, rows * scale)
