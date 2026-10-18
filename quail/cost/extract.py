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

from dataclasses import dataclass
from itertools import repeat

from quail.cost.budgets import parallel_copies
from quail.cost.dense_decoder_cost import decoder_seconds
from quail.cost.draft_policy import DraftPolicy
from quail.cost.work import Work, ask, decode, scan


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
        return (1 - 1 / self.answer_tokens) ** tokens

    def rounds(self, max_answer_tokens: int, draft_tokens: int) -> float:
        """Return the expected rounds an answer takes, limited by the answer limit.

        The cue chooses the first token. Round r, counting from zero,
        runs when the answer has more than r (1 + draft_tokens) tokens
        after it; at the limit no stop is read, so the limit minus one
        token bounds the rounds.
        """
        per_round = 1 + draft_tokens
        most = -(-(max_answer_tokens - 1) // per_round)
        return float(sum(self.survival(r * per_round) for r in range(most)))


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


def document_work(length: int, head_tokens: int, tail_tokens: int, *,
                  window: int = 0, resident: bool = False,
                  shared: int = 0, max_answer_tokens: int, draft_tokens: int,
                  draft_policy: str = "adaptive", round_cost_per_token: float = 0.0,
                  assumptions: ExtractAssumptions | None = None,
                  ) -> tuple[Work, float, float]:
    """Return one document's work, its fed tokens, and its readout rows.

    Args:
        length: The document's tokens.
        head_tokens: Prompt tokens before the document.
        tail_tokens: Prompt tokens after the document, through the cue.
        window: The model's sliding window, 0 for none.
        resident: Whether the document's KV is already available, as
            it is for the second and later extractions of a pipeline.
        shared: Prefix tokens borrowed from an earlier document.
        max_answer_tokens: Answer limit from the extraction plan.
        draft_tokens: Source tokens proposed per round from the extraction
            plan; the cap under the adaptive policy.
        draft_policy: "fixed" or "adaptive", from the extraction plan.
        round_cost_per_token: One document token's KV read per round, in
            fed tokens, from the extraction plan.
        assumptions: Expected answer length, separate from execution limits.
    """
    assumptions = ExtractAssumptions() if assumptions is None else assumptions
    prefix = head_tokens + length
    if draft_policy == "adaptive":
        # the policy's first-round choice for this document, with one stream
        policy = DraftPolicy(draft_tokens, max_answer_tokens, assumptions.answer_tokens)
        draft_tokens = policy.draft_length(
            round_cost_per_token * (prefix + tail_tokens), 1, num_streams=1)
    if resident:
        work = ask(prefix, tail_tokens, window=window)
    elif shared:
        # a borrowed prefix includes the prompt head the documents share
        borrowed = shared + head_tokens
        work = ask(borrowed, prefix + tail_tokens - borrowed, window=window)
    else:
        work = scan(prefix, tail_tokens, window=window)
    context = prefix + tail_tokens
    fed = float(tail_tokens)
    rows = 1.0
    # each round appends the chosen token and the draft, and reads
    # every row; the last row read decides the closing quote
    round_tokens = 1 + draft_tokens
    rounds = assumptions.rounds(max_answer_tokens, draft_tokens)
    whole = int(rounds)
    work += decode(context, repeat(round_tokens, whole), window=window)
    if rounds > whole:
        work += ask(context + whole * round_tokens, round_tokens,
                    window=window) * (rounds - whole)
    fed += rounds * round_tokens
    rows += rounds * round_tokens
    return work, fed, rows


def estimate(expected_rows: float, lengths, head_tokens: int, tail_tokens: int, *,
             chunk: int, model, device, resident: bool = False,
             shared=None, max_answer_tokens: int, draft_tokens: int,
             draft_policy: str = "adaptive", round_cost_per_token: float = 0.0,
             gpus: int = 1,
             assumptions: ExtractAssumptions | None = None) -> ExtractCost:
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

    Returns:
        Estimated execution time, work, and token counts.
    """
    n = len(lengths)
    if expected_rows <= 0 or n == 0:
        return ExtractCost(0.0, 0.0, Work(), 0.0, 0.0)
    assumptions = ExtractAssumptions() if assumptions is None else assumptions
    work = Work()
    fed = rows = 0.0
    for index, length in enumerate(lengths):
        own, own_fed, own_rows = document_work(
            int(length), head_tokens, tail_tokens,
            window=model.sliding_window, resident=resident,
            shared=0 if shared is None else int(shared[index]),
            max_answer_tokens=max_answer_tokens, draft_tokens=draft_tokens,
            draft_policy=draft_policy, round_cost_per_token=round_cost_per_token,
            assumptions=assumptions)
        work += own
        fed += own_fed
        rows += own_rows
    scale = expected_rows / n
    total = work * scale
    passes = total.tokens / chunk
    copies = parallel_copies(gpus, min(expected_rows, n))
    seconds = decoder_seconds(total * (1 / copies), rows * scale / copies,
                              passes / copies, model, device)
    return ExtractCost(seconds, passes, total, fed * scale, rows * scale)
