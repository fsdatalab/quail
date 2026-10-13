"""Extraction costs: expected work priced like a classification's work.

The estimate sums each document's work over every document, scales it
to the live count, and prices the total with the component roofline
at ideal packing, as quail/cost/classify.py does. The steps follow the
executor (quail/backends/quail/executor/extract/): the prompt with
the document, then the answer, one chosen token and a configured draft of
the document per round, every row read. The expected answer length is a
cost assumption; the planner does not see the answer. The second and later
extractions of a pipeline reuse the document's KV. Each answer round reads
the document, question, and previously accepted answer tokens from KV.
"""

from dataclasses import dataclass
from itertools import repeat

from quail.cost.dense_decoder_cost import decoder_seconds
from quail.cost.work import Work, ask, decode, scan


@dataclass(frozen=True)
class ExtractAssumptions:
    """Expected answer length used only for extraction cost estimates.

    The default assumes 16 tokens and fully accepted drafts. Actual answer
    lengths and rejected drafts are not available during planning.
    """

    answer_tokens: int = 16

    def __post_init__(self):
        if type(self.answer_tokens) is not int or self.answer_tokens < 0:
            raise ValueError("answer_tokens must be a nonnegative integer")

    def rounds(self, max_answer_tokens: int, draft_tokens: int) -> int:
        """Return expected rounds, limited by the configured answer length."""
        tokens = min(self.answer_tokens, max_answer_tokens)
        if tokens == max_answer_tokens:
            # The first token comes from the cue; the cap needs no stop read.
            tokens -= 1
        return -(-tokens // (1 + draft_tokens))


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
        draft_tokens: Source tokens proposed per round from the extraction plan.
        assumptions: Expected answer length, separate from execution limits.
    """
    assumptions = ExtractAssumptions() if assumptions is None else assumptions
    prefix = head_tokens + length
    if resident:
        work = ask(prefix, tail_tokens, window=window)
    elif shared:
        work = ask(shared, prefix + tail_tokens - shared, window=window)
    else:
        work = scan(prefix, tail_tokens, window=window)
    context = prefix + tail_tokens
    fed = float(tail_tokens)
    rows = 1.0
    # each round appends the chosen token and the draft, and reads
    # every row; the last row read decides the closing quote
    round_tokens = 1 + draft_tokens
    rounds = assumptions.rounds(max_answer_tokens, draft_tokens)
    work += decode(context, repeat(round_tokens, rounds), window=window)
    fed += rounds * round_tokens
    rows += rounds * round_tokens
    return work, fed, rows


def estimate(live: float, lengths, head_tokens: int, tail_tokens: int, *,
             chunk: int, model, device, resident: bool = False,
             shared=None, max_answer_tokens: int, draft_tokens: int,
             assumptions: ExtractAssumptions | None = None) -> ExtractCost:
    """Estimate the cost of one extraction over a table.

    Args:
        live: Expected number of documents to extract from.
        lengths: Document lengths in tokens.
        head_tokens: Prompt tokens before the document.
        tail_tokens: Prompt tokens after the document, through the cue.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether the documents' KV is already available.
        shared: Per-document borrowed prefix lengths, or None.
        max_answer_tokens: Answer limit from the extraction plan.
        draft_tokens: Source tokens proposed per round from the extraction plan.
        assumptions: Expected answer length, separate from execution limits.

    Returns:
        Estimated execution time, work, and token counts.
    """
    n = len(lengths)
    if live <= 0 or n == 0:
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
            assumptions=assumptions)
        work += own
        fed += own_fed
        rows += own_rows
    scale = live / n
    total = work * scale
    passes = total.tokens / chunk
    seconds = decoder_seconds(total, rows * scale, passes, model, device)
    return ExtractCost(seconds, passes, total, fed * scale, rows * scale)
