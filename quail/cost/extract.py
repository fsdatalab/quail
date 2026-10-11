"""Extraction costs: expected work priced like a classification's work.

The estimate sums each document's work over every document, scales it
to the live count, and prices the total with the component roofline
at ideal packing, as quail/cost/classify.py does. The steps follow the
executor (quail/backends/quail/executor/extract.py): the prompt with
the document, plain or as numbered lines; the lines answer, one token
per round; the phrase cue; the start step; and the candidate spans fed
after the cue. How many tokens the spans take is a measured constant,
not a function of the document: the planner does not see the answer.
"""

from dataclasses import dataclass

import numpy as np

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work, ask, scan, stream

# the executor's settings
MAX_STARTS = 8          # candidate start positions fed
CHUNK = 16              # document tokens fed per pass from each start
LOCATE_MIN_LINES = 3    # documents with fewer lines are shown plain
# the lines answer "a-b" and its line break, one token each
RANGE_TOKENS = 4
# the phrase cue after the lines answer: a line break, "Phrase", ":",
# and the opening quote
CUE_TOKENS = 5
# the start step, one token per ambiguous start token, at most the
# TOP_TOKENS read at the cue; measured 1.9 to 2.0 rounds per document
START_STEP_TOKENS = 8
# a numbered line's number, colon, and space
LINE_NUMBER_TOKENS = 3
# passes each start takes, as a mean: measured 1.05 on SQuAD, whose
# answers are 2.8 words, and 2.9 on CUAD, whose answers are clauses of
# 48 words (/results/extract_spans/20261011T010106Z_qwen3-4b-fp8_squad.json
# and 20261011T005925Z_qwen3-4b-fp8_cuad.json)
PASSES = 2.0
FED_PER_START = CHUNK * PASSES


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


def _readout(model, rows: float, passes: float) -> CostComponent:
    """Price the full-vocabulary readout of the answer rows."""
    head = 2.0 * model.hidden * model.vocab
    return CostComponent(name="readout", flops=head * rows,
                         bytes_moved=head * passes, precision="bf16")


def document_work(length: int, lines: int, head_tokens: int, tail_tokens: int,
                  numbered_head_tokens: int, numbered_tail_tokens: int, *,
                  window: int = 0, resident: bool = False) -> tuple[Work, float, float]:
    """Return one document's work, its fed tokens, and its readout rows.

    Args:
        length: The document's tokens.
        lines: The document's lines.
        head_tokens: Prompt tokens before a plain document.
        tail_tokens: Prompt tokens after a plain document, through the cue.
        numbered_head_tokens: Prompt tokens before a numbered document.
        numbered_tail_tokens: Prompt tokens after it, through the lines cue.
        window: The model's sliding window, 0 for none.
        resident: Whether a plain document's KV is already available.
    """
    numbered = lines >= LOCATE_MIN_LINES
    fed = rows = 0.0
    if numbered:
        prefix = numbered_head_tokens + length + LINE_NUMBER_TOKENS * lines
        work = scan(prefix, numbered_tail_tokens, window=window)
        context = prefix + numbered_tail_tokens
        rows += 1
        for _ in range(RANGE_TOKENS):
            work += ask(context, 1, window=window)
            context += 1
            rows += 1
        work += ask(context, CUE_TOKENS, window=window)
        context += CUE_TOKENS
        fed += numbered_tail_tokens + RANGE_TOKENS + CUE_TOKENS
    else:
        prefix = head_tokens + length
        work = (ask(prefix, tail_tokens, window=window) if resident
                else scan(prefix, tail_tokens, window=window))
        context = prefix + tail_tokens
        fed += tail_tokens
    rows += 1
    work += stream(context, [1] * START_STEP_TOKENS, window=window)
    fed += START_STEP_TOKENS
    rows += START_STEP_TOKENS
    work += stream(context, [FED_PER_START] * MAX_STARTS, window=window)
    fed += FED_PER_START * MAX_STARTS
    rows += FED_PER_START * MAX_STARTS
    return work, fed, rows


def estimate(live: float, lengths, lines, head_tokens: int, tail_tokens: int,
             numbered_head_tokens: int, numbered_tail_tokens: int, *,
             chunk: int, model, device, resident: bool = False) -> ExtractCost:
    """Estimate the cost of one extraction over a table.

    Args:
        live: Expected number of documents to extract from.
        lengths: Document lengths in tokens.
        lines: Document line counts, or None when unknown: every
            document is then priced as numbered.
        head_tokens: Prompt tokens before a plain document.
        tail_tokens: Prompt tokens after a plain document, through the cue.
        numbered_head_tokens: Prompt tokens before a numbered document.
        numbered_tail_tokens: Prompt tokens after it, through the lines cue.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        resident: Whether plain documents' KV is already available.

    Returns:
        Estimated execution time, work, and token counts.
    """
    n = len(lengths)
    if live <= 0 or n == 0:
        return ExtractCost(0.0, 0.0, Work(), 0.0, 0.0)
    if lines is None:
        lines = [LOCATE_MIN_LINES] * n
    work = Work()
    fed = rows = 0.0
    for length, count in zip(lengths, lines):
        own, own_fed, own_rows = document_work(
            int(length), int(count), head_tokens, tail_tokens,
            numbered_head_tokens, numbered_tail_tokens,
            window=model.sliding_window, resident=resident)
        work += own
        fed += own_fed
        rows += own_rows
    scale = live / n
    total = work * scale
    passes = total.tokens / chunk
    components = dense_decoder_components(total, model, passes=passes)
    components += (_readout(model, rows * scale, passes),)
    seconds = sum(component.seconds
                  for component in component_latencies(components, device))
    return ExtractCost(seconds, passes, total, fed * scale, rows * scale)


def line_counts(texts) -> np.ndarray:
    """Return each text's line count: its line breaks plus one.

    Blank lines count, though the executor does not number them, so a
    document with blank lines may be priced as numbered when it is
    shown plain.
    """
    import pyarrow.compute as pc

    return pc.count_substring(texts, "\n").to_numpy(zero_copy_only=False) + 1
