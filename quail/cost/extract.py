"""Extraction costs: expected work priced like a classification's work.

The estimate sums each document's work over every document, scales it
to the live count, and prices the total with the component roofline
at ideal packing, as quail/cost/classify.py does. The steps follow the
executor (quail/backends/quail/executor/extract.py): the prompt with
the document, then the answer, one chosen token and DRAFT_TOKENS of
the document per round, every row read. How many tokens the answer
takes is a measured constant, not a function of the document: the
planner does not see the answer. The second and later extractions of
a pipeline find the document resident and are priced without reading
it.
"""

from dataclasses import dataclass

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work, ask, scan, stream
from quail.extraction import DRAFT_TOKENS

# the executor's settings
# the answer's tokens, as a mean: the answers measured were 2.8 words
# on SQuAD and 48 words on CUAD's clause fields
# (/results/extract_spans/20261011T010106Z_qwen3-4b-fp8_squad.json and
# 20261011T005925Z_qwen3-4b-fp8_cuad.json)
ANSWER_TOKENS = 16
# rounds an answer of ANSWER_TOKENS takes when every draft is accepted
ANSWER_ROUNDS = -(-ANSWER_TOKENS // (DRAFT_TOKENS + 1))


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


def document_work(length: int, head_tokens: int, tail_tokens: int, *,
                  window: int = 0, resident: bool = False,
                  shared: int = 0) -> tuple[Work, float, float]:
    """Return one document's work, its fed tokens, and its readout rows.

    Args:
        length: The document's tokens.
        head_tokens: Prompt tokens before the document.
        tail_tokens: Prompt tokens after the document, through the cue.
        window: The model's sliding window, 0 for none.
        resident: Whether the document's KV is already available, as
            it is for the second and later extractions of a pipeline.
        shared: Prefix tokens borrowed from an earlier document.
    """
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
    round_tokens = 1 + DRAFT_TOKENS
    work += stream(context, [round_tokens] * ANSWER_ROUNDS, window=window)
    fed += ANSWER_ROUNDS * round_tokens
    rows += ANSWER_ROUNDS * round_tokens
    return work, fed, rows


def estimate(live: float, lengths, head_tokens: int, tail_tokens: int, *,
             chunk: int, model, device, resident: bool = False,
             shared=None) -> ExtractCost:
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

    Returns:
        Estimated execution time, work, and token counts.
    """
    n = len(lengths)
    if live <= 0 or n == 0:
        return ExtractCost(0.0, 0.0, Work(), 0.0, 0.0)
    work = Work()
    fed = rows = 0.0
    for index, length in enumerate(lengths):
        own, own_fed, own_rows = document_work(
            int(length), head_tokens, tail_tokens,
            window=model.sliding_window, resident=resident,
            shared=0 if shared is None else int(shared[index]))
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
