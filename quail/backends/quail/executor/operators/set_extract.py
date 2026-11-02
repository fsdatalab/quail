"""AI.EXTRACT under set attention: pointer reads packed in passes."""

from dataclasses import dataclass, field

import numpy as np

from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import extraction_rows


@dataclass
class SetReads:
    """One extraction's answers from set-attention passes.

    Attributes:
        answers: Per row, the text read, or None for nothing.
        spans: Per row, the text's character span in the document, or None.
        tokens: The fresh tokens of the passes this extraction led.
        gpu_s: Their forward-pass GPU seconds.
        chunks: Their forward passes.
        suffix_tokens: The tokens of this extraction's question, summed
            over the rows it read.
        found: Per row, the calibrated probability that the text is there.
    """

    answers: list = field(default_factory=list)
    spans: list = field(default_factory=list)
    found: list = field(default_factory=list)
    tokens: int = 0
    gpu_s: float = 0.0
    chunks: int = 0
    suffix_tokens: int = 0


def calibrated(knots, value):
    """A value, or an array of them, through increasing (raw, calibrated) knots.

    Linear between knots and flat past the ends; no knots keep the value.
    """
    if not knots:
        return value
    raw, out = zip(*knots)
    result = np.interp(value, raw, out)
    return float(result) if np.ndim(result) == 0 else result


def token_span(text: str, tokens, vocabulary, first: int, last: int):
    """The characters of document tokens first to last, spaces trimmed.

    Args:
        text: The document text the tokens encode.
        tokens: The document's token ids.
        vocabulary: Token id -> its bytes.
        first: The span's first token.
        last: Its last token.

    Returns:
        (text, (start, end)), or (None, None) when only spaces are left.
    """
    sizes = [len(vocabulary.get(int(token), b"")) for token in tokens[:last + 1]]
    data = text.encode("utf-8")
    begin, end = sum(sizes[:first]), sum(sizes)
    start = len(data[:begin].decode("utf-8", "ignore"))
    stop = len(data[:end].decode("utf-8", "ignore"))
    while start < stop and text[start].isspace():
        start += 1
    while stop > start and text[stop - 1].isspace():
        stop -= 1
    if start == stop:
        return None, None
    return text[start:stop], (start, stop)


def found_reads(spans, knots):
    """Each read's calibrated P(found), and whether its span is returned.

    Args:
        spans: (reads, 3) first row, last row, and raw P(found).
        knots: The model's (raw, calibrated) P(found) knots.
    """
    found = calibrated(knots, spans[:, 2])
    return found, (spans[:, 0] > 0) & (found >= 0.5)


def read_texts(spans, keep, documents, texts, head, vocabulary):
    """The text and character span of each kept read, None elsewhere.

    Args:
        spans: (reads, 3) first row, last row, and raw P(found), a row
            counted as the document's token plus one.
        keep: (reads,) whether each read returns its span.
        documents: Per read, its prompt head and document tokens.
        texts: Per read, its document text.
        head: The prompt head's tokens.
        vocabulary: Token id -> its bytes.
    """
    answers, characters = [None] * len(spans), [None] * len(spans)
    for index in np.flatnonzero(keep):
        text = texts[index]
        text = text.as_py() if hasattr(text, "as_py") else str(text)
        tokens = [int(token) for token in documents[index]][head:]
        # a read row's document token is one before it: row 0 is nothing
        answers[index], characters[index] = token_span(
            text, tokens, vocabulary, int(spans[index, 0]) - 1,
            int(spans[index, 1]) - 1)
    return answers, characters


def read_result(node, ids, reads: SetReads) -> NodeResult:
    """An extraction's NodeResult: one answer and span per row."""
    spec = node.spec
    answers = extraction_rows(spec.alias, spec.name, spec.span_name,
                              np.asarray(ids), reads.answers, reads.spans)
    return NodeResult({node.rows_port: answers}, NodeMetrics(
        input_rows=len(ids), output_rows=len(ids), evaluated_documents=len(ids),
        fresh_tokens=reads.tokens, gpu_s=reads.gpu_s, chunks=reads.chunks,
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(ids), "suffix_tokens": reads.suffix_tokens}))


