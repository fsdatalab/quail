"""AI.EXTRACT under set attention: pointer reads packed in passes."""

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.parts import gpu_seconds, input_staging
from quail.backends.quail.executor.readout import AsyncSlotPointers
from quail.backends.quail.executor.set_passes import (
    group_passes,
    spec_questions,
    state_tails,
)
from quail.backends.quail.executor.stages import Stage, run_stages
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


def calibrated(knots, value: float) -> float:
    """A value through increasing (raw, calibrated) knots.

    Linear between knots and flat past the ends; no knots keep the value.
    """
    if not knots:
        return value
    index = bisect_left([raw for raw, _ in knots], value)
    if index == 0:
        return knots[0][1]
    if index == len(knots):
        return knots[-1][1]
    (a, low), (b, high) = knots[index - 1], knots[index]
    return low + (value - a) * (high - low) / (b - a)


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


def read_passes(state, specs, rows, documents, texts) -> list[SetReads]:
    """Read each extraction's span from every row, its questions packed in passes.

    Each pass reads a document and the state tail, then up to the
    model's set size of read questions. Each question's start and end
    slots point at the document's rows; the first row of the pass
    stands for nothing.

    Args:
        state: The query's execution state.
        specs: Read ExtractSpecs over one table.
        rows: The document ids to read.
        documents: The table's tokenized documents, by document id.
        texts: The table's texts, by document id.
    """
    first = specs[0]
    head, state_tail, questions = spec_questions(specs)
    tails = state_tails(state)
    tokens_of = [[int(token) for token in documents[row]] for row in rows]
    prefixes = [tails.prefix(head + tokens, state_tail) for tokens in tokens_of]
    # the first row, then each document row the prefix kept
    state_rows = [np.array([0, *range(len(head), min(len(head) + len(tokens),
                                                     len(prefix)))])
                  for tokens, prefix in zip(tokens_of, prefixes)]
    model_spec = state.loaded_model.model_spec
    vocabulary = tails.encoder.token_bytes()
    out = [SetReads([None] * len(rows), [None] * len(rows), [0.0] * len(rows))
           for _ in specs]
    torch = state.torch
    for members, suffix, shape in group_passes(
            prefixes, questions, first.question_header_token_ids,
            model_spec.set_size, state.chunk_tokens):
        readout = AsyncSlotPointers(torch, torch.nn.functional, len(members),
                                    model_spec.read_span_tokens)
        stage = Stage(suffixes=[suffix], readout=readout, questions=(shape,),
                      question_slots=first.read_slots,
                      state_reads=lambda anchor: state_rows[anchor],
                      decide=lambda _, row: True,
                      label=f"extract pass ({len(members)} questions)")
        answers, spans, tokens = run_stages(
            torch, state.loaded_model.arena, state.loaded_model.pipeline, [stage],
            prefixes, state.chunk_tokens, staging=input_staging(state),
            gpu_timing=state.gpu_timing, paged=False, label=stage.label)
        for position, answer in answers[0].items():
            reads = np.asarray(answer[0], dtype=float)
            text = texts[rows[position]]
            text = text.as_py() if hasattr(text, "as_py") else str(text)
            for read, member in zip(reads, members):
                start, end, found = int(read[0]), int(read[1]), float(read[2])
                result = out[member]
                result.suffix_tokens += shape[members.index(member)]
                found = calibrated(model_spec.read_found, found)
                result.found[position] = found
                if start == 0 or found < 0.5:
                    continue
                # a read row's document token is one before it: row 0 is nothing
                result.answers[position], result.spans[position] = token_span(
                    text, tokens_of[position], vocabulary, start - 1, end - 1)
        lead = out[members[0]]
        lead.tokens += tokens
        lead.gpu_s += gpu_seconds(spans)
        lead.chunks += len(spans)
    return out


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


def execute_set_extractions(state, nodes, inputs) -> dict:
    """Run a pipeline's extractions under set attention.

    Args:
        state: The query's execution state.
        nodes: The AiExtract nodes, over one table.
        inputs: The documents' ids, and the tokenized documents and texts
            by alias.

    Raises:
        ValueError: The table's text was not loaded.
    """
    alias = nodes[0].spec.alias
    texts = inputs.get("texts", {}).get(alias)
    if texts is None:
        raise ValueError(f"AI.EXTRACT reads the text of {alias!r}, which "
                         f"was not loaded")
    ids = np.asarray(inputs["document_ids"], dtype=np.int64)
    every = read_passes(state, [node.spec for node in nodes], ids,
                        inputs["documents"][alias], texts)
    return {node.node_id: read_result(node, ids, reads)
            for node, reads in zip(nodes, every)}
