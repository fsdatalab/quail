"""Extraction costs: the executor's steps priced like a classification."""

import numpy as np
import pyarrow as pa

from quail.cost.extract import (
    CUE_TOKENS,
    FED_PER_START,
    LINE_NUMBER_TOKENS,
    LOCATE_MIN_LINES,
    MAX_STARTS,
    RANGE_TOKENS,
    START_STEP_TOKENS,
    document_work,
    estimate,
    line_counts,
)
from quail.cost.work import Work, ask, scan, stream
from quail.specs import H100_SXM, QWEN3_4B_FP8

HEAD, TAIL, NUMBERED_HEAD, NUMBERED_TAIL = 2, 40, 5, 47
CHUNK = 10000


def _cost(lengths, lines, **kwargs):
    options = dict(chunk=CHUNK, model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    live = options.pop("live", len(lengths))
    return estimate(live, lengths, lines, HEAD, TAIL, NUMBERED_HEAD,
                    NUMBERED_TAIL, **options)


def test_a_plain_document_scans_once_then_feeds_the_step_and_the_starts():
    work, fed, rows = document_work(100, 1, HEAD, TAIL, NUMBERED_HEAD,
                                    NUMBERED_TAIL)
    context = HEAD + 100 + TAIL
    expected = (scan(HEAD + 100, TAIL)
                + stream(context, [1] * START_STEP_TOKENS)
                + stream(context, [FED_PER_START] * MAX_STARTS))
    assert work == expected
    assert fed == TAIL + START_STEP_TOKENS + FED_PER_START * MAX_STARTS
    assert rows == 1 + START_STEP_TOKENS + FED_PER_START * MAX_STARTS
    # resident KV turns the scan into a question over it
    resident, _, _ = document_work(100, 1, HEAD, TAIL, NUMBERED_HEAD,
                                   NUMBERED_TAIL, resident=True)
    assert resident == (ask(HEAD + 100, TAIL)
                        + stream(context, [1] * START_STEP_TOKENS)
                        + stream(context, [FED_PER_START] * MAX_STARTS))
    assert resident.tokens < work.tokens


def test_a_numbered_document_answers_its_lines_then_the_cue():
    lines = 9
    work, fed, rows = document_work(100, lines, HEAD, TAIL, NUMBERED_HEAD,
                                    NUMBERED_TAIL)
    prefix = NUMBERED_HEAD + 100 + LINE_NUMBER_TOKENS * lines
    expected = scan(prefix, NUMBERED_TAIL)
    context = prefix + NUMBERED_TAIL
    for _ in range(RANGE_TOKENS):
        expected += ask(context, 1)
        context += 1
    expected += ask(context, CUE_TOKENS)
    context += CUE_TOKENS
    expected += stream(context, [1] * START_STEP_TOKENS)
    expected += stream(context, [FED_PER_START] * MAX_STARTS)
    assert work == expected
    assert fed == (NUMBERED_TAIL + RANGE_TOKENS + CUE_TOKENS + START_STEP_TOKENS
                   + FED_PER_START * MAX_STARTS)
    assert rows == (1 + RANGE_TOKENS + 1 + START_STEP_TOKENS
                    + FED_PER_START * MAX_STARTS)
    # a numbered document is never resident: its tokens are not the table's
    assert document_work(100, lines, HEAD, TAIL, NUMBERED_HEAD, NUMBERED_TAIL,
                         resident=True)[0] == work
    # the line count decides the layout
    plain = document_work(100, LOCATE_MIN_LINES - 1, HEAD, TAIL,
                          NUMBERED_HEAD, NUMBERED_TAIL)[0]
    assert plain.tokens < work.tokens


def test_estimate_sums_documents_scales_to_live_and_prices_passes():
    lengths = [100, 300, 50]
    lines = [1, 9, 2]
    cost = _cost(lengths, lines)
    expected = Work()
    fed = rows = 0.0
    for length, count in zip(lengths, lines):
        work, own_fed, own_rows = document_work(
            length, count, HEAD, TAIL, NUMBERED_HEAD, NUMBERED_TAIL)
        expected += work
        fed += own_fed
        rows += own_rows
    assert cost.work == expected
    assert (cost.suffix_tokens, cost.rows) == (fed, rows)
    assert cost.passes == expected.tokens / CHUNK
    assert cost.seconds > 0
    # half the documents live halves the work; none costs nothing
    half = _cost(lengths, lines, live=1.5)
    assert half.work.tokens == expected.tokens / 2
    assert half.seconds < cost.seconds
    assert _cost(lengths, lines, live=0).seconds == 0
    assert _cost([], []).work == Work()
    # unknown line counts price every document as numbered
    assert _cost(lengths, None).work == _cost(lengths, [LOCATE_MIN_LINES] * 3).work
    # a smaller chunk reads the weights more often: more time, same work
    small = _cost(lengths, lines, chunk=100)
    assert small.work == cost.work and small.seconds > cost.seconds


def test_line_counts_read_the_texts_line_breaks():
    counts = line_counts(pa.chunked_array([pa.array(["a\nb\nc", "one", "x\n\ny"])]))
    assert isinstance(counts, np.ndarray)
    assert counts.tolist() == [3, 1, 3]
