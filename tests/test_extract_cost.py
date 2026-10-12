"""Extraction costs: the executor's rounds priced like a classification."""

from quail.cost.extract import (
    ANSWER_ROUNDS,
    DRAFT_TOKENS,
    document_work,
    estimate,
)
from quail.cost.work import Work, ask, scan, stream
from quail.specs import H100_SXM, QWEN3_4B_FP8

HEAD, TAIL = 2, 40
CHUNK = 10000


def _cost(lengths, **kwargs):
    options = dict(chunk=CHUNK, model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    live = options.pop("live", len(lengths))
    return estimate(live, lengths, HEAD, TAIL, **options)


def test_a_document_scans_once_then_writes_its_answer_in_rounds():
    work, fed, rows = document_work(100, HEAD, TAIL)
    context = HEAD + 100 + TAIL
    rounds = [1 + DRAFT_TOKENS] * ANSWER_ROUNDS
    expected = scan(HEAD + 100, TAIL) + stream(context, rounds)
    assert work == expected
    answer = ANSWER_ROUNDS * (1 + DRAFT_TOKENS)
    assert fed == TAIL + answer
    assert rows == 1 + answer
    # resident KV turns the scan into a question over it
    resident, _, _ = document_work(100, HEAD, TAIL, resident=True)
    assert resident == ask(HEAD + 100, TAIL) + stream(context, rounds)
    assert resident.tokens < work.tokens


def test_estimate_sums_documents_scales_to_live_and_prices_passes():
    lengths = [100, 300, 50]
    cost = _cost(lengths)
    expected = Work()
    fed = rows = 0.0
    for length in lengths:
        work, own_fed, own_rows = document_work(length, HEAD, TAIL)
        expected += work
        fed += own_fed
        rows += own_rows
    assert cost.work == expected
    assert (cost.suffix_tokens, cost.rows) == (fed, rows)
    assert cost.passes == expected.tokens / CHUNK
    assert cost.seconds > 0
    # half the documents live halves the work; none costs nothing
    half = _cost(lengths, live=1.5)
    assert half.work.tokens == expected.tokens / 2
    assert half.rows == rows / 2
    assert _cost(lengths, live=0).seconds == 0
    # a resident pipeline costs less than reading the documents
    assert _cost(lengths, resident=True).work.tokens < cost.work.tokens
    # a smaller chunk means more passes and more time
    small = _cost(lengths, chunk=100)
    assert small.passes > cost.passes and small.seconds > cost.seconds
