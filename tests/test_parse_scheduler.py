from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from quail.backends.quail.executor.parse import PageParser

STOP = 0


class _Parser(PageParser):
    """PageParser's loop with fake prefill and decode steps.

    Page i emits i % 5 + 1 tokens, the last of them STOP. The fakes
    check that no row holds two pages at once.
    """

    def __init__(self, rows, prefill_tokens):
        self.torch = SimpleNamespace(inference_mode=nullcontext)
        self.rows = rows
        self.prefill_tokens = prefill_tokens
        self.max_output_tokens = 100
        self.context_tokens = 1000
        self.stops = frozenset({STOP})
        self.admit_pages = 1
        self.holding = {}
        self.most_in_flight = 0

    def _emit(self, key, made):
        return STOP if made + 1 >= key % 5 + 1 else 1

    def _prefill(self, admitted, rows):
        assert len(set(rows)) == len(rows)
        for (key, page), row in zip(admitted, rows):
            assert row not in self.holding, f"row {row} admitted twice"
            assert 0 <= row < self.rows
            self.holding[row] = key
            page["offset"] = 0
        self.most_in_flight = max(self.most_in_flight, len(self.holding))
        return [self._release(row, key, 0) for (key, _), row in zip(admitted, rows)]

    def _decode(self, states):
        return [self._release(s.row, s.key, len(s.tokens)) for s in states]

    def _release(self, row, key, made):
        token = self._emit(key, made)
        if token == STOP:
            del self.holding[row]
        return token


def _pages(count, tokens=10):
    return [(i, dict(input_ids=np.zeros(tokens, dtype=np.int32)))
            for i in range(count)]


def test_more_pages_than_rows_never_share_a_row():
    parser = _Parser(rows=3, prefill_tokens=1000)
    done = list(parser.parse(_pages(20)))
    assert sorted(key for key, _, _ in done) == list(range(20))
    assert parser.most_in_flight == 3
    for key, tokens, prompt_tokens in done:
        assert len(tokens) == key % 5 + 1 and tokens[-1] == STOP
        assert prompt_tokens == 10


def test_a_step_prefills_no_more_than_its_budget_past_the_first_page():
    parser = _Parser(rows=8, prefill_tokens=25)
    admitted = []
    original = parser._prefill

    def recording(pages, rows):
        admitted.append(len(pages))
        return original(pages, rows)

    parser._prefill = recording
    list(parser.parse(_pages(6)))
    assert max(admitted) == 2


def test_a_step_waits_for_enough_free_rows_while_pages_decode():
    parser = _Parser(rows=6, prefill_tokens=1000)
    parser.admit_pages = 3
    admitted = []
    original = parser._prefill

    def recording(pages, rows):
        admitted.append(len(pages))
        return original(pages, rows)

    parser._prefill = recording
    done = list(parser.parse(_pages(30)))
    assert len(done) == 30
    # the first prefill fills every row; later ones wait for three
    # free rows unless the input runs out or nothing is decoding
    assert admitted[0] == 6
    assert min(admitted[1:-1]) >= 3
