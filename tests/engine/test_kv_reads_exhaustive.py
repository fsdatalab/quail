"""Every small corpus reads the right KV, on the tightest arena it fits.

A document is one to three whole pages, each all A tokens or all B
tokens, with or without a five-token tail. That covers the ways two
documents can share pages: not at all, part of a page, whole pages, a
whole document, and a duplicate whose length is a whole number of
pages. Every multiset of up to two such documents, up to swapping A
and B, runs through a filter, a two-stage filter, a join, and a
filter feeding a join, on both attention paths, a sliding-window
arena, and a canvas row, with the fewest arena pages and the smallest
chunk budget it fits, where admission blocks and parents wait.
test_kv_reads covers large sizes. See kv_checker for the check.
"""

import itertools

import pytest
from helpers.fakes import cpu_staging
from helpers.kv_checker import Setup, check_feed, check_filter, check_join

pytest.importorskip("torch")


@pytest.fixture(params=[("unified", False, False), ("tree", False, False),
                        ("unified", True, False), ("unified", True, True)],
                ids=["unified", "tree", "sliding", "sliding-canvas"])
def arena_kind(request, monkeypatch):
    """(attention path, sliding layers, canvas row), with CPU staging."""
    cpu_staging(monkeypatch)
    return request.param


def corpora(page_tokens):
    """Every multiset of one or two small documents, up to swapping A and B.

    Swapping the A and B tokens keeps which pages documents share, so
    of two corpora that differ only by the swap, one runs.
    """
    pages = ([1] * page_tokens, [2] * page_tokens)
    shapes = []
    for count in range(1, 4):
        for contents in itertools.product(pages, repeat=count):
            body = [token for page in contents for token in page]
            shapes += [body, body + [3] * 5]
    swap = {1: 2, 2: 1}
    for size in (1, 2):
        for picks in itertools.combinations_with_replacement(
                range(len(shapes)), size):
            documents = [list(shapes[i]) for i in picks]
            swapped = [[swap.get(token, token) for token in document]
                       for document in documents]
            if sorted(documents) <= sorted(swapped):
                yield documents


def tightest(documents, kind, extra, page_tokens=16):
    """The smallest arena and budget the longest document fits.

    The arena holds the longest document, its extra rows, and one page
    of temporary rows; the budget packs the document and its extra rows.
    """
    path, sliding, canvas = kind
    longest = max(map(len, documents)) + extra
    pages = -(-longest // page_tokens) + 1
    return Setup(path=path, window=2 * page_tokens if sliding else None,
                 canvas=(99,) if canvas else (), pages=pages,
                 sliding_pages=pages, budget=longest,
                 page_tokens=page_tokens)


def described(check, documents, setup, *inputs, **options):
    """Run a check, naming the corpus and arena in any failure."""
    try:
        check(documents, *inputs, setup, **options)
    except AssertionError as error:
        starts = [document[::setup.page_tokens] for document in documents]
        raise AssertionError(
            f"documents of {[len(d) for d in documents]} tokens starting "
            f"{starts}, {setup}: {error}") from error


def test_every_small_filter_and_join(arena_kind):
    first, second = [90, 91, 92, 93], [90, 91, 94]
    frame, partners = [95, 96], [[97, 60, 98], [97, 61, 98]]
    for documents in corpora(16):
        # a question row, a canvas row, and the second question's tail
        setup = tightest(documents, arena_kind, len(first) + 2)
        described(check_filter, documents, setup, [first])
        described(check_filter, documents, setup, [first, second])
        # the frame, one partner, and a canvas row after each
        setup = tightest(documents, arena_kind, len(frame) + len(partners[0]) + 2)
        described(check_join, documents, setup, frame, partners)
        described(check_join, documents, setup, frame, partners, retain=True)
        described(check_feed, documents, setup, [90, 91, 92], frame, partners)
