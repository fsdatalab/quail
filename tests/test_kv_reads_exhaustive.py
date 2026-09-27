"""Every small corpus reads the right KV, on the tightest arenas it fits.

A document is one to three whole pages, each all A tokens or all B
tokens, with or without a five-token tail. That covers the ways two
documents can share pages: not at all, part of a page, whole pages, a
whole document, and a duplicate whose length is a whole number of
pages. Every multiset of up to two such documents runs through a
filter, a two-stage filter, a join, and a filter feeding a join, on
both attention paths, a sliding-window arena, and a canvas row, with
the fewest arena pages and the smallest chunk budget each case fits,
where admission blocks and parents wait. KV_EXHAUSTIVE=1 adds every
three-document corpus and roomier arenas and budgets; it takes about
twenty minutes. See kv_checker for the check.
"""

import itertools
import os

import pytest
from fakes import cpu_staging
from kv_checker import CANVAS, PAGE, check_feed, check_filter, check_join

pytest.importorskip("torch")

FULL = bool(os.environ.get("KV_EXHAUSTIVE"))
PAGES = ([1] * PAGE, [2] * PAGE)
TAIL = [3] * 5
QUESTIONS = [[90, 91, 92, 93], [90, 91, 94]]
FRAME = [95, 96]
PARTNERS = [[97, 60, 98], [97, 61, 98]]
PATHS = [("unified", False, False), ("tree", False, False),
         ("unified", True, False), ("unified", True, True)]


def documents():
    for pages in range(1, 4):
        for contents in itertools.product(PAGES, repeat=pages):
            body = [token for page in contents for token in page]
            yield body
            yield body + TAIL


def corpora():
    shapes = list(documents())
    sizes = (1, 2, 3) if FULL else (1, 2)
    for size in sizes:
        for picks in itertools.combinations_with_replacement(
                range(len(shapes)), size):
            yield [list(shapes[i]) for i in picks]


def arenas(docs, extra):
    """(every-token pages, sliding pages, budget) from tightest up.

    The tightest arena holds the longest document, its extra rows,
    and one page of temporary rows; the tightest budget packs the
    longest document and its extra rows.
    """
    longest = max(map(len, docs)) + extra
    pages = -(-longest // PAGE) + 1
    budget = longest
    yield pages, pages, budget
    if FULL:
        yield pages + 2, pages + 1, 2 * budget
        yield 400, 400, 600


def run(check, docs, extra, path, sliding, canvas, *args):
    tried = 0
    for pages, sliding_pages, budget in arenas(docs, extra):
        try:
            check(docs, *args, path=path, sliding=sliding, canvas=canvas,
                  pages=pages, sliding_pages=sliding_pages, budget=budget)
        except ValueError as error:
            # an arena or budget too small for one document is refused
            # up front; anything else is a failure
            if not any(text in str(error) for text in (
                    "more pages than", "exceed", "chunk budget",
                    "no room")):
                raise AssertionError(f"{docs}: {error}") from error
            continue
        except AssertionError as error:
            lengths = [len(doc) for doc in docs]
            raise AssertionError(
                f"documents {lengths}, first tokens "
                f"{[doc[0::PAGE] for doc in docs]}, pages {pages}, "
                f"sliding {sliding_pages}, budget {budget}: {error}") from error
        tried += 1
    return tried


@pytest.mark.parametrize("path,sliding,canvas", PATHS)
def test_every_small_filter(monkeypatch, path, sliding, canvas):
    cpu_staging(monkeypatch)
    extra = len(QUESTIONS[0]) + len(CANVAS) + 1
    tried = 0
    for docs in corpora():
        tried += run(check_filter, docs, extra, path, sliding, canvas,
                     QUESTIONS[:1])
        tried += run(check_filter, docs, extra, path, sliding, canvas,
                     QUESTIONS)
    assert tried


@pytest.mark.parametrize("path,sliding,canvas", PATHS)
def test_every_small_join(monkeypatch, path, sliding, canvas):
    cpu_staging(monkeypatch)
    extra = len(FRAME) + len(PARTNERS[0]) + 2 * len(CANVAS)
    tried = 0
    for docs in corpora():
        tried += run(check_join, docs, extra, path, sliding, canvas,
                     FRAME, PARTNERS)
        tried += run(check_feed, docs, extra, path, sliding, canvas,
                     QUESTIONS[0][:3], FRAME, PARTNERS)
    assert tried
