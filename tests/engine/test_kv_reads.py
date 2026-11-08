"""Filters and joins over random corpora read the right KV.

The corpora repeat one another's starts, as agent snapshots do, and
run on arenas a few documents fill. See kv_checker for the check.
"""

import random

import pytest
from helpers.fakes import cpu_staging
from helpers.kv_checker import (
    Setup,
    check_feed,
    check_filter,
    check_join,
    check_join_after_join,
)

pytest.importorskip("torch")


@pytest.fixture(params=[("tree", False, False), ("unified", True, True)],
                ids=["tree", "sliding-canvas"])
def arena_kind(request, monkeypatch):
    """(attention path, sliding layers, canvas row), with CPU staging."""
    cpu_staging(monkeypatch)
    return request.param


def corpus(rng, n, header_tokens, piece_tokens):
    """Documents that repeat one another's starts, as agent snapshots do.

    Each trace starts with a shared header and grows by pieces; every
    prefix of it is a document. Some documents are exact duplicates,
    and some branch off a trace part way through.
    """
    header = [rng.randrange(50) for _ in range(rng.choice(header_tokens))]
    documents = []
    while len(documents) < n:
        trace = list(header)
        for _ in range(rng.randrange(1, 5)):
            trace = trace + [rng.randrange(50)
                             for _ in range(rng.choice(piece_tokens))]
            documents.append(trace)
            if rng.random() < 0.2:
                documents.append(list(trace))
            if rng.random() < 0.3 and len(trace) > 20:
                cut = rng.randrange(1, len(trace))
                documents.append(trace[:cut] + [50 + rng.randrange(20)])
    documents = documents[:n]
    rng.shuffle(documents)
    return documents


def small(rng, kind, n, pages, sliding_pages):
    """A corpus of short documents and an arena a few of them fill."""
    path, sliding, canvas = kind
    documents = corpus(rng, n, (0, 5, 16, 20), (1, 7, 16, 30, 45))
    return documents, Setup(
        path=path, window=32 if sliding else None,
        canvas=(99,) if canvas else (), pages=rng.choice(pages),
        sliding_pages=rng.choice(sliding_pages),
        budget=max(map(len, documents)) + rng.choice((8, 60, 600)))


def questions(rng):
    return [[90, 91, 92, 93], [90, 91, 94]][:rng.choice((1, 2))]


def join_parts():
    """A join's frame and three partner suffixes."""
    return [95, 96], [[97, 60 + p, 98] for p in range(3)]


def named(case, check, *args, **options):
    """Run a check, naming the case in any failure."""
    try:
        check(*args, **options)
    except AssertionError as error:
        raise AssertionError(f"{case}: {error}") from error


def test_small_filters_joins_and_filters_feeding_joins(arena_kind):
    """Short documents on arenas a few of them fill.

    A join whose child shares its first page with a retained anchor of
    an earlier join runs last: the anchor trimmed its window, so on a
    sliding arena the child cannot borrow the rows the anchor dropped
    and packs whole; elsewhere it borrows the page.
    """
    for seed in range(6):
        rng = random.Random(seed)
        documents, setup = small(rng, arena_kind, 40, (40, 80, 400), (24, 60))
        named(f"filter seed {seed}", check_filter, documents, questions(rng),
              setup)
        for retain in (False, True):
            rng = random.Random(100 + seed)
            documents, setup = small(rng, arena_kind, 30, (60, 400), (40, 80))
            named(f"join seed {seed} retain {retain}", check_join, documents,
                  *join_parts(), setup, retain=retain)
        rng = random.Random(200 + seed)
        documents, setup = small(rng, arena_kind, 40, (80, 400), (60, 120))
        named(f"feed seed {seed}", check_feed, documents, [90, 91, 92],
              *join_parts(), setup)

    path, sliding, canvas = arena_kind
    parent = [1] * 64                  # four pages, two of them the window
    child = [1] * 16 + [3] * 5
    setup = Setup(path=path, window=32 if sliding else None,
                  canvas=(99,) if canvas else (), pages=12,
                  sliding_pages=12, budget=200, page_tokens=16)
    check_join_after_join([parent, child], *join_parts(), setup)


@pytest.mark.parametrize("path,window,canvas,state_slots", [
    ("tree", None, (), 0),
    ("tree", 32, (), 0),
    ("unified", None, (), 0),
    ("unified", 32, (99,), 0),
    ("unified", None, (), 24),
    ("unified", 32, (), 24),
], ids=["tree", "tree-sliding", "unified", "sliding-canvas", "hybrid",
        "hybrid-sliding"])
def test_a_document_longer_than_a_pass_prefills_over_passes_and_answers_alike(
        monkeypatch, path, window, canvas, state_slots):
    """Documents of up to 160 tokens on a 40-token pass, then a roomy one.

    A document longer than two passes writes its prefix over three or
    more, its KV accumulating in its pages; borrowers wait for it, and
    the checking model fails any read of a row not yet written.
    """
    from quail.backends.quail.executor import chunk as chunk_mod

    cpu_staging(monkeypatch)
    pieces = {}
    real_pack = chunk_mod.pack_chunk

    def counting_pack(torch_, arena, specs, **options):
        for spec in specs:
            if spec.get("prefix") is not None:
                pieces[spec["key"]] = pieces.get(spec["key"], 0) + 1
        return real_pack(torch_, arena, specs, **options)

    monkeypatch.setattr(chunk_mod, "pack_chunk", counting_pack)
    rng = random.Random(11)
    documents = corpus(rng, 16, (0, 20, 40), (30, 45, 60))
    assert max(map(len, documents)) > 80
    for budget in (40, 400):
        setup = Setup(path=path, window=window, canvas=canvas, pages=120,
                      sliding_pages=120, budget=budget, state_slots=state_slots)
        pieces.clear()
        named(f"filter budget {budget}", check_filter, documents,
              [[90, 91, 92, 93], [90, 91, 94]], setup)
        assert (max(pieces.values()) >= 3) == (budget == 40)
        named(f"join budget {budget}", check_join, documents, *join_parts(),
              setup, retain=True)
        named(f"feed budget {budget}", check_feed, documents, [90, 91, 92],
              *join_parts(), setup)


def test_an_unpaged_filter_writes_a_long_document_in_pages_over_passes(
        monkeypatch):
    from helpers.kv_checker import Run, answer, chain, passthrough

    from quail.backends.quail.executor.stages import filter_stages, run_stages

    cpu_staging(monkeypatch)
    documents = [list(range(100)), [7] * 20]
    question = [90, 91, 92]
    answers = {chain(document + question): answer(chain(document + question))
               for document in documents}
    run = Run(Setup(path="unified", pages=40, budget=32), answers)
    got, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline,
        filter_stages([question], passthrough()), documents, 32,
        anchor_keys=[("d", 0), ("d", 1)], paged=False)
    assert {d: [int(bit) for bit in row] for d, row in got[0].items()} == {
        d: [answers[chain(document + question)]]
        for d, document in enumerate(documents)}
    assert tokens == 100 + 20 + 2 * len(question)
    assert not run.arena.accounting.owned
