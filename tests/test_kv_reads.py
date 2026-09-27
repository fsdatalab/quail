"""Filters and joins over random corpora read the right KV.

The corpora repeat one another's starts, as agent snapshots do, and
run on both attention paths, a sliding-window arena, and a canvas row;
see kv_checker for the check. KV_SEEDS=400 runs more corpora per case.
"""

import os
import random

import pytest
from fakes import cpu_staging
from kv_checker import check_feed, check_filter, check_join

pytest.importorskip("torch")


def corpus(rng, n):
    """Documents that repeat one another's starts, as agent snapshots do."""
    header = [rng.randrange(50) for _ in range(rng.choice([0, 5, 16, 20]))]
    docs = []
    while len(docs) < n:
        trace = list(header)
        for _ in range(rng.randrange(1, 5)):
            trace = trace + [rng.randrange(50)
                             for _ in range(rng.choice([1, 7, 16, 30, 45]))]
            docs.append(trace)
            if rng.random() < 0.2:
                docs.append(list(trace))          # an exact duplicate
            if rng.random() < 0.3 and len(trace) > 20:
                cut = rng.randrange(1, len(trace))
                docs.append(trace[:cut] + [50 + rng.randrange(20)])
    docs = docs[:n]
    rng.shuffle(docs)
    return docs


SEEDS = int(os.environ.get("KV_SEEDS", 6))
CASES = [(seed, path, sliding, canvas)
         for seed in range(SEEDS)
         for path, sliding, canvas in (("unified", False, False),
                                       ("tree", False, False),
                                       ("unified", True, False),
                                       ("unified", True, True))]
FRAME = [95, 96]
PARTNERS = [[97, 60 + p, 98] for p in range(3)]


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_filter_reads_the_right_kv(monkeypatch, seed, path, sliding, canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(seed)
    docs = corpus(rng, 40)
    check_filter(docs, [[90, 91, 92, 93], [90, 91, 94]][:rng.choice([1, 2])],
                 path=path, sliding=sliding, canvas=canvas,
                 pages=rng.choice([40, 80, 400]),
                 sliding_pages=rng.choice([24, 60]),
                 budget=max(map(len, docs)) + rng.choice([8, 60, 600]))


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_join_reads_the_right_kv(monkeypatch, seed, path, sliding, canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(100 + seed)
    anchors = corpus(rng, 30)
    check_join(anchors, FRAME, PARTNERS, path=path, sliding=sliding,
               canvas=canvas, pages=rng.choice([60, 400]),
               sliding_pages=rng.choice([40, 80]),
               budget=max(map(len, anchors)) + rng.choice([8, 60, 600]))


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_filter_survivors_feed_a_join(monkeypatch, seed, path, sliding,
                                      canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(200 + seed)
    docs = corpus(rng, 40)
    check_feed(docs, [90, 91, 92], FRAME, PARTNERS, path=path,
               sliding=sliding, canvas=canvas, pages=rng.choice([80, 400]),
               sliding_pages=rng.choice([60, 120]),
               budget=max(map(len, docs)) + rng.choice([8, 60, 600]))
