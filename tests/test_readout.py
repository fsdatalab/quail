"""Readouts carry their answer type; joins pick their attention path."""

from types import SimpleNamespace

import numpy as np
import pytest

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.attention import (
    FILTER_ATTENTION,
    JOIN_ATTENTION,
    )
from quail.backends.quail.executor.readout import AsyncAnswers, AsyncScores


def test_readouts_declare_their_answer_type():
    assert AsyncAnswers.dtype is None
    assert AsyncScores.dtype is np.float32


def test_run_join_packs_every_chunk_for_its_path(monkeypatch):
    from fakes import cpu_arena, fake_pipeline, fake_torch

    modes = []

    def forward(chunk):
        modes.append(chunk.attention_mode)
        return [1] * len(chunk.specs)

    for expected in (JOIN_ATTENTION, FILTER_ATTENTION):
        modes.clear()
        pipeline = fake_pipeline(join_attention=expected, forward_chunk=forward)
        monkeypatch.setattr(loop, "pack_chunk", lambda torch, arena, specs, **kw:
                            SimpleNamespace(specs=specs, tokens=len(specs),
                                            attention_mode=kw["attention_mode"],
                                            temporary_keys=(),
                                            fresh_keys=()))
        answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                  dtype=None)
        loop.run_join(fake_torch(), cpu_arena(64), pipeline, answers,
                      [[1] * 8, [2] * 8], [[[3, 4]]], 64,
                      anchor_keys=[("a", 0), ("a", 1)])
        assert modes and all(mode == expected for mode in modes)


class _Images:
    """A PageImages stand-in: counts takes and opens per anchor."""

    def __init__(self):
        self.opened = None
        self.taken = []
        self.closed = False

    def open(self, chunk_tokens=None):
        self.opened = chunk_tokens

    def take(self, doc):
        self.taken.append(doc)
        return (("block", f"page-{doc}"),)

    def close(self):
        self.closed = True


def test_run_join_packs_an_anchor_page_with_its_prefix_once(monkeypatch):
    from fakes import cpu_arena, fake_pipeline, fake_torch

    packed = []
    failed = []

    def pack(torch, arena, specs, **kw):
        if len(specs) > 1 and not failed:
            # the split retries the same anchors: pages are not re-taken
            failed.append(len(specs))
            raise loop.ArenaFullError("unified suffix pages exceed the arena")
        packed.append([(s["key"], s["prefix"] is not None, s.get("images"))
                       for s in specs])
        return SimpleNamespace(specs=specs, tokens=len(specs),
                               attention_mode=kw["attention_mode"],
                               temporary_keys=(), fresh_keys=())

    monkeypatch.setattr(loop, "pack_chunk", pack)
    arena = cpu_arena(64)
    monkeypatch.setattr(arena, "evict_retained", lambda need: ())
    pipeline = fake_pipeline(
        join_attention=FILTER_ATTENTION,
        forward_chunk=lambda chunk: [1] * sum(len(s["suffixes"])
                                              for s in chunk.specs))
    pipeline.takes_images = True
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    images = _Images()
    out, _, _ = loop.run_join(fake_torch(), arena, pipeline, answers,
                              [[1] * 8, [2] * 8], [[[3, 4], [5, 6]]], 64,
                              anchor_keys=[("p", 0), ("p", 1)], images=images)
    assert images.opened == 64 and not images.closed   # the caller closes
    assert images.taken == [0, 1]
    carried = [(key, imgs) for chunk in packed for key, fresh, imgs in chunk
               if fresh]
    assert carried == [(("p", 0), (("block", "page-0"),)),
                       (("p", 1), (("block", "page-1"),))]
    assert all(imgs is None for chunk in packed
               for _, fresh, imgs in chunk if not fresh)
    assert out == [{0: [1, 1], 1: [1, 1]}]


def test_run_join_checks_its_image_source_fits_the_path():
    from fakes import cpu_arena, fake_pipeline, fake_torch

    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    text_pipeline = fake_pipeline(join_attention=FILTER_ATTENTION)
    text_pipeline.takes_images = False
    with pytest.raises(ValueError, match="does not embed images"):
        loop.run_join(fake_torch(), cpu_arena(64), text_pipeline, answers,
                      [[1] * 8], [[[3]]], 64, images=_Images())
    merge_pipeline = fake_pipeline(join_attention=JOIN_ATTENTION)
    merge_pipeline.takes_images = True
    with pytest.raises(ValueError, match="paged unified path"):
        loop.run_join(fake_torch(), cpu_arena(64), merge_pipeline, answers,
                      [[1] * 8], [[[3]]], 64, images=_Images())
    with pytest.raises(ValueError, match="streamed anchor"):
        loop.run_join(fake_torch(), cpu_arena(64), merge_pipeline, answers,
                      [], [[[3]]], 64, anchor_keys=[],
                      anchor_source=SimpleNamespace(done=True, chunks=0),
                      images=_Images())


def test_run_join_evicts_then_halves_a_chunk_that_does_not_fit(monkeypatch):
    from fakes import cpu_arena, fake_pipeline, fake_torch

    sizes = []
    failed = []

    def pack(torch, arena, specs, **kw):
        if len(specs) > 1 and not failed:
            failed.append(len(specs))
            raise loop.ArenaFullError("unified suffix pages exceed the free KV arena")
        sizes.append(len(specs))
        return SimpleNamespace(specs=specs, tokens=len(specs),
                               attention_mode=kw["attention_mode"],
                               temporary_keys=(), fresh_keys=())

    monkeypatch.setattr(loop, "pack_chunk", pack)
    arena = cpu_arena(64)
    evictions = []
    monkeypatch.setattr(arena, "evict_retained",
                        lambda need: evictions.append(need) or ())
    pipeline = fake_pipeline(forward_chunk=lambda chunk: [1] * len(chunk.specs))
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    answers_out, _, _ = loop.run_join(fake_torch(), arena, pipeline, answers,
                                 [[1] * 8, [2] * 8], [[[3, 4]]], 64,
                                 anchor_keys=[("a", 0), ("a", 1)])
    # nothing retained to evict, so the two-group chunk ran as two chunks
    assert failed == [2] and len(evictions) == 1 and evictions[0] > 0
    assert sizes == [1, 1]
    assert answers_out == [{0: [1], 1: [1]}]
