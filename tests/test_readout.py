"""Readouts carry their answer type; joins pick their attention path."""

from types import SimpleNamespace

import numpy as np

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
    result, _, _ = loop.run_join(fake_torch(), arena, pipeline, answers,
                                 [[1] * 8, [2] * 8], [[[3, 4]]], 64,
                                 anchor_keys=[("a", 0), ("a", 1)])
    # nothing retained to evict, so the two-group chunk ran as two chunks
    assert failed == [2] and len(evictions) == 1 and evictions[0] > 0
    assert sizes == [1, 1]
    assert len(result) == 2
