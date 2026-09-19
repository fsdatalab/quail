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
