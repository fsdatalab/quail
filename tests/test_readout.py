"""Readouts carry their answer type; joins pick their attention path."""

from types import SimpleNamespace

import numpy as np

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.attention import (
    FILTER_ATTENTION,
    JOIN_ATTENTION,
    join_attention_mode,
)
from quail.backends.quail.executor.readout import AsyncAnswers, AsyncScores


def test_readouts_declare_their_answer_type():
    assert AsyncAnswers.dtype is None
    assert AsyncScores.dtype is np.float32


def test_join_path_follows_weight_precision():
    assert join_attention_mode(True) == JOIN_ATTENTION
    assert join_attention_mode(False) == FILTER_ATTENTION


def test_run_join_states_its_path_before_each_chunk(monkeypatch):
    from fakes import cpu_arena, fake_torch

    modes = []

    def forward(chunk):
        modes.append(pipeline.attention_mode)
        return [1] * len(chunk.specs)

    for fp8, expected in ((True, JOIN_ATTENTION), (False, FILTER_ATTENTION)):
        modes.clear()
        pipeline = SimpleNamespace(attention_mode="unified", is_fp8=fp8,
                                   forward_chunk=forward)
        monkeypatch.setattr(loop, "pack_chunk", lambda torch, arena, specs, **kw:
                            SimpleNamespace(specs=specs, tokens=len(specs)))
        answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                  dtype=None)
        loop.run_join(fake_torch(), cpu_arena(64), pipeline, answers,
                      [[1] * 8, [2] * 8], [[[3, 4]]], 64,
                      anchor_keys=[("a", 0), ("a", 1)])
        assert modes and all(mode == expected for mode in modes)
