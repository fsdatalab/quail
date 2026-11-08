"""DiffusionGemma: spec geometry, chat-turn text, canvas rows, and the layer loop."""

from types import SimpleNamespace

import pytest
from helpers.fakes import (
    cpu_arena,
    cpu_staging,
    fake_pack,
    fake_pipeline,
    fake_readout,
    fake_torch,
)

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.admission import JoinAdmission
from quail.backends.quail.executor.model_operator import gated_stages
from quail.backends.quail.executor.operators.filter import FilterOperator
from quail.backends.quail.executor.stage_executor import StageExecutor
from quail.backends.quail.executor.stages import filter_stages, frame_writes


def test_canvas_rows_are_charged_in_filter_streams_and_packed_chunks(
        monkeypatch):
    answers = fake_readout()
    documents = [[5] * 20, [6] * 30]
    questions = [[40, 41, 42], [40, 41, 43, 44]]
    # the questions' shared preamble is the frame, written once; each
    # stage's suffix is its question past it, and canvas rows follow
    # every suffix
    parts = [FilterOperator(SimpleNamespace(question_token_ids=q), [0, 1],
                        answers, canvas_tokens=3) for q in questions]
    stages = gated_stages(parts)
    assert [stage.frame for stage in stages] == [[40, 41], [40, 41]]
    assert [stage.suffixes for stage in stages] == [[[42]], [[43, 44]]]
    assert frame_writes(stages) == [True, False]
    assert parts[0].read_answers([{0: [1], 1: [0]}]) == 2 * (3 + 3)
    assert parts[1].read_answers([{0: [1]}]) == 2 + 3
    # the stages' suffix counts include their canvas rows; a frame
    # entry's canvas is counted from frame_canvas_tokens
    sched = JoinAdmission([20, 30], [[1 + 3], [2 + 3]], 200, arena_pages=64,
                          page_tokens=16, frame_tokens=[2, 2],
                          frame_writes=[True, False], frame_canvas_tokens=3)
    assert [lengths.tolist() for lengths in sched.stages] == [[1 + 3], [2 + 3]]
    assert sched.frame_rows == [2 + 3, 0]

    sched = JoinAdmission([10], [[9, 9]], 100, arena_pages=100, page_tokens=16,
                          frame_tokens=[3], frame_canvas_tokens=4)
    assert [lengths.tolist() for lengths in sched.stages] == [[9, 9]]
    assert sched.frame_rows == [7]

    # a model whose attention reads paged KV writes pages even when the
    # plan skipped them; the retention cap set earlier stays
    paged = fake_pipeline(
        needs_pages=True, canvas_ids=(1, 2, 3),
        forward_chunk=lambda chunk: [1 for spec in chunk.specs
                                     for _ in spec["suffixes"]])
    arena = cpu_arena(64)
    arena.retention_cap_pages = 8
    activated = []
    real_activate = arena.activate

    def activate(key, tokens, **kw):
        activated.append(key)
        return real_activate(key, tokens, **kw)

    arena.activate = activate
    real_pack = chunk_mod.pack_chunk
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    StageExecutor(
        fake_torch(), arena, paged,
        filter_stages(questions[:1], answers), documents[:1], 200,
        paged=False,
        default_attention='unified',
        anchor_keys=[('d', 0)]).run()
    monkeypatch.setattr(chunk_mod, "pack_chunk", real_pack)
    assert activated == [("d", 0)]
    assert arena.retention_cap_pages == 8

    # pack_chunk appends the canvas rows after each suffix
    torch = cpu_staging(monkeypatch)
    arena = cpu_arena(64)
    canvas = (90, 91, 92, 93)
    groups = [dict(key=("d", 0), prefix=[1, 2, 3], f=3, suffixes=[[10, 11]]),
              dict(key=("d", 1), prefix=[4, 5], f=2, suffixes=[[12]])]
    chunk = chunk_mod.pack_chunk(torch, arena, groups, attention_mode="unified",
                                 canvas=canvas)
    assert chunk.tokens == 3 + 2 + 4 + 2 + 1 + 4
    assert chunk.input_ids.tolist() == [1, 2, 3, 10, 11, *canvas, 4, 5, 12, *canvas]
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6]
    # the answer row is the first canvas row of each group
    assert chunk.final_indices.tolist() == [5, 12]
    meta = chunk.meta
    assert meta["cu_a"].tolist() == [0, 9, 16]
    assert meta["unified"] is None
    assert meta["canvas"]["rows"].tolist() == [5, 6, 7, 8, 12, 13, 14, 15]
    assert meta["canvas"]["cu_q"].tolist() == [0, 4, 8]
    assert meta["canvas"]["max_q"] == 4

    later = chunk_mod.pack_chunk(torch, arena, groups, attention_mode="unified",
                                 canvas=canvas, answer_row=2)
    assert later.final_indices.tolist() == [7, 14]
    with pytest.raises(ValueError, match="answer_row"):
        chunk_mod.pack_chunk(torch, arena, groups, attention_mode="unified",
                             canvas=canvas, answer_row=4)

    plain = chunk_mod.pack_chunk(torch, arena, groups, attention_mode="unified")
    assert plain.meta["canvas"] is None
    assert plain.final_indices.tolist() == [4, 7]
    with pytest.raises(ValueError, match="unified"):
        chunk_mod.pack_chunk(torch, arena, groups, attention_mode="tree",
                             canvas=canvas)
