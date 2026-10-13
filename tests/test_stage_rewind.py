"""A rewinding stage writes after the frame, past what earlier stages appended."""

import pytest
from fakes import cpu_staging
from kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.stages import Stage, run_stages

DOCS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
FIRST, SECOND = [70, 71], [72, 73, 74, 75]     # two questions' frames
CUE, STEP = [41], [42, 43]


def _question(frame, seen, index, readout):
    """A question's stages: its cue after the frame, then an appended step."""
    def keep(j):
        def decide(a, row):
            seen[j][a] = list(row)
            return True
        return decide

    return [
        Stage(suffixes=[CUE], frame=frame, rewind=True, readout=readout,
              decide=keep(index), single=True, append=True),
        Stage(suffixes=[STEP], frame=frame, readout=readout,
              decide=keep(index + 1), single=True, append=True),
    ]


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_each_question_reads_its_frame_right_after_the_document(path, monkeypatch):
    cpu_staging(monkeypatch)
    answers = {}
    want = {}
    for d, doc in enumerate(DOCS):
        rows = []
        for frame in (FIRST, SECOND, SECOND):
            cue = chain(doc + frame + CUE)
            step = chain(doc + frame + CUE + STEP)
            answers[cue], answers[step] = answer(cue), answer(step)
            rows += [[answers[cue]], [answers[step]]]
        want[d] = rows
    setup = Setup(path=path)
    run = Run(setup, answers)
    seen = [{} for _ in range(6)]
    # the second frame replaces the first and its step; the third
    # question has the second's frame, rewound past its step
    stages = (_question(FIRST, seen, 0, passthrough())
              + _question(SECOND, seen, 2, passthrough())
              + _question(SECOND, seen, 4, passthrough()))
    out, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline, stages, DOCS, setup.budget,
        anchor_keys=[("d", d) for d in range(len(DOCS))], attention_mode=path)
    # the checking model fails any row that reads KV no request wrote
    for d, rows in want.items():
        assert [out[j][d] for j in range(6)] == rows, f"document {d}"
    # each document packs its prefix once, two frames, and three
    # questions' cue and step
    per_document = len(FIRST) + len(SECOND) + 3 * (len(CUE) + len(STEP))
    assert tokens == sum(map(len, DOCS)) + len(DOCS) * per_document
    assert not run.arena.accounting.owned
