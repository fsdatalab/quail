"""The extraction's stage shape, two questions long, reads consistent KV."""

import pytest
from fakes import cpu_staging
from kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.stages import Stage, run_stages

DOCS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
TAILS = [[70, 71, 72, 41], [73, 74, 75, 76, 77, 41]]   # two questions' tails
DIGIT, CUE2, ANSWER = [20], [30, 31], [[50], [51]]      # two answer tokens


def _question(tail, readout, keep):
    frame, last = tail[:-1], tail[-1]
    return [
        Stage(suffixes=[[last]], frame=frame, rewind=True, readout=readout,
              decide=keep, single=True, append=True),
        Stage(suffixes=[DIGIT], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
        Stage(suffixes=[CUE2], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
        Stage(suffixes=[ANSWER[0]], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
        Stage(suffixes=[ANSWER[1]], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
    ]


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_two_questions_of_the_extraction_shape_read_their_own_tails(path, monkeypatch):
    cpu_staging(monkeypatch)
    answers = {}
    want = {}
    for d, doc in enumerate(DOCS):
        rows = []
        for tail in TAILS:
            context = doc + tail
            rows.append([_plant(answers, context)])
            context = context + DIGIT
            rows.append([_plant(answers, context)])
            context = context + CUE2
            rows.append([_plant(answers, context)])
            for token in ANSWER:
                context = context + token
                rows.append([_plant(answers, context)])
        want[d] = rows
    setup = Setup(path=path)
    run = Run(setup, answers)

    def keep(a, row):
        return True

    stages = _question(TAILS[0], passthrough(), keep) + _question(
        TAILS[1], passthrough(), keep)
    out, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline, stages, DOCS, setup.budget,
        anchor_keys=[("d", d) for d in range(len(DOCS))], attention_mode=path)
    # the checking model fails any row that reads KV no request wrote;
    # the answers show every stage read the row it asked for
    for d, rows in want.items():
        got = [out[j][d] for j in range(len(stages))]
        for j, (want_row, got_row) in enumerate(zip(rows, got)):
            assert list(_flatten(got_row)) == want_row, f"document {d} stage {j}"
    assert not run.arena.accounting.owned
    assert tokens > 0


def _plant(answers, tokens):
    h = chain(tokens)
    answers[h] = answer(h)
    return answers[h]


def _flatten(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value
