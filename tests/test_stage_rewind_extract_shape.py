"""The extraction's stage shape, two questions long, reads consistent KV."""

from types import SimpleNamespace

import pytest
from fakes import cpu_staging
from kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.readout import split_rows
from quail.backends.quail.executor.stages import Stage, run_stages

DOCS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
TAILS = [[70, 71, 72, 41], [73, 74, 75, 76, 77, 41]]   # two questions' tails
DIGIT, CUE2, STEPS = [20], [30, 31], [[50], [51]]
STREAMS = [[60, 61, 62], [63, 64]]           # two starts' spans, read whole


def _whole():
    """A readout of every row of each suffix, as the feed readout is."""
    return SimpleNamespace(
        submit=lambda values, rows_per_answer=None: (list(values), rows_per_answer),
        result=lambda handle: split_rows(handle[0], handle[1]),
        dtype=object)


def _question(tail, readout, keep):
    frame, last = tail[:-1], tail[-1]
    return [
        Stage(suffixes=[[last]], frame=frame, rewind=True, readout=readout,
              decide=keep, single=True, append=True),
        Stage(suffixes=[DIGIT], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
        Stage(suffixes=[CUE2], frame=frame, readout=readout, decide=keep,
              single=True, append=True),
        Stage(suffixes=STEPS, frame=frame, readout=readout, decide=keep),
        Stage(suffixes=STREAMS, frame=frame, readout=_whole(), decide=keep,
              read_all_rows=True),
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
            rows.append([_plant(answers, context + step) for step in STEPS])
            whole = []
            for stream in STREAMS:
                whole.append([_plant(answers, context + stream[:j + 1])
                              for j in range(len(stream))])
            rows.append(whole)
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
            flat = [x for group in (want_row if j % 5 == 4 else [want_row])
                    for x in group] if j % 5 == 4 else want_row
            assert list(_flatten(got_row)) == flat, f"document {d} stage {j}"
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
