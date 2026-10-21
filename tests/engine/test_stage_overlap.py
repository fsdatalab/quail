"""Append stages overlap decisions with independent document work."""

import pytest
from helpers.fakes import cpu_staging
from helpers.kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.stage_executor import StageExecutor
from quail.backends.quail.executor.stages import Stage, StageRequests, run_stages


@pytest.mark.parametrize("path", ["unified", "tree"])
@pytest.mark.parametrize("budget,pages,shared_readout", [
    (1000, 400, True), (1000, 400, False), (18, 400, True), (1000, 3, True),
])
def test_append_stages_launch_between_document_decisions(path, budget, pages,
                                                       shared_readout, monkeypatch):
    cpu_staging(monkeypatch)
    documents = [[a + 1, 20, 21, 22] for a in range(8)]
    frame = [70, 71]
    suffixes = [[40], [50], [51], [52]]
    answers, expected = {}, {}
    for a, document in enumerate(documents):
        context = document + frame
        for j, suffix in enumerate(suffixes):
            context += suffix
            hashed = chain(context)
            answers[hashed] = answer(hashed)
            expected[a, j] = [answers[hashed]]
    events = []
    executors = []
    original = StageExecutor._run_one

    def launch(self, groups):
        assert len(self.own) <= len(documents)
        executors[:] = [self]
        events.append(("launch", tuple(groups)))
        return original(self, groups)

    def decide(j):
        def keep(a, row):
            events.append(("decide", j, a))
            return True
        return keep

    monkeypatch.setattr(StageExecutor, "_run_one", launch)
    readout = passthrough()
    stages = [Stage(suffixes=lambda _, suffix=suffix: StageRequests([suffix]),
                    suffix_tokens=1,
                    frame=frame,
                    readout=readout if shared_readout else passthrough(),
                    decide=decide(j), single=True, append=True)
              for j, suffix in enumerate(suffixes)]
    setup = Setup(path=path, budget=budget, pages=pages)
    run = Run(setup, answers)
    out, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline, stages, documents, budget,
        anchor_keys=[("d", a) for a in range(len(documents))], attention_mode=path)
    for (a, j), want in expected.items():
        assert out[j][a] == want
    assert tokens == sum(map(len, documents)) + len(documents) * (
        len(frame) + len(suffixes)
    )
    assert not run.arena.accounting.owned
    assert not executors[0].own
    first_round = next(i for i, event in enumerate(events)
                       if event[0] == "launch" and any(j > 0 for _, j, *_ in event[1]))
    assert any(event[0:2] == ("decide", 0) for event in events[first_round + 1:])
    if budget == 18:
        assert any(event[0] == "launch"
                   and any(j > 0 for _, j, *_ in event[1])
                   and any(carried for *_, carried in event[1])
                   for event in events)
    elif pages == 400:
        decided = [event[2] for event in events[:first_round]
                   if event[0:2] == ("decide", 0)]
        assert len(decided) == len(documents) // 2
        assert {a for a, *_ in events[first_round][1]} == set(decided)
