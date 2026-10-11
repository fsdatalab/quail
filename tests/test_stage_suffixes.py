"""A stage whose suffixes are each document's own, chosen by an earlier stage."""

import pytest
from fakes import cpu_staging
from kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.stages import Stage, run_stages

DOCS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
CUE = [40]


def _second(bit):
    """Stage 1's two suffixes, chosen by the document's stage 0 answer."""
    return [[50 + bit, 51 + bit], [60 + bit]]


def _third(bits):
    """Stage 2's one suffix, chosen by the document's stage 1 answers."""
    return [[70 + sum(bits)]]


def _fourth(bit):
    """Stage 3's one suffix, chosen by the document's stage 2 answer."""
    return [[80 + bit]]


def _plant(answers, tokens):
    h = chain(tokens)
    answers[h] = answer(h)
    return answers[h]


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_own_suffixes_follow_each_documents_answers(path, monkeypatch):
    cpu_staging(monkeypatch)
    # every suffix a document can be fed, and its answer, planted by hash
    answers = {}
    want = {}
    for d, doc in enumerate(DOCS):
        bit = _plant(answers, doc + CUE)
        if d == 1:
            bits = []
        else:
            bits = [_plant(answers, doc + CUE + suffix) for suffix in _second(bit)]
        (third,) = _third(bits)
        bit2 = _plant(answers, doc + CUE + third)
        (fourth,) = _fourth(bit2)
        bit3 = _plant(answers, doc + CUE + third + fourth)
        want[d] = (bit, bits, bit2, bit3)
    setup = Setup(path=path)
    run = Run(setup, answers)
    keys = [("d", d) for d in range(len(DOCS))]
    seen = [{} for _ in range(4)]      # per stage, document -> its row

    def keep(j):
        def decide(a, row):
            seen[j][a] = list(row)
            return True
        return decide

    def second(key):
        d = key[1]
        return Stage.SKIP if d == 1 else _second(seen[0][d][0])

    stages = [
        Stage(suffixes=[CUE], readout=passthrough(), decide=keep(0),
              single=True, append=True, label="cue"),
        Stage(suffixes=second, suffix_tokens=2, readout=passthrough(),
              decide=keep(1), label="second"),
        Stage(suffixes=lambda key: _third(seen[1].get(key[1], [])),
              suffix_tokens=1, readout=passthrough(), decide=keep(2),
              single=True, append=True, label="third"),
        Stage(suffixes=lambda key: _fourth(seen[2][key[1]][0]),
              suffix_tokens=1, readout=passthrough(), decide=keep(3),
              single=True, append=True, label="fourth"),
    ]
    out, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline, stages, DOCS, setup.budget,
        anchor_keys=keys, attention_mode=path)
    for d, (bit, bits, bit2, bit3) in want.items():
        assert out[0][d] == [bit], f"document {d}"
        if d == 1:
            assert d not in out[1]
        else:
            assert out[1][d] == bits, f"document {d}"
        assert out[2][d] == [bit2], f"document {d}"
        assert out[3][d] == [bit3], f"document {d}"
    # each document packs its prefix and cue once, its two second-stage
    # suffixes (3 tokens, none for the skipped document), and one token
    # at each of the two appended stages
    assert tokens == sum(len(doc) for doc in DOCS) + 4 + 3 * 3 + 4 + 4
    assert not run.arena.accounting.owned


def test_own_suffixes_name_their_most_tokens():
    with pytest.raises(ValueError, match="suffix_tokens"):
        Stage(suffixes=lambda key: [[1]], readout=passthrough())
    with pytest.raises(ValueError, match="requests"):
        Stage(suffixes=lambda key: [[1]], suffix_tokens=1,
              readout=passthrough(), requests=lambda key: [0])


def test_own_suffix_over_its_most_tokens_is_refused(monkeypatch):
    cpu_staging(monkeypatch)
    answers = {}
    for doc in DOCS:
        _plant(answers, doc + CUE)
    setup = Setup(path="unified")
    run = Run(setup, answers)
    stages = [
        Stage(suffixes=[CUE], readout=passthrough(), single=True, append=True,
              decide=lambda a, row: True),
        Stage(suffixes=lambda key: [[50, 51, 52]], suffix_tokens=2,
              readout=passthrough(), label="long"),
    ]
    with pytest.raises(ValueError, match="long: a document's suffix has 3"):
        run_stages(run.torch, run.arena, run.pipeline, stages, DOCS,
                   setup.budget, anchor_keys=[("d", d) for d in range(len(DOCS))],
                   attention_mode="unified")
