"""A resident key whose KV was built for another prefix is fed afresh."""

import pytest
from helpers.fakes import cpu_staging
from helpers.kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.stages import Stage, StageRequests, run_stages
from quail.catalog.tokens import prefix_tree

DOCS = [[1, 2, 3, 4], [5, 6, 7]]
QUESTION = [40]
HEAD = [90, 91]
CUE = [41]


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_a_resident_prefix_of_another_length_is_fed_afresh(path, monkeypatch):
    cpu_staging(monkeypatch)
    # a filter leaves each document's KV resident under its table key;
    # a later operator shows the document after a prompt head
    answers = {}
    for document in DOCS:
        answers[chain(document + QUESTION)] = 1
        h = chain(HEAD + document + CUE)
        answers[h] = answer(h)
    setup = Setup(path=path)
    run = Run(setup, answers)
    run.arena.retention_cap_pages = setup.pages
    keys = [("d", d) for d in range(len(DOCS))]

    def retain(anchor, survived, row):
        run.arena.retain(keys[anchor], len(DOCS[anchor]))

    run_stages(run.torch, run.arena, run.pipeline,
               [Stage(suffixes=[QUESTION], readout=passthrough(), single=True)],
               DOCS, setup.budget, anchor_keys=keys, on_settled=retain,
               attention_mode=path)
    assert all(run.arena.is_resident(key) for key in keys)

    headed = [HEAD + document for document in DOCS]
    out, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline,
        [Stage(suffixes=[CUE], readout=passthrough(), single=True)],
        headed, setup.budget, anchor_keys=keys, attention_mode=path)
    # the checking model fails any row that reads KV no request wrote
    for d, document in enumerate(DOCS):
        assert out[0][d] == [answers[chain(HEAD + document + CUE)]], f"document {d}"
    assert tokens == sum(len(prefix) for prefix in headed) + len(DOCS)
    assert not run.arena.accounting.owned


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_borrowed_prefix_and_accepted_decode_tokens_survive_rounds(path, monkeypatch):
    cpu_staging(monkeypatch)
    shared = list(range(1000, 1016))
    documents = [shared + [1020, 1021], shared + [1030, 1031]]
    frame = [30, 31]
    paths = [frame + [32], frame + [32, 41, 42, 43],
             frame + [32, 41, 42, 44, 45], [50, 51, 52], [50, 51, 52, 53]]
    expected = {chain(document + tail): float(index + 1)
                for document in documents for index, tail in enumerate(paths)}
    setup = Setup(path=path)
    run = Run(setup, expected)
    stages = [
        Stage(suffixes=[[32]], frame=frame, single=True, append=True,
              rewind=True, readout=passthrough()),
        Stage(suffixes=lambda key: StageRequests(
                  [[41, 42, 43]], write_suffix_tokens=2, read_rows=[1]),
              suffix_tokens=3, frame=frame, single=True, append=True,
              readout=passthrough()),
        Stage(suffixes=[[44, 45]], frame=frame, single=True, append=True,
              readout=passthrough()),
        Stage(suffixes=[[52]], frame=[50, 51], single=True, append=True,
              rewind=True, readout=passthrough()),
        Stage(suffixes=[[53]], frame=[50, 51], single=True, append=True,
              readout=passthrough()),
    ]
    stats = {}
    output, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline, stages, documents, setup.budget,
        attention_mode=path, prefix_tree=prefix_tree(documents, setup.page_tokens),
        stats=stats)
    assert stats["borrowed_tokens"] == 16
    assert tokens == sum(map(len, documents)) - 16 + len(documents) * (
        3 + 3 + 2 + 3 + 1
    )
    for index in range(len(stages)):
        assert output[index] == {0: [float(index + 1)], 1: [float(index + 1)]}
    assert not run.arena.accounting.owned
