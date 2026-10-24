"""A second operator over saved documents reads their prefix back from the host."""

import pytest
from helpers.fakes import cpu_staging
from helpers.kv_checker import Run, Setup, answer, chain, passthrough

from quail.backends.quail.executor.host_kv import HostKV
from quail.backends.quail.executor.stages import Stage, run_stages

torch = pytest.importorskip("torch")

HEAD = [90, 91]
DOCS = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
QUESTIONS = [[40, 41], [50, 51, 52]]


def planted(documents, questions):
    answers = {}
    for document in documents:
        for question in questions:
            h = chain(HEAD + document + question)
            answers[h] = answer(h)
    return answers


def run_filter(run, question, keys):
    """One filter stage over every document; returns its answers and tokens."""
    prefixes = [HEAD + document for document in DOCS]
    got, _, tokens = run_stages(
        run.torch, run.arena, run.pipeline,
        [Stage(suffixes=[question], readout=passthrough(), single=True)],
        prefixes, run.setup.budget, anchor_keys=keys,
        attention_mode=run.setup.path)
    return {d: int(got[0][d][0]) for d in range(len(DOCS))}, tokens


@pytest.mark.parametrize("path", ["unified", "tree"])
@pytest.mark.parametrize("window", [None, 4])
def test_a_query_over_saved_documents_packs_no_prefix_tokens(
        path, window, monkeypatch):
    cpu_staging(monkeypatch)
    setup = Setup(path=path, window=window, pages=64, sliding_pages=64,
                  page_tokens=4, budget=64)
    run = Run(setup, planted(DOCS, QUESTIONS))
    host = HostKV(run.arena, 4 * 2**20, torch, pinned=False)
    run.arena.host = host
    keys = [("d", d) for d in range(len(DOCS))]
    want = {d: run.model.answers[chain(HEAD + DOCS[d] + QUESTIONS[0])]
            for d in range(len(DOCS))}
    first, first_tokens = run_filter(run, QUESTIONS[0], keys)
    assert first == want
    # every document was freed when it finished, and saved first
    assert not run.arena.accounting.owned
    assert host.saved_keys == len(DOCS)
    # the next query's keys name the same documents under the same head
    host.begin_query()
    second, second_tokens = run_filter(run, QUESTIONS[1], keys)
    want = {d: run.model.answers[chain(HEAD + DOCS[d] + QUESTIONS[1])]
            for d in range(len(DOCS))}
    assert second == want
    assert host.restored_keys == len(DOCS)
    prefix_tokens = sum(len(HEAD) + len(document) for document in DOCS)
    assert second_tokens == first_tokens - prefix_tokens + (
        len(QUESTIONS[1]) - len(QUESTIONS[0])) * len(DOCS)
    assert not run.arena.accounting.owned


def test_an_entry_dropped_before_its_reserve_is_computed_instead(monkeypatch):
    cpu_staging(monkeypatch)
    setup = Setup(path="tree", pages=64, page_tokens=4, budget=64)
    run = Run(setup, planted(DOCS, QUESTIONS))
    host = HostKV(run.arena, 4 * 2**20, torch, pinned=False)
    run.arena.host = host
    keys = [("d", d) for d in range(len(DOCS))]
    run_filter(run, QUESTIONS[0], keys)
    host.begin_query()
    # the store loses every entry between the lookup and the reserve
    restore = host.restore
    monkeypatch.setattr(host, "restore", lambda key, digest: (
        host.entries.clear(), restore(key, digest))[1])
    second, second_tokens = run_filter(run, QUESTIONS[1], keys)
    want = {d: run.model.answers[chain(HEAD + DOCS[d] + QUESTIONS[1])]
            for d in range(len(DOCS))}
    assert second == want
    assert host.restored_keys == 0 and host.misses == len(DOCS)
    assert second_tokens == sum(
        len(HEAD) + len(document) + len(QUESTIONS[1]) for document in DOCS)
