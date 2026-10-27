"""AI.MAP's decode rows on a model that checks every KV read.

The model gives each row a hash of its token and of the KV it reads,
as tests/helpers/kv_checker.py does, and chooses the row's next token
from that hash. An answer comes out right only if every decode row
read its own prompt and answer so far at the right positions and
wrote its KV where the next row reads it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers.fakes import cpu_staging, fake_pipeline
from helpers.kv_checker import HashModel, Setup, chain, cpu_torch, make_arena

from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.operators.filter import FilterPart
from quail.backends.quail.executor.operators.map import MapPart
from quail.backends.quail.executor.parts import gated_stages
from quail.backends.quail.executor.stages import run_stages

VOCABULARY = 1024
STOP = VOCABULARY - 1


def next_token(h: int) -> int:
    """The token the model chooses after a row whose hash is h."""
    return STOP if h % 5 == 0 else 10 + h % 900


class TokenModel(HashModel):
    """Answer each read row with a one-hot of the token its hash chooses."""

    def __init__(self, arena):
        super().__init__(arena, answers={}, canvas=())

    def forward_chunk(self, chunk):
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        hashes = [None] * len(ids)
        self._unified(chunk.meta["unified"], ids, positions, hashes)
        rows = chunk.final_indices.tolist()
        out = torch.zeros(len(rows), VOCABULARY)
        out[torch.arange(len(rows)),
            torch.tensor([next_token(hashes[row]) for row in rows])] = 1.0
        return out


class Bits:
    """A filter readout: a row passes when its chosen token is odd."""

    dtype = None
    trailing_rows = 1
    normalize = False

    @staticmethod
    def submit(normed):
        return (normed.argmax(dim=1) % 2).tolist()

    @staticmethod
    def result(handle):
        return handle


def greedy(prompt, limit):
    """The answer greedy decoding writes after a prompt."""
    tokens = []
    while len(tokens) < limit:
        token = next_token(chain(prompt + tokens))
        if token == STOP:
            break
        tokens.append(token)
    return tokens


def map_node(name, tail, limit):
    return SimpleNamespace(spec=SimpleNamespace(
        name=name, prompt_token_parts=((), tuple(tail)), max_answer_tokens=limit))


def events_torch(lag):
    """CPU torch whose events report done only after lag queries."""
    torch_ = cpu_torch()

    def event(**kw):
        asked = []

        def query():
            asked.append(True)
            return len(asked) > lag

        return SimpleNamespace(record=lambda: None, query=query,
                               synchronize=lambda: None,
                               elapsed_time=lambda other: 1.0)

    torch_.cuda.Event = event
    return torch_


@pytest.mark.parametrize("lag", [0, 3])
@pytest.mark.parametrize("budget", [40, 400])
@pytest.mark.parametrize("question", [None, [700, 701]])
def test_map_answers_equal_greedy_decoding(monkeypatch, budget, question, lag):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(0)
    documents = [rng.integers(20, 900, size=int(n)).tolist()
                 for n in rng.integers(3, 30, size=12)]
    tails = [[500, 501, 502], [600, 601]]
    limits = [6, 4]
    setup = Setup(path="unified", pages=200, budget=budget)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk,
                             tree_attention=True, canvas_ids=())
    torch_ = events_torch(lag)
    ids = list(range(len(documents)))
    maps = MapPart(None, [map_node("a", tails[0], limits[0]),
                          map_node("b", tails[1], limits[1])], ids, 0)
    parts = [maps]
    if question is not None:
        parts = [FilterPart(SimpleNamespace(question_token_ids=question), ids,
                            Bits(), 0), maps]
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(documents), limits,
                         [STOP], setup.page_tokens)
    maps.decoder = decoder
    run_stages(torch_, arena, pipeline, gated_stages(parts), documents, budget,
               anchor_keys=[("d", i) for i in ids], attention_mode="unified",
               decoder=decoder)
    passed = [question is None
              or next_token(chain(document + question)) % 2 == 1
              for document in documents]
    for q, (tail, limit) in enumerate(zip(tails, limits)):
        assert maps.asked[q].tolist() == passed
        asked = np.flatnonzero(maps.asked[q])
        got = [answer.tolist() for answer in decoder.answers(q, asked)]
        want = [greedy(documents[d] + tail, limit) for d in asked]
        assert got == want, f"map {q}"
        # answers that stop early and answers cut at the limit
        assert {len(answer) == limit for answer in want} == {True, False}
    assert decoder.rows_fed.min() > 0
    assert (question is None) == all(passed)
    assert not decoder.active
    assert not arena.accounting.owned
