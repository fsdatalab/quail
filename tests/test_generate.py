"""GreedyGenerator's scheduling, with a fake model on the CPU."""

import numpy as np
import pytest
from fakes import cpu_staging
from test_state_pool import state_arena

from quail.backends.quail.executor.generate import (
    Generation,
    GreedyGenerator,
    admission_ready,
)

torch = pytest.importorskip("torch")

VOCAB = 32
STOP = 0


class _CountingModel:
    """Chooses the fed token plus one, then STOP once it passes a limit.

    The head is the identity, so the hidden state is the next token's
    one-hot row. Every chunk is recorded.
    """

    def __init__(self, limit):
        self.limit = limit
        self.chunks = []

    def forward_chunk(self, chunk):
        self.chunks.append(chunk)
        fed = chunk.input_ids[chunk.final_indices]
        nxt = torch.where(fed + 1 > self.limit, torch.zeros_like(fed), fed + 1)
        return torch.nn.functional.one_hot(nxt, VOCAB).float()


def _generator(arena, model, lanes=2, budget=64):
    return GreedyGenerator(torch, arena, model, torch.eye(VOCAB), lanes=lanes,
                           budget=budget, max_new_tokens=8, context_tokens=48,
                           stops=[STOP], graphs=False)


def test_each_document_generates_until_its_stop_token(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=4)
    model = _CountingModel(limit=9)
    prompts = {"a": [3, 4, 5], "b": [6], "c": [8, 7, 6, 5]}
    done = dict(_generator(arena, model).run(
        Generation(key, np.array(ids)) for key, ids in prompts.items()))
    # each continues counting from its prompt's last token to the limit
    assert done == {"a": [6, 7, 8, 9, STOP], "b": [7, 8, 9, STOP],
                    "c": [6, 7, 8, 9, STOP]}
    # every page went back to the arena
    assert arena.resident_keys() == []


def test_lanes_bound_the_documents_in_flight(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=4)
    model = _CountingModel(limit=6)
    keys = [f"d{i}" for i in range(5)]
    done = list(_generator(arena, model, lanes=2).run(
        Generation(key, np.array([1, 2])) for key in keys))
    assert sorted(key for key, _ in done) == keys
    # one row per document in flight, plus two-token prompts
    assert max(len(c.final_indices) for c in model.chunks) <= 2


def test_the_output_limit_ends_a_document(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=4)
    model = _CountingModel(limit=VOCAB)
    ((key, tokens),) = _generator(arena, model).run(
        [Generation("long", np.array([1]))])
    assert tokens == [2, 3, 4, 5, 6, 7, 8, 9]


def test_admission_waits_for_a_budget_of_prompts_while_documents_decode():
    # nothing decoding, or the input done: admit what fits
    assert admission_ready(free=1, decoding=0, mean_prompt=100, budget=500,
                           exhausted=False)
    assert admission_ready(free=1, decoding=10, mean_prompt=100, budget=500,
                           exhausted=True)
    # 3 free lanes x 100 tokens < 500 - 10: keep decoding alone
    assert not admission_ready(free=3, decoding=10, mean_prompt=100, budget=500,
                               exhausted=False)
    # 5 free lanes take 500 >= 490
    assert admission_ready(free=5, decoding=10, mean_prompt=100, budget=500,
                           exhausted=False)
    assert not admission_ready(free=0, decoding=0, mean_prompt=100, budget=500,
                               exhausted=True)
