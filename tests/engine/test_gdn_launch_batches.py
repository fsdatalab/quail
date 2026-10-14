"""The recurrence kernel runs in launches that bound a layer's state copies."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import quail.backends.quail.executor.gdn as gdn  # noqa: E402


def test_launches_cap_the_sequences_and_match_one_pass(monkeypatch):
    key_heads, value_heads, key_dim, value_dim = 2, 4, 8, 8
    spec = SimpleNamespace(gdn_value_heads=value_heads)
    lengths = [1, 2, 3, 1, 2, 1, 3]
    rows = sum(lengths)
    torch.manual_seed(0)
    q = torch.randn(rows, key_heads, key_dim)
    k = torch.randn(rows, key_heads, key_dim)
    v = torch.randn(rows, value_heads, value_dim)
    gate = -torch.rand(rows, value_heads)
    beta = torch.rand(rows, value_heads)
    start = torch.randn(len(lengths), value_heads, value_dim, key_dim)
    cu = torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int32)
    phase = SimpleNamespace(lengths=tuple(lengths), cu_seqlens=cu)

    launches = []

    def kernel(q, k, v, g, beta, state, cu_seqlens, scale):
        launches.append(len(cu_seqlens) - 1)
        sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return gdn.gated_delta_reference(q, k, v, g, beta, state, sizes, scale)

    monkeypatch.setattr(gdn, "gated_delta_kernel", kernel)
    monkeypatch.setattr(gdn, "SEQUENCES_PER_LAUNCH", 3)
    written = start.clone()

    def load(first, last):
        return start[first:last].clone()

    def store(first, last, state):
        written[first:last] = state

    out = gdn._kernel_in_batches(spec, q, k, v, gate, beta, load, store,
                                 phase, 0.5)
    assert launches == [3, 3, 1]
    expected_state = start.clone()
    expected = gdn.gated_delta_reference(q, k, v, gate, beta, expected_state,
                                         lengths, 0.5)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(written, expected_state)


def test_a_launch_holds_two_state_copies_within_the_pass_reserve():
    from quail.cost.budgets import CHECKPOINT_PASS_QUESTIONS

    # the gathered state and the final state are two copies per sequence,
    # and the reserve counts one copy per question
    assert 2 * gdn.SEQUENCES_PER_LAUNCH <= CHECKPOINT_PASS_QUESTIONS
