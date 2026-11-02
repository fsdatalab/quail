"""The recurrence kernel runs in launches that bound a layer's state copies."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import quail.backends.quail.executor.gdn as gdn  # noqa: E402
from experiments.hybrid import (  # noqa: E402
    gated_delta_reference,
)


def test_launches_cap_the_sequences_and_match_one_pass(monkeypatch):
    from quail.cost.budgets import CHECKPOINT_PASS_QUESTIONS

    assert 2 * gdn.SEQUENCES_PER_LAUNCH <= CHECKPOINT_PASS_QUESTIONS
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

    def kernel(q, k, v, g, beta, state, cu_seqlens, scale, *, output_final_state):
        launches.append((len(cu_seqlens) - 1, output_final_state))
        sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        final = state.clone()
        out = gated_delta_reference(q, k, v, g, beta, final, sizes, scale)
        return out, final if output_final_state else None

    real_kernel = gdn.gated_delta_kernel
    monkeypatch.setattr(gdn, "gated_delta_kernel", kernel)
    monkeypatch.setattr(gdn, "SEQUENCES_PER_LAUNCH", 3)
    written = start.clone()

    def load(first, last):
        return start[first:last].clone()

    def store(first, last, state):
        written[first:last] = state

    out = gdn._kernel_in_batches(spec, q, k, v, gate, beta, load, store,
                                 phase, 0.5)
    assert launches == [(3, True), (3, True), (1, True)]
    expected_state = start.clone()
    expected = gated_delta_reference(q, k, v, gate, beta, expected_state,
                                     lengths, 0.5)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(written, expected_state)
    launches.clear()
    read_only = gdn._kernel_in_batches(
        spec, q, k, v, gate, beta, load, None, phase, 0.5)
    assert launches == [(3, False), (3, False), (1, False)]
    torch.testing.assert_close(read_only, expected)

    for backend in ("flashinfer", "triton"):
        monkeypatch.setattr(gdn, "gdn_prefill_backend", lambda: backend)

        def fused(*, output_final_state, **kwargs):
            torch.testing.assert_close(
                kwargs["q"], q if backend == "flashinfer" else q[None])
            torch.testing.assert_close(
                kwargs["k"], k if backend == "flashinfer" else k[None])
            torch.testing.assert_close(
                kwargs["g"], gate.exp() if backend == "flashinfer" else gate[None])
            if backend == "triton":
                assert kwargs["use_qk_l2norm_in_kernel"] is False
            output = expected if backend == "flashinfer" else expected[None]
            if backend == "flashinfer" and not output_final_state:
                return output
            return output, expected_state if output_final_state else None

        for module in ("flashinfer.gdn_prefill",
                       "vllm.third_party.flash_linear_attention.ops"):
            monkeypatch.setitem(sys.modules, module, SimpleNamespace(
                chunk_gated_delta_rule=fused))
        for keep in (False, True):
            initial = start.clone()
            output, final = real_kernel(
                q, k, v, gate, beta, initial, cu, 0.5, output_final_state=keep)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(initial, start)
            assert final is expected_state if keep else final is None
