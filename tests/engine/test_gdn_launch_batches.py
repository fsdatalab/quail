"""The recurrence kernel runs in launches that bound a layer's state copies."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import quail.backends.quail.executor.gdn as gdn  # noqa: E402
from experiments.hybrid import (  # noqa: E402
    causal_conv_reference,
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


def test_convolution_reuses_metadata_without_overwriting_shared_state(monkeypatch):
    lengths, channels, width = (1, 4, 2), 8, 3
    rows = sum(lengths)
    bounds = torch.tensor([0, 1, 5, rows], dtype=torch.int32)
    builds = []

    def metadata(cu, *, device):
        assert cu.device.type == "cpu"
        torch.testing.assert_close(cu, bounds)
        builds.append(device)
        return {}, torch.empty(0), torch.empty(0)

    def convolution(x, weight, bias, *, conv_states, query_start_loc,
                    cache_indices, has_initial_state, null_block_id,
                    activation, metadata):
        assert null_block_id is None
        torch.testing.assert_close(query_start_loc, bounds)
        torch.testing.assert_close(cache_indices, torch.arange(3, dtype=torch.int32))
        assert conv_states.shape[0] == 4
        history = (conv_states[:3].transpose(1, 2)
                   if has_initial_state.all() else None)
        out, tails = causal_conv_reference(
            x.T, weight, lengths, history, bias, activate=activation == "silu")
        conv_states[:3].copy_(tails.transpose(1, 2))
        return out.T

    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.utils",
                        SimpleNamespace(compute_causal_conv1d_metadata=metadata))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        SimpleNamespace(causal_conv1d_fn=convolution))
    inputs, weight = torch.randn(rows, channels), torch.randn(channels, width)
    arena = SimpleNamespace(conv_state=torch.randn(4, 1, width - 1, channels))
    original = arena.conv_state.clone()
    phase = gdn.GdnPhase(
        rows=torch.arange(rows), cu_seqlens=bounds, lengths=lengths,
        read_slots=torch.tensor([0, 0, 0]))
    expected, tails = causal_conv_reference(
        inputs, weight, lengths, original[[0, 0, 0], 0], activate=False)
    for _ in range(2):
        output = gdn.short_conv_core(inputs, weight, (phase,), arena, 0)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(arena.conv_state, original)
    assert len(builds) == 1
    phase.write_slots = torch.tensor([1, 2, 3])
    gdn.short_conv_core(inputs, weight, (phase,), arena, 0)
    torch.testing.assert_close(arena.conv_state[0], original[0])
    torch.testing.assert_close(arena.conv_state[[1, 2, 3], 0], tails)
    assert len(builds) == 1

    phase = gdn.GdnPhase(rows=None, cu_seqlens=bounds, lengths=lengths)
    expected, tails = causal_conv_reference(inputs, weight, lengths)
    for _ in range(2):
        output, state = gdn.causal_conv_kernel(inputs, weight, phase, None)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(state, tails)
    assert len(builds) == 2


def test_gdn_prepares_selected_rows_once_before_batched_recurrence(monkeypatch):
    from experiments.hybrid import l2_normalize

    spec = SimpleNamespace(gdn_key_heads=2, gdn_value_heads=4,
                           gdn_key_dim=8, gdn_value_dim=8)
    rows, channels = 4, 64
    tensors = dict(qkv=torch.randn(rows, channels), a=torch.randn(rows, 4),
                   b=torch.randn(rows, 4), A_log=torch.randn(4),
                   dt_bias=torch.randn(4), conv_weight=None, conv_bias=None)
    selected = torch.tensor([2, 0, 3])
    phase = gdn.GdnPhase(rows=selected, lengths=(1, 2),
                         cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32))
    prepared = []

    def prepare(*, conv_output, a, b, A_log, dt_bias, num_k_heads,
                head_k_dim, head_v_dim, apply_l2norm, output_g_exp):
        torch.testing.assert_close(conv_output, tensors["qkv"][selected])
        torch.testing.assert_close(a, tensors["a"][selected])
        torch.testing.assert_close(b, tensors["b"][selected])
        assert (num_k_heads, head_k_dim, head_v_dim) == (2, 8, 8)
        assert apply_l2norm and not output_g_exp
        q, k, v = conv_output.split([16, 16, 32], dim=-1)
        result = (l2_normalize(q.reshape(-1, 2, 8)),
                  l2_normalize(k.reshape(-1, 2, 8)), v.reshape(-1, 4, 8),
                  -A_log.exp() * torch.nn.functional.softplus(a + dt_bias),
                  b.sigmoid())
        prepared.append(result)
        return result

    consumed = []

    def recurrence(q, k, v, g, beta, state, cu_seqlens, scale, *,
                   output_final_state):
        low = sum(consumed)
        high = low + len(q)
        for actual, expected in zip((q, k, v, g, beta), prepared[0], strict=True):
            torch.testing.assert_close(actual, expected[low:high])
        assert not output_final_state
        assert scale == 8 ** -0.5
        assert state.count_nonzero() == 0
        torch.testing.assert_close(cu_seqlens, torch.tensor([0, len(q)],
                                                          dtype=torch.int32))
        consumed.append(len(q))
        return v, None

    monkeypatch.setitem(sys.modules, "vllm.third_party.flash_linear_attention.ops",
                        SimpleNamespace(fused_post_conv_prep=prepare))
    monkeypatch.setattr(gdn, "causal_conv_kernel", lambda x, *_: (x, None))
    monkeypatch.setattr(gdn, "gated_delta_kernel", recurrence)
    monkeypatch.setattr(gdn, "SEQUENCES_PER_LAUNCH", 1)
    out = gdn.gdn_core(spec, tensors, (phase,), SimpleNamespace(), 0)
    expected = torch.zeros(rows, 4, 8)
    expected[selected] = tensors["qkv"][selected, 32:].reshape(-1, 4, 8)
    torch.testing.assert_close(out, expected)
    assert len(prepared) == 1
    assert consumed == [1, 2]


def test_convolution_warmup_covers_rounded_capacity_and_reuses_storage(monkeypatch):
    from quail.backends.quail.executor.warmup import _warm_convolutions

    weight, bias = torch.randn(8, 3), torch.randn(8)
    pipeline = SimpleNamespace(convolutions=lambda: ((weight, bias, False),))
    launches, storage = [], set()

    def convolution(x, w, phase, history, b, activate):
        assert w is weight and b is bias and not activate
        assert x.shape == (sum(phase.lengths), 8)
        assert history.stride() == (16, 8, 1)
        launches.append((len(x), len(history)))
        storage.add(history.untyped_storage().data_ptr())

    monkeypatch.setattr(gdn, "causal_conv_kernel", convolution)
    _warm_convolutions(torch, pipeline, 17)
    assert launches == [(2, capacity) for capacity in (1, 2, 4, 8, 16, 32)]
    assert len(storage) == 1
