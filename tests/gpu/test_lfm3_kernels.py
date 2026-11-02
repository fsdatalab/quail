"""The LFM3 Triton kernels against the PyTorch code they replace, on a GPU.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword lfm3_kernels`.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

# d1-2B's geometry
HIDDEN, HEADS, KV_HEADS, DIM, WIDTH = 2048, 16, 4, 128, 7


def _rms(x, weight, eps=1e-5):
    """VLLM's rms_norm: the scaled row rounded to bf16, then the weight."""
    scaled = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return scaled.to(torch.bfloat16) * weight


def _steps(actual, expected, size) -> float:
    """The largest difference, in bf16 steps at each value's size."""
    exponent = torch.floor(torch.log2(size.float().clamp_min(2 ** -100)))
    step = torch.exp2(exponent - 7)
    return float(((actual.float() - expected.float()).abs() / step).max())


def _bf16(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


def test_the_gated_convolution_matches_shifted_sums_within_each_sequence():
    from quail.backends.quail.executor.models import lfm3_kernels as kernels

    torch.manual_seed(0)
    lengths = [1, 5, 300, 64]
    rows = sum(lengths)
    projected = _bf16(rows, 3 * HIDDEN)
    taps = _bf16(HIDDEN, WIDTH, scale=0.3)
    starts = kernels.sequence_starts(torch, lengths, "cuda")

    out = kernels.gated_conv(projected, taps, starts)

    b, c, x = projected.chunk(3, -1)
    gated = b * x
    expected, first = [], 0
    for length in lengths:
        own = gated[first:first + length].float()
        padded = torch.cat((own.new_zeros(WIDTH - 1, HIDDEN), own))
        conv = sum(padded[tap:tap + length] * taps[:, tap].float()
                   for tap in range(WIDTH))
        expected.append(c[first:first + length] * conv.to(torch.bfloat16))
        first += length
    torch.testing.assert_close(out, torch.cat(expected), atol=2e-2, rtol=2e-2)


def test_the_fused_norms_match_vllms_norm_add_norm():
    from vllm import _custom_ops as ops

    from quail.backends.quail.executor.models import lfm3_kernels as kernels

    torch.manual_seed(1)
    rows = 517
    x, hidden = _bf16(rows, HIDDEN), _bf16(rows, HIDDEN, scale=4.0)
    hidden_in = hidden.clone()
    post, following = _bf16(HIDDEN), _bf16(HIDDEN)

    def vllm_norm(rows_in, weight):
        out = torch.empty_like(rows_in)
        ops.rms_norm(out, rows_in, weight, 1e-5)
        return out

    expected_hidden = hidden + vllm_norm(x, post)
    expected = vllm_norm(expected_hidden, following)
    out = kernels.post_norm_add_norm(x, hidden, post, following, 1e-5)
    last = hidden_in.clone()
    kernels.post_norm_add_norm(x, last, post, None, 1e-5)

    # the norms sum their squares in another order than vLLM's, so a
    # normed value can round one bf16 step apart, and the residual sum
    # rounds once more: two steps at the size of the values summed,
    # even where their sum cancels
    summed = hidden_in.abs() + vllm_norm(x, post).abs()
    assert _steps(hidden, expected_hidden, summed) <= 2
    assert _steps(last, expected_hidden, summed) <= 2
    # the next norm scales those values by the row's norm and its weight,
    # and rounds twice more
    rows_scale = torch.rsqrt(expected_hidden.float().pow(2).mean(-1, keepdim=True)
                             + 1e-5)
    assert _steps(out, expected, summed * rows_scale * following.abs()) <= 4


def test_the_head_gate_scales_each_head_by_its_sigmoid():
    from quail.backends.quail.executor.models import lfm3_kernels as kernels

    torch.manual_seed(2)
    rows = 300
    out = _bf16(rows, HEADS * DIM)
    projected = _bf16(rows, 64 + HEADS)
    expected = (out.view(rows, HEADS, DIM)
                * torch.sigmoid(projected[:, 64:])[:, :, None]).view(rows, -1)

    kernels.gate_heads(out, projected[:, 64:])

    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_the_question_merge_weights_each_read_by_its_log_sum_exp():
    from quail.backends.quail.executor.models import lfm3_kernels as kernels

    torch.manual_seed(3)
    rows, count = 40, 12
    out = _bf16(rows, HEADS, DIM)
    asked = torch.arange(20, 32, device="cuda")
    a, b = _bf16(count, HEADS, DIM), _bf16(count, HEADS, DIM)
    lse_a = torch.randn(HEADS, count, device="cuda")
    lse_b = torch.randn(HEADS, count, device="cuda")
    expected = out.clone()
    wa = (lse_a - lse_a.maximum(lse_b)).exp().transpose(0, 1)[:, :, None]
    wb = (lse_b - lse_a.maximum(lse_b)).exp().transpose(0, 1)[:, :, None]
    expected[asked] = ((a.float() * wa + b.float() * wb) / (wa + wb)).to(
        torch.bfloat16)

    kernels.merge_questions(out, asked, a, lse_a, b, lse_b)

    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def test_the_qk_norm_and_partial_rotary_match_norming_then_rotating():
    from quail.backends.quail.executor.attention import Engine
    from quail.backends.quail.executor.models.lfm3 import rotary_tables

    torch.manual_seed(4)
    rows = 257
    projected = _bf16(rows, (HEADS + 2 * KV_HEADS) * DIM + HEADS, scale=3.0)
    q_weight, k_weight = _bf16(DIM), _bf16(DIM)
    positions = torch.randint(0, 4000, (rows,), device="cuda")
    # the full layers' rotary: half the pairs rotate, the rest stay
    config = dict(head_dim=DIM, rope_parameters={"full_attention": dict(
                      rope_type="default", rope_theta=1e6,
                      partial_rotary_factor=0.5)})
    cos, sin = rotary_tables(torch, config, "full_attention", 4096, "cuda")
    engine = Engine.__new__(Engine)
    engine.torch = torch

    q, k = engine.qk_norm_rope_heads(
        projected, positions, num_q_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=DIM, q_weight=q_weight, k_weight=k_weight, eps=1e-5,
        cos_sin_cache=torch.cat((cos, sin), -1).contiguous())

    def expected(part, heads, weight):
        normed = part.reshape(rows, heads, DIM).float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + 1e-5)
        normed = normed * weight.float()
        first, second = normed.chunk(2, -1)
        c, s = cos[positions][:, None], sin[positions][:, None]
        return torch.cat((first * c - second * s, second * c + first * s),
                         -1).to(torch.bfloat16).reshape(rows, heads * DIM)

    torch.testing.assert_close(
        q, expected(projected[:, :HEADS * DIM], HEADS, q_weight),
        atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        k, expected(projected[:, HEADS * DIM:(HEADS + KV_HEADS) * DIM],
                    KV_HEADS, k_weight), atol=3e-2, rtol=2e-2)
