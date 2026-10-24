"""The fused SAE encode against the blocked torch path, on a GPU.

The two paths round their GEMMs differently (tl.dot against cuBLAS),
so near the top-k boundary a feature can swap with its neighbour. The
test checks that each row's feature sets differ by at most a few
boundary features, that the shared features' strengths agree within
GEMM rounding, and that the 'max' accumulation over a fake chunk
agrees the same way. Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py`.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

NUM_FEATURES = 32_768
HIDDEN = 2_048
KEEP = 100


def _encoders():
    import torch.nn.functional as functional

    from quail.backends.quail.executor.sae import SaeEncoder

    torch.manual_seed(7)
    # spread the feature scales over two decades so the pre-activation
    # distribution is heavy-tailed like a real SAE's; with a flat
    # Gaussian, hundreds of values crowd the top-k boundary bin and
    # every row overflows the fused path's candidate buffer
    scales = torch.logspace(-1.0, 1.0, NUM_FEATURES, device="cuda")
    weight = (torch.randn(NUM_FEATURES, HIDDEN, device="cuda")
              / HIDDEN ** 0.5 * scales[:, None]).to(torch.bfloat16)
    bias = torch.randn(NUM_FEATURES, device="cuda") * 0.01
    fused = SaeEncoder(torch, functional, weight, bias, KEEP,
                       kernels="quail")
    plain = SaeEncoder(torch, functional, weight, bias, KEEP,
                       kernels="vllm")
    return fused, plain


def _agreeing_rows(fused_index, fused_value, plain_index, plain_value):
    """Check one batch's fused and torch readings row by row."""
    for row in range(fused_index.shape[0]):
        fused_set = {int(feature) for feature, value in
                     zip(fused_index[row].tolist(), fused_value[row].tolist())
                     if value > 0}
        plain_set = {int(feature) for feature, value in
                     zip(plain_index[row].tolist(), plain_value[row].tolist())
                     if value > 0}
        boundary = fused_set ^ plain_set
        assert len(boundary) <= 8, f"row {row} swaps {len(boundary)} features"
        shared = fused_set & plain_set
        fused_of = dict(zip(fused_index[row].tolist(),
                            fused_value[row].tolist()))
        plain_of = dict(zip(plain_index[row].tolist(),
                            plain_value[row].tolist()))
        # a swapped feature must sit at the top-k boundary, not be a
        # feature one path plainly missed
        cut = min(plain_value[row].tolist())
        for feature in boundary:
            value = fused_of.get(feature) or plain_of.get(feature)
            assert abs(value - cut) <= 5e-2 * max(1.0, abs(cut)), (
                f"row {row} feature {feature} is {value} against cut {cut}")
        for feature in shared:
            assert abs(fused_of[feature] - plain_of[feature]) <= (
                2e-2 * max(1.0, abs(plain_of[feature])))


@pytest.mark.parametrize("rows", [1, 63, 517])
def test_fused_top_features_match_the_torch_path(rows):
    fused, plain = _encoders()
    hidden = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fused_index, fused_value = fused.top_features(hidden)
    plain_index, plain_value = plain.top_features(hidden)
    assert fused_index.shape == plain_index.shape == (rows, KEEP)
    # if every row overflowed the candidate buffer, the fused call
    # answered from the torch path and the comparison checks nothing
    assert fused.overflow_rows == 0
    _agreeing_rows(fused_index, fused_value, plain_index, plain_value)


def test_a_transposed_checkpoint_weight_still_encodes_correctly():
    """The checkpoint's W_enc is a transposed view.

    Its inner stride is num_features, while the fused kernels index the
    weight with an inner stride of one, so the encoder must make it
    contiguous.
    """
    import torch.nn.functional as functional

    from quail.backends.quail.executor.sae import SaeEncoder

    torch.manual_seed(7)
    scales = torch.logspace(-1.0, 1.0, NUM_FEATURES, device="cuda")
    weight = (torch.randn(HIDDEN, NUM_FEATURES, device="cuda")
              / HIDDEN ** 0.5 * scales[None, :]).to(torch.bfloat16).t()
    assert not weight.is_contiguous()
    bias = torch.randn(NUM_FEATURES, device="cuda") * 0.01
    fused = SaeEncoder(torch, functional, weight, bias, KEEP,
                       kernels="quail")
    plain = SaeEncoder(torch, functional, weight, bias, KEEP,
                       kernels="vllm")
    hidden = torch.randn(63, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fused_index, fused_value = fused.top_features(hidden)
    plain_index, plain_value = plain.top_features(hidden)
    assert fused.overflow_rows == 0
    _agreeing_rows(fused_index, fused_value, plain_index, plain_value)


def _maxima(encoder, rows):
    """Per-feature maxima over the rows, as the accumulator folds them."""
    buffer = torch.zeros(NUM_FEATURES, device="cuda")
    indices, strengths = encoder.top_features(rows)
    buffer.scatter_reduce_(0, indices.reshape(-1).long(),
                           strengths.reshape(-1), reduce="amax")
    return buffer


def test_fused_maxima_match_the_torch_path():
    fused, plain = _encoders()
    hidden = torch.randn(200, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fused_buffer = _maxima(fused, hidden)
    plain_buffer = _maxima(plain, hidden)
    fused_set = set(fused_buffer.nonzero().flatten().tolist())
    plain_set = set(plain_buffer.nonzero().flatten().tolist())
    # over 200 tokens a few boundary features may differ per token
    assert len(fused_set ^ plain_set) <= 40
    shared = torch.tensor(sorted(fused_set & plain_set), device="cuda")
    torch.testing.assert_close(fused_buffer[shared], plain_buffer[shared],
                               rtol=2e-2, atol=1e-3)


def test_the_accumulator_splits_document_and_question_rows():
    from types import SimpleNamespace

    from quail.backends.quail.executor.sae import SaeMaxAccumulator

    _fused, plain = _encoders()
    document_tokens = {("r", 0): 6, ("r", 1): 4}
    accumulator = SaeMaxAccumulator(torch, plain,
                                    lambda key: document_tokens[key])
    accumulator.register_call()
    stream = torch.randn(14, HIDDEN, device="cuda", dtype=torch.bfloat16)
    chunk = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5],
                               device="cuda"),
        row_spans=((("r", 0), 0, 6, 8), (("r", 1), 8, 12, 14)))
    accumulator.observe(stream, chunk)
    features = SaeMaxAccumulator.features_of(accumulator.settle(("r", 0)))
    # the settled document's buffer was freed after the only registered
    # call; the other document's stays until its own settle
    assert ("r", 0) not in accumulator.document_max
    assert ("r", 1) in accumulator.document_max
    expected = _maxima(plain, stream[:8])
    kept = {item["feature"] for item in features}
    reference = set(expected.nonzero().flatten().tolist())
    assert kept == reference
