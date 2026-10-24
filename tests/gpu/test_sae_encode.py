"""The SAE encode and the 'max' accumulator, on a GPU.

The encode runs its GEMM in bf16, and the reference runs it in fp32,
so near the top-k boundary a feature can swap with its neighbour. The
test checks that each row's feature sets differ only by features at
the top-k boundary, and that the shared features' strengths agree
within bf16 rounding. Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py`.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

NUM_FEATURES = 32_768
HIDDEN = 2_048
KEEP = 100


def _checkpoint_weight():
    """A W_enc laid out as the published checkpoint stores it.

    The checkpoint holds W_enc as a transposed view, strides
    (1, num_features). The feature scales spread over two decades, so
    the pre-activations are heavy-tailed like a real SAE's.
    """
    torch.manual_seed(7)
    scales = torch.logspace(-1.0, 1.0, NUM_FEATURES, device="cuda")
    weight = (torch.randn(HIDDEN, NUM_FEATURES, device="cuda")
              / HIDDEN ** 0.5 * scales[None, :]).to(torch.bfloat16).t()
    bias = torch.randn(NUM_FEATURES, device="cuda") * 0.01
    return weight, bias


def _encoder():
    import torch.nn.functional as functional

    from quail.backends.quail.executor.sae import SaeEncoder

    weight, bias = _checkpoint_weight()
    return SaeEncoder(torch, functional, weight, bias, KEEP)


def _reference(encoder, rows):
    """The fp32 encode of every row at once: (indices, strengths)."""
    pre = (rows.float() @ encoder.weight.float().T + encoder.bias)
    values, indices = torch.topk(pre.clamp_(min=0.0), KEEP, dim=1)
    return indices, values


def _maxima(encoder, rows):
    """Per-feature maxima over the rows, as the accumulator folds them."""
    buffer = torch.zeros(NUM_FEATURES, device="cuda")
    indices, strengths = encoder.top_features(rows)
    buffer.scatter_reduce_(0, indices.reshape(-1).long(),
                           strengths.reshape(-1), reduce="amax")
    return buffer


def test_the_blocked_encode_matches_an_fp32_reference():
    encoder = _encoder()
    assert not encoder.weight.is_contiguous()
    # 2,049 rows cross the 2,048-row block boundary
    hidden = torch.randn(2_049, HIDDEN, device="cuda", dtype=torch.bfloat16)
    indices, strengths = encoder.top_features(hidden)
    reference_indices, reference_strengths = _reference(encoder, hidden)
    assert indices.shape == strengths.shape == (2_049, KEEP)
    for row in range(hidden.shape[0]):
        found = dict(zip(indices[row].tolist(), strengths[row].tolist()))
        expected = dict(zip(reference_indices[row].tolist(),
                            reference_strengths[row].tolist()))
        cut = min(expected.values())
        for feature in found.keys() ^ expected.keys():
            value = found.get(feature, expected.get(feature))
            assert abs(value - cut) <= 5e-2 * max(1.0, abs(cut)), (
                f"row {row} feature {feature} is {value} against cut {cut}")
        for feature in found.keys() & expected.keys():
            assert abs(found[feature] - expected[feature]) <= (
                2e-2 * max(1.0, abs(expected[feature])))


def test_the_accumulator_splits_document_and_question_rows():
    from types import SimpleNamespace

    from quail.backends.quail.executor.sae import SaeMaxAccumulator

    encoder = _encoder()
    document_tokens = {("r", 0): 6, ("r", 1): 4}
    accumulator = SaeMaxAccumulator(torch, encoder,
                                    lambda key: document_tokens[key])
    accumulator.register_call()
    stream = torch.randn(14, HIDDEN, device="cuda", dtype=torch.bfloat16)
    chunk = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5],
                               device="cuda"),
        row_spans=((("r", 0), 0, 6, 8), (("r", 1), 8, 12, 14)))
    accumulator.observe(stream, chunk)
    handle = accumulator.settle(("r", 0))
    # the settled document's slot was freed after the only registered
    # call; the other document's stays until its own settle
    assert ("r", 0) not in accumulator.document_slot
    assert ("r", 1) in accumulator.document_slot
    # the slab is still filling, so a non-waiting harvest reads nothing
    assert accumulator.harvest([handle], completed_only=True)[0].size == 0
    taken, lengths, indices, _strengths = accumulator.harvest(
        [handle], completed_only=False)
    assert taken.tolist() == [0] and lengths.tolist() == [len(indices)]
    expected = _maxima(encoder, stream[:8])
    assert set(indices.tolist()) == set(expected.nonzero().flatten().tolist())
