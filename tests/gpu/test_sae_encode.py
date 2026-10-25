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


def test_a_failed_feature_run_removes_its_forward_hook():
    from types import SimpleNamespace

    from quail.backends.quail.executor.operators.sae_features import SaeFeaturesPart
    from quail.physical import AiSaeFeatures, SaeFeaturesSpec

    pipeline = SimpleNamespace(forward_hook=None)
    state = SimpleNamespace(
        torch=torch, loaded_model=SimpleNamespace(pipeline=pipeline, sae=_encoder()))
    node = AiSaeFeatures(
        node_id="features", spec=SaeFeaturesSpec(
            aliases=("r",), name="features", pooling="max",
            query_template="{0}?", arguments=(("r", "body"),),
            expected_rows=1, estimated_seconds=0,
            prompt_token_parts=((), (1,))))
    part = SaeFeaturesPart(state, node, [0], {"r": [[0]]})
    SaeFeaturesPart.prepare_group([part])
    with pytest.raises(RuntimeError, match="interrupted"):
        with part.execution_context():
            raise RuntimeError("interrupted")
    assert pipeline.forward_hook is None


def test_repeated_questions_match_separate_feature_queries():
    import numpy as np
    import pyarrow as pa

    import quail

    config = quail.EngineConfig(model="qwen3.5-2b-sae-bf16", device="h100-sxm")
    with quail.Session(config) as session:
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": [0, 1],
            "body": ["This camera takes clear photos. " * 40 + ending
                     for ending in ("But the battery broke.", "I love it.")],
        }), id_col="id"))

        def collect(poolings):
            query = session.docs("reviews").alias("r")
            names = []
            for index, pooling in enumerate(poolings):
                name = f"features{index}"
                names.append(name)
                query = query.ai_sae_features(
                    quail.prompt("{0}\n\nWhat does the reviewer think of the "
                                 "battery, the photos, and the camera?",
                                 quail.col("r.body")), pooling=pooling, name=name)
            return query.select("r.id", *names).collect().sort_by("r.id")

        expected = {pooling: collect([pooling]).column("features0").to_pylist()
                    for pooling in ("last", "max")}
        comparisons = []
        for poolings in (("max", "max"), ("last", "max", "last", "max")):
            table = collect(poolings)
            assert table.column("r.id").to_pylist() == [0, 1]
            for index, pooling in enumerate(poolings):
                for actual, reference in zip(
                        table.column(f"features{index}").to_pylist(),
                        expected[pooling]):
                    actual = {item["feature"]: item["strength"] for item in actual}
                    reference = {item["feature"]: item["strength"]
                                 for item in reference}
                    common = actual.keys() & reference.keys()
                    features = actual.keys() | reference.keys()
                    overlap = len(common) / len(features)
                    expected_values = np.array([reference.get(key, 0)
                                                for key in features])
                    actual_values = np.array([actual.get(key, 0) for key in features])
                    relative_error = (np.linalg.norm(actual_values - expected_values)
                                      / np.linalg.norm(expected_values))
                    comparisons.append((poolings, index, overlap, relative_error))
        minimum_overlap = min(overlap for _, _, overlap, _ in comparisons)
        maximum_error = max(error for _, _, _, error in comparisons)
        # Prefill and retained-state paths round bf16 differently.
        assert minimum_overlap > .95 and maximum_error < .05, (
            minimum_overlap, maximum_error)
