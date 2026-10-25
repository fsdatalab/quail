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


@pytest.mark.parametrize("poolings,questions", [
    (("last", "max"), ((1,), (2,))),
    (("max", "last", "max"), ((1,), (2,), (3,))),
    (("max", "last"), ((1,), (2,))),
    (("max", "max"), ((1, 2), (1, 2))),
])
def test_max_pooling_reads_its_own_question_tokens(poolings, questions):
    from contextlib import ExitStack
    from types import SimpleNamespace

    import torch.nn.functional as functional

    from quail.backends.quail.executor.operators.sae_features import SaeFeaturesPart
    from quail.backends.quail.executor.parts import gated_stages
    from quail.backends.quail.executor.sae import SaeEncoder
    from quail.backends.quail.executor.stages import frame_writes
    from quail.physical import AiSaeFeatures, SaeFeaturesSpec

    width = len(poolings) + 1
    encoder = SaeEncoder(
        torch, functional, torch.eye(width, device="cuda", dtype=torch.bfloat16),
        torch.zeros(width, device="cuda"), 1)
    pipeline = SimpleNamespace(forward_hook=None)
    state = SimpleNamespace(
        torch=torch, loaded_model=SimpleNamespace(pipeline=pipeline, sae=encoder))
    parts = []
    for index, pooling in enumerate(poolings):
        node = AiSaeFeatures(
            node_id=f"features{index}", spec=SaeFeaturesSpec(
                aliases=("r",), name=f"features{index}", pooling=pooling,
                query_template="{0}?", arguments=(("r", "body"),),
                expected_rows=1, estimated_seconds=0,
                prompt_token_parts=((), questions[index])))
        accumulator = next((part.accumulator for part in parts
                            if part.accumulator is not None), None)
        parts.append(SaeFeaturesPart(state, node, [0], {"r": [[0]]},
                                     accumulator=accumulator))
    stages = gated_stages(parts)
    with ExitStack() as resources:
        for part in parts:
            resources.enter_context(part.execution_context())
        for index, (stage, writes_frame) in enumerate(zip(stages,
                                                         frame_writes(stages))):
            rows = ([0] if index == 0 else [])
            rows += stage.frame if writes_frame else []
            rows += stage.suffixes[0]
            stream = encoder.weight[rows] * 3
            chunk = SimpleNamespace(
                positions=torch.arange(index != 0, len(rows) + (index != 0),
                                       device="cuda"),
                row_spans=((("r", 0), 0, int(index == 0), len(rows)),))
            pipeline.forward_hook(stream, chunk)
            stage.decide(0, stage.readout.result(stage.readout.submit(stream[-1:])))
    assert pipeline.forward_hook is None
    for index, part in enumerate(parts):
        result = part.result(0, 0, 0, {})
        listed = result.outputs[part.node.rows_port]
        features = listed.values.column(part.node.spec.name)[0].as_py()
        expected = ({0, *questions[index]} if poolings[index] == "max"
                    else {questions[index][-1]})
        assert {item["feature"] for item in features} == expected
        assert all(item["strength"] == 3 for item in features)
    assert not parts[0].accumulator.document_slot
    assert not parts[0].accumulator.tail_slot


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
