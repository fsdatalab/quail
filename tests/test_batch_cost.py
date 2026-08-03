import pytest

from docengine.runtime.batch_cost import (
    AttentionShapePoint,
    AttentionShapeTable,
    BatchFeatures,
    MeasuredBatchTimeEstimator,
    PrimitivePoint,
    PrimitiveTimeTable,
    SharedPrefixGroup,
    ValidationRow,
    summarize_validation,
)


def table(scale=10):
    return PrimitiveTimeTable([
        PrimitivePoint(10, 10 * scale),
        PrimitivePoint(20, 15 * scale),
        PrimitivePoint(40, 25 * scale),
    ])


def estimator():
    return MeasuredBatchTimeEstimator(
        dense=table(10),
        standard_attention=table(2),
        cascade_attention=table(1),
        output=table(3),
        kv_transfer=table(4),
        fixed_ns=7,
    )


def test_piecewise_time_table_interpolates_and_extrapolates():
    measured = table(10)
    assert measured.estimate_ns(5) == 50
    assert measured.estimate_ns(15) == 125
    assert measured.estimate_ns(80) == 500


def test_time_table_rejects_nonmonotone_measurements():
    with pytest.raises(ValueError):
        PrimitiveTimeTable([
            PrimitivePoint(10, 20),
            PrimitivePoint(20, 10),
        ])


def test_batch_features_use_exact_lengths_and_shared_groups():
    features = BatchFeatures(
        new_tokens=(3, 5),
        cached_prefix_tokens=(10, 20),
        standard_query_tokens=(3, 5),
        output_tokens=2,
        shared_prefix_groups=(
            SharedPrefixGroup(prefix_tokens=30, query_tokens=(2, 2)),
        ),
        kv_transfer_bytes=4,
    )
    assert features.total_new_tokens == 8
    assert features.standard_attention_pairs == 151
    assert features.cascade_attention_pairs == 126


def test_estimator_sums_measured_primitive_groups():
    features = BatchFeatures(
        new_tokens=(5, 5),
        cached_prefix_tokens=(0, 0),
        output_tokens=10,
        kv_transfer_bytes=10,
    )
    estimate = estimator().estimate(features)
    assert estimate.dense_ns == 100
    assert estimate.output_ns == 30
    assert estimate.kv_transfer_ns == 40
    assert estimate.standard_attention_ns == 40
    assert estimate.total_ns == 217


def test_estimator_round_trips_json(tmp_path):
    path = tmp_path / "catalog.json"
    estimator().save(path)
    restored = MeasuredBatchTimeEstimator.load(path)
    features = BatchFeatures(
        new_tokens=(10,),
        cached_prefix_tokens=(0,),
        output_tokens=0,
    )
    assert restored.estimate(features) == estimator().estimate(features)


def test_validation_gate():
    summary = summarize_validation([
        ValidationRow(100, 100, "a"),
        ValidationRow(104, 100, "b"),
        ValidationRow(95, 100, "c"),
        ValidationRow(108, 100, "d"),
    ])
    assert summary.median_absolute_error == pytest.approx(0.045)
    assert summary.p95_absolute_error == pytest.approx(0.08)
    assert summary.passes


def test_attention_shape_table_keeps_k_specific_curves():
    table = AttentionShapeTable([
        AttentionShapePoint(8, 2, 300, 32, 100),
        AttentionShapePoint(8, 2, 3000, 32, 200),
        AttentionShapePoint(8, 4, 300, 32, 80),
        AttentionShapePoint(8, 4, 3000, 32, 160),
    ])
    assert table.estimate_ns(
        groups=8,
        k=2,
        prefix_tokens=300,
        tail_tokens=32,
    ) == 100
    assert table.estimate_ns(
        groups=8,
        k=4,
        prefix_tokens=300,
        tail_tokens=32,
    ) == 80
    assert 100 < table.estimate_ns(
        groups=8,
        k=2,
        prefix_tokens=1500,
        tail_tokens=32,
    ) < 200


def test_attention_shape_table_round_trips_rows():
    rows = [{
        "groups": 4,
        "k": 2,
        "prefix_tokens": 512,
        "tail_tokens": 32,
        "time_ns": 1000,
    }]
    table = AttentionShapeTable.from_rows(rows)
    assert table.to_rows() == rows


def test_attention_shape_table_interpolates_group_count():
    table = AttentionShapeTable([
        AttentionShapePoint(4, 2, 1000, 32, 100),
        AttentionShapePoint(4, 2, 2000, 32, 200),
        AttentionShapePoint(16, 2, 1000, 32, 300),
        AttentionShapePoint(16, 2, 2000, 32, 500),
    ])
    assert table.estimate_ns(
        groups=10,
        k=2,
        prefix_tokens=1000,
        tail_tokens=32,
    ) == 200
    assert table.estimate_ns(
        groups=10,
        k=2,
        prefix_tokens=2000,
        tail_tokens=32,
    ) == 350


def test_attention_shape_table_interpolates_tail_length():
    table = AttentionShapeTable([
        AttentionShapePoint(8, 4, 1000, 16, 100),
        AttentionShapePoint(8, 4, 1000, 48, 300),
    ])
    assert table.estimate_ns(
        groups=8,
        k=4,
        prefix_tokens=1000,
        tail_tokens=32,
    ) == 200
