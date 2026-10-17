"""Numeric aggregates keep valid answers through large intermediate totals."""

import math
import sys
from decimal import ROUND_HALF_UP, Decimal, Inexact, localcontext
from fractions import Fraction

import pyarrow as pa
import pytest

from quail.execution.runtimes.relational import aggregate_table
from quail.physical import Aggregate
from quail.physical.base import input_ports


def _aggregate(values, dtype, kernel, grouped):
    table = pa.table({"key": ["a"] * len(values), "value": pa.array(values, dtype)})
    node = Aggregate(node_id="aggregate", inputs=input_ports(()),
                     keys=("key",) if grouped else (),
                     aggregates=(("answer", kernel, "value"),), columns=("answer",))
    return aggregate_table(table, node).column("answer")


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [pa.decimal128(38, 0), pa.decimal256(76, 0)])
def test_decimal_average_does_not_overflow_its_intermediate_sum(dtype, grouped):
    value = Decimal("9" * dtype.precision)
    result = _aggregate([value] * 10 + [None], dtype, "avg", grouped)
    assert result.type == dtype
    assert result.to_pylist() == [value]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [pa.decimal128(38, 0), pa.decimal256(76, 0)])
def test_decimal_sum_rejects_an_unrepresentable_result(dtype, grouped):
    value = Decimal("9" * dtype.precision)
    with pytest.raises(pa.ArrowInvalid):
        _aggregate([value] * 10, dtype, "sum", grouped)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [pa.decimal128(38, 2), pa.decimal256(76, 2)])
@pytest.mark.parametrize("kernel", ["sum", "avg"])
def test_decimal_cancellation_is_exact_and_independent_of_python_context(
        dtype, kernel, grouped):
    value = Decimal("9" * (dtype.precision - 2) + ".99")
    values = [value, value, value.copy_negate(), None]
    with localcontext() as context:
        context.prec = 100
        expected = (value if kernel == "sum" else (value / 3).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP))
    with localcontext() as context:
        context.prec = 6
        context.traps[Inexact] = True
        result = _aggregate(values, dtype, kernel, grouped)
    assert result.to_pylist() == [expected]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("values", [
    [1e308, 1e308, None], [-1e308, -1e308],
    [1e308, 1e308, -1e308], [1e308, 1e-300, -1e308],
    [sys.float_info.max / 3] * 3, [sys.float_info.max / 9] * 9,
])
def test_float_average_avoids_intermediate_overflow(values, grouped):
    present = [value for value in values if value is not None]
    expected = float(sum(map(Fraction, present)) / len(present))
    assert _aggregate(values, pa.float64(), "avg", grouped).to_pylist() == [expected]


@pytest.mark.parametrize("grouped", [False, True])
def test_float_sum_finishes_cancellation_before_converting_to_float(grouped):
    values = [1e308, 1e308, -1e308, None]
    assert _aggregate(values, pa.float64(), "sum", grouped).to_pylist() == [1e308]
    with pytest.raises(OverflowError):
        _aggregate([1e308, 1e308], pa.float64(), "sum", grouped)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("kernel", ["sum", "avg"])
@pytest.mark.parametrize("values,expected", [
    ([1e308, float("inf"), None], float("inf")),
    ([1e308, -float("inf")], -float("inf")),
    ([float("inf"), -float("inf")], float("nan")),
    ([1e308, float("nan")], float("nan")),
])
def test_float_aggregates_preserve_nonfinite_inputs(values, expected, kernel, grouped):
    (actual,) = _aggregate(values, pa.float64(), kernel, grouped).to_pylist()
    assert math.isnan(actual) if math.isnan(expected) else actual == expected


@pytest.mark.parametrize("dtype", [pa.decimal256(76, 2), pa.float64()])
@pytest.mark.parametrize("kernel", ["sum", "avg"])
def test_numeric_aggregates_preserve_empty_and_null_groups(dtype, kernel):
    assert _aggregate([], dtype, kernel, False).to_pylist() == [None]
    assert _aggregate([], dtype, kernel, True).to_pylist() == []
    assert _aggregate([None] * 10, dtype, kernel, False).to_pylist() == [None]
    assert _aggregate([None] * 10, dtype, kernel, True).to_pylist() == [None]


@pytest.mark.parametrize("dtype", [pa.decimal256(76, 0), pa.float64()])
@pytest.mark.parametrize("grouped", [False, True])
def test_wide_aggregates_share_chunked_inputs_with_other_kernels(dtype, grouped):
    decimal = pa.types.is_decimal(dtype)
    large = Decimal("8" * 76) if decimal else 1e308
    negative = large.copy_negate() if decimal else -large
    small = [Decimal(1), Decimal(2)] if decimal else [1.0, 2.0]
    values = [large, small[0], large, None, small[1], negative]
    column = pa.chunked_array([pa.array(values[:2], dtype),
                               pa.array(values[2:4], dtype),
                               pa.array(values[4:], dtype)])
    table = pa.table({"key": ["big", "small", "big", "null", "small", "big"],
                      "value": column})
    node = Aggregate(node_id="aggregate", inputs=input_ports(()),
                     keys=("key",) if grouped else (),
                     aggregates=(("total", "sum", "value"),
                                 ("again", "sum", "value"),
                                 ("average", "avg", "value"),
                                 ("n", "count", "value")),
                     columns=(*(("key",) if grouped else ()),
                              "total", "again", "average", "n"))
    result = aggregate_table(table, node)
    with localcontext() as context:
        context.prec = 100
        big_mean = ((large / 3).quantize(Decimal(1), rounding=ROUND_HALF_UP)
                    if decimal else float(Fraction(large) / 3))
        total = large + 3 if decimal else float(Fraction(large) + 3)
        average = ((total / 5).quantize(Decimal(1), rounding=ROUND_HALF_UP)
                   if decimal else float((Fraction(large) + 3) / 5))
    expected = [
        {"key": "big", "total": large, "again": large, "average": big_mean, "n": 3},
        {"key": "small", "total": 3, "again": 3,
         "average": Decimal(2) if decimal else 1.5, "n": 2},
        {"key": "null", "total": None, "again": None, "average": None, "n": 0}]
    assert result.to_pylist() == (expected if grouped else [
        {"total": total, "again": total, "average": average, "n": 5}])


@pytest.mark.parametrize("dtype", [pa.decimal32(9, 2), pa.decimal64(18, 2),
                                  pa.decimal128(38, 2), pa.decimal256(76, 2)])
@pytest.mark.parametrize("negative", [False, True])
def test_decimal_average_preserves_scale_and_rounds_ties_away_from_zero(
        dtype, negative):
    values = [Decimal("1.00"), Decimal("2.01")]
    expected = Decimal("1.51")
    if negative:
        values = [value.copy_negate() for value in values]
        expected = expected.copy_negate()
    result = _aggregate(values, dtype, "avg", True)
    assert result.type == dtype
    assert result.to_pylist() == [expected]
