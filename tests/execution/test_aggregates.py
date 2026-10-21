"""SUM and AVG keep an exact total through wide intermediate values."""

import math
import sys
from decimal import ROUND_HALF_UP, Decimal, localcontext
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
def test_a_decimal_average_keeps_its_scale_and_a_sum_refuses_overflow(dtype, grouped):
    value = Decimal("9" * dtype.precision)
    result = _aggregate([value] * 10 + [None], dtype, "avg", grouped)
    assert (result.type, result.to_pylist()) == (dtype, [value])
    with pytest.raises(pa.ArrowInvalid):
        _aggregate([value] * 10, dtype, "sum", grouped)
    with localcontext() as context:
        context.prec = 100
        third = (Decimal("1.00") + Decimal("2.01")) / 2
    assert _aggregate([Decimal("1.00"), Decimal("2.01")], pa.decimal128(38, 2),
                      "avg", grouped).to_pylist() == [
        third.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("values", [
    [1e308, 1e308, None], [1e308, 1e308, -1e308], [sys.float_info.max / 3] * 3])
def test_a_float_total_is_finished_before_it_is_rounded(values, grouped):
    present = [value for value in values if value is not None]
    expected = float(sum(map(Fraction, present)) / len(present))
    assert _aggregate(values, pa.float64(), "avg", grouped).to_pylist() == [expected]
    assert _aggregate([1e308, 1e308, -1e308, None], pa.float64(), "sum",
                      grouped).to_pylist() == [1e308]
    with pytest.raises(OverflowError):
        _aggregate([1e308, 1e308], pa.float64(), "sum", grouped)
    (total,) = _aggregate([1e308, float("nan")], pa.float64(), "sum",
                          grouped).to_pylist()
    assert math.isnan(total)
