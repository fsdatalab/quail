"""Accumulators for SUM and AVG that keep an exact total.

Arrow sums an integer column in its own width and wraps on overflow,
and sums decimals and floats in their own width, so a total that fits
the output type can still overflow on the way. Each aggregate here
picks a wider accumulator, or reduces the values in Python when no
Arrow type is wide enough.
"""

import math
import sys
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from fractions import Fraction
from functools import partial

import pyarrow as pa
import pyarrow.compute as pc


def prepare_aggregate(values, kernel: str, *, grouped: bool):
    """Choose how one aggregate accumulates its values.

    Args:
        values: The column aggregated, dictionary-decoded.
        kernel: The Arrow kernel asked for: sum, mean, or another.
        grouped: Whether the aggregation has GROUP BY keys.

    Returns:
        The values to hand Arrow, the kernel to run over them, and a
        function finishing the aggregated column, or None when Arrow's
        result is final. A None kernel means the function computes the
        whole aggregate itself from the values.
    """
    if kernel not in ("sum", "mean"):
        return values, kernel, None
    dtype = values.type
    if pa.types.is_integer(dtype) and kernel == "sum":
        if dtype.bit_width + len(values).bit_length() <= 64:
            return values, kernel, None
        output = pa.uint64() if pa.types.is_unsigned_integer(dtype) else pa.int64()
        # 38 digits hold any int64 times Arrow's largest row count
        return (values.cast(pa.decimal128(38, 0)), kernel,
                partial(pc.cast, target_type=output))
    if pa.types.is_decimal(dtype):
        output = dtype
        if kernel == "sum":
            output = (pa.decimal256(76, dtype.scale) if dtype.bit_width == 256
                      else pa.decimal128(38, dtype.scale))
        precision = dtype.precision + len(str(max(1, len(values))))
        if precision <= 76:
            accumulator = (pa.decimal128(38, dtype.scale) if precision <= 38
                           else pa.decimal256(76, dtype.scale))
            return (values.cast(accumulator), kernel,
                    partial(pc.cast, target_type=output))
        reduce = partial(_decimal_total, kernel=kernel, dtype=dtype)
    elif pa.types.is_float64(dtype):
        bounds = pc.min_max(values).as_py()
        largest = max((abs(value) for value in bounds.values() if value is not None),
                      default=0.0)
        # halved: a float sum may round up on the way to its total
        if largest <= sys.float_info.max / 2 / max(1, len(values)):
            return values, kernel, None
        output = pa.float64()
        reduce = partial(_float_total, kernel=kernel)
    else:
        return values, kernel, None

    def finish(column):
        groups = (item.values for item in column) if grouped else (values,)
        return pa.array([reduce(group.drop_null().to_pylist()) for group in groups],
                        type=output)

    return values, "list" if grouped else None, finish


def _decimal_total(values, *, kernel: str, dtype):
    """Sum decimals with enough digits for the exact total."""
    if not values:
        return None
    precision = dtype.precision + len(str(len(values))) + 2
    with localcontext(Context(prec=precision)):
        total = sum(values, Decimal(0))
        if kernel == "mean":
            unit = Decimal((0, (1,), -dtype.scale))
            total = (total / len(values)).quantize(unit, rounding=ROUND_HALF_UP)
        return total


def _float_total(values, *, kernel: str):
    """Sum finite floats exactly and round only the finished total."""
    if not values:
        return None
    special = [value for value in values if not math.isfinite(value)]
    if special:
        return sum(special)
    total = sum(map(Fraction, values), Fraction())
    return float(total / len(values) if kernel == "mean" else total)
