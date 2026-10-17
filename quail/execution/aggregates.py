"""Select numeric accumulators before Arrow groups aggregate inputs."""

import math
import sys
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from fractions import Fraction
from functools import partial

import pyarrow as pa
import pyarrow.compute as pc


def prepare_aggregate(values, kernel, *, grouped):
    """Return the aggregate's input, Arrow kernel, and optional finalizer.

    Args:
        values: Input values after dictionary decoding and key normalization.
        kernel: Arrow aggregate kernel name.
        grouped: Whether the aggregation has GROUP BY keys.

    Returns:
        An input array, a kernel name, and a callable on the aggregated
        column, or None when its values need no conversion. A None kernel
        computes a scalar directly through the callable.
    """
    if kernel not in ("sum", "mean"):
        return values, kernel, None
    dtype = values.type
    if pa.types.is_integer(dtype) and kernel == "sum":
        if dtype.bit_width + len(values).bit_length() <= 64:
            return values, kernel, None
        output = pa.uint64() if pa.types.is_unsigned_integer(dtype) else pa.int64()
        # An int128 accumulator fits int64/uint64 values times
        # Arrow's signed int64 maximum row count.
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
        reduce = partial(_decimal_aggregate, kernel=kernel, dtype=dtype)
    elif pa.types.is_float64(dtype):
        bounds = pc.min_max(values).as_py()
        maximum = max((abs(value) for value in bounds.values() if value is not None),
                      default=0.0)
        # Leave room for rounding during native floating-point accumulation.
        if maximum <= sys.float_info.max / 2 / max(1, len(values)):
            return values, kernel, None
        output = pa.float64()
        reduce = partial(_float_aggregate, kernel=kernel)
    else:
        return values, kernel, None

    def finish(column):
        groups = (item.values for item in column) if grouped else (values,)
        return pa.array([reduce(group.drop_null().to_pylist()) for group in groups],
                        type=output)

    return values, "list" if grouped else None, finish


def _decimal_aggregate(values, *, kernel, dtype):
    """Accumulate decimal values with enough digits for their exact total."""
    if not values:
        return None
    precision = dtype.precision + len(str(len(values))) + 2
    with localcontext(Context(prec=precision)):
        total = sum(values, Decimal(0))
        if kernel == "mean":
            unit = Decimal((0, (1,), -dtype.scale))
            total = (total / len(values)).quantize(unit, rounding=ROUND_HALF_UP)
        return total


def _float_aggregate(values, *, kernel):
    """Accumulate finite floats exactly and convert only the completed result."""
    if not values:
        return None
    special = [value for value in values if not math.isfinite(value)]
    if special:
        return sum(special)
    total = sum(map(Fraction, values), Fraction())
    return float(total / len(values) if kernel == "mean" else total)
