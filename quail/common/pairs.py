"""Pair tables for joins with ordinary conditions.

A pair table has one int32 column per joined alias holding row indices
into that alias's document store. It lists exactly the (left, right)
rows whose key columns are equal, or every pair when the join has no
keys, where the join's filter is TRUE, so the model is asked about those
pairs and no others. The HashJoin node builds it at run time from the
columns the request carries as ``columns:<alias>`` relations.
"""

from __future__ import annotations

import _quail_native
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.common.expressions import accepts
from quail.common.selection import equality_keys

COLUMNS_PREFIX = "columns:"
# the value column a deduplicated table carries: each row's first row
# with the same tokens and the same deduplicate setting columns
FIRST_DOCUMENT = "__first_document"
# the value column that is TRUE for each document of a scan that is
# NULL, present only when one is; Rust's planner writes it
NULL_DOCUMENT = _quail_native.NULL_DOCUMENT
# the value column of a deduplicated rows input to an AI filter with a
# stop key: the documents each key document stands for
COPIES = "__copies"
# the most candidate pairs a join's filter tests at once, which bounds
# the memory of a join without keys to its pairs that pass
FILTER_BATCH_PAIRS = 1 << 20


def pair_keys(anchors, partners):
    """Encode document ID pairs as int64 keys or an Arrow expression."""
    keys = pc.add(pc.shift_left(anchors.cast(pa.int64()), 32),
                  partners.cast(pa.int64()))
    return keys.combine_chunks() if isinstance(keys, pa.ChunkedArray) else keys


def columns_key(alias: str) -> str:
    """The request relation key of one alias's value table."""
    return f"{COLUMNS_PREFIX}{alias}"


def pair_table(left_alias: str, left_keys, right_alias: str,
               right_keys) -> pa.Table:
    """Row index pairs whose key columns are all equal.

    Args:
        left_alias: Alias of the left table; names its index column.
        left_keys: Its key columns, one Arrow array per equality.
        right_alias: Alias of the right table.
        right_keys: Its key columns, in the same equality order.

    Returns:
        A table with columns (left_alias, right_alias), sorted by both.
    """
    left, right = _join_key_tables(left_keys, right_keys)
    names = left.column_names
    left = left.append_column("left", pa.array(range(len(left)), pa.int32()))
    right = right.append_column("right", pa.array(range(len(right)), pa.int32()))
    joined = left.join(right, keys=names, join_type="inner")
    return joined.select(["left", "right"]).sort_by(
        [("left", "ascending"), ("right", "ascending")]).rename_columns(
            [left_alias, right_alias])


def join_pairs(left_alias: str, left_keys, right_alias: str, right_keys, *,
               join_filter=None, values=None) -> pa.Table:
    """Row index pairs whose keys are equal and where the filter is TRUE.

    For example, ``r JOIN t ON r.n < t.n`` has no keys and the filter
    ``r.n < t.n``: the pairs are those of every r row and t row where
    the r row's n is less than the t row's.

    Args:
        left_alias: Alias of the left table; names its index column.
        left_keys: Its key columns, one Arrow array per equality; empty
            for a join without keys, which pairs every row.
        right_alias: Alias of the right table.
        right_keys: Its key columns, in the same equality order.
        join_filter: The join's other conditions, an Expression over
            both tables' columns, or None.
        values: alias -> the columns the filter reads, one row per key
            row, as ``filter_columns`` returns them.

    Returns:
        A table with columns (left_alias, right_alias), sorted by both.

    Raises:
        ValueError: The join has neither keys nor a filter.
    """
    if left_keys:
        candidates = pair_table(left_alias, left_keys, right_alias, right_keys)
        if join_filter is None:
            return candidates
        batches = (candidates.slice(start, FILTER_BATCH_PAIRS)
                   for start in range(0, candidates.num_rows, FILTER_BATCH_PAIRS))
    elif join_filter is None:
        raise ValueError("a join needs an equality or a filter")
    else:
        batches = _cross_pairs(left_alias, values[left_alias].num_rows,
                               right_alias, values[right_alias].num_rows)
    kept = [batch.filter(_passes(join_filter, batch, values)) for batch in batches]
    if not kept:
        return pa.table({left_alias: pa.array([], pa.int32()),
                         right_alias: pa.array([], pa.int32())})
    return pa.concat_tables(kept).combine_chunks()


def filter_columns(join_filter, alias: str, columns) -> pa.Table:
    """The columns of one side a join's filter reads.

    A filter of two tables reads at least one column of each.

    Args:
        join_filter: The join's filter, an Expression.
        alias: The side's table alias.
        columns: The side's columns, as anything with ``column(name)``.
    """
    names = sorted({ref.column for ref in join_filter.references
                    if ref.alias == alias})
    return pa.table({name: columns.column(name) for name in names})


def _cross_pairs(left_alias: str, num_left: int, right_alias: str, num_right: int):
    """Yield every (left, right) row pair, left-major, in bounded batches."""
    block = max(1, FILTER_BATCH_PAIRS // max(1, num_right))
    right = np.arange(num_right, dtype=np.int32)
    for start in range(0, num_left, block):
        left = np.arange(start, min(start + block, num_left), dtype=np.int32)
        yield pa.table({left_alias: np.repeat(left, num_right),
                        right_alias: np.tile(right, len(left))})


def _passes(join_filter, pairs: pa.Table, values) -> pa.Array:
    """Whether the filter is TRUE for each pair, with NULL rejected."""
    columns = {}
    for ref in join_filter.references:
        name = f"{ref.alias}.{ref.column}"
        if name not in columns:
            found = values[ref.alias].column(ref.column)
            columns[name] = pc.take(found, pairs.column(ref.alias))
    return accepts(join_filter, pa.table(columns), qualified=True)


def _join_key_tables(left_keys, right_keys) -> tuple[pa.Table, pa.Table]:
    """Build both sides' key tables, with one type per equality."""
    if len(left_keys) != len(right_keys) or not left_keys:
        raise ValueError("a join needs one key column per side "
                         "and at least one equality")
    names = [f"key{index}" for index in range(len(left_keys))]
    columns = [_join_key_columns(left, right)
               for left, right in zip(left_keys, right_keys)]
    return tuple(pa.Table.from_arrays([pair[side] for pair in columns], names=names)
                 for side in (0, 1))


def _join_key_columns(left, right):
    """Return both key columns in one type, equal only for equal values.

    A float joins an integer only at an integral value; a uint64 joins
    a signed integer as a decimal, so neither side wraps.
    """
    left, right = (equality_keys(column, nan_equal=False) for column in (left, right))
    types = (left.type, right.type)
    if (any(pa.types.is_integer(kind) for kind in types)
            and any(pa.types.is_floating(kind) for kind in types)):
        return _integral_join_key(left), _integral_join_key(right)
    if pa.uint64() in types and any(pa.types.is_signed_integer(kind) for kind in types):
        kind = pa.decimal128(20, 0)
        return left.cast(kind), right.cast(kind)
    schema = pa.unify_schemas([pa.schema([("key", kind)]) for kind in types],
                              promote_options="permissive")
    kind = schema.field(0).type
    # Arrow's hash join needs a concrete key type when every value is null
    kind = pa.bool_() if pa.types.is_null(kind) else kind
    return left.cast(kind), right.cast(kind)


def _integral_join_key(column):
    """Return numbers as decimals; a float that is no 64-bit integer becomes null."""
    if pa.types.is_floating(column.type):
        integral = pc.equal(column, pc.floor(column))
        in_range = pc.and_kleene(pc.greater_equal(column, float(-2**63)),
                                 pc.less(column, float(2**64)))
        column = pc.if_else(pc.and_kleene(integral, in_range), column, None)
    return column.cast(pa.decimal128(20, 0))


def pair_ids_table(left_alias: str, right_alias: str, pairs) -> pa.Table:
    """Build an int32 pair table from an array of (left, right) id rows."""
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pa.table({
        left_alias: pa.array(rows[:, 0], type=pa.int32()),
        right_alias: pa.array(rows[:, 1], type=pa.int32()),
    })


def pair_partner(anchor: str, partners) -> str:
    """The partner alias a pair stage pairs with its anchor."""
    if len(partners) != 1:
        raise ValueError(
            f"a join over pairs relates the anchor {anchor!r} to one "
            f"partner, not {list(partners)}")
    return partners[0]


class RowGroups:
    """Row IDs grouped by a sorted key column, preserving order within each key."""

    def __init__(self, keys, rows):
        keys = np.asarray(keys, dtype=np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        if not len(keys):
            starts = np.empty(0, dtype=np.int64)
        self.keys = sorted_keys[starts]
        self.offsets = np.r_[starts, len(keys)]
        self.rows = np.asarray(rows, dtype=np.int64)[order]

    def get(self, key, default=()):
        """Return a view of one key's rows, or the supplied default."""
        position = np.searchsorted(self.keys, key)
        if position == len(self.keys) or self.keys[position] != key:
            return default
        return self.rows[self.offsets[position]:self.offsets[position + 1]]

    def take(self, keys):
        """Return the rows for each supplied key, including repeated keys."""
        keys = np.asarray(keys, dtype=np.int64)
        positions = np.searchsorted(self.keys, keys)
        valid = positions < len(self.keys)
        positions, keys = positions[valid], keys[valid]
        positions = positions[self.keys[positions] == keys]
        starts = self.offsets[positions]
        counts = self.offsets[positions + 1] - starts
        ends = np.cumsum(counts)
        indices = np.repeat(starts - ends + counts, counts)
        indices += np.arange(ends[-1] if len(ends) else 0)
        return self.rows[indices]


def partner_map(pairs: pa.Table, anchor_alias: str,
                partner_alias: str) -> RowGroups:
    """Anchor row -> its partner rows, from a pair table."""
    return RowGroups(pairs.column(anchor_alias).to_numpy(),
                     pairs.column(partner_alias).to_numpy())


def members_by_partner(members, position: int) -> RowGroups:
    """Partner row -> indices of the member tuples that hold it."""
    keys = np.asarray(members, dtype=np.int64)[:, position] if len(members) else ()
    return RowGroups(keys, np.arange(len(members)))


def allowed_members(rows, by_partner: RowGroups, anchor) -> np.ndarray:
    """The sorted member indices one anchor's pair rows allow."""
    return np.sort(by_partner.take(rows.get(int(anchor), ())))
