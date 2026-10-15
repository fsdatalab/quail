"""Pair tables for joins with ordinary equality conditions.

A pair table has one int32 column per joined alias holding row indices
into that alias's document store. It lists exactly the (left, right)
rows whose key columns are equal, so the model is asked about those
pairs and no others. The HashJoin node builds it at run time from the
key columns the request carries as ``columns:<alias>`` relations.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

COLUMNS_PREFIX = "columns:"


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
    if len(left_keys) != len(right_keys) or not left_keys:
        raise ValueError("a pair table needs one key column per side "
                         "and at least one equality")
    n_left = len(left_keys[0])
    n_right = len(right_keys[0])
    names = [f"key{index}" for index in range(len(left_keys))]
    left_columns = {left_alias: pa.array(range(n_left), type=pa.int32())}
    right_columns = {right_alias: pa.array(range(n_right), type=pa.int32())}
    for name, left, right in zip(names, left_keys, right_keys):
        if right.type != left.type:
            right = pc.cast(right, left.type)
        left_columns[name] = left
        right_columns[name] = right
    joined = pa.table(left_columns).join(
        pa.table(right_columns), keys=names, join_type="inner")
    return joined.select([left_alias, right_alias]).sort_by(
        [(left_alias, "ascending"), (right_alias, "ascending")])


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


def estimate_pair_fraction(left_keys, right_keys) -> float:
    """Estimate the equality pairs over the cross product from key samples.

    Each side is a list of key arrays sampled from its table; a row's
    key is its values across them, and a null never matches. The
    estimate is the sum, over the keys seen on both sides, of the
    product of the two sides' value counts, over the product of the
    sample sizes.
    """
    if len(left_keys) != len(right_keys) or not left_keys:
        raise ValueError("a pair estimate needs one key column per side "
                         "and at least one equality")
    names = [f"key{index}" for index in range(len(left_keys))]
    n_left, n_right = len(left_keys[0]), len(right_keys[0])
    if not n_left or not n_right:
        return 0.0

    def value_counts(columns, count):
        table = pa.table(dict(zip(names, columns))).drop_null()
        return table.group_by(names).aggregate(
            [([], "count_all")]).rename_columns([*names, count])

    # integer keys on one side are cast to the other side's type
    right_keys = [right if right.type == left.type else pc.cast(right, left.type)
                  for left, right in zip(left_keys, right_keys)]
    matched = value_counts(left_keys, "left").join(
        value_counts(right_keys, "right"), keys=names, join_type="inner")
    pairs = pc.sum(pc.multiply(matched.column("left"),
                               matched.column("right"))).as_py() or 0
    # samples that share no key still say little about the tables, so
    # the estimate stays off zero
    return max(pairs, 1) / (n_left * n_right)


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
