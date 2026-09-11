"""Pair tables for joins with ordinary equality conditions.

A pair table has one int32 column per joined alias holding row indices
into that alias's document store. It lists exactly the (left, right)
rows whose key columns are equal, so the model is asked about those
pairs and no others.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

PAIRS_PREFIX = "pairs:"


def pairs_key(written_pos: int) -> str:
    """The request relation key of one join's pair table."""
    return f"{PAIRS_PREFIX}{written_pos}"


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
        left = pa.chunked_array([left]) if isinstance(left, pa.Array) \
            else left
        right = pa.chunked_array([right]) if isinstance(right, pa.Array) \
            else right
        if right.type != left.type:
            right = pc.cast(right, left.type)
        left_columns[name] = left
        right_columns[name] = right
    joined = pa.table(left_columns).join(
        pa.table(right_columns), keys=names, join_type="inner")
    return joined.select([left_alias, right_alias]).sort_by(
        [(left_alias, "ascending"), (right_alias, "ascending")])


def pair_fraction(pairs: pa.Table, n_left: int, n_right: int) -> float:
    """The pair count as a fraction of the cross product."""
    total = n_left * n_right
    return pairs.num_rows / total if total else 0.0


def partner_map(pairs: pa.Table, anchor_alias: str,
                partner_alias: str) -> dict[int, list[int]]:
    """Anchor row -> its partner rows, from a pair table."""
    out: dict[int, list[int]] = {}
    for anchor, partner in zip(pairs.column(anchor_alias).to_pylist(),
                               pairs.column(partner_alias).to_pylist()):
        out.setdefault(int(anchor), []).append(int(partner))
    return out
