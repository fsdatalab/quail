"""Pair tables for joins with ordinary equality conditions.

A pair table has one int32 column per joined alias holding row indices
into that alias's document store. It lists exactly the (left, right)
rows whose key columns are equal, so the model is asked about those
pairs and no others. The HashJoin node builds it at run time from the
key columns the request carries as ``columns:<alias>`` relations.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

COLUMNS_PREFIX = "columns:"


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
    """Build an int32 pair table from row-id tuples."""
    rows = list(pairs)
    return pa.table({
        left_alias: pa.array([left for left, _ in rows], type=pa.int32()),
        right_alias: pa.array([right for _, right in rows], type=pa.int32()),
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
    return pairs / (n_left * n_right)


def partner_map(pairs: pa.Table, anchor_alias: str,
                partner_alias: str) -> dict[int, list[int]]:
    """Anchor row -> its partner rows, from a pair table."""
    out: dict[int, list[int]] = {}
    for anchor, partner in zip(pairs.column(anchor_alias).to_pylist(),
                               pairs.column(partner_alias).to_pylist()):
        out.setdefault(int(anchor), []).append(int(partner))
    return out


def members_by_partner(members, position: int) -> dict:
    """Partner row -> indices of the member tuples that hold it."""
    by_partner = {}
    for index, member in enumerate(members):
        by_partner.setdefault(int(member[position]), []).append(index)
    return by_partner


def allowed_members(rows: dict, by_partner: dict, anchor) -> list:
    """The sorted member indices one anchor's pair rows allow."""
    return sorted(index for partner in rows.get(int(anchor), ())
                  for index in by_partner.get(int(partner), ()))
