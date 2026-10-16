"""Select the rows of a scan that pass its column predicates."""

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.predicates import accepted_rows


def selected_rows(columns: dict, predicates) -> pa.Array:
    """Return the positions of the rows that pass every predicate.

    Args:
        columns: Each predicate's column values, keyed by column name.
        predicates: The RegularPredicate tests, combined with AND.
    """
    mask = None
    for predicate in predicates:
        test = predicate.evaluate(columns[predicate.column.column])
        mask = test if mask is None else pc.and_kleene(mask, test)
    return pc.indices_nonzero(accepted_rows(mask))


def document_positions(ids) -> np.ndarray:
    """Map document IDs to positions in a selection; absent IDs map to -1."""
    ids = np.asarray(ids, dtype=np.int64)
    positions = np.full(int(ids.max()) + 1 if len(ids) else 0, -1, dtype=np.int32)
    positions[ids] = np.arange(len(ids), dtype=np.int32)
    return positions
