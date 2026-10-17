"""Row selection preserves positions and values across input representations."""

import numpy as np
import pyarrow as pa
import pytest

from quail.execution.ids import document_ids
from quail.execution.rows import Rows


@pytest.mark.parametrize("indices", [[], (), np.array([]), pa.array([], pa.int64())])
def test_taking_no_rows_retains_the_schema(indices):
    rows = Rows({"left": [2, 1], "right": [4, 3]},
                pa.table({"answer": ["yes", "no"]}))
    selected = rows.take(indices)
    assert selected.num_rows == 0
    assert selected.aliases == rows.aliases
    assert selected.values.schema == rows.values.schema
    assert all(ids.type == pa.int32() and len(ids) == 0
               for ids in selected.ids.values())


@pytest.mark.parametrize("indices", [[1, 0, 1], np.array([1, 0, 1]),
                                     pa.array([1, 0, 1], pa.int64())])
def test_taking_rows_preserves_order_and_repeated_positions(indices):
    rows = Rows({"r": [8, 5]}, pa.table({"answer": ["first", "second"]}))
    assert rows.take(indices).to_pydict() == {
        "r": [5, 8, 5], "answer": ["second", "first", "second"]}


@pytest.mark.parametrize("indices", [[0.5], np.array([0.5]), pa.array([0.5])])
def test_integer_positions_do_not_truncate_fractional_values(indices):
    with pytest.raises(pa.ArrowInvalid, match="truncated"):
        document_ids(indices)
    with pytest.raises(pa.ArrowInvalid, match="truncated"):
        Rows({"r": [2, 3]}).take(indices)
