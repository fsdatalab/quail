"""Document rows: the value every ``rows:`` port carries between operators."""

import numpy as np
import pyarrow as pa

ID_TYPE = pa.int32()


def document_ids(value) -> pa.Array:
    """Return document positions as one int32 Arrow array.

    Args:
        value: An Arrow array or chunked array, a one-column Arrow table,
            a NumPy array, or any sequence of integers.

    Raises:
        ValueError: The value holds a null, or a table with several
            columns.
    """
    if hasattr(value, "ids") and hasattr(value, "values"):
        if len(value.ids) != 1:
            raise ValueError("rows over several tables name no single document")
        (value,) = value.ids.values()
    if isinstance(value, pa.Table):
        if value.num_columns != 1:
            raise ValueError(
                "document ids must be one column, an array, or a sequence")
        value = value.column(0)
    if isinstance(value, pa.ChunkedArray):
        value = value.combine_chunks()
    elif isinstance(value, np.ndarray):
        value = pa.array(value)
    elif not isinstance(value, pa.Array):
        value = pa.array(value, type=pa.int64())
    if value.null_count:
        raise ValueError("a document id is null")
    return value if value.type == ID_TYPE else value.cast(ID_TYPE)


def id_positions(value) -> np.ndarray:
    """Return document ids as a NumPy array for indexing document stores."""
    return document_ids(value).to_numpy()
