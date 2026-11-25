"""Build score and label columns."""

import numpy as np
import pyarrow as pa

from quail.execution.rows import Rows
from quail.physical import ClassifySpec


def probability_column(labels, probabilities) -> pa.MapArray:
    """Build an Arrow map column of label probabilities.

    Args:
        labels: Labels in probability-column order.
        probabilities: Array with shape (documents, labels).

    Returns:
        A MapArray from label to probability for each document. A row with
        a NaN probability is null: its document has no answer.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(
        -1, len(labels))
    count, width = probabilities.shape
    return pa.MapArray.from_arrays(
        pa.array(np.arange(0, count * width + 1, width), pa.int32()),
        pa.array(np.tile(np.asarray(labels, dtype=object), count), pa.string()),
        pa.array(np.nan_to_num(probabilities).ravel(), pa.float64()),
        mask=pa.array(np.isnan(probabilities).any(axis=1)))


def score_rows(rows, aliases, name, scores, value_type=None) -> Rows:
    """The rows a score or classification returns: the ids and one value column."""
    return Rows({alias: rows[:, index] for index, alias in enumerate(aliases)},
                pa.table({name: pa.array(scores, type=value_type or pa.float64())}))


def scored_batch(node, rows, values) -> dict:
    """The answer-sink payload for one batch of scored rows.

    ``rows`` are the batch's row indices into each alias table, one int
    per row for a single alias and one list per row for a pair;
    ``values`` are their scores or labels, in the same order.
    """
    rows = np.asarray(rows)
    values = list(values)
    payload = {"kind": "score", "node": node.node_id,
               "output": node.spec.name, "aliases": list(node.spec.aliases),
               "rows": (rows[:, 0].tolist() if rows.shape[1] == 1
                        else rows.tolist())}
    if isinstance(node.spec, ClassifySpec):
        return {**payload, "kind": "label", "labels": values}
    return {**payload, "scores": [round(float(value), 4) for value in values]}


def classify_outputs(node, rows: Rows) -> dict:
    """Return the label rows on the node's rows port."""
    return {node.rows_port: rows}
