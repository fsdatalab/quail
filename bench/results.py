"""Write Modal run metadata and the measurements table across methods."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, data):
    """Replace a JSON file only after its contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def combine_measurements(directory, methods) -> pa.Table:
    """Concatenate the methods' QUAIL-B measurements into one table.

    Each `<method>/measurements.parquet` that QUAIL-B wrote gets a
    `method` column; the result is `<directory>/measurements.parquet`.
    """
    directory = Path(directory)
    tables = []
    for method in methods:
        table = pq.read_table(directory / method / "measurements.parquet")
        tables.append(table.append_column(
            "method", pa.array([method] * table.num_rows, pa.string())))
    table = pa.concat_tables(tables)
    pq.write_table(table, directory / "measurements.parquet", compression="zstd")
    return table
