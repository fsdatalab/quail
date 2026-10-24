"""Turn a CTE's result into the table the next stage reads."""

import pyarrow as pa


def cte_table(result: pa.Table, schema: pa.Schema) -> pa.Table:
    """Return a stage's result as the table the next stage reads.

    Label columns are decoded from dictionaries to text, and the
    columns take the CTE's names.

    Raises:
        ValueError: The result's types differ from the schema the next
            stage was compiled against.
    """
    columns = [
        column.cast(column.type.value_type)
        if pa.types.is_dictionary(column.type) else column
        for column in result.columns]
    if len(columns) != len(schema) or [
            column.type for column in columns] != [field.type for field in schema]:
        raise ValueError(
            f"the CTE's result has types {[c.type for c in columns]}, but "
            f"the query reading it was compiled for {schema.types}")
    return pa.table(dict(zip(schema.names, columns)))
