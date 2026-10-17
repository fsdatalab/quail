"""Project document columns and model answers into an Arrow query result."""

import numpy as np
import pyarrow as pa
from pyarrow import acero
from pyarrow import compute as pc

from quail.catalog.pairs import pair_keys
from quail.execution.result import (
    IndexRelation,
    QueryResult,
    StructEntry,
    true_answer_rows,
)
from quail.logical import (
    Alias,
    CompileError,
)
from quail.physical import Project, ValueType


def projection_for(logical, token_inputs, documents):
    """Bind result projection to its scanned columns and classification calls."""
    operators = logical.operators()
    scans_by_alias = {scan.alias: scan for scan in operators.scans}
    model_calls = {node.name: node.call
                   for node in (*operators.classifies, *operators.extracts)}
    model_calls.update({node.span_name: node.call for node in operators.extracts})
    model_calls.update({column.name: column.expression
                        for column in logical.projection.columns
                        if isinstance(column, Alias)
                        and column.expression.kind in ("label", "extract", "score")})

    def project(node, value, labels=None):
        if not isinstance(node, Project):
            raise TypeError(type(node).__name__)
        if node.inputs[0].value_type is ValueType.JOIN_ANSWERS:
            value = true_answer_rows(value)
        relation = (
            IndexRelation.from_table(value)
            if isinstance(value, pa.Table) else value
        )
        if not isinstance(relation, IndexRelation):
            raise TypeError("Project needs an index relation")
        projection = []
        fields = []
        declaration = relation.declaration
        for name in node.columns:
            if name in relation.schema.names:
                # a score column the graph computed; document
                # index columns are aliases, never "alias.column"
                projection.append((name, None))
                fields.append(pa.field(
                    name, relation.schema.field(name).type
                ))
                continue
            table = next((table for table in (labels or {}).values()
                          if name in table.column_names), None)
            call = model_calls.get(name)
            aliases = () if call is None else call.aliases()
            if table is not None and len(aliases) == 2:
                # a label of joined rows: each result row looks its
                # label up by its two ids; a row without one leaves
                # the result
                if any(alias not in relation.schema.names
                       for alias in aliases):
                    raise CompileError(
                        f"label column {name!r} belongs to the joined "
                        f"rows of {aliases}, not all in the result")
                order = (None if call is None
                         else pa.array(call.labels, pa.string()))
                keys = pair_keys(table.column(aliases[0]),
                                  table.column(aliases[1]))
                declaration = acero.Declaration(
                    "filter", acero.FilterNodeOptions(
                        pair_keys(*(pc.field(alias) for alias in aliases))
                        .isin(keys)),
                    inputs=[declaration])
                projection.append((tuple(aliases), (keys, table, order)))
                fields.append(pa.field(
                    name,
                    pa.string() if order is None
                    else pa.dictionary(pa.int32(), pa.string()),
                    nullable=True,
                    metadata={b"quail.alias": aliases[0].encode("utf-8"),
                              b"quail.label": name.encode("utf-8")}))
                continue
            if table is not None:
                # a label the graph computed for some of an alias's
                # documents, widened to every document of the alias
                (alias,) = aliases
                if alias not in relation.schema.names:
                    raise CompileError(
                        f"label column {name!r} belongs to {alias!r}, "
                        f"which is not in the result")
                labeled = table.column(name).combine_chunks()
                if pa.types.is_dictionary(labeled.type):
                    labeled = labeled.cast(labeled.type.value_type)
                # a document whose answer names no label has no row
                # in the table, and leaves the result
                declaration = acero.Declaration(
                    "filter", acero.FilterNodeOptions(
                        pc.field(alias).isin(table.column(alias))),
                    inputs=[declaration])
                # position of each document's label, or the null
                # appended past the labels for an unlabeled one
                slots = np.full(len(documents[alias]), len(labeled),
                                dtype=np.int64)
                slots[table.column(alias).to_numpy()] = np.arange(
                    len(labeled))
                widened = pc.take(
                    pa.concat_arrays([labeled, pa.array([None], labeled.type)]),
                    pa.array(slots))
                if call is not None and call.kind == "label":
                    order = pa.array(call.labels, pa.string())
                    values = pa.DictionaryArray.from_arrays(
                        pc.index_in(widened, order).cast(pa.int32()),
                        order)
                else:
                    values = widened
                projection.append((alias, values))
                fields.append(pa.field(
                    name, values.type, nullable=True,
                    metadata={b"quail.alias": alias.encode("utf-8"),
                              b"quail.label": name.encode("utf-8")}))
                continue
            try:
                alias, column = name.split(".", 1)
                scan = scans_by_alias[alias]
            except (ValueError, KeyError) as error:
                raise CompileError(
                    f"unknown projection column {name!r}"
                ) from error
            if alias not in relation.schema.names:
                raise CompileError(
                    f"projection column {name!r} is not in the result"
                )
            store = token_inputs[alias]
            if column not in store.projected_columns:
                raise CompileError(
                    f"projection column {name!r} was not loaded by the "
                    f"scan of {alias!r}; the projection_pushdown "
                    f"logical rule is not registered")
            values = store.column(column)
            projection.append((alias, values))
            fields.append(pa.field(
                name,
                values.type,
                nullable=values.null_count > 0,
                metadata={
                    b"quail.alias": alias.encode("utf-8"),
                    b"quail.provider": scan.provider.encode("utf-8"),
                    b"quail.column": column.encode("utf-8"),
                },
            ))
        if node.structs:
            projection, fields = pack_structs(
                node.columns, node.structs, projection, fields)
        return QueryResult(
            columns=list(node.result_columns),
            declaration=declaration,
            document_index_schema=relation.schema,
            output_schema=pa.schema(
                fields,
                metadata={b"quail.kind": b"query_result"},
            ),
            projection=projection,
            report={},
            row_count=(value.num_rows if isinstance(value, pa.Table)
                       and declaration is relation.declaration
                       else None),
        )

    return project


def pack_structs(columns, structs, projection, fields) -> tuple[list, list]:
    """Pack member columns into struct columns.

    Each struct column takes the place of its first member and holds
    one field per member; the members leave the result.

    Args:
        columns: The projected column names, one per projection entry.
        structs: (struct column, ((field, member column), ...)) pairs.
        projection: The (alias, values) entry of each column.
        fields: The Arrow field of each column.

    Returns:
        The packed projection entries and fields.
    """
    entries = dict(zip(columns, zip(projection, fields)))
    owner = {member: name for name, members in structs for _, member in members}
    packed = {}
    for name, members in structs:
        parts = [entries[member] for _, member in members]
        struct_fields = tuple(pa.field(field, part.type)
                              for (field, _), (_, part) in zip(members, parts))
        packed[name] = (
            (None, StructEntry(tuple(entry for entry, _ in parts), struct_fields)),
            pa.field(name, pa.struct(struct_fields), nullable=True,
                     metadata=parts[0][1].metadata))
    out_projection, out_fields = [], []
    for column, entry in zip(columns, zip(projection, fields)):
        name = owner.get(column)
        if name is not None:
            if name not in packed:
                continue
            entry = packed.pop(name)
        out_projection.append(entry[0])
        out_fields.append(entry[1])
    return out_projection, out_fields
