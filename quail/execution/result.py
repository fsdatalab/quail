"""Arrow tables and Acero execution for query results."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass

import pyarrow as pa
from pyarrow import acero
from pyarrow import compute as pc

DEFAULT_BATCH_ROWS = 65_536


@dataclass(frozen=True)
class IndexRelation:
    """A lazy Arrow relation over document index columns."""

    declaration: acero.Declaration
    schema: pa.Schema

    @classmethod
    def from_table(cls, table: pa.Table) -> "IndexRelation":
        """Create a lazy relation from one Arrow table."""
        return cls(_table_source(table), table.schema)


def document_index_schema(aliases: Iterable[str], kind: str) -> pa.Schema:
    """Return the shared schema for document-index relations."""
    fields = [
        pa.field(
            alias,
            pa.int32(),
            nullable=False,
            metadata={b"quail.alias": alias.encode("utf-8")},
        )
        for alias in aliases
    ]
    return pa.schema(
        fields,
        metadata={
            b"quail.kind": kind.encode("utf-8"),
        },
    )


def document_index_table(columns: dict[str, Iterable[int]],
                         kind: str) -> pa.Table:
    """Build a document-index table with the shared relation schema."""
    schema = document_index_schema(columns, kind)
    arrays = [pa.array(columns[field.name], type=field.type)
              for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def answer_table(columns: dict[str, Iterable[int]],
                 answers: Iterable[bool], kind: str,
                 metadata: dict[str, str] | None = None) -> pa.Table:
    """Build a filter or join answer table."""
    schema = document_index_schema(columns, kind)
    fields = list(schema)
    fields.append(pa.field("answer", pa.bool_(), nullable=False))
    schema_metadata = dict(schema.metadata or {})
    for key, value in (metadata or {}).items():
        schema_metadata[f"quail.{key}".encode("utf-8")] = \
            str(value).encode("utf-8")
    schema = pa.schema(fields, metadata=schema_metadata)
    arrays = [pa.array(columns[field.name], type=field.type)
              for field in schema if field.name != "answer"]
    arrays.append(pa.array(answers, type=pa.bool_()))
    return pa.Table.from_arrays(arrays, schema=schema)


def true_answer_rows(table: pa.Table) -> pa.Table:
    """Select the document-index rows whose answer is true."""
    if "answer" not in table.schema.names:
        raise ValueError("answer table has no answer column")
    aliases = [name for name in table.schema.names if name != "answer"]
    selected = table.filter(table.column("answer")).select(aliases)
    metadata = dict(table.schema.metadata or {})
    metadata[b"quail.kind"] = b"join_answers"
    return selected.replace_schema_metadata(metadata)


def _table_source(table: pa.Table) -> acero.Declaration:
    return acero.Declaration(
        "table_source", acero.TableSourceNodeOptions(table))


def _inner_join(left: acero.Declaration, left_schema: pa.Schema,
                right: acero.Declaration, right_schema: pa.Schema,
                keys: Iterable[str] | None = None,
                ) -> tuple[acero.Declaration, pa.Schema]:
    """Join two Acero inputs and keep one copy of each shared key."""
    right_names = set(right_schema.names)
    shared = list(keys) if keys is not None else [
        name for name in left_schema.names if name in right_names]
    if not shared:
        raise ValueError("ordinary query relations have no shared alias")
    for name in shared:
        if name not in left_schema.names or name not in right_schema.names:
            raise ValueError(f"join key {name!r} is missing")
        if left_schema.field(name).type != right_schema.field(name).type:
            raise TypeError(f"join key {name!r} has different Arrow types")

    right_output = [name for name in right_schema.names
                    if name not in shared]
    options = acero.HashJoinNodeOptions(
        "inner",
        shared,
        shared,
        left_output=left_schema.names,
        right_output=right_output,
    )
    declaration = acero.Declaration(
        "hashjoin", options, [left, right])
    fields = list(left_schema)
    fields.extend(right_schema.field(name) for name in right_output)
    metadata = dict(left_schema.metadata or {})
    metadata[b"quail.kind"] = b"joined_indices"
    return declaration, pa.schema(fields, metadata=metadata)


def build_result_declaration(
        true_join_tables: Iterable[pa.Table],
        survivor_indices: dict[str, pa.Array],
        base_alias: str,
        ) -> tuple[acero.Declaration, pa.Schema]:
    """Build the Acero joins for the rows returned by the SQL query."""
    pending = list(true_join_tables)
    if pending:
        table = pending.pop(0)
        declaration = _table_source(table)
        schema = table.schema
        while pending:
            for index, table in enumerate(pending):
                if set(schema.names) & set(table.schema.names):
                    declaration, schema = _inner_join(
                        declaration,
                        schema,
                        _table_source(table),
                        table.schema,
                    )
                    pending.pop(index)
                    break
            else:
                raise ValueError("AI join relations form a disconnected graph")
    else:
        if base_alias not in survivor_indices:
            raise ValueError(f"no survivors for alias {base_alias!r}")
        table = document_index_table(
            {base_alias: survivor_indices[base_alias]},
            "filter_survivors",
        )
        declaration = _table_source(table)
        schema = table.schema

    for alias in schema.names:
        if alias not in survivor_indices:
            raise ValueError(f"no survivors for alias {alias!r}")
        survivor_table = document_index_table(
            {alias: survivor_indices[alias]},
            "final_survivors",
        )
        declaration, schema = _inner_join(
            declaration,
            schema,
            _table_source(survivor_table),
            survivor_table.schema,
            keys=[alias],
        )
    return declaration, schema


def intersect_tables(left: pa.Table, right: pa.Table) -> pa.Table:
    """Return rows present in both document-index relations."""
    if set(left.schema.names) != set(right.schema.names):
        raise ValueError("relation intersection requires the same aliases")
    keys = left.schema.names
    declaration, _ = _inner_join(
        _table_source(left), left.schema,
        _table_source(right), right.schema,
        keys=keys,
    )
    table = declaration.to_table()
    return table.select(keys).replace_schema_metadata(left.schema.metadata)


def intersect_indices(left: pa.Array, right: pa.Array) -> pa.Array:
    """Return document indices present in both arrays."""
    mask = pc.is_in(left, value_set=right)
    return pc.filter(left, mask)


def count_rows(declaration: acero.Declaration) -> int:
    """Count rows inside Acero without returning the rows to Python."""
    options = acero.AggregateNodeOptions(
        [([], "count_all", None, "row_count")])
    count_declaration = acero.Declaration(
        "aggregate", options, [declaration])
    table = count_declaration.to_table()
    return int(table.column("row_count")[0].as_py())


def execute_stream(declaration: acero.Declaration, schema: pa.Schema,
                   limit: int | None = None,
                   batch_rows: int = DEFAULT_BATCH_ROWS,
                   ) -> pa.RecordBatchReader:
    """Return bounded Arrow batches from an Acero declaration."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    source = declaration.to_reader()

    def batches():
        remaining = limit
        try:
            for batch in source:
                if remaining == 0:
                    break
                if remaining is not None and len(batch) > remaining:
                    batch = batch.slice(0, remaining)
                offset = 0
                while offset < len(batch):
                    length = min(batch_rows, len(batch) - offset)
                    yield batch.slice(offset, length)
                    offset += length
                    if remaining is not None:
                        remaining -= length
                        if remaining == 0:
                            break
        finally:
            source.close()

    return pa.RecordBatchReader.from_batches(schema, batches())


class QueryResult:
    """A lazy Arrow result backed by an Acero execution plan."""

    def __init__(self, columns: list[str],
                 declaration: acero.Declaration,
                 document_index_schema: pa.Schema,
                 output_schema: pa.Schema,
                 projection: list[tuple[str, pa.Array | pa.ChunkedArray]],
                 report: dict,
                 answer_tables: dict | None = None,
                 limit: int | None = None,
                 survivor_indices: dict[str, pa.Array] | None = None,
                 true_join_tables: dict[int, pa.Table] | None = None,
                 row_count: int | None = None):
        self.columns = columns
        self.schema = output_schema
        self.report = report
        # the executed physical graph and each node's measured metrics,
        # filled in by Query.finish
        self.plan = None
        self.node_metrics: dict = {}
        self.answer_tables = answer_tables or {"filters": {}, "joins": {}}
        self.limit = limit
        self._declaration = declaration
        self._document_index_schema = document_index_schema
        self._projection = projection
        self.survivor_indices = survivor_indices or {}
        self.true_join_tables = true_join_tables or {}
        self._row_count = row_count
        self._materialized = None

    @classmethod
    def from_table(cls, table: pa.Table,
                   report: dict | None = None) -> "QueryResult":
        """Create a result from a table produced by a local runtime."""
        result = cls.__new__(cls)
        result.columns = list(table.column_names)
        result.schema = table.schema
        result.report = report or {}
        result.answer_tables = {"filters": {}, "joins": {}}
        result.plan = None
        result.node_metrics = {}
        result.limit = None
        result._declaration = None
        result._document_index_schema = None
        result._projection = []
        result.survivor_indices = {}
        result.true_join_tables = {}
        result._row_count = len(table)
        result._materialized = table
        return result

    def execute_stream(self, batch_rows: int = DEFAULT_BATCH_ROWS,
                       limit: int | None = None
                       ) -> pa.RecordBatchReader:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        effective_limit = self.limit
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be nonnegative")
            effective_limit = (limit if effective_limit is None
                               else min(limit, effective_limit))
        if self._materialized is not None:
            table = self._materialized
            if effective_limit is not None:
                table = table.slice(0, effective_limit)
            return table.to_reader(max_chunksize=batch_rows)
        indices = execute_stream(
            self._declaration,
            self._document_index_schema,
            effective_limit,
            batch_rows,
        )

        def batches():
            try:
                for batch in indices:
                    arrays = []
                    for alias, values in self._projection:
                        selected = pc.take(
                            values, batch.column(batch.schema.get_field_index(
                                alias)))
                        if isinstance(selected, pa.ChunkedArray):
                            selected = selected.combine_chunks()
                        arrays.append(selected)
                    yield pa.RecordBatch.from_arrays(
                        arrays, schema=self.schema)
            finally:
                indices.close()

        return pa.RecordBatchReader.from_batches(self.schema, batches())

    def collect(self, limit: int | None = None,
                batch_rows: int = DEFAULT_BATCH_ROWS
                ) -> pa.Table:
        reader = self.execute_stream(batch_rows=batch_rows, limit=limit)
        try:
            return reader.read_all()
        finally:
            reader.close()

    def attach_executed_plan(self, codecs) -> "QueryResult":
        """Decode the executed plan and node metrics from a saved report."""
        from quail.execution.runner import NodeMetrics
        from quail.physical import decode_graph

        encoded = self.report.get("executed_plan")
        if encoded is not None:
            self.plan = decode_graph(encoded, codecs)
            self.node_metrics = {
                node_id: NodeMetrics(**metrics)
                for node_id, metrics in self.report.get(
                    "node_metrics", {}).items()
            }
        return self

    def explain(self, *, verbose: bool = False) -> str:
        """Return the executed operator tree with measured rows and time."""
        from quail.explain import measured_stages, physical_tree

        if self.plan is None:
            return "no physical plan was executed"
        return physical_tree(self.plan, verbose=verbose,
                             metrics=self.node_metrics,
                             stages=measured_stages(self.report))

    def observer(self, observer) -> dict:
        """Return the report one execution observer attached.

        Args:
            observer: The observer class or instance, or its name.
        """
        name = observer if isinstance(observer, str) else observer.name
        reports = self.report.get("observers", {})
        if name not in reports:
            raise KeyError(
                f"no report from observer {name!r}; the query ran with "
                f"{sorted(reports) or 'no observers'}")
        return reports[name]

    def to_rows(self, limit: int | None = None) -> list[tuple]:
        table = self.collect(limit=limit)
        columns = [column.to_pylist() for column in table.columns]
        return list(zip(*columns))

    def known_count(self) -> int | None:
        """Return the row count when it is known without running the plan."""
        return self._row_count

    def count(self) -> int:
        if self._row_count is None:
            count = count_rows(self._declaration)
            self._row_count = (count if self.limit is None
                               else min(count, self.limit))
        return self._row_count

    def with_limit(self, limit: int) -> "QueryResult":
        """Return the same result with a smaller row limit."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = copy(self)
        result.limit = limit if self.limit is None else min(self.limit, limit)
        if self._row_count is not None:
            result._row_count = min(self._row_count, result.limit)
        return result

    def __len__(self) -> int:
        return self.count()
