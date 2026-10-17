"""Arrow tables and Acero execution for query results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import copy
from dataclasses import dataclass
from itertools import repeat
from weakref import finalize

import numpy as np
import pyarrow as pa
from pyarrow import acero
from pyarrow import compute as pc

from quail.catalog.pairs import pair_keys

DEFAULT_BATCH_ROWS = 65_536


@dataclass(frozen=True)
class StructEntry:
    """Projection entries a result packs into one struct column.

    Attributes:
        entries: The (alias, values) projection entry of each field.
        fields: The Arrow field of each field, named as the struct
            field.
    """

    entries: tuple
    fields: tuple


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


def _answer_field(columns, role: str, dtype) -> pa.Field:
    """Name an answer-metadata field without shadowing a document ID."""
    name = role
    while name in columns:
        name = "_" + name
    return pa.field(name, dtype, nullable=False,
                    metadata={b"quail.role": role.encode("utf-8")})


def answer_table(columns: dict[str, Iterable[int]],
                 answers: Iterable[bool], kind: str,
                 metadata: dict[str, str] | None = None) -> pa.Table:
    """Build a filter or join answer table."""
    schema = document_index_schema(columns, kind)
    fields = list(schema)
    fields.append(_answer_field(columns, "answer", pa.bool_()))
    schema_metadata = dict(schema.metadata or {})
    for key, value in (metadata or {}).items():
        schema_metadata[f"quail.{key}".encode("utf-8")] = \
            str(value).encode("utf-8")
    schema = pa.schema(fields, metadata=schema_metadata)
    arrays = [pa.array(values, type=pa.int32()) for values in columns.values()]
    arrays.append(pa.array(answers, type=pa.bool_()))
    return pa.Table.from_arrays(arrays, schema=schema)


def filter_answer_table(alias, documents, positions, answers) -> pa.Table:
    """Build filter answers with predicate positions and alias metadata."""
    schema = pa.schema(
        [pa.field(alias, pa.int32(), nullable=False),
         _answer_field((alias,), "predicate", pa.int32()),
         _answer_field((alias,), "answer", pa.bool_())],
        metadata={b"quail.kind": b"filter_answers",
                  b"quail.alias": alias.encode("utf-8")})
    return pa.Table.from_arrays(
        [pa.array(documents, pa.int32()), pa.array(positions, pa.int32()),
         pa.array(answers, pa.bool_())], schema=schema)


def extract_answers(alias, name, span_name, document_ids, answers, spans):
    """Build the rows of an extraction: its answers and nullable character spans.

    Returns:
        Rows of the alias with the answer and span columns.
    """
    from quail.execution.rows import Rows

    valid = np.fromiter((span is not None for span in spans), dtype=bool,
                        count=len(spans))
    bounds = np.zeros((len(spans), 2), dtype=np.int32)
    bounds[valid] = np.asarray([span for span in spans if span is not None],
                               dtype=np.int32).reshape(-1, 2)
    span_array = pa.StructArray.from_arrays(
        [pa.array(bounds[:, 0]), pa.array(bounds[:, 1])],
        names=["start", "end"], mask=pa.array(~valid))
    return Rows({alias: document_ids},
                pa.table({name: pa.array(answers, pa.string()), span_name: span_array}))


def answer_values(table: pa.Table) -> pa.ChunkedArray:
    """Return the final boolean column of a filter or join answer table.

    Table aliases may also be named "answer", so the answer's position
    identifies it independently of the document ID columns.

    Raises:
        ValueError: The final column is not the boolean answer column.
    """
    if table.num_columns:
        field = table.schema.field(-1)
        role = (field.metadata or {}).get(b"quail.role", field.name.encode("utf-8"))
        if role == b"answer" and field.type == pa.bool_():
            return table.column(-1)
    raise ValueError("answer table has no final boolean answer column")


def true_answer_rows(table: pa.Table) -> pa.Table:
    """Select the document-index rows whose answer is true."""
    selected = table.filter(answer_values(table)).select(range(table.num_columns - 1))
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


def joined_survivors(tables, survivors) -> dict[str, pa.Array]:
    """Keep document IDs occurring in a complete join of each connected component.

    Args:
        tables: Relations of document IDs whose join predicates passed.
        survivors: Input document IDs by alias, in their existing order.

    Returns:
        Surviving IDs by alias, preserving their input order and duplicates.
    """
    components = []
    for table in tables:
        aliases = set(table.column_names)
        group = [table.cast(document_index_schema(table.column_names, "join_answers"))]
        separate = []
        for names, members in components:
            if aliases & names:
                aliases |= names
                group.extend(members)
            else:
                separate.append((names, members))
        components = [*separate, (aliases, group)]
    result = dict(survivors)
    for aliases, group in components:
        limits = dict(survivors)
        for table in group:
            for alias in table.column_names:
                if alias not in limits:
                    limits[alias] = pc.unique(table[alias])
        declaration, _ = build_result_declaration(
            group, limits, group[0].column_names[0])
        found = {alias: set() for alias in aliases if alias in survivors}
        with declaration.to_reader() as reader:
            for batch in reader:
                for alias, values in found.items():
                    values.update(pc.unique(batch.column(alias)).to_pylist())
        for alias, values in found.items():
            ids = survivors[alias]
            result[alias] = pc.filter(ids, pc.is_in(
                ids, value_set=pa.array(sorted(values), type=ids.type)))
    return result


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


def count_rows(declaration: acero.Declaration) -> int:
    """Count rows inside Acero without returning the rows to Python."""
    options = acero.AggregateNodeOptions(
        [([], "count_all", None, "row_count")])
    count_declaration = acero.Declaration(
        "aggregate", options, [declaration])
    table = count_declaration.to_table()
    return int(table.column("row_count")[0].as_py())


def bounded_reader(source: pa.RecordBatchReader, limit: int | None = None,
                   batch_rows: int = DEFAULT_BATCH_ROWS, *,
                   schema: pa.Schema | None = None,
                   project: Callable[[pa.RecordBatch], pa.RecordBatch] | None = None,
                   ) -> pa.RecordBatchReader:
    """Read and project bounded batches while owning the source reader."""
    def batches():
        remaining = limit
        try:
            while remaining is None or remaining > 0:
                batch = next(source, None)
                if batch is None:
                    return
                if remaining is not None:
                    batch = batch[:remaining]
                    remaining -= len(batch)
                for offset in range(0, len(batch), batch_rows):
                    part = batch.slice(offset, batch_rows)
                    yield part if project is None else project(part)
        finally:
            close()

    iterator = batches()
    # A generator's finally block does not run if it was never started.
    close = finalize(iterator, source.close)
    # The C stream releases its iterator on close; from_batches alone does not.
    return pa.RecordBatchReader.from_stream(pa.RecordBatchReader.from_batches(
        source.schema if schema is None else schema, iterator))


def projected_column(batch: pa.RecordBatch, alias, values) -> pa.Array:
    """Return one result column for a batch of document index rows.

    Args:
        batch: The index rows.
        alias: The index column that selects the values, the two ids
            of a joined label, or None for a struct column.
        values: The values by document index, None for a column the
            batch carries, a joined label's lookup, or a StructEntry.
    """
    if isinstance(values, StructEntry):
        return pa.StructArray.from_arrays(
            [projected_column(batch, *entry) for entry in values.entries],
            fields=list(values.fields))
    if isinstance(alias, tuple):
        # a label of joined rows, looked up by the row's ids
        keys, labeled = values
        positions = pc.index_in(
            pair_keys(*(batch.column(batch.schema.get_field_index(name))
                        for name in alias)),
            value_set=keys)
        return pc.take(labeled, positions)
    column = batch.column(batch.schema.get_field_index(alias))
    # a computed column such as a score is carried as is
    selected = column if values is None else pc.take(values, column)
    if isinstance(selected, pa.ChunkedArray):
        selected = selected.combine_chunks()
    return selected


def iter_rows(table: pa.Table) -> Iterator[tuple]:
    """Yield tuples preserving column order and rows with no columns."""
    if not table.num_columns:
        return repeat((), table.num_rows)
    return zip(*(column.to_pylist() for column in table.columns))


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
        """Stream projected result rows as Arrow record batches.

        Args:
            batch_rows: Maximum rows in each returned batch.
            limit: Maximum rows to return, or all rows when omitted.

        Returns:
            An Arrow reader that the caller should exhaust or close.
        """
        return self._result_reader(batch_rows, limit, materialize=False)

    def _result_reader(self, batch_rows, limit, *, materialize):
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        effective_limit = self.limit
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be nonnegative")
            effective_limit = (limit if effective_limit is None
                               else min(limit, effective_limit))
        if self._materialized is not None:
            return self._materialized[:effective_limit].to_reader(
                max_chunksize=batch_rows)
        def project(batch):
            if not self._projection:
                return batch.select([]).replace_schema_metadata(self.schema.metadata)
            arrays = [projected_column(batch, alias, values)
                      for alias, values in self._projection]
            return pa.RecordBatch.from_arrays(arrays, schema=self.schema)

        if materialize and effective_limit is None:
            source = self._declaration.to_table().to_reader(max_chunksize=batch_rows)
        else:
            # Acero's threaded hash joins can hang on early close. Only complete
            # table execution can use them without an intermediate stream.
            source = self._declaration.to_reader(use_threads=False)
        return bounded_reader(
            source, effective_limit, batch_rows, schema=self.schema, project=project)

    def collect(self, limit: int | None = None,
                batch_rows: int = DEFAULT_BATCH_ROWS
                ) -> pa.Table:
        """Collect projected result rows into one Arrow table.

        Args:
            limit: Maximum rows to collect, or all rows when omitted.
            batch_rows: Maximum rows read from the result at once.

        Returns:
            A table containing only the columns projected by the query.
        """
        reader = self._result_reader(batch_rows, limit, materialize=True)
        try:
            return reader.read_all()
        finally:
            reader.close()

    def attach_executed_plan(self, codecs) -> "QueryResult":
        """Decode the executed plan and node metrics from a saved report."""
        from quail.execution.metrics import NodeMetrics
        from quail.physical import decode_graph

        encoded = self.report.get("executed_plan")
        if encoded is not None:
            self.plan = decode_graph(encoded, codecs)
            self.node_metrics = {
                node_id: NodeMetrics.from_scalars(metrics)
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
        """Collect projected result rows as Python tuples."""
        return list(iter_rows(self.collect(limit=limit)))

    def known_count(self) -> int | None:
        """Return the row count when it is known without running the plan."""
        return self._row_count

    def count(self) -> int:
        """Return the result row count, computing it when needed."""
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
