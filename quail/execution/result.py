"""Query results, and the id relations the executor joins with Acero."""

from __future__ import annotations

import itertools
import json
import weakref
from collections.abc import Callable, Iterable

import pyarrow as pa
from pyarrow import acero
from pyarrow import compute as pc

DEFAULT_BATCH_ROWS = 65_536


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


def free_name(base: str, taken: Iterable[str]) -> str:
    """Return base, or base behind enough underscores to be unlike every taken name."""
    taken = set(taken)
    name = base
    while name in taken:
        name = "_" + name
    return name


def answer_table(columns: dict[str, Iterable[int]],
                 answers: Iterable[bool], kind: str,
                 metadata: dict[str, str] | None = None) -> pa.Table:
    """Build a filter or join answer table: the id columns, then the answers.

    The answer column comes last, under a name no alias uses.
    """
    schema = document_index_schema(columns, kind)
    fields = list(schema)
    fields.append(pa.field(free_name("answer", schema.names), pa.bool_(),
                           nullable=False))
    schema_metadata = dict(schema.metadata or {})
    for key, value in (metadata or {}).items():
        schema_metadata[f"quail.{key}".encode("utf-8")] = \
            str(value).encode("utf-8")
    schema = pa.schema(fields, metadata=schema_metadata)
    arrays = [pa.array(columns[field.name], type=field.type)
              for field in list(schema)[:-1]]
    arrays.append(pa.array(answers, type=pa.bool_()))
    return pa.Table.from_arrays(arrays, schema=schema)


def answer_values(table: pa.Table) -> pa.ChunkedArray:
    """Return the answers of an answer table: its last column."""
    return table.column(table.num_columns - 1)


def filter_answer_table(alias, documents, positions, answers) -> pa.Table:
    """Build filter answers: the ids, the predicate positions, then the answers."""
    predicate = free_name("predicate", (alias,))
    schema = pa.schema(
        [pa.field(alias, pa.int32(), nullable=False),
         pa.field(predicate, pa.int32(), nullable=False),
         pa.field(free_name("answer", (alias, predicate)), pa.bool_(),
                  nullable=False)],
        metadata={b"quail.kind": b"filter_answers",
                  b"quail.alias": alias.encode("utf-8")})
    return pa.Table.from_arrays(
        [pa.array(documents, pa.int32()), pa.array(positions, pa.int32()),
         pa.array(answers, pa.bool_())], schema=schema)


def true_answer_rows(table: pa.Table) -> pa.Table:
    """Select the document-index rows whose answer is true."""
    answers = answer_values(table)
    if not pa.types.is_boolean(answers.type):
        raise ValueError("answer table has no answer column")
    selected = table.filter(answers).select(table.column_names[:-1])
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


def joined_survivors(tables: Iterable[pa.Table],
                     survivors: dict[str, pa.Array]) -> dict[str, pa.Array]:
    """Keep the ids that take part in a complete join of their relations.

    Relations sharing an alias form one component, joined as a whole,
    so an id survives only with a partner in every relation at once.

    Args:
        tables: Relations of the id pairs whose join predicates passed.
        survivors: alias -> its ids before the joins, in order.

    Returns:
        alias -> the surviving ids, in their input order with duplicates.
    """
    components = []
    for table in tables:
        aliases = set(table.column_names)
        group = [table.cast(document_index_schema(table.column_names, "join_answers"))]
        apart = []
        for names, members in components:
            if aliases & names:
                aliases |= names
                group.extend(members)
            else:
                apart.append((names, members))
        components = [*apart, (aliases, group)]
    kept = dict(survivors)
    for aliases, group in components:
        limits = {alias: survivors.get(alias, pc.unique(table[alias]))
                  for table in group for alias in table.column_names}
        declaration, _ = build_result_declaration(
            group, limits, group[0].column_names[0])
        joined = declaration.to_table(use_threads=False)
        for alias in aliases & set(survivors):
            ids = survivors[alias]
            kept[alias] = pc.filter(ids, pc.is_in(
                ids, value_set=pc.unique(joined[alias]).cast(ids.type)))
    return kept


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


def bounded_reader(source: pa.RecordBatchReader, limit: int | None = None,
                   batch_rows: int = DEFAULT_BATCH_ROWS, *,
                   schema: pa.Schema | None = None,
                   project: Callable | None = None) -> pa.RecordBatchReader:
    """Return a reader of at most limit rows that owns the source reader.

    The source is read only while rows are still wanted, so a limit
    never reads a batch past itself, and it is closed when the
    returned reader is closed, exhausted, or dropped unread.

    Args:
        source: The reader of the rows.
        limit: Maximum rows to return, or every row when None.
        batch_rows: Maximum rows in each returned batch.
        schema: The returned batches' schema, when project changes it.
        project: A function from a source batch to a returned batch.
    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")

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
    # a generator never started runs no finally block
    close = weakref.finalize(iterator, source.close)
    # the C stream releases the iterator on close; from_batches alone does not
    return pa.RecordBatchReader.from_stream(pa.RecordBatchReader.from_batches(
        source.schema if schema is None else schema, iterator))


def iter_rows(table: pa.Table):
    """Yield the rows of a table as tuples, by column position."""
    if table.num_columns == 0:
        return itertools.repeat((), table.num_rows)
    return zip(*(column.to_pylist() for column in table.columns))


class QueryResult:
    """A query's rows as one Arrow table, with the run's plan, report, and answers."""

    def __init__(self, table: pa.Table, report: dict | None = None):
        self.columns = list(table.column_names)
        self.schema = table.schema
        self.report = report or {}
        # the executed physical graph and each node's measured metrics,
        # filled in by Query.finish
        self.plan = None
        self.node_metrics: dict = {}
        self.answer_tables = {"filters": {}, "joins": {}}
        self.survivor_indices: dict[str, pa.Array] = {}
        self.true_join_tables: dict[int, pa.Table] = {}
        self._table = table
        # DataFusion's plan around the model region, when it ran one
        self.datafusion_plan: str | None = None
        self.region_rows: int | None = None

    @classmethod
    def from_table(cls, table: pa.Table,
                   report: dict | None = None) -> "QueryResult":
        """Create a result over a table of rows."""
        return cls(table, report)

    def execute_stream(self, batch_rows: int = DEFAULT_BATCH_ROWS,
                       limit: int | None = None
                       ) -> pa.RecordBatchReader:
        """Stream result rows as Arrow record batches.

        Args:
            batch_rows: Maximum rows in each returned batch.
            limit: Maximum rows to return, or all rows when omitted.

        Returns:
            An Arrow reader that the caller should exhaust or close.
        """
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        table = self._table if limit is None else self._table[:limit]
        return table.to_reader(max_chunksize=batch_rows)

    def collect(self, limit: int | None = None,
                batch_rows: int = DEFAULT_BATCH_ROWS
                ) -> pa.Table:
        """Collect result rows into one Arrow table.

        Args:
            limit: Maximum rows to collect, or all rows when omitted.
            batch_rows: Maximum rows read from the result at once.

        Returns:
            A table containing only the columns projected by the query.
        """
        reader = self.execute_stream(batch_rows=batch_rows, limit=limit)
        try:
            return reader.read_all()
        finally:
            reader.close()

    def attach_executed_plan(self) -> "QueryResult":
        """Decode the executed plan and node metrics from a saved report."""
        from quail.execution.metrics import NodeMetrics
        from quail.physical import PhysicalGraph

        encoded = self.report.get("executed_plan")
        if encoded is not None:
            self.plan = PhysicalGraph.from_json(json.dumps(encoded))
            self.node_metrics = {
                node_id: NodeMetrics.from_scalars(metrics)
                for node_id, metrics in self.report.get(
                    "node_metrics", {}).items()
            }
        return self

    def explain(self, *, verbose: bool = False) -> str:
        """Return the executed operator tree with measured rows and time."""
        from quail.explain import physical_tree

        if self.plan is None:
            return "no physical plan was executed"
        tree = physical_tree(self.plan, verbose=verbose,
                             node_metrics=self.node_metrics, report=self.report)
        if self.datafusion_plan is None:
            return tree
        return f"DataFusion:\n{self.datafusion_plan}\n\nmodel region:\n{tree}"

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
        """Collect result rows as Python tuples."""
        return list(iter_rows(self.collect(limit=limit)))

    def count(self) -> int:
        """Return the result row count."""
        return self._table.num_rows

    def __len__(self) -> int:
        return self.count()
