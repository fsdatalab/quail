"""File backed token storage and Arrow token views."""

from __future__ import annotations

import bisect
import os
from collections.abc import Sequence

import pyarrow as pa
from pyarrow import compute as pc


class TokenView(Sequence):
    def __init__(self, values):
        self._values = values

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        try:
            return iter(self.numpy())
        except (TypeError, ValueError):
            return iter(self._values.to_pylist())

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return list(self)[index]
            return TokenView(self._values.slice(start, stop - start))
        return self._values[index].as_py()

    def numpy(self):
        return self._values.to_numpy(zero_copy_only=True)

    @property
    def arrow_array(self):
        return self._values

    @property
    def token_parts(self):
        return (self,)


class TokenChain(Sequence):
    def __init__(self, *parts):
        flattened = []
        for part in parts:
            if isinstance(part, TokenChain):
                flattened.extend(part.token_parts)
            elif len(part):
                flattened.append(part)
        self._parts = tuple(flattened)
        self._length = sum(len(part) for part in self._parts)

    def __len__(self):
        return self._length

    def __iter__(self):
        for part in self._parts:
            yield from part

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        for part in self._parts:
            if index < len(part):
                return part[index]
            index -= len(part)
        raise IndexError(index)

    @property
    def token_parts(self):
        return self._parts


class ArrowTokenDocuments(Sequence):
    def __init__(self, array):
        self._array = array
        self._chunks = tuple(array.chunks) if hasattr(array, "chunks") \
            else (array,)
        self._ends = []
        total = 0
        for chunk in self._chunks:
            total += len(chunk)
            self._ends.append(total)
        self._offsets = tuple(
            chunk.offsets.to_numpy(zero_copy_only=True)
            for chunk in self._chunks)
        self._values_by_chunk = tuple(chunk.values for chunk in self._chunks)
        self._values = (self._values_by_chunk[0]
                        if len(self._values_by_chunk) == 1 else None)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chunk_index = bisect.bisect_right(self._ends, index)
        prior = self._ends[chunk_index - 1] if chunk_index else 0
        local = index - prior
        offsets = self._offsets[chunk_index]
        start = int(offsets[local])
        end = int(offsets[local + 1])
        values = self._values_by_chunk[chunk_index]
        return TokenView(values.slice(start, end - start))


class TokenLengths(Sequence):
    """Read document lengths from a token store without copying them."""

    def __init__(self, store: "TokenStore"):
        self._store = store

    def __len__(self):
        return len(self._store)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        batch_index, local = self._store._locate(index)
        return self._store._length_batches[batch_index][local].as_py()


class TokenStore(Sequence):
    """Provide random token access from one memory mapped Arrow file."""

    _token_column = "__quail_token_ids"
    _count_column = "__quail_token_count"

    def __init__(self, path: str):
        self.path = path
        self._source = pa.memory_map(path, "r")
        self._reader = pa.ipc.open_file(self._source)
        self._documents_by_batch = []
        self._length_batches = []
        self._ends = []
        total = 0
        for index in range(self._reader.num_record_batches):
            batch = self._reader.get_batch(index)
            self._documents_by_batch.append(
                ArrowTokenDocuments(batch.column(self._token_column))
            )
            self._length_batches.append(batch.column(self._count_column))
            total += len(batch)
            self._ends.append(total)
        self._lengths = TokenLengths(self)

    @classmethod
    def write(
        cls,
        path: str,
        batches,
        *,
        document_column: str,
        projected_columns: tuple[str, ...],
        tokenizer,
        token_type: pa.DataType,
        source_schema: pa.Schema,
    ) -> "TokenStore":
        """Tokenize bounded input batches into one Arrow IPC file."""
        reserved = {cls._token_column, cls._count_column}
        conflict = reserved & set(projected_columns)
        if conflict:
            raise ValueError(
                f"source columns use reserved Quail names {sorted(conflict)}"
            )
        token_list_type = pa.large_list(token_type)
        fields = [
            pa.field(cls._token_column, token_list_type, nullable=False),
            pa.field(cls._count_column, pa.int64(), nullable=False),
        ]
        fields.extend(source_schema.field(name) for name in projected_columns)
        schema = pa.schema(
            fields,
            metadata={b"quail.kind": b"token_store"},
        )
        try:
            with pa.OSFile(path, "wb") as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    for batch in batches:
                        texts = batch.column(
                            batch.schema.get_field_index(document_column)
                        )
                        tokens = pa.array(
                            [tokenizer(text.as_py()) for text in texts],
                            type=token_list_type,
                        )
                        counts = pc.list_value_length(tokens).cast(
                            pa.int64()
                        )
                        arrays = [tokens, counts]
                        arrays.extend(
                            batch.column(batch.schema.get_field_index(name))
                            for name in projected_columns
                        )
                        writer.write_batch(
                            pa.RecordBatch.from_arrays(arrays, schema=schema)
                        )
        except Exception:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            raise
        return cls(path)

    @property
    def lengths(self) -> TokenLengths:
        """Return the file backed document length sequence."""
        return self._lengths

    @property
    def projected_columns(self) -> tuple[str, ...]:
        """Return source columns stored beside the token batches."""
        return tuple(
            name for name in self._reader.schema.names
            if name not in {self._token_column, self._count_column}
        )

    def column(self, name: str) -> pa.ChunkedArray:
        """Return one memory mapped source column."""
        if name not in self.projected_columns:
            raise KeyError(name)
        return pa.chunked_array(
            [
                self._reader.get_batch(index).column(name)
                for index in range(self._reader.num_record_batches)
            ],
            type=self._reader.schema.field(name).type,
        )

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        batch_index = bisect.bisect_right(self._ends, index)
        prior = self._ends[batch_index - 1] if batch_index else 0
        return batch_index, index - prior

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        batch_index, local = self._locate(index)
        return self._documents_by_batch[batch_index][local]

    def close(self) -> None:
        """Close the memory mapped Arrow file."""
        self._source.close()

    def select(self, indices) -> "TokenSelection":
        """Return a file reference for selected document positions."""
        return TokenSelection(self.path, indices)


class TokenSelection(Sequence):
    """Open selected token documents inside a child process."""

    def __init__(self, path: str, indices):
        self._path = path
        self._indices = indices
        self._store = None

    def _open(self) -> TokenStore:
        if self._store is None:
            self._store = TokenStore(self._path)
        return self._store

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._open()[self._indices[index]]

    def __getstate__(self):
        return self._path, self._indices

    def __setstate__(self, state):
        self._path, self._indices = state
        self._store = None


class DocumentPrefixes(Sequence):
    """Add common prefix tokens when a document is accessed."""

    def __init__(self, prefix, documents, indices):
        self._prefix = prefix
        self._documents = documents
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return chain_tokens(
            self._prefix,
            self._documents[self._indices[index]],
        )


class DocumentKeys(Sequence):
    """Build stable arena keys when a document is accessed."""

    def __init__(self, alias: str, indices):
        self._alias = alias
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._alias, self._indices[index]


def select_documents(documents, indices):
    """Select documents without copying file backed token buffers."""
    if isinstance(documents, TokenStore):
        return documents.select(indices)
    return [documents[index] for index in indices]


def chain_tokens(*parts):
    return TokenChain(*parts)


def decode_token_documents(value):
    if isinstance(value, ArrowTokenDocuments):
        return value
    import pyarrow as pa
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return ArrowTokenDocuments(value)
    return value


def decode_payload_documents(documents):
    return {alias: decode_token_documents(value)
            for alias, value in documents.items()}
