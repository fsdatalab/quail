"""Arrow transport and views for tokenized document columns."""

from __future__ import annotations

import bisect
from collections.abc import Sequence


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
    def __init__(self, array, owner=None):
        self._array = array
        self._owner = owner
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


def chain_tokens(*parts):
    return TokenChain(*parts)


def encode_token_documents(tokens) -> bytes:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table({"tokens": tokens})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def decode_token_documents(value):
    if isinstance(value, ArrowTokenDocuments):
        return value
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return value

    import pyarrow as pa
    import pyarrow.ipc as ipc

    owner = pa.py_buffer(value)
    table = ipc.open_stream(owner).read_all()
    array = table.column("tokens")
    return ArrowTokenDocuments(array, owner=(owner, table))


def decode_payload_documents(documents):
    return {alias: decode_token_documents(value)
            for alias, value in documents.items()}
