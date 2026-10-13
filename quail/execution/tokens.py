"""File backed token storage and Arrow token views."""

from __future__ import annotations

import bisect
import os
from collections.abc import Sequence

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc


class TokenView(Sequence):
    def __init__(self, values, character_offsets=None):
        self._values = values
        self._character_offsets = character_offsets

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
            offsets = (None if self._character_offsets is None else
                       self._character_offsets.slice(start, stop - start))
            return TokenView(self._values.slice(start, stop - start), offsets)
        return self._values[index].as_py()

    def numpy(self):
        return self._values.to_numpy(zero_copy_only=True)

    @property
    def character_offsets(self):
        """Return a zero-copy view of each token's original character range."""
        if self._character_offsets is None:
            return None
        return self._character_offsets.flatten().to_numpy(
            zero_copy_only=True).reshape(-1, 2)

    @property
    def arrow_array(self):
        return self._values

    @property
    def token_parts(self):
        return (self,)


class TokenChain(Sequence):
    """Token parts read in sequence without copying them."""

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
            start, stop, step = index.indices(self._length)
            if step != 1:
                return list(self)[index]
            return TokenChain(*self._sliced_parts(start, stop))
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        for part in self._parts:
            if index < len(part):
                return part[index]
            index -= len(part)
        raise IndexError(index)

    def _sliced_parts(self, start, stop):
        """The parts covering [start, stop), each sliced to it."""
        parts = []
        offset = 0
        for part in self._parts:
            end = offset + len(part)
            if end > start and offset < stop:
                parts.append(part[max(0, start - offset):stop - offset])
            offset = end
            if offset >= stop:
                break
        return parts

    @property
    def token_parts(self):
        return self._parts


class ArrowTokenDocuments(Sequence):
    def __init__(self, array, character_offsets=None):
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
        self._character_offsets = (
            None if character_offsets is None else
            tuple(character_offsets.chunks) if hasattr(character_offsets, "chunks")
            else (character_offsets,))

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
        bounds = None
        if self._character_offsets is not None:
            column = self._character_offsets[chunk_index]
            if column[local].is_valid:
                offset_start = column.offsets[local].as_py()
                offset_end = column.offsets[local + 1].as_py()
                bounds = column.values.slice(offset_start, offset_end - offset_start)
        return TokenView(values.slice(start, end - start), bounds)


class TokenLengths(Sequence):
    """Read document lengths from a token store without copying them."""

    def __init__(self, store: "TokenStore"):
        self._store = store

    def __len__(self):
        return len(self._store)

    @property
    def arrow_array(self) -> pa.ChunkedArray:
        return pa.chunked_array(self._store._length_batches, type=pa.int64())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        batch_index, local = self._store._locate(index)
        return self._store._length_batches[batch_index][local].as_py()


def _write_all(writer, batches):
    """Write every batch through one store writer and open the store."""
    try:
        for batch in batches:
            writer.write_batch(batch)
    except Exception:
        writer.abort()
        raise
    return writer.finish()


class _StoreWriter:
    """Write one memory mappable Arrow IPC file batch by batch."""

    def __init__(self, path: str, schema: pa.Schema):
        self.path = path
        self._schema = schema
        self._sink = pa.OSFile(path, "wb")
        self._writer = pa.ipc.new_file(self._sink, schema)

    def _write(self, arrays) -> None:
        self._writer.write_batch(
            pa.RecordBatch.from_arrays(arrays, schema=self._schema)
        )

    def _close(self) -> None:
        self._writer.close()
        self._sink.close()

    def abort(self) -> None:
        """Close the file and delete it."""
        try:
            self._close()
        finally:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class GigatokenEncoder:
    """Encode individual texts and Arrow batches with one Gigatoken tokenizer."""

    def __init__(self, source):
        from gigatoken import Tokenizer

        self.tokenizer = Tokenizer(source)

    def __call__(self, text):
        return [int(token) for token in self.tokenizer.encode(text)]

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def token_bytes(self):
        return self.tokenizer.vocab

    def batch_arrow(self, texts):
        """Return token IDs in Arrow without constructing Python token lists."""
        import awkward as ak

        source = ak.from_arrow(texts.cast(pa.large_string()))
        return ak.to_arrow(self.tokenizer.encode_batch(source),
                           extensionarray=False).cast(pa.large_list(pa.int32()))

    def batch_numpy(self, texts):
        """Return flat token IDs and row boundaries for short continuation texts."""
        encoded = self.tokenizer.encode_batch(texts, parallel=False).layout
        return np.asarray(encoded.content.data), np.asarray(encoded.offsets.data)


class TokenOffsets:
    """Map byte-preserving token sequences to original Unicode character ranges."""

    def __init__(self, vocabulary):
        self.vocabulary = pa.array(
            [vocabulary.get(index) for index in range(max(vocabulary, default=-1) + 1)],
            type=pa.binary())

    def __call__(self, tokens, texts):
        positions, exact = self.numpy(tokens, texts)
        offsets = tokens.offsets.to_numpy() - tokens.offsets[0].as_py()
        pairs = pa.FixedSizeListArray.from_arrays(pa.array(positions.reshape(-1)), 2)
        return pa.LargeListArray.from_arrays(offsets, pairs, mask=pc.invert(exact))

    def numpy(self, tokens, texts):
        """Return flat character ranges and the rows that preserve source bytes."""
        pieces = pc.take(self.vocabulary, tokens.flatten())
        offsets = tokens.offsets.to_numpy() - tokens.offsets[0].as_py()
        decoded = pc.binary_join(pa.LargeListArray.from_arrays(offsets, pieces), b"")
        source = texts.cast(pa.large_binary())
        exact = pc.fill_null(pc.equal(decoded, source), False)
        lengths = pc.fill_null(pc.binary_length(pieces), 0).to_numpy()
        _, boundaries, data = source.buffers()
        boundaries = np.frombuffer(boundaries, dtype=np.int64)[
            source.offset:source.offset + len(source) + 1]
        raw = np.frombuffer(data, dtype=np.uint8)[boundaries[0]:boundaries[-1]]
        boundaries = boundaries - boundaries[0]
        counts = np.diff(offsets)
        cumulative = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        ends = cumulative[1:] + np.repeat(
            boundaries[:-1] - cumulative[offsets[:-1]], counts)
        starts = ends - lengths
        if np.all(raw < 128):
            origin = np.repeat(boundaries[:-1], counts)
            positions = np.column_stack((starts - origin, ends - origin))
        else:
            chars = np.concatenate((np.flatnonzero((raw & 0xC0) != 0x80), [len(raw)]))
            origin = np.repeat(np.searchsorted(chars, boundaries[:-1]), counts)
            positions = np.column_stack((
                np.searchsorted(chars, starts, side="right") - 1 - origin,
                np.searchsorted(chars, ends, side="left") - origin))
        return positions, exact


class TokenStoreWriter(_StoreWriter):
    """Tokenize source batches into a token store file."""

    def __init__(self, path: str, *, document_column: str, tokenizer,
                 token_type: pa.DataType, offsets_encoder=None, existing_tokens=None):
        self._document_column = document_column
        self._tokenizer = tokenizer
        self._token_list_type = pa.large_list(token_type)
        self._offsets_encoder = offsets_encoder
        self._existing_tokens = existing_tokens
        self._row = 0
        self._offset_list_type = pa.large_list(pa.list_(pa.int64(), 2))
        schema = pa.schema(
            [
                pa.field(TokenStore._token_column, self._token_list_type,
                         nullable=False),
                pa.field(TokenStore._count_column, pa.int64(),
                         nullable=False),
            ],
            metadata={b"quail.kind": b"token_store"},
        )
        if offsets_encoder is not None:
            schema = schema.append(pa.field(
                TokenStore._offset_column, self._offset_list_type))
        super().__init__(path, schema)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Tokenize the document column of one source batch."""
        texts = batch.column(
            batch.schema.get_field_index(self._document_column)
        )
        offsets = None
        batch_arrow = getattr(self._tokenizer, "batch_arrow", None)
        if self._existing_tokens is not None:
            selected = self._existing_tokens.slice(self._row, len(texts))
            if len(selected) != len(texts):
                raise ValueError("source rows no longer match the stored tokens")
            tokens = (selected.chunk(0) if selected.num_chunks == 1
                      else selected.combine_chunks())
            self._row += len(texts)
        elif batch_arrow is None:
            tokens = pa.array(
                [self._tokenizer(text.as_py()) for text in texts],
                type=self._token_list_type)
        else:
            tokens = batch_arrow(texts).cast(self._token_list_type)
        if self._offsets_encoder is not None:
            offsets = self._offsets_encoder(tokens, texts)
        counts = pc.list_value_length(tokens).cast(pa.int64())
        self._write([tokens, counts] if offsets is None else [tokens, counts, offsets])

    def finish(self) -> "TokenStore":
        """Close the file and open it as a token store."""
        self._close()
        return TokenStore(self.path)


class ColumnStoreWriter(_StoreWriter):
    """Copy one source column into a column store file."""

    def __init__(self, path: str, field: pa.Field):
        self._name = field.name
        super().__init__(
            path,
            pa.schema([field], metadata={b"quail.kind": b"column_store"}),
        )

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Copy the stored column out of one source batch."""
        self._write([batch.column(batch.schema.get_field_index(self._name))])

    def finish(self) -> "ColumnStore":
        """Close the file and open it as a column store."""
        self._close()
        return ColumnStore(self.path)


class ColumnStore:
    """Provide one memory mapped source column for result rows."""

    def __init__(self, path: str):
        self.path = path
        self._source = pa.memory_map(path, "r")
        self._reader = pa.ipc.open_file(self._source)
        self.name = self._reader.schema.names[0]
        self._values = None

    @classmethod
    def write(cls, path: str, batches, field: pa.Field) -> "ColumnStore":
        """Copy one column of bounded input batches into an Arrow IPC file."""
        return _write_all(ColumnStoreWriter(path, field), batches)

    @property
    def values(self) -> pa.ChunkedArray:
        """Return the column as one memory mapped chunked array."""
        if self._values is None:
            self._values = pa.chunked_array(
                [
                    self._reader.get_batch(index).column(0)
                    for index in range(self._reader.num_record_batches)
                ],
                type=self._reader.schema.field(0).type,
            )
        return self._values

    def __len__(self) -> int:
        return len(self.values)

    def close(self) -> None:
        """Close the memory mapped Arrow file."""
        self._source.close()


class TokenStore(Sequence):
    """Provide random token access from one memory mapped Arrow file."""

    _token_column = "__quail_token_ids"
    _count_column = "__quail_token_count"
    _offset_column = "__quail_token_offsets"

    def __init__(self, path: str):
        self.path = path
        self._source = pa.memory_map(path, "r")
        self._reader = pa.ipc.open_file(self._source)
        self.has_offsets = self._offset_column in self._reader.schema.names
        self._documents_by_batch = []
        self._length_batches = []
        self._ends = []
        total = 0
        for index in range(self._reader.num_record_batches):
            batch = self._reader.get_batch(index)
            self._documents_by_batch.append(
                ArrowTokenDocuments(
                    batch.column(self._token_column),
                    batch.column(self._offset_column) if self.has_offsets else None)
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
        tokenizer,
        token_type: pa.DataType,
        offsets_encoder=None,
    ) -> "TokenStore":
        """Tokenize bounded input batches into one Arrow IPC file."""
        writer = TokenStoreWriter(
            path,
            document_column=document_column,
            tokenizer=tokenizer,
            token_type=token_type,
            offsets_encoder=offsets_encoder,
        )
        return _write_all(writer, batches)

    @property
    def lengths(self) -> TokenLengths:
        """Return the file backed document length sequence."""
        return self._lengths

    @property
    def arrow_array(self) -> pa.ChunkedArray:
        """Return the stored token batches without copying their buffers."""
        return pa.chunked_array(
            [self._reader.get_batch(index).column(self._token_column)
             for index in range(self._reader.num_record_batches)],
            type=self._reader.schema.field(self._token_column).type)

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


class ScanInput:
    """The tokenized documents of one scan and its stored value columns."""

    def __init__(self, tokens: "TokenStore", columns: dict):
        self.tokens = tokens
        self._columns = dict(columns)

    @property
    def lengths(self) -> TokenLengths:
        """Return the document token lengths."""
        return self.tokens.lengths

    @property
    def projected_columns(self) -> tuple[str, ...]:
        """Return the value columns stored for this scan."""
        return tuple(self._columns)

    def column(self, name: str) -> pa.ChunkedArray:
        """Return one stored value column."""
        return self._columns[name].values


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

    def select(self, indices) -> "TokenSelection":
        """Return a file reference for positions of this selection."""
        return TokenSelection(
            self._path, [self._indices[index] for index in indices])


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
    if isinstance(documents, (TokenStore, TokenSelection)):
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


def longest_common_prefix(left, right) -> int:
    """Return the length of the token prefix two sequences share."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def shared_prefix_lengths(sequences) -> list[int]:
    """Return, per sequence, the prefix tokens another sequence also has.

    The values sum to the tokens a prefix trie over the sequences
    saves: an execution with unlimited KV that computes every distinct
    prefix once pays for each sequence only beyond its shared length.
    The sum does not depend on the order the sequences are computed
    in; a sequence's shared length is the longest common prefix with
    the lexicographic predecessor.
    """
    return prefix_tree(sequences, 1).shared


def _token_array(document):
    """A document's token ids as an int32 array, without a Python list."""
    import numpy as np

    parts = getattr(document, "token_parts", (document,))
    arrays = []
    for part in parts:
        if hasattr(part, "numpy"):
            try:
                arrays.append(np.asarray(part.numpy(), dtype=np.int32))
                continue
            except (TypeError, ValueError):
                pass
        arrays.append(np.asarray(list(part), dtype=np.int32))
    if not arrays:
        return np.empty(0, dtype=np.int32)
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


class PrefixTree:
    """Which document each document borrows its KV prefix from.

    Documents are visited in sorted token order. A document's shared
    length is the tokens it has in common with its predecessor,
    rounded down to a whole page and short of its last token, so it
    can read those pages instead of computing them. Its parent is the
    earliest document that computed those pages itself: the
    predecessor, or an ancestor of the predecessor that borrowed at
    least as much. Documents sharing
    one prefix thus all borrow from the same parent, and a forward
    pass can read that parent's pages once for all of them. A
    document with nothing to borrow has parent None and shared 0.

    Attributes:
        order: Document positions in sorted token order; a parent
            always comes before its children.
        parent: Per document, the position it borrows from, or None.
        shared: Per document, the borrowed tokens, a multiple of the
            page size.
    """

    def __init__(self, order, parent, shared):
        self.order = order
        self.parent = parent
        self.shared = shared

    @property
    def shared_tokens(self) -> int:
        """Tokens the tree lets the engine read instead of compute."""
        return sum(self.shared)


def prefix_tree(documents, page_tokens: int) -> PrefixTree:
    """Build the prefix tree of a document set.

    Args:
        documents: Token sequences, indexable by position.
        page_tokens: Tokens per KV page; shares round down to it.
    """
    import numpy as np

    n = len(documents)
    try:
        arrays = [_token_array(documents[i]) for i in range(n)]
    except (TypeError, ValueError):
        # tokens that are not integers (a test tokenizer's words)
        arrays = [tuple(documents[i]) for i in range(n)]
        keys = arrays
        common = longest_common_prefix
    else:
        # big-endian bytes sort like the token ids they encode
        keys = [a.astype(">i4").tobytes() for a in arrays]

        def common(a, b):
            limit = min(len(a), len(b))
            differ = np.flatnonzero(a[:limit] != b[:limit])
            return int(differ[0]) if len(differ) else limit
    order = sorted(range(n), key=keys.__getitem__)
    parent = [None] * n
    shared = [0] * n
    previous = None
    for index in order:
        if previous is not None:
            # a document computes at least its last token, so a
            # duplicate still packs a prefix of its own
            lcp = min(common(arrays[previous], arrays[index]),
                      max(len(arrays[index]) - 1, 0))
            pages = lcp // page_tokens * page_tokens
            if pages:
                # climb while the ancestor borrowed these pages too
                owner = previous
                while parent[owner] is not None and shared[owner] >= pages:
                    owner = parent[owner]
                parent[index] = owner
                shared[index] = pages
        previous = index
    return PrefixTree(order, parent, shared)
