"""In-memory token stores and Arrow token views."""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import asdict

import _quail_native
import numpy as np
import pyarrow as pa

from quail.common.selection import equality_keys


class TokenView(Sequence):
    def __init__(self, values, character_offsets=None):
        self._values = values
        self._character_offsets = character_offsets

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self.numpy())

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


class TokenConcat(Sequence):
    """Token parts read in sequence without copying them."""

    def __init__(self, *parts):
        flattened = []
        for part in parts:
            if isinstance(part, TokenConcat):
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
            return TokenConcat(*self._sliced_parts(start, stop))
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


def _locate_chunk(index, length, ends) -> tuple[int, int]:
    """Locate a sequence index in chunks with cumulative end positions."""
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError(index)
    chunk = bisect.bisect_right(ends, index)
    return chunk, index - (ends[chunk - 1] if chunk else 0)


class ArrowTokenDocuments(Sequence):
    def __init__(self, array, character_offsets=None):
        self._array = array
        self.character_offsets = character_offsets
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
        chunk_index, local = _locate_chunk(index, len(self), self._ends)
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
    """Per document token counts, exact from a token store or estimated.

    Attributes:
        arrow_array: The counts as an int64 chunked array.
        store: The token store the counts were read from, or None for
            an estimate.
    """

    def __init__(self, values, store: "TokenStore | None" = None):
        self.arrow_array = (values.cast(pa.int64())
                            if isinstance(values, pa.ChunkedArray)
                            else pa.chunked_array([pa.array(values, pa.int64())]))
        self.store = store

    def __len__(self):
        return len(self.arrow_array)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.arrow_array[index].to_pylist()
        return self.arrow_array[index].as_py()

    def __array__(self, dtype=None, copy=None):
        values = self.arrow_array.to_numpy()
        return values if dtype is None else values.astype(dtype, copy=False)


# a model's tokenizer, run in Rust: the planner tokenizes documents with it
# and the GPU worker encodes continuations and decodes answers with it
Tokenizer = _quail_native.Tokenizer


class TokenStore(Sequence):
    """Provide random token access to a token store Rust built in memory.

    Args:
        table: The store's columns: each document's token ids and token
            count, and optionally each token's character range.
    """

    _token_column = "__quail_token_ids"
    _count_column = "__quail_token_count"
    _offset_column = "__quail_token_offsets"

    def __init__(self, table: pa.Table):
        self._table = table
        self.has_offsets = self._offset_column in table.column_names
        self._documents_by_batch = []
        self._ends = []
        total = 0
        for batch in table.to_batches():
            self._documents_by_batch.append(ArrowTokenDocuments(
                batch.column(self._token_column),
                batch.column(self._offset_column) if self.has_offsets else None))
            total += len(batch)
            self._ends.append(total)
        self._lengths = TokenLengths(table.column(self._count_column), self)
        self._prefix_index = None
        self._trees = {}

    def __reduce__(self):
        return TokenStore, (self._table,)

    @property
    def table(self) -> pa.Table:
        """Return the store's columns, over the buffers Rust built."""
        return self._table

    @property
    def lengths(self) -> TokenLengths:
        """Return the document length sequence."""
        return self._lengths

    def prefix_tree(self, page_tokens: int) -> "PrefixTree":
        """Return the store's prefix tree at a page size.

        The documents are sorted once; each page size's tree is built
        once from that sort.
        """
        if self._prefix_index is None:
            self._prefix_index = _quail_native.PrefixIndex(
                pa.table({"tokens": self.arrow_array}))
        if page_tokens not in self._trees:
            self._trees[page_tokens] = PrefixTree(
                *self._prefix_index.tree(page_tokens))
        return self._trees[page_tokens]

    def first_documents_with(self, columns) -> pa.Array:
        """Each document's first document with its tokens and its row values.

        Args:
            columns: Arrow arrays, one value per document; two documents
                share a first document only if they are equal in each,
                with nulls equal, NaNs equal, and -0.0 equal to 0.0.
        """
        values = (pa.table({str(index): equality_keys(column, nan_equal=True)
                            for index, column in enumerate(columns)})
                  if columns else None)
        return _quail_native.first_documents(
            pa.table({"tokens": self.arrow_array}), values)

    @property
    def arrow_array(self) -> pa.ChunkedArray:
        """Return the stored token batches without copying their buffers."""
        return self._table.column(self._token_column)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        batch_index, local = _locate_chunk(index, len(self), self._ends)
        return self._documents_by_batch[batch_index][local]


class ScanInput:
    """The tokenized documents of one scan and its value columns.

    Args:
        tokens: The TokenStore of the scan's document column.
        columns: The value columns the executor reads, one row per
            document.
    """

    def __init__(self, tokens: "TokenStore", columns: pa.Table):
        self.tokens = tokens
        self._columns = columns

    @property
    def lengths(self) -> TokenLengths:
        """Return the document token lengths."""
        return self.tokens.lengths

    @property
    def projected_columns(self) -> tuple[str, ...]:
        """Return the value columns kept for this scan."""
        return tuple(self._columns.column_names)

    def column(self, name: str) -> pa.ChunkedArray:
        """Return one value column."""
        return self._columns.column(name)


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
        return concat_tokens(
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


def concat_tokens(*parts):
    return TokenConcat(*parts)


def decode_token_documents(value):
    if isinstance(value, ArrowTokenDocuments):
        return value
    import pyarrow as pa
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return ArrowTokenDocuments(value)
    return value


def token_table(documents) -> pa.Table:
    """Return token documents as a table another process can read.

    ``documents_from_table`` reads the documents back over the table's
    buffers.
    """
    if isinstance(documents, TokenStore):
        return documents.table
    if isinstance(documents, ArrowTokenDocuments):
        columns = {"tokens": documents._array}
        if documents.character_offsets is not None:
            columns["offsets"] = documents.character_offsets
        return pa.table(columns)
    return _token_lists(documents)


def documents_from_table(table: pa.Table):
    """Return the token documents ``token_table`` wrote."""
    if TokenStore._token_column in table.column_names:
        return TokenStore(table)
    return ArrowTokenDocuments(
        table.column("tokens"),
        table.column("offsets") if "offsets" in table.column_names else None)


def _token_array(document):
    """A document's integer token ids as an int32 array."""
    parts = getattr(document, "token_parts", (document,))
    arrays = [np.asarray(part.numpy() if hasattr(part, "numpy") else list(part),
                         dtype=np.int32)
              for part in parts]
    if not arrays:
        return np.empty(0, dtype=np.int32)
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def _token_lists(documents) -> pa.Table:
    """Documents' integer token ids as a table of one int32 list column."""
    arrays = [_token_array(documents[index]) for index in range(len(documents))]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([len(array) for array in arrays], out=offsets[1:])
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)
    return pa.table({"tokens": pa.LargeListArray.from_arrays(
        pa.array(offsets), pa.array(values))})


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
        documents: A TokenStore, or integer token sequences indexable by
            position.
        page_tokens: Tokens per KV page; shares round down to it.
    """
    if isinstance(documents, TokenStore):
        return documents.prefix_tree(page_tokens)
    index = _quail_native.PrefixIndex(_token_lists(documents))
    return PrefixTree(*index.tree(page_tokens))


def model_answers(spec, tokenizer) -> tuple[set, set, tuple[int, ...]]:
    """Return a model's TRUE ids, FALSE ids, and decision readout offsets.

    Rust's planner reads the same ids into every plan; the GPU worker
    calls this while it loads the model, before any plan arrives.

    Args:
        spec: The model's ModelSpec.
        tokenizer: Callable text -> token ids, without special tokens.
    """
    true_ids, false_ids, offsets = _quail_native.model_answers(
        json.dumps(asdict(spec)), tokenizer)
    return set(true_ids), set(false_ids), tuple(offsets)
