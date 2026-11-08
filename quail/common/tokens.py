"""File backed token storage and Arrow token views."""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict

import _quail_native
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.common.selection import group_ids


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


# a model's tokenizer, run in Rust: the planner writes token stores with it
# and the GPU worker encodes continuations and decodes answers with it
Tokenizer = _quail_native.Tokenizer


class ColumnStore:
    """Provide one memory mapped source column for result rows."""

    def __init__(self, path: str):
        self.path = path
        with ExitStack() as resources:
            self._source = resources.enter_context(pa.memory_map(path, "r"))
            self._reader = resources.enter_context(pa.ipc.open_file(self._source))
            self.name = self._reader.schema.names[0]
            self._values = None
            self._resources = resources.pop_all()

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
        self._resources.close()


def token_keys(tokens: pa.ChunkedArray) -> pa.ChunkedArray:
    """Each document's tokens as one binary value, to group equal documents.

    Integer tokens are viewed as their bytes without copying the token
    buffer; the words of a test tokenizer are joined.
    """
    chunks = []
    for chunk in tokens.chunks:
        values = chunk.values
        if not pa.types.is_integer(values.type):
            # a word holds no control character, so the join is exact
            chunks.append(pc.binary_join(chunk, "\x00").cast(pa.large_binary()))
            continue
        width = values.type.byte_width
        offsets = (chunk.offsets.to_numpy().astype(np.int64)
                   + values.offset) * width
        chunks.append(pa.Array.from_buffers(
            pa.large_binary(), len(chunk),
            [None, pa.py_buffer(offsets), values.buffers()[1]]))
    return pa.chunked_array(chunks, pa.large_binary())


class TokenStore(Sequence):
    """Provide random token access from one memory mapped Arrow file."""

    _token_column = "__quail_token_ids"
    _count_column = "__quail_token_count"
    _offset_column = "__quail_token_offsets"

    def __init__(self, path: str):
        self.path = path
        with ExitStack() as resources:
            self._source = resources.enter_context(pa.memory_map(path, "r"))
            self._reader = resources.enter_context(pa.ipc.open_file(self._source))
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
                        batch.column(self._offset_column) if self.has_offsets
                        else None))
                self._length_batches.append(batch.column(self._count_column))
                total += len(batch)
                self._ends.append(total)
            self._lengths = TokenLengths(
                pa.chunked_array(self._length_batches, type=pa.int64()), self)
            self._prefix_index = None
            self._first_documents = None
            self._resources = resources.pop_all()

    @property
    def lengths(self) -> TokenLengths:
        """Return the file backed document length sequence."""
        return self._lengths

    @property
    def prefix_index(self) -> "PrefixIndex":
        """Return the document order and common prefix lengths, computed once."""
        if self._prefix_index is None:
            self._prefix_index = PrefixIndex(self)
        return self._prefix_index

    @property
    def first_documents(self) -> pa.Array:
        """Return each document's first document with the same tokens, as int32.

        A document whose tokens no earlier document has is its own first.
        """
        if self._first_documents is None:
            self._first_documents = self.first_documents_with([])
        return self._first_documents

    def first_documents_with(self, columns) -> pa.Array:
        """Each document's first document with its tokens and its row values.

        The documents are grouped by hashing their token bytes, like a
        GROUP BY, so no prefix index is built.

        Args:
            columns: Arrow arrays, one value per document; two documents
                share a first document only if they are equal in each.
        """
        if not len(self):
            return pa.array([], pa.int32())
        groups = group_ids([token_keys(self.arrow_array), *columns])
        _, first, inverse = np.unique(groups, return_index=True,
                                      return_inverse=True)
        return pa.array(first[inverse], pa.int32())

    @property
    def arrow_array(self) -> pa.ChunkedArray:
        """Return the stored token batches without copying their buffers."""
        return pa.chunked_array(
            [self._reader.get_batch(index).column(self._token_column)
             for index in range(self._reader.num_record_batches)],
            type=self._reader.schema.field(self._token_column).type)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        batch_index, local = _locate_chunk(index, len(self), self._ends)
        return self._documents_by_batch[batch_index][local]

    def close(self) -> None:
        """Close the memory mapped Arrow file."""
        self._prefix_index = None
        self._first_documents = None
        self._resources.close()

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


def select_documents(documents, indices):
    """Select documents without copying file backed token buffers."""
    if isinstance(documents, (TokenStore, TokenSelection)):
        return documents.select(indices)
    return [documents[index] for index in indices]


def concat_tokens(*parts):
    return TokenConcat(*parts)


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
        page_tokens: The page size of a tree built from the documents'
            tokens in sorted order, which GrowingPrefixTree can extend;
            None for a tree given by its parents.
    """

    def __init__(self, order, parent, shared, page_tokens=None):
        self.order = order
        self.parent = parent
        self.shared = shared
        self.page_tokens = page_tokens
        self._reader_groups = None

    @property
    def shared_tokens(self) -> int:
        """Tokens the tree lets the engine read instead of compute."""
        return sum(self.shared)

    def reader_groups(self, lengths):
        """Group document rows by their borrowed parent and prefix length."""
        if self._reader_groups is None:
            rows = pc.subtract(lengths.arrow_array, pa.array(self.shared, pa.int64()))
            table = pa.table({"parent": pa.array(self.parent, pa.int64()),
                              "shared": pa.array(self.shared, pa.int64()),
                              "rows": rows}).drop_null()
            groups = table.group_by(["parent", "shared"]).aggregate(
                [("rows", "count"), ("rows", "sum")])
            self._reader_groups = tuple(
                groups[name].to_numpy()
                for name in ("shared", "rows_count", "rows_sum"))
        return self._reader_groups


class PrefixIndex:
    """Sorted document positions and common prefixes shared by page-size trees."""

    def __init__(self, documents):
        n = len(documents)
        try:
            arrays = [_token_array(documents[i]) for i in range(n)]
        except (TypeError, ValueError):
            # Test tokenizers may return words instead of integer token IDs.
            arrays = [tuple(documents[i]) for i in range(n)]
            keys = arrays
            common = longest_common_prefix
        else:
            # Big-endian bytes sort in token order for nonnegative token IDs.
            keys = [a.astype(">i4").tobytes() for a in arrays]
            common = _array_common_prefix
        self.order = sorted(range(n), key=keys.__getitem__)
        self.shared = np.zeros(n, dtype=np.int64)
        # Every document computes its final token, including exact duplicates.
        for previous, index in zip(self.order, self.order[1:]):
            self.shared[index] = min(common(arrays[previous], arrays[index]),
                                     max(len(arrays[index]) - 1, 0))
        self._trees = {}

    def tree(self, page_tokens: int) -> PrefixTree:
        """Return borrowed parents and lengths rounded to a page size."""
        if page_tokens not in self._trees:
            shared = (self.shared // page_tokens * page_tokens).tolist()
            parent = [None] * len(shared)
            for previous, index in zip(self.order, self.order[1:]):
                pages = shared[index]
                if pages:
                    owner = previous
                    while parent[owner] is not None and shared[owner] >= pages:
                        owner = parent[owner]
                    parent[index] = owner
            self._trees[page_tokens] = PrefixTree(self.order, parent, shared,
                                                  page_tokens)
        return self._trees[page_tokens]


def _sort_key(document):
    """A document's key in sorted token order, and its tokens to compare.

    Returns (key, tokens, common), where common(left, right) counts the
    tokens two documents' tokens share; PrefixIndex sorts by the same keys.
    """
    try:
        array = _token_array(document)
    except (TypeError, ValueError):
        # Test tokenizers may return words instead of integer token IDs.
        words = tuple(document)
        return words, words, longest_common_prefix
    return array.astype(">i4").tobytes(), array, _array_common_prefix


class GrowingPrefixTree:
    """A prefix tree over the documents seen so far, extended as more arrive.

    It starts from a PrefixTree that prefix_tree built in sorted token
    order and keeps the same rules as documents arrive: add() places
    each new document in sorted order, ties after the documents before
    it, and recomputes from the first place that changed the shared
    tokens (with the document before it in the order, page-rounded and
    short of its last token) and the parent of every document not
    fixed. A fixed document, one admitted already, keeps its parent and
    shared tokens. With no document fixed, the tree after any sequence
    of add() calls equals prefix_tree over every document: the same
    order, parents, and shared tokens.

    Args:
        tree: A PrefixTree with page_tokens set, over the first documents.

    Attributes:
        order, parent, shared: As PrefixTree's, over the documents seen.
    """

    def __init__(self, tree):
        if tree.page_tokens is None:
            raise ValueError("only a tree in sorted token order can grow")
        self.page_tokens = tree.page_tokens
        self.order = list(tree.order)
        self.parent = list(tree.parent)
        self.shared = list(tree.shared)

    @property
    def shared_tokens(self) -> int:
        """Tokens the tree lets the engine read instead of compute."""
        return sum(self.shared)

    def add(self, documents, fixed=frozenset()) -> list:
        """Place the documents past the ones seen so far.

        Args:
            documents: Token sequences of every document, those seen
                and the new ones after them, indexable by position.
            fixed: Positions whose parent and shared tokens stay.

        Returns:
            The positions seen before whose parent or shared tokens changed.
        """
        seen = len(self.parent)
        if len(documents) <= seen:
            return []
        keys = {}

        def key(position):
            if position not in keys:
                keys[position] = _sort_key(documents[position])
            return keys[position]

        order = self.order
        for position in range(seen, len(documents)):
            low, high = 0, len(order)
            mine = key(position)[0]
            while low < high:
                middle = (low + high) // 2
                if key(order[middle])[0] <= mine:
                    low = middle + 1
                else:
                    high = middle
            order.insert(low, position)
            self.parent.append(None)
            self.shared.append(0)
        place = {document: index for index, document in enumerate(order)}
        new = range(seen, len(documents))
        first = min(place[position] for position in new)
        before = {document: (self.parent[document], self.shared[document])
                  for document in order[first:] if document < seen}
        # a document's shared tokens change only with the one before it
        moved = set(new)
        moved.update(order[place[position] + 1] for position in new
                     if place[position] + 1 < len(order))
        page = self.page_tokens
        for document in moved - set(fixed):
            index = place[document]
            if not index:
                self.shared[document] = 0
                continue
            mine, tokens, common = key(document)
            _, previous, _ = key(order[index - 1])
            count = min(common(previous, tokens), max(len(tokens) - 1, 0))
            self.shared[document] = count // page * page
        for index in range(first, len(order)):
            document = order[index]
            if document in fixed:
                continue
            pages = self.shared[document]
            owner = None
            if pages and index:
                owner = order[index - 1]
                while (self.parent[owner] is not None
                       and self.shared[owner] >= pages):
                    owner = self.parent[owner]
            self.parent[document] = owner
        return [document for document, was in before.items()
                if (self.parent[document], self.shared[document]) != was]


def _array_common_prefix(left, right) -> int:
    """Find the first unequal token without materializing every mismatch index."""
    limit = min(len(left), len(right))
    if not limit:
        return 0
    differ = left[:limit] != right[:limit]
    first = int(differ.argmax())
    return first if differ[first] else limit


def prefix_tree(documents, page_tokens: int) -> PrefixTree:
    """Build the prefix tree of a document set.

    Args:
        documents: Token sequences, indexable by position.
        page_tokens: Tokens per KV page; shares round down to it.
    """
    index = (documents.prefix_index if isinstance(documents, TokenStore)
             else PrefixIndex(documents))
    return index.tree(page_tokens)


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
