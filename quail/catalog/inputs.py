"""Tokenized document columns and value columns, cached per session.

An ``InputStore`` tokenizes a provider's document column once into an
Arrow token file and keeps each value column a query needs in its own
file, so later queries read only what they project. It also samples a
column for the planner's estimates.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing
from itertools import chain
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog.providers import Catalog, ScanRequest
from quail.catalog.tokens import (
    ColumnStoreWriter,
    ScanInput,
    TokenOffsets,
    TokenStoreWriter,
)
from quail.progress import Progress, say

# rows the planner samples to estimate selectivities and token lengths
ESTIMATE_SAMPLE = 1024
# rows written per tokenization step, so progress stays responsive
TOKENIZE_ROWS = 2048


class InputStore:
    """Token and value column files for the tables of one session.

    Args:
        catalog: The session's registered tables.
        tokenizer: A callable returning the session's encoder; called
            when a column is first tokenized.
        tokenizer_name: The encoder's name for the report notes.
    """

    def __init__(self, catalog: Catalog, *, tokenizer, tokenizer_name: str):
        self.catalog = catalog
        self._tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.notes = []            # tokenizer picks etc., for reports
        self._token_offsets = None
        self._token_stores = {}
        self._replaced_token_stores = []
        self._column_stores = {}
        self.num_stores = 0
        self._token_directory = None
        self._corpus_tokenizers = {}
        self._length_estimates = {}
        self._lock = threading.RLock()
        self._background = None

    @property
    def tokenizer(self):
        return self._tokenizer()

    def close(self):
        """Wait for background tokenization and remove the token files."""
        if self._background is not None:
            self._background.shutdown(cancel_futures=True)
        for store in self._token_stores.values():
            store.close()
        self._token_stores.clear()
        for store in self._replaced_token_stores:
            store.close()
        self._replaced_token_stores.clear()
        for store in self._column_stores.values():
            store.close()
        self._column_stores.clear()
        if self._token_directory is not None:
            self._token_directory.cleanup()
            self._token_directory = None

    def _offsets_encoder(self):
        """Return the original-text offset mapper for the session tokenizer."""
        token_bytes = getattr(self.tokenizer, "token_bytes", None)
        if token_bytes is None:
            return None
        if self._token_offsets is None:
            self._token_offsets = TokenOffsets(token_bytes())
        return self._token_offsets

    def tokenize(self, provider_name: str, column: str,
                 projected_columns=(), *, with_offsets=False) -> ScanInput:
        """Tokenize one document column and keep value columns beside it.

        The token file is cached per document column. Each value column
        is cached in its own file, so a later query that returns other
        columns reads only those columns from the provider and does not
        tokenize the documents again.
        """
        with self._lock:
            return self._tokenize(provider_name, column, projected_columns,
                                  with_offsets)

    def _tokenize(self, provider_name, column, projected_columns, with_offsets):
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        projected_columns = tuple(dict.fromkeys(projected_columns))
        token_key = (identity, column)
        missing = [
            name for name in projected_columns
            if (identity, name) not in self._column_stores
        ]
        stored = self._token_stores.get(token_key)
        offsets_encoder = (self._offsets_encoder() if with_offsets
                           and (stored is None or not stored.has_offsets) else None)
        needs_tokens = stored is None or offsets_encoder is not None
        if needs_tokens or missing:
            self._load(
                provider_name,
                column if needs_tokens else None,
                tuple(missing),
                offsets_encoder=offsets_encoder,
            )
        return ScanInput(
            self._token_stores[token_key],
            {
                name: self._column_stores[(identity, name)]
                for name in projected_columns
            },
        )

    def _background_executor(self) -> ThreadPoolExecutor:
        if self._background is None:
            self._background = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="quail-tokenize")
        return self._background

    def warm_tokenizer(self) -> Future:
        """Load the background thread's encoder before any column needs it.

        The encoder keeps one instance per thread, and loading one takes
        a few hundred milliseconds that would otherwise land in the first
        query's input preparation.
        """
        return self._background_executor().submit(self.tokenizer, "")

    def tokenize_async(self, provider_name: str, column: str,
                       projected_columns=(), *, with_offsets=False) -> Future:
        """Start tokenize() on a background thread and return its Future."""
        return self._background_executor().submit(
            self.tokenize, provider_name, column, projected_columns,
            with_offsets=with_offsets)

    def sample_columns(self, provider_name: str, columns) -> dict:
        """Return the first ESTIMATE_SAMPLE rows of some source columns.

        The planner estimates selectivities from this sample; it never
        reads a whole column.
        """
        provider = self.catalog.get(provider_name)
        columns = tuple(dict.fromkeys(columns))
        reader = provider.scan(ScanRequest(columns=columns,
                                           limit=ESTIMATE_SAMPLE))
        batches = []
        rows = 0
        try:
            for batch in reader:
                batches.append(batch.slice(0, ESTIMATE_SAMPLE - rows))
                rows += batches[-1].num_rows
                if rows >= ESTIMATE_SAMPLE:
                    break
        finally:
            reader.close()
        schema = provider.schema()
        return {
            column: pa.chunked_array(
                [batch.column(index) for batch in batches],
                type=schema.field(column).type)
            for index, column in enumerate(columns)
        }

    def token_lengths(self, provider_name: str, column: str):
        """Return the exact token counts when the column is tokenized."""
        identity = self.catalog.get(provider_name).content_identity()
        store = self._token_stores.get((identity, column))
        return None if store is None else store.lengths

    def estimate_lengths(self, provider_name: str, column: str) -> list[int]:
        """Estimate document token counts from a tokenized sample.

        Tokenizes the first ESTIMATE_SAMPLE documents and repeats their
        lengths over the provider's row count, so the planner sees the
        sample's length distribution without a pass over the table. A
        provider that reports no row count is scanned for its byte
        lengths, which the sample's tokens per byte scale.
        """
        provider = self.catalog.get(provider_name)
        key = (provider.content_identity(), column)
        if key in self._length_estimates:
            return self._length_estimates[key]
        started = time.perf_counter()
        num_rows = provider.statistics().num_rows
        sample = self.sample_columns(provider_name, (column,))[column].to_pylist()
        # Defer fast-tokenizer initialization to the background tokenization pass.
        tokenizer = self.tokenizer
        sample_lengths = [max(1, len(tokenizer(text))) for text in sample]
        if num_rows is not None:
            lengths = [sample_lengths[index % len(sample_lengths)]
                       for index in range(num_rows)] if sample_lengths else []
        else:
            sample_tokens = sum(sample_lengths)
            sample_bytes = sum(len(text.encode("utf-8")) for text in sample)
            ratio = sample_tokens / sample_bytes if sample_bytes else 0.0
            lengths = []
            reader = provider.scan(ScanRequest(columns=(column,)))
            try:
                for batch in reader:
                    lengths.extend(
                        max(1, round(n * ratio)) for n in pc.binary_length(
                            batch.column(0)).cast(pa.int64()).to_pylist())
            finally:
                reader.close()
        self._length_estimates[key] = lengths
        say(f"estimated {provider_name}.{column}: {len(lengths):,} documents, "
            f"about {sum(lengths):,} tokens from a {len(sample)} document "
            f"sample, {time.perf_counter() - started:.1f} s")
        return lengths

    def _corpus_tokenizer(self, provider_name, column, texts):
        """Cache the session tokenizer and token type for one column."""
        key = (self.catalog.get(provider_name).content_identity(), column)
        if key not in self._corpus_tokenizers:
            tokenizer = self.tokenizer
            self.notes.append(
                f"{provider_name}.{column}: tokenizer: {self.tokenizer_name}"
            )
            first_token = next(
                (token for text in texts for token in tokenizer(text)), None
            )
            token_type = (
                pa.int32()
                if first_token is None or isinstance(first_token, Integral)
                else pa.string()
            )
            self._corpus_tokenizers[key] = (tokenizer, token_type)
        return self._corpus_tokenizers[key]

    def _store_path(self) -> str:
        if self._token_directory is None:
            self._token_directory = TemporaryDirectory(
                prefix="quail-tokens-"
            )
        self.num_stores += 1
        return str(
            Path(self._token_directory.name)
            / f"input-{self.num_stores}.arrow"
        )

    def _pick_tokenizer(self, reader, provider_name: str, column: str):
        """Sample the first documents to choose a tokenizer and token type."""
        buffered = []
        text_sample = []
        while len(text_sample) < 25:
            try:
                batch = next(reader)
            except StopIteration:
                break
            buffered.append(batch)
            texts = batch.column(batch.schema.get_field_index(column))
            needed = 25 - len(text_sample)
            text_sample.extend(texts.slice(0, needed).to_pylist())
        tokenizer, token_type = self._corpus_tokenizer(
            provider_name, column, text_sample)
        return buffered, tokenizer, token_type

    def _load(self, provider_name: str, column: str | None,
              value_columns: tuple[str, ...], *, offsets_encoder=None) -> None:
        """Scan the provider once and write the missing store files.

        Args:
            provider_name: The registered provider.
            column: The document column to tokenize, or None when its
                token file already exists.
            value_columns: Value columns without a column file yet.
            offsets_encoder: Optional mapper from token IDs to character ranges.
        """
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        scan_columns = tuple(dict.fromkeys(
            ((column,) if column is not None else ()) + value_columns
        ))
        with ExitStack() as rollback:
            token_writer = None
            column_writers = {}
            writers = []
            with closing(provider.scan(ScanRequest(columns=scan_columns))) as scan:
                reader = iter(scan)
                batches = reader
                if column is not None:
                    existing = self._token_stores.get((identity, column))
                    existing_tokens = (existing.arrow_array if existing is not None
                                       else None)
                    if existing_tokens is None:
                        buffered, tokenizer, token_type = self._pick_tokenizer(
                            reader, provider_name, column)
                        batches = chain(buffered, reader)
                    else:
                        tokenizer, token_type = (
                            self.tokenizer,
                            existing_tokens.type.value_type,
                        )
                    if offsets_encoder is not None:
                        self.notes.append(
                            f"{provider_name}.{column}: tokenizer: "
                            f"{self.tokenizer_name} with character offsets")
                    token_writer = TokenStoreWriter(
                        self._store_path(),
                        document_column=column,
                        tokenizer=tokenizer,
                        token_type=token_type,
                        offsets_encoder=offsets_encoder,
                        existing_tokens=existing_tokens,
                    )
                    rollback.callback(token_writer.abort)
                    writers.append(token_writer)
                source_schema = provider.schema()
                for name in value_columns:
                    writer = ColumnStoreWriter(
                        self._store_path(), source_schema.field(name))
                    rollback.callback(writer.abort)
                    column_writers[name] = writer
                    writers.append(writer)
                progress = None
                if column is not None:
                    say(f"tokenizing {provider_name}.{column} "
                        f"({self.notes[-1].split(': ', 1)[-1]})")
                    progress = Progress(f"tokenizing {provider_name}.{column}")
                rows = 0
                for batch in batches:
                    for start in range(0, batch.num_rows, TOKENIZE_ROWS):
                        piece = batch.slice(start, TOKENIZE_ROWS)
                        for writer in writers:
                            writer.write_batch(piece)
                        rows += piece.num_rows
                        if progress is not None:
                            progress.update(rows)
            # a finished store is closed and deleted with the rest on failure
            token_store = token_writer.finish() if token_writer is not None else None
            if token_store is not None:
                rollback.callback(token_store.close)
            column_stores = {}
            for name, writer in column_writers.items():
                store = writer.finish()
                rollback.callback(store.close)
                column_stores[name] = store
            # every file for one provider must line up row for row, so a
            # later scan must return as many rows as the earlier one did
            known_rows = next(
                (len(store) for (store_identity, _), store in chain(
                    self._token_stores.items(), self._column_stores.items())
                 if store_identity == identity),
                None)
            written = list(column_stores.items())
            if token_store is not None:
                written.append((column, token_store))
            for name, store in written:
                rows = len(store)
                if known_rows is not None and rows != known_rows:
                    raise RuntimeError(
                        f"{provider_name}.{name} returned {rows} rows but an "
                        f"earlier scan returned {known_rows}; the provider "
                        f"does not scan in a stable order")
            if progress is not None:
                progress.finish(f"tokenized {provider_name}.{column}",
                                f"{sum(token_store.lengths):,} tokens")
            rollback.pop_all()
        if token_store is not None:
            previous = self._token_stores.get((identity, column))
            if previous is not None:
                # In-flight requests can still reference the previous store.
                self._replaced_token_stores.append(previous)
            self._token_stores[(identity, column)] = token_store
        for name, store in column_stores.items():
            self._column_stores[(identity, name)] = store
