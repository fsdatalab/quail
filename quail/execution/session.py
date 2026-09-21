"""Register documents, plan queries, and run them in the current process."""

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from itertools import chain
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import compute as pc

from quail.builtins import built_in_registry
from quail.catalog import (
    Catalog,
    OcrProvider,
    PDFProvider,
    ScanRequest,
    TableProvider,
)
from quail.execution.pairs import (
    columns_key,
    pair_fraction,
    pair_table,
)
from quail.execution.pdf_inputs import PdfScanInput
from quail.execution.result import IndexRelation, QueryResult, true_answer_rows
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    scalar_node_metrics,
)
from quail.execution.tokens import (
    ColumnStoreWriter,
    ScanInput,
    TokenStoreWriter,
)
from quail.execution.types import PhysicalRequest
from quail.extensions import ExtensionRegistry
from quail.frontend.builder import Query as BuilderQuery
from quail.frontend.sql import SQLDialect, compile_sql
from quail.logical import (
    CompileError,
    LogicalPlan,
    join_conditions,
    oriented_join_conditions,
)
from quail.pdf.prompt import PagePrompts
from quail.physical import (
    PhysicalScan,
    PortRef,
    Project,
    ValueType,
    encode_graph,
)
from quail.planner import explain, plan_query
from quail.planner.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.planner.plan import (
    EngineConfig,
    PdfDocuments,
    Refusal,
    resolve_model,
)
from quail.progress import Progress, say
from quail.specs.vision import resolve_image_tokens


class RefusalError(RuntimeError):
    """Raised when run() is called on a refused plan."""

    def __init__(self, refusal: Refusal):
        self.refusal = refusal
        super().__init__(
            f"{refusal.constraint}: needed {refusal.needed} "
            f"{refusal.unit}, available {refusal.available}. "
            + " ".join(refusal.reasons))


def pick_corpus_tokenizer(primary, fast, texts, sample=25):
    """Pick the corpus tokenizer for one column.

    Returns the fast tokenizer if it matches the primary on a sample,
    otherwise the primary.
    """
    if fast is None:
        return primary, "tokenizer: transformers"
    for t in texts[:sample]:
        if list(fast(t)) != list(primary(t)):
            return primary, ("tokenizer: transformers (Gigatoken "
                             "failed parity on this column's sample)")
    return fast, "tokenizer: Gigatoken (parity-checked on sample)"


TOKENIZE_ROWS = 2048

# documents tokenized to measure tokens per byte for a length estimate
ESTIMATE_SAMPLE = 1024


class Session:
    def __init__(self, config: EngineConfig, *,
                 tokenizer=None,
                 registry: ExtensionRegistry | None = None):
        self.registry = registry or built_in_registry()
        model = resolve_model(config.model, self.registry.models)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = self.registry.device(config.device)
        try:
            backend = self.registry.backend(config.backend)
        except ValueError as error:
            raise RefusalError(Refusal(
                reasons=(str(error),),
                constraint="unknown_backend",
                needed=1,
                available=0,
                unit="backends",
            )) from error
        support = backend.supports(model, self.device, config.gpus)
        if not support.supported:
            raise RefusalError(Refusal(
                reasons=(support.reason or
                         "unsupported backend configuration",),
                constraint="unsupported_backend_configuration",
                needed=1,
                available=0,
                unit="configurations",
            ))
        try:
            self.image_tokens = resolve_image_tokens(model, config.image_tokens)
        except ValueError as error:
            raise RefusalError(Refusal(
                reasons=(str(error),),
                constraint="image_tokens_unsupported",
                needed=1,
                available=0,
                unit="image token budgets",
            )) from error
        self.catalog = Catalog()
        self._pdf_inputs = {}
        self._tok = tokenizer      # injectable for tests; lazy HF load
        self._tok_injected = tokenizer is not None
        self._fast = None          # lazy Gigatoken instance
        self._fast_tried = False
        self.notes = []            # tokenizer picks etc., for reports
        self._token_stores = {}
        self._column_stores = {}
        self._store_count = 0
        self._token_directory = None
        self._corpus_tokenizers = {}
        self._length_estimates = {}
        self._lock = threading.RLock()
        self._background = None

    def close(self):
        """Wait for background tokenization and remove temporary token files."""
        if self._background is not None:
            self._background.shutdown(cancel_futures=True)
        for store in self._token_stores.values():
            store.close()
        self._token_stores.clear()
        for store in self._column_stores.values():
            store.close()
        self._column_stores.clear()
        if self._token_directory is not None:
            self._token_directory.cleanup()
            self._token_directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def register(self, name: str, provider: TableProvider) -> None:
        self.catalog.register(name, provider)

    def sql(self, text: str, order: str | None = None,
            dialect: SQLDialect | str = SQLDialect.SNOWFLAKE) -> "Query":
        logical = compile_sql(
            text, self.catalog, self.tokenizer, dialect=dialect,
            turn=self.model.turn,
        )
        return Query(self, logical, order=order)

    def docs(self, name: str) -> "BoundBuilder":
        return BoundBuilder(self,
                            BuilderQuery(self.catalog, name,
                                         self.tokenizer,
                                         turn=self.model.turn))

    @property
    def tokenizer(self):
        """Return the primary tokenizer, loading from HuggingFace if needed."""
        if self._tok is None:
            from transformers import AutoTokenizer
            hf = AutoTokenizer.from_pretrained(self.model.hf_name)
            self._tok = lambda text: hf(
                text, add_special_tokens=False)["input_ids"]
        return self._tok

    def _fast_tokenizer(self):
        """Return the Gigatoken tokenizer, or None if unavailable."""
        if self._tok_injected:
            return None
        if not self._fast_tried:
            self._fast_tried = True
            try:
                from gigatoken import Tokenizer
                fast = Tokenizer(self.model.hf_name)
                self._fast = fast.encode
            except Exception:
                self._fast = None
        return self._fast

    def pdf_documents(self, provider_name: str) -> PdfDocuments | None:
        """What the planner needs to know about an alias over PDF pages.

        None for a table that is not PDF pages. A PDF provider's pages
        are rendered; the OCR operator over one gives their text.
        """
        provider = self.catalog.get(provider_name)
        if not isinstance(provider, (PDFProvider, OcrProvider)):
            return None
        pdf_input = provider.pdf_input()
        rows = dict(row_mode=pdf_input.row_mode, n_pages=pdf_input.page_count,
                    pages_per_row_max=pdf_input.pages_per_row_max)
        if isinstance(provider, OcrProvider):
            return PdfDocuments(reading="ocr", **rows)
        return PdfDocuments(reading="image", visual_tokens=self.image_tokens,
                            **rows)

    def tokenize(self, provider_name: str, column: str,
                 projected_columns=()) -> ScanInput:
        """Tokenize one document column and keep value columns beside it.

        The token file is cached per document column. Each value column
        is cached in its own file, so a later query that returns other
        columns reads only those columns from the provider and does not
        tokenize the documents again.
        """
        with self._lock:
            return self._tokenize(provider_name, column, projected_columns)

    def _tokenize(self, provider_name, column, projected_columns):
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        projected_columns = tuple(dict.fromkeys(projected_columns))
        token_key = (identity, column)
        missing = [
            name for name in projected_columns
            if (identity, name) not in self._column_stores
        ]
        if token_key not in self._token_stores or missing:
            self._load(
                provider_name,
                None if token_key in self._token_stores else column,
                tuple(missing),
            )
        return ScanInput(
            self._token_stores[token_key],
            {
                name: self._column_stores[(identity, name)]
                for name in projected_columns
            },
        )

    def prepare_pdf(self, provider_name: str,
                    projected_columns=()) -> PdfScanInput:
        """Bind one PDF provider's rows and keep value columns beside them.

        Reads the page manifest through the provider (cached there),
        prices every row for this session's image budget, and stores
        the requested value columns the same way tokenize() does.
        """
        with self._lock:
            provider = self.catalog.get(provider_name)
            identity = provider.content_identity()
            projected_columns = tuple(dict.fromkeys(projected_columns))
            missing = tuple(
                name for name in projected_columns
                if (identity, name) not in self._column_stores)
            if missing:
                self._load(provider_name, None, missing)
            if identity not in self._pdf_inputs:
                pdf_input = provider.pdf_input()
                self._pdf_inputs[identity] = (
                    pdf_input, PagePrompts(pdf_input, self.model,
                                           self.image_tokens).lengths)
            pdf_input, lengths = self._pdf_inputs[identity]
            return PdfScanInput(pdf_input, lengths, {
                name: self._column_stores[(identity, name)]
                for name in projected_columns
            })

    def tokenize_async(self, provider_name: str, column: str,
                       projected_columns=()) -> Future:
        """Start tokenize() on a background thread and return its Future."""
        if self._background is None:
            self._background = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="quail-tokenize")
        return self._background.submit(
            self.tokenize, provider_name, column, projected_columns)

    def register_functions(self, functions: dict) -> None:
        """Register a query's apply() functions once each, by name."""
        for name, function in functions.items():
            known = self.registry.functions.get(name)
            if known is function:
                continue
            if known is not None:
                raise ValueError(
                    f"apply name {name!r} is registered for another "
                    f"function on this session")
            self.registry.register_function(function, name=name)

    def column_values(self, provider_name: str, column: str) -> pa.ChunkedArray:
        """Return one source column in scan order.

        Reads the column store when the column is loaded, else scans
        the provider for that one column.
        """
        provider = self.catalog.get(provider_name)
        with self._lock:
            store = self._column_stores.get(
                (provider.content_identity(), column))
        if store is not None:
            return store.values
        reader = provider.scan(ScanRequest(columns=(column,)))
        try:
            batches = [batch.column(0) for batch in reader]
        finally:
            reader.close()
        return pa.chunked_array(
            batches, type=provider.schema().field(column).type)

    def token_lengths(self, provider_name: str, column: str):
        """Return the exact token counts when the column is tokenized."""
        identity = self.catalog.get(provider_name).content_identity()
        store = self._token_stores.get((identity, column))
        return None if store is None else store.lengths

    def estimate_lengths(self, provider_name: str, column: str) -> list[int]:
        """Estimate document token counts from a tokenized sample.

        Reads the column's byte lengths and scales them by the tokens
        per byte measured on the first ESTIMATE_SAMPLE documents.
        """
        provider = self.catalog.get(provider_name)
        key = (provider.content_identity(), column)
        if key in self._length_estimates:
            return self._length_estimates[key]
        started = time.perf_counter()
        estimate = getattr(provider, "estimate_token_lengths", None)
        if estimate is not None:
            # the provider knows its rows better than their byte
            # lengths would tell: PDF pages before their text exists
            tok = self.tokenizer
            lengths = estimate(lambda texts: [len(tok(text)) for text in texts])
            self._length_estimates[key] = lengths
            say(f"estimated {provider_name}.{column}: {len(lengths):,} rows, "
                f"about {sum(lengths):,} tokens from a page sample, "
                f"{time.perf_counter() - started:.1f} s")
            return lengths
        reader = provider.scan(ScanRequest(columns=(column,)))
        byte_lengths = []
        sample = []
        try:
            for batch in reader:
                texts = batch.column(0)
                byte_lengths.append(pc.binary_length(texts).cast(pa.int64()))
                if len(sample) < ESTIMATE_SAMPLE:
                    sample.extend(
                        texts.slice(0, ESTIMATE_SAMPLE - len(sample)).to_pylist()
                    )
        finally:
            reader.close()
        # Defer fast-tokenizer initialization to the background tokenization pass.
        tok = self.tokenizer
        sample_tokens = sum(len(tok(text)) for text in sample)
        sample_bytes = sum(len(text.encode("utf-8")) for text in sample)
        ratio = sample_tokens / sample_bytes if sample_bytes else 0.0
        lengths = []
        for chunk in byte_lengths:
            lengths.extend(
                max(1, round(n * ratio)) for n in chunk.to_pylist()
            )
        self._length_estimates[key] = lengths
        say(f"estimated {provider_name}.{column}: {len(lengths):,} documents, "
            f"about {sum(lengths):,} tokens from a {len(sample)} document "
            f"sample, {time.perf_counter() - started:.1f} s")
        return lengths

    def _corpus_tokenizer(self, provider_name, column, texts):
        """Pick and cache the tokenizer and token type for one column."""
        key = (self.catalog.get(provider_name).content_identity(), column)
        if key not in self._corpus_tokenizers:
            tok, note = pick_corpus_tokenizer(
                self.tokenizer, self._fast_tokenizer(), texts)
            self.notes.append(f"{provider_name}.{column}: {note}")
            first_token = next(
                (token for text in texts for token in tok(text)), None
            )
            token_type = (
                pa.int32()
                if first_token is None or isinstance(first_token, Integral)
                else pa.string()
            )
            self._corpus_tokenizers[key] = (tok, token_type)
        return self._corpus_tokenizers[key]

    def _store_path(self) -> str:
        if self._token_directory is None:
            self._token_directory = TemporaryDirectory(
                prefix="quail-tokens-"
            )
        self._store_count += 1
        return str(
            Path(self._token_directory.name)
            / f"input-{self._store_count}.arrow"
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
        tok, token_type = self._corpus_tokenizer(
            provider_name, column, text_sample)
        return buffered, tok, token_type

    def _load(self, provider_name: str, column: str | None,
              value_columns: tuple[str, ...]) -> None:
        """Scan the provider once and write the missing store files.

        Args:
            provider_name: The registered provider.
            column: The document column to tokenize, or None when its
                token file already exists.
            value_columns: Value columns without a column file yet.
        """
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        scan_columns = tuple(dict.fromkeys(
            ((column,) if column is not None else ()) + value_columns
        ))
        scan_reader = provider.scan(ScanRequest(columns=scan_columns))
        reader = iter(scan_reader)
        token_writer = None
        column_writers = {}
        writers = []
        try:
            batches = reader
            if column is not None:
                buffered, tok, token_type = self._pick_tokenizer(
                    reader, provider_name, column)
                batches = chain(buffered, reader)
                token_writer = TokenStoreWriter(
                    self._store_path(),
                    document_column=column,
                    tokenizer=tok,
                    token_type=token_type,
                )
                writers.append(token_writer)
            source_schema = provider.schema()
            for name in value_columns:
                column_writers[name] = ColumnStoreWriter(
                    self._store_path(), source_schema.field(name))
                writers.append(column_writers[name])
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
        except Exception:
            for writer in writers:
                writer.abort()
            raise
        finally:
            scan_reader.close()
        token_store = (
            token_writer.finish() if token_writer is not None else None)
        if progress is not None:
            progress.finish(f"tokenized {provider_name}.{column}",
                            f"{sum(token_store.lengths):,} tokens")
        column_stores = {
            name: writer.finish() for name, writer in column_writers.items()
        }
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
                for _, bad in written:
                    bad.close()
                    os.unlink(bad.path)
                raise RuntimeError(
                    f"{provider_name}.{name} returned {rows} rows but an "
                    f"earlier scan returned {known_rows}; the provider "
                    f"does not scan in a stable order")
        if token_store is not None:
            self._token_stores[(identity, column)] = token_store
        for name, store in column_stores.items():
            self._column_stores[(identity, name)] = store


class BoundBuilder:
    """Builder wrapper that returns a runnable Query from select()."""

    def __init__(self, session: Session, inner):
        self._session = session
        self._inner = inner

    def alias(self, a):
        self._inner.alias(a)
        return self

    def ai_filter(self, p, selectivity=None):
        self._inner.ai_filter(p, selectivity=selectivity)
        return self

    def join(self, other, on=None):
        inner = other._inner if isinstance(other, BoundBuilder) else other
        self._inner.join(inner, on=on)
        return self

    def apply(self, fn, columns=(), **options):
        self._inner.apply(fn, columns, **options)
        self._session.register_functions(self._inner.functions)
        return self

    def apply_table(self, fn, columns=(), **options):
        return self.apply(fn, columns, kind="barrier", **options)

    def ai_join(self, others, p, selectivity=None, anchor=None,
                semantics="full"):
        if not isinstance(others, (list, tuple)):
            others = [others]
        unwrapped = [o._inner if isinstance(o, BoundBuilder) else o
                     for o in others]
        self._inner.ai_join(unwrapped, p, selectivity=selectivity,
                            anchor=anchor, semantics=semantics)
        return self

    def limit(self, n):
        self._inner.limit(n)
        return self

    def select(self, *cols, order=None) -> "Query":
        return Query(self._session, self._inner.select(*cols),
                     order=order)


class Query:
    def __init__(self, session: Session, logical: LogicalPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self.order = order
        self._plan = None
        self._doc_tokens = None
        self._token_inputs = None
        self._pdf_documents = {}
        self._providers = {}
        self._token_futures = {}
        self._estimated = ()
        self.token_wait_s = 0.0

    def ocr_metrics(self) -> dict[str, dict]:
        """The OCR operator's counters for every alias over one, by alias.

        An alias whose text was never read (the plan was refused, or
        another alias over the same operator read it) is left out.
        """
        out = {}
        for alias, pdf in self._pdf_documents.items():
            if pdf.reading != "ocr":
                continue
            metrics = self.session.catalog.get(self._providers[alias]).metrics()
            if metrics is not None:
                out[alias] = metrics
        return out

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        Plans first, then waits for any background tokenization, so
        planning and the speed of light estimate share these stores.
        """
        self.plan()
        self.wait_for_tokens()
        return self._token_inputs

    def plan(self):
        if self._plan is None:
            self.logical, _ = apply_logical_rules(
                self.logical,
                tuple(self.session.registry.logical_rules.values()),
                LogicalPlanningContext(
                    self.session.catalog, self.session.config
                ),
            )
            operators = self.logical.operators()
            scans, joins = operators.scans, operators.joins
            self._doc_tokens = {}
            self._token_inputs = {}
            self._pdf_documents = {}
            self._providers = {}
            estimated = []
            for s in scans:
                self._providers[s.alias] = s.provider
                pdf = self.session.pdf_documents(s.provider)
                if pdf is not None:
                    self._pdf_documents[s.alias] = pdf
                if pdf is not None and pdf.reading == "image":
                    if not self.session.image_tokens:
                        self._plan = Refusal(
                            reasons=(f"model {self.session.model.name!r} "
                                     f"takes text only, but {s.alias!r} "
                                     f"binds PDF pages; register the table "
                                     f"through .ocr() to read their text",),
                            constraint="model_takes_text_only",
                            needed=1, available=0, unit="image models")
                        return self._plan
                    store = self.session.prepare_pdf(s.provider, s.columns)
                    self._token_inputs[s.alias] = store
                    self._doc_tokens[s.alias] = store.lengths
                    continue
                # the OCR operator's rows take the text path below
                exact = self.session.token_lengths(s.provider, s.column)
                if exact is not None:
                    store = self.session.tokenize(
                        s.provider, s.column, s.columns)
                    self._token_inputs[s.alias] = store
                    self._doc_tokens[s.alias] = store.lengths
                    continue
                self._doc_tokens[s.alias] = self.session.estimate_lengths(
                    s.provider, s.column)
                self._token_futures[s.alias] = self.session.tokenize_async(
                    s.provider, s.column, s.columns)
                estimated.append(s.alias)
            self._estimated = tuple(estimated)
            pair_fractions = self._pair_fractions(scans, joins)
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                order=self.order,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                pdf_documents=self._pdf_documents)
            if self.session.config.gpu_timing:
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, "gpu_timing": True})
        return self._plan

    def _pair_fractions(self, scans, joins) -> dict:
        """Pairs kept over the cross product, per join with conditions."""
        providers = {scan.alias: scan.provider for scan in scans}
        fractions = {}
        for position, join in enumerate(joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            left_alias, right_alias, conditions = oriented
            left_keys = [
                self.session.column_values(
                    providers[left.alias], left.column)
                for left, _ in conditions
            ]
            right_keys = [
                self.session.column_values(
                    providers[right.alias], right.column)
                for _, right in conditions
            ]
            fractions[position] = pair_fraction(
                pair_table(left_alias, left_keys, right_alias, right_keys),
                len(self._doc_tokens[left_alias]),
                len(self._doc_tokens[right_alias]))
        return fractions

    def explain(self, *, verbose: bool = False,
                analyze: bool = False) -> str:
        """Return the optimized plan, optionally measured by running it.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first, like EXPLAIN ANALYZE, and show
                each node's measured rows, time, and tokens beside the
                planner's estimates, then the measured totals.
        """
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        result = None
        if analyze:
            result = self.run()
            # the root relation is lazy, so its row count is taken here
            rows = result.count()
            root = physical.graph.root.node_id
            result.node_metrics[root] = replace(
                result.node_metrics.get(root, NodeMetrics()),
                output_rows=rows)
            result.report.setdefault("node_metrics", {}).setdefault(
                root, {})["output_rows"] = rows
        text = explain(self.logical, physical, verbose=verbose,
                       result=result,
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        if self._estimated:
            text += ("\n\n  note: token counts for "
                     + ", ".join(repr(a) for a in self._estimated)
                     + f" are estimated from a {ESTIMATE_SAMPLE} document "
                     "sample")
        return text

    def wait_for_tokens(self) -> None:
        """Block until every background tokenization has finished."""
        started = time.perf_counter()
        for alias, future in list(self._token_futures.items()):
            self._token_inputs[alias] = future.result()
            del self._token_futures[alias]
        self.token_wait_s += time.perf_counter() - started

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan from plan().insert(),
                remove(), or move(); the planner's own plan when
                omitted.
        """
        from quail.execution.execute import execute_query

        return execute_query(self, plan=plan)

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run().execute_stream(
            batch_rows=batch_rows, limit=limit)

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and explicitly collect one Arrow table."""
        return self.run().collect(
            limit=limit, batch_rows=batch_rows)

    def _prepare_physical(self):
        """Bind the query's tokenized inputs to its physical plan."""
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        self.wait_for_tokens()
        inputs = {
            node.input_id: self._token_inputs[node.alias].physical_input()
            for node in plan.nodes if isinstance(node, PhysicalScan)
        }
        envelope = plan.to_envelope(self.session.registry.codecs)
        return PhysicalRequest(envelope, inputs, self._column_tables())

    def _column_tables(self) -> dict:
        """One value table per alias a HashJoin or an apply() reads."""
        needed = {}
        operators = self.logical.operators()
        for join in operators.joins:
            for condition in join_conditions(join):
                for ref in (condition.left, condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for apply in operators.applies:
            for ref in apply.columns:
                needed.setdefault(ref.alias, {})[ref.column] = None
        tables = {}
        for alias, columns in needed.items():
            store = self._token_inputs[alias]
            arrays = {alias: pa.array(range(len(store.lengths)), pa.int32())}
            for name in columns:
                arrays[name] = store.column(name)
            tables[columns_key(alias)] = pa.table(arrays)
        return tables

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        from quail.physical import AiJoin, Barrier

        expected_nodes = tuple(node for node in plan.nodes
                               if isinstance(node, (AiJoin, Barrier)))
        report = dict(
            backend=out.get("backend", plan.backend),
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            estimated_seconds=plan.estimated_seconds,
            boot_kind=out.get("boot_kind"),
            boot=out.get("boot"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"],
            cached_tokens=out.get("cached_tokens"),
            stages=[],
            peak_gib=out.get("peak_gib"),
            order_rule=plan.settings.get("order_rule"),
            expected_join_plan=[
                {
                    "type": node.type_name,
                    "id": node.node_id,
                    **node.explain_fields(),
                }
                for node in expected_nodes
            ],
            executed_join_plan=out.get("executed_join_plan", []),
            kv_manager=out.get("kv_manager"),
            node_metrics=out.get("node_metrics", {}),
            backend_metrics=out.get("backend_metrics"),
            remarks=list(plan.remarks) + list(self.session.notes))
        for key in ("gpu_s", "chunks"):
            if key in out:
                report[key] = out[key]
        for key in (
            "evaluated_documents",
            "evaluated_document_pairs",
            "documents_per_second",
            "document_pairs_per_second",
            "usd_per_query",
        ):
            if key in out:
                report[key] = out[key]

        operators = self.logical.operators()
        scans, logical_filters, logical_joins = (
            operators.scans, operators.filters, operators.joins
        )
        scans_by_alias = {scan.alias: scan for scan in scans}

        def project(node, value):
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
            for name in node.columns:
                if name in relation.schema.names:
                    # a score column the graph computed; document
                    # index columns are aliases, never "alias.column"
                    projection.append((name, None))
                    fields.append(pa.field(
                        name, relation.schema.field(name).type
                    ))
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
                store = self._token_inputs[alias]
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
            return QueryResult(
                columns=list(node.columns),
                declaration=relation.declaration,
                document_index_schema=relation.schema,
                output_schema=pa.schema(
                    fields,
                    metadata={b"quail.kind": b"query_result"},
                ),
                projection=projection,
                report={},
                row_count=(value.num_rows if isinstance(value, pa.Table)
                           else None),
            )

        sources = {
            node.input_id: range(node.n_docs)
            for node in plan.nodes
            if isinstance(node, PhysicalScan)
        }
        observers = self.session.registry.new_observers()
        run = GenericRunner().run(
            plan.graph,
            ExecutionContext(
                runtimes=self.session.registry.runtimes,
                sources=sources,
                project=project,
                observers=observers,
            ),
            initial_outputs=response.outputs,
            initial_metrics={
                node_id: NodeMetrics(**metrics)
                for node_id, metrics in out.get("node_metrics", {}).items()
            },
        )
        if not isinstance(run.value, QueryResult):
            raise TypeError("physical graph root must return QueryResult")
        result = run.value
        result.plan = plan.graph
        result.node_metrics = {
            node_id: node_result.metrics
            for node_id, node_result in run.nodes.items()
        }
        # Saved reports need the graph and metrics without Python objects.
        report["executed_plan"] = encode_graph(
            plan.graph, self.session.registry.codecs)
        report["node_metrics"] = scalar_node_metrics(run.nodes)
        observer_reports = {
            observer.name: dict(observer.report())
            for observer in observers
        }
        if observer_reports:
            report["observers"] = observer_reports
        result.report = report

        answer_tables = {"filters": {}, "joins": {}}
        survivors = {
            scan.alias: list(range(len(self._doc_tokens[scan.alias])))
            for scan in scans
        }

        output_types = {
            PortRef(node.node_id, output.name): output.value_type
            for node in plan.nodes
            for output in node.outputs
        }
        filter_relations = {}
        join_relations = {}
        for ref, table in response.outputs.items():
            value_type = output_types.get(ref)
            metadata = table.schema.metadata or {}
            if value_type is ValueType.FILTER_ANSWERS:
                alias = metadata.get(b"quail.alias")
                if alias is None:
                    raise ValueError(
                        "a filter answer relation needs quail.alias metadata"
                    )
                filter_relations.setdefault(
                    alias.decode("utf-8"), []
                ).append(table)
            elif value_type is ValueType.JOIN_ANSWERS:
                written_pos = metadata.get(b"quail.written_pos")
                if written_pos is None:
                    raise ValueError(
                        "a join answer relation needs "
                        "quail.written_pos metadata"
                    )
                position = int(written_pos.decode("ascii"))
                if position in join_relations:
                    raise ValueError(
                        f"duplicate join answer relation {position}"
                    )
                join_relations[position] = table

        missing_filter_relations = set(logical_filters) - set(
            filter_relations
        )
        if missing_filter_relations:
            raise ValueError(
                "execution response is missing filter answer relations for "
                f"{sorted(missing_filter_relations)}"
            )
        missing_join_relations = set(range(len(logical_joins))) - set(
            join_relations
        )
        if missing_join_relations:
            raise ValueError(
                "execution response is missing join answer relations for "
                f"{sorted(missing_join_relations)}"
            )

        for node in plan.graph.topological_nodes():
            for output in node.outputs:
                if output.value_type is not ValueType.DOCUMENT_IDS:
                    continue
                ref = PortRef(node.node_id, output.name)
                if ref not in response.outputs:
                    continue
                table = response.outputs[ref]
                if len(table.column_names) != 1:
                    raise ValueError(
                        "a document id relation needs one alias column"
                    )
                alias = table.column_names[0]
                survivors[alias] = table.column(alias).to_pylist()

        for alias, relations in filter_relations.items():
            table = relations[0] if len(relations) == 1 else \
                pa.concat_tables(relations)
            positions = table.column("predicate").to_pylist()
            predicate_order = list(dict.fromkeys(int(pos) for pos in positions))
            predicate_order.extend(
                position for position in range(len(logical_filters[alias]))
                if position not in predicate_order
            )
            for index, written_pos in enumerate(predicate_order):
                mask = pc.equal(table.column("predicate"), written_pos)
                stage_table = table.filter(mask)
                answered = stage_table.column(alias).to_pylist()
                answers = stage_table.column("answer").to_pylist()
                passed = sum(bool(answer) for answer in answers)
                answer_tables["filters"][(alias, written_pos)] = stage_table
                report["stages"].append(dict(
                    op="filter", alias=alias, stage=index,
                    written_pos=written_pos,
                    provided_selectivity=(
                        logical_filters[alias][written_pos].selectivity
                    ),
                    observed_selectivity=round(
                        passed / max(1, len(answered)), 4
                    ),
                    evaluated=len(answered),
                ))

        true_join_tables = {}
        expected_order = []
        for node in expected_nodes:
            for output in node.outputs:
                if not output.name.startswith("join_answers:"):
                    continue
                expected_order.append(int(output.name.split(":", 1)[1]))
        expected_order.extend(
            position for position in sorted(join_relations)
            if position not in expected_order
        )
        for written_pos in expected_order:
            if written_pos not in join_relations:
                continue
            table = join_relations[written_pos]
            metadata = table.schema.metadata or {}
            logical_join = logical_joins[written_pos]
            logical_aliases = [
                argument.alias for argument in logical_join.prompt.args
            ]
            required_metadata = {
                b"quail.anchor", b"quail.partners", b"quail.semantics"
            }
            missing = required_metadata - set(metadata)
            if missing:
                raise ValueError(
                    "a join answer relation is missing metadata "
                    f"{sorted(key.decode('utf-8') for key in missing)}"
                )
            anchor = metadata[b"quail.anchor"].decode("utf-8")
            partner_text = metadata[b"quail.partners"].decode("utf-8")
            partners = [] if not partner_text else partner_text.split(",")
            semantics = metadata[b"quail.semantics"].decode("utf-8")
            missing_columns = set(logical_aliases) - set(table.column_names)
            if missing_columns:
                raise ValueError(
                    "a join answer relation is missing alias columns "
                    f"{sorted(missing_columns)}"
                )
            answers = table.column("answer")
            answer_tables["joins"][written_pos] = table
            report["stages"].append(dict(
                op="join", written_pos=written_pos, anchor=anchor,
                partners=partners,
                semantics=semantics,
                provided_selectivity=logical_join.selectivity,
                observed_selectivity=round(
                    (pc.sum(answers).as_py() or 0) / max(1, len(answers)), 4
                ),
                tuples=len(answers),
            ))
            if semantics == "full":
                true_join_tables[written_pos] = true_answer_rows(table)
        survivor_arrays = {
            alias: pa.array(indices, type=pa.int32())
            for alias, indices in survivors.items()
        }
        result.answer_tables = answer_tables
        result.survivor_indices = survivor_arrays
        result.true_join_tables = true_join_tables
        return result
