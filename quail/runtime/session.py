"""Register documents, plan queries, and run them through a compute provider."""

import os
from itertools import chain
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import compute as pc

from quail.builder import Query as BuilderQuery
from quail.builtins import built_in_registry
from quail.catalog import Catalog, ScanRequest, TableProvider
from quail.execution import PhysicalRequest, document_input
from quail.extensions import ExtensionRegistry
from quail.logical import CompileError, LogicalPlan
from quail.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.physical import DocumentInput, PortRef, Project, ValueType, encode_graph
from quail.planner import collect_operators, explain, plan_query
from quail.planner.plan import EngineConfig, Refusal, resolve_model
from quail.runtime.compute import ModalComputeProvider, QueryRequest
from quail.runtime.prefixes import prefix_metrics
from quail.runtime.result import IndexRelation, QueryResult, true_answer_rows
from quail.runtime.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    scalar_node_metrics,
)
from quail.runtime.tokens import (
    ColumnStoreWriter,
    ScanInput,
    TokenStoreWriter,
)
from quail.sqlfront import SQLDialect, compile_sql


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
            return primary, ("tokenizer: transformers (bpe-qwen "
                             "failed parity on this column's sample)")
    return fast, "tokenizer: bpe-qwen (parity-checked on sample)"


class Session:
    def __init__(self, config: EngineConfig = EngineConfig(),
                 device: str = "h100-sxm", tokenizer=None,
                 registry: ExtensionRegistry | None = None,
                 compute_provider=None):
        self.registry = registry or built_in_registry()
        model = resolve_model(config.model, self.registry.models)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = self.registry.device(device)
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
        self.catalog = Catalog()
        self._tok = tokenizer      # injectable for tests; lazy HF load
        self._tok_injected = tokenizer is not None
        self._fast = None          # lazy bpe-qwen instance
        self._fast_tried = False
        self.notes = []            # tokenizer picks etc., for reports
        self._token_stores = {}
        self._column_stores = {}
        self._store_count = 0
        self._token_directory = None
        self.compute_provider = compute_provider

    def close(self):
        """Close compute and temporary token storage."""
        try:
            if self.compute_provider is not None:
                self.compute_provider.close()
        finally:
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

    # ---- the two entry points ---------------------------------------

    def sql(self, text: str, order: str | None = None,
            dialect: SQLDialect | str = SQLDialect.SNOWFLAKE) -> "Query":
        logical = compile_sql(
            text, self.catalog, self.tokenizer, dialect=dialect
        )
        return Query(self, logical, order=order)

    def docs(self, name: str) -> "BoundBuilder":
        return BoundBuilder(self,
                            BuilderQuery(self.catalog, name,
                                         self.tokenizer))

    # ---- shared machinery --------------------------------------------

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
        """Return the bpe-qwen fast tokenizer, or None if unavailable."""
        if self._tok_injected:
            return None
        if not self._fast_tried:
            self._fast_tried = True
            try:
                from bpe_qwen import AutoLinearTokenizer
                fast = AutoLinearTokenizer.from_pretrained(
                    self.model.hf_name)
                self._fast = lambda text: fast(
                    text, add_special_tokens=False)["input_ids"]
            except Exception:
                self._fast = None
        return self._fast

    def tokenize(self, provider_name: str, column: str,
                 projected_columns=()) -> ScanInput:
        """Tokenize one document column and keep value columns beside it.

        The token file is cached per document column. Each value column
        is cached in its own file, so a later query that returns other
        columns reads only those columns from the provider and does not
        tokenize the documents again.
        """
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
        tok, note = pick_corpus_tokenizer(
            self.tokenizer, self._fast_tokenizer(), text_sample)
        self.notes.append(f"{provider_name}.{column}: {note}")
        sample_rows = [tok(text) for text in text_sample]
        first_token = next(
            (token for row in sample_rows for token in row), None
        )
        token_type = (
            pa.int32()
            if first_token is None or isinstance(first_token, Integral)
            else pa.string()
        )
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
            for batch in batches:
                for writer in writers:
                    writer.write_batch(batch)
        except Exception:
            for writer in writers:
                writer.abort()
            raise
        finally:
            scan_reader.close()
        token_store = (
            token_writer.finish() if token_writer is not None else None)
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

    def select(self, *cols, order="as_written") -> "Query":
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

    # ---- planning (the optimization) ---------------------------------

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        The logical rules run first, then each scanned column is
        tokenized once. Planning and the speed of light estimate share
        these stores.
        """
        if self._token_inputs is None:
            self.logical, _ = apply_logical_rules(
                self.logical,
                tuple(self.session.registry.logical_rules.values()),
                LogicalPlanningContext(
                    self.session.catalog, self.session.config
                ),
            )
            scans, _, _ = collect_operators(self.logical)
            self._doc_tokens = {}
            self._token_inputs = {}
            # each Scan lists the columns it must load; the projection
            # pushdown rule filled that in above
            for s in scans:
                store = self.session.tokenize(
                    s.provider,
                    s.column,
                    s.columns,
                )
                self._token_inputs[s.alias] = store
                self._doc_tokens[s.alias] = store.lengths
        return self._token_inputs

    def plan(self):
        if self._plan is None:
            self.token_inputs()
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                order=self.order,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer)
        return self._plan

    def explain(self) -> str:
        return explain(self.logical, self.plan())

    # ---- execution -----------------------------------------------------

    def run(self) -> QueryResult:
        """Execute the query through the session compute provider."""
        if self.session.compute_provider is None:

            self.session.compute_provider = ModalComputeProvider()
        result = self.session.compute_provider.execute(self._request())
        if not isinstance(result, QueryResult):
            raise TypeError("a compute provider must return QueryResult")
        return result

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
        """Build the physical request used inside a compute worker."""
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        inputs = {}
        for node in plan.nodes:
            if not isinstance(node, DocumentInput):
                continue
            inputs[node.input_id] = document_input(
                self._token_inputs[node.alias].tokens
            )
        envelope = plan.to_envelope(self.session.registry.codecs)
        return PhysicalRequest(envelope, inputs)

    def _request(self):
        """Build the logical request sent to a compute provider."""
        scans, _, _ = collect_operators(self.logical)
        return QueryRequest(
            logical_plan=self.logical,
            providers={
                scan.provider: self.session.catalog.get(scan.provider)
                for scan in scans
            },
            config=self.session.config,
            device=self.session.device.name,
            order=self.order,
            registry=self.session.registry,
        )

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        from quail.physical import AnchoredJoin, Exchange

        expected_nodes = tuple(node for node in plan.nodes
                               if isinstance(node, (AnchoredJoin, Exchange)))
        report = dict(
            backend=out.get("backend", plan.backend),
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            boot_kind=out.get("boot_kind"),
            boot=out.get("boot"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"],
            cached_tokens=out.get("cached_tokens"),
            regret_tokens=out.get("regret_tokens"), stages=[],
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
            result_volume_path=out.get("result_volume_path"),
            remarks=list(plan.remarks) + list(self.session.notes))

        scans, logical_filters, logical_joins = collect_operators(self.logical)
        scans_by_alias = {scan.alias: scan for scan in scans}

        def project(node, value):
            if not isinstance(node, Project):
                raise TypeError(type(node).__name__)
            if node.inputs[0].value_type is ValueType.JOIN_ANSWERS:
                # a join answers table read directly: keep the true pairs
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
            )

        sources = {
            node.input_id: range(node.n_docs)
            for node in plan.nodes
            if isinstance(node, DocumentInput)
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
                argument.alias for argument in logical_join.predicate.args
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
            answers = table.column("answer").to_pylist()
            answer_tables["joins"][written_pos] = table
            report["stages"].append(dict(
                op="join", anchor=anchor,
                partners=partners,
                semantics=semantics,
                provided_selectivity=logical_join.selectivity,
                observed_selectivity=round(
                    sum(bool(answer) for answer in answers)
                    / max(1, len(answers)), 4
                ),
                tuples=len(answers),
            ))
            if semantics == "full":
                true_join_tables[written_pos] = true_answer_rows(table)
        report.update(prefix_metrics(report, scans, self._token_inputs))
        survivor_arrays = {
            alias: pa.array(indices, type=pa.int32())
            for alias, indices in survivors.items()
        }
        result.answer_tables = answer_tables
        result.survivor_indices = survivor_arrays
        result.true_join_tables = true_join_tables
        return result
