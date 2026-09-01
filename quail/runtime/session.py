"""Session and Query: the user-facing API for registering documents,
compiling queries, planning, and executing on Modal.
"""

import re
import time
from dataclasses import replace

import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog import Catalog, ScanRequest, TableProvider
from quail.extensions import ExtensionRegistry, built_in_registry
from quail.logical import (
    SHARED_PRE,
    CompileError,
    LogicalPlan,
    join_label,
    render_join_frame,
)
from quail.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
)
from quail.planner.decide import _collect, explain, plan_query
from quail.planner.plan import EngineConfig, Refusal, resolve_model
from quail.physical import AdaptiveJoinPlan, ExecutionLocation, PackedFilter
from quail.runtime.result import (
    QueryResult,
    answer_table,
    build_result_declaration,
    true_answer_rows,
)
from quail.specs import DEVICES


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
                 registry: ExtensionRegistry | None = None):
        model = resolve_model(config.model)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = DEVICES[device]
        self.registry = registry or built_in_registry()
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
        self._scan_cache = {}      # (source, column) -> token lists
        self._app_ctx = None       # the Modal app held open for the
        #                            session, so the worker container
        #                            (its booted model) survives
        #                            between run() calls

    def worker(self):
        """Return the worker module, opening this session's Modal app if needed."""
        from quail.runtime import worker
        if self._app_ctx is None:
            self._app_ctx = worker.app.run()
            self._app_ctx.__enter__()
        return worker

    def close(self):
        """Stop the Modal app and release the container."""
        if self._app_ctx is not None:
            self._app_ctx.__exit__(None, None, None)
            self._app_ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def register(self, name: str, provider: TableProvider) -> None:
        self.catalog.register(name, provider)

    # ---- the two entry points ---------------------------------------

    def sql(self, text: str, order: str | None = None) -> "Query":
        from quail.sqlfront import compile_sql
        logical = compile_sql(text, self.catalog, self.tokenizer)
        return Query(self, logical, order=order)

    def docs(self, name: str) -> "BoundBuilder":
        from quail.builder import Query as BuilderQuery
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

    def scan(self, provider_name: str, column: str):
        """Return Arrow ID, text, and token columns for one scan."""
        provider = self.catalog.get(provider_name)
        key = ("scan", provider.content_identity(), column)
        if key not in self._scan_cache:
            columns = (provider.id_col,) if column == provider.id_col \
                else (provider.id_col, column)
            reader = provider.scan(ScanRequest(columns=columns))
            id_chunks = []
            text_chunks = []
            for batch in reader:
                id_chunks.append(batch.column(
                    batch.schema.get_field_index(provider.id_col)
                ))
                text_chunks.append(batch.column(
                    batch.schema.get_field_index(column)
                ))
            schema = provider.schema()
            ids = pa.chunked_array(
                id_chunks, type=schema.field(provider.id_col).type
            )
            texts = pa.chunked_array(
                text_chunks, type=schema.field(column).type
            )
            text_sample = texts.slice(0, 25).to_pylist()
            tok, note = pick_corpus_tokenizer(
                self.tokenizer, self._fast_tokenizer(), text_sample)
            self.notes.append(f"{provider_name}.{column}: {note}")
            token_rows = [tok(text.as_py()) for chunk in texts.chunks
                          for text in chunk]
            first_token = next(
                (token for row in token_rows for token in row), None)
            token_type = (pa.int32() if first_token is None
                          or isinstance(first_token, int)
                          else pa.string())
            toks = pa.array(
                token_rows, type=pa.large_list(token_type))
            self._scan_cache[key] = (ids, texts, toks)
        return self._scan_cache[key]

    def column_values(self, provider_name: str, column: str):
        """Return one raw Arrow column, cached."""
        provider = self.catalog.get(provider_name)
        key = ("vals", provider.content_identity(), column)
        if key not in self._scan_cache:
            reader = provider.scan(ScanRequest(columns=(column,)))
            chunks = [
                batch.column(batch.schema.get_field_index(column))
                for batch in reader
            ]
            self._scan_cache[key] = pa.chunked_array(
                chunks, type=provider.schema().field(column).type
            )
        return self._scan_cache[key]


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


def _true_false_ids(tok):
    """Return (true_ids, false_ids) first-token ids for TRUE/FALSE spellings."""
    true, false = set(), set()
    for w in ("TRUE", " TRUE", "True", " True"):
        ids = tok(w)
        if ids:
            true.add(ids[0])
    for w in ("FALSE", " FALSE", "False", " False"):
        ids = tok(w)
        if ids:
            false.add(ids[0])
    return sorted(true), sorted(false)


def _question_ids(session: Session, prompt) -> list:
    """Return token ids for the question suffix after the document."""
    text = re.sub(r"\{\d+\}", "", prompt.tail)
    return session.tokenizer(text)


def _join_spec(session: Session, prompt, anchor: str,
               partners: list) -> dict:
    """Build one join stage spec with tokenized frames and labels for all tables."""
    tok = session.tokenizer
    slot = {r.alias: i for i, r in enumerate(prompt.args)}
    aliases = [r.alias for r in prompt.args]
    return dict(
        anchor=anchor, partners=list(partners), aliases=aliases,
        frames={a: tok(render_join_frame(prompt.template, slot[a]))
                for a in aliases},
        labels={a: tok(join_label(slot[a])) for a in aliases},
        tail=tok(prompt.tail))


class Query:
    def __init__(self, session: Session, logical: LogicalPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self.order = order
        self._plan = None
        self._doc_tokens = None
        self._logical_rule_trace = ()

    # ---- planning (the optimization) ---------------------------------

    def plan(self):
        if self._plan is None:
            self.logical, self._logical_rule_trace = apply_logical_rules(
                self.logical,
                tuple(self.session.registry.logical_rules.values()),
                LogicalPlanningContext(
                    self.session.catalog, self.session.config
                ),
            )
            scans, _, _ = _collect(self.logical)
            self._doc_tokens = {}
            for s in scans:
                _, _, toks = self.session.scan(s.provider, s.column)
                self._doc_tokens[s.alias] = pc.list_value_length(
                    toks).to_pylist()
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                order=self.order,
                backend=self.session.config.backend,
                registry=self.session.registry)
        return self._plan

    def explain(self) -> str:
        return explain(self.logical, self.plan())

    # ---- execution -----------------------------------------------------

    def run(self, _execute=None) -> QueryResult:
        """Plan, execute on Modal, and return a lazy Arrow result.

        Args:
            _execute: Optional callable(payload) -> output for testing.
                None sends to the Modal worker.
        """
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        plan.graph.validate(runtime_keys=set(self.session.registry.runtimes))
        plan.graph.validate_backend(plan.backend)
        if plan.workers > 8:
            raise NotImplementedError(
                "more than 8 GPUs means multiple containers; the "
                "multi-container coordinator is a later step")
        scans, filters, joins = _collect(self.logical)
        if _execute is None and all(
                node.location is not ExecutionLocation.GPU_EXECUTOR
                for node in plan.nodes):
            return self._run_local(plan, scans)
        payload = self._payload(plan, scans, filters, joins)
        t0 = time.time()
        if _execute is None:
            worker = self.session.worker()
            k = plan.workers
            fn = (worker.execute if k == 1 else
                  worker.execute_2 if k == 2 else
                  worker.execute_4 if k <= 4 else worker.execute_8)
            out = fn.remote(payload)
        else:
            out = _execute(payload)
        coordinator_wall = time.time() - t0
        return self._assemble(plan, scans, filters, joins, out,
                              coordinator_wall)

    def _run_local(self, plan, scans) -> QueryResult:
        """Run a graph that has no GPU executor nodes."""
        from quail.runtime.runner import ExecutionContext, GenericRunner

        sources = {
            scan.alias: list(range(len(self._doc_tokens[scan.alias])))
            for scan in scans
        }
        scans_by_alias = {scan.alias: scan for scan in scans}

        def project(node, value):
            if isinstance(value, pa.Table):
                return value.select(node.columns)
            if len(scans_by_alias) != 1:
                raise ValueError(
                    "a local Project over document ids needs one input alias")
            alias = next(iter(scans_by_alias))
            indices = value.get(alias) if isinstance(value, dict) else value
            arrays = []
            for name in node.columns:
                column_alias, column = name.split(".", 1)
                if column_alias != alias:
                    raise ValueError(
                        f"local Project cannot read alias {column_alias!r}")
                source = scans_by_alias[column_alias]
                values = self.session.column_values(source.provider, column)
                arrays.append(pc.take(values, pa.array(indices)))
            return pa.table(arrays, names=list(node.columns))

        started = time.time()
        result = GenericRunner().run(
            plan.graph,
            ExecutionContext(
                runtimes=self.session.registry.runtimes,
                sources=sources,
                project=project,
            ),
        )
        value = result.value
        if not isinstance(value, pa.Table):
            columns = [f"{column.alias}.{column.column}"
                       for column in self.logical.root.columns]
            if len(columns) != 1:
                raise TypeError(
                    "a local graph root must return an Arrow table")
            value = pa.table({columns[0]: value})
        report = {
            "wall_s": round(time.time() - started, 4),
            "fresh_tokens": result.metrics.fresh_tokens,
            "regret_tokens": result.metrics.regret_tokens,
            "nodes": {
                node_id: node_result.metrics.__dict__
                for node_id, node_result in result.nodes.items()
            },
            "remarks": list(plan.remarks) + list(self.session.notes),
        }
        return QueryResult.from_table(value, report=report)

    def execute_stream(self, _execute=None, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run(_execute=_execute).execute_stream(
            batch_rows=batch_rows, limit=limit)

    def collect(self, _execute=None, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and explicitly collect one Arrow table."""
        return self.run(_execute=_execute).collect(
            limit=limit, batch_rows=batch_rows)

    # ---- payload -------------------------------------------------------

    def _payload(self, plan, scans, filters, joins) -> dict:
        from quail.runtime.tokens import encode_token_documents

        sess = self.session
        docs = {}
        encoded = {}
        for s in scans:
            _, _, toks = sess.scan(s.provider, s.column)
            key = (s.provider, s.column)
            if key not in encoded:
                encoded[key] = encode_token_documents(toks)
            docs[s.alias] = encoded[key]
        filter_qids = {}
        for node in plan.nodes:
            if not isinstance(node, PackedFilter):
                continue
            alias = node.alias
            preds = filters[alias]
            filter_qids[alias] = [
                _question_ids(sess, preds[stage.written_pos].prompt)
                for stage in node.stages]
        # one spec per stage in expected execution order
        join_specs = []
        for stage in plan.expected_join_stages():
            j = joins[stage.written_pos]
            spec = _join_spec(sess, j.predicate, stage.anchor,
                              list(stage.partners))
            spec["semantics"] = stage.semantics
            spec["selectivity"] = j.selectivity
            spec["written_pos"] = stage.written_pos
            # a full join without a user override lets the post-filter
            # join DP choose either orientation
            spec["anchor_free"] = (j.anchor is None
                                   and j.semantics == "full")
            join_specs.append(spec)
        true_ids, false_ids = _true_false_ids(sess.tokenizer)
        encoded_nodes = []
        for node in plan.nodes:
            if isinstance(node, PackedFilter):
                node = replace(
                    node,
                    question_token_ids=tuple(
                        tuple(question)
                        for question in filter_qids[node.alias]
                    ),
                )
            elif isinstance(node, AdaptiveJoinPlan):
                node = replace(node, join_specs=tuple(join_specs))
            encoded_nodes.append(node)
        runtime_plan = replace(
            plan, nodes=tuple(encoded_nodes), root=plan.root
        )
        return dict(
            physical_plan=runtime_plan.to_envelope(
                sess.registry.codecs, include_runtime_data=True
            ),
            model=sess.model.name,
            kv_dtype=plan.kv_dtype,
            chunk_tokens=plan.chunk_tokens,
            # the worker re-runs the join search on actual survivors
            # under the same order rule
            order_rule=plan.order_rule,
            workers=plan.workers,
            # the payload limit is the per-filter admission cap. With
            # joins, capping a table's filter would drop join inputs
            # and change the result, so it is only sent for pure
            # filter queries; _assemble truncates the output rows
            # either way
            limit=plan.limit if not join_specs else None,
            true_ids=true_ids, false_ids=false_ids,
            # the engine preamble, once: the worker prepends it to
            # every KV-owning document (filter scans, join anchors)
            pre_ids=sess.tokenizer(SHARED_PRE),
            docs=docs)

    # ---- sink: gate, replay-check, project ------------------------------

    def _assemble(self, plan, scans, filters, joins, out,
                  coordinator_wall) -> QueryResult:
        from quail.executor.pack import gate

        report = dict(
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            boot_kind=out.get("boot_kind"),
            boot=out.get("boot"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"],
            regret_tokens=out.get("regret_tokens"), stages=[],
            peak_gib=out.get("peak_gib"),
            order_rule=plan.order_rule,
            join_optimizer=out.get("join_optimizer"),
            expected_join_plan=[
                {
                    "type": node.type_name,
                    "id": node.node_id,
                    **node.explain_fields(),
                }
                for node in plan.expected_join_nodes()
            ],
            executed_join_plan=(out.get("join_optimizer") or {}).get(
                "executed_plan", []
            ),
            kv_manager=out.get("kv_manager"),
            result_volume_path=out.get("result_volume_path"),
            remarks=list(plan.remarks) + list(self.session.notes))
        answer_rows = dict(filters=out["filters"], joins=out["joins"])
        answer_tables = {"filters": {}, "joins": {}}

        # filter survivors + observed selectivities
        survivors = {}
        for s in scans:
            n = len(self._doc_tokens[s.alias])
            survivors[s.alias] = list(range(n))
        for node in plan.nodes:
            if not isinstance(node, PackedFilter):
                continue
            alias = node.alias
            rows = out["filters"][alias]
            n_stages = len(node.stages)
            for si, stage in enumerate(node.stages):
                answered = [d for d, row in rows.items()
                            if len(row) > si]
                passed = [d for d in answered if rows[d][si]]
                written_pos = stage.written_pos
                answer_tables["filters"][(alias, written_pos)] = \
                    answer_table(
                        {alias: answered},
                        [bool(rows[d][si]) for d in answered],
                        "filter_answers",
                        {"alias": alias, "written_pos": written_pos},
                    )
                report["stages"].append(dict(
                    op="filter", alias=alias, stage=si,
                    provided_selectivity=stage.selectivity,
                    observed_selectivity=round(
                        len(passed) / max(1, len(answered)), 4),
                    evaluated=len(answered)))
            survivors[alias] = sorted(
                d for d, row in rows.items()
                if len(row) == n_stages and all(row))

        # join stages in expected execution order: rows are over local
        # indices; map through the
        # index lists the worker reports. partner_index entries are
        # index tuples, one global index per partner alias. The
        # worker reports each stage's ACTUAL anchor (a barrier-time
        # re-pick may differ from the compile-time one); the plan's
        # stage dict is the fallback for executors that do not.
        stage_plan = plan.expected_join_stages()
        true_join_tables = {}
        full_join_order = []
        for st, jout in zip(stage_plan, out["joins"]):
            written_pos = jout.get("written_pos", st.written_pos)
            anchor = jout.get("anchor", st.anchor)
            partners = list(jout.get("partners", st.partners))
            semantics = jout.get("semantics", st.semantics)
            selectivity = jout.get("selectivity", st.selectivity)
            rows = jout["rows"]
            anchor_map = jout["anchor_index"]
            partner_map = jout["partner_index"]
            evaluated = sum(len(r) for r in rows.values())
            yes = sum(sum(r) for r in rows.values())
            report["stages"].append(dict(
                op="join", anchor=anchor,
                partners=partners,
                semantics=semantics,
                provided_selectivity=selectivity,
                observed_selectivity=round(yes / max(1, evaluated), 4),
                tuples=evaluated))
            global_rows = {anchor_map[a]: r for a, r in rows.items()}
            columns = {alias: [] for alias in [anchor, *partners]}
            bits = []
            for raw_local, answers in rows.items():
                global_anchor = anchor_map[int(raw_local)]
                for tuple_index, answer in enumerate(answers):
                    columns[anchor].append(global_anchor)
                    for alias, global_partner in zip(
                            partners, partner_map[tuple_index]):
                        columns[alias].append(global_partner)
                    bits.append(bool(answer))
            answers_table = answer_table(
                columns,
                bits,
                "join_answers",
                {
                    "written_pos": written_pos,
                    "anchor": anchor,
                    "partners": ",".join(partners),
                },
            )
            answer_tables["joins"][written_pos] = answers_table
            if semantics == "full":
                true_join_tables[written_pos] = true_answer_rows(
                    answers_table)
                full_join_order.append(written_pos)
            else:
                keep = set(gate(global_rows))
                if semantics == "exists":
                    survivors[anchor] = [d for d in survivors[anchor]
                                         if d in keep]
                else:
                    survivors[anchor] = [d for d in survivors[anchor]
                                         if d not in keep]

        cols = [f"{c.alias}.{c.column}" for c in
                self.logical.root.columns]
        survivor_arrays = {
            alias: pa.array(indices, type=pa.int32())
            for alias, indices in survivors.items()
        }
        result_declaration, result_index_schema = build_result_declaration(
            [true_join_tables[pos] for pos in full_join_order],
            survivor_arrays,
            self.logical.root.columns[0].alias,
        )

        projection = []
        output_fields = []
        for name, column in zip(cols, self.logical.root.columns):
            if column.alias not in result_index_schema.names:
                raise CompileError(
                    f"projection column {name} is not part of the result")
            values = self.session.column_values(
                column.provider, column.column)
            if isinstance(values, pa.ChunkedArray):
                values = values.combine_chunks()
            projection.append((column.alias, values))
            output_fields.append(pa.field(
                name,
                values.type,
                nullable=values.null_count > 0,
                metadata={
                    b"quail.alias": column.alias.encode("utf-8"),
                    b"quail.provider": column.provider.encode("utf-8"),
                    b"quail.column": column.column.encode("utf-8"),
                },
            ))
        output_schema = pa.schema(
            output_fields,
            metadata={
                b"quail.schema_version": b"1",
                b"quail.kind": b"query_result",
            },
        )
        return QueryResult(
            columns=cols,
            declaration=result_declaration,
            document_index_schema=result_index_schema,
            output_schema=output_schema,
            projection=projection,
            report=report,
            answer_rows=answer_rows,
            answer_tables=answer_tables,
            limit=plan.limit,
            survivor_indices=survivor_arrays,
            true_join_tables=true_join_tables,
        )
