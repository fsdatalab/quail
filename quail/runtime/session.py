"""Session and Query: the user-facing API for registering documents,
compiling queries, planning, and executing on Modal.
"""

import re
import time

import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog import Catalog, DocumentProvider
from quail.logical import (
    SHARED_PRE,
    CompileError,
    LogicalPlan,
    join_label,
    render_join_frame,
)
from quail.planner.decide import _collect, explain, plan_query
from quail.planner.plan import EngineConfig, Refusal, resolve_model
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
                 device: str = "h100-sxm", tokenizer=None):
        model = resolve_model(config.model)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = DEVICES[device]
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

    def register(self, name: str, provider: DocumentProvider) -> None:
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
        key = ("scan", provider.source, column)
        if key not in self._scan_cache:
            table = provider.read_column(column)
            ids = table.column(provider.id_col)
            texts = table.column(column)
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
        key = ("vals", provider.source, column)
        if key not in self._scan_cache:
            table = provider.read_column(column)
            self._scan_cache[key] = table.column(column)
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

    # ---- planning (the optimization) ---------------------------------

    def plan(self):
        if self._plan is None:
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
                order=self.order)
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
        if plan.workers > 8:
            raise NotImplementedError(
                "more than 8 GPUs means multiple containers; the "
                "multi-container coordinator is a later step")
        scans, filters, joins = _collect(self.logical)
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
        filter_writes = {}
        for node in plan.nodes:
            if node["op"] != "FilterChain":
                continue
            alias = node["alias"]
            preds = filters[alias]
            filter_qids[alias] = [
                _question_ids(sess, preds[st["written_pos"]].prompt)
                for st in node["stages"]]
            filter_writes[alias] = node["arena_writes"]
        # one spec per stage, in execution order (the JoinGroup nodes'
        # stage_idxs index into this list)
        join_specs = []
        for node in plan.nodes:
            if node["op"] != "JoinGroup":
                continue
            for st in node["stages"]:
                j = joins[st["written_pos"]]
                spec = _join_spec(sess, j.predicate, st["anchor"],
                                  st["partners"])
                spec["semantics"] = st["semantics"]
                spec["selectivity"] = j.selectivity
                spec["written_pos"] = st["written_pos"]
                # a full join without a user override lets the post-filter
                # join DP choose either orientation
                spec["anchor_free"] = (j.anchor is None
                                       and j.semantics == "full")
                join_specs.append(spec)
        true_ids, false_ids = _true_false_ids(sess.tokenizer)
        shards = {node["alias"]: node["shards"] for node in plan.nodes
                  if node["op"] == "DocScan"}
        # the plan's node graph rides along so the worker executes the
        # structure the planner emitted (groups, barriers) instead of
        # re-deriving it; shard lists already ship separately
        plan_nodes = []
        for n in plan.nodes:
            n = dict(n)
            n.pop("shards", None)
            n.pop("shard_token_loads", None)
            plan_nodes.append(n)
        return dict(
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
            shards=shards,
            true_ids=true_ids, false_ids=false_ids,
            # the engine preamble, once: the worker prepends it to
            # every KV-owning document (filter scans, join anchors)
            pre_ids=sess.tokenizer(SHARED_PRE),
            docs=docs,
            filters=filter_qids,
            # the planner's per-chain call on whether the arena is
            # written; run_filter requires it and never derives it
            filter_arena_writes=filter_writes,
            joins=join_specs,
            plan_nodes=plan_nodes)

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
            if node["op"] != "FilterChain":
                continue
            alias = node["alias"]
            rows = out["filters"][alias]
            n_stages = len(node["stages"])
            for si, st in enumerate(node["stages"]):
                answered = [d for d, row in rows.items()
                            if len(row) > si]
                passed = [d for d in answered if rows[d][si]]
                written_pos = st["written_pos"]
                answer_tables["filters"][(alias, written_pos)] = \
                    answer_table(
                        {alias: answered},
                        [bool(rows[d][si]) for d in answered],
                        "filter_answers",
                        {"alias": alias, "written_pos": written_pos},
                    )
                report["stages"].append(dict(
                    op="filter", alias=alias, stage=si,
                    provided_selectivity=st["selectivity"],
                    observed_selectivity=round(
                        len(passed) / max(1, len(answered)), 4),
                    evaluated=len(answered)))
            survivors[alias] = sorted(
                d for d, row in rows.items()
                if len(row) == n_stages and all(row))

        # join stages, in execution order (the plan's JoinGroup nodes
        # flattened): rows are over local indices; map through the
        # index lists the worker reports. partner_index entries are
        # index tuples, one global index per partner alias. The
        # worker reports each stage's ACTUAL anchor (a barrier-time
        # re-pick may differ from the compile-time one); the plan's
        # stage dict is the fallback for executors that do not.
        stage_plan = []
        for node in plan.nodes:
            if node["op"] == "JoinGroup":
                stage_plan.extend(node["stages"])
        true_join_tables = {}
        full_join_order = []
        for st, jout in zip(stage_plan, out["joins"]):
            written_pos = jout.get("written_pos", st["written_pos"])
            anchor = jout.get("anchor", st["anchor"])
            partners = list(jout.get("partners", st["partners"]))
            semantics = jout.get("semantics", st["semantics"])
            selectivity = jout.get("selectivity", st["selectivity"])
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
