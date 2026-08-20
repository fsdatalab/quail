"""Session: the user surface from the design's section 2.

    sess = quail.Session(EngineConfig(gpus=1))
    sess.register("reviews", DocumentProvider.from_parquet(...))
    q = sess.sql("SELECT r.id FROM reviews r WHERE AI_FILTER(...)")
    q.explain()   # logical tree + chosen physical plan, no run
    res = q.run() # plans (the optimization), executes on Modal,
                  # replay-checks, applies the projection

Both entry points return the same runnable Query: sess.sql() compiles
AI SQL, sess.docs() starts the builder.

What run() supports: filter-only queries, exists/anti gates, and the
one full join - however many tables it spans, since an n-way join is
a single cross-product stage (every document of a tuple in one
prompt), never a chain of pairwise stages. The KV store and GPU
snapshots are later steps; cpu_memory_gb is carried but not yet
consumed.
"""

import hashlib
import os
import re
import time
from dataclasses import dataclass, field

from quail.catalog import Catalog, DocumentProvider
from quail.logical import (SHARED_PRE, CompileError, LogicalPlan,
                           join_anchor_note, join_label,
                           render_join_question)
from quail.planner.calibration import channel_bandwidths
from quail.planner.decide import _collect, explain, plan_query
from quail.planner.plan import (EngineConfig, Refusal, StoreSpec,
                                resolve_model)
from quail.specs import DEVICES

STORE_HEADROOM_GB = 16    # container memory the store must leave for
#                           weights loading, activations paging, and
#                           the Python process itself


class RefusalError(RuntimeError):
    """run() on a refused plan. The Refusal rides along."""

    def __init__(self, refusal: Refusal):
        self.refusal = refusal
        super().__init__(
            f"{refusal.constraint}: needed {refusal.needed} "
            f"{refusal.unit}, available {refusal.available}. "
            + " ".join(refusal.reasons))


@dataclass
class Result:
    columns: list          # projection column names, "alias.column"
    rows: list             # tuples of projected values
    report: dict           # measured walls, tokens, tuples per stage,
    #                        provided vs observed selectivity
    answer_rows: dict      # per-stage answer matrices, as returned

    def __len__(self):
        return len(self.rows)


def pick_corpus_tokenizer(primary, fast, texts, sample=25):
    """The corpus tokenizer for one scanned column: the fast one when
    it matches the primary on a sample of the column's real texts,
    the primary otherwise. bpe-qwen's own README warns that some
    multi-byte UTF-8 is mishandled, so the fast path earns each
    column instead of being trusted."""
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
        self._warm_hashes = set()  # content hashes stored by earlier
        #                            runs in this session
        self._app_ctx = None       # the Modal app held open for the
        #                            session, so the worker container
        #                            (its booted model and its store)
        #                            survives between run() calls
        self.store_enabled = True  # the benchmark's cold pass runs
        #                            with the store disabled
        self._flush_next = False

    def set_store(self, enabled: bool) -> None:
        self.store_enabled = enabled

    def flush_store(self) -> None:
        """The next run tells the worker to flush its store first -
        the benchmark's cold pass is a store flush, not a restart."""
        self._flush_next = True
        self._warm_hashes.clear()

    def worker(self):
        """The worker module, inside this session's long-lived app
        context. One app per session is what keeps the container -
        and with it the loaded model and the pinned store - warm
        across queries."""
        from quail.runtime import worker
        if self._app_ctx is None:
            self._app_ctx = worker.app.run()
            self._app_ctx.__enter__()
        return worker

    def close(self):
        """End the session: the app stops, the container scales down,
        the store is gone. A new session starts cold."""
        if self._app_ctx is not None:
            self._app_ctx.__exit__(None, None, None)
            self._app_ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def store_spec(self, hashes=()) -> StoreSpec | None:
        """The planner's view of the KV store: pinned bandwidth,
        capacity from the config, warm when every scanned content
        hash was stored by an earlier run of this session."""
        if not self.store_enabled:
            return None
        if self.config.cpu_memory_gb <= STORE_HEADROOM_GB:
            return None
        capacity = (self.config.cpu_memory_gb
                    - STORE_HEADROOM_GB) * 1e9
        warm = bool(hashes) and all(h in self._warm_hashes
                                    for h in hashes)
        return StoreSpec(read_bw=channel_bandwidths()["pinned_h2d"],
                         warm=warm, capacity_bytes=capacity)

    def content_hash(self, provider_name: str, column: str) -> str:
        """The store key prefix for one scanned column: provenance of
        (provider data, column, tokenizer, engine preamble). File
        identity is (path, size, mtime) - cheaper than hashing the
        bytes, and a rewritten file changes it. The preamble is part
        of every stored prefix, so changing its text must invalidate
        every stored extent."""
        provider = self.catalog.get(provider_name)
        ident = [provider.kind, provider.source, column,
                 self.model.name, SHARED_PRE]
        if provider.kind == "parquet" and os.path.exists(provider.source):
            st = os.stat(provider.source)
            ident += [str(st.st_size), str(int(st.st_mtime))]
        return hashlib.sha256(":".join(ident).encode()).hexdigest()[:16]

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
        """The exact tokenizer: prompts, YES/NO ids, parity samples."""
        if self._tok is None:
            from transformers import AutoTokenizer
            hf = AutoTokenizer.from_pretrained(self.model.hf_name)
            self._tok = lambda text: hf(
                text, add_special_tokens=False)["input_ids"]
        return self._tok

    def _fast_tokenizer(self):
        """bpe-qwen when installed (measured 12.8x on corpus text);
        None when unavailable or when a test injected its own."""
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
        """(ids, texts, token lists) for one provider column, the
        tokenization cached per session (the design's content-hash
        cache; in-memory for now)."""
        provider = self.catalog.get(provider_name)
        key = ("scan", provider.source, column)
        if key not in self._scan_cache:
            ids, texts = provider.read_column(column)
            tok, note = pick_corpus_tokenizer(
                self.tokenizer, self._fast_tokenizer(), texts)
            self.notes.append(f"{provider_name}.{column}: {note}")
            toks = [tok(t) for t in texts]
            self._scan_cache[key] = (ids, texts, toks)
        return self._scan_cache[key]

    def column_values(self, provider_name: str, column: str) -> list:
        """One column's values, for the projection - never tokenized."""
        provider = self.catalog.get(provider_name)
        key = ("vals", provider.source, column)
        if key not in self._scan_cache:
            _, values = provider.read_column(column)
            self._scan_cache[key] = values
        return self._scan_cache[key]


class BoundBuilder:
    """sess.docs(...): the builder, returning a runnable Query at
    select() instead of a bare LogicalPlan."""

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

    def select(self, *cols) -> "Query":
        # the builder is always as_written: the order you chain calls
        # is the order that runs (design section 2.5)
        return Query(self._session, self._inner.select(*cols),
                     order="as_written")


def _yes_no_ids(tok):
    """First-token ids of the YES/NO spellings, from the session's
    tokenizer callable (same rule as the executor's yes_no_ids)."""
    yes, no = set(), set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w)
        if ids:
            yes.add(ids[0])
    for w in ("NO", " NO", "No", " No", "N", " N"):
        ids = tok(w)
        if ids:
            no.add(ids[0])
    return sorted(yes), sorted(no)


def _question_ids(session: Session, prompt) -> list:
    """The question suffix the executor attaches after the document:
    the template text from the first placeholder on, placeholders
    excluded. The engine preamble (before the placeholder) ships once
    as the payload's pre_ids, never inside a question."""
    text = re.sub(r"\{\d+\}", "", prompt.tail)
    return session.tokenizer(text)


def _join_spec(session: Session, prompt, anchor: str,
               partners: list) -> dict:
    """The executor's view of one join: per-alias block labels (paid
    once per tuple, ahead of each partner document), the anchor's
    naming line (written into kept KV once per anchor, so the
    question can call it by alias), and the rendered question (paid
    once per tuple, after the last block). The engine preamble ships
    once as the payload's pre_ids."""
    tok = session.tokenizer
    return dict(
        anchor=anchor, partners=list(partners),
        frame=tok(join_anchor_note(anchor)),
        labels={p: tok(join_label(p)) for p in partners},
        tail=tok(render_join_question(prompt.template, prompt.args)))


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
            self._hashes = {}
            for s in scans:
                _, _, toks = self.session.scan(s.provider, s.column)
                self._doc_tokens[s.alias] = [len(t) for t in toks]
                self._hashes[s.alias] = self.session.content_hash(
                    s.provider, s.column)
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                store=self.session.store_spec(self._hashes.values()),
                kv_dtype=self.session.config.kv_dtype,
                order=self.order)
        return self._plan

    def explain(self) -> str:
        return explain(self.logical, self.plan())

    # ---- execution -----------------------------------------------------

    def run(self, _execute=None) -> Result:
        """Plan, execute, project. _execute is the worker seam: None
        ships to the Modal worker; tests inject a callable
        payload -> worker output."""
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

    # ---- payload -------------------------------------------------------

    def _payload(self, plan, scans, filters, joins) -> dict:
        sess = self.session
        docs = {}
        for s in scans:
            _, _, toks = sess.scan(s.provider, s.column)
            docs[s.alias] = toks
        filter_qids = {}
        for op in plan.operators:
            if op["op"] != "FilterChain":
                continue
            alias = op["alias"]
            preds = filters[alias]
            filter_qids[alias] = [
                _question_ids(sess, preds[st["written_pos"]].prompt)
                for st in op["stages"]]
        join_specs = []
        for op in plan.operators:
            if op["op"] != "JoinStage":
                continue
            j = joins[op["written_pos"]]
            spec = _join_spec(sess, j.predicate, op["anchor"],
                              op["partners"])
            spec["semantics"] = op["semantics"]
            join_specs.append(spec)
        yes_ids, no_ids = _yes_no_ids(sess.tokenizer)
        store = None
        spec = sess.store_spec(self._hashes.values())
        if spec is not None:
            store = dict(capacity_bytes=spec.capacity_bytes,
                         min_doc_tokens=max(1,
                                            plan.store_min_doc_tokens),
                         hashes=dict(self._hashes))
        shards = {op["alias"]: op["shards"] for op in plan.operators
                  if op["op"] == "DocScan"}
        flush = sess._flush_next
        sess._flush_next = False
        return dict(
            store_flush=flush,
            model=sess.model.name,
            kv_dtype=plan.kv_dtype,
            chunk_tokens=plan.chunk_tokens,
            workers=plan.workers,
            shards=shards,
            yes_ids=yes_ids, no_ids=no_ids,
            # the engine preamble, once: the worker prepends it to
            # every KV-owning document (filter scans, join anchors)
            pre_ids=sess.tokenizer(SHARED_PRE),
            docs=docs,
            filters=filter_qids,
            joins=join_specs,
            store=store)

    # ---- sink: gate, replay-check, project ------------------------------

    def _assemble(self, plan, scans, filters, joins, out,
                  coordinator_wall) -> Result:
        from quail.executor.pack import gate, matches

        report = dict(
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            boot_kind=out.get("boot_kind"),
            boot=out.get("boot"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"], stages=[],
            peak_gib=out.get("peak_gib"),
            store=out.get("store"),
            order_rule=plan.order_rule,
            calibration=plan.calibration_source,
            remarks=list(plan.remarks) + list(self.session.notes))
        if getattr(self, "_hashes", None) and out.get("store"):
            # later runs of this session may now plan access=restore
            self.session._warm_hashes.update(self._hashes.values())
        answer_rows = dict(filters=out["filters"], joins=out["joins"])

        # filter survivors + observed selectivities
        survivors = {}
        for s in scans:
            n = len(self._doc_tokens[s.alias])
            survivors[s.alias] = list(range(n))
        for op in plan.operators:
            if op["op"] != "FilterChain":
                continue
            alias = op["alias"]
            rows = out["filters"][alias]
            n_stages = len(op["stages"])
            for si, st in enumerate(op["stages"]):
                answered = [d for d, row in rows.items()
                            if len(row) > si]
                passed = [d for d in answered if rows[d][si]]
                report["stages"].append(dict(
                    op="filter", alias=alias, stage=si,
                    provided_selectivity=st["selectivity"],
                    observed_selectivity=round(
                        len(passed) / max(1, len(answered)), 4),
                    evaluated=len(answered)))
            survivors[alias] = sorted(
                d for d, row in rows.items()
                if len(row) == n_stages and all(row))

        # join stages: rows are over local indices; map through the
        # index lists the worker reports. partner_index entries are
        # index tuples, one global index per partner alias.
        join_ops = [op for op in plan.operators
                    if op["op"] == "JoinStage"]
        full_stages = []
        for op, jout in zip(join_ops, out["joins"]):
            rows = jout["rows"]
            anchor_map = jout["anchor_index"]
            partner_map = jout["partner_index"]
            evaluated = sum(len(r) for r in rows.values())
            yes = sum(sum(r) for r in rows.values())
            report["stages"].append(dict(
                op="join", anchor=op["anchor"],
                partners=list(op["partners"]),
                semantics=op["semantics"],
                provided_selectivity=op["selectivity"],
                observed_selectivity=round(yes / max(1, evaluated), 4),
                tuples=evaluated))
            global_rows = {anchor_map[a]: r for a, r in rows.items()}
            if op["semantics"] == "full":
                full_stages.append((op, global_rows, partner_map))
            else:
                keep = set(gate(global_rows))
                alias = op["anchor"]
                if op["semantics"] == "exists":
                    survivors[alias] = [d for d in survivors[alias]
                                        if d in keep]
                else:
                    survivors[alias] = [d for d in survivors[alias]
                                        if d not in keep]

        # ---- output tuples: the one full join's YES rows, each
        # member checked against its table's final survivor set (a
        # gate written after the join still applies - order changes
        # cost, never results)
        cols = [f"{c.alias}.{c.column}" for c in
                self.logical.root.columns]
        if not full_stages:
            alias_order = [scans[0].alias]
            tuples = [(d,) for d in survivors[scans[0].alias]]
        else:
            (op, rows, partner_map), = full_stages  # one by compile
            alias_order = [op["anchor"]] + list(op["partners"])
            keep = {a: set(survivors[a]) for a in alias_order}
            tuples = []
            for a, ts in matches(rows).items():
                if a not in keep[op["anchor"]]:
                    continue
                for ti in ts:
                    member = partner_map[ti]
                    if all(g in keep[al] for al, g in
                           zip(op["partners"], member)):
                        tuples.append((a, *member))

        proj_rows = []
        for tup in tuples:
            pos = {alias: idx for alias, idx in zip(alias_order, tup)}
            row = []
            for c in self.logical.root.columns:
                if c.alias not in pos:
                    raise CompileError(
                        f"projection column {c.alias}.{c.column} is "
                        f"not part of the result tuple")
                vals = self.session.column_values(c.provider, c.column)
                row.append(vals[pos[c.alias]])
            proj_rows.append(tuple(row))
        return Result(columns=cols, rows=proj_rows, report=report,
                      answer_rows=answer_rows)
