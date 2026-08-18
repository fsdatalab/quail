"""Session: the user surface from the design's section 2.

    sess = quail.Session(EngineConfig(gpus=1))
    sess.register("reviews", DocumentProvider.from_parquet(...))
    q = sess.sql("SELECT r.id FROM reviews r WHERE AI_FILTER(...)")
    q.explain()   # logical tree + chosen physical plan, no run
    res = q.run() # plans (the optimization), executes on Modal,
                  # replay-checks, applies the projection

Both entry points return the same runnable Query: sess.sql() compiles
AI SQL, sess.docs() starts the builder.

What run() supports today (one worker, one GPU): filter-only queries,
one full join (with filters pushed down on either side), a two-stage
chain sharing one anchor, and exists/anti stages on the chain's
anchor. Deeper shapes raise a plain error until the coordinator's
n-way merge lands. The KV store and GPU snapshots are later steps;
cpu_memory_gb is carried but not yet consumed.
"""

import re
import time
from dataclasses import dataclass, field

from quail.catalog import Catalog, DocumentProvider
from quail.logical import CompileError, LogicalPlan
from quail.planner.decide import _collect, _join_sides, explain, plan_query
from quail.planner.plan import EngineConfig, Refusal, resolve_model
from quail.specs import DEVICES


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
    report: dict           # measured walls, tokens, pairs per stage,
    #                        provided vs observed selectivity
    answer_rows: dict      # per-stage answer matrices (replay input)

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

    def ai_join(self, other, p, selectivity=None, anchor=None,
                semantics="full"):
        inner_other = other._inner if isinstance(other, BoundBuilder) \
            else other
        self._inner.ai_join(inner_other, p, selectivity=selectivity,
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
    every part of the template except the document placeholders."""
    text = re.sub(r"\{\d+\}", "", prompt.template)
    return session.tokenizer(text)


def _join_segments(session: Session, prompt):
    """(pre_ids, mid_ids, tail_ids): the template text before the
    first placeholder, between the two, and after the second."""
    m = list(re.finditer(r"\{\d+\}", prompt.template))
    pre = prompt.template[:m[0].start()]
    mid = prompt.template[m[0].end():m[1].start()]
    tail = prompt.template[m[1].end():]
    tok = session.tokenizer
    return tok(pre), tok(mid), tok(tail)


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
                self._doc_tokens[s.alias] = [len(t) for t in toks]
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                kv_dtype=self.session.config.kv_dtype,
                order=self.order)
        return self._plan

    def explain(self) -> str:
        return explain(self.logical, self.plan())

    # ---- execution -----------------------------------------------------

    def run(self, _execute=None) -> Result:
        """Plan, execute, replay-check, project. _execute is the
        worker seam: None ships to the Modal worker; tests inject a
        callable payload -> worker output."""
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        if plan.workers != 1:
            raise NotImplementedError(
                "multi-GPU dispatch is a later step; run with gpus "
                "sized for one worker")
        scans, filters, joins = _collect(self.logical)
        payload = self._payload(plan, scans, filters, joins)
        t0 = time.time()
        if _execute is None:
            from quail.runtime import worker
            with worker.app.run():
                out = worker.execute.remote(payload)
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
            if len(j.predicate.args) != 2:
                raise NotImplementedError(
                    "join prompts with more than two placeholders are "
                    "not executable yet")
            pre, mid, tail = _join_segments(sess, j.predicate)
            a1, _ = _join_sides(j)
            join_specs.append(dict(
                anchor=op["anchor"], partner=op["partner"],
                semantics=op["semantics"],
                # the anchor document always sits first so its KV is
                # pair-independent; when the anchor is the template's
                # second placeholder the two documents swap slots
                swapped=op["anchor"] != a1,
                pre=pre, mid=mid, tail=tail))
        yes_ids, no_ids = _yes_no_ids(sess.tokenizer)
        return dict(
            model=sess.model.name,
            kv_dtype=plan.kv_dtype,
            chunk_tokens=plan.chunk_tokens,
            yes_ids=yes_ids, no_ids=no_ids,
            docs=docs,
            filters=filter_qids,
            joins=join_specs)

    # ---- sink: gate, replay-check, project ------------------------------

    def _assemble(self, plan, scans, filters, joins, out,
                  coordinator_wall) -> Result:
        from quail.executor.pack import (assemble, brute_force_triples,
                                         gate, matches)

        report = dict(
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"], stages=[],
            order_rule=plan.order_rule,
            calibration=plan.calibration_source,
            remarks=list(plan.remarks) + list(self.session.notes))
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
        # index lists the worker reports
        join_ops = [op for op in plan.operators
                    if op["op"] == "JoinStage"]
        full_stages = []
        for op, jout in zip(join_ops, out["joins"]):
            rows = jout["rows"]
            anchor_map = jout["anchor_index"]
            partner_map = jout["partner_index"]
            pairs = sum(len(r) for r in rows.values())
            yes = sum(sum(r) for r in rows.values())
            report["stages"].append(dict(
                op="join", anchor=op["anchor"], partner=op["partner"],
                semantics=op["semantics"],
                provided_selectivity=op["selectivity"],
                observed_selectivity=round(yes / max(1, pairs), 4),
                pairs=pairs))
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

        # ---- output tuples per supported shape
        cols = [f"{c.alias}.{c.column}" for c in
                self.logical.root.columns]
        if not full_stages:
            alias_order = [scans[0].alias]
            tuples = [(d,) for d in survivors[scans[0].alias]]
        elif len(full_stages) == 1:
            op, rows, partner_map = full_stages[0]
            sem_keep = set(survivors[op["anchor"]])
            tuples, alias_order = [], [op["anchor"], op["partner"]]
            for a, ps in matches(rows).items():
                if a not in sem_keep:
                    continue
                for p in ps:
                    tuples.append((a, partner_map[p]))
        elif (len(full_stages) == 2
              and full_stages[0][0]["anchor"]
              == full_stages[1][0]["anchor"]):
            (op1, rows1, pm1), (op2, rows2, pm2) = full_stages
            triples = assemble(rows1, rows2)
            reference = brute_force_triples(rows1, rows2)
            report["replay_check"] = (triples == reference)
            alias_order = [op1["partner"], op1["anchor"],
                           op2["partner"]]
            tuples = [(pm1[a], b, pm2[c]) for a, b, c in triples]
        else:
            raise NotImplementedError(
                "the coordinator assembles filter-only queries, one "
                "full join, or a two-stage chain on one anchor; "
                "deeper shapes are a later step")

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
