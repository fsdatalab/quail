"""Python builder that mirrors AI SQL construct for construct."""

from dataclasses import dataclass
from typing import Optional

from quail.catalog import Catalog
from quail.logical import (
    ColumnRef,
    CompileError,
    Equality,
    FilterPredicate,
    JoinSpec,
    LogicalPlan,
    LogicalPlanBuilder,
    bind_join_prompt,
    bind_prompt,
)


@dataclass(frozen=True, eq=False)
class ColSpec:
    alias: Optional[str]
    column: str

    def __eq__(self, other):
        """``col("c.url") == col("e.url")`` is a join condition."""
        if not isinstance(other, ColSpec):
            return NotImplemented
        return EqualsSpec(self, other)

    def __hash__(self):
        return hash((self.alias, self.column))


@dataclass(frozen=True)
class EqualsSpec:
    """An unresolved ``col(...) == col(...)`` join condition."""
    left: ColSpec
    right: ColSpec


@dataclass(frozen=True)
class PromptSpec:
    template: str
    cols: tuple


def col(ref: str) -> ColSpec:
    """"r.review" or "review" (unqualified resolves when unambiguous)."""
    if "." in ref:
        alias, column = ref.split(".", 1)
        return ColSpec(alias=alias, column=column)
    return ColSpec(alias=None, column=ref)


def prompt(template: str, *cols: ColSpec) -> PromptSpec:
    return PromptSpec(template=template, cols=tuple(cols))


class Query:
    def __init__(self, catalog: Catalog, provider: str, tokenizer=None):
        catalog.get(provider)     # unknown provider -> CompileError
        self._catalog = catalog
        self._tokenizer = tokenizer
        self._tables = [(provider, provider)]   # (alias, provider)
        self._doc_columns = {}
        self._filters = {}
        self._joins = []
        self._pending_join = None    # (new aliases, conditions, applies)
        #                              awaiting the AI predicate over
        #                              its pairs
        self._applies = {}           # alias -> [(name, kind, ids, refs)]
        self._functions = {}         # name -> the Python function
        self._limit = None

    # ---- scope -------------------------------------------------------

    def alias(self, a: str) -> "Query":
        if self._joins or self._filters or len(self._tables) != 1:
            raise CompileError("alias() must come right after docs()")
        self._tables[0] = (a, self._tables[0][1])
        return self

    def _scope(self) -> dict:
        return dict(self._tables)

    def _resolve(self, c: ColSpec) -> ColumnRef:
        scope = self._scope()
        if c.alias is not None:
            if c.alias not in scope:
                raise CompileError(f"unknown table alias {c.alias!r}")
            provider = scope[c.alias]
            if c.column not in self._catalog.get(provider).columns:
                raise CompileError(
                    f"column {c.column!r} not in provider {provider!r} "
                    f"(schema: {self._catalog.get(provider).columns})")
            return ColumnRef(alias=c.alias, provider=provider,
                             column=c.column)
        owners = [(a, p) for a, p in scope.items()
                  if c.column in self._catalog.get(p).columns]
        if len(owners) != 1:
            raise CompileError(
                f"column {c.column!r} is "
                f"{'ambiguous' if owners else 'unknown'}; qualify it "
                f"with a table alias")
        return ColumnRef(alias=owners[0][0], provider=owners[0][1],
                         column=c.column)

    def _note_doc_column(self, ref: ColumnRef) -> None:
        seen = self._doc_columns.get(ref.alias)
        if seen and seen != ref.column:
            raise CompileError(
                f"alias {ref.alias!r} is referenced through two "
                f"columns ({seen!r}, {ref.column!r}); predicates over "
                f"one table must share one document column so its KV "
                f"is computed once")
        self._doc_columns[ref.alias] = ref.column

    def _bind(self, p: PromptSpec, join: bool = False):
        refs = tuple(self._resolve(c) for c in p.cols)
        binder = bind_join_prompt if join else bind_prompt
        bound = binder(p.template, refs, self._tokenizer)
        for r in refs:
            self._note_doc_column(r)
        aliases = []
        for r in refs:
            if r.alias not in aliases:
                aliases.append(r.alias)
        return bound, aliases

    # ---- the query surface --------------------------------------------

    def ai_filter(self, p: PromptSpec,
                  selectivity: Optional[float] = None) -> "Query":
        """Ask the prompt of every document, or of every joined pair.

        A prompt over one table filters its documents. A prompt over
        the tables of the preceding join() is the AI predicate asked
        of that join's pairs.
        """
        if self._pending_join is not None:
            return self._ai_join_pairs(p, selectivity)
        bound, aliases = self._bind(p)
        if len(aliases) != 1:
            raise CompileError(
                f"ai_filter must reference exactly one provider, got "
                f"{aliases}; a two-provider predicate follows join() or "
                f"is ai_join")
        self._filters.setdefault(aliases[0], []).append(
            FilterPredicate(prompt=bound, selectivity=selectivity))
        return self

    def join(self, other: "Query", on=None) -> "Query":
        """Join one table on ordinary column equalities.

        The next ai_filter() must name this table and one already in
        the query; the model then sees only the pairs the equalities
        allow. With on=None every pair is a candidate.

        Args:
            other: One docs() query, optionally filtered.
            on: ``col("c.url") == col("e.url")`` or a list of them.
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "add that predicate before joining another table")
        new_aliases = self._absorb([other])
        conditions = [] if on is None else (
            [on] if isinstance(on, EqualsSpec) else list(on))
        resolved = []
        for condition in conditions:
            if not isinstance(condition, EqualsSpec):
                raise CompileError(
                    "join(on=...) takes col(...) == col(...) conditions")
            left, right = (self._resolve(condition.left),
                           self._resolve(condition.right))
            sides = {left.alias, right.alias}
            if new_aliases[0] not in sides or len(sides) != 2:
                raise CompileError(
                    f"join condition {left.alias}.{left.column} = "
                    f"{right.alias}.{right.column} must relate the "
                    f"joined table {new_aliases[0]!r} to a table already "
                    f"in the query")
            resolved.append(Equality(left, right))
        self._pending_join = (new_aliases, tuple(resolved), [])
        return self

    def _ai_join_pairs(self, p: PromptSpec, selectivity):
        new_aliases, conditions, applies = self._pending_join
        bound, aliases = self._bind(p, join=True)
        if len(aliases) != 2 or new_aliases[0] not in aliases:
            raise CompileError(
                f"the predicate after join() must name the joined "
                f"table {new_aliases[0]!r} and one other table, got "
                f"{aliases}")
        for name, _kind, refs in applies:
            outside = sorted({ref.alias for ref in refs} - set(aliases))
            if outside:
                raise CompileError(
                    f"apply {name!r} reads tables {outside} that the join "
                    f"predicate over {aliases} does not join")
        self._pending_join = None
        self._joins.append(JoinSpec(aliases=tuple(new_aliases),
                                    prompt=bound, selectivity=selectivity,
                                    on=conditions, applies=tuple(applies)))
        return self

    def apply(self, fn, columns=(), *, name=None, ids=None,
              kind="per_batch") -> "Query":
        """Call a Python function between two operators.

        The function receives a dict of Arrow tables keyed by alias:
        each table holds the alias's row indices under the alias name
        plus the listed columns. After ``join()`` it returns the pairs
        the next AI predicate is asked about, as a table with both
        alias columns. Otherwise it works on one table and returns the
        ids to keep. A function never invents an id.

        Args:
            fn: The function. Registered on the session under name.
            columns: ``col(...)`` references the function reads.
            name: Registered name; the function's name by default.
            ids: "drop" (default, may leave ids out), "preserve"
                (returns every id), or "pairs" (after join()).
            kind: "per_batch" runs on each batch a streaming operator
                hands over; "barrier" runs once over every survivor.
        """
        if not callable(fn):
            raise CompileError("apply() needs a callable")
        name = name or getattr(fn, "__name__", None)
        if not name or name == "<lambda>":
            raise CompileError("apply() needs a name for a lambda")
        if kind not in ("per_batch", "barrier"):
            raise CompileError(
                f"apply kind must be per_batch or barrier, got {kind!r}")
        known = self._functions.get(name)
        if known is not None and known is not fn:
            raise CompileError(
                f"apply name {name!r} is already used by another function")
        columns = [columns] if isinstance(columns, ColSpec) else list(columns)
        refs = tuple(self._resolve(spec) for spec in columns)
        if self._pending_join is not None:
            if ids not in (None, "pairs"):
                raise CompileError(
                    "an apply() after join() returns the pairs to ask "
                    "about; ids must be 'pairs'")
            self._pending_join[2].append((name, kind, refs))
            self._functions[name] = fn
            return self
        if ids == "pairs":
            raise CompileError(
                "an apply() returning pairs follows join()")
        ids = ids or "drop"
        if ids not in ("preserve", "drop"):
            raise CompileError(
                f"apply ids must be preserve, drop, or pairs, got {ids!r}")
        owners = {ref.alias for ref in refs}
        if len(owners) != 1:
            raise CompileError(
                f"apply {name!r} must work on one table; its columns "
                f"name {sorted(owners) or 'none'}")
        (alias,) = owners
        self._applies.setdefault(alias, []).append((name, kind, ids, refs))
        self._functions[name] = fn
        return self

    def apply_table(self, fn, columns=(), *, name=None, ids=None) -> "Query":
        """Call a function once over every survivor: apply() as a barrier."""
        return self.apply(fn, columns, name=name, ids=ids, kind="barrier")

    @property
    def functions(self) -> dict:
        """The Python functions apply() calls, by registered name."""
        return dict(self._functions)

    def _absorb(self, others) -> list:
        """Bring other single-table queries into scope; returns aliases."""
        new_aliases = []
        for other in others:
            if not isinstance(other, Query) or other._joins \
                    or other._pending_join is not None \
                    or len(other._tables) != 1:
                raise CompileError(
                    "every joined side must be a single (optionally "
                    "filtered) docs(...) query")
            alias, provider = other._tables[0]
            if alias in self._scope():
                raise CompileError(f"duplicate table alias {alias!r}")
            self._tables.append((alias, provider))
            for a, preds in other._filters.items():
                self._filters.setdefault(a, []).extend(preds)
            for a, applies in other._applies.items():
                self._applies.setdefault(a, []).extend(applies)
            for name, fn in other._functions.items():
                if self._functions.get(name, fn) is not fn:
                    raise CompileError(
                        f"apply name {name!r} is used by two functions")
                self._functions[name] = fn
            for a, c in other._doc_columns.items():
                self._note_doc_column(
                    ColumnRef(alias=a, provider=self._scope()[a],
                              column=c))
            new_aliases.append(alias)
        return new_aliases

    def ai_join(self, others, p: PromptSpec,
                selectivity: Optional[float] = None,
                anchor: Optional[str] = None,
                semantics: str = "full") -> "Query":
        """Join one or more tables with a single prompt over every tuple.

        Shorthand for join(other) followed by ai_filter(p) when every
        pair is a candidate.

        Args:
            others: One docs() query or a list of them.
            p: Prompt with one placeholder per table it references.
            selectivity: Fraction of tuples expected to pass.
            anchor: Table alias whose KV is kept across tuples.
            semantics: "full", "exists", or "anti".
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "add that predicate before ai_join")
        if semantics not in ("full", "exists", "anti"):
            raise CompileError(f"semantics must be full, exists, or "
                               f"anti, got {semantics!r}")
        others = [others] if isinstance(others, Query) else list(others)
        if semantics != "full" and len(others) != 1:
            raise CompileError(
                "an exists/anti gate takes exactly one inner table; "
                "use one ai_join call per gate")
        new_aliases = self._absorb(others)
        bound, aliases = self._bind(p, join=True)
        if semantics == "full":
            # each call's prompt must cover the tables that call
            # joins and touch at least one table already in the
            # query, so every join connects to the join graph
            missing = [a for a in new_aliases if a not in aliases]
            if missing:
                raise CompileError(
                    f"the join prompt must reference every table this "
                    f"call joins; {missing} joined but not referenced "
                    f"(got {aliases})")
            if all(a in new_aliases for a in aliases):
                raise CompileError(
                    f"the join prompt must reference at least one "
                    f"table already in the query, so this join "
                    f"connects to it; got only new tables {aliases}")
            if anchor is not None and anchor not in aliases:
                raise CompileError(
                    f"anchor {anchor!r} is not a table of this join "
                    f"({aliases})")
        else:
            inner = new_aliases[0]
            if len(aliases) != 2 or inner not in aliases:
                raise CompileError(
                    f"an exists/anti prompt must reference the inner "
                    f"table {inner!r} and exactly one outer table, "
                    f"got {aliases}")
            outer = next(a for a in aliases if a != inner)
            if anchor is not None and anchor != outer:
                raise CompileError(
                    f"exists/anti always anchor on the outer table "
                    f"{outer!r} - the gate applies to its documents - "
                    f"got anchor {anchor!r}")
            anchor = outer
        self._joins.append(JoinSpec(aliases=tuple(new_aliases),
                                    prompt=bound,
                                    semantics=semantics,
                                    selectivity=selectivity,
                                    anchor=anchor))
        return self

    def limit(self, n: int) -> "Query":
        if not isinstance(n, int) or n <= 0:
            raise CompileError("LIMIT must be a positive integer")
        self._limit = n
        return self

    def select(self, *cols) -> LogicalPlan:
        if self._pending_join is not None:
            raise CompileError(
                f"join() of {self._pending_join[0]} has no AI predicate "
                f"over its pairs; a plain join belongs in the database "
                f"the ids came from")
        if not self._joins and not self._filters:
            raise CompileError("the query has no AI predicate; a plain "
                               "scan belongs in the database the ids "
                               "came from")
        columns = []
        for c in cols:
            if c == "*":
                for alias, provider in self._tables:
                    for name in self._catalog.get(provider).columns:
                        columns.append(ColumnRef(alias=alias,
                                                 provider=provider,
                                                 column=name))
                continue
            spec = col(c) if isinstance(c, str) else c
            columns.append(self._resolve(spec))
        logical = LogicalPlanBuilder()
        for alias, provider in self._tables:
            logical.add_scan(
                alias,
                provider,
                self._doc_columns.get(alias, ""),
                tuple(self._filters.get(alias, ())),
                tuple(self._applies.get(alias, ())),
            )
        for join in self._joins:
            logical.add_join(join)
        return logical.project(tuple(columns), self._limit)


def docs(catalog: Catalog, provider: str, tokenizer=None) -> Query:
    return Query(catalog, provider, tokenizer)
