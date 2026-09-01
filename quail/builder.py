"""Python builder that mirrors AI SQL construct for construct."""

from dataclasses import dataclass
from typing import Optional

from quail.catalog import Catalog
from quail.logical import (ColumnRef, CompileError, FilterPredicate,
                           JoinSpec, LogicalPlan, LogicalPlanBuilder,
                           bind_join_prompt, bind_prompt)


@dataclass(frozen=True)
class ColSpec:
    alias: Optional[str]
    column: str


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
        bound, aliases = self._bind(p)
        if len(aliases) != 1:
            raise CompileError(
                f"ai_filter must reference exactly one provider, got "
                f"{aliases}; a two-provider predicate is ai_join")
        self._filters.setdefault(aliases[0], []).append(
            FilterPredicate(prompt=bound, selectivity=selectivity))
        return self

    def ai_join(self, others, p: PromptSpec,
                selectivity: Optional[float] = None,
                anchor: Optional[str] = None,
                semantics: str = "full") -> "Query":
        """Join one or more tables with a single prompt.

        Args:
            others: One docs() query or a list of them.
            p: Prompt with one placeholder per table it references.
            selectivity: Fraction of tuples expected to pass.
            anchor: Table alias whose KV is kept across tuples.
            semantics: "full", "exists", or "anti".
        """
        if semantics not in ("full", "exists", "anti"):
            raise CompileError(f"semantics must be full, exists, or "
                               f"anti, got {semantics!r}")
        others = [others] if isinstance(others, Query) else list(others)
        if semantics != "full" and len(others) != 1:
            raise CompileError(
                "an exists/anti gate takes exactly one inner table; "
                "use one ai_join call per gate")
        new_aliases = []
        for other in others:
            if not isinstance(other, Query) or other._joins \
                    or len(other._tables) != 1:
                raise CompileError(
                    "every joined side of ai_join must be a single "
                    "(optionally filtered) docs(...) query")
            alias, provider = other._tables[0]
            if alias in self._scope():
                raise CompileError(f"duplicate table alias {alias!r}")
            self._tables.append((alias, provider))
            for a, preds in other._filters.items():
                self._filters.setdefault(a, []).extend(preds)
            for a, c in other._doc_columns.items():
                self._note_doc_column(
                    ColumnRef(alias=a, provider=self._scope()[a],
                              column=c))
            new_aliases.append(alias)
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
            )
        for join in self._joins:
            logical.add_join(join)
        return logical.project(tuple(columns), self._limit)


def docs(catalog: Catalog, provider: str, tokenizer=None) -> Query:
    return Query(catalog, provider, tokenizer)
