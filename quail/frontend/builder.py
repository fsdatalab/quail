"""Python builder that mirrors AI SQL construct for construct."""

from dataclasses import dataclass
from typing import Optional

from quail.catalog import Catalog
from quail.frontend.label_tables import read_label_table
from quail.logical import (
    AggregateCall,
    Aggregation,
    Alias,
    ColumnPredicate,
    ColumnRef,
    CompileError,
    Equality,
    FilterPredicate,
    HavingTest,
    InList,
    JoinSpec,
    LogicalPlan,
    LogicalPlanBuilder,
    ModelCall,
    SortKey,
    bind_classify_prompt,
    bind_join_prompt,
    bind_prompt,
    bind_score_prompt,
)
from quail.logical.nodes import validate_task_description


@dataclass(frozen=True, eq=False)
class ColSpec:
    alias: Optional[str]
    column: str

    def __eq__(self, other):
        """``col("c.url") == col("e.url")`` is a join condition.

        Compared with a literal, it is a column test for ``where()``.
        """
        if isinstance(other, ColSpec):
            return EqualsSpec(self, other)
        return PredicateSpec(self, "=", other)

    def __ne__(self, other):
        return PredicateSpec(self, "<>", other)

    def __lt__(self, other):
        return PredicateSpec(self, "<", other)

    def __le__(self, other):
        return PredicateSpec(self, "<=", other)

    def __gt__(self, other):
        return PredicateSpec(self, ">", other)

    def __ge__(self, other):
        return PredicateSpec(self, ">=", other)

    def isin(self, values) -> "PredicateSpec":
        return PredicateSpec(self, "in", tuple(values))

    def is_null(self) -> "PredicateSpec":
        return PredicateSpec(self, "is null")

    def is_not_null(self) -> "PredicateSpec":
        return PredicateSpec(self, "is not null")

    def asc(self) -> "SortSpec":
        return SortSpec(self)

    def desc(self) -> "SortSpec":
        return SortSpec(self, descending=True)

    def nulls_first(self) -> "SortSpec":
        return SortSpec(self, nulls="first")

    def nulls_last(self) -> "SortSpec":
        return SortSpec(self, nulls="last")

    def __hash__(self):
        return hash((self.alias, self.column))


@dataclass(frozen=True)
class SortSpec:
    """A sort key for ``order_by()``: ``col("r.stars").desc().nulls_last()``.

    Attributes:
        col: The column or projected name to sort by.
        descending: Whether to sort from the largest value down.
        nulls: ``"first"`` or ``"last"``, or None for the default: nulls
            last when ascending and first when descending.
    """

    col: ColSpec
    descending: bool = False
    nulls: Optional[str] = None

    def asc(self) -> "SortSpec":
        return SortSpec(self.col, False, self.nulls)

    def desc(self) -> "SortSpec":
        return SortSpec(self.col, True, self.nulls)

    def nulls_first(self) -> "SortSpec":
        return SortSpec(self.col, self.descending, "first")

    def nulls_last(self) -> "SortSpec":
        return SortSpec(self.col, self.descending, "last")


@dataclass(frozen=True)
class EqualsSpec:
    """An unresolved ``col(...) == col(...)`` join condition."""
    left: ColSpec
    right: ColSpec


@dataclass(frozen=True)
class PredicateSpec:
    """An unresolved column test for ``where()``."""
    col: ColSpec
    comparison: str
    value: object = None


@dataclass(frozen=True)
class PromptSpec:
    template: str
    cols: tuple


@dataclass(frozen=True)
class AggSpec:
    """An unresolved aggregate for ``agg()``."""
    function: str
    col: object = None    # ColSpec, a label or score name, or None

    def _test(self, comparison, value):
        return HavingSpec(self, comparison, value)

    def __eq__(self, other):
        return self._test("=", other)

    def __ne__(self, other):
        return self._test("<>", other)

    def __lt__(self, other):
        return self._test("<", other)

    def __le__(self, other):
        return self._test("<=", other)

    def __gt__(self, other):
        return self._test(">", other)

    def __ge__(self, other):
        return self._test(">=", other)

    def __hash__(self):
        return hash((self.function, self.col))


@dataclass(frozen=True)
class HavingSpec:
    """An unresolved ``having()`` test of an aggregate against a number."""
    agg: AggSpec
    comparison: str
    value: object


def count(c=None) -> AggSpec:
    """``count()`` counts rows; ``count("r.id")`` counts non-null values."""
    return AggSpec("count", c)


def count_distinct(c) -> AggSpec:
    return AggSpec("count_distinct", c)


def sum_(c) -> AggSpec:
    return AggSpec("sum", c)


def avg(c) -> AggSpec:
    return AggSpec("avg", c)


def min_(c) -> AggSpec:
    return AggSpec("min", c)


def max_(c) -> AggSpec:
    return AggSpec("max", c)


def col(ref: str) -> ColSpec:
    """"r.review" or "review" (unqualified resolves when unambiguous)."""
    if "." in ref:
        alias, column = ref.split(".", 1)
        return ColSpec(alias=alias, column=column)
    return ColSpec(alias=None, column=ref)


def prompt(template: str, *cols: ColSpec) -> PromptSpec:
    return PromptSpec(template=template, cols=tuple(cols))


class Query:
    def __init__(self, catalog: Catalog, provider: str, tokenizer=None,
                 turn: tuple[str, str] = ("", ""), layout: str = "ai-if"):
        catalog.get(provider)     # unknown provider -> CompileError
        self._catalog = catalog
        self._tokenizer = tokenizer
        self._turn = turn
        self._layout = layout
        self._tables = [(provider, provider)]   # (alias, provider)
        self._doc_columns = {}
        self._filters = {}
        self._label_filters = {}     # alias -> [(name, labels, selectivity)]
        self._joins = []
        self._pending_join = None    # (new aliases, conditions, applies)
        #                              awaiting the AI predicate over
        #                              its pairs
        self._applies = {}           # alias -> [(name, kind, ids, refs)]
        self._functions = {}         # name -> the Python function
        self._labels = {}            # name -> Alias of an AI.CLASSIFY call
        self._scores = {}            # name -> Alias of an AI.SCORE call
        self._column_predicates = {}  # alias -> [ColumnPredicate]
        self._limit = None
        self._order = []             # (column or name, descending, nulls_first)
        self._offset = 0
        self._distinct = False
        self._group = []             # key columns or names
        self._aggs = {}              # name -> AggSpec
        self._having = []            # HavingSpec

    # ---- scope -------------------------------------------------------

    def alias(self, a: str) -> "Query":
        if (self._joins or self._filters or self._label_filters
                or len(self._tables) != 1):
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
        bound = binder(p.template, refs, self._tokenizer, turn=self._turn,
                       layout=self._layout)
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
            FilterPredicate(ModelCall(bound), selectivity=selectivity))
        return self

    ai_if = ai_filter

    def ai_score(self, p: PromptSpec, *, name: str) -> "Query":
        """Add a named AI.SCORE column: the model's belief that a prompt is TRUE.

        Args:
            p: A prompt over one or two columns.
            name: The score column's name, used in select(), order_by(),
                group_by() arguments, and agg() arguments.
        """
        if not name or "." in name:
            raise CompileError(
                f"a score needs a column name without a dot, got {name!r}")
        if name in self._labels or name in self._scores or name in self._scope():
            raise CompileError(f"the name {name!r} is already used")
        refs = tuple(self._resolve(c) for c in p.cols)
        bound = bind_score_prompt(p.template, refs, self._tokenizer,
                                  turn=self._turn)
        for ref in refs:
            self._note_doc_column(ref)
        call = ModelCall(bound, "score")
        call.validate()
        self._scores[name] = Alias(call, name)
        return self

    def ai_classify(self, p: PromptSpec, labels, *, name: str,
                    descriptions=None,
                    task_description: str = "",
                    probabilities: bool = False) -> "Query":
        """Give each document one of the labels, in a named result column.

        Use select() to return the column and label_in() to keep documents
        with chosen labels. The planner chooses how to score the labels.

        Args:
            p: The prompt over one document column, or over the anchor and
                partner columns of a join's rows.
            labels: Label strings, (label, description) pairs, or the name
                of a registered label table.
            name: Result column name, without a dot.
            descriptions: Optional descriptions in the same order as labels.
                Leave empty when labels holds (label, description) pairs.
            task_description: Additional instructions, up to 50 words.
            probabilities: Whether to also return a map of label probabilities
                in the column named name + "_probabilities".

        Returns:
            This query builder, with the classification added.

        Raises:
            CompileError: The name, labels, descriptions, or prompt are
                invalid, or a pending join has no AI predicate yet.
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "classify before joining")
        if not name or "." in name:
            raise CompileError(
                f"a classification needs a column name without a dot, "
                f"got {name!r}")
        if name in self._labels or name in self._scope():
            raise CompileError(f"the name {name!r} is already used")
        if isinstance(labels, str):
            labels, table_descriptions = read_label_table(self._catalog, labels)
            descriptions = descriptions or table_descriptions
        else:
            labels = tuple(labels)
            if labels and all(isinstance(label, (tuple, list))
                              for label in labels):
                if descriptions:
                    raise CompileError(
                        "give descriptions in the pairs or separately, not both")
                descriptions = tuple(description for _, description in labels)
                labels = tuple(label for label, _ in labels)
        descriptions = tuple(descriptions or ())
        validate_task_description(task_description)
        refs = tuple(self._resolve(c) for c in p.cols)
        bound = bind_classify_prompt(p.template, refs, labels, descriptions,
                                     self._tokenizer, turn=self._turn,
                                     task_description=task_description,
                                     layout=self._layout)
        for ref in refs:
            self._note_doc_column(ref)
        call = ModelCall(bound, "label", labels, descriptions,
                         probabilities=probabilities)
        call.validate()
        self._labels[name] = Alias(call, name)
        return self

    def label_in(self, name: str, labels,
                 selectivity: Optional[float] = None) -> "Query":
        """Keep documents whose label is one of the given labels.

        This uses the same membership check as SQL IN.

        Args:
            name: Column name from an earlier ai_classify() call.
            labels: Labels to keep.
            selectivity: Estimated fraction of documents kept. None uses 0.2.

        Returns:
            This query builder, with the condition added.

        Raises:
            CompileError: No classification has this name, the classification
                refers to document pairs, or labels contains duplicates.
        """
        if name not in self._labels:
            raise CompileError(f"no classification is named {name!r}")
        call = self._labels[name].expression
        if len(call.aliases()) != 1:
            raise CompileError(
                f"a filter on a label tests a one-document classification; "
                f"{name!r} classifies pairs of {call.aliases()}")
        (alias,) = call.aliases()
        labels = tuple(labels)
        InList(ColumnRef(alias, self._scope()[alias], name), labels).validate()
        self._label_filters.setdefault(alias, []).append(
            (name, labels, selectivity))
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
                                    predicate=ModelCall(bound),
                                    selectivity=selectivity,
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
            for name, column in other._labels.items():
                if name in self._labels or name in self._scope():
                    raise CompileError(f"the name {name!r} is already used")
                self._labels[name] = column
            self._tables.append((alias, provider))
            for a, preds in other._filters.items():
                self._filters.setdefault(a, []).extend(preds)
            for a, tests in other._label_filters.items():
                self._label_filters.setdefault(a, []).extend(tests)
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
                                    predicate=ModelCall(bound),
                                    semantics=semantics,
                                    selectivity=selectivity,
                                    anchor=anchor))
        return self

    def where(self, *tests) -> "Query":
        """Keep the documents that pass column tests, before any model call.

        Args:
            tests: Comparisons of a ``col(...)`` with a literal, such as
                ``col("r.year") > 2020``, or ``col(...).isin([...])``,
                ``col(...).is_null()``, or ``col(...).is_not_null()``.
        """
        for test in tests:
            if not isinstance(test, PredicateSpec):
                raise CompileError(
                    f"where() takes column tests such as col('r.year') > "
                    f"2020, got {test!r}")
            ref = self._resolve(test.col)
            predicate = ColumnPredicate(ref, test.comparison, test.value)
            predicate.validate()
            self._column_predicates.setdefault(ref.alias, []).append(predicate)
        return self

    def limit(self, n: int) -> "Query":
        if not isinstance(n, int) or n <= 0:
            raise CompileError("LIMIT must be a positive integer")
        self._limit = n
        return self

    def order_by(self, *keys) -> "Query":
        """Sort the result rows.

        Args:
            keys: Each a column name such as ``"r.year"`` or a projected
                name such as ``"score"``, sorted ascending, or a
                ``col(...)`` with ``.asc()`` or ``.desc()`` and an
                optional ``.nulls_first()`` or ``.nulls_last()``.
        """
        for key in keys:
            if isinstance(key, str):
                key = col(key)
            if isinstance(key, ColSpec):
                key = SortSpec(key)
            if not isinstance(key, SortSpec):
                raise CompileError(
                    "order_by takes a column name or col(...).asc() or "
                    f".desc(), got {key!r}")
            name = (key.col.column if key.col.alias is None
                    else f"{key.col.alias}.{key.col.column}")
            nulls_first = (key.descending if key.nulls is None
                           else key.nulls == "first")
            self._order.append((name, key.descending, nulls_first))
        return self

    def group_by(self, *keys) -> "Query":
        """Group the result rows by columns or by named AI outputs."""
        self._group.extend(keys)
        return self

    def agg(self, **aggregates) -> "Query":
        """Add named aggregates, such as ``n=count()`` or ``s=avg("r.stars")``."""
        for name, spec in aggregates.items():
            if not isinstance(spec, AggSpec):
                raise CompileError(
                    f"agg() takes count(), count_distinct(), sum_(), avg(), "
                    f"min_(), or max_(), got {spec!r} for {name!r}")
            self._aggs[name] = spec
        return self

    def having(self, *tests) -> "Query":
        """Keep the groups whose aggregates pass tests such as ``count() > 2``."""
        for test in tests:
            if not isinstance(test, HavingSpec):
                raise CompileError(
                    f"having() takes tests such as count() > 2, got {test!r}")
            self._having.append(test)
        return self

    def offset(self, n: int) -> "Query":
        if not isinstance(n, int) or n < 0:
            raise CompileError("OFFSET must be a nonnegative integer")
        self._offset = n
        return self

    def distinct(self) -> "Query":
        self._distinct = True
        return self

    def select(self, *cols) -> LogicalPlan:
        if self._pending_join is not None:
            raise CompileError(
                f"join() of {self._pending_join[0]} has no AI predicate "
                f"over its pairs; a plain join belongs in the database "
                f"the ids came from")
        if not (self._joins or self._filters or self._labels or self._scores):
            raise CompileError("the query has no AI predicate; a plain "
                               "scan belongs in the database the ids "
                               "came from")
        columns = []
        aggregation = None
        if self._group or self._aggs or self._having:
            columns, aggregation = self._aggregation(cols)
        for c in () if aggregation is not None else cols:
            if isinstance(c, str) and c in self._labels:
                columns.append(self._labels[c])
                continue
            if isinstance(c, str) and c in self._scores:
                columns.append(self._scores[c])
                continue
            if isinstance(c, str) and c == "*":
                for alias, provider in self._tables:
                    for name in self._catalog.get(provider).columns:
                        columns.append(ColumnRef(alias=alias,
                                                 provider=provider,
                                                 column=name))
                continue
            spec = col(c) if isinstance(c, str) else c
            columns.append(self._resolve(spec))
        # a classification is planned when the query returns or tests
        # its label
        tested = {name for tests in self._label_filters.values()
                  for name, _, _ in tests}
        wanted = [column for column in self._labels.values()
                  if column in columns or column.name in tested]
        logical = LogicalPlanBuilder()
        for alias, provider in self._tables:
            logical.add_scan(
                alias,
                provider,
                self._doc_columns.get(alias, ""),
                tuple(self._filters.get(alias, ())),
                tuple(self._applies.get(alias, ())),
                tuple((column.expression, column.name) for column in wanted
                      if column.expression.aliases() == (alias,)),
                tuple(self._label_filters.get(alias, ())),
                column_predicates=tuple(
                    self._column_predicates.get(alias, ())),
            )
        for join in self._joins:
            logical.add_join(join)
        for column in wanted:
            if len(column.expression.aliases()) == 2:
                logical.add_classify(column.expression, column.name)
        named = {column.name: column for column in columns
                 if isinstance(column, Alias)}
        if aggregation is not None:
            named.update({a.name: a for a in aggregation.aggregates})
        order = tuple(
            SortKey(named[name] if name in named else self._resolve(col(name)),
                    descending=descending, nulls_first=nulls_first)
            for name, descending, nulls_first in self._order)
        return logical.project(tuple(columns), self._limit, order=order,
                               offset=self._offset, distinct=self._distinct,
                               aggregation=aggregation)

    def _aggregation(self, cols) -> tuple:
        """Bind group_by(), agg(), and having() to the selected names.

        Returns:
            The projected columns and the Aggregation.
        """
        columns = []

        def project(c) -> str:
            if isinstance(c, str) and c in self._labels:
                column = self._labels[c]
            elif isinstance(c, str) and c in self._scores:
                column = self._scores[c]
            else:
                column = self._resolve(col(c) if isinstance(c, str) else c)
            if column not in columns:
                columns.append(column)
            return (column.name if isinstance(column, Alias)
                    else f"{column.alias}.{column.column}")

        keys = tuple(project(c) for c in self._group)
        aggregates = []
        for name, spec in self._aggs.items():
            argument = None if spec.col is None else project(spec.col)
            aggregates.append(AggregateCall(spec.function, argument, name))
        tests = []
        for index, test in enumerate(self._having):
            argument = None if test.agg.col is None else project(test.agg.col)
            same = next((a for a in aggregates
                         if (a.function, a.argument)
                         == (test.agg.function, argument)), None)
            if same is None:
                same = AggregateCall(test.agg.function, argument,
                                     f"__having_{index}")
                aggregates.append(same)
            tests.append(HavingTest(same, test.comparison, test.value))
        output = []
        for c in cols:
            if isinstance(c, str) and c in self._aggs:
                output.append(c)
            else:
                output.append(project(c))
        return columns, Aggregation(keys, tuple(aggregates), tuple(output),
                                    tuple(tests))


def docs(catalog: Catalog, provider: str, tokenizer=None,
         turn: tuple[str, str] = ("", ""), layout: str = "ai-if") -> Query:
    return Query(catalog, provider, tokenizer, turn, layout)
