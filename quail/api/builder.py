"""The builder API bound to a session, so select() returns a Query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quail.api.query import Query

if TYPE_CHECKING:
    from quail.api.session import Session


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

    ai_if = ai_filter

    def ai_score(self, p, *, name):
        self._inner.ai_score(p, name=name)
        return self

    def ai_classify(self, p, labels, *, name, descriptions=None,
                    task_description="", probabilities=False):
        self._inner.ai_classify(p, labels, name=name,
                                descriptions=descriptions,
                                task_description=task_description,
                                probabilities=probabilities)
        return self

    def ai_extract(self, c, question, *, name):
        self._inner.ai_extract(c, question, name=name)
        return self

    def ai_sae_features(self, p, *, pooling, name):
        self._inner.ai_sae_features(p, pooling=pooling, name=name)
        return self

    def unnest(self, *names, outer=False):
        self._inner.unnest(*names, outer=outer)
        return self

    def ai_map(self, c, instruction, *, name):
        self._inner.ai_map(c, instruction, name=name)
        return self

    def label_in(self, name, labels, selectivity=None):
        self._inner.label_in(name, labels, selectivity=selectivity)
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

    def where(self, *tests):
        self._inner.where(*tests)
        return self

    filter = where

    def limit(self, n):
        self._inner.limit(n)
        return self

    def order_by(self, *keys):
        self._inner.order_by(*keys)
        return self

    def group_by(self, *keys):
        self._inner.group_by(*keys)
        return self

    def agg(self, **aggregates):
        self._inner.agg(**aggregates)
        return self

    def having(self, *tests):
        self._inner.having(*tests)
        return self

    def window(self, **calls):
        self._inner.window(**calls)
        return self

    def offset(self, n):
        self._inner.offset(n)
        return self

    def distinct(self):
        self._inner.distinct()
        return self

    def select(self, *cols, order=None) -> "Query":
        return Query(self._session, self._inner.select(*cols),
                     order=order)
