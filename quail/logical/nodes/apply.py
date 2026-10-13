"""The Apply node for user functions between operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
)
from quail.logical.nodes.base import LogicalNode

APPLY_KINDS = ("per_batch", "barrier")


APPLY_IDS = ("preserve", "drop", "pairs")


@dataclass(frozen=True)
class Apply:
    """A user function between two operators.

    The function is registered on the session under ``function`` and
    receives an Arrow table per alias: the alias's row indices under
    the alias name plus the listed columns. ``kind`` says how it runs:
    ``per_batch`` on each batch of survivors a streaming operator hands
    over, ``barrier`` once over every survivor. ``ids`` says what it
    returns: ``preserve`` every input id, ``drop`` a subset of them,
    ``pairs`` a relation of (left alias, right alias) rows for the join
    at ``written_pos``. A function never invents an id.
    """

    input: LogicalNode
    function: str
    kind: str
    ids: str
    columns: tuple  # tuple[ColumnRef, ...]
    aliases: tuple  # the alias, or the join's two aliases
    written_pos: Optional[int] = None

    type_name: ClassVar[str] = "quail.apply"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.kind not in APPLY_KINDS:
            raise CompileError(
                f"apply kind must be one of {APPLY_KINDS}, got {self.kind!r}"
            )
        if self.ids not in APPLY_IDS:
            raise CompileError(
                f"apply ids must be one of {APPLY_IDS}, got {self.ids!r}"
            )
        if not self.function:
            raise CompileError("apply needs a registered function name")
        present = {field.alias for field in self.input.output_schema()}
        if not set(self.aliases) <= present:
            raise CompileError(
                f"apply {self.function!r} names tables {self.aliases} "
                f"outside its input ({sorted(present)})"
            )
        for ref in self.columns:
            if ref.alias not in self.aliases:
                raise CompileError(
                    f"apply {self.function!r} reads {ref.alias}.{ref.column} "
                    f"but works on {self.aliases}"
                )
        if (self.ids == "pairs") != (len(self.aliases) == 2):
            raise CompileError(
                "an apply returning pairs works on exactly two tables; "
                "one returning ids works on one"
            )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Apply needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        return replace(self, columns=tuple(expressions))

    def explain_fields(self) -> dict:
        return {
            "function": self.function,
            "kind": self.kind,
            "ids": self.ids,
            "columns": [f"{ref.alias}.{ref.column}" for ref in self.columns],
        }
