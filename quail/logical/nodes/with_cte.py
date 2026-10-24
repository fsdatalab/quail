"""Named relations scoped to one query."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import CompileError
from quail.logical.nodes.base import LogicalNode


@dataclass(frozen=True)
class WithCTE:
    """Bind CTE definitions before the input that reads them."""

    input: LogicalNode
    ctes: tuple[tuple[str, LogicalNode], ...]

    type_name: ClassVar[str] = "quail.with_cte"

    def children(self) -> tuple:
        return (*(node for _, node in self.ctes), self.input)

    def expressions(self) -> tuple:
        return ()

    def output_schema(self) -> tuple:
        return self.input.output_schema()

    def validate(self) -> None:
        names = [name for name, _ in self.ctes]
        if len(set(names)) != len(names):
            raise CompileError("CTE names must be unique within a query")

    def with_children(self, children: tuple) -> "WithCTE":
        if len(children) != len(self.ctes) + 1:
            raise CompileError("WithCTE needs its definitions and its input")
        return replace(self, input=children[-1], ctes=tuple(
            (name, child) for (name, _), child in zip(self.ctes, children)))

    def with_expressions(self, expressions: tuple) -> "WithCTE":
        if expressions:
            raise CompileError("WithCTE has no expressions")
        return self

    def explain_fields(self) -> dict:
        return {"ctes": tuple(name for name, _ in self.ctes)}
