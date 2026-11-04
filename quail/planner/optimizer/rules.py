"""The optimizer rules a session's registry lists, by their Rust names.

The Rust planner in ``_quail_native`` implements every rule
(``crates/quail-planner/src/rules.rs`` for the logical rules and
``physical_rules.rs`` for the physical ones). The registry lists the
rules a session applies and their order; a rule is added by writing it
in Rust and registering its name.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LogicalRule:
    """A rule that rewrites a logical plan or proposes alternatives to it."""

    name: str
    kind: ClassVar[str] = "logical"


@dataclass(frozen=True)
class PhysicalRule:
    """A rule that rewrites a physical graph and its settings."""

    name: str
    kind: ClassVar[str] = "physical"
