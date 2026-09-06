"""Order AI_FILTER predicates by selectivities observed in earlier runs.

The planner orders a filter chain by cost, which needs each predicate's
selectivity: the fraction of documents that pass. A predicate written
without one gets a default. This rule fills the missing selectivities
from a table keyed by prompt template, so a query written today is
ordered by what the same prompts did yesterday.

The table comes from a JSON file named by QUAIL_SELECTIVITY_HINTS,
a mapping from prompt template to selectivity:

    {"Does this review mention the acting? {0}": 0.31,
     "Is the reviewer recommending the film? {0}": 0.58}

Without the variable the rule registers with an empty table and
changes nothing. Load it like any extension:

    registry = quail.ExtensionRegistry.with_built_ins()
    registry.load_extension("quail_ext_examples.selectivity_hints",
                            local_python_sources=("quail_ext_examples",))
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Mapping

from quail.logical import SemanticFilter, split_frame

ENV_VAR = "QUAIL_SELECTIVITY_HINTS"


def canonical(template: str) -> str:
    """Return the template as bound prompts store it.

    The front end moves any text written before the first placeholder
    after it, so a hint written as the user wrote the prompt still
    matches the bound predicate.
    """
    return split_frame(template)[1]


class SelectivityHints:
    """Fill missing filter selectivities from observed values."""

    name = "example.selectivity_hints"

    def __init__(self, hints: Mapping[str, float]):
        self.hints = {}
        for template, value in hints.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"selectivity for {template!r} must be in [0, 1], "
                    f"got {value!r}")
            self.hints[canonical(template)] = float(value)

    def rewrite(self, node, context):
        if not isinstance(node, SemanticFilter) or not self.hints:
            return None
        predicates = tuple(
            replace(predicate,
                    selectivity=self.hints[predicate.prompt.template])
            if predicate.selectivity is None
            and predicate.prompt.template in self.hints
            else predicate
            for predicate in node.predicates
        )
        if predicates == node.predicates:
            return None
        return replace(node, predicates=predicates)


def load_hints(path: str | None = None) -> dict[str, float]:
    """Read the hint table from a JSON file, or return an empty one."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        return {}
    with open(path) as source:
        return {str(key): float(value) for key, value in json.load(source).items()}


def register_quail_extension(registry) -> None:
    """Register the rule with the hints named by QUAIL_SELECTIVITY_HINTS."""
    registry.register_logical_rule(
        SelectivityHints.name, SelectivityHints(load_hints()))
