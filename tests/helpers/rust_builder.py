"""Build every SELECT the front end binds a second time, in Rust, and compare.

While the Python and Rust planners both exist, the front end's
``place_relation`` and ``LogicalPlanBuilder.project`` calls are
repeated through ``_quail_native.build_select``. A query whose Rust
plan, or whose CompileError message, differs from Python's fails the
test that compiled it.
"""

from __future__ import annotations

import inspect
import json

from quail.common.expressions import And, CompileError
from quail.logical.naive import Exists, Hint, _equalities, _scans
from quail.logical.nodes import Filter
from quail.physical.codec import encode_value

# The CompileError messages the Rust builder matched, in order.
COMPARED_REFUSALS: list[str] = []


def _place_arguments(relation, columns, catalog, unnested, tested,
                     allow_relational) -> dict:
    """The place_relation arguments, with the relation split as it splits it."""
    condition = relation.condition if isinstance(relation, Filter) else None
    scans = _scans(relation.input if isinstance(relation, Filter) else relation)
    parts = (() if condition is None else condition.operands
             if isinstance(condition, And) else (condition,))
    gates = [part.condition if isinstance(part, Hint) else part for part in parts]
    providers = {scan.provider for scan in scans}
    providers.update(gate.relation.provider for gate in gates
                     if isinstance(gate, Exists))
    return {
        "scans": encode_value(scans),
        "equalities": [[alias, encode_value(found)]
                       for alias, found in _equalities(relation).items()],
        "parts": encode_value(parts),
        "columns": encode_value(columns),
        "schemas": {provider: encode_value(catalog.get(provider).schema())
                    for provider in sorted(providers)},
        "unnested": list(unnested),
        "tested": [[encode_value(call), name] for call, name in (tested or {}).items()],
        "allow_relational": allow_relational,
    }


def _check(native, place: dict, project: dict | None, plan, message) -> None:
    """Raise AssertionError when the Rust builder disagrees with Python's."""
    rust_plan, rust_message = native.build_select(
        json.dumps(place), None if project is None else json.dumps(project))
    if message is not None or rust_message is not None:
        assert rust_message == message, (
            f"the Rust builder refused with {rust_message!r}; Python with {message!r}")
        COMPARED_REFUSALS.append(message)
        return
    if plan is None:
        return
    expected = json.loads(json.dumps(encode_value(plan)))
    built = json.loads(rust_plan)
    assert built == expected, (
        f"the Rust builder built another plan: {first_difference(built, expected)}")


def first_difference(built, expected, path="") -> str:
    """The path of the first value that differs, with both values."""
    if isinstance(built, dict) and isinstance(expected, dict):
        for key in sorted(set(built) | set(expected)):
            if built.get(key) != expected.get(key):
                return first_difference(built.get(key), expected.get(key),
                                        f"{path}.{key}")
    if (isinstance(built, list) and isinstance(expected, list)
            and len(built) == len(expected)):
        for index, (mine, theirs) in enumerate(zip(built, expected)):
            if mine != theirs:
                return first_difference(mine, theirs, f"{path}[{index}]")
    mine, theirs = json.dumps(built)[:300], json.dumps(expected)[:300]
    return f"{path}: Rust {mine}, Python {theirs}"


def shadow_place_relation(native, original):
    """Return place_relation, also building each plan in Rust."""

    def place_relation(relation, columns, catalog, unnested=(), *, tested=None,
                       allow_relational=False):
        place = _place_arguments(relation, columns, catalog, unnested, tested,
                                 allow_relational)
        try:
            placed = original(relation, columns, catalog, unnested, tested=tested,
                              allow_relational=allow_relational)
        except CompileError as error:
            _check(native, place, None, None, str(error))
            raise
        _check(native, place, None, None, None)
        project = placed.builder.project
        signature = inspect.signature(project)

        def recorded_project(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {name: encode_value(value)
                         for name, value in bound.arguments.items()
                         if name != "row_tests"}
            try:
                plan = project(*args, **kwargs)
            except CompileError as error:
                _check(native, place, arguments, None, str(error))
                raise
            _check(native, place, arguments, plan, None)
            return plan

        placed.builder.project = recorded_project
        return placed

    return place_relation


def shadow_prompt_binder(native, original):
    """Return PromptBinder.bind, also binding each prompt in Rust."""

    def bind(self, kind, template, args, *, labels=(), descriptions=(),
             task_description=""):
        turn = self.turn if self.model is None else self.model.turn
        layout = self.layout if self.model is None else self.model.prompt_layout
        rust_prompt, rust_message = native.bind_prompt(
            kind, template, json.dumps(encode_value(tuple(args))), list(labels),
            list(descriptions), task_description, tuple(turn), layout)
        try:
            prompt = original(self, kind, template, args, labels=labels,
                              descriptions=descriptions,
                              task_description=task_description)
        except CompileError as error:
            assert rust_message == str(error), (
                f"Rust bound the prompt with {rust_message!r}; Python refused with "
                f"{error}")
            raise
        assert rust_message is None, f"Rust refused the prompt: {rust_message}"
        built = json.loads(rust_prompt)
        expected = json.loads(json.dumps(encode_value(prompt)))
        assert built == expected, (
            f"Rust bound another prompt: {first_difference(built, expected)}")
        return prompt

    return bind
