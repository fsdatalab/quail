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


def _schemas_as_text(value):
    """The value with each encoded Arrow schema replaced by its text."""
    import base64

    import pyarrow as pa

    if isinstance(value, dict):
        if value.get("type") == "pyarrow.Schema":
            schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(value["ipc"])))
            return schema.to_string(show_schema_metadata=True)
        return {key: _schemas_as_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_schemas_as_text(item) for item in value]
    return value


def rust_logical_plans(native, sql, catalog, prompts):
    """Bind a query with DataFusion and the Rust binder: (results, whole)."""
    import pyarrow as pa

    def refuse(columns):
        raise RuntimeError("the binder check does not scan tables")

    def run_region(text):
        raise RuntimeError("the binder check does not run regions")

    context = native.NativeContext(run_region)
    for name, provider in catalog.providers.items():
        sink = pa.BufferOutputStream()
        schema = provider.schema()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(schema.empty_table())
        context.register(name, sink.getvalue().to_pybytes(), refuse)
    model = prompts.model
    turn = prompts.turn if model is None else model.turn
    layout = prompts.layout if model is None else model.prompt_layout
    return context.logical_plans(sql, tuple(turn), layout)


def binder_outcome(native, sql, catalog, prompts, plan, message) -> str:
    """One line on how the Rust binder's plan compares with Python's."""
    try:
        results, whole = rust_logical_plans(native, sql, catalog, prompts)
    except Exception as error:  # noqa: BLE001 - every failure is an outcome
        return f"datafusion error: {str(error).splitlines()[0][:200]}"
    if len(results) != 1 or not whole:
        return f"not one whole region: {len(results)} regions, whole={whole}"
    rust_plan, rust_message = results[0]
    if message is not None:
        return ("same refusal" if rust_message == message
                else f"refusal differs: Rust {rust_message!r}, Python {message!r}")
    if rust_message is not None:
        return f"rust refused: {rust_message[:200]}"
    built = _schemas_as_text(json.loads(rust_plan))
    expected = _schemas_as_text(json.loads(json.dumps(encode_value(plan))))
    if built == expected:
        return "same plan"
    return f"plan differs: {first_difference(built, expected)}"


# Queries Python's front end compiles that DataFusion or the Rust binder
# cannot yet, by a phrase of the outcome, with the reason.
KNOWN_GAPS = {
    "table function 'UNNEST' not found":
        "DataFusion's Snowflake dialect has no UNNEST in FROM",
    "a subquery in FROM is outside what Quail binds":
        "sqlglot merges a CTE into one SELECT; the Rust binder reads a table",
    "For SELECT DISTINCT, ORDER BY expressions":
        "DataFusion refuses DISTINCT with ORDER BY a struct field",
}


def check_outcome(outcome: str, accepted: bool) -> None:
    """Raise AssertionError where the Rust binder disagrees with Python's.

    A query Python accepts must bind to the same plan, unless a known
    gap stops it; a query Python refuses must be refused, in any words.
    """
    if outcome in ("same plan", "same refusal"):
        return
    if accepted:
        assert any(phrase in outcome for phrase in KNOWN_GAPS), (
            f"the Rust binder disagrees: {outcome}")
        return
    assert not outcome.startswith("refusal differs: Rust None"), (
        f"the Rust binder accepted a query Python refuses: {outcome}")


def checked_compile_sql(native, original, path=None):
    """Return compile_sql, also binding each query with DataFusion in Rust.

    Args:
        native: The _quail_native module.
        original: Python's compile_sql.
        path: A file to log each query's outcome to, or None.
    """

    def compile_sql(sql, catalog, dialect="snowflake", turn=("", ""),
                    layout="ai-if", prompts=None):
        from quail.frontend.prompts import PromptBinder
        from quail.logical.nodes import WithCTE

        binder = prompts or PromptBinder(turn=turn, layout=layout)
        plan = message = None
        try:
            plan = original(sql, catalog, dialect=dialect, prompts=binder)
        except CompileError as error:
            message = str(error)
            raise
        finally:
            if str(dialect).lower().endswith("snowflake") and not (
                    plan is not None and isinstance(plan.root, WithCTE)):
                outcome = binder_outcome(native, sql, catalog, binder, plan, message)
                if path:
                    with open(path, "a") as log:
                        log.write(json.dumps({"sql": sql, "outcome": outcome}) + "\n")
                check_outcome(outcome, accepted=message is None)
        return plan

    return compile_sql


def shadow_logical_alternatives(native, original):
    """Return logical_alternatives, also applying the rules in Rust.

    Rules other than the built-in ones are Python's alone; a call with
    one is not compared.
    """
    from quail.planner.optimizer.rules import LogicalRule
    from quail.planner.rules import logical as built_in

    classes = {getattr(built_in, name) for name in dir(built_in)
               if name.startswith("Should")}

    def logical_alternatives(plan, rules, context, **options):
        found = original(plan, rules, context, **options)
        used = [rule for rule in rules if isinstance(rule, LogicalRule)]
        if options or any(type(rule) not in classes for rule in used):
            return found
        catalog = context.catalog
        id_columns = (None if catalog is None else {
            name: provider.id_col for name, provider in catalog.providers.items()
            if hasattr(provider, "id_col")})
        built, message = native.logical_alternatives(
            json.dumps(encode_value(plan)), [rule.name for rule in used], id_columns)
        assert message is None, f"the Rust rules refused: {message}"
        built = [(json.loads(text), list(names)) for text, names in built]
        expected = [(json.loads(json.dumps(encode_value(alternative.plan))),
                     list(alternative.rules)) for alternative in found]
        assert [names for _, names in built] == [names for _, names in expected], (
            "the Rust rules made other alternatives")
        for (mine, _), (theirs, _) in zip(built, expected):
            difference = first_difference(mine, theirs)
            assert mine == theirs, f"the Rust rules rewrote another plan: {difference}"
        return found

    return logical_alternatives
