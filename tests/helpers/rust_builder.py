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
import os

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


# The prompt_lowering functions repeated in Rust, by name.
LOWERING_FUNCTIONS = ("lower_prompt", "score_token_parts", "answer_ids",
                      "option_spellings", "read_slot_offsets",
                      "classify_prompt_tokens", "choice_token_parts")


def _lowered_fields(lowered, intern):
    """A LoweredPrompt as the Rust type writes it, its tokens interned."""
    return {
        "prompt": json.loads(json.dumps(encode_value(lowered.prompt))),
        "preamble_token_ids": [intern(t) for t in lowered.preamble_token_ids],
        "tail_token_ids": [intern(t) for t in lowered.tail_token_ids],
        "frame_tokens": lowered.frame_tokens,
        "label_token_ids": [
            [alias, [intern(t) for t in label], [intern(t) for t in frame]]
            for alias, label, frame in lowered.label_token_ids],
        "lettered": (None if lowered.lettered is None
                     else _lowered_fields(lowered.lettered, intern)),
    }


def _model_fields(model) -> dict:
    return {"role": model.role,
            "answer_spellings": [list(words) for words in model.answer_spellings],
            "turn": list(model.turn), "prompt_layout": model.prompt_layout}


def _interned(value, intern):
    """Every token id in a nested result, interned."""
    if isinstance(value, (list, tuple)):
        return [_interned(item, intern) for item in value]
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return intern(value)
    return value


def shadow_lowering(native, name, original):
    """Return a prompt_lowering function that also runs in Rust.

    The tokenizer's tokens are interned to integers on both sides, so a
    test tokenizer that returns words compares as one returning ids.
    """
    from quail.common.expressions import CompileError as Refused

    def lowering(*args, **kwargs):
        if name == "lower_prompt":
            prompt, tokenizer = args
            arguments = [encode_value(prompt)]
        elif name == "score_token_parts":
            prompt, model, tokenizer = args
            arguments = [encode_value(prompt), _model_fields(model)]
        elif name == "answer_ids":
            model, tokenizer = args
            arguments = [_model_fields(model)]
        elif name in ("option_spellings",):
            codes, tokenizer = args
            arguments = [list(codes)]
        elif name == "read_slot_offsets":
            segment, names, tokenizer = args
            arguments = [segment, list(names)]
        else:
            lowered, *rest = args
            tokenizer = rest[-1]
            arguments = None
        if tokenizer is None:
            return original(*args, **kwargs)
        table = {}

        def intern(token):
            # an integer id passes through; a test tokenizer's word gets
            # an id past every real one
            if isinstance(token, int) and not isinstance(token, bool):
                return token
            return table.setdefault(token, 1 << 40 | len(table))

        def tokens(text):
            return [intern(token) for token in tokenizer(text)]

        if arguments is None:
            arguments = [_lowered_fields(lowered, intern), *(
                [list(rest[0])] if name == "classify_prompt_tokens" else [])]
        rust, message = native.prompt_lowering(name, json.dumps(arguments), tokens)
        if os.environ.get("QUAIL_RUST_LOWERING_LOG"):
            with open(os.environ["QUAIL_RUST_LOWERING_LOG"], "a") as log:
                log.write(f"{name} {getattr(args[0], 'kind', '')}\n")
        try:
            found = original(*args, **kwargs)
        except (Refused, ValueError) as error:
            assert message is not None, f"Rust {name} did not refuse: {error}"
            raise
        assert message is None, f"Rust {name} refused: {message}"
        expected = (_lowered_fields(found, intern) if name == "lower_prompt"
                    else _interned(found, intern))
        built = json.loads(rust)
        if name == "lower_prompt":
            assert built == expected, (
                f"Rust lowered another prompt: {first_difference(built, expected)}")
        elif name == "answer_ids":
            # each side is sorted by token id, and a word's id is made up
            assert [set(side) for side in built] == [set(side) for side in expected], (
                f"Rust answer_ids differs: {built} != {expected}")
        else:
            assert _interned(built, lambda token: token) == expected, (
                f"Rust {name} differs: {built} != {expected}")
        return found

    return lowering


def _untagged(value):
    """Encoded data without the codec's class tags, as the Rust specs read it."""
    if isinstance(value, dict):
        return {key: _untagged(item) for key, item in value.items() if key != "type"}
    if isinstance(value, list):
        return [_untagged(item) for item in value]
    return value


def same_values(built, expected, path="") -> str | None:
    """The path of the first difference, floats to 1e-12 relative; None if none."""
    if isinstance(built, float) or isinstance(expected, float):
        both = (built, expected)
        if all(isinstance(value, (int, float)) for value in both):
            scale = max(abs(built), abs(expected))
            if abs(built - expected) <= 1e-12 * scale:
                return None
        return f"{path}: Rust {built!r}, Python {expected!r}"
    if isinstance(built, dict) and isinstance(expected, dict):
        if built.keys() != expected.keys():
            return f"{path}: keys {sorted(built)} != {sorted(expected)}"
        for key in built:
            found = same_values(built[key], expected[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(built, list) and isinstance(expected, list):
        if len(built) != len(expected):
            return f"{path}: {len(built)} items != {len(expected)}"
        for index, (mine, theirs) in enumerate(zip(built, expected)):
            found = same_values(mine, theirs, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if built == expected else f"{path}: Rust {built!r}, Python {expected!r}"


def _interning(tokenizer):
    """(intern, tokens): a token's id, and the tokenizer with ids interned."""
    table = {}

    def intern(token):
        if isinstance(token, int) and not isinstance(token, bool):
            return token
        return table.setdefault(token, 1 << 40 | len(table))

    def tokens(text):
        return [intern(token) for token in tokenizer(text)]

    return intern, tokens


def _token_fields(spec: dict, intern) -> dict:
    """A spec's token id fields with their ids interned."""
    out = dict(spec)
    for key in ("prompt_token_parts", "label_token_ids", "question_header_token_ids"):
        if key in out and out[key] is not None:
            out[key] = _interned(out[key], intern)
    return out


def _context_json(context) -> str:
    """The planning context's fields, as the native module reads them."""
    import numpy as np

    return json.dumps({
        "model": _untagged(encode_value(context.model)),
        "device": _untagged(encode_value(context.device)),
        "gpus": context.gpus, "canvas_draws": context.canvas_draws,
        "document_tokens": {alias: np.asarray(values).astype(int).tolist()
                            for alias, values in context.document_tokens.items()},
        "plan": (None if context.logical_plan is None
                 else encode_value(context.logical_plan)),
    })


def _plan_operator(native, name, context, arguments):
    """(intern, the Rust result or None, its refusal or None)."""
    intern, tokens = _interning(context.tokenizer)
    if os.environ.get("QUAIL_RUST_OPERATOR_LOG"):
        with open(os.environ["QUAIL_RUST_OPERATOR_LOG"], "a") as log:
            log.write(f"{name} {context.model.attention_mask}\n")
    rust, message = native.plan_operator(
        name, _context_json(context), json.dumps(arguments), tokens)
    return intern, (None if rust is None else json.loads(rust)), message


def _work_fields(work) -> dict:
    from dataclasses import asdict

    return {key: float(value) for key, value in asdict(work).items()}


def shadow_column_spec(native, original):
    """Return a column planner's spec method, also building the spec in Rust."""
    from quail.planner.validation import SaeFeaturesRefusedError

    def spec(self, context, node, expected_rows, resident=False):
        if context.tokenizer is None:
            return original(self, context, node, expected_rows, resident)
        intern, rust, message = _plan_operator(native, "column_spec", context, {
            "node": encode_value(node), "expected_rows": float(expected_rows),
            "resident": resident})
        try:
            found = original(self, context, node, expected_rows, resident)
        except SaeFeaturesRefusedError:
            assert message is not None, "Rust built a spec Python refused"
            raise
        assert message is None, f"Rust refused the spec: {message}"
        expected = _token_fields(json.loads(json.dumps(encode_value(found))), intern)
        difference = same_values(rust, expected)
        assert difference is None, f"Rust built another spec: {difference}"
        return found

    return spec


def shadow_score_spec(native, original):
    """Return score_spec, also building the spec and its work in Rust."""

    def score_spec(prompt, *, name, expected_rows, mean_tokens, context,
                   chunk_tokens, pair_fraction=1.0, prefix_groups=None):
        found = original(prompt, name=name, expected_rows=expected_rows,
                         mean_tokens=mean_tokens, context=context,
                         chunk_tokens=chunk_tokens, pair_fraction=pair_fraction,
                         prefix_groups=prefix_groups)
        if context.tokenizer is None:
            return found
        intern, rust, message = _plan_operator(native, "score_spec", context, {
            "prompt": encode_value(prompt), "name": name,
            "expected_rows": float(expected_rows),
            "mean_tokens": None if mean_tokens is None else float(mean_tokens),
            "chunk": chunk_tokens, "pair_fraction": float(pair_fraction),
            "prefix_groups": None if prefix_groups is None else float(prefix_groups)})
        assert message is None, f"Rust refused the score: {message}"
        spec, work = found
        expected = [_token_fields(json.loads(json.dumps(encode_value(spec))), intern),
                    _work_fields(work)]
        difference = same_values(rust, expected)
        assert difference is None, f"Rust built another score: {difference}"
        return found

    return score_spec


# the document counts filter and score costs are compared at
SAMPLE_ROWS = (0.0, 1.0, 37.5, 4000.0)


def shadow_filter_costs(native, original):
    """Return prepare_filter_costs, also pricing the predicates in Rust."""

    def prepare_filter_costs(predicates, *, mean_tokens, model, device,
                             chunk_tokens, context=None, lower=None):
        found = original(predicates, mean_tokens=mean_tokens, model=model,
                         device=device, chunk_tokens=chunk_tokens,
                         context=context, lower=lower)
        if (context is None or lower is not None or context.tokenizer is None
                or model is not context.model or device is not context.device):
            return found
        _, rust, message = _plan_operator(native, "filter_costs", context, {
            "predicates": encode_value(tuple(predicates)),
            "mean_tokens": float(mean_tokens), "chunk": chunk_tokens,
            "rows": list(SAMPLE_ROWS)})
        assert message is None, f"Rust refused the filter costs: {message}"
        expected = [{
            "selectivity": cost.selectivity,
            "reuse_key": (None if cost.reuse_key is None
                          else json.loads(json.dumps(encode_value(cost.reuse_key)))),
            "first": [float(cost.first(rows)) for rows in SAMPLE_ROWS],
            "later": [float(cost.later(rows)) for rows in SAMPLE_ROWS],
        } for cost in found]
        difference = same_values(rust, expected)
        assert difference is None, f"Rust priced the filters otherwise: {difference}"
        return found

    return prepare_filter_costs


def shadow_join_specs(native, original):
    """Return join_specs, also pricing the joins in Rust."""
    import numpy as np

    def join_specs(joins, pair_fractions=None, canvas=0, *, context=None, chunk=0):
        found = original(joins, pair_fractions, canvas, context=context, chunk=chunk)
        if context is None or context.tokenizer is None:
            return found
        # a score join's work, compared at half of every table's documents
        counts = {alias: len(np.asarray(values)) / 2
                  for alias, values in context.document_tokens.items()}
        _, rust, message = _plan_operator(native, "join_specs", context, {
            "joins": [_untagged_top(encode_value(join)) for join in joins],
            "pair_fractions": {str(key): float(value)
                               for key, value in (pair_fractions or {}).items()},
            "canvas": canvas, "chunk": chunk, "counts": counts})
        assert message is None, f"Rust refused the join specs: {message}"
        expected = [{
            "written_position": spec.written_position, "aliases": list(spec.aliases),
            "anchor": spec.anchor, "anchor_free": spec.anchor_free,
            "semantics": spec.semantics, "selectivity": spec.selectivity,
            "frame_tokens": dict(spec.frame_tokens),
            "label_tokens": dict(spec.label_tokens), "tail_tokens": spec.tail_tokens,
            "on": [list(key) for key in spec.on], "pair_fraction": spec.pair_fraction,
            "work": None if spec.cost is None else _work_fields(spec.cost.work(counts)),
        } for spec in found]
        difference = same_values(rust, expected)
        assert difference is None, f"Rust priced the joins otherwise: {difference}"
        return found

    return join_specs


def _untagged_top(value: dict) -> dict:
    """An encoded dataclass without its own class tag."""
    return {key: item for key, item in value.items() if key != "type"}
