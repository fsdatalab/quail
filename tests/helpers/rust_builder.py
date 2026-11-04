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
        intern, tokens, _ = _interning(tokenizer)

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
    """Return a token's id, the tokenizer with ids interned, and a decoder.

    The decoder is the tokenizer's decode of a list of interned ids, or None.
    """
    table, words = {}, {}

    def intern(token):
        if isinstance(token, int) and not isinstance(token, bool):
            return token
        if token not in table:
            table[token] = 1 << 40 | len(table)
            words[table[token]] = token
        return table[token]

    def tokens(text):
        return [intern(token) for token in tokenizer(text)]

    decoder = getattr(tokenizer, "decode", None)
    decode = None
    if decoder is not None:
        def decode(ids):
            piece = decoder([words.get(token, token) for token in ids])
            return (piece.decode("utf-8", errors="replace")
                    if isinstance(piece, bytes) else str(piece))

    return intern, tokens, decode


def _token_fields(spec: dict, intern) -> dict:
    """A spec's token id fields with their ids interned."""
    out = dict(spec)
    for key in ("prompt_token_parts", "label_token_ids", "question_header_token_ids",
                "join_layout"):
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
        "attention": context.attention,
    })


def _plan_operator(native, name, context, arguments, tokenizer=None, interning=None):
    """(intern, the Rust result or None, its refusal or None)."""
    intern, tokens, decode = interning or _interning(tokenizer or context.tokenizer)
    if os.environ.get("QUAIL_RUST_OPERATOR_LOG"):
        with open(os.environ["QUAIL_RUST_OPERATOR_LOG"], "a") as log:
            log.write(f"{name} {context.model.attention_mask}\n")
    rust, message = native.plan_operator(
        name, _context_json(context), json.dumps(arguments), tokens, decode)
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


def _table_fields(table) -> dict:
    """A ClassifyStatistics' fields, as the Rust ClassifyTable reads them."""
    import numpy as np

    return {
        "alias": table.alias, "mean": float(table.mean), "longest": int(table.longest),
        "budget": int(table.budget), "chunk": int(table.chunk),
        "model": _untagged(encode_value(table.model)),
        "device": _untagged(encode_value(table.device)),
        "capacity": int(table.capacity), "draws": int(table.draws),
        "lengths": np.asarray(table.lengths, dtype=float).tolist(),
        "shared": [float(value) for value in table.shared],
        "tree": bool(table.tree), "gpus": int(table.gpus),
    }


class _TableContext:
    """The context fields a classification table's calls do not read."""

    def __init__(self, table):
        self.model, self.device, self.gpus = table.model, table.device, table.gpus
        self.canvas_draws, self.document_tokens = 1, {}
        self.logical_plan, self.attention = None, None
        self.tokenizer = table.tokenizer


def _spec_json(spec, intern) -> dict:
    return _token_fields(json.loads(json.dumps(encode_value(spec))), intern)


def shadow_classify(native, planner, scoring, builder):
    """Patch the classification planner's methods to also run in Rust.

    Args:
        native: The _quail_native module.
        planner: The ClassifyPlanner class.
        scoring: The ClassifyScoring class.
        builder: The ClassifyBuilder class.

    Returns:
        (attribute owner, name, replacement) for each method.
    """
    from quail.planner.validation import ClassifyRefusedError

    def compare(name, context, arguments, call, expected_of, interning=None):
        intern, rust, message = _plan_operator(native, name, context, arguments,
                                               interning=interning)
        try:
            found = call()
        except ClassifyRefusedError as error:
            assert message == str(error), (
                f"Rust {name} refused with {message!r}, Python with {error}")
            raise
        assert message is None, f"Rust {name} refused: {message}"
        difference = same_values(rust, expected_of(found, intern))
        assert difference is None, f"Rust {name} differs: {difference}"
        return found

    original_spec, original_seconds = planner.spec, planner.seconds
    original_choose, original_joined = scoring.choose_scoring, builder.classify_joined

    def spec(self, context, node, expected_rows, resident=False):
        if context.tokenizer is None:
            return original_spec(self, context, node, expected_rows, resident)
        return compare(
            "classify_spec", context,
            {"node": encode_value(node), "expected_rows": float(expected_rows)},
            lambda: original_spec(self, context, node, expected_rows, resident),
            _spec_json)

    def seconds(self, context, node, resident, expected_rows):
        if context.tokenizer is None:
            return original_seconds(self, context, node, resident, expected_rows)
        intern, rust, message = _plan_operator(
            native, "classify_seconds", context,
            {"node": encode_value(node), "expected_rows": float(expected_rows),
             "resident": resident})
        found = original_seconds(self, context, node, resident, expected_rows)
        assert message is None, f"Rust classify_seconds refused: {message}"
        rust = float("inf") if rust is None else rust
        assert same_values(rust, float(found)) is None or rust == found, (
            f"Rust classify_seconds {rust} != Python {found}")
        return found

    def choose_scoring(self, spec, call, resident):
        context = _TableContext(self)
        interning = _interning(self.tokenizer)
        return compare(
            "choose_scoring", context,
            {"table": _table_fields(self), "spec": _spec_json(spec, interning[0]),
             "call": encode_value(call), "resident": resident},
            lambda: original_choose(self, spec, call, resident), _spec_json,
            interning)

    def classify_joined(self, call, name, partner, expected_rows, partner_tokens,
                        anchors=None):
        context = _TableContext(self)
        return compare(
            "classify_joined", context,
            {"table": _table_fields(self), "call": encode_value(call), "name": name,
             "partner": partner, "expected_rows": float(expected_rows),
             "partner_tokens": float(partner_tokens),
             "anchors": None if anchors is None else float(anchors)},
            lambda: original_joined(self, call, name, partner, expected_rows,
                                    partner_tokens, anchors),
            lambda found, intern: [_spec_json(found[0], intern),
                                   _work_fields(found[1])])

    return ((planner, "spec", spec), (planner, "seconds", seconds),
            (scoring, "choose_scoring", choose_scoring),
            (builder, "classify_joined", classify_joined))


def _operator_order_fields(found, intern) -> dict:
    """An OperatorOrder, or its Refusal, as the Rust binding writes it."""
    from quail.planner.plan import Refusal

    if isinstance(found, Refusal):
        return {"refusal": {"reasons": list(found.reasons),
                            "constraint": found.constraint,
                            "needed": float(found.needed),
                            "available": float(found.available), "unit": found.unit}}
    statistics = found.statistics
    return {
        "statistics": {
            "stats": {alias: [s.num_documents, s.total_tokens, s.max_document_tokens]
                      for alias, s in statistics.stats.items()},
            "lengths": {alias: [s.count, s.total, str(s.squared), s.maximum, s.window,
                                [list(pair) for pair in s.short_counts]]
                        for alias, s in statistics.lengths.items()},
            "chunk": statistics.chunk, "arena_pages": list(statistics.arena_pages),
            "admission": statistics.admission, "cap_pages": statistics.cap_pages,
            "pre": statistics.pre,
            "asks": {alias: list(positions)
                     for alias, positions in statistics.asks.items()},
            "specs": [spec.written_position for spec in statistics.specs],
        },
        "filter_orders": {alias: list(positions)
                          for alias, positions in found.filter_orders.items()},
        "step_orders": {alias: [{
            "kind": step.kind, "key": step.key, "selectivity": step.selectivity,
            "reuse_key": step.reuse_key is not None,
            "document_head": [intern(token) for token in step.document_head],
        } for step in steps] for alias, steps in found.step_orders.items()},
        "join_sequence": [[spec.written_position, anchor]
                          for spec, anchor in found.join_sequence],
        "join_steps": [{
            "written_position": step.written_position, "anchor": step.anchor,
            "resident": step.resident, "resident_documents": step.resident_documents,
            "anchor_documents": float(step.anchor_documents),
            "tuples": float(step.tuples), "tokens": float(step.tokens),
            "work": _work_fields(step.work),
        } for step in found.join_steps],
        "expected_rows": {alias: float(rows)
                          for alias, rows in found.expected_rows.items()},
        "filter_work": {alias: _work_fields(work)
                        for alias, work in found.filter_work.items()},
        "order_rule": found.order_rule, "order_source": found.order_source,
    }


def shadow_operator_order(native, original):
    """Return choose_operator_order, also searching in Rust."""
    import numpy as np

    def choose_operator_order(plan, operators, model, device, document_tokens, order,
                              pair_fractions, context, scan_fractions):
        found = original(plan, operators, model, device, document_tokens, order,
                         pair_fractions, context, scan_fractions)
        if (context is None or context.tokenizer is None
                or model is not context.model or device is not context.device):
            return found
        fields = json.loads(_context_json(context))
        fields.update(
            document_tokens={alias: np.asarray(values).astype(int).tolist()
                             for alias, values in document_tokens.items()},
            order=order,
            pair_fractions={str(key): float(value)
                            for key, value in (pair_fractions or {}).items()},
            scan_fractions={alias: float(value)
                            for alias, value in (scan_fractions or {}).items()})
        intern, tokens, decode = _interning(context.tokenizer)
        if os.environ.get("QUAIL_RUST_OPERATOR_LOG"):
            joins = getattr(found, "join_sequence", ())
            with open(os.environ["QUAIL_RUST_OPERATOR_LOG"], "a") as log:
                log.write(f"operator_order {len(joins)} joins "
                          f"{type(found).__name__}\n")
        rust, message = native.plan_operator(
            "operator_order", json.dumps(fields),
            json.dumps({"plan": encode_value(plan)}), tokens, decode)
        assert message is None, f"Rust refused the operator order: {message}"
        difference = same_values(json.loads(rust),
                                 _operator_order_fields(found, intern))
        assert difference is None, f"Rust ordered the operators otherwise: {difference}"
        return found

    return choose_operator_order


# node attribute and spec fields that hold token ids
_TOKEN_FIELDS = ("question_token_ids", "document_head", "question_header_token_ids",
                 "frame_token_ids", "tail_token_ids", "prompt_token_parts",
                 "join_layout")


def _interned_tokens(value, intern):
    """A node's or spec's fields with their token ids interned."""
    if isinstance(value, list):
        return [_interned_tokens(item, intern) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key in _TOKEN_FIELDS and item is not None:
            out[key] = _interned(item, intern)
        elif key == "label_token_ids" and item:
            # a spec's are id lists; an AiJoin's are (alias, ids) pairs
            out[key] = (_interned(item, intern) if "type" in value
                        else [[alias, _interned(ids, intern)] for alias, ids in item])
        else:
            out[key] = _interned_tokens(item, intern)
    return out


def _first_documents(context) -> dict:
    """Map each alias with copies to whether each document is its text's first."""
    import numpy as np

    found = {}
    for alias, lengths in context.document_tokens.items():
        store = getattr(lengths, "store", None)
        if store is None:
            continue
        first = store.first_documents.to_numpy()
        keys = first == np.arange(len(first))
        if not keys.all():
            found[alias] = keys.tolist()
    return found


def _candidate_fields(candidate, intern) -> dict:
    """A plan_region candidate as the Rust binding writes it."""
    from quail.physical.codec import built_in_codecs
    from quail.planner.plan import Refusal

    plan = candidate.plan
    if isinstance(plan, Refusal):
        return {"refusal": {"reasons": list(plan.reasons),
                            "constraint": plan.constraint,
                            "needed": float(plan.needed),
                            "available": float(plan.available), "unit": plan.unit}}
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    envelope = json.loads(json.dumps(plan.to_envelope(codecs)))
    envelope["graph"] = _interned_tokens(envelope["graph"], intern)
    for key in ("true_ids", "false_ids"):
        if key in envelope["settings"]:
            envelope["settings"][key] = _interned(envelope["settings"][key], intern)
    return {"envelope": envelope, "estimated_seconds": float(plan.estimated_seconds),
            "estimates": json.loads(json.dumps(dict(plan.estimates)))}


def shadow_plan_region(native, original):
    """Return plan_region, also building the plan in Rust."""
    import numpy as np

    def plan_region(region, context):
        if context.tokenizer is None or context.backend != "quail":
            return original(region, context)
        fields = json.loads(_context_json(context))
        fields.update(
            order=context.order,
            pair_fractions={str(key): float(value)
                            for key, value in (context.pair_fractions or {}).items()},
            scan_fractions={alias: float(value)
                            for alias, value in (context.scan_fractions or {}).items()},
            document_tokens={alias: np.asarray(values).astype(int).tolist()
                             for alias, values in context.document_tokens.items()},
            first_documents=_first_documents(context))
        intern, tokens, decode = _interning(context.tokenizer)
        rust, message = native.plan_operator(
            "plan_region", json.dumps(fields),
            json.dumps({"plan": encode_value(region.logical_plan)}), tokens, decode)
        try:
            found = original(region, context)
        except Exception:
            assert message is not None, "Rust planned a region Python could not"
            raise
        assert message is None, f"Rust refused the region: {message}"
        built, expected = json.loads(rust), _candidate_fields(found[0], intern)
        # the answer ids are a set, sorted by word in Python and by id in Rust
        for plan in (built, expected):
            for key in ("true_ids", "false_ids"):
                if key in plan.get("envelope", {}).get("settings", {}):
                    plan["envelope"]["settings"][key].sort()
        difference = same_values(built, expected)
        if os.environ.get("QUAIL_RUST_OPERATOR_LOG"):
            with open(os.environ["QUAIL_RUST_OPERATOR_LOG"], "a") as log:
                log.write(f"plan_region {type(found[0].plan).__name__} "
                          f"{'same' if difference is None else 'differs'} "
                          f"{context.model.attention_mask} gpus={context.gpus} "
                          f"copies={bool(fields['first_documents'])}\n")
        assert difference is None, f"Rust built another plan: {difference}"
        return found

    return plan_region


def _sharing_fields(context) -> dict:
    """Each table store's page sharing, reader groups, and last tokens."""
    import pyarrow as pa

    from quail.common.document_ends import last_tokens
    from quail.planner.statistics.prefixes import page_tree

    page_shared, reader_groups, last = {}, {}, {}
    for alias, lengths in context.document_tokens.items():
        store = getattr(lengths, "store", None)
        if store is None:
            continue
        tree = page_tree(store, context.device.page_tokens)
        page_shared[alias] = [int(value) for value in tree.shared]
        reader_groups[alias] = [[int(value) for value in group]
                                for group in zip(*tree.reader_groups(lengths))]
        try:
            ends = last_tokens(store.arrow_array)
            last[alias] = sorted({int(token) for token in ends})
        except (TypeError, ValueError, pa.ArrowInvalid):
            # a test store of words has no integer last tokens
            pass
    return {"page_shared": page_shared, "reader_groups": reader_groups,
            "last_tokens": last}


def _document_ends_sorted(value):
    """A plan with each node's document ends in token order."""
    if isinstance(value, list):
        return [_document_ends_sorted(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: _document_ends_sorted(item) for key, item in value.items()}
    if out.get("document_ends"):
        out["document_ends"] = [[alias, tail, sorted(joins, key=repr)]
                                for alias, tail, joins in out["document_ends"]]
    return out


def shadow_search_plan(native, original):
    """Return search_plan, also selecting the plan in Rust.

    The shadow runs only for the Quail backend with the built-in rules
    and no other physical planner, which is every plan the suite makes
    unless a test registers its own.
    """
    import numpy as np

    from quail.planner.optimizer.tree_rewrites import logical_alternatives
    from quail.planner.rules import built_in_rules

    built_in = [rule.name for rule in built_in_rules()]
    # a test that patches a rule's rewrite plans with a rule Rust lacks
    methods = {rule.name: (getattr(type(rule), "rewrite", None),
                           getattr(type(rule), "rewrites", None))
               for rule in built_in_rules()}

    def search_plan(logical, context, registry, *, logical_candidates=None):
        found = original(logical, context, registry,
                         logical_candidates=logical_candidates)
        logical_rules = [rule.name for rule in registry.logical_rules.values()]
        rules = [rule.name for rule in registry.physical_rules.values()]
        if (context.tokenizer is None or context.backend != "quail"
                or registry.physical_planners or logical_rules + rules != built_in
                or any((getattr(type(rule), "rewrite", None),
                        getattr(type(rule), "rewrites", None)) != methods[rule.name]
                       for rule in (*registry.logical_rules.values(),
                                    *registry.physical_rules.values()))):
            return found
        alternatives = (logical_candidates if logical_candidates is not None
                        else logical_alternatives(
                            logical, tuple(registry.logical_rules.values()), context))
        fields = json.loads(_context_json(context))
        fields.update(
            order=context.order,
            pair_fractions={str(key): float(value)
                            for key, value in (context.pair_fractions or {}).items()},
            scan_fractions={alias: float(value)
                            for alias, value in (context.scan_fractions or {}).items()},
            document_tokens={alias: np.asarray(values).astype(int).tolist()
                             for alias, values in context.document_tokens.items()},
            first_documents=_first_documents(context), **_sharing_fields(context))
        intern, tokens, decode = _interning(context.tokenizer)
        rust, message = native.plan_operator(
            "search_plan", json.dumps(fields),
            json.dumps({"alternatives": [[encode_value(alternative.plan),
                                          list(alternative.rules)]
                                         for alternative in alternatives],
                        "rules": rules}), tokens, decode)
        assert message is None, f"Rust refused the search: {message}"
        built = json.loads(rust)
        expected = _candidate_fields(_Selected(found), intern)
        built.pop("alternative", None)
        for plan in (built, expected):
            for key in ("true_ids", "false_ids"):
                if key in plan.get("envelope", {}).get("settings", {}):
                    plan["envelope"]["settings"][key].sort()
        difference = same_values(_document_ends_sorted(built),
                                 _document_ends_sorted(expected))
        if os.environ.get("QUAIL_RUST_OPERATOR_LOG"):
            settings = expected.get("envelope", {}).get("settings", {})
            with open(os.environ["QUAIL_RUST_OPERATOR_LOG"], "a") as log:
                log.write(f"search_plan {type(found).__name__} "
                          f"{'same' if difference is None else 'differs'} "
                          f"{context.model.attention_mask} "
                          f"sharing={bool(fields['page_shared'])} "
                          f"retention={'retention' in settings} "
                          f"candidates={len(alternatives)} "
                          f"fired={_fired(expected)}\n")
        assert difference is None, f"Rust selected another plan: {difference}"
        return found

    return search_plan


class _Selected:
    """A selected plan or refusal, shaped as a candidate."""

    def __init__(self, plan):
        self.plan = plan


def _fired(plan) -> str:
    """Which physical rules left a mark on an encoded plan, for the log."""
    nodes = plan.get("envelope", {}).get("graph", {}).get("nodes", [])
    marks = set()
    for node in nodes:
        attributes = node["attributes"]
        spec = attributes.get("spec") or {}
        if attributes.get("share_prefixes") or spec.get("share_prefixes"):
            marks.add("share")
        if attributes.get("pass_group", -1) >= 0:
            marks.add("passes")
        if attributes.get("document_ends"):
            marks.add("ends")
        if attributes.get("attention") == "tree":
            marks.add("tree")
        if attributes.get("keep_kv"):
            marks.add("keep")
        if spec.get("scoring"):
            marks.add("classify")
    settings = plan.get("envelope", {}).get("settings", {})
    marks.update(key for key in ("deduplicate", "filter_limit") if key in settings)
    return ",".join(sorted(marks)) or "-"
