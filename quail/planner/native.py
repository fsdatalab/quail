"""Plan a query with Quail's Rust planner in the ``_quail_native`` module.

The Rust planner runs the same search, graph builder, physical rules,
and pricing as the Python planner, which ``optimize.search_plan`` keeps
as the fallback: for another backend, a registry with rules or physical
planners of its own, a session without a tokenizer, a missing native
module, or ``QUAIL_PLANNER=python``.

A tokenizer that returns words instead of integer ids, as the tests'
does, crosses to Rust with each word interned as an integer id, and the
plan comes back with the words in place.
"""

from __future__ import annotations

import json
import os

import numpy as np

from quail.common.document_ends import last_tokens
from quail.common.expressions import CompileError
from quail.common.tokens import prefix_tree
from quail.logical import LogicalPlan
from quail.physical.codec import (
    built_in_codecs,
    decode_graph,
    decode_value,
    encode_graph,
    encode_value,
)
from quail.planner.optimizer.candidates import LogicalAlternative
from quail.planner.plan import PhysicalPlan, Refusal

# the built-in logical rules, in registry order
LOGICAL_RULES = (
    "should_eliminate_distinct", "should_push_down_distinct",
    "should_push_down_projection", "should_push_down_filters",
    "should_lift_column_operators",
)
# the built-in logical then physical rules, in registry order
BUILT_IN_RULES = (
    "should_eliminate_distinct", "should_push_down_distinct",
    "should_push_down_projection", "should_push_down_filters",
    "should_lift_column_operators", "should_eliminate_window_sort",
    "should_push_down_limit_through_window", "should_push_down_limit",
    "choose_classify_implementation", "pack_set_passes", "join_document_ends",
    "should_retain_kv", "should_share_prefixes", "choose_attention_path",
    "should_deduplicate",
)

# node attribute and spec fields that hold token ids
_TOKEN_FIELDS = ("question_token_ids", "document_head", "question_header_token_ids",
                 "frame_token_ids", "tail_token_ids", "prompt_token_parts",
                 "join_layout")

# interned word ids start here, above any vocabulary
_WORDS = 1 << 40


def _module():
    if os.environ.get("QUAIL_PLANNER") == "python":
        return None
    try:
        import _quail_native
    except ImportError:
        return None
    return _quail_native


class Vocabulary:
    """A tokenizer's ids as Rust reads them, with words interned."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids, self.words = {}, {}

    def intern(self, token):
        """The integer id of a token: itself, or an interned word's."""
        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            return int(token)
        if token not in self.ids:
            self.ids[token] = _WORDS | len(self.ids)
            self.words[self.ids[token]] = token
        return self.ids[token]

    def word(self, token):
        """The token an id stands for."""
        return self.words.get(token, token)

    def tokens(self, text):
        """Tokenize text into integer ids."""
        return [self.intern(token) for token in self.tokenizer(text)]

    def decoder(self):
        """Decode a list of ids, or None for a tokenizer that cannot decode."""
        decode = getattr(self.tokenizer, "decode", None)
        if decode is None:
            return None

        def decoded(ids):
            text = decode([self.word(token) for token in ids])
            return (text.decode("utf-8", errors="replace")
                    if isinstance(text, bytes) else str(text))

        return decoded


def _mapped(value, convert):
    """Every token in a nested list, converted."""
    if isinstance(value, (list, tuple)):
        return [_mapped(item, convert) for item in value]
    if isinstance(value, (int, str, np.integer)) and not isinstance(value, bool):
        return convert(value)
    return value


def _tokens_mapped(value, convert):
    """A node's or spec's fields with their token ids converted."""
    if isinstance(value, list):
        return [_tokens_mapped(item, convert) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key in _TOKEN_FIELDS and item is not None:
            out[key] = _mapped(item, convert)
        elif key == "label_token_ids" and item:
            # a spec's are id lists; an AiJoin's are (alias, ids) pairs
            out[key] = (_mapped(item, convert) if "type" in value
                        else [[alias, _mapped(ids, convert)] for alias, ids in item])
        elif key == "document_ends" and item:
            out[key] = [[alias, _mapped(tail, convert),
                         [[convert(token), _mapped(joined, convert)]
                          for token, joined in ends]]
                        for alias, tail, ends in item]
        else:
            out[key] = _tokens_mapped(item, convert)
    return out


def _untagged(value):
    if isinstance(value, dict):
        return {key: _untagged(item) for key, item in value.items() if key != "type"}
    if isinstance(value, list):
        return [_untagged(item) for item in value]
    return value


def _table_statistics(context, vocabulary) -> dict:
    """Each table's copies, shared pages, reader groups, and last tokens."""
    first, page_shared, reader_groups, last = {}, {}, {}, {}
    for alias, lengths in context.document_tokens.items():
        store = getattr(lengths, "store", None)
        if store is None:
            continue
        documents = store.first_documents.to_numpy()
        keys = documents == np.arange(len(documents))
        if not keys.all():
            first[alias] = keys.tolist()
        tree = prefix_tree(store, context.device.page_tokens)
        page_shared[alias] = [int(value) for value in tree.shared]
        reader_groups[alias] = [[int(value) for value in group]
                                for group in zip(*tree.reader_groups(lengths))]
        try:
            ends = last_tokens(store.arrow_array)
        except (TypeError, ValueError):
            # a store of words has no integer last tokens
            continue
        last[alias] = sorted({vocabulary.intern(token) for token in ends.tolist()})
    return {"first_documents": first, "page_shared": page_shared,
            "reader_groups": reader_groups, "last_tokens": last}


def context_fields(context, vocabulary) -> dict:
    """The planning context's fields, as the native module reads them."""
    return {
        "model": _untagged(encode_value(context.model)),
        "device": _untagged(encode_value(context.device)),
        "gpus": context.gpus, "canvas_draws": context.canvas_draws,
        "document_tokens": {alias: np.asarray(values).astype(int).tolist()
                            for alias, values in context.document_tokens.items()},
        "plan": (None if context.logical_plan is None
                 else encode_value(context.logical_plan)),
        "attention": context.attention, "order": context.order,
        "pair_fractions": {str(key): float(value)
                           for key, value in (context.pair_fractions or {}).items()},
        "scan_fractions": {alias: float(value)
                           for alias, value in (context.scan_fractions or {}).items()},
        "backend": context.backend,
        **_table_statistics(context, vocabulary),
    }


class NativePricer:
    """Price an edited graph with the Rust plan pricer, as PlanPricer does."""

    def __init__(self, native, fields, pricer, vocabulary):
        self._native, self._fields = native, fields
        self._pricer, self._vocabulary = pricer, vocabulary

    def __call__(self, graph) -> tuple[float, dict]:
        codecs = {codec.type_name: codec for codec in built_in_codecs()}
        encoded = _tokens_mapped(json.loads(json.dumps(encode_graph(graph, codecs))),
                                 self._vocabulary.intern)
        found, message = self._native.plan_operator(
            "price", json.dumps(self._fields),
            json.dumps({"pricer": self._pricer, "graph": encoded}),
            self._vocabulary.tokens, self._vocabulary.decoder())
        if message is not None:
            raise ValueError(message)
        found = json.loads(found)
        return found["seconds"], _estimates(found["estimates"])


def _estimates(estimates: dict) -> dict:
    """The per-node estimates with their member lists as tuples."""
    return {node_id: {key: tuple(value) if isinstance(value, list) else value
                      for key, value in entry.items()}
            for node_id, entry in estimates.items()}


def search_plan(alternatives, context, registry):
    """Select the cheapest refined plan in Rust, or None to plan in Python.

    Args:
        alternatives: The logical alternatives, each with its plan and rules.
        context: The planning context.
        registry: The session extension registry.

    Returns:
        A PhysicalPlan or a Refusal, or None when the Rust planner does
        not cover the request.
    """
    native = _module()
    rules = tuple(rule.name for rule in (*registry.logical_rules.values(),
                                         *registry.physical_rules.values()))
    if (native is None or context.tokenizer is None or context.backend != "quail"
            or registry.physical_planners or rules != BUILT_IN_RULES):
        return None
    vocabulary = Vocabulary(context.tokenizer)
    fields = context_fields(context, vocabulary)
    found, message = native.plan_operator(
        "search_plan", json.dumps(fields),
        json.dumps({"alternatives": [[encode_value(alternative.plan),
                                      list(alternative.rules)]
                                     for alternative in alternatives],
                    "rules": [rule.name for rule in registry.physical_rules.values()]}),
        vocabulary.tokens, vocabulary.decoder())
    if message is not None:
        raise ValueError(message)
    found = json.loads(found)
    if "refusal" in found:
        refusal = found["refusal"]
        return Refusal(reasons=tuple(refusal["reasons"]),
                       constraint=refusal["constraint"], needed=refusal["needed"],
                       available=refusal["available"], unit=refusal["unit"])
    envelope = found["envelope"]
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    graph = decode_graph(_tokens_mapped(envelope["graph"], vocabulary.word), codecs)
    settings = dict(envelope["settings"])
    settings["candidates"] = tuple(
        {**candidate, "rewrites": tuple(candidate["rewrites"])}
        for candidate in settings["candidates"])
    for key in ("true_ids", "false_ids"):
        if key in settings:
            settings[key] = [vocabulary.word(token) for token in settings[key]]
    return PhysicalPlan(
        model=envelope["model"], device=envelope["device"], gpus=envelope["gpus"],
        backend=envelope["backend"], estimated_seconds=found["estimated_seconds"],
        nodes=graph.nodes, settings=settings, root=graph.root,
        estimates=_estimates(found["estimates"]),
        estimator=NativePricer(native, fields, found["pricer"], vocabulary),
        logical_plan=alternatives[found["alternative"]].plan)


def logical_alternatives(plan, registry, context):
    """Apply the registry's logical rules in Rust, or None to apply them in Python.

    Args:
        plan: The input logical plan.
        registry: The session extension registry.
        context: The planning context, whose catalog names each table's
            id column.

    Returns:
        The alternatives in order, each with its plan and the rules that
        made it, or None when a rule is not a built-in one.

    Raises:
        CompileError: A rule refuses the plan.
    """
    native = _module()
    names = tuple(rule.name for rule in registry.logical_rules.values())
    if native is None or names != LOGICAL_RULES:
        return None
    catalog = context.catalog
    id_columns = (None if catalog is None else {
        name: provider.id_col for name, provider in catalog.providers.items()
        if hasattr(provider, "id_col")})
    found, message = native.logical_alternatives(
        json.dumps(encode_value(plan)), list(names), id_columns)
    if message is not None:
        raise CompileError(message)
    return tuple(LogicalAlternative(decode_value(LogicalPlan, json.loads(text)),
                                    tuple(rules))
                 for text, rules in found)


def _plan_from(found: dict, native, fields, vocabulary, logical_plan, pricer=None):
    """A PhysicalPlan, or a Refusal, from what a native planning call returns."""
    if "refusal" in found:
        refusal = found["refusal"]
        return Refusal(reasons=tuple(refusal["reasons"]),
                       constraint=refusal["constraint"], needed=refusal["needed"],
                       available=refusal["available"], unit=refusal["unit"])
    envelope = found["envelope"]
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    graph = decode_graph(_tokens_mapped(envelope["graph"], vocabulary.word), codecs)
    pricer = found.get("pricer", pricer)
    return PhysicalPlan(
        model=envelope["model"], device=envelope["device"], gpus=envelope["gpus"],
        backend=envelope["backend"],
        estimated_seconds=found.get("estimated_seconds", 0.0),
        nodes=graph.nodes, settings=dict(envelope["settings"]), root=graph.root,
        estimates=_estimates(found.get("estimates", {})),
        estimator=(None if pricer is None
                   else NativePricer(native, fields, pricer, vocabulary)),
        logical_plan=logical_plan)


def relational_plan(logical, context, num_documents: int):
    """The plan of a stage that asks no model, or None to build it in Python."""
    native = _module()
    if native is None:
        return None
    vocabulary = Vocabulary(context.tokenizer or (lambda text: []))
    fields = context_fields(context, vocabulary)
    found, message = native.plan_operator(
        "relational_plan", json.dumps(fields),
        json.dumps({"plan": encode_value(logical), "num_documents": num_documents}),
        vocabulary.tokens, None)
    if message is not None:
        raise ValueError(message)
    envelope = json.loads(found)
    return _plan_from({"envelope": envelope}, native, fields, vocabulary, logical)


def refine_plan(plan, context, registry):
    """Apply the physical rules to an edited plan in Rust, or None for Python.

    Rust refines a plan whose pricer is Rust's, with the built-in rules.
    """
    native = _module()
    estimator = getattr(plan, "estimator", None)
    rules = tuple(rule.name for rule in (*registry.logical_rules.values(),
                                         *registry.physical_rules.values()))
    if (native is None or not isinstance(estimator, NativePricer)
            or context.tokenizer is None or rules != BUILT_IN_RULES):
        return None
    vocabulary = estimator._vocabulary
    fields = context_fields(context, vocabulary)
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    envelope = json.loads(json.dumps(plan.to_envelope(codecs)))
    envelope["graph"] = _tokens_mapped(envelope["graph"], vocabulary.intern)
    found, message = native.plan_operator(
        "refine_plan", json.dumps(fields),
        json.dumps({"envelope": envelope, "pricer": estimator._pricer,
                    "estimated_seconds": plan.estimated_seconds,
                    "estimates": json.loads(json.dumps(dict(plan.estimates))),
                    "rules": [rule.name for rule in registry.physical_rules.values()]}),
        vocabulary.tokens, vocabulary.decoder())
    if message is not None:
        raise ValueError(message)
    return _plan_from(json.loads(found), native, fields, vocabulary,
                      plan.logical_plan, estimator._pricer)
