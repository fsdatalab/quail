"""Plan a query with Quail's Rust planner in the ``_quail_native`` module.

The Rust planner applies the logical rules, searches the alternatives,
builds and prices each graph, and applies the physical rules, each in
the order the session's registry lists them by name.

A tokenizer that returns words instead of integer ids, as the tests'
does, crosses to Rust with each word interned as an integer id, and the
plan comes back with the words in place.
"""

from __future__ import annotations

import json

import _quail_native
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
from quail.physical.lowering import Vocabulary
from quail.physical.plan import PhysicalPlan, PlanEditError, Refusal
from quail.planner.candidates import LogicalAlternative

# the optimizer rules each model region runs, in order, as
# quail_datafusion::REGION_RULES names them
LOGICAL_RULES = (
    "eliminate_distinct", "place_semantic_nodes", "push_down_distinct",
    "push_down_filter", "optimize_projections",
)
# the built-in physical rules, in the order they apply
PHYSICAL_RULES = (
    "should_push_down_limit", "choose_classify_implementation", "pack_set_passes",
    "join_document_ends", "should_retain_kv", "should_share_prefixes",
    "choose_attention_path", "should_deduplicate",
)

# node attribute and spec fields that hold token ids
_TOKEN_FIELDS = ("question_token_ids", "document_head", "question_header_token_ids",
                 "frame_token_ids", "tail_token_ids", "prompt_token_parts",
                 "join_layout")

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


def untagged(value):
    """Encoded data without the codec's type tags, as the Rust specs read it."""
    if isinstance(value, dict):
        return {key: untagged(item) for key, item in value.items() if key != "type"}
    if isinstance(value, list):
        return [untagged(item) for item in value]
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
        "model": untagged(encode_value(context.model)),
        "device": untagged(encode_value(context.device)),
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


    def edit(self, graph, edit: dict, retained: set) -> tuple:
        """Apply a plan edit and price the edited graph in Rust.

        Returns:
            (the edited graph, its seconds, its per-node estimates).

        Raises:
            PlanEditError: The edit fails.
        """
        codecs = {codec.type_name: codec for codec in built_in_codecs()}
        intern = self._vocabulary.intern
        encoded = _tokens_mapped(json.loads(json.dumps(encode_graph(graph, codecs))),
                                 intern)
        found, message = self._native.plan_operator(
            "edit_plan", json.dumps(self._fields),
            json.dumps({"pricer": self._pricer, "graph": encoded,
                        "edit": _tokens_mapped(json.loads(json.dumps(edit)), intern),
                        "retained": sorted(retained)}),
            self._vocabulary.tokens, self._vocabulary.decoder())
        if message is not None:
            raise PlanEditError(message)
        found = json.loads(found)
        edited = decode_graph(_tokens_mapped(found["graph"], self._vocabulary.word),
                              codecs)
        return edited, found["seconds"], _estimates(found["estimates"])


def _estimates(estimates: dict) -> dict:
    """The per-node estimates with their member lists as tuples."""
    return {node_id: {key: tuple(value) if isinstance(value, list) else value
                      for key, value in entry.items()}
            for node_id, entry in estimates.items()}


def search_plan(alternatives, context, registry):
    """Select the cheapest refined Quail plan of the logical alternatives.

    Args:
        alternatives: The column placements, each with its plan and rewrites.
        context: The planning context.
        registry: The session extension registry, whose physical rules
            the planner applies in order.

    Returns:
        A PhysicalPlan, or a Refusal explaining why the query cannot run.
    """
    native = _quail_native
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


def planned_region(found: dict, tokenizer):
    """The PhysicalPlan, or Refusal, and chosen logical plan Rust planned.

    Args:
        found: What ``NativePlanner.plan`` returns, read from JSON.
        tokenizer: The session's tokenizer, to price an edited plan.

    Returns:
        (the plan or refusal, the logical plan it lowers or None).
    """
    if "refusal" in found:
        refusal = found["refusal"]
        return Refusal(reasons=tuple(refusal["reasons"]),
                       constraint=refusal["constraint"], needed=refusal["needed"],
                       available=refusal["available"], unit=refusal["unit"]), None
    logical = decode_value(LogicalPlan, json.loads(found["logical"]))
    envelope = found["envelope"]
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    graph = decode_graph(envelope["graph"], codecs)
    settings = dict(envelope["settings"])
    settings["candidates"] = tuple(
        {**candidate, "rewrites": tuple(candidate["rewrites"])}
        for candidate in settings.get("candidates", ()))
    vocabulary = Vocabulary(tokenizer)
    plan = PhysicalPlan(
        model=envelope["model"], device=envelope["device"], gpus=envelope["gpus"],
        backend=envelope["backend"], estimated_seconds=found["estimated_seconds"],
        nodes=graph.nodes, settings=settings, root=graph.root,
        estimates=_estimates(found["estimates"]),
        estimator=NativePricer(_quail_native, found["fields"], found["pricer"],
                               vocabulary),
        logical_plan=logical)
    return plan, logical


def column_placements(plan):
    """The placements of a plan's column operators that planning prices.

    Args:
        plan: The bound logical plan.

    Returns:
        The plan, then each plan with a classification, extraction, or
        map moved above a join, each with the rewrites that made it.

    Raises:
        CompileError: A moved plan is not valid.
    """
    found, message = _quail_native.column_placements(
        json.dumps(encode_value(plan)))
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
    """Plan a stage that asks no model."""
    native = _quail_native
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


def request_plan(logical, context, settings: dict):
    """Plan a region for a request backend in Rust.

    Args:
        logical: The region's logical plan.
        context: The planning context.
        settings: The backend's ``backend_name``, ``filter_submission``,
            ``join_submission``, ``answer_submission``, and
            ``scores_labels``.

    Returns:
        A Refusal, or (the graph as JSON, the plan's settings): the
        request node is the backend's own type, which its codecs decode.

    Raises:
        CompileError: A prompt cannot be lowered, or the prompts do not
            share a preamble.
    """
    vocabulary = Vocabulary(context.tokenizer)
    found, message = _quail_native.plan_operator(
        "request_plan", json.dumps(context_fields(context, vocabulary)),
        json.dumps({"plan": encode_value(logical), "settings": settings}),
        vocabulary.tokens, vocabulary.decoder())
    if message is not None:
        raise CompileError(message)
    found = json.loads(found)
    if "refusal" in found:
        refusal = found["refusal"]
        return Refusal(reasons=tuple(refusal["reasons"]),
                       constraint=refusal["constraint"], needed=refusal["needed"],
                       available=refusal["available"], unit=refusal["unit"])
    plan_settings = dict(found["settings"])
    for key in ("true_ids", "false_ids"):
        plan_settings[key] = [vocabulary.word(token) for token in plan_settings[key]]
    return _tokens_mapped(found["graph"], vocabulary.word), plan_settings


def refine_plan(plan, context, registry):
    """Apply the registry's physical rules again to an edited plan.

    A plan the Rust planner did not price, such as another backend's,
    is returned unchanged.
    """
    native = _quail_native
    estimator = getattr(plan, "estimator", None)
    if not isinstance(estimator, NativePricer):
        return plan
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
