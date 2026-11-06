"""Render a region's logical and physical plans with Quail's Rust planner."""

import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass

import _quail_native

from quail.physical.codec import built_in_codecs, encode_value
from quail.physical.plan import Refusal

_BUILT_IN_MODULE = "quail.physical.nodes"


def _json(value) -> str:
    return json.dumps(value, default=str)


def _shown(value):
    """Return a value as explain's ``key=value`` lines show it.

    A mapping, or a list of mappings, keeps its items to nest; any other
    value is its ``str``.
    """
    if isinstance(value, Mapping):
        return {str(key): _shown(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
        return [_shown(item) for item in value]
    return str(value)


def _graph(graph) -> tuple[dict, dict]:
    """Encode a graph for Rust, with each extension node's own report.

    Returns:
        (the encoded graph, node id -> an extension node's type name,
        inputs, outputs, and explain fields).
    """
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    nodes, extensions = [], {}
    for node in graph.nodes:
        codec = codecs.get(node.type_name)
        if codec is not None and type(node).__module__.startswith(_BUILT_IN_MODULE):
            nodes.append(codec.encode(node))
            continue
        nodes.append({"id": node.node_id})
        extensions[node.node_id] = {
            "type_name": node.type_name,
            "inputs": [[port.name, port.source.node_id, port.source.port]
                       for port in node.inputs],
            "outputs": [output.name for output in node.outputs],
            "fields": [(name, _shown(value))
                       for name, value in node.explain_fields().items()],
        }
    return {"nodes": nodes, "root": encode_value(graph.root)}, extensions


def _metrics(node_metrics) -> dict:
    """Each node's measured metrics as (field, value) pairs."""
    return {
        node_id: ([(field.name, getattr(measured, field.name))
                   for field in fields(measured)]
                  if is_dataclass(measured) else list(dict(measured).items()))
        for node_id, measured in node_metrics.items()
    }


def _model_spec(model: str):
    from quail.planner.native import untagged
    from quail.specs import MODELS

    spec = MODELS.get(model)
    return None if spec is None else untagged(encode_value(spec))


def explain(logical, physical, *, verbose: bool = False, result=None,
            usd_per_hour: float | None = None, rules=None) -> str:
    """Format the logical and physical operator trees.

    Args:
        logical: The optimized logical plan.
        physical: The physical plan or planning refusal.
        verbose: Include runtime settings, internal node fields, and
            the optimizer rules.
        result: The QueryResult of running the plan. When given, each
            node shows its measured rows, time, and tokens next to the
            estimates, and the measured totals follow the tree.
        usd_per_hour: Price of one GPU, for the measured cost per query.
        rules: Registered rules in order, listed when verbose.

    Returns:
        The explain text.
    """
    request = {
        "logical": encode_value(logical),
        "verbose": verbose,
        "usd_per_hour": usd_per_hour,
        "rules": (None if rules is None
                  else [[rule.name, rule.kind] for rule in rules]),
    }
    if isinstance(physical, Refusal):
        request["refusal"] = asdict(physical)
    else:
        graph, extensions = _graph(physical.graph)
        request["physical"] = {
            "backend": physical.backend, "model": physical.model,
            "device": physical.device, "gpus": physical.gpus,
            "settings": list(physical.settings.items()),
            "shown_settings": [(name, _shown(value))
                               for name, value in physical.settings.items()],
            "graph": graph,
            "estimates": list((physical.estimates or {}).items()),
            "extensions": extensions,
        }
        request["model_spec"] = _model_spec(physical.model)
        if result is not None:
            request["run"] = {"report": result.report,
                              "metrics": _metrics(result.node_metrics)}
    return _quail_native.explain_plan(_json(request))


def physical_tree(graph, *, verbose: bool = False, node_metrics=None,
                  report=None) -> str:
    """Format an executed graph with each node's measured rows and time.

    Args:
        graph: The executed physical graph.
        verbose: Include node ids, ports, and fields.
        node_metrics: Measured metrics by node id.
        report: The execution report, for each predicate's measurements.

    Returns:
        The tree text.
    """
    encoded, extensions = _graph(graph)
    request = {
        "tree_only": True,
        "verbose": verbose,
        "physical": {"graph": encoded, "extensions": extensions},
        "run": {"report": report or {},
                "metrics": _metrics(node_metrics or {})},
    }
    return _quail_native.explain_plan(_json(request))
