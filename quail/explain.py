"""Render logical and physical operator trees."""

from collections import Counter
from collections.abc import Mapping

from quail import logical as logical_nodes
from quail.physical import (
    AnchoredJoin,
    DocumentInput,
    Exchange,
    Limit,
    PackedFilter,
    PortRef,
    Project,
    Recombine,
    RequestExecution,
)


def _number(value):
    if 0 < abs(value) < 1:
        return f"{value:.3g}"
    return f"{value:,.1f}".removesuffix(".0")


def _prompt(prompt):
    args = ", ".join(f"{arg.alias}.{arg.column}" for arg in prompt.args)
    return f"PROMPT({prompt.template!r}, {args})"


def _selectivity(value):
    return "unknown" if value is None else f"{value * 100:g}%"


def _fields(fields, depth):
    lines = []
    pad = "  " * depth
    for key, value in fields.items():
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            lines.extend(_fields(value, depth + 1))
        elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], Mapping):
            for index, item in enumerate(value, 1):
                lines.append(f"{pad}{key} {index}:")
                lines.extend(_fields(item, depth + 1))
        else:
            lines.append(f"{pad}{key}={value}")
    return lines


def logical_tree(logical):
    """Return the logical operators and their expressions."""
    lines = []

    def visit(node, depth):
        details = []
        if isinstance(node, logical_nodes.Project):
            columns = ", ".join(f"{c.alias}.{c.column}" for c in node.columns)
            title = f"Project: {columns}"
            if node.limit is not None:
                lines.append(f"{'  ' * depth}Limit: {node.limit:,}")
                depth += 1
        elif isinstance(node, logical_nodes.Scan):
            title = f"Scan {node.provider} as {node.alias}"
            columns = ", ".join(dict.fromkeys((node.column, *node.columns)))
            title += f" [{columns}]"
        elif isinstance(node, logical_nodes.SemanticFilter):
            title = "SemanticFilter"
            details = [f"{_prompt(p.prompt)} "
                       f"(selectivity={_selectivity(p.selectivity)})"
                       for p in node.predicates]
        elif isinstance(node, logical_nodes.SemanticJoin):
            title = f"SemanticJoin ({node.semantics})"
            details = [f"{_prompt(node.predicate)} "
                       f"(selectivity={_selectivity(node.selectivity)})"]
            if node.anchor is not None:
                title += f" anchor={node.anchor}"
        else:
            title = node.type_name
            details = _fields(node.explain_fields(), 0)
        lines.append(f"{'  ' * depth}{title}")
        lines.extend(f"{'  ' * (depth + 1)}{detail}" for detail in details)
        for child in node.children():
            visit(child, depth + 1)

    visit(logical.root, 0)
    return "\n".join(lines)


def _logical_context(logical):
    scans, filters, joins = {}, {}, []

    def visit(node):
        for child in node.children():
            visit(child)
        if isinstance(node, logical_nodes.Scan):
            scans[node.alias] = node
        elif isinstance(node, logical_nodes.SemanticFilter):
            for predicate in node.predicates:
                filters.setdefault(predicate.prompt.args[0].alias, []).append(
                    predicate)
        elif isinstance(node, logical_nodes.SemanticJoin):
            joins.append(node)

    if logical is not None:
        visit(logical.root)
    return scans, filters, joins


def _estimated_rows(graph):
    rows = {}
    for node in graph.topological_nodes():
        inputs = [rows.get(port.source) for port in node.inputs]
        value = None
        if isinstance(node, DocumentInput):
            value = node.n_docs
        elif isinstance(node, PackedFilter):
            value = inputs[0] if inputs else None
            for stage in node.stages:
                if value == 0 or stage.selectivity == 0:
                    value = 0
                elif value is None or stage.selectivity is None:
                    value = None
                else:
                    value *= stage.selectivity
        elif isinstance(node, (Project, Limit)) and len(inputs) == 1:
            value = inputs[0]
            if isinstance(node, Limit) and value is not None:
                value = min(value, node.count)
        # Join nodes expose both document ids and predicate answers.
        # Their evaluated tuple counts are not output row estimates.
        for output in node.outputs:
            if output.name.startswith("filter_answers:"):
                continue
            rows[PortRef(node.node_id, output.name)] = value
    return rows


def physical_tree(graph, *, logical=None, verbose=False, metrics=None):
    """Return a physical tree, with references for shared inputs.

    Args:
        graph: The physical graph to display.
        logical: Optional logical plan supplying source names and predicates.
        verbose: Include node ids, ports, settings, and stage details.
        metrics: Optional measured metrics indexed by node id.
    """
    scans, filters, joins = _logical_context(logical)
    estimates = _estimated_rows(graph)
    uses = Counter(child for node in graph.nodes
                   for child in {port.source.node_id for port in node.inputs})
    shared = {node.node_id: index for index, node in enumerate(
        (node for node in graph.topological_nodes() if uses[node.node_id] > 1), 1)}
    visited = set()
    lines = []
    streamed = {}
    for node in graph.nodes:
        ports = {port.name: port for port in node.inputs}
        for name in node.streamed_inputs():
            streamed[ports[name].source.node_id] = node.node_id

    def describe(node):
        details = []
        title = type(node).__name__
        if isinstance(node, DocumentInput):
            source = scans.get(node.alias)
            title = (f"Scan {source.provider} as {node.alias}" if source else
                     f"DocumentInput: {node.alias}")
            mean = node.total_tokens / node.n_docs if node.n_docs else 0
            details.append(f"tokens={node.total_tokens:,}, "
                           f"mean_doc_tokens={_number(mean)}")
        elif isinstance(node, Project):
            title += ": " + ", ".join(node.columns)
        elif isinstance(node, Limit):
            title += f": {node.count:,}"
        elif isinstance(node, PackedFilter):
            title += f": {node.alias}"
            kv = []
            if len(node.stages) > 1 and node.arena_writes:
                kv.append("KV rewind=on")
            if node.keep_kv:
                fraction = _selectivity(node.keep_resident_fraction)
                kv.append("retain KV for joins "
                          f"(estimated resident survivors={fraction})")
            if node.node_id in streamed:
                kv.append("survivors stream into the join with KV pinned")
            details.append(", ".join(kv) if kv else
                           "KV: stored" if node.arena_writes else "KV: not stored")
            predicates = filters.get(node.alias, ())
            for index, stage in enumerate(node.stages, 1):
                predicate = (_prompt(predicates[stage.written_pos].prompt)
                             if stage.written_pos < len(predicates)
                             else f"predicate {stage.written_pos + 1}")
                details.append(f"{index}. {predicate} "
                               f"(selectivity={_selectivity(stage.selectivity)}, "
                               f"question_tokens={stage.question_tokens:,})")
        elif isinstance(node, AnchoredJoin):
            title += f": anchor={node.anchor}"
            source = {"none": "not resident", "filter": "from filters",
                      "kept": "from an earlier join"}.get(
                          node.anchor_resident, node.anchor_resident)
            if node.stream_anchor:
                source = "streamed from its filter"
            keep = "yes" if node.keep_anchor_kv else "no"
            details.append(f"KV: anchor={source}, retain after join={keep}")
            for index, stage in enumerate(node.stages, 1):
                predicate = (_prompt(joins[stage.written_pos].predicate)
                             if stage.written_pos < len(joins)
                             else f"predicate {stage.written_pos + 1}")
                details.append(
                    f"{index}. {stage.semantics} ({stage.anchor}, "
                    f"{', '.join(stage.partners)}): {predicate} "
                    f"(selectivity={_selectivity(stage.selectivity)}, "
                    f"estimated_evaluations={_number(stage.expected_tuples)})")
        elif isinstance(node, Exchange):
            title += f": next_anchor={node.next_anchor}"
        elif isinstance(node, Recombine):
            title += ": " + ", ".join(node.alias_order)
        elif isinstance(node, RequestExecution):
            title += f": {node.backend_name}"
            for spec in node.filters:
                predicates = filters.get(spec.alias, ())
                for position in spec.written_positions:
                    predicate = (_prompt(predicates[position].prompt)
                                 if position < len(predicates)
                                 else f"predicate {position + 1}")
                    details.append(f"Filter {spec.alias}: {predicate}")
            for spec in node.joins:
                predicate = (_prompt(joins[spec.written_pos].predicate)
                             if spec.written_pos < len(joins)
                             else f"predicate {spec.written_pos + 1}")
                details.append(f"Join {spec.semantics}: {predicate}")
        else:
            title = node.type_name
            if not verbose:
                details.extend(_fields(node.explain_fields(), 0))
        return title, details

    def visit(node, depth):
        pad = "  " * depth
        title, details = describe(node)
        reference = shared.get(node.node_id)
        if node.node_id in visited:
            lines.append(f"{pad}Reuse [{reference}]")
            return
        visited.add(node.node_id)
        if reference:
            title += f" [{reference}]"
        if metrics is None:
            values = [estimates.get(PortRef(node.node_id, output.name))
                      for output in node.outputs
                      if not output.name.startswith("filter_answers:")]
            value = values[0] if len(values) == 1 else None
            count = _number(value) if value is not None else "unknown"
            title += f" (estimated_rows={count})"
        elif node.node_id in metrics:
            measured = metrics[node.node_id]
            title += (f" (actual_rows={measured.output_rows:,}, "
                      f"wall_s={measured.wall_s:.3f})")
            if verbose:
                details.extend(_fields(vars(measured), 0))
        else:
            title += " (metrics unavailable)"
        lines.append(pad + title)
        lines.extend(pad + "  " + detail for detail in details)
        if verbose:
            lines.extend(_fields({"node_id": node.node_id,
                                  "type": node.type_name,
                                  **node.explain_fields()}, depth + 1))
            for port in node.inputs:
                lines.append(f"{pad}  {port.name} <- "
                             f"{port.source.node_id}[{port.source.port}]")
        children = dict.fromkeys(port.source.node_id for port in node.inputs)
        for child in children:
            visit(graph.node(child), depth + 1)
        for child in node.embedded_nodes():
            visit(child, depth + 1)

    visit(graph.node(graph.root.node_id), 0)
    for node in graph.topological_nodes():
        if node.node_id not in visited:
            visit(node, 0)
    return "\n".join(lines)
