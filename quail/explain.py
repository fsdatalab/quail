"""Render logical and physical operator trees."""

from collections import Counter
from collections.abc import Mapping

from quail import logical as logical_nodes
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    Foreign,
    HashJoin,
    Limit,
    PortRef,
    Project,
    Recombine,
    RequestExecution,
    Scan,
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


def measured_stages(report) -> dict:
    """Index a report's stage measurements by the predicate they ran.

    Args:
        report: An execution report (QueryResult.report).

    Returns:
        ("filter", alias, written_pos) and ("join", written_pos) keys
        mapped to the stage dicts of report["stages"]. Stages saved
        before written_pos was recorded are left out.
    """
    stages = {}
    for stage in (report or {}).get("stages", ()):
        written_pos = stage.get("written_pos")
        if written_pos is None:
            continue
        if stage.get("op") == "filter":
            stages[("filter", stage["alias"], written_pos)] = stage
        elif stage.get("op") == "join":
            stages[("join", written_pos)] = stage
    return stages


def _measured_stage(stage, evaluated_key, unit):
    if stage is None:
        return ""
    return (f", observed={_selectivity(stage['observed_selectivity'])}, "
            f"evaluated={stage[evaluated_key]:,} {unit}")


def measured_summary(report, graph, workers, usd_per_hour=None) -> list:
    """Return the measured totals of one run as indented explain lines.

    Args:
        report: The execution report (QueryResult.report).
        graph: The executed physical graph, for the input document count.
        workers: GPUs the query ran on.
        usd_per_hour: Price of one GPU, for the cost per query.
    """
    wall = report.get("wall_s")
    lines = []
    if wall is None:
        return ["  query time: not measured"]
    lines.append(f"  query time={wall:.3f} s (model startup excluded)")
    pairs = sum(stage.get("tuples", 0) for stage in report.get("stages", ())
                if stage.get("op") == "join")
    documents = sum(node.n_docs for node in graph.nodes
                    if isinstance(node, Scan))
    if wall > 0:
        if pairs:
            lines.append(f"  throughput={_number(pairs / wall)} document "
                         f"pairs/second ({pairs:,} pairs evaluated across "
                         "the join stages)")
        else:
            lines.append(f"  throughput={_number(documents / wall)} "
                         f"documents/second ({documents:,} input "
                         "documents)")
    tokens = [f"fresh_tokens={report['fresh_tokens']:,}"]
    if report.get("cached_tokens") is not None:
        tokens.append(f"cached_tokens={report['cached_tokens']:,}")
    lines.append("  " + ", ".join(tokens))
    if usd_per_hour is not None:
        cost = wall / 3600 * workers * usd_per_hour
        lines.append(f"  GPU cost=${cost:.4f}/query ({workers} GPU x "
                     f"${usd_per_hour:.4f}/hour, startup excluded)")
    if report.get("boot_s") is not None:
        kind = report.get("boot_kind")
        lines.append(f"  model startup={report['boot_s']:.3f} s"
                     + (f" ({kind})" if kind else ""))
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
        elif isinstance(node, logical_nodes.Join):
            title = "Join"
            details = ([f"on {condition}" for condition in node.on]
                       or ["cross"])
        elif isinstance(node, logical_nodes.Apply):
            title = f"Apply {node.function} ({node.kind}, {node.ids})"
            details = ["columns: " + ", ".join(
                f"{ref.alias}.{ref.column}" for ref in node.columns)]
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
        if isinstance(node, Scan):
            value = node.n_docs
        elif isinstance(node, AiFilter):
            value = inputs[0] if inputs else None
            for stage in node.stages:
                if value == 0 or stage.selectivity == 0:
                    value = 0
                elif value is None or stage.selectivity is None:
                    value = None
                else:
                    value *= stage.selectivity
        elif isinstance(node, (Project, Limit, Exchange)) and len(inputs) == 1:
            value = inputs[0]
            if isinstance(node, Limit) and value is not None:
                value = min(value, node.count)
        elif isinstance(node, Foreign) and node.ids == "preserve":
            value = inputs[0] if inputs else None
        # Join nodes expose both document ids and predicate answers.
        # Their evaluated tuple counts are not output row estimates.
        for output in node.outputs:
            if output.name.startswith("filter_answers:"):
                continue
            rows[PortRef(node.node_id, output.name)] = value
    return rows


def physical_tree(graph, *, logical=None, verbose=False, metrics=None,
                  estimates=None, stages=None):
    """Return a physical tree, with references for shared inputs.

    Args:
        graph: The physical graph to display.
        logical: Optional logical plan supplying source names and predicates.
        verbose: Include node ids, ports, settings, and stage details.
        metrics: Optional measured metrics indexed by node id.
        estimates: Optional per node estimates (PhysicalPlan.estimates).
        stages: Optional measured stages from measured_stages(report).
    """
    estimates_by_node = estimates or {}
    stages = stages or {}
    scans, filters, joins = _logical_context(logical)
    estimates = _estimated_rows(graph)
    uses = Counter(child for node in graph.nodes
                   for child in {port.source.node_id for port in node.inputs})
    shared = {node.node_id: index for index, node in enumerate(
        (node for node in graph.topological_nodes() if uses[node.node_id] > 1), 1)}
    visited = set()
    lines = []

    def describe(node):
        details = []
        title = type(node).__name__
        if isinstance(node, Scan):
            source = scans.get(node.alias)
            title = (f"Scan {source.provider} as {node.alias}" if source else
                     f"Scan: {node.alias}")
            mean = node.total_tokens / node.n_docs if node.n_docs else 0
            details.append(f"tokens={node.total_tokens:,}, "
                           f"mean_doc_tokens={_number(mean)}")
        elif isinstance(node, Project):
            title += ": " + ", ".join(node.columns)
        elif isinstance(node, Limit):
            title += f": {node.count:,}"
        elif isinstance(node, AiFilter):
            title += f": {node.alias}"
            kv = []
            if len(node.stages) > 1 and node.arena_writes:
                kv.append("KV rewind=on")
            if node.keep_kv:
                kv.append("retain KV for later joins")
            if node.pin_survivors:
                kv.append("survivors stream into the join with KV pinned")
            details.append(", ".join(kv) if kv else
                           "KV: stored" if node.arena_writes else "KV: not stored")
            predicates = filters.get(node.alias, ())
            for index, stage in enumerate(node.stages, 1):
                predicate = (_prompt(predicates[stage.written_pos].prompt)
                             if stage.written_pos < len(predicates)
                             else f"predicate {stage.written_pos + 1}")
                measured = _measured_stage(
                    stages.get(("filter", node.alias, stage.written_pos)),
                    "evaluated", "documents")
                details.append(f"{index}. {predicate} "
                               f"(selectivity={_selectivity(stage.selectivity)}"
                               f"{measured}, "
                               f"question_tokens={stage.question_tokens:,})")
        elif isinstance(node, AiJoin):
            title += f": anchor={node.anchor}"
            source = {"none": "not resident", "filter": "from filters",
                      "kept": "from an earlier join"}.get(
                          node.anchor_resident, node.anchor_resident)
            anchor_port = next(
                (port for port in node.inputs
                 if port.source.port == f"ids:{node.anchor}"), None)
            producer = (graph.node(anchor_port.source.node_id)
                        if anchor_port else None)
            if isinstance(producer, AiFilter) and producer.pin_survivors:
                source = "streamed from its filter"
            keep = "yes" if node.keep_anchor_kv else "no"
            details.append(f"KV: anchor={source}, retain after join={keep}")
            for index, stage in enumerate(node.stages, 1):
                predicate = (_prompt(joins[stage.written_pos].predicate)
                             if stage.written_pos < len(joins)
                             else f"predicate {stage.written_pos + 1}")
                pairs = (f" over pairs from {stage.pairs_from}"
                         if stage.pairs_from else "")
                measured = _measured_stage(
                    stages.get(("join", stage.written_pos)), "tuples", "pairs")
                details.append(
                    f"{index}. {stage.semantics} ({stage.anchor}, "
                    f"{', '.join(stage.partners)}){pairs}: {predicate} "
                    f"(selectivity={_selectivity(stage.selectivity)}"
                    f"{measured}, "
                    f"estimated_evaluations={_number(stage.expected_tuples)})")
        elif isinstance(node, HashJoin):
            title += ": " + " and ".join(
                f"{node.left}.{left} = {node.right}.{right}"
                for left, right in node.on)
            details.append(
                f"pairs kept: {node.pair_fraction:.4g} of the cross product")
        elif isinstance(node, Foreign):
            title += (f": {node.function} ({node.kind}, {node.ids}) on "
                      f"{', '.join(node.aliases)}")
            if node.columns:
                details.append("columns: " + ", ".join(
                    f"{alias}.{column}" for alias, column in node.columns))
        elif isinstance(node, Exchange):
            title += f": {node.anchor} to the GPU holding its KV"
        elif isinstance(node, Barrier):
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
        values = [estimates.get(PortRef(node.node_id, output.name))
                  for output in node.outputs
                  if not output.name.startswith("filter_answers:")]
        value = values[0] if len(values) == 1 else None
        count = _number(value) if value is not None else "unknown"
        estimate = estimates_by_node.get(node.node_id, {})
        if metrics is None:
            title += f" (estimated_rows={count}"
            if "seconds" in estimate:
                title += f", estimated_seconds={estimate['seconds']:.3f}"
            title += ")"
            if "release_recompute_tokens" in estimate:
                pinned = node.pin_survivors
                details.append(
                    ("if the KV were released here instead of pinned: "
                     if pinned else "expected recompute at the join: ")
                    + f"{estimate['release_recompute_tokens']:,} tokens, "
                    f"{estimate['release_recompute_seconds']:.3f} s")
        elif node.node_id in metrics:
            measured = metrics[node.node_id]
            parts = [f"actual_rows={measured.output_rows:,}"]
            if value is not None:
                parts.append(f"estimated_rows={count}")
            parts.append(f"wall_s={measured.wall_s:.3f}")
            if "seconds" in estimate:
                parts.append(f"estimated_seconds={estimate['seconds']:.3f}")
            if measured.fresh_tokens:
                parts.append(f"fresh_tokens={measured.fresh_tokens:,}")
            title += f" ({', '.join(parts)})"
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
