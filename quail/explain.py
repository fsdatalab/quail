"""Render logical and physical operator trees."""

from collections import Counter
from collections.abc import Mapping

from quail import logical as logical_nodes
from quail.logical import DEFAULT_SELECTIVITY, effective_selectivity
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    Foreign,
    HashJoin,
    Limit,
    PhysicalScan,
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
            columns = ", ".join(node.explain_fields()["columns"])
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
            details = [f"{_prompt(node.prompt)} "
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


DEFAULT_MARK = "*"
DEFAULT_NOTE = (f"{DEFAULT_MARK} no selectivity was given; the planner "
                f"assumes {DEFAULT_SELECTIVITY * 100:g}%")
PROMPT_CHARS = 32

# table columns in display order: (cell key, header)
_COLUMNS = (
    ("est_rows", "est. rows"),
    ("rows", "rows"),
    ("est_pass", "est. pass"),
    ("pass", "pass"),
    ("est_time", "est. time"),
    ("time", "time"),
    ("tokens", "fresh tokens"),
)


def _selectivity(value, mark=False):
    if value is None:
        default = f"{DEFAULT_SELECTIVITY * 100:g}%"
        return default + DEFAULT_MARK if mark else default + " (default)"
    return f"{value * 100:.3g}%"


def _seconds(value):
    if value is None:
        return ""
    if value < 0.001:
        return "<1 ms"
    if value < 1:
        return f"{value * 1000:.3g} ms"
    return f"{value:.3g} s"


def _short_prompt(prompt):
    template = prompt.template
    if len(template) > PROMPT_CHARS:
        template = template[:PROMPT_CHARS] + "…"
    return f"PROMPT({template!r})"


def _ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10, "th")
    return f"{n}{suffix}"


def _table(rows):
    """Lay out (text, cells) rows as a tree column and aligned columns."""
    columns = [(key, label) for key, label in _COLUMNS
               if any(cells.get(key) for _, cells in rows)]
    width = max(len(text) for text, _ in rows)
    widths = {
        key: max(len(label), *(len(cells.get(key, "")) for _, cells in rows))
        for key, label in columns
    }
    lines = []
    if columns:
        lines.append(" " * width + "".join(
            f"   {label:>{widths[key]}}" for key, label in columns))
    for text, cells in rows:
        line = text.ljust(width) + "".join(
            f"   {cells.get(key, ''):>{widths[key]}}" for key, _ in columns)
        lines.append(line.rstrip())
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


def run_summary(report, graph, workers, usd_per_hour=None) -> list:
    """Return the measured totals of one run as labeled lines.

    Args:
        report: The execution report (QueryResult.report).
        graph: The executed physical graph, for the input document count.
        workers: GPUs the query ran on.
        usd_per_hour: Price of one GPU, for the cost per query.
    """
    wall = report.get("wall_s")
    if wall is None:
        return ["query time   not measured"]
    items = [("query time", f"{_seconds(wall)} (model startup excluded)")]
    if report.get("boot_s") is not None:
        kind = f" ({report['boot_kind']})" if report.get("boot_kind") else ""
        items.append(("startup", _seconds(report["boot_s"]) + kind))
    pairs = sum(stage.get("tuples", 0) for stage in report.get("stages", ())
                if stage.get("op") == "join")
    if not pairs:
        # a score over document pairs has no join stage records
        pairs = report.get("evaluated_document_pairs") or 0
    documents = sum(node.n_docs for node in graph.nodes
                    if isinstance(node, PhysicalScan))
    if wall > 0 and pairs:
        items.append(("throughput", f"{_number(pairs / wall)} document "
                      f"pairs/second over {pairs:,} evaluated pairs"))
    elif wall > 0:
        items.append(("throughput", f"{_number(documents / wall)} "
                      f"documents/second over {documents:,} input documents"))
    tokens = f"{report['fresh_tokens']:,} fresh"
    if report.get("cached_tokens") is not None:
        tokens += f", {report['cached_tokens']:,} read from KV"
    items.append(("tokens", tokens))
    if usd_per_hour is not None:
        cost = wall / 3600 * workers * usd_per_hour
        items.append(("GPU cost", f"${cost:.4f} per query ({workers} GPU at "
                      f"${usd_per_hour:.4f}/hour, startup excluded)"))
    width = max(len(label) for label, _ in items)
    return [f"{label.ljust(width)}   {value}" for label, value in items]


def _logical_context(logical):
    if logical is None:
        return {}, {}, []
    operators = logical.operators()
    scans = {scan.alias: scan for scan in operators.scans}
    return scans, operators.filters, list(operators.joins)


def _estimated_rows(graph):
    rows = {}
    for node in graph.topological_nodes():
        inputs = [rows.get(port.source) for port in node.inputs]
        value = None
        if isinstance(node, PhysicalScan):
            value = node.n_docs
        elif isinstance(node, AiFilter):
            value = inputs[0] if inputs else None
            for stage in node.stages:
                if value is not None:
                    value *= effective_selectivity(stage.selectivity)
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
    """Return a physical tree with estimated and measured columns.

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
    rows = []
    used_default = []

    def stage_cells(selectivity, expected, measured, evaluated_key):
        if selectivity is None:
            used_default.append(True)
        cells = {"est_rows": _number(expected),
                 "est_pass": _selectivity(selectivity, mark=True)}
        if measured is not None:
            cells["rows"] = f"{measured[evaluated_key]:,}"
            cells["pass"] = _selectivity(measured["observed_selectivity"])
        return cells

    def describe(node):
        """Return the node's title, detail lines, and predicate rows."""
        details = []
        predicates = []
        title = type(node).__name__
        if isinstance(node, PhysicalScan):
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
            written = filters.get(node.alias, ())
            order = [stage.written_pos for stage in node.stages]
            reordered = order != sorted(order)
            for index, stage in enumerate(node.stages, 1):
                label = f"predicate {stage.written_pos + 1}"
                if reordered:
                    label = f"{_ordinal(index)}: {label}"
                if stage.written_pos < len(written):
                    label += "  " + _short_prompt(
                        written[stage.written_pos].prompt)
                predicates.append((label, stage_cells(
                    stage.selectivity, stage.expected_docs,
                    stages.get(("filter", node.alias, stage.written_pos)),
                    "evaluated")))
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
            order = [stage.written_pos for stage in node.stages]
            reordered = order != sorted(order)
            for index, stage in enumerate(node.stages, 1):
                label = (f"join {stage.written_pos + 1} {stage.semantics} "
                         f"({stage.anchor}, {', '.join(stage.partners)})")
                if stage.pairs_from:
                    label += f" over pairs from {stage.pairs_from}"
                if reordered:
                    label = f"{_ordinal(index)}: {label}"
                if stage.written_pos < len(joins):
                    label += "  " + _short_prompt(
                        joins[stage.written_pos].prompt)
                predicates.append((label, stage_cells(
                    stage.selectivity, stage.expected_tuples,
                    stages.get(("join", stage.written_pos)), "tuples")))
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
                written = filters.get(spec.alias, ())
                for position in spec.written_positions:
                    predicate = (_short_prompt(written[position].prompt)
                                 if position < len(written)
                                 else f"predicate {position + 1}")
                    details.append(f"Filter {spec.alias}: {predicate}")
            for spec in node.joins:
                predicate = (_short_prompt(joins[spec.written_pos].prompt)
                             if spec.written_pos < len(joins)
                             else f"predicate {spec.written_pos + 1}")
                details.append(f"Join {spec.semantics}: {predicate}")
        else:
            title = node.type_name
            if not verbose:
                details.extend(_fields(node.explain_fields(), 0))
        return title, details, predicates

    def visit(node, depth):
        pad = "  " * depth
        title, details, predicates = describe(node)
        reference = shared.get(node.node_id)
        if node.node_id in visited:
            rows.append((f"{pad}Reuse [{reference}]", {}))
            return
        visited.add(node.node_id)
        if reference:
            title += f" [{reference}]"
        values = [estimates.get(PortRef(node.node_id, output.name))
                  for output in node.outputs
                  if not output.name.startswith("filter_answers:")]
        value = values[0] if len(values) == 1 else None
        estimate = estimates_by_node.get(node.node_id, {})
        cells = {}
        if value is not None:
            cells["est_rows"] = _number(value)
        if "seconds" in estimate:
            cells["est_time"] = _seconds(estimate["seconds"])
        if metrics is not None:
            if node.node_id in metrics:
                measured = metrics[node.node_id]
                cells["rows"] = f"{measured.output_rows:,}"
                cells["time"] = _seconds(measured.wall_s)
                if measured.fresh_tokens:
                    cells["tokens"] = f"{measured.fresh_tokens:,}"
                if verbose:
                    details.extend(_fields(vars(measured), 0))
            else:
                title += " (metrics unavailable)"
        if "release_recompute_tokens" in estimate:
            details.append(
                ("if the KV were released here instead of pinned: "
                 if node.pin_survivors else "expected recompute at the join: ")
                + f"{estimate['release_recompute_tokens']:,} tokens, "
                f"{estimate['release_recompute_seconds']:.3f} s")
        rows.append((pad + title, cells))
        rows.extend((pad + "  " + detail, {}) for detail in details)
        rows.extend((pad + "  " + label, stage) for label, stage in predicates)
        if verbose:
            rows.extend((line, {}) for line in _fields(
                {"node_id": node.node_id, "type": node.type_name,
                 **node.explain_fields()}, depth + 1))
            rows.extend((f"{pad}  {port.name} <- "
                         f"{port.source.node_id}[{port.source.port}]", {})
                        for port in node.inputs)
        children = dict.fromkeys(port.source.node_id for port in node.inputs)
        for child in children:
            visit(graph.node(child), depth + 1)
        for child in node.embedded_nodes():
            visit(child, depth + 1)

    visit(graph.node(graph.root.node_id), 0)
    for node in graph.topological_nodes():
        if node.node_id not in visited:
            visit(node, 0)
    lines = _table(rows)
    if used_default:
        lines.append(DEFAULT_NOTE)
    return "\n".join(lines)
