"""Plan an AI.RANK filter as one AiTopK node."""

from dataclasses import replace

from quail.cost.rank import (
    NUM_PIVOTS,
    RANK_MARGIN,
    RANK_SEED,
    expected_comparisons,
    kept_rows,
    rank_seconds,
)
from quail.logical.expressions import column_name, column_references
from quail.physical import PhysicalGraph, PortRef, Project, Window
from quail.physical.base import input_ports
from quail.physical.nodes.rank import AiTopK, RankSpec
from quail.planner.prompt_lowering import rank_token_parts
from quail.planner.statistics.tables import summarize_alias


def plan_ai_top_k(graph, context) -> PhysicalGraph | None:
    """Replace a filtered AI.RANK window with an AiTopK node.

    The node reads the scan rows, before projection, and writes them
    back in rank order. Later sort and limit nodes read the projection.
    A second pass finds no rank window and leaves the graph as it is.

    Args:
        graph: The physical graph, with the rank still as a window.
        context: The planning context. Its logical plan and tokenizer
            are set.

    Returns:
        The rewritten graph, or None when the graph has no AI.RANK.
    """
    window = next((
        node for node in graph.nodes
        if isinstance(node, Window)
        and any(call.function == "ai_rank" for call in node.calls)), None)
    if window is None or context is None or context.logical_plan is None:
        return None
    call = next(call for call in window.calls if call.function == "ai_rank")
    project = graph.node(window.inputs[0].source.node_id)
    if not isinstance(project, Project):
        return None
    source = project.inputs[0].source
    spec = build_rank_spec(call, context, graph.node(source.node_id))
    top = AiTopK(
        node_id="ai-top-k",
        inputs=input_ports((source,)),
        backend_name=context.backend,
        model="" if context.model is None else context.model.name,
        spec=spec)
    rewritten = replace(
        project,
        inputs=input_ports((PortRef(top.node_id, "rows"),)),
        columns=window.columns)
    nodes = []
    for node in graph.nodes:
        if node is window:
            continue
        if node is project:
            nodes.append(top)
            nodes.append(rewritten)
            continue
        inputs = tuple(
            replace(port, source=PortRef(project.node_id, "rows"))
            if port.source.node_id == window.node_id else port
            for port in node.inputs)
        nodes.append(node if inputs == node.inputs
                     else replace(node, inputs=inputs))
    root = graph.root
    if root.node_id == window.node_id:
        root = PortRef(project.node_id, "rows")
    return PhysicalGraph(tuple(nodes), root)


def build_rank_spec(call, context, scan) -> RankSpec:
    """Bind one AI.RANK window to the comparison spec AiTopK runs.

    Args:
        call: The AI.RANK window call.
        context: The planning context.
        scan: The physical scan the rows come from.

    Returns:
        The rank spec, including the comparison estimate.
    """
    ref = call.prompt.args[0]
    alias = ref.alias
    logical = context.logical_plan
    logical_scan = next(
        item for item in logical.operators().scans if item.alias == alias)
    provider = context.catalog.get(logical_scan.provider)
    partition = tuple(
        _source(logical, name).column for name in call.partition)
    num_rows = (float(scan.num_documents) if scan.expected_rows is None
                else float(scan.expected_rows))
    k = kept_rows(call.rank_comparison, call.rank_k)
    parts = rank_token_parts(call.prompt, context.model, context.tokenizer)
    mean, variance = _length_stats(context, alias)
    seconds = rank_seconds(
        num_rows, k, tuple(len(part) for part in parts), mean, variance,
        context.model, context.device, num_pivots=NUM_PIVOTS)
    return RankSpec(
        name=call.name,
        aliases=(alias, alias),
        query_template=call.prompt.template,
        arguments=((alias, column_name(ref, qualified=False)),) * 2,
        expected_rows=min(float(k), num_rows),
        estimated_seconds=seconds,
        prompt_token_parts=parts,
        k=k,
        comparison=call.rank_comparison,
        bound=call.rank_k,
        partition=partition,
        num_pivots=NUM_PIVOTS,
        margin=RANK_MARGIN,
        seed=RANK_SEED,
        id_column=provider.id_col,
        expected_comparisons=expected_comparisons(num_rows, k, NUM_PIVOTS))


def _source(plan, name):
    """Return the one source column a rank partition name projects."""
    for column in plan.projection.columns:
        if column_name(column) != name:
            continue
        refs = column_references(column)
        if len(refs) == 1:
            return refs[0]
    raise ValueError(f"AI.RANK partition {name!r} is not one source column")


def _length_stats(context, alias) -> tuple[float, float]:
    lengths = context.document_tokens.get(alias)
    if lengths is None or context.model is None:
        return 0.0, 0.0
    summary = summarize_alias(
        lengths, context.model.sliding_window, memo=context.memo)
    return summary.mean, summary.variance
