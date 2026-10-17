"""Query tree transformations used by optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    Aggregate,
    Alias,
    Apply,
    ColumnRef,
    Equality,
    Filter,
    Join,
    LogicalNode,
    LogicalPlan,
    ModelCall,
    RegularPredicate,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    SortKey,
    is_score,
    model_call,
)
from quail.logical.expressions import ColumnExpression


def _column_refs(expression) -> tuple[ColumnRef, ...]:
    """Return source columns whose values an expression reads."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, ColumnExpression):
        return (expression.reference,)
    if isinstance(expression, Alias):
        return _column_refs(expression.expression)
    if isinstance(expression, ModelCall) and expression.kind == "extract":
        # Extractions copy answer strings from the original document text.
        return tuple(ref for arg in expression.prompt.args for ref in _column_refs(arg))
    if isinstance(expression, RegularPredicate):
        return (expression.column,)
    if isinstance(expression, Equality):
        return (*_column_refs(expression.left), *_column_refs(expression.right))
    if isinstance(expression, SortKey):
        return _column_refs(expression.expression)
    return ()


def push_down_projection(root: LogicalNode) -> LogicalNode:
    """Rewrite every Scan to keep only the column values the plan reads.

    One pass from the root toward the scans. Each node on the way adds
    the columns its expressions read to the set carried down, in first
    use order; a Scan keeps the columns collected for its alias. Every
    node that reads an alias's column is an ancestor of that alias's
    Scan, so the path holds the complete set.

    A source column is retained as values when a projection, predicate,
    join, callback, or extraction reads it. Other model inputs need only
    their tokens.
    """
    def descend(node, needed):
        needed = {alias: dict(columns) for alias, columns in needed.items()}
        expressions = node.expressions()
        for expression in expressions:
            for ref in _column_refs(expression):
                needed.setdefault(ref.alias, {})[ref.column] = None
        if isinstance(node, (SemanticClassify, SemanticExtract)):
            # the columns a classification or extraction adds are not
            # read below it
            for field in node.output_schema()[len(node.input.output_schema()):]:
                needed.get(field.alias, {}).pop(field.column, None)
        if isinstance(node, Scan):
            columns = tuple(needed.get(node.alias, ()))
            return node if columns == node.columns else replace(
                node, columns=columns)
        children = tuple(descend(child, needed) for child in node.children())
        if children == node.children():
            return node
        return node.with_children(children)

    return descend(root, {})


def _lets_through(node, alias: str) -> bool:
    """Return whether a filter on one table may move below node.

    True for a Join, a full SemanticJoin, a classification of joined
    rows, and any node on another table. False for the table's own
    sequence, a gate (an exists or anti join), a callback, and any other node.
    """
    if isinstance(node, Join):
        return True
    if isinstance(node, SemanticJoin):
        return node.semantics == "full"
    if isinstance(node, SemanticClassify):
        return len(node.call.aliases()) == 2 or node.alias != alias
    if isinstance(node, SemanticExtract):
        return node.alias != alias
    if isinstance(node, Filter):
        return node.condition.column.alias != alias
    if isinstance(node, SemanticFilter):
        return all(model_call(p.expression).aliases()[0] != alias
                   for p in node.predicates)
    return False


def _place_filter(filter_node, node, alias: str):
    """Put a one-table filter below the joins above its table.

    Returns:
        The subtree under node with the filter as low as it goes, or
        None when a gate, a callback, or the table's own
        sequence stops it before any Join.
    """
    if isinstance(node, Join):
        left = {field.alias for field in node.left.output_schema()}
        side = "left" if alias in left else "right"
        child = getattr(node, side)
        placed = _place_filter(filter_node, child, alias)
        if placed is None:
            placed = filter_node.with_children((child,))
        return replace(node, **{side: placed})
    if _lets_through(node, alias):
        (child,) = node.children()
        placed = _place_filter(filter_node, child, alias)
        return None if placed is None else node.with_children((placed,))
    return None


def push_down_filters(root: LogicalNode) -> LogicalNode | None:
    """Move every cheap one-table filter below the joins above its table.

    A Filter on a label column tests each row independently, so it
    commutes with the full joins, the classifications of
    joined rows, and the other tables' operators above that table.
    Each moves to the top of its own table's sequence and keeps its order
    with the table's other filters. A filter stays where it is when a
    gate (an exists or anti join) or a callback
    sits between it and the first Join below.

    Returns:
        The rewritten root, or None when no filter moved.
    """
    moved = False

    def visit(node):
        nonlocal moved
        children = tuple(visit(child) for child in node.children())
        if children != node.children():
            node = node.with_children(children)
        if not isinstance(node, Filter):
            return node
        alias = node.condition.column.alias
        placed = _place_filter(node, node.input, alias)
        if placed is None:
            return node
        moved = True
        return placed

    rewritten = visit(root)
    return rewritten if moved else None


def _label_sequence(node) -> tuple:
    """Split a table's sequence at its classifications and extractions.

    Returns:
        The node below the table's lowest SemanticClassify or
        SemanticExtract, and the SemanticClassify, label filter, and
        SemanticExtract nodes above it, lowest first.
    """
    lifted = []
    while isinstance(node, (SemanticClassify, SemanticExtract, Filter)):
        lifted.append(node)
        node = node.input
    return node, tuple(reversed(lifted))


def _sequence_alias(node) -> str:
    """Return the table alias a classification, extraction, or label filter reads."""
    if isinstance(node, (SemanticClassify, SemanticExtract)):
        return node.alias
    return node.condition.column.alias


def lift_maps(root: LogicalNode) -> LogicalNode | None:
    """Move every joined table's classifications and extractions above the joins.

    Each one-table SemanticClassify of a table that a SemanticJoin
    reads, with the Filters on its label column, and each
    SemanticExtract of such a table, leaves the table's sequence and
    moves above the topmost join, under the root Project, in scan
    order. A classification or extraction of a table that no join
    reads stays. The map operators here are SemanticClassify and
    SemanticExtract. The planner prices the rewritten plan alongside
    the original plan. A callback keeps the model work below it in place.
    Filters on an exists or anti join's inner table stay before that join.

    Returns:
        The rewritten root, or None when no classification or extraction moved.
    """
    if isinstance(root, (Result, Aggregate)):
        rewritten = lift_maps(root.input)
        return None if rewritten is None else root.with_children((rewritten,))
    nodes = LogicalPlan(root).walk()
    fixed = {node.alias for apply in nodes if isinstance(apply, Apply)
             for node in LogicalPlan(apply.input).walk()
             if isinstance(node, (SemanticClassify, SemanticExtract))}
    fixed.update(alias for node in nodes if isinstance(node, SemanticJoin)
                 and node.semantics != "full"
                 for alias in model_call(node.predicate).aliases()
                 if alias != node.anchor)
    # Placement is chosen for a table's complete model sequence.
    joined = {alias for node in nodes
              if isinstance(node, SemanticJoin)
              for alias in model_call(node.predicate).aliases()} - fixed
    sequences = {}

    def cut(node):
        base, sequence = _label_sequence(node)
        if sequence and _sequence_alias(sequence[0]) in joined:
            sequences.setdefault(_sequence_alias(sequence[0]), []).extend(sequence)
            return base
        return node

    def visit(node):
        if isinstance(node, Join):
            return replace(node, left=visit(node.left), right=cut(node.right))
        if isinstance(node, SemanticJoin) or (
                isinstance(node, SemanticClassify)
                and len(node.call.aliases()) == 2):
            return node.with_children((visit(node.input),))
        return cut(node)

    top, lifted = _label_sequence(root.input)
    for node in lifted:
        sequences.setdefault(_sequence_alias(node), []).append(node)
    spine = visit(top)
    if not sequences:
        return None
    for scan in LogicalPlan(root).walk():
        if isinstance(scan, Scan):
            for node in sequences.get(scan.alias, ()):
                spine = node.with_children((spine,))
    lifted = root.with_children((spine,))
    return None if lifted == root else lifted


def _filtered_alias(node: LogicalNode) -> str | None:
    """Return the scan alias beneath a chain containing only AI.IF filters."""
    if not isinstance(node, SemanticFilter):
        return None
    while isinstance(node, SemanticFilter):
        if any(is_score(predicate.expression) for predicate in node.predicates):
            return None
        node = node.input
    return node.alias if isinstance(node, Scan) else None


def _stop_key(columns, alias) -> tuple[str, ...] | None:
    """Return the alias's columns the projection returns, or None."""
    key = []
    for column in columns:
        if not isinstance(column, ColumnRef) or column.alias != alias:
            return None
        if column.column not in key:
            key.append(column.column)
    return tuple(key)
