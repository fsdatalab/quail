"""AI.IF over two tables: the joins as the cost model prices them."""

from quail.cost.joins import PricedJoin
from quail.logical import (
    is_score,
    join_conditions,
)
from quail.logical.expressions import column_name
from quail.logical.prompts import prompt_aliases
from quail.physical import AiJoin, Foreign, PortRef
from quail.physical.base import input_ports
from quail.planner.operators.common import question_tokens
from quail.planner.operators.score import ScoreJoinCost, score_statistics
from quail.planner.prompt_lowering import LoweredPrompt


def _priced_key(condition) -> tuple:
    """Return one equality as (left alias, left key, right alias, right key)."""
    return (condition.left.alias, column_name(condition.left, qualified=False),
            condition.right.alias, column_name(condition.right, qualified=False))

def _join_aliases(join) -> list:
    """Return the join's table aliases in placeholder order."""
    return [r.alias for r in join.prompt.args]


def _label_counts(lowered: LoweredPrompt) -> dict:
    """Return alias -> (block_label_tokens, anchor_frame_tokens)."""
    return {a: (lt, nt) for a, lt, nt in lowered.labels}


def join_specs(joins, pair_fractions=None, canvas: int = 0, *,
               context=None, chunk: int = 0) -> list[PricedJoin]:
    """The joins as the cost model prices them, in written order.

    pair_fractions maps a written position to the fraction of the
    cross product its equality conditions keep; a join with
    conditions but no entry is priced as the full cross product.
    canvas is the rows a diffusion model appends to every pair.
    """
    pair_fractions = pair_fractions or {}
    out = []
    for i, j in enumerate(joins):
        conditions = join_conditions(j)
        fraction = pair_fractions.get(i, 1.0) if conditions else 1.0
        if is_score(j.predicate):
            parts, cost = score_statistics(j.prompt, context, chunk)
            left, right = prompt_aliases(j.prompt)
            head, middle, tail = map(len, parts)
            out.append(PricedJoin(
                written_position=i, aliases=(left, right), anchor=left,
                anchor_free=False, semantics=j.semantics,
                selectivity=j.selectivity,
                frame_tokens={left: head + middle, right: 0},
                label_tokens={left: 0, right: 0}, tail_tokens=tail + canvas,
                on=tuple(_priced_key(c) for c in conditions),
                pair_fraction=fraction,
                cost=ScoreJoinCost((left, right), cost, fraction)))
            continue
        lowered = context.lowered(j.prompt)
        labels = _label_counts(lowered)
        out.append(PricedJoin(
            written_position=i, aliases=tuple(_join_aliases(j)), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=question_tokens(lowered, canvas),
            on=tuple(_priced_key(c) for c in conditions),
            pair_fraction=fraction))
    return out


def emit_join(
    builder, spec, step, anchor, join_id, exec_index, port_by_alias
) -> tuple:
    """Append one AI join stage and the functions that supply its pairs.

    Args:
        builder: The graph builder.
        spec: The join's PricedJoin.
        step: The join's step in the chosen sequence.
        anchor: The anchor alias.
        join_id: The AiJoin node id.
        exec_index: The join's position in execution order.
        port_by_alias: alias -> the port its documents stand at.

    Returns:
        The anchor's ports after the join, and the port of the pairs a
        full join keeps, else None.
    """
    pair_inputs = []
    partners = [a for a in spec.aliases if a != anchor]
    pairs_from = ""
    if spec.written_position in builder.pairs_src:
        producer = builder.pairs_src[spec.written_position]
        pair_inputs.append(
            PortRef(producer.node_id, f"pairs:{spec.written_position}"))
        pairs_from = producer.node_id
    for apply in builder.join_applies.get(spec.written_position, ()):
        # the function's pairs reach the join on their own port
        apply_id = f"apply:{apply.function}"
        builder.nodes.append(Foreign(
            node_id=apply_id,
            inputs=input_ports(tuple(
                port_by_alias[a] for a in apply.aliases)),
            function=apply.function, kind=apply.kind, ids="pairs",
            columns=apply.columns, aliases=tuple(apply.aliases),
            written_position=spec.written_position))
        pair_inputs.append(PortRef(apply_id, f"pairs:{spec.written_position}"))
        pairs_from = apply_id
    builder.nodes.append(AiJoin(
        node_id=join_id,
        inputs=input_ports(tuple(
            port_by_alias[a] for a in [anchor, *partners])
                           + tuple(pair_inputs)),
        written_position=spec.written_position, exec_index=exec_index,
        anchor=anchor, partners=tuple(partners),
        semantics=spec.semantics,
        selectivity=spec.selectivity,
        expected_rows=round(step.tuples, 1),
        anchor_documents=round(step.anchor_documents, 1),
        anchor_frame_tokens=spec.frame_tokens[anchor],
        pair_tail_tokens=spec.tail_tokens,
        document_head=tuple(builder.context.lowered(
            builder.operators.joins[spec.written_position].prompt).preamble_token_ids),
        anchor_resident=step.resident,
        tuple_tokens=round(step.tokens, 1),
        pairs_from=pairs_from))
    return (PortRef(join_id, f"rows:{anchor}"),
            PortRef(join_id, f"join_answers:{spec.written_position}")
            if spec.semantics == "full" else None)
