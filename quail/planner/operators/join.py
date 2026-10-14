"""AI.IF over two tables: the joins as the cost model prices them."""

from quail.cost.joins import PricedJoin
from quail.logical import (
    is_score,
    join_conditions,
)
from quail.logical.prompts import prompt_aliases
from quail.physical import AiJoin, Foreign, PortRef
from quail.physical.base import input_ports
from quail.planner.operators.common import question_tokens
from quail.planner.operators.score import ScoreJoinCost, score_statistics
from quail.planner.prompt_lowering import LoweredPrompt


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
                written_pos=i, aliases=(left, right), anchor=left,
                anchor_free=False, semantics=j.semantics,
                selectivity=j.selectivity,
                frame_tokens={left: head + middle, right: 0},
                label_tokens={left: 0, right: 0}, tail_tokens=tail + canvas,
                on=tuple((c.left.alias, c.left.column, c.right.alias,
                          c.right.column) for c in conditions),
                pair_fraction=fraction,
                cost=ScoreJoinCost((left, right), cost, fraction)))
            continue
        lowered = context.lowered(j.prompt)
        labels = _label_counts(lowered)
        out.append(PricedJoin(
            written_pos=i, aliases=tuple(_join_aliases(j)), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=question_tokens(lowered, canvas),
            on=tuple((c.left.alias, c.left.column, c.right.alias,
                      c.right.column) for c in conditions),
            pair_fraction=fraction))
    return out


def emit_join(builder, spec, step, anchor, jid, exec_idx):
    """Append one AI join stage and the functions that supply its pairs."""
    pair_inputs = []
    partners = [a for a in spec.aliases if a != anchor]
    pairs_from = ""
    if spec.written_pos in builder.pairs_src:
        producer = builder.pairs_src[spec.written_pos]
        pair_inputs.append(
            PortRef(producer.node_id, f"pairs:{spec.written_pos}"))
        pairs_from = producer.node_id
    for apply in builder.join_applies.get(spec.written_pos, ()):
        # the function's pairs reach the join on their own port
        aid = f"apply:{apply.function}"
        builder.nodes.append(Foreign(
            node_id=aid,
            inputs=input_ports(tuple(
                builder.ids_src[a] for a in apply.aliases)),
            function=apply.function, kind=apply.kind, ids="pairs",
            columns=tuple((ref.alias, ref.column)
                          for ref in apply.columns),
            aliases=tuple(apply.aliases),
            written_pos=spec.written_pos))
        pair_inputs.append(PortRef(aid, f"pairs:{spec.written_pos}"))
        pairs_from = aid
    builder.nodes.append(AiJoin(
        node_id=jid,
        inputs=input_ports(tuple(
            builder.ids_src[a] for a in [anchor, *partners])
                           + tuple(pair_inputs)),
        written_pos=spec.written_pos, exec_idx=exec_idx,
        anchor=anchor, partners=tuple(partners),
        semantics=spec.semantics,
        selectivity=spec.selectivity,
        expected_tuples=round(step.tuples, 1),
        anchor_frame_tokens=spec.frame_tokens[anchor],
        pair_tail_tokens=spec.tail_tokens,
        anchor_resident=step.resident,
        tuple_tokens=round(step.tokens, 1),
        pairs_from=pairs_from))
    builder.ids_src[anchor] = PortRef(jid, f"ids:{anchor}")
    if spec.semantics == "full":
        builder.pairs_edges.append(PortRef(
            jid, f"join_answers:{spec.written_pos}"
        ))
        for a in [anchor] + partners:
            if a not in builder.out_aliases:
                builder.out_aliases.append(a)
