"""Execute join stages and collect their partner answers."""

from quail.backends.quail.executor.parts import (
    StagesPart,
    chunks,
    complete_chain,
    execution_prefix_tree,
    gated_stages,
    gpu_seconds,
    input_staging,
    report_chain_transitions,
    require_execution_state,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.graph import stage_partner_lists
from quail.execution.runner import NodeMetrics, NodeResult
from quail.execution.tokens import DocumentKeys


def execute_join(state, node, inputs) -> NodeResult:
    """Run a join with any preceding and following pipeline parts."""
    require_execution_state(state)
    torch = state["torch"]
    arena = state["arena"]
    pipeline = state["pipeline"]
    async_answers = state["async_answers"]
    chunk_tokens = state["chunk_tokens"]

    stage_frames = inputs["stage_frames"]
    chain = inputs.get("chain")
    lists_for = inputs.get("anchor_partners")
    join_stages = [
        Stage(suffixes=suffixes, readout=async_answers, frame=frame,
              requests=(None if lists_for is None
                        else (lambda key, j=j: lists_for(key[1])[j])),
              label=f"join stage {j}")
        for j, (suffixes, frame) in enumerate(
            zip(inputs["stage_suffixes"], stage_frames))
    ]
    join_stats = {}
    leading = 0
    parts = []
    after = []
    if chain is not None:
        # the chain's stages lead the join's: a survivor goes on to
        # the join with its KV resident, past the chain's gates;
        # a classification of the rows kept follows the join's
        filter_ids = chain["document_ids"]
        parts = chain["parts"]
        after = chain.get("after", [])
        join_part = StagesPart(join_stages)
        stages = gated_stages(parts + [join_part] + after)
        leading = len(stages) - len(join_stages) - sum(
            len(part.stages) for part in after)
        root = parts[0].node if parts else node
        prefixes = chain["documents"]
        anchor_keys = DocumentKeys(node.anchor, filter_ids)
        tree = execution_prefix_tree(root, prefixes, arena)
        attention = (node.attention or getattr(root, "attention", None)
                     or None)
    else:
        stages = join_stages
        prefixes = inputs["prefixes"]
        anchor_keys = inputs["anchor_keys"]
        tree = execution_prefix_tree(node, inputs["prefixes"], arena)
        attention = node.attention or None
    anchor_done = inputs["anchor_done"]
    settled = set()
    join_rows = {}
    if after:
        # the join's last stage records each anchor's kept partners
        # for the parts after it, and its row settles the anchor
        last = len(join_stages) - 1

        def kept(a, row, stage=join_stages[last]):
            indices = (None if lists_for is None
                       else lists_for(anchor_keys[a][1])[last])
            join_rows[a] = row
            for part in after:
                part.kept.record(a, row, indices)
            return bool(any(row))

        join_stages[last].decide = kept

    def on_settled(anchor, survived, row):
        settled.add(anchor)
        anchor_done(anchor, join_rows.get(anchor, row) if after else row)

    def on_chunk(transitions):
        report_chain_transitions(parts, transitions)

    every, spans, tokens = run_stages(
        torch, arena, pipeline, stages, prefixes, chunk_tokens,
        anchor_keys=anchor_keys, on_settled=on_settled,
        attention_mode=attention, prefix_tree=tree, stats=join_stats,
        on_chunk=on_chunk, label=f"join ({len(join_stages)} stages)",
        staging=input_staging(state))
    answers = every[leading:leading + len(join_stages)]
    after_tokens = 0
    if after:
        low = leading + len(join_stages)
        for part in after:
            high = low + len(part.stages)
            own = part.finish(every[low:high]) or 0
            part_spans = stage_spans(spans, low, high)
            part.result_value = part.result(
                own, gpu_seconds(torch, part_spans, inputs),
                chunks(part_spans, inputs), {})
            after_tokens += own
            low = high
        spans = stage_spans(spans, 0, leading + len(join_stages))
        tokens -= after_tokens
    if chain is not None:
        # the join packs frames and partner suffixes; the rest is the chain's
        join_tokens = _streamed_tokens(join_stages, answers, lists_for,
                                       anchor_keys)
        # the join's anchors are the documents that reached it, in
        # the chain's order: those with a first-stage row and those
        # that settled there with no partner to ask; their rows are
        # re-keyed to that order
        reached = sorted(set(answers[0] if answers else ()) | settled)
        local_of = {position: local for local, position in enumerate(reached)}
        answers = [{local_of[position]: row for position, row in stage.items()}
                   for stage in answers]
        anchor_ids = [filter_ids[position] for position in reached]
        if parts:
            kv_round = {"hits": len(reached), "misses": 0}
            complete_chain(parts, every[:leading], spans,
                           tokens - join_tokens, join_stats, torch, inputs)
            for part in parts:
                if part.result_value is None:
                    raise RuntimeError(
                        "a pipeline part finished without a result")
            join_stats = {}
            spans = stage_spans(spans, leading, len(stages))
            tokens = join_tokens
        else:
            # the join led the pipeline and packed the anchors itself
            kv_round = {"hits": 0, "misses": len(reached)}
    else:
        anchor_ids = list(inputs["anchor_ids"])
        kv_round = inputs.get("kv_round") or {}
    group = inputs["group"]
    last = answers[-1] if answers else {}
    matched = {
        anchor_ids[int(local)]
        for local, row in last.items() if any(row)
    }
    if group[-1]["semantics"] == "anti":
        survivors = [document for document in anchor_ids
                     if document not in matched]
    else:
        survivors = [document for document in anchor_ids
                     if document in matched]
    outputs = {f"ids:{node.anchor}": survivors}
    partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
    for stage, stage_answers, members in zip(
            node.stages, answers, partner_lists):
        outputs[f"join_answers:{stage.written_pos}"] = {
            "rows": stage_answers,
            "anchor_index": anchor_ids,
            "partner_index": inputs["partner_indices"][
                stage.written_pos
            ],
            "anchor_partners": members,
            "anchor": node.anchor,
            "partners": list(stage.partners),
            "semantics": stage.semantics,
            "selectivity": stage.selectivity,
            "written_pos": stage.written_pos,
        }
    return NodeResult(
        outputs=outputs,
        metrics=NodeMetrics(
            input_rows=len(anchor_ids),
            output_rows=len(survivors),
            evaluated_document_pairs=sum(
                sum(len(row) for row in stage.values())
                for stage in answers
            ),
            kv_hits=kv_round.get("hits", 0),
            kv_misses=kv_round.get("misses", 0),
            fresh_tokens=tokens,
            gpu_s=gpu_seconds(torch, spans, inputs),
            chunks=chunks(spans, inputs),
            extension={
                "answers": answers,
                **({"borrowed_prefix_tokens": join_stats["borrowed_tokens"]}
                   if join_stats.get("borrowed_tokens") else {}),
                **({"pack_s": round(join_stats["pack_s"], 3)}
                   if join_stats.get("pack_s") else {}),
            },
        ),
    )


def _streamed_tokens(join_stages, answers, lists_for, anchor_keys) -> int:
    """Tokens the join's stages packed: frames and partner suffixes."""
    total = 0
    previous = None
    for j, (stage, rows) in enumerate(zip(join_stages, answers)):
        frame = list(stage.frame)
        writes = bool(frame) and frame != previous
        previous = frame
        lengths = [len(suffix) for suffix in stage.suffixes]
        for anchor in rows:
            indices = (None if lists_for is None
                       else lists_for(anchor_keys[anchor][1])[j])
            total += (len(frame) if writes else 0) + (
                sum(lengths) if indices is None
                else sum(lengths[i] for i in indices))
    return total
