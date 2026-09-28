"""Physical planning for AI.CLASSIFY queries on one table.

A plan scans the table, then for each label filter in written order
classifies the documents still alive and keeps those with an accepted
label, then classifies what the projection still needs. Each document
is scored the way the join path scores a pair: its prompt head and
document stay in KV, the question and category list are written once
after it, and one short suffix per label-trie node reads the next
token's log probabilities.
"""

from quail.cost import budgets
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, scan, stream
from quail.execution.labels import label_trie
from quail.logical import (
    Alias,
    Apply,
    LabelIn,
    effective_selectivity,
    label_text,
    true_false_token_ids,
)
from quail.physical import (
    AiClassify,
    ClassifySpec,
    LabelFilter,
    Limit,
    PortRef,
    Project,
    Scan,
)
from quail.physical.base import input_ports
from quail.planner.physical_optimizer import PhysicalCandidate
from quail.planner.plan import PhysicalPlan, Refusal


def _refused(reason: str, constraint: str = "unsupported_classify_query",
             needed: int = 1, available: int = 0, unit: str = "queries"):
    refusal = Refusal(reasons=(reason,), constraint=constraint,
                      needed=needed, available=available, unit=unit)
    return (PhysicalCandidate(None, refusal, float("inf")),)


def has_label(logical) -> bool:
    """Return whether a logical plan classifies or tests a label."""
    operators = logical.operators()
    return any(
        isinstance(predicate.expression, LabelIn)
        for predicates in operators.filters.values()
        for predicate in predicates
    ) or any(
        isinstance(column, Alias) and column.expression.kind == "label"
        for column in logical.root.columns
    )


def suffix_tokens(trie) -> int:
    """Tokens of every node's suffix: the answer cue's last token and the prefix."""
    return sum(1 + len(prefix) for prefix in trie)


def classify_work(documents: float, mean_tokens: float, head_tokens: int,
                  frame_tokens: int, trie) -> Work:
    """Work to classify documents of a mean length.

    Each document computes its head, document, and frame once, then
    streams one suffix per trie node against them.
    """
    prefix = head_tokens + mean_tokens + frame_tokens
    suffixes = [1 + len(node) for node in trie]
    per_document = scan(prefix, 0) + stream(prefix, suffixes)
    return per_document * documents


def plan_classify(region, context, *, backend_name: str):
    """Build one Quail plan for a table's AI.CLASSIFY columns and label filters."""
    logical = region.logical_plan
    model = context.model
    if any(isinstance(node, Apply) for node in logical.walk()):
        return _refused("AI.CLASSIFY cannot be mixed with apply() yet")
    operators = logical.operators()
    if operators.joins or len(operators.scans) != 1:
        return _refused("AI.CLASSIFY runs on one table without joins for now")
    (source,) = operators.scans
    alias = source.alias
    predicates = operators.filters.get(alias, ())
    if not all(isinstance(p.expression, LabelIn) for p in predicates):
        return _refused(
            "AI.CLASSIFY cannot be mixed with AI.IF or AI.SCORE yet")
    if model.canvas_tokens:
        return _refused(
            f"AI.CLASSIFY is not supported on {model.name!r} yet: it reads "
            f"answers from a canvas")
    if not model.tied_head:
        return _refused(
            f"AI.CLASSIFY needs the full output head, which Quail keeps "
            f"only for models whose head is tied to the embedding; "
            f"{model.name!r} has a separate head")
    if context.tokenizer is None:
        return _refused("AI.CLASSIFY planning needs the model's tokenizer")

    lengths = context.document_tokens[alias]
    count = len(lengths)
    total = sum(int(length) for length in lengths)
    mean = total / max(1, count)
    longest = max((int(length) for length in lengths), default=0)
    chunk = budgets.chunk_budget(model, context.device)
    capacity = budgets.arena_tokens(model, context.device, chunk)

    nodes = [Scan(node_id=f"scan:{alias}", alias=alias, input_id=alias,
                  n_docs=count, total_tokens=total, shard_ranges=((0, count),),
                  shard_token_loads=(total,))]
    current = PortRef(nodes[0].node_id, f"ids:{alias}")
    projected = {
        column.expression: column.name for column in logical.root.columns
        if isinstance(column, Alias)
    }
    named = {}      # ModelCall -> output column name
    work = Work()
    seconds = 0.0
    live = float(count)

    def classify(call, name):
        nonlocal current, work, seconds
        tokenizer = context.tokenizer
        head = tuple(tokenizer(call.prompt.preamble))
        tail = tuple(call.prompt.tail_token_ids)
        labels = tuple(tuple(tokenizer(label_text(label)))
                       for label in call.labels)
        trie = label_trie(labels)
        # head, document, and frame stay resident while the longest
        # suffix and the frame entry's own rows are packed beside them
        need = len(head) + longest + len(tail) + max(1 + len(n) for n in trie)
        if need > min(chunk, capacity):
            raise _RefusedError(
                f"a document in {alias!r} needs {need} tokens with its "
                f"classification prompt, but the forward pass budget is "
                f"{min(chunk, capacity)} tokens", need, min(chunk, capacity))
        step = classify_work(live, mean, len(head), len(tail) - 1, trie)
        estimate = speed_of_light(step, model, context.device, chunk).seconds
        spec = ClassifySpec(
            name=name, aliases=(alias,), query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=live, estimated_seconds=estimate,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels,
        )
        node = AiClassify(node_id=f"ai-classify:{len(named)}",
                          inputs=input_ports((current,)),
                          backend_name=backend_name, model=model.name,
                          spec=spec)
        nodes.append(node)
        current = PortRef(node.node_id, "scores")
        named[call] = name
        work += step
        seconds += estimate

    try:
        for written_pos, predicate in enumerate(predicates):
            test = predicate.expression
            if test.call not in named:
                classify(test.call, projected.get(test.call)
                         or test.name or f"__label_{alias}_{written_pos}")
            node = LabelFilter(
                node_id=f"label-filter:{alias}:{written_pos}",
                inputs=input_ports((current,)), score_name=named[test.call],
                aliases=(alias,), comparison="in", threshold=0.0,
                selectivity=predicate.selectivity, written_pos=written_pos,
                accepted=tuple(test.accepted))
            nodes.append(node)
            current = PortRef(node.node_id, "scores")
            live *= effective_selectivity(predicate.selectivity)
        for call, name in projected.items():
            if call not in named:
                classify(call, name)
    except _RefusedError as refused:
        return _refused(refused.reason, "suffix_over_chunk", refused.needed,
                        refused.available, "tokens")

    columns = tuple(
        column.name if isinstance(column, Alias)
        else f"{column.alias}.{column.column}"
        for column in logical.root.columns)
    nodes.append(Project(node_id="project", inputs=input_ports((current,)),
                         columns=columns))
    if logical.root.limit is not None:
        nodes.append(Limit(node_id="limit",
                           inputs=input_ports((PortRef("project", "rows"),)),
                           count=logical.root.limit))
    # the GPU boots with the TRUE/FALSE rows every Quail query retains,
    # so a loaded model serves filters and classifications alike
    true_ids, false_ids = true_false_token_ids(context.tokenizer)
    plan = PhysicalPlan(
        model=model.name,
        device=context.device.name,
        workers=context.gpu_count,
        backend=backend_name,
        estimated_seconds=seconds,
        nodes=tuple(nodes),
        settings={
            "chunk_tokens": chunk,
            "true_ids": true_ids,
            "false_ids": false_ids,
            "retained_kv_tokens": capacity,
            "prefix_reuse": "document KV shared by every label-trie node",
            "survivor_assumption": "uniform independent selection",
            "estimated_fresh_tokens": work.tokens,
            "estimated_attention_pairs": work.pairs,
            "batching": "token_based_admission",
            "data_parallel_copies": context.gpu_count,
            "label_scoring": "exhaustive label trie, full-vocabulary log "
                             "probabilities",
            "order_rule": "written order",
        },
    )
    return (PhysicalCandidate(plan.graph, plan, seconds),)


class _RefusedError(Exception):
    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available
