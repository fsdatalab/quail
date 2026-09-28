"""Physical planning for AI.CLASSIFY queries on one table.

A plan scans the table, then for each label filter in written order
classifies the documents still alive and keeps those with an accepted
label, then classifies what the projection still needs. Each document
is scored the way the join path scores a pair: its prompt head and
document stay in KV, the question and category list are written once
after it, and one short suffix per label-trie node reads the next
token's log probabilities.
"""

from dataclasses import dataclass

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


# The label scoring rules the executor runs, in the order they were
# added; each later rule must return the labels of the first.
LABEL_SCORINGS = ("trie_nodes", "label_chains")
DEFAULT_LABEL_SCORING = "label_chains"


def suffix_lengths(scoring: str, labels) -> list[int]:
    """Tokens of each suffix a document streams under one scoring rule.

    Every suffix starts with the answer cue's last token. Under
    ``trie_nodes`` it continues with a label-trie node's tokens and
    only its last row is read. Under ``label_chains`` it continues
    with all but a label's last token and every row is read.
    """
    if scoring == "label_chains":
        return [len(ids) for ids in labels]
    return [1 + len(node) for node in label_trie(labels)]


def classify_work(documents: float, mean_tokens: float, head_tokens: int,
                  frame_tokens: int, suffixes) -> Work:
    """Work to classify documents of a mean length.

    Each document computes its head, document, and frame once, then
    streams the suffixes against them.
    """
    prefix = head_tokens + mean_tokens + frame_tokens
    per_document = scan(prefix, 0) + stream(prefix, suffixes)
    return per_document * documents


@dataclass(frozen=True)
class _Table:
    """The one table a classification plan scans, and the budgets it plans under.

    Attributes:
        alias: The table's alias in the query.
        mean: Mean document length in tokens.
        longest: Longest document in tokens.
        budget: Tokens one forward pass may hold: the smaller of the
            chunk budget and the KV arena.
        chunk: The chunk budget in tokens.
        scoring: The label scoring rule every classification uses.
    """

    alias: str
    mean: float
    longest: int
    budget: int
    chunk: int
    scoring: str
    backend_name: str
    model: object
    device: object
    tokenizer: object

    def classify(self, call, name, input_port, live, index):
        """Return an AiClassify node for one prompt and the work it does.

        Args:
            call: The logical AI.CLASSIFY call.
            name: The output column.
            input_port: The port carrying the documents still alive.
            live: How many documents are expected to reach it.
            index: The node's position among the plan's classifications.

        Raises:
            _RefusedError: A document and its prompt exceed the budget.
        """
        head = tuple(self.tokenizer(call.prompt.preamble))
        tail = tuple(call.prompt.tail_token_ids)
        labels = tuple(tuple(self.tokenizer(label_text(label)))
                       for label in call.labels)
        suffixes = suffix_lengths(self.scoring, labels)
        # head, document, and frame stay resident while the longest
        # suffix and the frame entry's own rows are packed beside them
        need = len(head) + self.longest + len(tail) + max(suffixes)
        if need > self.budget:
            raise _RefusedError(
                f"a document in {self.alias!r} needs {need} tokens with its "
                f"classification prompt, but the forward pass budget is "
                f"{self.budget} tokens", need, self.budget)
        step = classify_work(live, self.mean, len(head), len(tail) - 1,
                             suffixes)
        estimate = speed_of_light(step, self.model, self.device,
                                  self.chunk).seconds
        spec = ClassifySpec(
            name=name, aliases=(self.alias,),
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=live, estimated_seconds=estimate,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels, scoring=self.scoring,
        )
        node = AiClassify(node_id=f"ai-classify:{index}",
                          inputs=input_ports((input_port,)),
                          backend_name=self.backend_name, model=self.model.name,
                          spec=spec)
        return node, step


def plan_classify(region, context, *, backend_name: str):
    """Build one Quail plan for a table's AI.CLASSIFY columns and label filters.

    The plan is a chain: scan the table, then for each label filter in
    written order an AiClassify node (if that prompt has not been
    classified yet) and a LabelFilter node, then an AiClassify node for
    each projected label column not yet classified, then the projection.
    Each AiClassify node gets a cost estimate from the documents
    expected to reach it and the suffix lengths of its scoring rule.
    """
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
    scoring = context.label_scoring or DEFAULT_LABEL_SCORING
    if scoring not in LABEL_SCORINGS:
        return _refused(f"unknown label scoring rule {scoring!r}; the rules "
                        f"are {LABEL_SCORINGS}")

    lengths = [int(length) for length in context.document_tokens[alias]]
    count = len(lengths)
    total = sum(lengths)
    chunk = budgets.chunk_budget(model, context.device)
    capacity = budgets.arena_tokens(model, context.device, chunk)
    table = _Table(
        alias=alias, mean=total / max(1, count), longest=max(lengths, default=0),
        budget=min(chunk, capacity), chunk=chunk, scoring=scoring,
        backend_name=backend_name, model=model, device=context.device,
        tokenizer=context.tokenizer)

    # the output column of each classified prompt
    named = {
        column.expression: column.name for column in logical.root.columns
        if isinstance(column, Alias)
    }
    # ("classify", call) and ("filter", written position) in plan order:
    # a prompt is classified where it is first needed
    steps = []
    classified = set()
    for written_pos, predicate in enumerate(predicates):
        test = predicate.expression
        if test.call not in classified:
            named.setdefault(test.call,
                             test.name or f"__label_{alias}_{written_pos}")
            classified.add(test.call)
            steps.append(("classify", test.call))
        steps.append(("filter", written_pos))
    steps.extend(("classify", call) for call in named if call not in classified)

    nodes = [Scan(node_id=f"scan:{alias}", alias=alias, input_id=alias,
                  n_docs=count, total_tokens=total, shard_ranges=((0, count),),
                  shard_token_loads=(total,))]
    current = PortRef(nodes[0].node_id, f"ids:{alias}")
    work = Work()
    seconds = 0.0
    live = float(count)
    for kind, item in steps:
        if kind == "classify":
            try:
                node, step = table.classify(item, named[item], current, live,
                                            sum(isinstance(n, AiClassify)
                                                for n in nodes))
            except _RefusedError as refused:
                return _refused(refused.reason, "suffix_over_chunk",
                                refused.needed, refused.available, "tokens")
            work += step
            seconds += node.spec.estimated_seconds
        else:
            predicate = predicates[item]
            node = LabelFilter(
                node_id=f"label-filter:{alias}:{item}",
                inputs=input_ports((current,)),
                score_name=named[predicate.expression.call],
                aliases=(alias,), comparison="in", threshold=0.0,
                selectivity=predicate.selectivity, written_pos=item,
                accepted=tuple(predicate.expression.accepted))
            live *= effective_selectivity(predicate.selectivity)
        nodes.append(node)
        current = PortRef(node.node_id, "scores")

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
            "prefix_reuse": "document KV shared by every label suffix",
            "survivor_assumption": "uniform independent selection",
            "estimated_fresh_tokens": work.tokens,
            "estimated_attention_pairs": work.pairs,
            "batching": "token_based_admission",
            "data_parallel_copies": context.gpu_count,
            "label_scoring": scoring,
            "order_rule": "written order",
        },
    )
    return (PhysicalCandidate(plan.graph, plan, seconds),)


class _RefusedError(Exception):
    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available
