"""Physical planning for AI.CLASSIFY queries on one table.

A plan scans the table, then for each label filter in written order
classifies the documents still alive and keeps those with an accepted
label, then classifies what the projection still needs. Each document
is scored the way the join path scores a pair: its prompt head and
document stay in KV, the question and category list are written once
after it, and one short suffix per label-trie node reads the next
token's log probabilities.
"""

import heapq
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from quail.cost import budgets
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work, ask, scan, stream
from quail.execution.labels import (
    label_trie,
    replay_rounds,
    trace_key,
    trie_paths,
)
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
    ClassifyStage,
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


def classification_refusal(context) -> Refusal | None:
    """Why the context's model cannot classify, or None when it can."""
    model = context.model
    forced = context.label_scoring
    if context.tokenizer is None:
        reason = "AI.CLASSIFY planning needs the model's tokenizer"
    elif forced is not None and forced not in LABEL_SCORINGS:
        reason = (f"unknown label scoring rule {forced!r}; "
                  f"the rules are {LABEL_SCORINGS}")
    elif model.canvas_tokens and forced not in (None, "canvas"):
        reason = (f"{model.name!r} scores labels on a canvas; the "
                  f"{forced!r} rule reads left-to-right log probabilities")
    elif forced == "canvas" and not model.canvas_tokens:
        reason = f"{model.name!r} has no canvas to score labels on"
    elif model.canvas_tokens and not (model.canvas_end_text
                                      and model.canvas_pad_text):
        reason = (f"{model.name!r} names no answer end and pad tokens "
                  f"for the classification canvas")
    else:
        return None
    return _refused(reason)[0].plan


def classify_table(context, alias: str, backend_name: str) -> "_Table":
    """The classification planner for one table of the context."""
    lengths = [int(length) for length in context.document_tokens[alias]]
    count = len(lengths)
    total = sum(lengths)
    chunk = budgets.chunk_budget(context.model, context.device)
    capacity = budgets.arena_tokens(context.model, context.device, chunk)
    return _Table(
        alias=alias, mean=total / max(1, count),
        longest=max(lengths, default=0), budget=min(chunk, capacity),
        chunk=chunk, scoring=context.label_scoring,
        backend_name=backend_name, model=context.model,
        device=context.device, tokenizer=context.tokenizer,
        capacity=capacity, lengths=tuple(lengths),
        traces=getattr(context, "label_traces", None))


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
LABEL_SCORINGS = ("trie_nodes", "label_chains", "trie_paths", "trie_rounds",
                  "trie_search", "canvas")

# A round launched in one chunk has its answers read while the next
# chunk runs, so a document's next round enters the chunk after that.
READOUT_LAG_CHUNKS = 2
EXHAUSTIVE_SCORING = "trie_paths"
ADAPTIVE_SCORINGS = ("trie_rounds", "trie_search")


def suffix_lengths(scoring: str, labels) -> list[int]:
    """Tokens of each suffix a document streams under one scoring rule.

    Under ``canvas`` the one suffix is the canvas: as many rows as the
    padded labels are long. Otherwise every suffix starts with the
    answer cue's last token. Under
    ``trie_nodes`` it continues with a label-trie node's tokens and
    only its last row is read. Under ``label_chains`` it continues
    with all but a label's last token and every row is read. Under
    ``trie_paths`` it continues with one of the trie's deepest proper
    prefixes and every row is read. Under ``trie_rounds`` and
    ``trie_search`` a document sends one chain per trie depth, the cue
    and that many tokens, assuming one label path is read; how many
    are read is measured, not planned, until the cost model.
    """
    if scoring == "canvas":
        return [len(labels[0])]
    if scoring == "label_chains":
        return [len(ids) for ids in labels]
    if scoring == "trie_paths":
        return [1 + len(path) for path in trie_paths(labels)]
    if scoring in ("trie_rounds", "trie_search"):
        return [depth + 1 for depth in range(max(len(ids) for ids in labels))]
    return [1 + len(node) for node in label_trie(labels)]


def readout_component(rows: float, model) -> CostComponent:
    """The label readout: every read row through the whole output head."""
    return CostComponent(
        name="readout", flops=2.0 * model.hidden * model.vocab * rows,
        # the bf16 head streams from memory once per chunk that reads
        bytes_moved=2.0 * model.hidden * model.vocab if rows else 0.0,
        precision="bf16")


def chunk_seconds(work: Work, rows: float, model, device) -> float:
    """Price one forward chunk at the roofline: weights stream once."""
    components = dense_decoder_components(work, model, passes=1.0)
    components += (readout_component(rows, model),)
    return sum(component.seconds
               for component in component_latencies(components, device))


@dataclass(frozen=True)
class Simulated:
    """What a scoring rule costs on a table, from the replayed scheduler.

    Attributes:
        seconds: The summed roofline time of every chunk.
        passes: Forward chunks launched.
        work: The token, attention, and KV work of every chunk.
        label_tokens: Suffix tokens streamed after the documents.
        rounds: The most rounds any document ran.
    """

    seconds: float
    passes: int
    work: Work
    label_tokens: float
    rounds: int


def simulate(prefixes, frame: int, rounds, chunk: int, capacity: int,
             model, device, resident: bool = False,
             read_all_rows: bool = True) -> Simulated:
    """Replay the stage scheduler on the CPU and price each chunk.

    Documents are admitted in order while their reservation, the
    prefix plus the frame and longest chain, fits the arena. A chunk
    takes the rounds that are ready, then fresh documents, up to the
    chunk budget; a round is atomic. A document's next round is ready
    READOUT_LAG_CHUNKS chunks after the one that launched the round,
    and a document leaves the arena after its last round. Each chunk
    costs its roofline time with the weights streamed once, plus the
    readout of the rows it reads.

    Args:
        prefixes: Per document, the tokens before the frame (the prompt
            head and the document).
        frame: The frame tokens written once per document.
        rounds: Per document, its rounds, each a list of chain lengths.
        chunk: The chunk budget in tokens.
        capacity: The arena's tokens.
        model: The ModelSpec.
        device: The DeviceSpec.
        resident: Whether the prefixes are in KV already.
        read_all_rows: Whether every chain row is read, or only the last.
    """
    window = model.sliding_window
    n = len(prefixes)
    longest = max((length for document in rounds for chains in document
                   for length in chains), default=0)
    extra = frame + longest

    def item_tokens(document, index):
        chains = rounds[document][index] if rounds[document] else []
        tokens = sum(chains)
        if index == 0:
            tokens += frame + (0 if resident else prefixes[document])
        return tokens

    def item_work(document, index):
        prefix = prefixes[document]
        chains = rounds[document][index] if rounds[document] else []
        if index > 0:
            return stream(prefix + frame, chains, window=window)
        if resident:
            return (ask(prefix, frame, window=window)
                    + stream(prefix + frame, chains, window=window))
        return (scan(prefix + frame, 0, window=window)
                + stream(prefix + frame, chains, window=window))

    ready = deque()
    waiting = []           # heap of (ready chunk, document, round)
    next_document = 0
    held = 0
    seconds = 0.0
    passes = 0
    total = Work()
    label_tokens = 0.0
    most_rounds = 0
    k = 0
    while next_document < n or ready or waiting:
        while waiting and waiting[0][0] <= k:
            _, document, index = heapq.heappop(waiting)
            ready.append((document, index))
        room = chunk
        work = Work()
        rows = 0.0
        launched = []
        while ready and (item_tokens(*ready[0]) <= room or room == chunk):
            document, index = ready.popleft()
            room -= item_tokens(document, index)
            work += item_work(document, index)
            chains = rounds[document][index] if rounds[document] else []
            rows += sum(chains) if read_all_rows else len(chains)
            label_tokens += sum(chains)
            launched.append((document, index))
        while (next_document < n
               and held + prefixes[next_document] + extra <= capacity
               and (item_tokens(next_document, 0) <= room or room == chunk)):
            document = next_document
            next_document += 1
            held += prefixes[document] + extra
            room -= item_tokens(document, 0)
            work += item_work(document, 0)
            chains = rounds[document][0] if rounds[document] else []
            rows += sum(chains) if read_all_rows else len(chains)
            label_tokens += sum(chains)
            launched.append((document, 0))
        if not launched:
            # nothing is ready: the loop waits for the readouts in flight
            k = waiting[0][0]
            continue
        seconds += chunk_seconds(work, rows, model, device)
        passes += 1
        total += work
        for document, index in launched:
            most_rounds = max(most_rounds, index + 1)
            if index + 1 < len(rounds[document]):
                heapq.heappush(waiting,
                               (k + READOUT_LAG_CHUNKS, document, index + 1))
            else:
                held -= prefixes[document] + extra
        k += 1
    return Simulated(seconds, passes, total, label_tokens, most_rounds)


def rule_rounds(scoring: str, labels, traces=None, demand=None) -> tuple:
    """The rounds each document runs under one rule, and how rows are read.

    An exhaustive rule runs one round with the same chains for every
    document. An adaptive rule's rounds come from replaying it on the
    traces, one entry per traced document; without traces they are
    its worst case, every trie node read.

    Returns:
        (rounds per document, read_all_rows).
    """
    if scoring == "canvas":
        return [[[len(labels[0])]]], True
    nodes = sorted(label_trie(labels), key=lambda node: (len(node), node))
    if scoring == "trie_nodes":
        return [[[1 + len(node) for node in nodes]]], False
    if scoring == "label_chains":
        return [[[len(ids) for ids in labels]]], True
    if scoring == "trie_paths":
        return [[[1 + len(path) for path in trie_paths(labels)]]], True
    if scoring not in ADAPTIVE_SCORINGS:
        raise ValueError(f"unknown label scoring rule {scoring!r}")
    search = scoring == "trie_search"
    if traces:
        replayed = replay_rounds(labels, traces, search=search, demand=demand)
        return [rounds for rounds, _ in replayed], True
    if search:
        return [[[1 + len(node)] for node in nodes]], True
    depth = max(len(ids) for ids in labels)
    return [[[1 + len(node) for node in nodes if len(node) == d]
             for d in range(depth)]], True


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
        capacity: The arena's tokens.
        scoring: The label scoring rule every classification uses, or
            None to choose per classification by simulated cost.
        lengths: Every document's length in tokens.
        traces: Callable(trace key) -> saved exhaustive traces, or
            None when the session keeps none.
    """

    alias: str
    mean: float
    longest: int
    budget: int
    chunk: int
    scoring: str | None
    backend_name: str
    model: object
    device: object
    tokenizer: object
    capacity: int = 0
    lengths: tuple = ()
    traces: object = None

    def sample(self, live: float) -> list[int]:
        """The lengths of the documents expected to reach a classification.

        Takes ``live`` lengths spread evenly over the sorted lengths, so
        the sample keeps the table's length distribution.
        """
        ordered = sorted(self.lengths)
        count = min(len(ordered), max(1, int(round(live))))
        if not ordered:
            return []
        picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return [ordered[i] for i in picks]

    def head(self, call) -> tuple:
        """The prompt tokens before the document."""
        return tuple(self.tokenizer(call.prompt.preamble))

    def classify(self, call, name, live, resident=False, demand=None):
        """Return a ClassifySpec for one prompt and the work it does.

        Args:
            call: The logical AI.CLASSIFY call.
            name: The output column.
            live: How many documents are expected to reach it.
            resident: Whether the documents' KV is resident from an
                earlier stage of the same chain.
            demand: The labels a filter accepts when only membership
                is needed, else None.

        Raises:
            ClassifyRefusedError: A document and its prompt exceed the budget.
        """
        head = self.head(call)
        tail = tuple(call.prompt.tail_token_ids)
        labels = tuple(tuple(self.tokenizer(label_text(label)))
                       for label in call.labels)
        if self.model.canvas_tokens:
            labels = self.padded(labels)
        traces = None
        if self.traces is not None:
            traces = self.traces(trace_key(
                self.model.name, call.prompt.template, call.labels, labels))
        demanded = (None if demand is None
                    else [call.labels.index(label) for label in demand])
        scoring, simulated = self.choose(live, len(head), len(tail) - 1,
                                         labels, resident, traces, demanded)
        suffixes = suffix_lengths(scoring, labels)
        # head, document, and frame stay resident while the longest
        # suffix and the frame entry's own rows are packed beside them
        need = len(head) + self.longest + len(tail) + max(suffixes)
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a document in {self.alias!r} needs {need} tokens with its "
                f"classification prompt, but the forward pass budget is "
                f"{self.budget} tokens", need, self.budget)
        spec = ClassifySpec(
            name=name, aliases=(self.alias,),
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=live, estimated_seconds=simulated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels, scoring=scoring,
            demand=None if demand is None else tuple(demand),
            traced_documents=len(traces) if traces else 0,
        )
        return spec, simulated.work

    def padded(self, labels) -> tuple:
        """Every label padded to the canvas with the end token, then pads.

        The canvas is one row longer than the longest label, so every
        label is followed by at least the answer end token.

        Raises:
            ClassifyRefusedError: The end or pad text is not one token.
        """
        end = tuple(self.tokenizer(self.model.canvas_end_text))
        pad = tuple(self.tokenizer(self.model.canvas_pad_text))
        if len(end) != 1 or len(pad) != 1:
            raise ClassifyRefusedError(
                f"the canvas end {self.model.canvas_end_text!r} and pad "
                f"{self.model.canvas_pad_text!r} must be one token each",
                1, 0)
        rows = max(len(ids) for ids in labels) + 1
        return tuple(ids + end + pad * (rows - len(ids) - 1) for ids in labels)

    def simulate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident, traces=None, demand=None) -> Simulated:
        """Replay one rule over the documents expected and price it."""
        per_document, read_all = rule_rounds(scoring, labels, traces, demand)
        lengths = self.sample(live)
        prefixes = [head_tokens + length for length in lengths]
        rounds = [per_document[i % len(per_document)]
                  for i in range(len(prefixes))]
        return simulate(prefixes, frame_tokens, rounds, self.chunk,
                        self.capacity or self.budget, self.model, self.device,
                        resident=resident, read_all_rows=read_all)

    def choose(self, live, head_tokens, frame_tokens, labels, resident,
               traces=None, demand=None) -> tuple[str, Simulated]:
        """The rule with the least simulated time, and its simulation.

        A forced rule is the one candidate. Otherwise the exhaustive
        rule is a candidate, and the adaptive rules are candidates
        only with traces to replay, since their work is not known
        without them. Ties go to fewer rounds, then fewer label
        tokens, then the order listed.
        """
        if self.model.canvas_tokens:
            candidates = ["canvas"]
        elif self.scoring:
            candidates = [self.scoring]
        else:
            candidates = [EXHAUSTIVE_SCORING]
            if traces:
                candidates += list(ADAPTIVE_SCORINGS)
        best = None
        for scoring in candidates:
            simulated = self.simulate(scoring, live, head_tokens, frame_tokens,
                                      labels, resident, traces, demand)
            key = (simulated.seconds, simulated.rounds, simulated.label_tokens)
            if best is None or key < best[0]:
                best = (key, scoring, simulated)
        return best[1], best[2]

    def node(self, spec, input_port, index) -> AiClassify:
        """The plan node running a classification and its chained stages."""
        return AiClassify(node_id=f"ai-classify:{index}",
                          inputs=input_ports((input_port,)),
                          backend_name=self.backend_name,
                          model=self.model.name, spec=spec)


def plan_classify(region, context, *, backend_name: str):
    """Build one Quail plan for a table's AI.CLASSIFY columns and label filters.

    The plan is a chain: scan the table, then for each label filter in
    written order an AiClassify node (if that prompt has not been
    classified yet) and a LabelFilter node, then an AiClassify node for
    each projected label column not yet classified, then the projection.
    Each classification gets a cost estimate from the documents
    expected to reach it and the suffix lengths of its scoring rule.

    A classification that follows another on the same documents, with
    the same prompt head and at most one label filter between them
    testing the earlier label, joins the earlier node as a stage: the
    executor runs it on the documents the filter accepts while their
    KV is still resident.
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
    refusal = classification_refusal(context)
    if refusal is not None:
        return (PhysicalCandidate(None, refusal, float("inf")),)
    # None lets the cost model choose per classification
    scoring = context.label_scoring
    table = classify_table(context, alias, backend_name)
    count = len(context.document_tokens[alias])
    total = sum(int(length) for length in context.document_tokens[alias])
    chunk = table.chunk
    capacity = budgets.arena_tokens(model, context.device, chunk)

    # the output column of each classified prompt
    named = {
        column.expression: column.name for column in logical.root.columns
        if isinstance(column, Alias)
    }
    # a label that is only tested for membership, with one accepted
    # set, is scored only as far as that decision needs
    demands = {}
    for predicate in predicates:
        test = predicate.expression
        demands.setdefault(test.call, set()).add(tuple(test.accepted))
    demands = {call: next(iter(accepted)) for call, accepted in demands.items()
               if call not in named and len(accepted) == 1}
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
    chain = None          # index in nodes of the open classify chain
    last_call = None      # the chain's latest stage
    since = []            # label filters since that stage
    for kind, item in steps:
        if kind == "classify":
            joins = (chain is not None and len(since) <= 1
                     and all(predicates[p].expression.call is last_call
                             for p in since)
                     and table.head(item) == table.head(last_call))
            try:
                spec, step = table.classify(item, named[item], live,
                                            resident=joins,
                                            demand=demands.get(item))
            except ClassifyRefusedError as refused:
                return (PhysicalCandidate(None, refused.refusal(),
                                          float("inf")),)
            work += step
            seconds += spec.estimated_seconds
            if joins:
                gate = (tuple(predicates[since[0]].expression.accepted)
                        if since else None)
                root = nodes[chain]
                nodes[chain] = replace(root, spec=replace(
                    root.spec, stages=root.spec.stages
                    + (ClassifyStage(spec=spec, accepted=gate),)))
            else:
                chain = len(nodes)
                nodes.append(table.node(
                    spec, current, sum(isinstance(n, AiClassify)
                                       for n in nodes)))
                current = PortRef(nodes[chain].node_id, "scores")
            last_call, since = item, []
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
            since.append(item)

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
            "prefix_reuse": "document KV shared by every label suffix and "
                            "by every chained classification",
            "survivor_assumption": "uniform independent selection",
            "estimated_fresh_tokens": work.tokens,
            "estimated_attention_pairs": work.pairs,
            "batching": "token_based_admission",
            "data_parallel_copies": context.gpu_count,
            "label_scoring": scoring or "cost model",
            "order_rule": "written order",
        },
    )
    return (PhysicalCandidate(plan.graph, plan, seconds),)


class ClassifyRefusedError(Exception):
    """A classification the forward pass budget cannot hold."""

    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available

    def refusal(self) -> Refusal:
        """The planning refusal this exception stands for."""
        return _refused(self.reason, "suffix_over_chunk", self.needed,
                        self.available, "tokens")[0].plan
