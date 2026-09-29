"""Physical planning for AI.CLASSIFY queries on one table.

A plan scans the table, then for each label filter in written order
classifies the documents still alive and keeps those with an accepted
label, then classifies what the projection still needs. Each document
is scored the way the join path scores a pair: its prompt head and
document stay in KV, the question and category list are written once
after it, and short suffixes over the label trie read the next
token's log probabilities. A diffusion model decodes its answer
instead: every denoising step sends the answer canvas after the
resident prompt (the ``canvas`` rule).
"""

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from quail.cost import budgets
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work, ask, scan, stream
from quail.execution.labels import label_trie, trie_paths
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
                  f"{forced!r} rule reads the rows of a causal chain")
    elif forced == "canvas" and not model.canvas_tokens:
        reason = f"{model.name!r} has no canvas to score labels on"
    elif model.canvas_tokens and model.denoising is None:
        reason = (f"{model.name!r} names no denoising sampler to decode "
                  f"answers with")
    else:
        return None
    return _refused(reason)[0].plan


def classify_table(context, alias: str, backend_name: str,
                   shared=()) -> "_Table":
    """The classification planner for one table of the context.

    Args:
        context: The PlanningContext.
        alias: The table's alias.
        backend_name: The backend running the plan.
        shared: Per document, the prefix tokens an earlier document
            also has, when the plan shares prefixes; else empty.
    """
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
        capacity=capacity, lengths=tuple(lengths), shared=tuple(shared),
        tree=(context.model.weight_precision == "fp8"
              and not context.model.canvas_tokens
              and getattr(context, "attention", None) != "unified"))


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
LABEL_SCORINGS = ("trie_paths", "trie_tree", "trie_decode", "canvas")

EXHAUSTIVE_SCORING = "trie_paths"
# every node once, in one request: fewest tokens, tree attention only
TREE_SCORING = "trie_tree"
# one token per round along the greedy path: fewest tokens, most rounds
DECODE_SCORING = "trie_decode"


def decodable(labels) -> bool:
    """Whether a greedy decode over the trie ends at a whole label.

    A label that is a proper prefix of another label would need the
    decode to choose between stopping and continuing, which the
    rules score no end token for.
    """
    ids = {tuple(label) for label in labels}
    return not any(tuple(label[:depth]) in ids
                   for label in labels for depth in range(1, len(label)))


def suffix_lengths(scoring: str, labels, canvas_rows: int = 0) -> list[int]:
    """Tokens of each suffix a document streams under one scoring rule.

    Every suffix starts with the answer cue's last token. Under
    ``trie_paths`` it continues with one of the trie's deepest proper
    prefixes, and under ``trie_tree`` the one suffix holds every trie
    node once; every row is read. Under ``trie_decode`` a document
    sends one chain per round, the cue and the tokens decoded so far,
    for as many rounds as the mean label length rounded up, and reads
    each chain's last row; how many rounds it needs is decided as it
    runs. Under ``canvas`` the one suffix is the cue and the
    ``canvas_rows`` rows of the answer canvas, sent once per denoising
    step; the canvas rows are read.

    Raises:
        ValueError: The rule is not one of LABEL_SCORINGS.
    """
    if scoring == "trie_decode":
        rounds = math.ceil(sum(len(ids) for ids in labels) / len(labels))
        return [1 + depth for depth in range(rounds)]
    if scoring == "canvas":
        return [1 + canvas_rows]
    if scoring == "trie_tree":
        return [len(label_trie(labels))]
    if scoring == "trie_paths":
        return [1 + len(path) for path in trie_paths(labels)]
    raise ValueError(f"unknown label scoring rule {scoring!r}")


def readout_component(rows: float, model, gemms: int = 1) -> CostComponent:
    """The label readout: every read row through the whole output head.

    A denoising step's readout multiplies the head twice: the rows by
    the head for the logits, then the probabilities by the tied
    embedding for the next step's self-conditioning input.
    """
    return CostComponent(
        name="readout", flops=2.0 * model.hidden * model.vocab * rows * gemms,
        # the bf16 head streams from memory once per GEMM of a chunk
        # that reads
        bytes_moved=2.0 * model.hidden * model.vocab * gemms if rows else 0.0,
        precision="bf16")


def chunk_seconds(work: Work, rows: float, model, device,
                  gemms: int = 1) -> float:
    """Price one forward chunk at the roofline: weights stream once."""
    components = dense_decoder_components(work, model, passes=1.0)
    components += (readout_component(rows, model, gemms),)
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


def simulate(prefixes, frame: int, chains, chunk: int, capacity: int,
             model, device, resident: bool = False, canvas_rows: int = 0,
             rounds: int = 1, one_per_round: bool = False,
             shared=None) -> Simulated:
    """Replay the stage scheduler on the CPU and price each chunk.

    Documents are admitted in order while their reservation, the
    prefix plus the frame and longest chain, fits the arena, and a
    chunk takes documents up to the chunk budget; a document's request
    is atomic. A document sends its chains ``rounds`` times, once per
    denoising step, or with ``one_per_round`` one of its chains per
    round, in order, reading only that chain's last row. A round's
    answers are read while the next chunk runs, so the document's next
    round enters the chunk after that, or the next chunk when nothing
    else is ready; waiting rounds enter a chunk before fresh
    documents. A document leaves the arena once its last round's chunk
    has run. Each chunk costs its roofline time with the weights
    streamed once, plus the readout of every row it reads.

    The self-conditioning MLP on the canvas rows (three hidden by
    intermediate GEMMs per row) is left out: it is under 1% of a
    row's readout. Every document is priced at ``rounds`` rounds; a
    canvas that settles sooner takes fewer.

    Args:
        prefixes: Per document, the tokens before the frame (the prompt
            head and the document).
        frame: The frame tokens written once per document.
        chains: Per document, the lengths of the chains it streams.
        chunk: The chunk budget in tokens.
        capacity: The arena's tokens.
        model: The ModelSpec.
        device: The DeviceSpec.
        resident: Whether the prefixes are in KV already.
        canvas_rows: The rows of the answer canvas each denoising step
            reads, which read the document a second time; 0 for a rule
            that reads its suffixes' rows.
        rounds: The denoising steps of every document; 1 for a rule
            that reads its answers once. With ``one_per_round`` the
            rounds are the document's chains.
        one_per_round: Each round sends one of the document's chains,
            in order, and reads its last row.
        shared: Per document, the leading prefix tokens an earlier
            document already computed, which it borrows from KV
            instead of computing; None when no prefix is shared.
    """
    window = model.sliding_window
    n = len(prefixes)
    shared = [0] * n if shared is None else shared
    longest = max((length for document in chains for length in document),
                  default=0)
    extra = frame + longest

    def round_chains(document, round_):
        if one_per_round:
            return [chains[document][round_]]
        return chains[document]

    def round_count(document):
        return len(chains[document]) if one_per_round else rounds

    def first_tokens(document):
        tokens = sum(round_chains(document, 0)) + frame
        if not resident:
            tokens += prefixes[document] - shared[document]
        return tokens

    def suffix_work(document, round_):
        prefix = prefixes[document]
        # a canvas longer than one row reads the document a second
        # time, in the non-causal call every canvas row runs
        suffixes = stream(prefix + frame, round_chains(document, round_),
                          window=window)
        if canvas_rows > 1:
            suffixes = suffixes + Work(
                pairs=suffixes.pairs, kv_read=suffixes.kv_read,
                sliding_pairs=suffixes.sliding_pairs,
                sliding_kv_read=suffixes.sliding_kv_read)
        return suffixes

    def first_work(document):
        prefix = prefixes[document]
        suffixes = suffix_work(document, 0)
        if resident:
            return ask(prefix, frame, window=window) + suffixes
        if shared[document]:
            return ask(shared[document], prefix - shared[document] + frame,
                       window=window) + suffixes
        return scan(prefix + frame, 0, window=window) + suffixes

    def read_rows(document, round_):
        if one_per_round:
            return 1
        return canvas_rows or sum(chains[document])

    next_document = 0
    seconds = 0.0
    passes = 0
    total = Work()
    label_tokens = 0.0
    most_rounds = 0
    held = 0
    waiting = deque()    # (first chunk it may enter, document, round)
    while next_document < n or waiting:
        room = chunk
        work = Work()
        rows = 0.0
        launched = []
        while waiting and waiting[0][0] <= passes:
            _, document, round_ = waiting[0]
            tokens = sum(round_chains(document, round_))
            if tokens > room and room != chunk:
                break
            waiting.popleft()
            room -= tokens
            work += suffix_work(document, round_)
            rows += read_rows(document, round_)
            label_tokens += tokens
            launched.append((document, round_))
        while (next_document < n
               and (held + prefixes[next_document] + extra <= capacity
                    or not held)
               and (first_tokens(next_document) <= room or room == chunk)):
            document = next_document
            next_document += 1
            held += prefixes[document] + extra
            room -= first_tokens(document)
            work += first_work(document)
            rows += read_rows(document, 0)
            label_tokens += sum(round_chains(document, 0))
            launched.append((document, 0))
        if not launched:
            # the waiting rounds' answers are read before anything packs
            waiting = deque((min(ready, passes), document, round_)
                            for ready, document, round_ in waiting)
            continue
        seconds += chunk_seconds(work, rows, model, device,
                                 gemms=2 if canvas_rows else 1)
        for document, round_ in launched:
            most_rounds = max(most_rounds, round_ + 1)
            if round_ + 1 < round_count(document):
                waiting.append((passes + 2, document, round_ + 1))
            else:
                held -= prefixes[document] + extra
        passes += 1
        total += work
    return Simulated(seconds, passes, total, label_tokens, most_rounds)


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
        shared: Per document, the leading tokens an earlier document
            also has, which prefix sharing borrows from KV; empty when
            the plan does not share prefixes or the corpus is not
            tokenized yet.
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
    shared: tuple = ()
    # whether the model runs the tree attention path, which the
    # packed trie needs: an fp8 model with the plan not forced unified
    tree: bool = False

    def sample(self, live: float) -> list[tuple[int, int]]:
        """The documents expected to reach a classification.

        Takes ``live`` documents spread evenly over the documents sorted
        by length, so the sample keeps the table's length distribution.

        Returns:
            Per sampled document, its length and its shared prefix
            tokens.
        """
        shared = self.shared or (0,) * len(self.lengths)
        ordered = sorted(zip(self.lengths, shared))
        count = min(len(ordered), max(1, int(round(live))))
        if not ordered:
            return []
        picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return [ordered[i] for i in picks]

    def head(self, call) -> tuple:
        """The prompt tokens before the document."""
        return tuple(self.tokenizer(call.prompt.preamble))

    def classify(self, call, name, live, resident=False):
        """Return a ClassifySpec for one prompt and the work it does.

        Args:
            call: The logical AI.CLASSIFY call.
            name: The output column.
            live: How many documents are expected to reach it.
            resident: Whether the documents' KV is resident from an
                earlier stage of the same chain.

        Raises:
            ClassifyRefusedError: A document and its prompt exceed the budget.
        """
        head = self.head(call)
        tail = tuple(call.prompt.tail_token_ids)
        labels = tuple(tuple(self.tokenizer(label_text(label)))
                       for label in call.labels)
        scoring, simulated = self.choose(live, len(head), len(tail) - 1,
                                         labels, resident)
        if scoring == DECODE_SCORING and not decodable(labels):
            raise ClassifyRefusedError(
                f"the {scoring!r} rule cannot decode the labels of "
                f"{name!r}: one label is a proper prefix of another",
                1, 0)
        suffixes = suffix_lengths(scoring, labels, self.canvas_rows)
        if scoring == "canvas" and max(map(len, labels)) > self.canvas_rows:
            raise ClassifyRefusedError(
                f"a label of {max(map(len, labels))} tokens does not fit "
                f"the {self.canvas_rows}-row answer canvas",
                max(map(len, labels)), self.canvas_rows)
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
        )
        return spec, simulated.work

    @property
    def canvas_rows(self) -> int:
        """The rows of the model's answer canvas; 0 without one."""
        denoising = self.model.denoising
        return denoising.canvas_rows if denoising is not None else 0

    def simulate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident) -> Simulated:
        """Replay one rule over the documents expected and price it."""
        chains = suffix_lengths(scoring, labels, self.canvas_rows)
        documents = self.sample(live)
        prefixes = [head_tokens + length for length, _ in documents]
        # a borrowed prefix includes the prompt head the documents share
        shared = [head_tokens + tokens if tokens else 0
                  for _, tokens in documents]
        return simulate(prefixes, frame_tokens, [chains] * len(prefixes),
                        self.chunk, self.capacity or self.budget, self.model,
                        self.device, resident=resident,
                        canvas_rows=(self.canvas_rows if scoring == "canvas"
                                     else self.model.canvas_tokens),
                        rounds=(self.model.denoising.max_steps
                                if scoring == "canvas" else 1),
                        one_per_round=scoring == DECODE_SCORING,
                        shared=shared)

    def reestimate(self, spec: ClassifySpec) -> ClassifySpec:
        """The spec with its seconds simulated again on this table.

        The plan is made before the corpus is tokenized, so its
        estimate prices every document from scratch. Once the token
        store is known and the plan shares prefixes, the documents
        borrowing a prefix pay only for the rest, as the filter and
        join estimates do.
        """
        head, tail = spec.prompt_token_parts
        simulated = self.simulate(
            spec.scoring, spec.expected_inputs, len(head), len(tail) - 1,
            spec.label_token_ids, False)
        return replace(spec, estimated_seconds=simulated.seconds)

    def choose(self, live, head_tokens, frame_tokens, labels, resident
               ) -> tuple[str, Simulated]:
        """The rule with the least simulated time, and its simulation.

        A forced rule is the one candidate, as is ``canvas`` on a
        canvas model. Otherwise ``trie_paths`` is a candidate,
        ``trie_tree`` under tree attention, and ``trie_decode`` for a
        classification whose documents are not resident from an
        earlier stage and whose labels a greedy decode can end at.
        Ties go to fewer label tokens, then the order listed.
        """
        if self.scoring:
            candidates = [self.scoring]
        elif self.model.canvas_tokens:
            candidates = ["canvas"]
        else:
            candidates = [EXHAUSTIVE_SCORING]
            if self.tree:
                candidates.append(TREE_SCORING)
            if not resident and decodable(labels):
                candidates.append(DECODE_SCORING)
        best = None
        for scoring in candidates:
            simulated = self.simulate(scoring, live, head_tokens, frame_tokens,
                                      labels, resident)
            key = (simulated.seconds, simulated.label_tokens)
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
            # a classification after another on the same documents
            # reads their resident KV (in one node, or in one pipeline
            # of two nodes)
            follows = (chain is not None and len(since) <= 1
                       and all(predicates[p].expression.call is last_call
                               for p in since)
                       and table.head(item) == table.head(last_call))
            # a decoded classification's rounds are its own node's
            # stages, as are a canvas classification's denoising steps
            joins = follows and nodes[chain].spec.scoring not in (
                DECODE_SCORING, "canvas")
            try:
                spec, step = table.classify(item, named[item], live,
                                            resident=follows)
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
