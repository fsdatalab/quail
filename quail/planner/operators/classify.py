"""AI.CLASSIFY: its statistics, scoring rule, cost, spec, and plan node."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from quail.cost import budgets
from quail.cost import classify as classify_cost
from quail.cost.work import Work
from quail.labels import (
    DECISION_SCORING,
    DECODE_SCORING,
    LETTERS_SCORING,
    TREE_SCORING,
    decodable,
    word_starts,
)
from quail.logical import LabelWork, effective_selectivity
from quail.physical import (
    AiClassify,
    ClassifySpec,
    Filter,
    PortRef,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import bind_inference, inference_node
from quail.planner.prompt_lowering import (
    choice_token_parts,
    classify_prompt_tokens,
    lower_prompt,
)
from quail.planner.statistics.memory import arena_capacity
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import ClassifyRefusedError


def classify_statistics(context, alias: str, shared=()) -> "ClassifyStatistics":
    """Gather document lengths and execution limits for one classification table.

    Args:
        context: The PlanningContext.
        alias: The table's alias.
        shared: Per document, the prefix tokens an earlier document
            also has, when the plan shares prefixes; else empty.

    Returns:
        The table's document lengths, budgets, and model settings.
    """
    lengths = context.document_tokens[alias]
    summary = summarize_alias(lengths, context.model.sliding_window, memo=context.memo)
    if hasattr(lengths, "arrow_array"):
        lengths = lengths.arrow_array.to_numpy()
    chunk = budgets.chunk_budget(context.model, context.device)
    capacity = arena_capacity(context, chunk)
    return ClassifyStatistics(
        alias=alias, mean=summary.mean,
        longest=summary.maximum, budget=min(chunk, capacity),
        chunk=chunk,
        draws=context.canvas_draws if context.model.answer_canvas else 1,
        model=context.model,
        device=context.device,
        capacity=capacity, lengths=lengths, shared=tuple(shared),
        tree=(budgets.tree_attention_allowed(context.model)
              and getattr(context, "attention", None) != "unified"))


@dataclass(frozen=True)
class ClassifyStatistics:
    """Document lengths, model settings, and execution budgets for one table.

    Attributes:
        alias: Table alias in the query.
        mean: Mean document length in tokens.
        longest: Maximum document length in tokens.
        budget: Smaller of the forward-pass token budget and KV capacity.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        capacity: KV arena capacity in tokens.
        draws: Maximum diffusion draws per document; one for causal models.
        lengths: Document lengths in tokens.
        shared: Shared prefix length per document, or an empty tuple.
        tree: Whether the tree attention path, which trie_tree needs, is
            available: an fp8 model without a canvas whose attention is not
            forced to unified.
    """

    alias: str
    mean: float
    longest: int
    budget: int
    chunk: int
    model: object
    device: object
    capacity: int = 0
    draws: int = 1
    lengths: Sequence = ()
    shared: tuple = ()
    tree: bool = False


def classify_scoring(context, alias, shared=()):
    """Prepare scoring-method choices for one table and execution context."""
    stats = classify_statistics(context, alias, shared)
    return ClassifyScoring(**vars(stats), tokenizer=context.tokenizer,
                           lower=context.lowered)


@dataclass(frozen=True, kw_only=True)
class ClassifyScoring(ClassifyStatistics):
    """Choose and price the scoring method of a prepared classification."""

    tokenizer: object
    lower: object = None

    def choose_scoring(self, spec: ClassifySpec, call,
                       resident: bool) -> ClassifySpec:
        """Pick the scoring rule for a prepared specification.

        Args:
            spec: A specification from prepare().
            call: Its logical AI.CLASSIFY call.
            resident: Whether document KV is available from an earlier operator.

        Returns:
            The specification with its scoring rule, token parts, draws,
            and estimated seconds.

        Raises:
            ClassifyRefusedError: No scoring rule can run, or the document and
                prompt exceed a token or KV budget.
        """
        if spec.scoring == DECISION_SCORING:
            return spec
        head, tail = spec.prompt_token_parts
        labels, starts = spec.label_token_ids, spec.label_word_starts
        lettered = None
        lower = self.lower or (lambda prompt: lower_prompt(prompt, self.tokenizer))
        lettered_prompt = lower(call.prompt).lettered
        if lettered_prompt is not None:
            letter_head, letter_tail, letter_ids = classify_prompt_tokens(
                lettered_prompt, lettered_prompt.letters, self.tokenizer)
            lettered = (len(letter_head), len(letter_tail) - 1, letter_ids)
        scoring, estimated = self.choose(
            spec.expected_inputs, len(head), len(tail) - 1, labels, resident,
            lettered, probabilities=spec.probabilities, starts=starts)
        if scoring == LETTERS_SCORING:
            head, tail, labels = letter_head, letter_tail, letter_ids
            starts = ()
        if scoring == DECODE_SCORING and not decodable(labels):
            raise ClassifyRefusedError(
                f"the {scoring!r} rule cannot decode the labels of "
                f"{spec.name!r}: one label is a proper prefix of another",
                1, 0)
        suffixes = classify_cost.suffix_lengths(scoring, labels, self.canvas_rows,
                                                starts=starts)
        # the head, longest document, frame, and longest suffix must fit
        # together within the forward pass and KV budget
        need = len(head) + self.longest + len(tail) + max(suffixes)
        if need > self.budget:
            raise ClassifyRefusedError.over_budget(
                f"a document in {self.alias!r}", need, self.budget)
        return replace(
            spec, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), label_token_ids=labels,
            label_word_starts=starts, scoring=scoring,
            draws=self.draws if scoring == LETTERS_SCORING else 1)

    @property
    def canvas_rows(self) -> int:
        """Return the number of answer canvas rows, or zero without a canvas."""
        canvas = self.model.answer_canvas
        return canvas.rows if canvas is not None else 0

    def estimate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident, starts=()) -> classify_cost.ClassifyCost:
        """Estimate one scoring rule using this table's lengths and budgets.

        Args:
            scoring: Label scoring rule name.
            live: Expected number of documents to classify.
            head_tokens: Number of prompt tokens before each document.
            frame_tokens: Number of prompt tokens written after each document.
            labels: Token sequence for each label.
            resident: Whether document KV is already available.
            starts: Per label, which tokens begin a new word.

        Returns:
            Estimated execution time, work, and token counts.
        """
        return classify_cost.estimate(
            scoring, live, head_tokens, frame_tokens, labels,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            model=self.model, device=self.device, resident=resident,
            draws=self.draws, starts=starts)

    def estimate_spec(self, spec: ClassifySpec,
                      resident=False) -> classify_cost.ClassifyCost:
        """Estimate a specification's chosen scoring rule on this table.

        Args:
            spec: Classification specification with its scoring rule.
            resident: Whether document KV is already available.
        """
        head, tail = spec.prompt_token_parts
        if spec.scoring == DECISION_SCORING:
            return classify_cost.estimate_suffixes(
                len(head), spec.frame_tokens, [len(tail) - spec.frame_tokens],
                live=spec.expected_inputs, lengths=self.lengths,
                shared=self.shared, chunk=self.chunk, model=self.model,
                device=self.device, resident=resident,
                answer_rows=len(spec.label_token_ids) + 1)
        return self.estimate(
            spec.scoring, spec.expected_inputs, len(head), len(tail) - 1,
            spec.label_token_ids, resident, spec.label_word_starts)

    def reestimate(self, spec: ClassifySpec, *, resident=False) -> ClassifySpec:
        """Update a classification's estimated time for this table.

        Args:
            spec: Classification specification to update.
            resident: Whether document KV is already available.

        Returns:
            A copy of spec with estimated_seconds recalculated.
        """
        return replace(spec,
                       estimated_seconds=self.estimate_spec(spec, resident).seconds)

    def choose(self, live, head_tokens, frame_tokens, labels, resident,
               lettered=None, probabilities=False, starts=()
               ) -> tuple[str, classify_cost.ClassifyCost]:
        """Choose the supported scoring rule with the lowest estimated time.

        The letters rule requires a prompt with one distinct token per
        label. The trie_tree rule requires the tree attention path. The
        trie_decode rule requires a model without a canvas, no request
        for probabilities, and no label whose token sequence is a prefix
        of another's. Ties favor fewer suffix tokens, then the first
        candidate.

        Args:
            live: Expected number of documents to classify.
            head_tokens: Number of prompt tokens before the document.
            frame_tokens: Number of prompt tokens after the document, excluding
                the final answer cue token.
            labels: Token sequence for each label.
            resident: Whether document KV is available from an earlier operator.
            lettered: Tuple of head length, frame length, and label letter
                token sequences, or None if the prompt has no lettered form.
            probabilities: Whether every label's probability is required.
            starts: Per label, which tokens begin a new word.

        Returns:
            A tuple containing the method name and its estimated cost.

        Raises:
            ClassifyRefusedError: No supported scoring rule can run.
        """
        candidates = [LETTERS_SCORING] if lettered is not None else []
        if self.tree:
            candidates.append(TREE_SCORING)
        if (not self.model.canvas_tokens and not probabilities
                and decodable(labels)):
            candidates.append(DECODE_SCORING)
        if not candidates:
            raise ClassifyRefusedError(
                "no label scoring rule can run: no one-token letter for "
                "every label, no tree attention for the packed trie, and "
                "no greedy decode (an answer canvas, or a label that is "
                "another's prefix)", 1, 0)
        best = None
        for scoring in candidates:
            head, frame, ids, flags = (
                (head_tokens, frame_tokens, labels, starts)
                if scoring != LETTERS_SCORING else (*lettered, ()))
            estimated = self.estimate(scoring, live, head, frame, ids, resident,
                                      flags)
            key = (estimated.seconds, estimated.suffix_tokens)
            if best is None or key < best[0]:
                best = (key, scoring, estimated)
        return best[1], best[2]


def classify_builder(context, alias, backend_name):
    """Prepare the inputs for constructing classification operators."""
    stats = classify_statistics(context, alias)
    return ClassifyBuilder(
        **vars(stats), tokenizer=context.tokenizer, lower=context.lowered,
        backend_name=backend_name)


@dataclass(frozen=True, kw_only=True)
class ClassifyBuilder(ClassifyStatistics):
    """Construct classification specifications and operators."""

    tokenizer: object
    lower: object
    backend_name: str

    def prepare(self, call, name, live) -> tuple[ClassifySpec, Work]:
        """Build a classification specification and work for a required method.

        The specification carries the prompt that names the labels; the
        choose_classify_implementation rule picks the scoring rule, swaps in the
        lettered prompt when it picks letters, and fills the estimate.

        Args:
            call: Logical AI.CLASSIFY call.
            name: Result column name.
            live: Expected number of documents to classify.

        Returns:
            The specification and its work when the model requires a scoring
            method. Otherwise, the choose_classify_implementation rule
            chooses the method and fills in its work later.
        """
        lowered = self.lower(call.prompt)
        if self.model.role == "decision":
            return self._choice_spec(call, name, live, (self.alias,),
                                     tuple(lowered.preamble_token_ids))
        head, tail, labels = classify_prompt_tokens(
            lowered, call.labels, self.tokenizer)
        return bind_inference(
            ClassifySpec, call.prompt, name=name, aliases=(self.alias,),
            expected_inputs=live, estimated_seconds=0.0,
            prompt_token_parts=(head, tail),
            labels=tuple(call.labels),
            label_token_ids=labels,
            label_word_starts=word_starts(
                labels, getattr(self.tokenizer, "decode", None)),
            scoring="", probabilities=call.probabilities,
        ), Work()

    def node(self, spec, input_port, index, *ports) -> AiClassify:
        """Build the physical node for one classification."""
        return inference_node(
            AiClassify, spec, (input_port, *ports), index=index,
            backend=self.backend_name, model=self.model.name)

    def classify_joined(self, call, name, partner, pairs, partner_tokens):
        """Build a classification specification for joined document pairs.

        Pair classification uses letters and reuses the anchor document's KV.

        Args:
            call: Logical AI.CLASSIFY call referring to both documents.
            name: Result column name.
            partner: Partner table alias.
            pairs: Expected number of joined pairs to classify.
            partner_tokens: Mean partner document length in tokens.

        Returns:
            A tuple containing the ClassifySpec and estimated Work.

        Raises:
            ClassifyRefusedError: Probabilities are requested, the prompt lacks
                a lettered form, or its tokens exceed an execution budget.
        """
        if call.probabilities:
            raise ClassifyRefusedError(
                f"a classification of joined rows returns its label only; "
                f"{name!r} asks for the labels' probabilities", 1, 0)
        if self.model.role == "decision":
            return self._joined_choice_spec(call, name, partner, pairs,
                                            partner_tokens)
        prompt = self.lower(call.prompt).lettered
        if prompt is None:
            raise ClassifyRefusedError(
                "the letters rule needs a one-token letter for every "
                "label, which the tokenizer does not have", 1, 0)
        _, tail, labels = classify_prompt_tokens(
            prompt, prompt.letters, self.tokenizer)
        head = tuple(prompt.preamble_token_ids)
        parts = {alias: (label, frame)
                 for alias, label, frame in prompt.label_token_ids}
        note, partner_label = parts[self.alias][1], parts[partner][0]
        block = int(round(partner_tokens)) + len(tail) - 1
        suffixes = [
            block + length
            for length in classify_cost.suffix_lengths(LETTERS_SCORING, labels)
        ]
        estimated = classify_cost.estimate_suffixes(
            len(head), len(note) + len(partner_label), suffixes, live=pairs,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            model=self.model, device=self.device, resident=True,
            canvas_rows=self.model.canvas_tokens,
            head_rows=len({tuple(ids) for ids in labels}))
        need = len(head) + self.longest + len(note) + max(suffixes)
        if need > self.budget:
            raise ClassifyRefusedError.over_budget(
                f"a row of {self.alias!r} x {partner!r}", need, self.budget)
        return bind_inference(
            ClassifySpec, call.prompt, name=name, aliases=(self.alias, partner),
            expected_inputs=pairs, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels, scoring=LETTERS_SCORING,
            join_layout=(tuple(note), tuple(partner_label)),
        ), estimated.work

    def _choice_spec(self, call, name, live, aliases, head):
        """Build a decision_choice specification for one document per row.

        Each document takes the question as its frame, then one request
        of every option block and the closing line.
        """
        tail, frame, blocks = choice_token_parts(
            self.lower(call.prompt), self.tokenizer)
        request = len(tail) - frame
        need = len(head) + self.longest + len(tail)
        if need > self.budget:
            raise ClassifyRefusedError.over_budget(
                f"a document in {self.alias!r}", need, self.budget)
        estimated = classify_cost.estimate_suffixes(
            len(head), frame, [request], live=live, lengths=self.lengths,
            shared=self.shared, chunk=self.chunk, model=self.model,
            device=self.device, answer_rows=len(blocks) + 1)
        return bind_inference(
            ClassifySpec, call.prompt, name=name, aliases=aliases,
            expected_inputs=live, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=blocks, scoring=DECISION_SCORING,
            probabilities=call.probabilities, frame_tokens=frame), estimated.work

    def _joined_choice_spec(self, call, name, partner, pairs, partner_tokens):
        """Build a decision_choice specification for joined document pairs.

        The anchor's KV ends in its note and the partner's label; each
        pair's request is the partner document, the question, every
        option block, and the closing line.
        """
        prompt = self.lower(call.prompt)
        head = tuple(prompt.preamble_token_ids)
        tail, frame, blocks = choice_token_parts(prompt, self.tokenizer)
        parts = {alias: (label, note)
                 for alias, label, note in prompt.label_token_ids}
        note, partner_label = parts[self.alias][1], parts[partner][0]
        request = int(round(partner_tokens)) + len(tail)
        estimated = classify_cost.estimate_suffixes(
            len(head), len(note) + len(partner_label), [request], live=pairs,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            model=self.model, device=self.device, resident=True,
            answer_rows=len(blocks) + 1)
        need = len(head) + self.longest + len(note) + request
        if need > self.budget:
            raise ClassifyRefusedError.over_budget(
                f"a row of {self.alias!r} x {partner!r}", need, self.budget)
        return bind_inference(
            ClassifySpec, call.prompt, name=name, aliases=(self.alias, partner),
            expected_inputs=pairs, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=blocks, scoring=DECISION_SCORING,
            join_layout=(tuple(note), tuple(partner_label)),
            frame_tokens=frame), estimated.work


def emit_classify(builder, alias, live, calls):
    """Classify the alias's documents, then keep the accepted labels.

    Args:
        builder: The graph builder.
        alias: The table.
        live: The documents the classifications read.
        calls: The classifications to emit, in run order.
    """
    table = classify_builder(builder.context, alias, "quail")
    steps = []
    for call in calls:
        steps.extend((test.call, test)
                     for test in builder.operators.label_filters.get(alias, ())
                     if test.call == call)
        if call not in builder.operators.labels.tests:
            steps.append((call, None))
    classified_calls = set()
    scores = None
    for call, test in steps:
        if call not in classified_calls:
            spec, work = table.prepare(
                call, builder.operators.labels.names[call], live)
            if builder.scoring is not None:
                builder.scoring.work += work
            node = table.node(
                spec, scores or builder.source[alias],
                              sum(isinstance(n, AiClassify) for n in builder.nodes))
            builder.nodes.append(node)
            scores = PortRef(node.node_id, "rows")
            builder.source[alias] = scores
            if call in builder.operators.labels.projected:
                builder.label_ports.append(scores)
            classified_calls.add(call)
        if test is not None:
            lid = f"filter:{alias}:{test.position}"
            builder.nodes.append(Filter(
                node_id=lid, inputs=input_ports((scores,)),
                predicate=test.condition, aliases=(alias,),
                selectivity=test.selectivity, written_pos=test.position))
            scores = PortRef(lid, "rows")
            builder.source[alias] = scores
            live *= effective_selectivity(test.selectivity)



def joined_calls(labels: LabelWork) -> list:
    """Return classifications of document pairs and their table aliases."""
    return [(call, call.aliases()) for call, _ in labels.calls
            if len(call.aliases()) == 2]


def emit_joined_classify(builder):
    """Append the classifications of joined rows after their joins."""
    stats = builder.operator_order.statistics.stats
    # a classification of joined rows follows the join that keeps them,
    # in the join's pipeline: each kept row's partner block, question,
    # and label paths run over the anchor's resident KV
    for call, aliases in joined_calls(builder.operators.labels):
        for g, group in enumerate(builder.sequence_groups):
            anchor = group[0][1]
            stage = next((spec for spec, _ in group
                          if set(spec.aliases) == set(aliases)), None)
            if stage is not None:
                break
        else:
            raise AssertionError(
                "a classification of joined rows without its join")
        partner = next(alias for alias in aliases if alias != anchor)
        table = classify_builder(builder.context, anchor, "quail")
        join = builder.operators.joins[stage.written_pos]
        pairs = (effective_selectivity(join.selectivity)
                 * builder.operator_order.live[anchor]
                 * builder.operator_order.live[partner])
        spec, _ = table.classify_joined(
            call, builder.operators.labels.names[call], partner, pairs,
            stats[partner].mean_doc_tokens)
        node = table.node(
            spec, PortRef(builder.join_ids[stage.written_pos],
                          f"join_answers:{stage.written_pos}"),
            sum(isinstance(n, AiClassify) for n in builder.nodes))
        builder.nodes.append(node)
        if call in builder.operators.labels.projected:
            builder.label_ports.append(PortRef(node.node_id, "rows"))
