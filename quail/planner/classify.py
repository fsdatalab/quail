"""Prepare classification prompts and choose a scoring method.

The planner compares estimated execution times for letters, trie_tree,
and trie_decode. It emits one AiClassify node for each classification;
decoder rounds are handled by the executor.
"""

from dataclasses import dataclass, replace

from quail.cost import budgets
from quail.cost import classify as classify_cost
from quail.execution.pipelines import build_pipelines
from quail.labels import (
    DECODE_SCORING,
    LETTERS_SCORING,
    TREE_SCORING,
    decodable,
)
from quail.logical import (
    SemanticClassify,
    label_text,
)
from quail.physical import (
    AiClassify,
    AiJoin,
    ClassifySpec,
)
from quail.physical.base import input_ports
from quail.planner.plan import Refusal


def classification_refusal(context) -> Refusal | None:
    """Check whether the model supports classification planning.

    Args:
        context: Planning context with the tokenizer and model specification.

    Returns:
        A Refusal describing missing support, or None if planning can proceed.
    """
    model = context.model
    if context.tokenizer is None:
        reason = "AI.CLASSIFY planning needs the model's tokenizer"
    elif model.canvas_tokens and model.answer_canvas is None:
        reason = f"{model.name!r} names no answer canvas to read labels on"
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_classify_query",
                   needed=1, available=0, unit="queries")


def joined_classification_refusal(graph, workers: int) -> Refusal | None:
    """Check that each pair classification can run in its join's pipeline.

    Args:
        graph: Physical operator graph.
        workers: Number of model workers.

    Returns:
        A Refusal if a pair classification cannot share its source join's
        pipeline on one worker, or None if every pair classification can run.
    """
    pipelines = build_pipelines(graph) if workers == 1 else {}
    for node in graph.nodes:
        if not isinstance(node, AiClassify) or node.spec.partner is None:
            continue
        pipeline = pipelines.get(node.node_id)
        if pipeline is not None and any(
                isinstance(member, AiJoin)
                and any(port.source.node_id == member.node_id for port in node.inputs)
                for member in pipeline.members):
            continue
        return Refusal(
            reasons=(f"AI.CLASSIFY {node.spec.name!r} over a document pair "
                     "must run in its join's pipeline on one GPU; this plan "
                     "requires unsupported standalone pair classification",),
            constraint="joined_classify_pipeline", needed=1, available=0,
            unit="supported join pipelines")
    return None


def classify_table(context, alias: str, backend_name: str,
                   shared=()) -> "_Table":
    """Prepare prompts and estimate classification work for one table.

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
        chunk=chunk,
        draws=context.canvas_draws if context.model.answer_canvas else 1,
        backend_name=backend_name, model=context.model,
        device=context.device, tokenizer=context.tokenizer,
        capacity=capacity, lengths=tuple(lengths), shared=tuple(shared),
        tree=(context.model.weight_precision == "fp8"
              and not context.model.canvas_tokens
              and getattr(context, "attention", None) != "unified"))


def has_label(logical) -> bool:
    """Return whether a logical plan classifies documents or joined rows."""
    return any(isinstance(node, SemanticClassify) for node in logical.walk())


@dataclass(frozen=True)
class _Table:
    """Document lengths, model settings, and execution budgets for one table.

    Attributes:
        alias: Table alias in the query.
        mean: Mean document length in tokens.
        longest: Maximum document length in tokens.
        budget: Smaller of the forward-pass token budget and KV capacity.
        chunk: Maximum tokens per forward pass.
        backend_name: Backend executing the classification.
        model: Model specification.
        device: Device specification.
        tokenizer: Callable mapping text to token IDs.
        capacity: KV arena capacity in tokens.
        draws: Maximum diffusion draws per document; one for causal models.
        lengths: Document lengths in tokens.
        shared: Shared prefix length per document, or an empty tuple.
        tree: Whether tree attention is available for packed label scoring.
    """

    alias: str
    mean: float
    longest: int
    budget: int
    chunk: int
    backend_name: str
    model: object
    device: object
    tokenizer: object
    capacity: int = 0
    draws: int = 1
    lengths: tuple = ()
    shared: tuple = ()
    # whether the model runs the tree attention path, which the
    # packed trie needs: an fp8 model with the plan not forced unified
    tree: bool = False

    def head(self, call) -> tuple:
        """Return the prompt token sequence before the document."""
        return tuple(self.tokenizer(call.prompt.preamble))

    def classify(self, call, name, live, resident=False):
        """Build a classification specification and estimate its work.

        Args:
            call: Logical AI.CLASSIFY call.
            name: Result column name.
            live: Expected number of documents to classify.
            resident: Whether document KV is available from an earlier operator.

        Returns:
            A tuple containing the ClassifySpec and estimated Work.

        Raises:
            ClassifyRefusedError: No scoring method can run, or the document and
                prompt exceed a token or KV budget.
        """
        head = self.head(call)
        tail = tuple(call.prompt.tail_token_ids)
        prefix = call.prompt.label_prefix
        labels = tuple(tuple(self.tokenizer(label_text(label, prefix)))
                       for label in call.labels)
        lettered = None
        if call.prompt.lettered is not None:
            letter_head = tuple(self.tokenizer(call.prompt.lettered.preamble))
            letter_tail = tuple(call.prompt.lettered.tail_token_ids)
            letter_ids = tuple(tuple(self.tokenizer(label_text(letter, prefix)))
                               for letter in call.prompt.lettered.letters)
            lettered = (len(letter_head), len(letter_tail) - 1, letter_ids)
        scoring, simulated = self.choose(live, len(head), len(tail) - 1,
                                         labels, resident, lettered,
                                         probabilities=call.probabilities)
        if scoring == LETTERS_SCORING:
            head, tail, labels = letter_head, letter_tail, letter_ids
        if scoring == DECODE_SCORING and not decodable(labels):
            raise ClassifyRefusedError(
                f"the {scoring!r} rule cannot decode the labels of "
                f"{name!r}: one label is a proper prefix of another",
                1, 0)
        suffixes = classify_cost.suffix_lengths(scoring, labels, self.canvas_rows)
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
            draws=self.draws if scoring == LETTERS_SCORING else 1,
            probabilities=call.probabilities,
        )
        return spec, simulated.work

    @property
    def canvas_rows(self) -> int:
        """Return the number of answer canvas rows, or zero without a canvas."""
        canvas = self.model.answer_canvas
        return canvas.rows if canvas is not None else 0

    def simulate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident) -> classify_cost.Simulated:
        """Estimate one scoring method using this table's lengths and budgets.

        Args:
            scoring: Scoring method name.
            live: Expected number of documents to classify.
            head_tokens: Number of prompt tokens before each document.
            frame_tokens: Number of prompt tokens written after each document.
            labels: Token sequence for each category.
            resident: Whether document KV is already available.

        Returns:
            Estimated execution time, work, and token counts.
        """
        return classify_cost.estimate(
            scoring, live, head_tokens, frame_tokens, labels,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device, resident=resident, draws=self.draws)

    def reestimate(self, spec: ClassifySpec, *, resident=False) -> ClassifySpec:
        """Update a classification's estimated time for this table.

        Args:
            spec: Classification specification to update.
            resident: Whether document KV is already available.

        Returns:
            A copy of spec with estimated_seconds recalculated.
        """
        head, tail = spec.prompt_token_parts
        simulated = self.simulate(
            spec.scoring, spec.expected_inputs, len(head), len(tail) - 1,
            spec.label_token_ids, resident)
        return replace(spec, estimated_seconds=simulated.seconds)

    def choose(self, live, head_tokens, frame_tokens, labels, resident,
               lettered=None, probabilities=False
               ) -> tuple[str, classify_cost.Simulated]:
        """Choose the supported scoring method with the lowest estimated time.

        Letters requires a prompt with one distinct token per category. Tree
        scoring requires tree attention. Greedy decoding requires documents
        without resident KV and labels that do not contain another label's
        complete token sequence as a prefix. Greedy decoding is excluded when
        probabilities are requested. Ties favor fewer suffix tokens, then the
        first candidate.

        Args:
            live: Expected number of documents to classify.
            head_tokens: Number of prompt tokens before the document.
            frame_tokens: Number of prompt tokens after the document, excluding
                the final answer cue token.
            labels: Token sequence for each category.
            resident: Whether document KV is available from an earlier operator.
            lettered: Tuple of head length, frame length, and category letter
                token sequences, or None if the prompt has no lettered form.
            probabilities: Whether every category's probability is required.

        Returns:
            A tuple containing the method name and its simulated cost.

        Raises:
            ClassifyRefusedError: No supported scoring method can run.
        """
        candidates = [LETTERS_SCORING] if lettered is not None else []
        if self.tree:
            candidates.append(TREE_SCORING)
        if (not self.model.canvas_tokens and not resident
                and not probabilities and decodable(labels)):
            candidates.append(DECODE_SCORING)
        if not candidates:
            raise ClassifyRefusedError(
                "no label scoring rule can run: no one-token letter for "
                "every label, no tree attention for the packed trie, and "
                "no greedy decode (resident documents, or a label that is "
                "another's prefix)", 1, 0)
        best = None
        for scoring in candidates:
            head, frame, ids = ((head_tokens, frame_tokens, labels)
                                if scoring != LETTERS_SCORING else lettered)
            simulated = self.simulate(scoring, live, head, frame, ids,
                                      resident)
            key = (simulated.seconds, simulated.suffix_tokens)
            if best is None or key < best[0]:
                best = (key, scoring, simulated)
        return best[1], best[2]

    def node(self, spec, input_port, index, *ports) -> AiClassify:
        """Build the physical node for one classification."""
        return AiClassify(node_id=f"ai-classify:{index}",
                          inputs=input_ports((input_port, *ports)),
                          backend_name=self.backend_name,
                          model=self.model.name, spec=spec)

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
        prompt = call.prompt.lettered
        if prompt is None:
            raise ClassifyRefusedError(
                "the letters rule needs a one-token letter for every "
                "label, which the tokenizer does not have", 1, 0)
        head = tuple(prompt.preamble_token_ids)
        tail = tuple(prompt.tail_token_ids)
        parts = {alias: (label, frame)
                 for alias, label, frame in prompt.label_token_ids}
        note, partner_label = parts[self.alias][1], parts[partner][0]
        labels = tuple(
            tuple(self.tokenizer(label_text(letter, prompt.label_prefix)))
            for letter in prompt.letters)
        block = len(partner_label) + int(round(partner_tokens)) + len(tail) - 1
        chains = [
            block + length
            for length in classify_cost.suffix_lengths(LETTERS_SCORING, labels)
        ]
        simulated = classify_cost.estimate_chains(
            len(head), len(note), chains, live=pairs, lengths=self.lengths,
            shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device, resident=True,
            canvas_rows=self.model.canvas_tokens)
        need = len(head) + self.longest + len(note) + max(chains)
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a row of {self.alias!r} x {partner!r} needs {need} tokens "
                f"with its classification prompt, but the forward pass "
                f"budget is {self.budget} tokens", need, self.budget)
        return ClassifySpec(
            name=name, aliases=(self.alias, partner),
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=pairs, estimated_seconds=simulated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels, scoring=LETTERS_SCORING,
            join_layout=(tuple(note), tuple(partner_label)),
        ), simulated.work


class ClassifyRefusedError(Exception):
    """A classification cannot run within the selected execution constraints."""

    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available

    def refusal(self) -> Refusal:
        """Convert this exception to a planning Refusal."""
        return Refusal(reasons=(self.reason,), constraint="suffix_over_chunk",
                       needed=self.needed, available=self.available, unit="tokens")
