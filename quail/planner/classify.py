"""Prompt preparation and scoring rule selection for AI.CLASSIFY operators.

The prompt head and document stay in KV, the question and category list
are written once after it, and a suffix reads the next token's log probabilities.
Under the ``letters`` rule the categories are lettered, so every
label is one token and the cue's row scores them all; under
``trie_tree`` the one suffix holds the labels' own tokens as a trie,
and under ``trie_decode`` a document decodes its label one token a
round along the trie. A diffusion model reads the letter at the first
row of a seeded answer canvas in one denoising step.
"""

from dataclasses import dataclass, replace

from quail.cost import budgets
from quail.cost import classify as classify_cost
from quail.labels import (
    DECODE_SCORING,
    LABEL_SCORINGS,
    LETTERS_SCORING,
    TREE_SCORING,
    decodable,
)
from quail.logical import (
    Alias,
    LabelIn,
    label_text,
)
from quail.physical import (
    AiClassify,
    ClassifySpec,
)
from quail.physical.base import input_ports
from quail.planner.plan import Refusal


def classification_refusal(context) -> Refusal | None:
    """Why the context's model cannot classify, or None when it can."""
    model = context.model
    forced = context.label_scoring
    if context.tokenizer is None:
        reason = "AI.CLASSIFY planning needs the model's tokenizer"
    elif forced is not None and forced not in LABEL_SCORINGS:
        reason = (f"unknown label scoring rule {forced!r}; "
                  f"the rules are {LABEL_SCORINGS}")
    elif model.canvas_tokens and forced not in (None, LETTERS_SCORING):
        reason = (f"{model.name!r} reads labels on a canvas; the "
                  f"{forced!r} rule reads the rows of a causal chain")
    elif model.canvas_tokens and model.answer_canvas is None:
        reason = f"{model.name!r} names no answer canvas to read labels on"
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_classify_query",
                   needed=1, available=0, unit="queries")


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
        chunk=chunk, scoring=context.label_scoring,
        draws=context.canvas_draws if context.model.answer_canvas else 1,
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
        draws: The noise draws a letters read on a canvas model may
            average; 1 on a causal model.
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
    draws: int = 1
    lengths: tuple = ()
    shared: tuple = ()
    # whether the model runs the tree attention path, which the
    # packed trie needs: an fp8 model with the plan not forced unified
    tree: bool = False

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
                                         labels, resident, lettered)
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
        )
        return spec, simulated.work

    @property
    def canvas_rows(self) -> int:
        """The rows of the model's answer canvas; 0 without one."""
        canvas = self.model.answer_canvas
        return canvas.rows if canvas is not None else 0

    def simulate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident) -> classify_cost.Simulated:
        """Price a prepared scoring candidate under this table's budgets."""
        return classify_cost.estimate(
            scoring, live, head_tokens, frame_tokens, labels,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device, resident=resident, draws=self.draws)

    def reestimate(self, spec: ClassifySpec, *, resident=False) -> ClassifySpec:
        """The spec with its seconds simulated again on this table.

        Documents borrowing a prefix pay only for the rest. A later
        classification in a pipeline reads resident document KV.
        """
        head, tail = spec.prompt_token_parts
        simulated = self.simulate(
            spec.scoring, spec.expected_inputs, len(head), len(tail) - 1,
            spec.label_token_ids, resident)
        return replace(spec, estimated_seconds=simulated.seconds)

    def choose(self, live, head_tokens, frame_tokens, labels, resident,
               lettered=None) -> tuple[str, classify_cost.Simulated]:
        """The rule with the least simulated time, and its simulation.

        A forced rule is the one candidate. Otherwise ``letters`` is a
        candidate when the prompt has a lettered form, priced with that
        prompt; on a causal model ``trie_tree`` under tree attention,
        and ``trie_decode`` for a classification whose documents are
        not resident from an earlier stage and whose labels a greedy
        decode can end at; a canvas model reads letters only. Ties go
        to fewer label tokens, then the order listed.

        Args:
            live: How many documents are expected.
            head_tokens: The prompt head's tokens.
            frame_tokens: The tail's tokens but its last.
            labels: Each label's token ids.
            resident: Whether the documents' KV is resident already.
            lettered: (head tokens, frame tokens, letter token ids) of
                the lettered prompt; None when the prompt has none.

        Raises:
            ClassifyRefusedError: No rule can run: ``letters`` without a
                lettered prompt, or ``trie_tree`` without tree attention.
        """
        if self.scoring:
            candidates = [self.scoring]
        else:
            candidates = [LETTERS_SCORING] if lettered is not None else []
            if self.tree:
                candidates.append(TREE_SCORING)
            if (not self.model.canvas_tokens and not resident
                    and decodable(labels)):
                candidates.append(DECODE_SCORING)
        if lettered is None and LETTERS_SCORING in candidates:
            raise ClassifyRefusedError(
                "the letters rule needs a one-token letter for every "
                "label, which the tokenizer does not have", 1, 0)
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
        """The plan node running one classification."""
        return AiClassify(node_id=f"ai-classify:{index}",
                          inputs=input_ports((input_port, *ports)),
                          backend_name=self.backend_name,
                          model=self.model.name, spec=spec)

    def classify_joined(self, call, name, partner, pairs, partner_tokens):
        """Return a ClassifySpec for a classification of joined rows.

        Each row the join kept is priced as a suffix over the anchor's
        resident KV: the partner's label and document, the question,
        and the cue, whose row scores the letters (the ``letters``
        rule).

        Args:
            call: The logical AI.CLASSIFY call over the two aliases.
            name: The output column.
            partner: The partner table's alias.
            pairs: How many joined rows are expected to reach it.
            partner_tokens: The partner documents' mean length.

        Raises:
            ClassifyRefusedError: A forced rule other than letters, or
                a prompt without a lettered form.
        """
        if self.scoring not in (None, LETTERS_SCORING):
            raise ClassifyRefusedError(
                f"a classification of joined rows reads letters "
                f"({LETTERS_SCORING}); the {self.scoring!r} rule was forced",
                1, 0)
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
    """A classification the forward pass budget cannot hold."""

    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available

    def refusal(self) -> Refusal:
        """The planning refusal this exception stands for."""
        return Refusal(reasons=(self.reason,), constraint="suffix_over_chunk",
                       needed=self.needed, available=self.available, unit="tokens")
