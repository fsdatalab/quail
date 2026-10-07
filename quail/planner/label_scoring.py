"""Scoring-method choices and re-estimation for classification physical rules."""

from dataclasses import dataclass, replace

from quail.cost import classify as classify_cost
from quail.labels import (
    DECISION_SCORING,
    DECODE_SCORING,
    LETTERS_SCORING,
    TREE_SCORING,
    decodable,
)
from quail.logical.prompts import classify_prompt_tokens
from quail.physical import ClassifySpec
from quail.planner.statistics import ClassifyStatistics, classify_statistics
from quail.planner.validation import ClassifyRefusedError


def classify_scoring(context, alias, shared=()):
    """Prepare scoring-method choices for one table and execution context."""
    stats = classify_statistics(context, alias, shared)
    return ClassifyScoring(**vars(stats), tokenizer=context.tokenizer)


@dataclass(frozen=True, kw_only=True)
class ClassifyScoring(ClassifyStatistics):
    """Choose and price the scoring method of a prepared classification."""

    tokenizer: object

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
        labels = spec.label_token_ids
        lettered = None
        if call.prompt.lettered is not None:
            letter_head, letter_tail, letter_ids = classify_prompt_tokens(
                call.prompt.lettered, call.prompt.lettered.letters, self.tokenizer)
            lettered = (len(letter_head), len(letter_tail) - 1, letter_ids)
        scoring, estimated = self.choose(
            spec.expected_inputs, len(head), len(tail) - 1, labels, resident,
            lettered, probabilities=spec.probabilities)
        if scoring == LETTERS_SCORING:
            head, tail, labels = letter_head, letter_tail, letter_ids
        if scoring == DECODE_SCORING and not decodable(labels):
            raise ClassifyRefusedError(
                f"the {scoring!r} rule cannot decode the labels of "
                f"{spec.name!r}: one label is a proper prefix of another",
                1, 0)
        suffixes = classify_cost.suffix_lengths(scoring, labels, self.canvas_rows)
        # the head, longest document, frame, and longest suffix must fit
        # together within the forward pass and KV budget
        need = len(head) + self.longest + len(tail) + max(suffixes)
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a document in {self.alias!r} needs {need} tokens with its "
                f"classification prompt, but the forward pass budget is "
                f"{self.budget} tokens", need, self.budget)
        return replace(
            spec, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), label_token_ids=labels,
            scoring=scoring,
            draws=self.draws if scoring == LETTERS_SCORING else 1)

    @property
    def canvas_rows(self) -> int:
        """Return the number of answer canvas rows, or zero without a canvas."""
        canvas = self.model.answer_canvas
        return canvas.rows if canvas is not None else 0

    def estimate(self, scoring, live, head_tokens, frame_tokens, labels,
                 resident) -> classify_cost.ClassifyCost:
        """Estimate one scoring rule using this table's lengths and budgets.

        Args:
            scoring: Label scoring rule name.
            live: Expected number of documents to classify.
            head_tokens: Number of prompt tokens before each document.
            frame_tokens: Number of prompt tokens written after each document.
            labels: Token sequence for each label.
            resident: Whether document KV is already available.

        Returns:
            Estimated execution time, work, and token counts.
        """
        return classify_cost.estimate(
            scoring, live, head_tokens, frame_tokens, labels,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            model=self.model, device=self.device, resident=resident,
            draws=self.draws)

    def estimate_spec(self, spec: ClassifySpec,
                      resident=False) -> classify_cost.ClassifyCost:
        """Estimate a specification's chosen scoring rule on this table.

        Args:
            spec: Classification specification with its scoring rule.
            resident: Whether document KV is already available.
        """
        head, tail = spec.prompt_token_parts
        if spec.scoring == DECISION_SCORING:
            return classify_cost.estimate_chains(
                len(head), spec.frame_tokens, [len(tail) - spec.frame_tokens],
                live=spec.expected_inputs, lengths=self.lengths,
                shared=self.shared, chunk=self.chunk, model=self.model,
                device=self.device, resident=resident)
        return self.estimate(
            spec.scoring, spec.expected_inputs, len(head), len(tail) - 1,
            spec.label_token_ids, resident)

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
               lettered=None, probabilities=False
               ) -> tuple[str, classify_cost.ClassifyCost]:
        """Choose the supported scoring rule with the lowest estimated time.

        The letters rule requires a prompt with one distinct token per
        label. The trie_tree rule requires the tree attention path. The
        trie_decode rule requires a model without a canvas, documents
        without resident KV, no request for probabilities, and no label
        whose token sequence is a prefix of another's. Ties favor fewer
        suffix tokens, then the first candidate.

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

        Returns:
            A tuple containing the method name and its estimated cost.

        Raises:
            ClassifyRefusedError: No supported scoring rule can run.
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
            estimated = self.estimate(scoring, live, head, frame, ids, resident)
            key = (estimated.seconds, estimated.suffix_tokens)
            if best is None or key < best[0]:
                best = (key, scoring, estimated)
        return best[1], best[2]
