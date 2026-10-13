"""Lower a score, classification, or extraction to its spec and node."""

from dataclasses import dataclass, replace

from quail.cost import budgets
from quail.cost import classify as classify_cost
from quail.cost import extract as extract_cost
from quail.cost.work import Work
from quail.labels import DECISION_SCORING, LETTERS_SCORING, word_starts
from quail.logical.prompts import (
    choice_token_parts,
    classify_prompt_tokens,
    prompt_aliases,
    score_query_template,
    shared_preamble,
)
from quail.physical import (
    AiClassify,
    AiScore,
    ClassifySpec,
    Comparison,
    ExtractSpec,
    Filter,
    PortRef,
    ScoreSpec,
)
from quail.physical.base import input_ports
from quail.planner.plan import Refusal
from quail.planner.statistics import (
    ClassifyStatistics,
    classify_statistics,
    score_statistics,
)
from quail.planner.validation import (
    ClassifyRefusedError,
    ScoreRefusedError,
    score_input_tokens,
)


def bind_inference(spec_type, prompt, *, name, expected_inputs,
                   prompt_token_parts, aliases=None, query_template=None,
                   estimated_seconds=0.0, **settings):
    """Build an operator's spec from its bound prompt and execution settings."""
    return spec_type(
        name=name,
        aliases=prompt_aliases(prompt) if aliases is None else aliases,
        query_template=(prompt.template if query_template is None
                        else query_template),
        arguments=tuple((ref.alias, ref.column) for ref in prompt.args),
        expected_inputs=expected_inputs, estimated_seconds=estimated_seconds,
        prompt_token_parts=prompt_token_parts, **settings)


def inference_node(node_type, spec, sources, *, index, backend, model):
    """Connect a model call to its input rows or document IDs."""
    name = node_type.type_name.removeprefix("quail.").replace("_", "-")
    return node_type(
        node_id=f"{name}:{index}", inputs=input_ports(tuple(sources)),
        backend_name=backend, model=model, spec=spec)


def score_name(prompt, projected, fallback: str) -> str:
    # the front end rejects one prompt projected under two names
    return next(
        (item.name for item in projected if item.expression.prompt == prompt),
        fallback,
    )


def score_spec(prompt, *, name, expected_inputs, mean_tokens, context,
               chunk_tokens, pair_fraction=1.0, prefix_groups=None):
    """Build a score specification from prepared prompt tokens and costs."""
    parts, cost = score_statistics(
        prompt, context, chunk_tokens, mean_tokens=mean_tokens)
    work, seconds = cost.estimate(expected_inputs, prefix_groups=prefix_groups)
    return bind_inference(
        ScoreSpec, prompt, name=name,
        query_template=score_query_template(prompt),
        expected_inputs=expected_inputs, estimated_seconds=seconds,
        pair_fraction=pair_fraction, prompt_token_parts=parts, draws=cost.draws), work


class ScoreLowering:
    """Append score and comparison operators to a shared physical graph."""

    def __init__(self, context, nodes, projected, chunk):
        self.context = context
        self.nodes = nodes
        self.projected = projected
        self.chunk = chunk
        self.capacity = budgets.arena_tokens(context.model, context.device, chunk)
        self.names = {}
        self.work = Work()
        self.index = 0

    def score(self, prompt, fallback, inputs, expected, mean,
              *, pair_fraction=1.0, prefix_groups=None) -> PortRef:
        name = score_name(prompt, self.projected, fallback)
        spec, work = score_spec(
            prompt, name=name, expected_inputs=expected, mean_tokens=mean,
            context=self.context, chunk_tokens=self.chunk,
            pair_fraction=pair_fraction, prefix_groups=prefix_groups)
        prefix, suffix = score_input_tokens(spec, self.context)
        what = (f"a document in {spec.aliases[0]!r}" if len(spec.aliases) == 1
                else f"one pair of {spec.aliases[0]!r} and {spec.aliases[1]!r}")
        needed, available, unit = prefix + suffix, self.chunk, "tokens"
        if needed <= available:
            needed = budgets.pages(prefix) + budgets.pages(suffix)
            available, unit = self.capacity // budgets.PAGE_TOKENS, "pages"
        if needed > available:
            raise ScoreRefusedError(Refusal(
                reasons=(f"{what} needs {needed} {unit} with its prompt, "
                         f"but the execution budget is {available} {unit}",),
                constraint="suffix_over_chunk", needed=needed,
                available=available, unit=unit))
        node = inference_node(
            AiScore, spec, inputs, index=self.index,
            backend=self.context.backend, model=self.context.model.name)
        self.index += 1
        self.nodes.append(node)
        self.names[prompt] = name
        self.work += work
        return PortRef(node.node_id, "rows")

    def comparison(self, predicate, source, aliases, position) -> PortRef:
        alias = aliases[0] if len(aliases) == 1 else "join"
        expression = (predicate.expression if len(aliases) == 1
                      else predicate.predicate)
        node = Filter(
            node_id=f"filter:{alias}:{position}",
            inputs=input_ports((source,)),
            predicate=Comparison(self.names[predicate.prompt],
                                 expression.comparison, expression.threshold),
            aliases=tuple(aliases), selectivity=predicate.selectivity,
            written_pos=position)
        self.nodes.append(node)
        return PortRef(node.node_id, "rows")


def extract_spec(context, call, name, live, shared=False) -> ExtractSpec:
    """Build an extraction's specification and price it.

    The specification carries the token ids of the prompt around the
    document. The documents' KV is priced as computed afresh unless
    the extraction shares a pipeline with the one before it, which
    read the documents.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.EXTRACT call.
        name: Result column name.
        live: Expected number of documents to extract from.
        shared: Whether an extraction of the same table runs right
            before this one.
    """
    prompt = call.prompt
    # under the query's shared preamble, the prompt's document KV is
    # the one a filter writes
    shares_head = context.tokenizer is not None and tuple(
        prompt.preamble_token_ids) == tuple(context.tokenizer(shared_preamble(
            context.model.turn_prefix, context.model.prompt_layout)))
    (ref,) = call.prompt.args
    lengths = context.document_tokens[ref.alias]
    spec = bind_inference(
        ExtractSpec, prompt, name=name,
        expected_inputs=live, estimated_seconds=0.0,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        question=call.question, trim=call.trim, shares_head=shares_head)
    cost = extract_cost.estimate(
        live, lengths, len(prompt.preamble_token_ids), len(prompt.tail_token_ids),
        chunk=budgets.chunk_budget(context.model, context.device),
        model=context.model, device=context.device, resident=shared,
        max_answer_tokens=spec.max_answer_tokens, draft_tokens=spec.draft_tokens,
        workers=context.gpu_count)
    return replace(spec, estimated_seconds=cost.seconds)


def classify_builder(context, alias, backend_name):
    """Prepare the inputs for constructing classification operators."""
    stats = classify_statistics(context, alias)
    return ClassifyBuilder(**vars(stats), tokenizer=context.tokenizer,
                           backend_name=backend_name)


@dataclass(frozen=True, kw_only=True)
class ClassifyBuilder(ClassifyStatistics):
    """Construct classification specifications and operators."""

    tokenizer: object
    backend_name: str

    def prepare(self, call, name, live) -> tuple[ClassifySpec, Work]:
        """Build a classification specification and work for a required method.

        The specification carries the prompt that names the labels; the
        label_scoring rule picks the scoring rule, swaps in the
        lettered prompt when it picks letters, and fills the estimate.

        Args:
            call: Logical AI.CLASSIFY call.
            name: Result column name.
            live: Expected number of documents to classify.

        Returns:
            The specification and its work when the model requires a scoring
            method. Otherwise, the physical label_scoring rule chooses the
            method and fills in its work later.
        """
        if self.model.role == "decision":
            return self._choice_spec(call, name, live, (self.alias,),
                                     tuple(self.tokenizer(call.prompt.preamble)))
        head, tail, labels = classify_prompt_tokens(
            call.prompt, call.labels, self.tokenizer)
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
        prompt = call.prompt.lettered
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
        tail, frame, blocks = choice_token_parts(call.prompt, self.tokenizer)
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
        prompt = call.prompt
        head = tuple(prompt.preamble_token_ids)
        tail, frame, blocks = choice_token_parts(call.prompt, self.tokenizer)
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
