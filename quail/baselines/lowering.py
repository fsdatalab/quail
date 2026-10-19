"""Lowering each operator's prompt into the request node's token ids."""

from __future__ import annotations

import re

from quail.baselines.nodes import (
    RequestClassifySpec,
    RequestExtractSpec,
    RequestFilterSpec,
    RequestJoinSpec,
)
from quail.logical import (
    filter_question_text,
    join_label,
    join_outer_input,
    label_text,
    render_join_frame,
    shared_preamble,
    without_bos,
)


def lower_filters(context, aliases, filters, ask_filters, filter_orders,
                  first_anchor) -> list[RequestFilterSpec]:
    """Build one filter spec per alias with AI.IF filters, anchor last."""
    filter_specs = []
    for alias in sorted(aliases, key=lambda alias: alias == first_anchor):
        predicates = filters.get(alias, ())
        if alias not in ask_filters:
            continue
        written_positions = tuple(filter_orders[alias])
        questions = tuple(
            tuple(context.lowered(predicates[position].prompt).tail_token_ids)
            for position in written_positions
        )
        question_texts = tuple(
            filter_question_text(predicates[position].prompt)
            for position in written_positions
        )
        if any(not question for question in questions):
            raise ValueError("filter prompts have no token ids")
        filter_specs.append(RequestFilterSpec(
            alias=alias,
            written_positions=written_positions,
            question_token_ids=questions,
            question_texts=question_texts,
        ))
    return filter_specs


def lower_classifications(context, operators,
                          decides: bool) -> list[RequestClassifySpec]:
    """Build one classify spec per AI.CLASSIFY call."""
    joins = operators.joins
    # each classification labels the alias's documents after its AI.IF
    # sequence; the filters on its labels keep the accepted documents for the joins
    label_plan = operators.labels
    classify_specs = []
    for call, alias in label_plan.calls:
        pair = {}
        if len(call.aliases()) == 2:
            # a classification of the rows a join keeps: decoded after
            # that join, one request per row
            partner = call.aliases()[1]
            parts = {part_alias: (label, frame)
                     for part_alias, label, frame
                     in context.lowered(call.prompt).label_token_ids}
            written_pos = next(
                (position for position, join in enumerate(joins)
                 if {argument.alias for argument in join.prompt.args}
                 == set(call.aliases())), None)
            if written_pos is None:
                raise ValueError(
                    f"the classification of {alias!r} x {partner!r} pairs "
                    f"needs a join of the two")
            pair = dict(partner=partner,
                        join_layout_token_ids=(tuple(parts[alias][1]),
                                        tuple(parts[partner][0])),
                        join_written_pos=written_pos)
        classify_specs.append(RequestClassifySpec(
            alias=alias,
            output=label_plan.names[call],
            tail_token_ids=tuple(context.lowered(call.prompt).tail_token_ids),
            labels=tuple(call.labels),
            label_token_ids=tuple(
                tuple(context.tokenizer(label_text(label)))
                for label in call.labels),
            tests=tuple(
                (test.position, test.values)
                for test in operators.label_filters.get(alias, ())
                if test.call == call),
            option_offsets=(_option_offsets(call.prompt, context.tokenizer)
                            if decides else ()),
            **pair,
        ))
    return classify_specs


def lower_joins(context, joins, join_sequence) -> list[RequestJoinSpec]:
    """Build one join spec per join, in the searched sequence."""
    join_specs = []
    for written_pos, selected_anchor in join_sequence:
        join = joins[written_pos]
        prompt = context.lowered(join.prompt)
        aliases_in_prompt = tuple(argument.alias for argument in prompt.args)
        labels = tuple(
            (str(alias), tuple(label))
            for alias, label, _frame in prompt.label_token_ids
        )
        frames = tuple(
            (str(alias), tuple(frame))
            for alias, _label, frame in prompt.label_token_ids
        )
        text_parts = tuple(
            (
                argument.alias,
                join_label(index),
                render_join_frame(prompt.template, index),
            )
            for index, argument in enumerate(prompt.args)
        )
        if len(labels) != len(aliases_in_prompt) or not prompt.tail_token_ids:
            raise ValueError("join prompts have no token ids")
        outer_aliases = tuple(dict.fromkeys(
            field.alias for field in join_outer_input(join).output_schema()
        ))
        join_specs.append(RequestJoinSpec(
            written_pos=written_pos,
            aliases=aliases_in_prompt,
            outer_aliases=outer_aliases,
            anchor=selected_anchor,
            semantics=join.semantics,
            selectivity=join.selectivity,
            label_token_ids=labels,
            frame_token_ids=frames,
            tail_token_ids=tuple(prompt.tail_token_ids),
            label_texts=tuple(
                (alias, label) for alias, label, _frame in text_parts
            ),
            frame_texts=tuple(
                (alias, frame) for alias, _label, frame in text_parts
            ),
            tail_text=prompt.tail,
        ))
    return join_specs


def lower_extractions(context, extracts) -> list[RequestExtractSpec]:
    """Build one extract spec per AI.EXTRACT column."""
    # each extraction decodes one answer per surviving document of its
    # alias, after the joins; its prompt carries its own preamble, the
    # model's extract turn, so it stays out of the shared one
    extract_specs = []
    for node in extracts:
        extract_specs.append(RequestExtractSpec(
            alias=node.alias, output=node.name,
            question=node.call.question,
            tail_token_ids=tuple(
                context.lowered(node.call.prompt).tail_token_ids),
            tail_text=re.sub(r"\{\d+\}", "", node.call.prompt.tail),
            preamble_token_ids=tuple(
                context.lowered(node.call.prompt).preamble_token_ids),
            # vLLM adds the model's beginning-of-text token itself
            preamble_text=without_bos(node.call.prompt.preamble,
                                      context.model.bos_text)))
    return extract_specs


def request_preamble(context, prompts) -> tuple[tuple[int, ...], str]:
    """Return the one preamble the prompts share, as token ids and text."""
    preambles = {
        tuple(context.lowered(prompt).preamble_token_ids)
        for prompt in prompts
        if context.lowered(prompt).preamble_token_ids
    }
    if len(preambles) > 1:
        raise ValueError("request prompts have different preambles")
    preamble = next(iter(preambles), ())
    preamble_texts = {prompt.preamble for prompt in prompts}
    if len(preamble_texts) > 1:
        raise ValueError("request prompts have different preamble text")
    if not preamble and prompts and context.tokenizer is not None:
        preamble = tuple(context.tokenizer(
            shared_preamble(context.model.turn_prefix)))
    # vLLM adds the model's beginning-of-text token to a text prompt
    return tuple(preamble), without_bos(
        next(iter(preamble_texts), "")
        or shared_preamble(context.model.turn_prefix),
        context.model.bos_text,
    )


def _option_offsets(prompt, tokenizer) -> tuple[int, ...]:
    """Distances before a choice prompt's last row of each option's end, then 0.

    The tail's segments are the question, one block per option, and
    the close, each tokenized apart.
    """
    lengths = [len(tokenizer(segment)) for segment in prompt.tail_segments]
    return tuple(sum(lengths[index + 1:])
                 for index in range(1, len(lengths) - 1)) + (0,)
