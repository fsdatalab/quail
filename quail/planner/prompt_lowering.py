"""Lower a prompt's text to the token ids and counts one model reads.

A logical plan's prompts are text. The planner lowers each once per
query with the session's tokenizer, through the planning context, and
every count the cost model and the executor read comes from the
lowered prompt. A classification prompt is also lowered in its
lettered form when the tokenizer has a one-token letter per label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quail.logical.expressions import Prompt
from quail.logical.prompts import (
    LABEL_PREFIX,
    MAX_LETTERS,
    PROMPT_LAYOUTS,
    CompileError,
    bind_classify_prompt,
    bind_join_prompt,
    bind_prompt,
    label_text,
    letter_candidates,
    render_qwen3_reranker_input,
    score_query_template,
)

_PLACEHOLDER = re.compile(r"\{\d+\}")


@dataclass(frozen=True)
class LoweredPrompt:
    """A prompt's text with the token ids one tokenizer gives it.

    Attributes:
        prompt: The text prompt.
        preamble_token_ids: The preamble before the document.
        tail_token_ids: The text after the document, or after a join's
            last partner.
        frame_tokens: Tokens of the frame after the anchor document.
        label_token_ids: Per placeholder of a join or a classification
            of joined rows, (alias, its label's ids, its frame's ids).
        lettered: The classification prompt lowered with one letter
            per label, or None.
    """

    prompt: Prompt
    preamble_token_ids: tuple
    tail_token_ids: tuple
    frame_tokens: int
    label_token_ids: tuple = ()
    lettered: "LoweredPrompt | None" = None

    @property
    def preamble_tokens(self) -> int | None:
        """The preamble's token count; None for a score, laid out later."""
        return None if self.prompt.kind == "score" else len(self.preamble_token_ids)

    @property
    def tail_tokens(self) -> int | None:
        """The tail's token count; None for a score, laid out later."""
        return None if self.prompt.kind == "score" else len(self.tail_token_ids)

    @property
    def labels(self) -> tuple:
        """Per placeholder, (alias, label tokens, frame tokens)."""
        return tuple((alias, len(label), len(frame))
                     for alias, label, frame in self.label_token_ids)

    @property
    def template(self) -> str:
        return self.prompt.template

    @property
    def args(self) -> tuple:
        return self.prompt.args

    @property
    def preamble(self) -> str:
        return self.prompt.preamble

    @property
    def tail(self) -> str:
        return self.prompt.tail

    @property
    def layout(self) -> str:
        return self.prompt.layout

    @property
    def tail_segments(self) -> tuple:
        return self.prompt.tail_segments

    @property
    def label_prefix(self) -> str:
        return self.prompt.label_prefix

    @property
    def letters(self) -> tuple:
        return self.prompt.letters


def tokenize_segments(segments, tokenizer) -> tuple:
    """Token ids of each segment, joined."""
    return tuple(t for segment in segments for t in tokenizer(segment))


def tail_segments(prompt) -> tuple[str, ...]:
    """The text after a filter document or a join's last partner, by segment."""
    if prompt.tail_segments:
        return prompt.tail_segments
    return (_PLACEHOLDER.sub("", prompt.tail),)


def choice_letters(count: int, tokenizer=None,
                   prefix: str = LABEL_PREFIX) -> tuple[str, ...]:
    """Choose distinct one-token letters to stand for the labels.

    Args:
        count: Number of letters needed, one per label.
        tokenizer: Optional callable mapping text to token IDs. Without one,
            candidate strings are returned in their original order.
        prefix: Text immediately before the scored letter.

    Returns:
        Letters in label order. The result may contain fewer than count
        entries if the tokenizer has too few distinct one-token candidates.

    Raises:
        CompileError: The requested count exceeds MAX_LETTERS.
    """
    if count > MAX_LETTERS:
        raise CompileError(
            f"AI.CLASSIFY takes at most {MAX_LETTERS} labels, got {count}")
    letters, seen = [], set()
    for letter in letter_candidates():
        if tokenizer is not None:
            ids = tuple(tokenizer(prefix + letter))
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
        letters.append(letter)
        if len(letters) == count:
            break
    return tuple(letters)


def lower_prompt(prompt: Prompt, tokenizer) -> LoweredPrompt:
    """Tokenize a prompt's pieces, and letter a classification when possible.

    Args:
        prompt: The text prompt.
        tokenizer: Callable text -> token ids, without special tokens.
    """
    if prompt.kind == "score":
        return LoweredPrompt(prompt, (), (), 0)
    lettered = None
    if (prompt.kind == "label" and prompt.categories and not prompt.letters
            and PROMPT_LAYOUTS[prompt.layout].choice_label is None):
        letters = choice_letters(
            len(prompt.categories), tokenizer, prompt.label_prefix)
        if len(letters) == len(prompt.categories):
            lettered = lower_prompt(bind_classify_prompt(
                prompt.source, prompt.args, prompt.categories,
                prompt.descriptions, turn=prompt.turn,
                task_description=prompt.task_description,
                layout=prompt.layout, letters=letters), tokenizer)
    return LoweredPrompt(
        prompt=prompt,
        preamble_token_ids=tuple(tokenizer(prompt.preamble)),
        tail_token_ids=tokenize_segments(tail_segments(prompt), tokenizer),
        frame_tokens=(len(tokenizer(prompt.frame_text))
                      if prompt.frame_text else 0),
        label_token_ids=tuple(
            (alias, tuple(tokenizer(label)), tuple(tokenizer(frame)))
            for alias, label, frame in prompt.label_texts),
        lettered=lettered)


def true_false_ids(tokenizer):
    """Return TRUE and FALSE token sets from a Hugging Face tokenizer."""
    true, false = true_false_token_ids(
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"])
    return set(true), set(false)


def true_false_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """The sorted token ids that mean TRUE and FALSE.

    Args:
        tokenizer: Callable text -> token ids, without special tokens.
    """
    true, false = set(), set()
    for words, ids in (
            (("TRUE", " TRUE", "True", " True"), true),
            (("FALSE", " FALSE", "False", " False"), false)):
        for word in words:
            tokens = tokenizer(word)
            if tokens:
                ids.add(tokens[0])
    return sorted(true), sorted(false)


def answer_row_offsets(layout: str, tokenizer,
                       turn_suffix: str = "") -> tuple[int, ...]:
    """Per row a readout reads, its distance before the prompt's last row.

    Args:
        layout: A PROMPT_LAYOUTS name.
        tokenizer: Callable text -> token ids, without special tokens.
        turn_suffix: The model's chat-turn text after the prompt.

    Returns:
        The offsets of the answer_rows segment ends, in order, then 0.
    """
    spec = PROMPT_LAYOUTS[layout]
    segments = list(spec.answer_segments)
    if not segments:
        return (0,)
    segments[-1] += turn_suffix
    lengths = [len(tokenizer(segment)) for segment in segments]
    return tuple(sum(lengths[i + 1:]) for i in spec.answer_rows) + (0,)


def render_filter_prompt_ids(lowered: LoweredPrompt, document_ids) -> list:
    """The complete canonical token ids for one filter document."""
    return (list(lowered.preamble_token_ids) + list(document_ids)
            + list(lowered.tail_token_ids))


def render_join_prompt_ids(lowered: LoweredPrompt, documents,
                           anchor: int) -> list:
    """The complete canonical token ids for one join tuple."""
    if len(documents) != len(lowered.args):
        raise ValueError(
            f"join has {len(lowered.args)} placeholders but received "
            f"{len(documents)} documents")
    if anchor < 0 or anchor >= len(lowered.args):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    pieces = {index: (label, frame) for index, (_, label, frame)
              in enumerate(lowered.label_token_ids)}
    out = (list(lowered.preamble_token_ids) + list(documents[anchor])
           + list(pieces[anchor][1]))
    for index, document_ids in enumerate(documents):
        if index != anchor:
            out += list(pieces[index][0]) + list(document_ids)
    return out + list(lowered.tail_token_ids)


def score_token_parts(prompt, model, tokenizer) -> tuple[tuple[int, ...], ...]:
    """Tokenize the fixed pieces of the rendered prompt around the documents."""
    if model.role != "reranker":
        return _answer_token_parts(prompt, model, tokenizer)
    query_template = score_query_template(prompt)
    rendered = render_qwen3_reranker_input(query_template, "{document}")
    before, after = rendered.split("{document}")
    parts = (before, after)
    aliases = tuple(dict.fromkeys(argument.alias for argument in prompt.args))
    if len(aliases) == 2:
        first, middle = before.split("{0}")
        parts = (first, middle, after)
    # Documents are tokenized separately; tokens cannot span these boundaries.
    # These ids can differ from tokenizing the complete prompt string.
    return tuple(tuple(tokenizer(part)) for part in parts)


def _answer_token_parts(prompt, model, tokenizer) -> tuple[tuple[int, ...], ...]:
    """Tokenize the AI.IF layout around the documents, for a generative model.

    A document gets the AI.IF filter prompt and a pair the AI.IF join
    prompt anchored on its first document.
    """
    turn = model.turn
    layout = model.prompt_layout
    aliases = tuple(dict.fromkeys(argument.alias for argument in prompt.args))
    if len(aliases) == 1:
        bound = lower_prompt(
            bind_prompt(prompt.template, prompt.args, turn, layout), tokenizer)
        return tuple(bound.preamble_token_ids), tuple(bound.tail_token_ids)
    if len(prompt.args) != 2:
        raise ValueError("AI.SCORE supports one document or one document pair")
    bound = lower_prompt(
        bind_join_prompt(prompt.template, prompt.args, turn, layout), tokenizer)
    pieces = {alias: (label, frame) for alias, label, frame in bound.label_token_ids}
    left, right = aliases
    return (tuple(bound.preamble_token_ids),
            tuple(pieces[left][1]) + tuple(pieces[right][0]),
            tuple(bound.tail_token_ids))


def answer_ids(model, tokenizer) -> tuple[list[int], list[int]]:
    """The token ids a score compares: yes and no, or TRUE and FALSE.

    Args:
        model: The ModelSpec; a reranker answers yes or no.
        tokenizer: Callable text -> token ids.
    """
    if model.role == "reranker":
        return list(tokenizer("yes")), list(tokenizer("no"))
    return true_false_token_ids(tokenizer)


def classify_prompt_tokens(lowered: LoweredPrompt, labels, tokenizer) -> tuple:
    """Return a classification's head, tail, and label token sequences."""
    return (
        tuple(lowered.preamble_token_ids), tuple(lowered.tail_token_ids),
        tuple(tuple(tokenizer(label_text(label, lowered.label_prefix)))
              for label in labels))


def choice_token_parts(lowered: LoweredPrompt, tokenizer) -> tuple:
    """Return decision classification tail tokens, frame length, and option blocks."""
    segments = lowered.tail_segments
    frame = len(tokenizer(segments[0]))
    blocks = tuple(tuple(tokenizer(block)) for block in segments[1:-1])
    return tuple(lowered.tail_token_ids), frame, blocks
