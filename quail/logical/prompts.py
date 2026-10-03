"""Prompt text, prompt token ids, and template binding."""

import json
from dataclasses import dataclass

from quail.logical.nodes import CompileError, Prompt

# Fixed preamble before every document. Must be a formatting label,
# not an instruction; instruction text here biases short-document
# completions. A model's chat-turn text (ModelSpec.turn), when it has
# any, wraps the whole prompt: its opening piece goes before this
# preamble and its closing piece after the answer cue, so the user
# message stays open across the reusable document prefix.
SHARED_PRE = "DOCUMENT:\n"


def shared_preamble(turn_prefix: str, layout: str = "ai-if") -> str:
    """The text before every document: the model's turn opener, then the label."""
    return turn_prefix + PROMPT_LAYOUTS[layout].document_label


def true_false_ids(tok):
    """The token ids that mean TRUE and FALSE.

    Args:
        tok: A HuggingFace tokenizer; called with add_special_tokens=False.
    """
    true, false = set(), set()
    for w in ("TRUE", " TRUE", "True", " True"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            true.add(ids[0])
    for w in ("FALSE", " FALSE", "False", " False"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            false.add(ids[0])
    return true, false


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

# Fixed strings for join prompt layout.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # anchor note -> question
DATA_PROCESSING_INSTRUCTION = "You are performing a data processing task."
TASK_INSTRUCTION = "Evaluate TRUE or FALSE for the following question: "
ANSWER_CUE = "\nANSWER:"


@dataclass(frozen=True)
class PromptLayout:
    """The fixed text of an AI.IF filter or join prompt.

    A filter prompt is the document label, the document, the question
    separator, the question label, the question, the question end,
    then the answer segments. A join prompt puts the anchor document
    after the label, then the anchor note, the partner blocks, and the
    question. The question text through the question end is tokenized
    as one segment and each answer segment on its own; the token lists
    are joined.

    Attributes:
        name: The ModelSpec.prompt_layout value that selects this layout.
        document_label: The text before the first document.
        question_label: The text before the question.
        question_end: The text after the question, in its segment.
        answer_segments: The text after the question end, one entry per
            separately tokenized segment.
        question_separator: The text between a document and the question
            label. None keeps the separator a filter template wrote.
        question_in_join_frame: Whether a join writes the question into
            the anchor frame, before the partner blocks, instead of
            after the last partner.
        answer_rows: The answer segments whose last token the readout
            reads, besides the prompt's last token.
    """

    name: str
    document_label: str = SHARED_PRE
    question_label: str = TASK_INSTRUCTION
    question_end: str = ANSWER_CUE
    answer_segments: tuple[str, ...] = ()
    question_separator: str | None = None
    question_in_join_frame: bool = True
    answer_rows: tuple[int, ...] = ()

    def question_segments(self, question: str, separator: str) -> list[str]:
        """The text after a document: one entry per tokenized segment."""
        if self.question_separator is not None:
            separator = self.question_separator
        return [separator + self.question_label + question + self.question_end,
                *self.answer_segments]


def _decision2_option(key: str, description: str) -> str:
    option = json.dumps({"key": key, "description": description},
                        ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"))
    return f"\n<option>\n{option}\n</option>"


# Every layout a ModelSpec can name. "decision2-noul" is the yes/no
# ("noul") question format of the Decision 2.0 models from vLLM
# Semantic Router: the readout reads the last token of each option
# block and the last token of the prompt.
PROMPT_LAYOUTS = {layout.name: layout for layout in (
    PromptLayout("ai-if"),
    PromptLayout(
        "decision2-noul",
        document_label="Context:\n",
        question_label="Task type: noul\nQuestion:\n",
        question_end="\nOptions:",
        answer_segments=(
            _decision2_option("false", "No"),
            _decision2_option("true", "Yes"),
            "\n\nSelect the single option best supported by the context "
            "and instructions.\nDecision:",
        ),
        question_separator="\n\n",
        question_in_join_frame=False,
        answer_rows=(0, 1),
    ),
)}


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


def tokenize_segments(segments, tokenizer) -> tuple:
    """Token ids of each segment, joined."""
    return tuple(t for segment in segments for t in tokenizer(segment))


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def join_label(placeholder: int) -> str:
    return JOIN_DOC_LABEL.format(_marker(placeholder))


def join_anchor_note(placeholder: int) -> str:
    return JOIN_ANCHOR_NOTE.format(_marker(placeholder))


def render_join_question(template: str, layout: str = "ai-if") -> str:
    """The join question text, without the question end."""
    spec = PROMPT_LAYOUTS[layout]
    return JOIN_QUESTION_SEP + spec.question_label + template


def render_join_frame(template: str, placeholder: int,
                      layout: str = "ai-if") -> str:
    """The complete anchor frame, tokenized as one string."""
    if not PROMPT_LAYOUTS[layout].question_in_join_frame:
        return join_anchor_note(placeholder)
    return join_anchor_note(placeholder) + render_join_question(template, layout)


def _tail_segments(prompt) -> tuple[str, ...]:
    """The text after a filter document or a join's last partner, by segment."""
    if prompt.tail_segments:
        return prompt.tail_segments
    if len(prompt.args) == 1:
        return (filter_question_text(prompt),)
    return (prompt.tail,)


def join_anchor_prefix_ids(prompt, placeholder: int, document_ids,
                           tokenizer) -> list:
    """The canonical token prefix shared by every tuple of an anchor."""
    if placeholder < 0 or placeholder >= len(prompt.args):
        raise ValueError(f"join anchor placeholder {placeholder} is out of range")
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(render_join_frame(prompt.template, placeholder,
                                               prompt.layout))))


def join_tuple_suffix_ids(prompt, partners, tokenizer) -> list:
    """Build token ids for partner blocks and answer cue of one tuple."""
    out = []
    for placeholder, document_ids in partners:
        if placeholder < 0 or placeholder >= len(prompt.args):
            raise ValueError(
                f"join partner placeholder {placeholder} is out of range")
        out += tokenizer(join_label(placeholder))
        out += list(document_ids)
    out += tokenize_segments(_tail_segments(prompt), tokenizer)
    return out


def render_join_prompt_ids(prompt, documents, anchor: int,
                           tokenizer) -> list:
    """The complete canonical token ids for one join tuple."""
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} documents")
    prefix = join_anchor_prefix_ids(prompt, anchor, documents[anchor],
                                    tokenizer)
    partners = [(i, ids) for i, ids in enumerate(documents) if i != anchor]
    return prefix + join_tuple_suffix_ids(prompt, partners, tokenizer)


def render_join_prompt_text(prompt, documents, anchor: int) -> str:
    """The complete canonical text for one join tuple."""
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} documents")
    if anchor < 0 or anchor >= len(prompt.args):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    out = prompt.preamble + documents[anchor]
    out += render_join_frame(prompt.template, anchor, prompt.layout)
    for i, document in enumerate(documents):
        if i != anchor:
            out += join_label(i) + document
    return out + prompt.tail


def filter_question_segments(tail: str, layout: str = "ai-if") -> list[str]:
    """Wrap a filter tail with the layout's question text, by segment."""
    content = tail.lstrip("\n")
    sep = tail[:len(tail) - len(content)]
    if not sep:
        sep = "\n\n"
    return PROMPT_LAYOUTS[layout].question_segments(content, sep)


def render_filter_question(tail: str, layout: str = "ai-if") -> str:
    """Wrap a filter tail with the task instruction and answer cue."""
    return "".join(filter_question_segments(tail, layout))


def filter_question_text(prompt) -> str:
    """Return the text appended after a filter document."""
    if len(prompt.args) != 1 or not prompt.tail.startswith("{0}"):
        raise ValueError("a filter prompt must start its tail with {0}")
    return prompt.tail[len("{0}"):]


def render_filter_prompt_ids(prompt, document_ids, tokenizer) -> list:
    """The complete canonical token ids for one filter document."""
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenize_segments(_tail_segments(prompt), tokenizer)))


def split_template(template: str) -> tuple[str, str]:
    """Split into (pre-placeholder text, placeholder-onward text)."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str,
                document_label: str = SHARED_PRE) -> tuple[str, str]:
    """Split into (frame, canonical_template).

    Relocates user text before the first placeholder to after it,
    so the document's KV stays query-independent.
    """
    import re
    user_pre, tail = split_template(template)
    if not tail:
        return "", template
    m = re.match(r"\{\d+\}", tail)
    if m is None:
        raise CompileError(
            f"template text before the first placeholder must not "
            f"contain a brace that is not a placeholder: {tail[:40]!r}")
    frame = user_pre.strip()
    rest = tail[m.end():]
    return frame, (document_label + m.group(0)
                   + (f"\n\n{frame}" if frame else "") + rest)


def canonicalize_template(template: str) -> str:
    return split_frame(template)[1]


def _check_placeholders(template: str, n_args: int) -> None:
    import re
    slots = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    if sorted(set(slots)) != list(range(n_args)):
        raise CompileError(
            f"PROMPT placeholders {sorted(set(slots))} do not match "
            f"{n_args} argument(s): expected {{0}}..{{{n_args - 1}}} "
            f"each used at least once")


def bind_prompt(template: str, args: tuple, tokenizer=None,
                turn: tuple[str, str] = ("", ""),
                layout: str = "ai-if") -> Prompt:
    """Build a filter Prompt from a template and column arguments.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order.
        tokenizer: Optional callable (text -> token list) for counting.
        turn: The model's chat-turn text: the piece before the
            preamble and the piece after the answer cue.
        layout: The PROMPT_LAYOUTS name of the model's prompt text.
    """
    import re
    _check_placeholders(template, len(args))
    frame, template = split_frame(template,
                                  PROMPT_LAYOUTS[layout].document_label)
    preamble, tail = split_template(template)
    segments = ()
    # Wrap the question text (after the placeholder) with the task
    # instruction and answer cue.
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m:
        ph, question = m.group(1), m.group(2)
        segments = filter_question_segments(question, layout)
        segments[-1] += turn[1]
        segments = tuple(segments)
        tail = ph + "".join(segments)
    else:
        tail = tail + turn[1]
    preamble = turn[0] + preamble
    pre_tok = tail_tok = frame_tok = None
    pre_ids = tail_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        pre_tok = len(pre_ids)
        tail_ids = tokenize_segments(
            segments or (re.sub(r"\{\d+\}", "", tail),), tokenizer)
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  layout=layout, tail_segments=segments)


def bind_score_prompt(template: str, args: tuple,
                      tokenizer=None,
                      turn: tuple[str, str] = ("", ""),
                      layout: str = "ai-if") -> Prompt:
    """Build an AI.SCORE Prompt that keeps the template as written.

    The reranker renders its own layout, with the query text and the
    document in separate fields, so the template is not rearranged.

    Args:
        template: Prompt template with {0}, and {1} for a pair.
        args: Column references in placeholder order.
        tokenizer: Unused; the planner tokenizes the rendered layout.
        turn: Unused; the reranker layout carries its own turn text.
        layout: Unused; AI.SCORE prompts are laid out by the planner.
    """
    _check_placeholders(template, len(args))
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            f"each AI.SCORE placeholder must name a distinct table, got "
            f"aliases {aliases}")
    return Prompt(template=template, args=tuple(args), preamble="", tail="")


def bind_join_prompt(template: str, args: tuple,
                     tokenizer=None,
                     turn: tuple[str, str] = ("", ""),
                     layout: str = "ai-if") -> Prompt:
    """Build a join Prompt with one placeholder per table.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order (one per table).
        tokenizer: Optional callable (text -> token list) for counting.
        turn: The model's chat-turn text: the piece before the
            preamble and the piece after the answer cue.
        layout: The PROMPT_LAYOUTS name of the model's prompt text.
    """
    _check_placeholders(template, len(args))
    if len(args) < 2:
        raise CompileError("a join prompt needs at least two "
                           "placeholders, one per joined table")
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            f"each join placeholder must name a distinct table (one "
            f"document block per table), got aliases {aliases}; to "
            f"mention a table's document again, use its marker in the "
            f"question text, not a second placeholder")
    spec = PROMPT_LAYOUTS[layout]
    question = render_join_question(template, layout)
    preamble = shared_preamble(turn[0], layout)
    if spec.question_in_join_frame:
        segments = [spec.question_end, *spec.answer_segments]
    else:
        segments = spec.question_segments(template, JOIN_QUESTION_SEP)
    segments[-1] += turn[1]
    segments = tuple(segments)
    tail = "".join(segments)
    pre_tok = tail_tok = frame_tok = None
    labels = tuple((a, None, None) for a in aliases)
    pre_ids = tail_ids = ()
    label_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        tail_ids = tokenize_segments(segments, tokenizer)
        pre_tok = len(pre_ids)
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(question))
        labels = tuple((a, len(tokenizer(join_label(i))),
                        len(tokenizer(render_join_frame(template, i, layout))))
                       for i, a in enumerate(aliases))
        label_ids = tuple(
            (
                alias,
                tuple(tokenizer(join_label(index))),
                tuple(tokenizer(render_join_frame(template, index, layout))),
            )
            for index, alias in enumerate(aliases)
        )
    return Prompt(template=template, args=tuple(args),
                  preamble=preamble, tail=tail,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=question, frame_tokens=frame_tok, labels=labels,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_token_ids=label_ids, layout=layout,
                  tail_segments=segments)
