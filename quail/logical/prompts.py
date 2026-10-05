"""Prompt text, prompt token ids, and template binding."""

import json
from dataclasses import dataclass, replace

from quail.logical.expressions import LABEL_PREFIX, CompileError, Prompt

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
    after the label, then the anchor note and the question label and
    question (the anchor frame, kept in the anchor's KV), then the
    partner blocks, the question end, and the answer segments. The
    text through the question end is tokenized as one segment and
    each answer segment on its own; the token lists are joined.

    Attributes:
        name: The ModelSpec.prompt_layout value that selects this layout.
        document_label: The text before the first document.
        question_label: The text before the question.
        question_end: The text after the question, in its segment.
        answer_segments: The text after the question end, one entry per
            separately tokenized segment.
        question_separator: The text between a document and the question
            label. None keeps the separator a filter template wrote.
        answer_rows: The answer segments whose last token the readout
            reads, besides the prompt's last token.
        choice_label: For a model that classifies by scoring one option
            block per label, the text before an AI.CLASSIFY question;
            None for a model that classifies with Quail's label prompt.
    """

    name: str
    document_label: str = SHARED_PRE
    question_label: str = TASK_INSTRUCTION
    question_end: str = ANSWER_CUE
    answer_segments: tuple[str, ...] = ()
    question_separator: str | None = None
    answer_rows: tuple[int, ...] = ()
    choice_label: str | None = None

    def question_segments(self, question: str, separator: str) -> list[str]:
        """The text after a document: one entry per tokenized segment."""
        if self.question_separator is not None:
            separator = self.question_separator
        return [separator + self.question_label + question + self.question_end,
                *self.answer_segments]


def _decision2_option(key: str, description: str | None) -> str:
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
        answer_rows=(0, 1),
        choice_label="Task type: choice\nQuestion:\n",
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
    # the stored template names the predicate whatever the layout
    canonical = canonicalize_template(template)
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
    return Prompt(template=canonical, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  layout=layout, tail_segments=segments)


CLASSIFY_INSTRUCTION = ("Answer with exactly one of the categories below "
                        "for the following question: ")
LETTERS_INSTRUCTION = ("Answer with the letter of exactly one of the "
                       "categories below for the following question: ")
CATEGORIES_HEADER = "\n\nCategories:"
# Jev's limit on one choice's options, which OpenJev keeps
MAX_LETTERS = 255


def letter_candidates() -> list[str]:
    """Return candidate label letters: A to Z, a to z, then AA to ZZ."""
    upper = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    lower = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    return upper + lower + [first + second for first in upper for second in upper]


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


def answer_prefix(turn: tuple[str, str]) -> str:
    """Return the spacing before a label in the model's answer.

    A label after the answer cue follows a space, as a word would; a
    model whose turn suffix opens its reply after the cue starts the
    reply without one.
    """
    return "" if turn[1] else LABEL_PREFIX


def render_categories(labels, descriptions=(), letters=()) -> str:
    """Return the category list, one line per label.

    A line is ``- label``, or ``- label: description``; lettered,
    ``- A: label`` or ``- A: label (description)``.
    """
    descriptions = descriptions or ("",) * len(labels)
    if letters:
        return CATEGORIES_HEADER + "".join(
            f"\n- {letter}: {label}"
            + (f" ({description})" if description else "")
            for letter, label, description in zip(letters, labels, descriptions))
    return CATEGORIES_HEADER + "".join(
        f"\n- {label}" + (f": {description}" if description else "")
        for label, description in zip(labels, descriptions))


def render_classify_question(question: str, labels, descriptions=(),
                             task_description: str = "",
                             letters=()) -> str:
    """Wrap a classification question with its instruction, categories, and cue."""
    content = question.lstrip("\n")
    sep = question[:len(question) - len(content)] or "\n\n"
    if task_description:
        content = (content + "\n" if content else "") + task_description
    instruction = LETTERS_INSTRUCTION if letters else CLASSIFY_INSTRUCTION
    return (sep + instruction + content
            + render_categories(labels, descriptions, letters) + ANSWER_CUE)


def label_text(label: str, prefix: str = LABEL_PREFIX) -> str:
    """Return the label text with its answer prefix."""
    return prefix + label


def bind_classify_prompt(template: str, args: tuple, labels, descriptions=(),
                         tokenizer=None,
                         turn: tuple[str, str] = ("", ""),
                         task_description: str = "",
                         layout: str = "ai-if") -> Prompt:
    """Bind a classification prompt to one or two document columns.

    The prompt names the labels and, when every label gets a distinct
    one-token letter, holds a lettered version for the letters rule. A
    prompt over joined rows uses the join document layout, then the
    classification question and the label list.

    Args:
        template: Prompt text with one placeholder per document column.
        args: One column reference, or two references for joined documents.
        labels: Labels in query order.
        descriptions: Optional descriptions in label order.
        tokenizer: Optional callable mapping text to token IDs.
        turn: Chat text before the prompt and after the answer cue.
        task_description: Additional classification instructions.
        layout: The PROMPT_LAYOUTS name of the model's prompt text. A
            layout with a choice_label writes one option block per label.

    Returns:
        A Prompt containing the document layout, label list, and token IDs.

    Raises:
        CompileError: The placeholders or number of document columns are
            invalid, or there are more than MAX_LETTERS labels.
    """
    _check_placeholders(template, len(args))
    if PROMPT_LAYOUTS[layout].choice_label is not None:
        return _bind_choice_classify_prompt(
            template, args, labels, descriptions, tokenizer, turn,
            task_description, layout)
    if len(args) == 2:
        return _bind_joined_classify_prompt(
            template, args, labels, descriptions, tokenizer, turn,
            task_description)
    if len(args) != 1:
        raise CompileError("AI.CLASSIFY reads one document per row, or one "
                           "from each side of a join")
    prefix = answer_prefix(turn)
    named = _bind_document_classify_prompt(
        template, args, labels, descriptions, tokenizer, turn,
        task_description, prefix)
    letters = choice_letters(len(labels), tokenizer, prefix)
    if len(letters) < len(labels):
        return named
    lettered = _bind_document_classify_prompt(
        template, args, labels, descriptions, tokenizer, turn,
        task_description, prefix, letters)
    return replace(named, lettered=lettered)


def _bind_document_classify_prompt(template, args, labels, descriptions,
                                   tokenizer, turn, task_description,
                                   prefix, letters=()):
    """Bind one AI.CLASSIFY prompt over one document, lettered when asked."""
    import re
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m is None:
        raise CompileError(f"unexpected AI.CLASSIFY template: {template!r}")
    question = render_classify_question(
        m.group(2), labels, descriptions, task_description, letters)
    preamble = turn[0] + preamble
    tail = m.group(1) + question + turn[1]
    pre_tok = tail_tok = frame_tok = None
    pre_ids = tail_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        pre_tok = len(pre_ids)
        tail_ids = tuple(tokenizer(re.sub(r"\{\d+\}", "", tail)))
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_prefix=prefix, letters=tuple(letters))


def _bind_joined_classify_prompt(template, args, labels, descriptions,
                                 tokenizer, turn, task_description):
    """Bind a classification of joined rows: the join layout, then the question.

    The anchor document (placeholder {0}) comes first, then the anchor
    note, the partner's label and document, and the classification
    question with its categories and cue. ``label_token_ids`` holds,
    as for a join, each alias's partner label and the anchor note.
    """
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            "each placeholder of a classification of joined rows names a "
            "distinct table")
    prefix = answer_prefix(turn)

    def bind(letters=()):
        question = render_classify_question(template, labels, descriptions,
                                            task_description, letters)
        preamble = shared_preamble(turn[0])
        tail = question + turn[1]
        frame = join_anchor_note(0)
        pre_tok = tail_tok = frame_tok = None
        pre_ids = tail_ids = ()
        label_ids = ()
        if tokenizer is not None:
            pre_ids = tuple(tokenizer(preamble))
            tail_ids = tuple(tokenizer(tail))
            pre_tok, tail_tok = len(pre_ids), len(tail_ids)
            frame_tok = len(tokenizer(frame))
            label_ids = tuple(
                (alias, tuple(tokenizer(join_label(index))),
                 tuple(tokenizer(join_anchor_note(index))))
                for index, alias in enumerate(aliases))
        return Prompt(template=template, args=tuple(args), preamble=preamble,
                      tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                      frame=frame, frame_tokens=frame_tok,
                      preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                      label_token_ids=label_ids, label_prefix=prefix,
                      letters=tuple(letters))

    named = bind()
    letters = choice_letters(len(labels), tokenizer, prefix)
    if len(letters) < len(labels):
        return named
    return replace(named, lettered=bind(letters))


CHOICE_QUESTION = "Which option best fits the context?"


def choice_segments(question: str, labels, descriptions=(),
                    layout: str = "decision2-noul") -> list[str]:
    """The text after a document: the question, one block per label, the close.

    Args:
        question: The classification question; empty asks CHOICE_QUESTION.
        labels: Labels in query order; each is an option's key.
        descriptions: Optional descriptions in label order; a label
            without one has a null description.
        layout: A PROMPT_LAYOUTS name with a choice_label.
    """
    spec = PROMPT_LAYOUTS[layout]
    descriptions = tuple(descriptions) or (None,) * len(labels)
    return [JOIN_QUESTION_SEP + spec.choice_label + (question or CHOICE_QUESTION)
            + spec.question_end,
            *(_decision2_option(label, description or None)
              for label, description in zip(labels, descriptions)),
            spec.answer_segments[-1]]


def _bind_choice_classify_prompt(template, args, labels, descriptions,
                                 tokenizer, turn, task_description, layout):
    """Bind AI.CLASSIFY in a layout that scores one option block per label.

    One document: the document label, the document, then the question
    and the option blocks. Joined rows: the anchor document, its note,
    the partner's label and document, then the question and the blocks.
    ``tail_segments`` holds the question, each block, and the close.
    """
    import re
    if len(args) not in (1, 2):
        raise CompileError("AI.CLASSIFY reads one document per row, or one "
                           "from each side of a join")
    spec = PROMPT_LAYOUTS[layout]
    preamble = turn[0] + spec.document_label
    label_ids = ()
    if len(args) == 2:
        if len({r.alias for r in args}) != 2:
            raise CompileError(
                "each placeholder of a classification of joined rows names a "
                "distinct table")
        stored, question, frame, head = template, template, join_anchor_note(0), ""
        if tokenizer is not None:
            label_ids = tuple(
                (ref.alias, tuple(tokenizer(join_label(index))),
                 tuple(tokenizer(join_anchor_note(index))))
                for index, ref in enumerate(args))
    else:
        stored = canonicalize_template(template)
        frame, template = split_frame(template, spec.document_label)
        _, rest = split_template(template)
        m = re.match(r"(\{\d+\})(.*)", rest, re.DOTALL)
        if m is None:
            raise CompileError(f"unexpected AI.CLASSIFY template: {template!r}")
        head, question = m.group(1), m.group(2).strip()
    if task_description:
        question = (question + "\n" if question else "") + task_description
    segments = choice_segments(question, labels, descriptions, layout)
    segments[-1] += turn[1]
    segments = tuple(segments)
    tail = head + "".join(segments)
    pre_ids = tail_ids = ()
    pre_tok = tail_tok = frame_tok = None
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        tail_ids = tokenize_segments(segments, tokenizer)
        pre_tok, tail_tok = len(pre_ids), len(tail_ids)
        frame_tok = len(tokenizer(frame)) if frame else 0
    return Prompt(template=stored, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_token_ids=label_ids, layout=layout,
                  tail_segments=segments)


def render_joined_classify_prompt_text(prompt, anchor: str,
                                       partner: str) -> str:
    """Render the complete classification prompt for one joined row."""
    return (prompt.preamble + anchor + join_anchor_note(0) + join_label(1)
            + partner + prompt.tail)


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
    segments = [spec.question_end, *spec.answer_segments]
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


def prompt_aliases(prompt) -> tuple[str, ...]:
    return tuple(dict.fromkeys(argument.alias for argument in prompt.args))


def _without_document(template: str, marker: str) -> str:
    """Return the query text with the scored document's placeholder gone.

    The reranker gives the document its own field, so a placeholder
    standing alone at either end of the template only marks where the
    document goes and is dropped. A placeholder inside a sentence
    becomes "this document".
    """
    text = template.strip()
    if text.startswith(marker):
        text = text[len(marker):].lstrip()
    if text.endswith(marker):
        text = text[:-len(marker)].rstrip()
    return text.replace(marker, "this document")


def score_query_template(prompt) -> str:
    aliases = prompt_aliases(prompt)
    if len(aliases) == 1:
        return _without_document(prompt.template, "{0}")
    if len(aliases) == 2 and len(prompt.args) == 2:
        return _without_document(prompt.template, "{1}")
    raise ValueError("AI.SCORE supports one document or one document pair")


def score_token_parts(prompt, model, tokenizer) -> tuple[tuple[int, ...], ...]:
    """Tokenize the fixed pieces of the rendered prompt around the documents."""
    if model.role != "reranker":
        return _answer_token_parts(prompt, model, tokenizer)
    query_template = score_query_template(prompt)
    rendered = render_qwen3_reranker_input(query_template, "{document}")
    before, after = rendered.split("{document}")
    parts = (before, after)
    if len(prompt_aliases(prompt)) == 2:
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
    aliases = prompt_aliases(prompt)
    if len(aliases) == 1:
        bound = bind_prompt(prompt.template, prompt.args, tokenizer, turn,
                            layout)
        return tuple(bound.preamble_token_ids), tuple(bound.tail_token_ids)
    if len(prompt.args) != 2:
        raise ValueError("AI.SCORE supports one document or one document pair")
    bound = bind_join_prompt(prompt.template, prompt.args, tokenizer, turn,
                             layout)
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


def classify_prompt_tokens(prompt, labels, tokenizer) -> tuple:
    """Return a classification's head, tail, and label token sequences."""
    return (
        tuple(tokenizer(prompt.preamble)), tuple(prompt.tail_token_ids),
        tuple(tuple(tokenizer(label_text(label, prompt.label_prefix)))
              for label in labels))


def choice_token_parts(prompt, tokenizer) -> tuple:
    """Return decision classification tail tokens, frame length, and option blocks."""
    segments = prompt.tail_segments
    frame = len(tokenizer(segments[0]))
    blocks = tuple(tuple(tokenizer(block)) for block in segments[1:-1])
    return tuple(prompt.tail_token_ids), frame, blocks


QWEN3_RERANKER_INSTRUCTION = (
    "Judge whether the document meets the requirements in the query."
)
QWEN3_RERANKER_SYSTEM_TEXT = (
    f"{DATA_PROCESSING_INSTRUCTION} "
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" "
    "or \"no\"."
)


def render_qwen3_reranker_input(query: str, document: str) -> str:
    """Render a complete Qwen3 reranker prompt."""
    return (
        f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM_TEXT}<|im_end|>\n"
        f'<|im_start|>user\n<Instruct>: {QWEN3_RERANKER_INSTRUCTION}\n'
        f'<Query>: {query}\n<Document>: {document}<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )
