"""Prompt text, prompt token ids, and template binding."""

import json
from dataclasses import dataclass

from quail.common.expressions import (
    CompileError,
)

# The model's turn stays open across the shared document prefix.
SHARED_PRE = "DOCUMENT:\n"


JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
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
        record_label: For a set-attention layout, the label before each
            document of a join, formatted with its table's name; the
            question then names each document "the <name> record". The
            anchor document is the state and each partner tuple one
            question. None keeps the numbered DOCUMENT blocks.
        label_segment: Whether the separator and question label are
            tokenized on their own, so a set-attention pass can keep
            them with the document and open each later question with
            the label alone.
        option_end: For a model that classifies by one option code
            per label, the text after its option list; the question,
            "Options:", and one "<code> <text>" line per label come
            before it. None classifies with Quail's label prompt.
        read_end: For a model that extracts by pointing at document
            positions, the text after an AI.EXTRACT question; None
            extracts by decoding the answer.
        read_slots: The names of the slots such a model points from,
            each written as "<name>:" on its own line after its turn
            text.
    """

    name: str
    document_label: str = SHARED_PRE
    question_label: str = TASK_INSTRUCTION
    question_end: str = ANSWER_CUE
    answer_segments: tuple[str, ...] = ()
    question_separator: str | None = None
    answer_rows: tuple[int, ...] = ()
    choice_label: str | None = None
    record_label: str | None = None
    label_segment: bool = False
    option_end: str | None = None
    read_end: str | None = None
    read_slots: tuple[str, ...] = ()

    def question_segments(self, question: str, separator: str) -> list[str]:
        """The text after a document: one entry per tokenized segment."""
        if self.question_separator is not None:
            separator = self.question_separator
        if self.label_segment:
            return [separator + self.question_label,
                    question + self.question_end, *self.answer_segments]
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
    # Liquid's d1 decision models: the document, then the question
    # after "QUESTION:", then the instruction to answer yes or no.
    PromptLayout(
        "d1",
        document_label="",
        question_label="QUESTION:\n",
        question_end="\n\nReply with yes or no only.",
        question_separator="\n\n\n",
        record_label="{} record:\n",
        label_segment=True,
        option_end="\n\nReply with the option code only.",
        read_end="\n\nRead the text from the state, or nothing if it is absent.",
        read_slots=("start", "end"),
    ),
)}


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def join_anchor_note(placeholder: int) -> str:
    return JOIN_ANCHOR_NOTE.format(_marker(placeholder))


OPTION_LETTERS = tuple(chr(ord("A") + index) for index in range(26))


def option_codes(labels) -> tuple[str, ...]:
    """d1's option codes.

    The codes are the labels when each is one letter, else A to Z, then
    AA, AB, and so on.
    """
    labels = [str(label).strip() for label in labels]
    if labels and all(len(label) == 1 and label.isalpha() for label in labels):
        return tuple(labels)
    pairs = [first + second for first in OPTION_LETTERS
             for second in OPTION_LETTERS]
    return tuple([*OPTION_LETTERS, *pairs][:len(labels)])


# an AI.MAP template: the document, then the instruction
MAP_TEMPLATE_HEAD = "{0}\n\n"


def map_instruction(template: str) -> str:
    """Return the instruction of an AI.MAP template, ``{0}`` and the instruction.

    Raises:
        CompileError: The template is not in that form.
    """
    if not template.startswith(MAP_TEMPLATE_HEAD):
        raise CompileError(
            f"an AI.MAP template is {MAP_TEMPLATE_HEAD!r} and the instruction, "
            f"got {template!r}")
    return template[len(MAP_TEMPLATE_HEAD):]


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
