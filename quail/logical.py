"""Logical plan operators and prompt binding."""

from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar, Optional, Protocol


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


# Fixed preamble before every document. Must be a formatting label,
# not an instruction; instruction text here biases short-document
# completions.
SHARED_PRE = "DOCUMENT:\n"

# Fixed strings for join prompt layout.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # anchor note -> question
TASK_INSTRUCTION = "Evaluate TRUE or FALSE for the following question: "
ANSWER_CUE = "\nANSWER:"


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def join_label(placeholder: int) -> str:
    return JOIN_DOC_LABEL.format(_marker(placeholder))


def join_anchor_note(placeholder: int) -> str:
    return JOIN_ANCHOR_NOTE.format(_marker(placeholder))


def render_join_question(template: str) -> str:
    """The static join question written once into each anchor's KV."""
    return JOIN_QUESTION_SEP + TASK_INSTRUCTION + template


def render_join_frame(template: str, placeholder: int) -> str:
    """The complete anchor frame, tokenized as one string."""
    return join_anchor_note(placeholder) + render_join_question(template)


def join_anchor_prefix_ids(prompt, placeholder: int, document_ids,
                           tokenizer) -> list:
    """The canonical token prefix shared by every tuple of an anchor."""
    if placeholder < 0 or placeholder >= len(prompt.args):
        raise ValueError(f"join anchor placeholder {placeholder} is out of range")
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(render_join_frame(prompt.template,
                                               placeholder))))


def join_tuple_suffix_ids(prompt, partners, tokenizer) -> list:
    """Build token ids for partner blocks and answer cue of one tuple."""
    out = []
    for placeholder, document_ids in partners:
        if placeholder < 0 or placeholder >= len(prompt.args):
            raise ValueError(
                f"join partner placeholder {placeholder} is out of range")
        out += tokenizer(join_label(placeholder))
        out += list(document_ids)
    out += tokenizer(prompt.tail)
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
    out += render_join_frame(prompt.template, anchor)
    for i, document in enumerate(documents):
        if i != anchor:
            out += join_label(i) + document
    return out + prompt.tail


def render_filter_question(tail: str) -> str:
    """Wrap a filter tail with the task instruction and answer cue."""
    content = tail.lstrip("\n")
    sep = tail[:len(tail) - len(content)]
    if not sep:
        sep = "\n\n"
    return sep + TASK_INSTRUCTION + content + ANSWER_CUE


def render_filter_prompt_ids(prompt, document_ids, tokenizer) -> list:
    """The complete canonical token ids for one filter document."""

    if len(prompt.args) != 1 or not prompt.tail.startswith("{0}"):
        raise ValueError("a filter prompt must start its tail with {0}")
    tail = prompt.tail.replace("{0}", "", 1)
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(tail)))


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")


@dataclass(frozen=True)
class Prompt:
    """A bound PROMPT call, split into preamble, frame, and tail.

    Token counts are filled at bind time when a tokenizer is given;
    None means the planner must supply counts.
    """
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None
    # join prompts only: (alias, label_tokens, anchor_frame_tokens)
    # per placeholder, in order. Empty for filter prompts.
    labels: tuple = ()


@dataclass(frozen=True)
class FilterPredicate:
    prompt: Prompt
    selectivity: Optional[float] = None   # fraction of documents that
    #                                       pass; ordering only


class LogicalNode(Protocol):
    """Node in a logical query plan."""

    type_name: ClassVar[str]

    def children(self) -> tuple["LogicalNode", ...]: ...

    def expressions(self) -> tuple[Any, ...]: ...

    def output_schema(self) -> tuple[ColumnRef, ...]: ...

    def validate(self) -> None: ...

    def with_children(
        self, children: tuple["LogicalNode", ...]
    ) -> "LogicalNode": ...

    def with_expressions(self, expressions: tuple[Any, ...]) -> "LogicalNode": ...

    def explain_fields(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text."""
    provider: str
    alias: str
    column: str

    type_name: ClassVar[str] = "quail.scan.v1"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return ()

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return (ColumnRef(self.alias, self.provider, self.column),)

    def validate(self) -> None:
        if not self.provider or not self.alias or not self.column:
            raise CompileError("Scan needs a provider, alias, and column")

    def with_children(self, children: tuple) -> "Scan":
        if children:
            raise CompileError("Scan has no children")
        return self

    def with_expressions(self, expressions: tuple) -> "Scan":
        if expressions:
            raise CompileError("Scan has no expressions")
        return self

    def explain_fields(self) -> dict:
        return {
            "provider": self.provider,
            "alias": self.alias,
            "column": self.column,
        }


@dataclass(frozen=True)
class SemanticFilter:
    input: LogicalNode
    predicates: tuple    # tuple[FilterPredicate, ...], written order

    type_name: ClassVar[str] = "quail.semantic_filter.v1"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if not self.predicates:
            raise CompileError("SemanticFilter needs at least one predicate")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticFilter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("SemanticFilter needs at least one predicate")
        return replace(self, predicates=expressions)

    def explain_fields(self) -> dict:
        return {
            "predicates": len(self.predicates),
            "selectivities": [p.selectivity for p in self.predicates],
        }


@dataclass(frozen=True)
class SemanticJoin:
    """One n-way join: cross product filtered by a single prompt."""
    inputs: tuple    # tuple[LogicalNode]: accumulated tree first, then
    #                  one scan per newly joined table
    predicate: Prompt
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of tuples that pass
    anchor: Optional[str] = None       # table alias whose KV is kept;
    #                                    None = planner picks

    type_name: ClassVar[str] = "quail.semantic_join.v1"

    def children(self) -> tuple[LogicalNode, ...]:
        return self.inputs

    def expressions(self) -> tuple:
        return (self.predicate,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = []
        for child in self.inputs:
            for field in child.output_schema():
                if field not in fields:
                    fields.append(field)
        return tuple(fields)

    def validate(self) -> None:
        if not self.inputs:
            raise CompileError("SemanticJoin needs at least one input")
        if self.semantics not in {"full", "exists", "anti"}:
            raise CompileError(
                f"unknown join semantics {self.semantics!r}")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if not children:
            raise CompileError("SemanticJoin needs at least one input")
        return replace(self, inputs=children)

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticJoin needs one prompt")
        return replace(self, predicate=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class Project:
    """Column projection. Always the root operator."""
    input: LogicalNode
    columns: tuple    # tuple[ColumnRef, ...]
    limit: Optional[int] = None

    type_name: ClassVar[str] = "quail.logical_project.v1"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.columns

    def validate(self) -> None:
        if not self.columns:
            raise CompileError("Project needs at least one column")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("Project needs at least one column")
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        return {
            "columns": [
                f"{column.alias}.{column.column}" for column in self.columns
            ],
            "limit": self.limit,
        }


Operator = LogicalNode


@dataclass(frozen=True)
class LogicalPlan:
    root: LogicalNode

    def to_dict(self) -> dict:
        return asdict(self)

    def walk(self) -> tuple[LogicalNode, ...]:
        """Return logical nodes with each child before its parent."""
        nodes = []

        def visit(node):
            for child in node.children():
                visit(child)
            nodes.append(node)

        visit(self.root)
        return tuple(nodes)

    def validate(self) -> None:
        """Validate every logical node."""
        for node in self.walk():
            node.validate()


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""
    aliases: tuple             # newly joined table aliases
    prompt: Prompt
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None


class LogicalPlanBuilder:
    """Build the built in logical nodes used by both front ends."""

    def __init__(self):
        self._tables = []
        self._nodes = {}
        self._root = None

    def add_scan(
        self,
        alias: str,
        provider: str,
        column: str,
        predicates: tuple[FilterPredicate, ...] = (),
    ) -> None:
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(provider=provider, alias=alias, column=column)
        if predicates:
            node = SemanticFilter(node, tuple(predicates))
        self._tables.append(alias)
        self._nodes[alias] = node
        if self._root is None:
            self._root = node

    def add_join(self, join: JoinSpec) -> None:
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        self._root = SemanticJoin(
            inputs=(self._root,) + tuple(
                self._nodes[alias] for alias in join.aliases
            ),
            predicate=join.prompt,
            semantics=join.semantics,
            selectivity=join.selectivity,
            anchor=join.anchor,
        )

    def project(
        self, columns: tuple[ColumnRef, ...], limit: int | None = None
    ) -> LogicalPlan:
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        plan = LogicalPlan(Project(self._root, tuple(columns), limit))
        plan.validate()
        return plan


def split_template(template: str) -> tuple[str, str]:
    """Split into (pre-placeholder text, placeholder-onward text)."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str) -> tuple[str, str]:
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
    return frame, (SHARED_PRE + m.group(0)
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


def bind_prompt(template: str, args: tuple, tokenizer=None) -> Prompt:
    """Build a filter Prompt from a template and column arguments.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order.
        tokenizer: Optional callable (text -> token list) for counting.
    """
    import re
    _check_placeholders(template, len(args))
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
    # Wrap the question text (after the placeholder) with the task
    # instruction and answer cue.
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m:
        ph, question = m.group(1), m.group(2)
        tail = ph + render_filter_question(question)
    pre_tok = tail_tok = frame_tok = None
    if tokenizer is not None:
        pre_tok = len(tokenizer(preamble))
        tail_text = re.sub(r"\{\d+\}", "", tail)
        tail_tok = len(tokenizer(tail_text))
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok)


def bind_join_prompt(template: str, args: tuple,
                     tokenizer=None) -> Prompt:
    """Build a join Prompt with one placeholder per table.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order (one per table).
        tokenizer: Optional callable (text -> token list) for counting.
    """
    _check_placeholders(template, len(args))
    if len(args) < 2:
        raise CompileError("a join prompt needs at least two "
                           "placeholders, one per joined table")
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            f"each join placeholder must name a distinct table (one "
            f"document block per table), got aliases {aliases}; refer "
            f"to a table's document again in the question text with "
            f"its marker instead of adding a second placeholder")
    question = render_join_question(template)
    pre_tok = tail_tok = frame_tok = None
    labels = tuple((a, None, None) for a in aliases)
    if tokenizer is not None:
        pre_tok = len(tokenizer(SHARED_PRE))
        tail_tok = len(tokenizer(ANSWER_CUE))
        frame_tok = len(tokenizer(question))
        labels = tuple((a, len(tokenizer(join_label(i))),
                        len(tokenizer(render_join_frame(template, i))))
                       for i, a in enumerate(aliases))
    return Prompt(template=template, args=tuple(args),
                  preamble=SHARED_PRE, tail=ANSWER_CUE,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=question, frame_tokens=frame_tok, labels=labels)
