"""The logical plan: four operators, three semantic and one relational.

Both entry points (AI SQL and the builder) produce this tree; the
planner prices it and the coordinator applies the Project at the sink.
Everything here is a frozen dataclass so two compilations of the same
query compare equal.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Union


class CompileError(ValueError):
    """The query is malformed or outside the language. Raised at
    sess.sql() / builder time; nothing is planned, nothing runs."""


# The engine-owned preamble, identical for every operator, every
# query, and every document. Because it is one fixed string, the KV
# of [SHARED_PRE + document] is identical wherever the document
# appears (same tokens, same positions, same attention context), so
# the KV store can hand a filter-scanned document to a join anchor
# and back. Changing this text invalidates every stored extent - the
# session folds it into the store content hash.
#
# The wording is measured, not guessed. Any instruction-like sentence
# here breaks the flag value completions on short documents: "Evaluate
# whether the following is true or false" dropped B1's observed
# selectivity from 0.398 to 0.074, and "You will read a document and
# answer a yes-or-no question about it" dropped it to 0.0014. Long
# documents (B4's ~5,850-token reports) were immune both times - the
# preamble sits too far from the answer position to prime it. So the
# preamble is a formatting label, not an instruction; the task text
# lives in the suffix, after the document.
SHARED_PRE = "DOCUMENT:\n"


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")


@dataclass(frozen=True)
class Prompt:
    """A PROMPT('template {0} ...', cols...) call, bound and split.

    The split feeds both the executor's KV reuse and the token
    arithmetic: `preamble` is the engine's fixed text before the
    first placeholder (shared across every document, computed once);
    `tail` is everything after it, with the placeholders still in
    place; `frame` is the user's pre-document text, relocated after
    the document by canonicalization. A filter carries the frame at
    the head of each stage's question. A join writes it into the
    anchor's kept KV once per anchor, so pairs read it from KV
    instead of paying its tokens per pair. Token counts are filled
    at bind time when a tokenizer is available; None means the
    planner must be given counts."""
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None


@dataclass(frozen=True)
class FilterPredicate:
    prompt: Prompt
    selectivity: Optional[float] = None   # fraction of documents that
    #                                       pass; ordering only


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text."""
    provider: str
    alias: str
    column: str


@dataclass(frozen=True)
class SemanticFilter:
    input: "Operator"
    predicates: tuple    # tuple[FilterPredicate, ...], written order


@dataclass(frozen=True)
class SemanticJoin:
    left: "Operator"
    right: "Operator"
    predicate: Prompt
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of pairs that pass
    anchor: Optional[str] = None       # table alias whose documents
    #                                    anchor this stage; None =
    #                                    planner picks the longer side


@dataclass(frozen=True)
class Project:
    """Always the root, never anywhere else. Column selection only:
    ids and pass-through text, nothing computed."""
    input: "Operator"
    columns: tuple    # tuple[ColumnRef, ...]


Operator = Union[Scan, SemanticFilter, SemanticJoin, Project]


@dataclass(frozen=True)
class LogicalPlan:
    root: Project

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JoinSpec:
    """One JOIN clause (or EXISTS/anti term), pre-assembly."""
    alias: str                 # the newly joined (or inner) table alias
    prompt: Prompt
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None


@dataclass(frozen=True)
class QueryDesc:
    """What both entry points collect before assembly. Assembling the
    tree from this one description is what makes a SQL query and its
    builder translation compare equal."""
    tables: tuple               # ((alias, provider), ...) appearance order
    doc_columns: dict           # alias -> the document column its prompts use
    filters: dict               # alias -> tuple[FilterPredicate], written order
    joins: tuple                # tuple[JoinSpec], written order
    columns: tuple              # tuple[ColumnRef], the projection


def assemble_plan(desc: QueryDesc) -> LogicalPlan:
    """The deterministic tree: scans (with their filters attached
    directly above - filters always run before joins, section 4's
    unconditional pushdown), folded left to right by the join specs,
    Project at the root."""
    def subtree(alias: str):
        provider = dict(desc.tables)[alias]
        node = Scan(provider=provider, alias=alias,
                    column=desc.doc_columns.get(alias, ""))
        preds = desc.filters.get(alias, ())
        if preds:
            node = SemanticFilter(input=node, predicates=tuple(preds))
        return node

    first = desc.tables[0][0]
    tree = subtree(first)
    for j in desc.joins:
        tree = SemanticJoin(left=tree, right=subtree(j.alias),
                            predicate=j.prompt, semantics=j.semantics,
                            selectivity=j.selectivity, anchor=j.anchor)
    return LogicalPlan(root=Project(input=tree,
                                    columns=tuple(desc.columns)))


def split_template(template: str) -> tuple[str, str]:
    """(preamble, tail): the text before the first placeholder and
    everything from it on."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str) -> tuple[str, str]:
    """(frame, canonical_template). The frame is the user's text
    before the first placeholder, stripped; the canonical template is
    SHARED_PRE, then the placeholder (the document whose KV the
    engine owns), then the frame, then everything else.

    User text before the first placeholder would sit inside the
    document's KV and make it query-specific, so it is relocated to
    just after the placeholder instead. A template without a
    placeholder has no document and is returned unchanged (it cannot
    execute anyway)."""
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


def bind_prompt(template: str, args: tuple, tokenizer=None) -> Prompt:
    """Build a Prompt, checking placeholders against arguments.

    Placeholders must be {0}, {1}, ... matching the argument count and
    order. The template is canonicalized to the engine layout
    (SHARED_PRE + document + frame + suffix) before splitting.
    `tokenizer` is any callable text -> token list; when given, the
    preamble, tail (question text, placeholders excluded), and frame
    are counted once here so the planner never tokenizes."""
    import re
    slots = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    if sorted(set(slots)) != list(range(len(args))):
        raise CompileError(
            f"PROMPT placeholders {sorted(set(slots))} do not match "
            f"{len(args)} argument(s): expected {{0}}..{{{len(args) - 1}}} "
            f"each used at least once")
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
    pre_tok = tail_tok = frame_tok = None
    if tokenizer is not None:
        pre_tok = len(tokenizer(preamble))
        tail_text = re.sub(r"\{\d+\}", "", tail)
        tail_tok = len(tokenizer(tail_text))
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok)
