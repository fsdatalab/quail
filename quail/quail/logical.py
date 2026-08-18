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


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")


@dataclass(frozen=True)
class Prompt:
    """A PROMPT('template {0} ...', cols...) call, bound and split.

    The split feeds both the executor's KV reuse and the token
    arithmetic: `preamble` is the template text before the first
    placeholder (shared across every document, computed once);
    `tail` is everything after it, with the placeholders still in
    place. Token counts are filled at bind time when a tokenizer is
    available; None means the planner must be given counts."""
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None


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


def bind_prompt(template: str, args: tuple, tokenizer=None) -> Prompt:
    """Build a Prompt, checking placeholders against arguments.

    Placeholders must be {0}, {1}, ... matching the argument count and
    order. `tokenizer` is any callable text -> token list; when given,
    the preamble and tail (question text, placeholders excluded) are
    counted once here so the planner never tokenizes."""
    import re
    slots = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    if sorted(set(slots)) != list(range(len(args))):
        raise CompileError(
            f"PROMPT placeholders {sorted(set(slots))} do not match "
            f"{len(args)} argument(s): expected {{0}}..{{{len(args) - 1}}} "
            f"each used at least once")
    preamble, tail = split_template(template)
    pre_tok = tail_tok = None
    if tokenizer is not None:
        pre_tok = len(tokenizer(preamble))
        tail_text = re.sub(r"\{\d+\}", "", tail)
        tail_tok = len(tokenizer(tail_text))
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok)
