"""Write the QUAIL-B query plans as Substrait ProtoJSON files.

    uv run python tools/make_substrait_plans.py [output_dir]

The output directory defaults to `quail_b/plans`. The checked-in files
there are the published query definitions; a test checks that they
equal this script's output, so a prompt or query shape is changed here
and in `quail_b/prompts.py`, then the files are regenerated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format
from substrait import algebra_pb2 as algebra
from substrait import plan_pb2, type_pb2
from substrait.extensions import extensions_pb2

from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    CARDIOVASCULAR_REACTION,
    CUAD_CHANGE_OF_CONTROL,
    CUAD_EXCLUSIVITY,
    CUAD_LICENSE_GRANT,
    CUAD_NON_COMPETE,
    CUAD_NON_TRANSFERABLE_LICENSE,
    CUAD_PAGE_CAPS_LIABILITY,
    CUAD_PAGE_UNCAPPED_LIABILITY,
    CUAD_PERPETUAL_LICENSE,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F11,
    F12,
    F13,
    FIN_NEEDS_CALCULATION,
    FIN_PAGE_EVIDENCE,
    LEP1,
    LEP2,
    LEPJOIN,
    LEPS1,
    NEUROLOGICAL_REACTION,
    P_LOC,
    P_MSG,
    REACTION,
    REFUTE,
    SCENARIO_MATCH,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
    TREAS_COMBINES_FIGURES,
    TREAS_PAGE_EVIDENCE,
)
from quail_b.substrait import (
    AI_EXTENSION_URN,
    AI_FILTER_NAME,
    AI_JOIN_NAME,
    AND_NAME,
    BOOLEAN_EXTENSION_URN,
    COMPARISON_EXTENSION_URN,
    EQUAL_NAME,
    LTE_NAME,
    SUBSTRAIT_VERSION,
)

PLANS_DIR = Path(__file__).resolve().parents[1] / "quail_b" / "plans"

# Function anchors are fixed so expressions can reference them while
# the tree is built; only the functions a plan uses are declared.
_URN_ANCHORS = {
    AI_EXTENSION_URN: 1,
    COMPARISON_EXTENSION_URN: 2,
    BOOLEAN_EXTENSION_URN: 3,
}
_FUNCTIONS = {
    AI_FILTER_NAME: (1, AI_EXTENSION_URN),
    AI_JOIN_NAME: (2, AI_EXTENSION_URN),
    EQUAL_NAME: (3, COMPARISON_EXTENSION_URN),
    AND_NAME: (4, BOOLEAN_EXTENSION_URN),
    LTE_NAME: (5, COMPARISON_EXTENSION_URN),
}


@dataclass(frozen=True)
class Scan:
    """Read one document table.

    Attributes:
        table: The table name.
        alias: The relation alias, unique within a query.
        text: The column the AI functions read.
        columns: Further string columns, for ordinary join conditions.
        int_columns: Integer columns, for ordinary bounds.
    """

    table: str
    alias: str
    text: str
    columns: tuple[str, ...] = ()
    int_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bound:
    """Keep the rows of one scan whose integer column is at most a value."""

    input: Scan
    column: str
    value: int


@dataclass(frozen=True)
class Filter:
    """Keep the documents of one relation that answer a prompt TRUE."""

    input: Scan | Bound | Filter
    prompt: str


@dataclass(frozen=True)
class Join:
    """Keep the pairs of documents that answer a prompt TRUE.

    Attributes:
        left: The left input.
        right: The right input.
        aliases: The relations the prompt's `{0}` and `{1}` refer to.
        prompt: The join prompt.
        on: Column pairs, in `aliases` order, that must also be equal.
    """

    left: Scan | Bound | Filter | Join
    right: Scan | Bound | Filter | Join
    aliases: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Query:
    id: str
    description: str
    tree: Scan | Bound | Filter | Join
    privacy: bool = False


def _string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _bool_type():
    return type_pb2.Type(
        bool=type_pb2.Type.Boolean(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _int_type():
    return type_pb2.Type(
        i32=type_pb2.Type.I32(nullability=type_pb2.Type.NULLABILITY_REQUIRED)
    )


def _common(alias):
    return algebra.RelCommon(
        direct=algebra.RelCommon.Direct(),
        hint=algebra.RelCommon.Hint(alias=alias),
    )


def _field(index):
    return algebra.Expression(
        selection=algebra.Expression.FieldReference(
            direct_reference=algebra.Expression.ReferenceSegment(
                struct_field=algebra.Expression.ReferenceSegment.StructField(
                    field=index
                )
            ),
            root_reference=algebra.Expression.FieldReference.RootReference(),
        )
    )


def _literal(text):
    return algebra.Expression(
        literal=algebra.Expression.Literal(string=text)
    )


def _int_literal(value):
    return algebra.Expression(
        literal=algebra.Expression.Literal(i32=value)
    )


def _call(name, arguments):
    return algebra.Expression(
        scalar_function=algebra.Expression.ScalarFunction(
            function_reference=_FUNCTIONS[name][0],
            arguments=[
                algebra.FunctionArgument(value=argument)
                for argument in arguments
            ],
            output_type=_bool_type(),
        )
    )


class _Emitter:
    """Turn a query tree into Substrait relations.

    Operators are numbered in post-order (inputs before the operator,
    left before right), which is the order the benchmark's plan reader
    reports them in.
    """

    def __init__(self):
        self.counts = {"filter": 0, "join": 0}
        self.functions = set()

    def _operator_id(self, kind):
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def emit(self, node):
        """Return (rel, fields, text columns by alias) for one tree."""
        if isinstance(node, Scan):
            names = ("id", node.text, *node.columns, *node.int_columns)
            types = (
                [_string_type() for _name in names[:len(names) - len(node.int_columns)]]
                + [_int_type() for _name in node.int_columns]
            )
            read = algebra.ReadRel(
                common=_common(node.alias),
                base_schema=type_pb2.NamedStruct(
                    names=names,
                    struct=type_pb2.Type.Struct(
                        types=types,
                        nullability=type_pb2.Type.NULLABILITY_REQUIRED,
                    ),
                ),
                named_table=algebra.ReadRel.NamedTable(names=[node.table]),
            )
            fields = tuple((node.alias, name) for name in names)
            return algebra.Rel(read=read), fields, {node.alias: node.text}

        if isinstance(node, Bound):
            rel, fields, text = self.emit(node.input)
            self.functions.add(LTE_NAME)
            relation = algebra.FilterRel(
                common=algebra.RelCommon(direct=algebra.RelCommon.Direct()),
                input=rel,
                condition=_call(LTE_NAME, [
                    _field(fields.index((node.input.alias, node.column))),
                    _int_literal(node.value),
                ]),
            )
            return algebra.Rel(filter=relation), fields, text

        if isinstance(node, Filter):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_FILTER_NAME)
            relation = algebra.FilterRel(
                common=_common(self._operator_id("filter")),
                input=rel,
                condition=_call(AI_FILTER_NAME, [
                    _literal(node.prompt),
                    _field(fields.index((alias, text[alias]))),
                ]),
            )
            return algebra.Rel(filter=relation), fields, text

        left, left_fields, left_text = self.emit(node.left)
        right, right_fields, right_text = self.emit(node.right)
        fields = left_fields + right_fields
        text = {**left_text, **right_text}
        first, second = node.aliases
        self.functions.add(AI_JOIN_NAME)
        conditions = [_call(AI_JOIN_NAME, [
            _literal(node.prompt),
            _field(fields.index((first, text[first]))),
            _field(fields.index((second, text[second]))),
        ])]
        for left_column, right_column in node.on:
            self.functions.update({EQUAL_NAME, AND_NAME})
            conditions.append(_call(EQUAL_NAME, [
                _field(fields.index((first, left_column))),
                _field(fields.index((second, right_column))),
            ]))
        expression = (
            conditions[0] if len(conditions) == 1
            else _call(AND_NAME, conditions)
        )
        relation = algebra.JoinRel(
            common=_common(self._operator_id("join")),
            left=left,
            right=right,
            expression=expression,
            type=algebra.JoinRel.JOIN_TYPE_INNER,
        )
        return algebra.Rel(join=relation), fields, text


def build_plan(tree, select=None) -> plan_pb2.Plan:
    """Return the Substrait plan selecting the id of each relation.

    Args:
        tree: The query tree.
        select: Aliases whose ids the query returns, or None for every
            relation in scan order.
    """
    emitter = _Emitter()
    rel, fields, text = emitter.emit(tree)
    aliases = list(text) if select is None else list(select)
    expressions = [_field(fields.index((alias, "id"))) for alias in aliases]
    project = algebra.Rel(
        project=algebra.ProjectRel(
            common=algebra.RelCommon(
                emit=algebra.RelCommon.Emit(
                    output_mapping=[
                        len(fields) + index
                        for index in range(len(expressions))
                    ]
                )
            ),
            input=rel,
            expressions=expressions,
        )
    )
    used = sorted(emitter.functions, key=lambda name: _FUNCTIONS[name][0])
    urns = sorted(
        {_FUNCTIONS[name][1] for name in used},
        key=_URN_ANCHORS.__getitem__,
    )
    return plan_pb2.Plan(
        version=plan_pb2.Version(
            major_number=SUBSTRAIT_VERSION[0],
            minor_number=SUBSTRAIT_VERSION[1],
            patch_number=SUBSTRAIT_VERSION[2],
            producer="quail-b",
        ),
        extension_urns=[
            extensions_pb2.SimpleExtensionURN(
                extension_urn_anchor=_URN_ANCHORS[urn],
                urn=urn,
            )
            for urn in urns
        ],
        extensions=[
            extensions_pb2.SimpleExtensionDeclaration(
                extension_function=(
                    extensions_pb2.SimpleExtensionDeclaration.ExtensionFunction(
                        extension_urn_reference=_URN_ANCHORS[_FUNCTIONS[name][1]],
                        function_anchor=_FUNCTIONS[name][0],
                        name=name,
                    )
                )
            )
            for name in used
        ],
        relations=[
            plan_pb2.PlanRel(
                root=algebra.RelRoot(input=project, names=aliases)
            )
        ],
    )


def _filters(node, *prompts):
    for prompt in prompts:
        node = Filter(node, prompt)
    return node


def _reviews(alias="r"):
    return Scan("reviews", alias, "body")


def _aspects(alias="a"):
    return Scan("aspects", alias, "aspect")


def _reports():
    return Scan("reports", "r", "report")


def _terms(alias="m"):
    return Scan("terms", alias, "term")


def _claims(alias="c", *columns):
    return Scan("claims", alias, "claim", columns)


def _evidence(alias="e"):
    return Scan("evidence", alias, "text")


def _contexts():
    return Scan("citation_contexts", "d", "destination_context")


def _passages():
    return Scan("citation_passages", "s", "passage_text")


def _traces():
    return Scan("agent_traces", "t", "trace")


def _policies():
    return Scan("policies", "p", "policy_text")


def _pages():
    return Scan("contract_pages", "p", "document")


# One prompt carries every page of a contract; the probe of
# DiffusionGemma found 32 pages the most one prompt holds. 430 of
# the 510 contracts have at most 32 pages.
CONTRACT_PAGES_MAX = 32


def _contracts():
    return Bound(Scan("contracts", "c", "document", int_columns=("page_count",)),
                 "page_count", CONTRACT_PAGES_MAX)


def _questions():
    return Scan("filing_questions", "q", "question", ("filing",))


def _filing_pages():
    return Scan("filing_pages", "p", "document", ("filing",))


def _evidence_join(questions):
    """Each question against the pages of its own filing."""
    return Join(questions, _filing_pages(), ("q", "p"), FIN_PAGE_EVIDENCE,
                on=(("filing", "filing"),))


def _treasury_questions():
    return Scan("treasury_questions", "q", "question", ("statement",))


def _treasury_pages():
    return Scan("treasury_pages", "p", "document", ("statement",))


def _provenance_join(questions):
    """Each question row against the pages of the statement it reads."""
    return Join(questions, _treasury_pages(), ("q", "p"), TREAS_PAGE_EVIDENCE,
                on=(("statement", "statement"),))


def _imdb_chain(first):
    """Chain r1-a1-r2-a2: both reviews discuss a1, and r2 is positive about a2."""
    return Join(
        Join(
            Join(first, _aspects("a1"), ("r1", "a1"), DISCUSS_ASPECT),
            _reviews("r2"), ("r2", "a1"), DISCUSS_ASPECT,
        ),
        _aspects("a2"), ("r2", "a2"), ASPECT_SENTIMENT,
    )


def _fever_chain(c1, e1, c2, e2):
    """Chain c1-e1-c2-e2: e1 supports c1 but refutes c2, and e2 supports c2."""
    return Join(
        Join(Join(c1, e1, ("c1", "e1"), SUPPORT), c2, ("c2", "e1"), REFUTE),
        e2, ("c2", "e2"), SUPPORT,
    )


QUERIES = (
    Query("IMDB-1", "filter: F1 (at least one positive aspect)",
          _filters(_reviews(), F1)),
    Query("IMDB-2", "join: J1 (reviews x aspects)",
          Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT)),
    Query("IMDB-3", "F1 -> J1, dependent",
          Join(_filters(_reviews(), F1), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-4", "F1 -> F4 -> J1, 2 filters then 1 join",
          Join(_filters(_reviews(), F1, F4), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-5", "F1 -> F4 -> F5 -> J1, 3 filters then 1 join",
          Join(_filters(_reviews(), F1, F4, F5), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-6", "F1 -> F4, 2 filters, no join",
          _filters(_reviews(), F1, F4)),
    Query("IMDB-7", "F1 -> F4 -> F5, 3 filters, no join",
          _filters(_reviews(), F1, F4, F5)),
    Query("IMDB-8", "2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
          "(ASPECT_SENTIMENT), reviews x aspects x aspects",
          Join(Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT),
               _aspects("a2"), ("r", "a2"), ASPECT_SENTIMENT)),
    Query("IMDB-9", "3J chain r1-a1-r2-a2: two reviews discuss the same "
          "aspect, second review positive about another",
          _imdb_chain(_reviews("r1"))),
    Query("IMDB-10", "F1 -> 3J chain r1-a1-r2-a2",
          _imdb_chain(_filters(_reviews("r1"), F1))),

    Query("BIO-1", "filter: serious adverse event",
          _filters(_reports(), SERIOUS_ADVERSE_EVENT)),
    Query("BIO-2", "join: J1 (reports x terms)",
          Join(_reports(), _terms(), ("r", "m"), REACTION)),
    Query("BIO-3", "serious adverse event -> reaction join",
          Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT), _terms(),
               ("r", "m"), REACTION)),
    Query("BIO-4", "3F + 2J: serious reports with neurological and "
          "cardiovascular reactions",
          Join(Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT),
                    _filters(_terms("n"), NEUROLOGICAL_REACTION),
                    ("r", "n"), REACTION),
               _filters(_terms("c"), CARDIOVASCULAR_REACTION),
               ("r", "c"), REACTION)),

    Query("FEV-1", "filter: F11 (about a person)", _filters(_claims(), F11)),
    Query("FEV-2", "join: J3 (claims x evidence)",
          Join(_claims(), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-3", "F11 -> J3, dependent",
          Join(_filters(_claims(), F11), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-4", "F11 -> F12 -> J3, 2 filters then 1 join",
          Join(_filters(_claims(), F11, F12), _evidence(), ("c", "e"),
               SUPPORT)),
    Query("FEV-5", "2F + 1J: two-sided pushdown - F11 on claims, F13 on "
          "evidence, each filtered before J3",
          Join(_filters(_claims(), F11), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-6", "3F + 1J: two-sided pushdown, deeper - F11 -> F12 on "
          "claims, F13 on evidence, each filtered before J3",
          Join(_filters(_claims(), F11, F12), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-7", "2J, same anchor: J1 (SUPPORT) -> J2 (REFUTE), "
          "claims x evidence x evidence",
          Join(Join(_claims(), _evidence(), ("c", "e"), SUPPORT),
               _evidence("e2"), ("c", "e2"), REFUTE)),
    Query("FEV-8", "3J chain c1-e1-c2-e2: evidence supports c1 but refutes "
          "c2, c2 supported by different evidence",
          _fever_chain(_claims("c1"), _evidence("e1"), _claims("c2"),
                       _evidence("e2"))),
    Query("FEV-9", "4F + 3J: F11 on c1 and c2, F13 on e1 and e2, then the "
          "c1-e1-c2-e2 join chain",
          _fever_chain(_filters(_claims("c1"), F11),
                       _filters(_evidence("e1"), F13),
                       _filters(_claims("c2"), F11),
                       _filters(_evidence("e2"), F13))),
    # A claim names its Wikipedia page, and an evidence row's id is its
    # page name, so SUPPORT is asked only of a claim and its own page.
    Query("FEV-10", "2F + 1J over pairs: F11 on claims, F13 on evidence, "
          "SUPPORT asked only of a claim and its own Wikipedia page",
          Join(_filters(_claims("c", "evidence_wiki_url"), F11),
               _filters(_evidence(), F13), ("c", "e"), SUPPORT,
               on=(("evidence_wiki_url", "id"),))),

    Query("LEP-1", "filter: LEP1 (reasoning does not apply)",
          _filters(_contexts(), LEP1)),
    Query("LEP-2", "join: citation contexts x cited passages",
          Join(_contexts(), _passages(), ("d", "s"), LEPJOIN)),
    Query("LEP-3", "LEP1 -> join, dependent",
          Join(_filters(_contexts(), LEP1), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-4", "LEP1 -> LEP2 -> join, 2 filters then 1 join",
          Join(_filters(_contexts(), LEP1, LEP2), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-5", "2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
          "LEPS1 on passages, each filtered before the join",
          Join(_filters(_contexts(), LEP1, LEP2),
               _filters(_passages(), LEPS1), ("d", "s"), LEPJOIN)),
    Query("AGENT-1", "filter: recovered after an unsuccessful approach",
          _filters(_traces(), AGENT_RECOVERED)),
    Query("AGENT-2", "filter: implemented a plausible fix",
          _filters(_traces(), AGENT_IMPLEMENTED_FIX)),

    # CUAD: contract PDFs as images. A page row is one page; a
    # contract row is every page of a contract with at most 32 pages.
    Query("CUAD-1", "filter over pages: caps liability",
          _filters(_pages(), CUAD_PAGE_CAPS_LIABILITY)),
    Query("CUAD-2", "2 filters over pages: caps liability -> leaves some "
          "liability uncapped (a carve-out on the same page)",
          _filters(_pages(), CUAD_PAGE_CAPS_LIABILITY,
                   CUAD_PAGE_UNCAPPED_LIABILITY)),
    Query("CUAD-3", "filter over contracts of at most 32 pages: rights upon "
          "a change of control",
          _filters(_contracts(), CUAD_CHANGE_OF_CONTROL)),
    Query("CUAD-4", "2 filters over contracts of at most 32 pages: "
          "exclusivity -> non-compete",
          _filters(_contracts(), CUAD_EXCLUSIVITY, CUAD_NON_COMPETE)),
    Query("CUAD-5", "3 filters over contracts of at most 32 pages: license "
          "grant -> non-transferable -> irrevocable or perpetual",
          _filters(_contracts(), CUAD_LICENSE_GRANT,
                   CUAD_NON_TRANSFERABLE_LICENSE, CUAD_PERPETUAL_LICENSE)),

    # PrivacyPolicies: only when that corpus is available.
    Query("PRIV-1", "2 filters: P_MSG + P_LOC",
          _filters(_policies(), P_MSG, P_LOC), privacy=True),
    Query("PRIV-2", "2 filters + 1 join: P_MSG + P_LOC -> scenarios",
          Join(_filters(_policies(), P_MSG, P_LOC),
               Scan("scenarios", "s", "scenario"), ("p", "s"),
               SCENARIO_MATCH), privacy=True),

    # A question's filing is known, so the join asks about a question
    # and the pages of that one filing; the engine anchors on the page.
    Query("FIN-1", "join over filing pages: which pages of a question's "
          "filing hold its evidence",
          _evidence_join(_questions())),
    Query("FIN-2", "filter on questions -> join over filing pages: the "
          "evidence pages of the questions that need a calculation",
          _evidence_join(_filters(_questions(), FIN_NEEDS_CALCULATION))),

    # A question row names one statement, so the join asks about the
    # question and the pages of that statement; the engine anchors on
    # the page.
    Query("TREAS-1", "join over Treasury statement pages: which pages of "
          "the statement a question reads hold its answer",
          _provenance_join(_treasury_questions())),
    Query("TREAS-2", "filter on questions -> join over statement pages: the "
          "answer pages of the questions that combine several figures",
          _provenance_join(_filters(_treasury_questions(),
                                    TREAS_COMBINES_FIGURES))),
)


def plan_json(plan: plan_pb2.Plan) -> str:
    """Return a plan as ProtoJSON with sorted keys and a trailing newline."""
    return json_format.MessageToJson(
        plan,
        preserving_proto_field_name=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_plans(directory: Path) -> None:
    """Write every query plan and the catalog into a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    catalog = []
    for query in QUERIES:
        (directory / f"{query.id}.json").write_text(
            plan_json(build_plan(query.tree))
        )
        catalog.append({
            "id": query.id,
            "description": query.description,
            "privacy": query.privacy,
        })
    (directory / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n"
    )


def main(argv=None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    directory = Path(arguments[0]) if arguments else PLANS_DIR
    write_plans(directory)
    print(f"wrote {len(QUERIES)} plans to {directory}")


if __name__ == "__main__":
    main()
