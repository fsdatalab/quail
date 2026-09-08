"""The QUAIL-B queries as data, for any engine's runner to build.

A query is a base relation with filters, then joins that each add one
more relation, then a projection. Written order matters: an alias's
filters apply in the order listed, and joins apply in the order listed.
The runner of one engine turns a `QuerySpec` into that engine's query;
the scoring reads the same spec to know which label answers each
predicate.
"""

from __future__ import annotations

from dataclasses import dataclass

from quail_bench.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F7,
    F11,
    F12,
    F13,
    LEP1,
    LEP2,
    LEP3,
    LEP4,
    LEP5,
    LEPJOIN,
    LEPS1,
    P_LOC,
    P_MSG,
    REACTION,
    REFUTE,
    SCENARIO_MATCH,
    SUPPORT,
)

# Fixed planner inputs from the sf=0.1 Qwen3 32B fp8 labels.
# They apply at every scale factor so query planning does not read answers.
SELECTIVITY_ESTIMATE_COLLECTION = "gt_77bb8b128743a79aedddaa24c808c3f8"
SELECTIVITY_ESTIMATE_CORPUS = "c_1aa2c4f0d0b6c816fd37aa5748c33341"
SELECTIVITY_ESTIMATE_SCALE_FACTOR = 0.1
FILTER_SELECTIVITY_ESTIMATES = {
    F1: 4004 / 5000,
    F4: 1218 / 5000,
    F5: 2853 / 5000,
    F7: 306 / 500,
    AGENT_RECOVERED: 570 / 1772,
    AGENT_IMPLEMENTED_FIX: 537 / 1772,
    F11: 296 / 500,
    F12: 69 / 500,
    F13: 159 / 287,
    LEP1: 14 / 500,
    LEP2: 229 / 500,
    LEP3: 51 / 500,
    LEP4: 31 / 500,
    LEP5: 14 / 500,
    LEPS1: 351 / 433,
}
JOIN_SELECTIVITY_ESTIMATES = {
    DISCUSS_ASPECT: 17683 / 60000,
    ASPECT_SENTIMENT: 9635 / 60000,
    REACTION: 19144 / 563500,
    SUPPORT: 311 / 143500,
    REFUTE: 477 / 143500,
    LEPJOIN: 500 / 216500,
}


@dataclass(frozen=True)
class AliasSpec:
    """One use of one document table in a query."""

    alias: str
    table: str
    column: str                    # the document text column
    filters: tuple[str, ...] = ()  # prompt templates, written order


@dataclass(frozen=True)
class JoinSpec:
    """One binary AI join; aliases in placeholder order."""

    template: str
    aliases: tuple[str, str]


@dataclass(frozen=True)
class QuerySpec:
    """One benchmark query.

    The first alias is the base relation. Join i adds alias i + 1, with
    that alias's filters applied before the join. `select` names the
    output columns as alias.column.
    """

    id: str
    description: str
    aliases: tuple[AliasSpec, ...]
    joins: tuple[JoinSpec, ...]
    select: tuple[str, ...]

    def __post_init__(self):
        if len(self.aliases) != len(self.joins) + 1:
            raise ValueError(
                f"{self.id}: {len(self.aliases)} aliases need "
                f"{len(self.aliases) - 1} joins, not {len(self.joins)}")
        names = {spec.alias for spec in self.aliases}
        for index, join in enumerate(self.joins):
            added = self.aliases[index + 1].alias
            if added not in join.aliases or not set(join.aliases) <= names:
                raise ValueError(
                    f"{self.id}: join {index} must add alias {added!r}")

    def alias(self, name: str) -> AliasSpec:
        return next(spec for spec in self.aliases if spec.alias == name)

    @property
    def base_alias(self) -> str:
        return self.aliases[0].alias

    @property
    def filter_templates(self) -> tuple[str, ...]:
        return tuple(
            template for spec in self.aliases for template in spec.filters)

    @property
    def has_estimates(self) -> bool:
        return (
            all(template in FILTER_SELECTIVITY_ESTIMATES
                for template in self.filter_templates)
            and all(join.template in JOIN_SELECTIVITY_ESTIMATES
                    for join in self.joins)
        )

    @property
    def order(self) -> str:
        """The filter order rule: by cost when every predicate has an estimate."""
        return "by_cost" if self.has_estimates else "as_written"

    @staticmethod
    def filter_selectivity(template: str) -> float | None:
        return FILTER_SELECTIVITY_ESTIMATES.get(template)

    @staticmethod
    def join_selectivity(template: str) -> float | None:
        return JOIN_SELECTIVITY_ESTIMATES.get(template)


def _query(query_id, description, base, joins=(), select=None) -> QuerySpec:
    """Build one query from a base table and joins on it.

    base: (table, alias, column, filters). joins: list of
    (partner_table, partner_alias, partner_column, template
    [, partner_filters]); each join pairs the base alias with the
    partner in that placeholder order.
    """
    table, alias, column, filters = base
    aliases = [AliasSpec(alias, table, column, tuple(filters))]
    join_specs = []
    for partner_table, partner_alias, partner_column, template, *rest in joins:
        aliases.append(AliasSpec(
            partner_alias, partner_table, partner_column,
            tuple(rest[0]) if rest else ()))
        join_specs.append(JoinSpec(template, (alias, partner_alias)))
    if select is None:
        select = tuple(f"{spec.alias}.id" for spec in aliases)
    return QuerySpec(query_id, description, tuple(aliases),
                     tuple(join_specs), tuple(select))


QUERIES = (
    # IMDB: filter alone, join alone, then filter-chain depth 1/2/3
    # feeding the one join (reviews x aspects).
    _query("IMDB-1", "filter: F1 (at least one positive aspect)",
           ("reviews", "r", "body", [F1])),
    _query("IMDB-2", "join: J1 (reviews x aspects)",
           ("reviews", "r", "body", []),
           [("aspects", "a", "aspect", DISCUSS_ASPECT)]),
    _query("IMDB-3", "F1 -> J1, dependent",
           ("reviews", "r", "body", [F1]),
           [("aspects", "a", "aspect", DISCUSS_ASPECT)]),
    _query("IMDB-4", "F1 -> F4 -> J1, 2 filters then 1 join",
           ("reviews", "r", "body", [F1, F4]),
           [("aspects", "a", "aspect", DISCUSS_ASPECT)]),
    _query("IMDB-5", "F1 -> F4 -> F5 -> J1, 3 filters then 1 join",
           ("reviews", "r", "body", [F1, F4, F5]),
           [("aspects", "a", "aspect", DISCUSS_ASPECT)]),
    # filter chains with no join: IMDB-4 and IMDB-5 without their join,
    # so the difference is the join's cost.
    _query("IMDB-6", "F1 -> F4, 2 filters, no join",
           ("reviews", "r", "body", [F1, F4])),
    _query("IMDB-7", "F1 -> F4 -> F5, 3 filters, no join",
           ("reviews", "r", "body", [F1, F4, F5])),
    # IMDB-8: star shape - A joins B and A joins C, same anchor
    # throughout, barrier between stages but no anchor switch.
    _query("IMDB-8", "2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
           "(ASPECT_SENTIMENT), reviews x aspects x aspects",
           ("reviews", "r", "body", []),
           [("aspects", "a", "aspect", DISCUSS_ASPECT),
            ("aspects", "a2", "aspect", ASPECT_SENTIMENT)]),
    # IMDB-9/IMDB-10: 3-join chain r1-a1-r2-a2. Two reviews that
    # discuss the same aspect; what other aspect does the second
    # review feel positively about?
    QuerySpec(
        "IMDB-9",
        "3J chain r1-a1-r2-a2: two reviews discuss the same aspect, "
        "second review positive about another",
        (AliasSpec("r1", "reviews", "body"),
         AliasSpec("a1", "aspects", "aspect"),
         AliasSpec("r2", "reviews", "body"),
         AliasSpec("a2", "aspects", "aspect")),
        (JoinSpec(DISCUSS_ASPECT, ("r1", "a1")),
         JoinSpec(DISCUSS_ASPECT, ("r2", "a1")),
         JoinSpec(ASPECT_SENTIMENT, ("r2", "a2"))),
        ("r1.id", "a1.id", "r2.id", "a2.id"),
    ),
    QuerySpec(
        "IMDB-10",
        "F1 -> 3J chain r1-a1-r2-a2",
        (AliasSpec("r1", "reviews", "body", (F1,)),
         AliasSpec("a1", "aspects", "aspect"),
         AliasSpec("r2", "reviews", "body"),
         AliasSpec("a2", "aspects", "aspect")),
        (JoinSpec(DISCUSS_ASPECT, ("r1", "a1")),
         JoinSpec(DISCUSS_ASPECT, ("r2", "a1")),
         JoinSpec(ASPECT_SENTIMENT, ("r2", "a2"))),
        ("r1.id", "a1.id", "r2.id", "a2.id"),
    ),

    # BioDEX: filter, join, filter->join on long medical reports.
    # Deeper chains and multi-join shapes (star, 3J) are covered by
    # the IMDB queries; BioDEX adds long-document behavior.
    _query("BIO-1", "filter: F7 (female patient)",
           ("reports", "r", "report", [F7])),
    _query("BIO-2", "join: J1 (reports x terms)",
           ("reports", "r", "report", []),
           [("terms", "m", "term", REACTION)]),
    _query("BIO-3", "F7 -> J1, dependent",
           ("reports", "r", "report", [F7]),
           [("terms", "m", "term", REACTION)]),

    # FEVER: filter alone, join alone, filter chain to depth 2 only
    # (depth 3 yields 0 rows), plus two-sided FEV-5/FEV-6 pushdown.
    _query("FEV-1", "filter: F11 (about a person)",
           ("claims", "c", "claim", [F11])),
    _query("FEV-2", "join: J3 (claims x evidence)",
           ("claims", "c", "claim", []),
           [("evidence", "e", "text", SUPPORT)]),
    _query("FEV-3", "F11 -> J3, dependent",
           ("claims", "c", "claim", [F11]),
           [("evidence", "e", "text", SUPPORT)]),
    _query("FEV-4", "F11 -> F12 -> J3, 2 filters then 1 join",
           ("claims", "c", "claim", [F11, F12]),
           [("evidence", "e", "text", SUPPORT)]),
    _query("FEV-5", "2F + 1J: two-sided pushdown - F11 on claims, F13 on "
           "evidence, each filtered before J3",
           ("claims", "c", "claim", [F11]),
           [("evidence", "e", "text", SUPPORT, [F13])]),
    _query("FEV-6", "3F + 1J: two-sided pushdown, deeper - F11 -> F12 on "
           "claims, F13 on evidence, each filtered before J3",
           ("claims", "c", "claim", [F11, F12]),
           [("evidence", "e", "text", SUPPORT, [F13])]),
    # FEV-7: star shape, both joins anchored on claims.
    _query("FEV-7", "2J, same anchor: J1 (SUPPORT) -> J2 (REFUTE), "
           "claims x evidence x evidence",
           ("claims", "c", "claim", []),
           [("evidence", "e", "text", SUPPORT),
            ("evidence", "e2", "text", REFUTE)]),
    # FEV-8/FEV-9: 3-join chain c1-e1-c2-e2. Evidence e1 supports
    # claim c1 but refutes claim c2; claim c2 is supported by
    # different evidence e2.
    QuerySpec(
        "FEV-8",
        "3J chain c1-e1-c2-e2: evidence supports c1 but refutes c2, "
        "c2 supported by different evidence",
        (AliasSpec("c1", "claims", "claim"),
         AliasSpec("e1", "evidence", "text"),
         AliasSpec("c2", "claims", "claim"),
         AliasSpec("e2", "evidence", "text")),
        (JoinSpec(SUPPORT, ("c1", "e1")),
         JoinSpec(REFUTE, ("c2", "e1")),
         JoinSpec(SUPPORT, ("c2", "e2"))),
        ("c1.id", "e1.id", "c2.id", "e2.id"),
    ),
    QuerySpec(
        "FEV-9",
        "4F + 3J: F11 on c1 and c2, F13 on e1 and e2, then the "
        "c1-e1-c2-e2 join chain",
        (AliasSpec("c1", "claims", "claim", (F11,)),
         AliasSpec("e1", "evidence", "text", (F13,)),
         AliasSpec("c2", "claims", "claim", (F11,)),
         AliasSpec("e2", "evidence", "text", (F13,))),
        (JoinSpec(SUPPORT, ("c1", "e1")),
         JoinSpec(REFUTE, ("c2", "e1")),
         JoinSpec(SUPPORT, ("c2", "e2"))),
        ("c1.id", "e1.id", "c2.id", "e2.id"),
    ),

    # LePaRD uses two deduplicated projections of sampled citation pairs.
    _query("LEP-1", "filter: LEP1 (reasoning does not apply)",
           ("citation_contexts", "d", "destination_context", [LEP1])),
    _query("LEP-2", "join: citation contexts x cited passages",
           ("citation_contexts", "d", "destination_context", []),
           [("citation_passages", "s", "passage_text", LEPJOIN)]),
    _query("LEP-3", "LEP1 -> join, dependent",
           ("citation_contexts", "d", "destination_context", [LEP1]),
           [("citation_passages", "s", "passage_text", LEPJOIN)]),
    _query("LEP-4", "LEP1 -> LEP2 -> join, 2 filters then 1 join",
           ("citation_contexts", "d", "destination_context", [LEP1, LEP2]),
           [("citation_passages", "s", "passage_text", LEPJOIN)]),
    _query("LEP-5", "LEP1 -> LEP2 -> LEP3 -> join, 3 filters then 1 join",
           ("citation_contexts", "d", "destination_context",
            [LEP1, LEP2, LEP3]),
           [("citation_passages", "s", "passage_text", LEPJOIN)]),
    _query("LEP-6", "LEP1..LEP5 -> join, 5 filters then 1 join",
           ("citation_contexts", "d", "destination_context",
            [LEP1, LEP2, LEP3, LEP4, LEP5]),
           [("citation_passages", "s", "passage_text", LEPJOIN)]),
    _query("LEP-7", "2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
           "LEPS1 on passages, each filtered before the join",
           ("citation_contexts", "d", "destination_context", [LEP1, LEP2]),
           [("citation_passages", "s", "passage_text", LEPJOIN, [LEPS1])]),
    # LEP-6 without its join: the deepest filter chain in the suite,
    # five stages of KV reuse with no join work mixed in.
    _query("LEP-8", "LEP1..LEP5, 5 filters, no join",
           ("citation_contexts", "d", "destination_context",
            [LEP1, LEP2, LEP3, LEP4, LEP5])),

    # SWE-Next: semantic filters over cumulative agent trace snapshots.
    _query("AGENT-1", "filter: recovered after an unsuccessful approach",
           ("agent_traces", "t", "trace", [AGENT_RECOVERED])),
    _query("AGENT-2", "filter: implemented a plausible fix",
           ("agent_traces", "t", "trace", [AGENT_IMPLEMENTED_FIX])),
)

# PrivacyPolicies: only when that corpus is available.
PRIVACY_QUERIES = (
    _query("PRIV-1", "2 filters: P_MSG + P_LOC",
           ("policies", "p", "policy_text", [P_MSG, P_LOC])),
    _query("PRIV-2", "2 filters + 1 join: P_MSG + P_LOC -> scenarios",
           ("policies", "p", "policy_text", [P_MSG, P_LOC]),
           [("scenarios", "s", "scenario", SCENARIO_MATCH)]),
)

QUERY_ORDER = tuple(spec.id for spec in QUERIES)

QUERY_FAMILY_WORKLOADS = {
    "IMDB": "imdb",
    "BIO": "biodex",
    "FEV": "fever",
    "LEP": "lepard",
    "AGENT": "agent",
}


def queries(include_privacy: bool = False) -> dict[str, QuerySpec]:
    """Return the benchmark queries by id, in benchmark order."""
    specs = QUERIES + (PRIVACY_QUERIES if include_privacy else ())
    return {spec.id: spec for spec in specs}


def split_query_ids(ids, containers):
    """Split query IDs into the same equal chunks as stock vLLM."""
    count, extra = divmod(len(ids), containers)
    chunks = []
    start = 0
    for index in range(containers):
        size = count + (1 if index < extra else 0)
        chunks.append(tuple(ids[start:start + size]))
        start += size
    return tuple(chunk for chunk in chunks if chunk)


def split_query_families(ids):
    """Return one ordered query group for each QUAIL-B family."""
    groups = tuple(
        tuple(query_id for query_id in ids
              if query_id.split("-", 1)[0] == prefix)
        for prefix in QUERY_FAMILY_WORKLOADS
    )
    assigned = {query_id for group in groups for query_id in group}
    unknown = [query_id for query_id in ids if query_id not in assigned]
    if unknown:
        raise ValueError(f"unknown query family for {unknown}")
    return tuple(group for group in groups if group)


def query_family_name(ids):
    """Return the name for one query family."""
    prefixes = {query_id.split("-", 1)[0] for query_id in ids}
    if len(prefixes) != 1:
        raise ValueError(
            f"expected one query family, found {sorted(prefixes)}")
    prefix = prefixes.pop()
    try:
        return QUERY_FAMILY_WORKLOADS[prefix]
    except KeyError as error:
        raise ValueError(f"unknown query family {prefix!r}") from error
