"""Ideal SoL work estimate for every QuailB query.

The estimate prices three model components on one H100! request:

  1. attention projections, with their weight bytes and fp8 FLOPs
  2. the MLP, with its weight bytes and fp8 FLOPs
  3. attention, with its KV bytes and bf16 pair FLOPs

No measured or fitted performance constant appears in the calculation. The
estimate omits runtime overhead and uses ideal query-wide packing and unlimited
document prefix KV. Its join search is restricted to eager binary full left
deep plans. It is therefore an optimistic comparison point for that modeled
execution, not the exact minimum for every possible execution. The dollar
metric uses Modal's published H100! price.

The equations are in plans/sol_model.md. Work counting, model components, and
the component calculation live in shared planner modules. The exact SoL join
search is separate from the production planner.

KV reuse, the part that has to be right
---------------------------------------
Every document has a PREFIX: the shared preamble plus the document text. In
this estimate, its KV is computed once and stays resident. A filter appends
its question and removes the question KV after the answer. A join appends one
anchor frame, then streams many tuple suffixes over that framed context. Each
tuple suffix contains the partner label, partner document, and answer cue. Its
KV is computed, used once, and removed (`executor/pack.py` never retains tuple
suffix KV).

The estimate therefore computes a document prefix once however many questions
get asked about it. Each later question still reads that prefix from KV. Three
operations follow:

    scan()      compute a prefix and its first suffix, from nothing
    ask()       reuse a resident prefix, attach one more suffix
    stream()    reuse a resident prefix, attach many suffixes (a join)

`ask` and `stream` never charge for the document again; `scan` is
the only one that does.

Join search
-----------
All filters run first. The script then checks every eager binary full left
deep relation order and anchor choice supported by the search. It immediately
applies every available crossing predicate. It does not check bushy plans or
plans that delay a crossing predicate. It uses exact ground truth survivors at
every step and unlimited KV. The search runs separately for 4B and 32B. It
does not call or simulate the production planner.

Running it
----------
The corpora and the per-document ground-truth labels are raw data
and live on the quail-results volume, so pull them first. The
answers go back to the volume too:

    W=<workdir>; SF=0.1
    G=/ground_truth/quailb/schema_v1
    mkdir -p $W/data $W/allabels
    modal volume get quail-results /quailb_data/sf$SF $W/data/
    modal volume get quail-results $G/label_sets $W/allabels/
    modal volume get quail-results \
        $G/collections/gt_363b5ab570635c33894e1a030c21f57e/manifest.json \
        $W/collection_manifest.json
    uv run --with transformers --with pyarrow \
        python reports/make_sol_quailb.py $W $SF
    modal volume put quail-results $W/sol_quailb_sf$SF.json \
        /sol/sol_quailb_sf$SF.json

The scale factor defaults to 0.1. The collection supplies all four
query families. The run stops if the collection is for a different
scale factor than the one requested.

The report is reports/2026-08-29-sol-quailb.md.
"""
import collections
import itertools
import json
import math
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

import quail
from quail.bench import quailb as Q
from quail.bench.evaluate import H100_PRICE_SOURCE, H100_USD_PER_HOUR
from quail.bench.sol_dp import PairRelation, exact_live_rows
from quail.logical import SHARED_PRE, ColumnRef, bind_join_prompt, bind_prompt
from quail.planner import budgets
from quail.planner.decide import (
    collect_operators,
    default_order_rule,
    join_specs,
    order_filters_indexed,
)
from quail.planner.joins import fit_resident_documents, search_joins, summarize_alias
from quail.planner.leftdeep import Extension, optimize_left_deep
from quail.planner.plan import EngineConfig, Refusal
from quail.planner.sol import speed_of_light
from quail.planner.work import Work, ask, scan
from quail.backends.quail.coordinator import runtime_join_steps, thin_survivors
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8, ModelSpec

W = Path(sys.argv[1])
SF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
# "%g" so 0.1 stays "0.1" and 0.01 stays "0.01", matching the volume's
# own directory names
TAG = f"sf{SF:g}"
OUT = W / f"sol_quailb_{TAG}.json"
ROOT = Path(__file__).resolve().parents[1]

# check the workdir holds this scale factor before tokenizing anything:
# both of these otherwise surface much later as a missing parquet file
if not (W / "data" / TAG).is_dir():
    raise SystemExit(f"no corpus at {W / 'data' / TAG}: pull "
                     f"/quailb_data/{TAG} off the volume")
COLLECTION = json.load(open(W / "collection_manifest.json"))
if COLLECTION["scale_factor"] != SF:
    raise SystemExit(
        f"the collection in {W} is scale factor "
        f"{COLLECTION['scale_factor']:g}, not {SF:g}: pull the labels for "
        f"the corpus you are asking about")
COLLECTION_ID = COLLECTION["collection_id"]
CORPUS_ID = COLLECTION["corpus_id"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")


@cache
def encode(text):
    return tuple(tok(text, add_special_tokens=False)["input_ids"])


def length(text):
    return len(encode(text))


PRE = length(SHARED_PRE)   # the engine preamble, 2 tokens

# The batch size each model's forward pass runs at: (2^31 - 1) over
# the widest projection, the fused kernels' 32-bit offset limit. It
# decides how often the weights are re-read, and it is a planner
# choice rather than a hardware property, so it is stated here.
CHUNK = {"qwen3-4b-fp8": 110_376, "qwen3-32b-fp8": 41_943}
MODELS = [QWEN3_4B_FP8, QWEN3_32B_FP8]


# ================================================================
# PART 2: measuring the three inputs
# ================================================================

COLUMNS = {
    "reviews.body": ("reviews", "body"),
    "aspects.aspect": ("aspects", "aspect"),
    "reports.report": ("reports", "report"),
    "terms.term": ("terms", "term"),
    "claims.claim": ("claims", "claim"),
    "evidence.text": ("evidence", "text"),
    "citation_contexts.destination_context":
        ("citation_contexts", "destination_context"),
    "citation_passages.passage_text":
        ("citation_passages", "passage_text"),
}

# 1. document lengths -------------------------------------------------
lengths = {}        # column -> {doc id: token count}
for key, (table, col) in COLUMNS.items():
    t = pq.read_table(W / "data" / TAG / f"{table}.parquet",
                      columns=["id", col])
    lengths[key] = {i: length(x) for i, x in
                    zip(t.column("id").to_pylist(), t.column(col).to_pylist())}
    v = lengths[key].values()
    print(f"{key:32} {len(v):>6} docs  {sum(v):>10,} tokens  "
          f"mean {sum(v) / len(v):>8.1f}", flush=True)

# 2. prompt lengths ---------------------------------------------------
FILTER_TEMPLATES = {c: getattr(Q, c) for c in (
    "F1", "F4", "F5", "F7", "F11", "F12", "F13",
    "LEP1", "LEP2", "LEP3", "LEP4", "LEP5", "LEPS1")}
JOIN_TEMPLATES = {c: getattr(Q, c) for c in (
    "DISCUSS_ASPECT", "ASPECT_SENTIMENT", "REACTION",
    "SUPPORT", "REFUTE", "LEPJOIN")}
col_ref = (ColumnRef("x", "t", "c"),)
question = {c: bind_prompt(t, col_ref, encode).tail_tokens
            for c, t in FILTER_TEMPLATES.items()}
join_refs = (ColumnRef("left", "left_table", "text"),
             ColumnRef("right", "right_table", "text"))
join_prompt = {}
for code, template in JOIN_TEMPLATES.items():
    prompt = bind_join_prompt(template, join_refs, encode)
    join_prompt[code] = {
        "left_frame": prompt.labels[0][2],
        "right_frame": prompt.labels[1][2],
        "left_label": prompt.labels[0][1],
        "right_label": prompt.labels[1][1],
        "tail": prompt.tail_tokens,
    }

# 3. labels -----------------------------------------------------------
active_by_predicate = dict(COLLECTION["label_sets"])
ACTIVE = set(active_by_predicate.values())
LABEL_MANIFESTS = list(
    (W / "allabels" / "label_sets").glob("*/*/*/manifest.json"))
filter_answers = {}
join_answers = {}
predicate_meta = {}
template_codes = {}
for m in LABEL_MANIFESTS:
    if Path(m).parent.name not in ACTIVE:
        continue
    meta = json.load(open(m))["predicate"]
    rows = pq.read_table(Path(m).parent / "labels.parquet",
                         columns=["left_id", "right_id",
                                  "answer"]).to_pylist()
    code = meta["key"]
    predicate_meta[code] = meta
    left_ref = ColumnRef("left", meta["left_table"], meta["left_column"])
    if meta["kind"] == "filter":
        prompt = bind_prompt(meta["template"], (left_ref,), encode)
        filter_answers[code] = {
            str(row["left_id"]): bool(row["answer"]) for row in rows}
    else:
        right_ref = ColumnRef(
            "right", meta["right_table"], meta["right_column"])
        prompt = bind_join_prompt(
            meta["template"], (left_ref, right_ref), encode)
        join_answers[code] = {
            (str(row["left_id"]), str(row["right_id"])):
                bool(row["answer"])
            for row in rows}
    if prompt.template in template_codes:
        raise ValueError(f"duplicate prompt template for {code}")
    template_codes[prompt.template] = code

if len(predicate_meta) != len(ACTIVE):
    raise ValueError(
        f"loaded {len(predicate_meta)} active predicates, expected "
        f"{len(ACTIVE)}")


def prompt_code(prompt) -> str:
    try:
        return template_codes[prompt.template]
    except KeyError as error:
        raise KeyError("query prompt has no active ground truth") from error


def prompt_answer(prompt, assignment, aliases) -> bool:
    code = prompt_code(prompt)
    ids = [str(aliases[arg.alias]["ids"][assignment[arg.alias]])
           for arg in prompt.args]
    if len(ids) == 1:
        return filter_answers[code][ids[0]]
    if len(ids) == 2:
        return join_answers[code][(ids[0], ids[1])]
    raise NotImplementedError("SoL supports prompts with one or two documents")


def prompt_token_counts(prompt):
    labels_by_alias = {
        alias: {"label": label_tokens, "frame": frame_tokens}
        for alias, label_tokens, frame_tokens in prompt.labels}
    return labels_by_alias, prompt.tail_tokens


def join_stage_work(anchor, partners, aliases, survivors, prompt,
                    resident_rows) -> Work:
    """One stage's Work. Anchor rows in resident_rows have their
    prefix KV in the arena and pay the frame only; the rest scan."""
    labels_by_alias, tail = prompt_token_counts(prompt)
    partner_rows = list(itertools.product(
        *[survivors[alias] for alias in partners]))
    suffixes = [
        tail + sum(labels_by_alias[alias]["label"]
                   + aliases[alias]["tokens"][row]
                   for alias, row in zip(partners, partner_row))
        for partner_row in partner_rows
    ]
    suffix_tokens = sum(suffixes)
    suffix_triangles = sum(suffix * (suffix + 1) / 2
                           for suffix in suffixes)
    frame = labels_by_alias[anchor]["frame"]
    resident_rows = set(resident_rows)
    work = Work()
    for row in survivors[anchor]:
        prefix = PRE + aliases[anchor]["tokens"][row]
        work = work + (ask(prefix, frame) if row in resident_rows
                       else scan(prefix, frame))
        anchor_prefix = prefix + frame
        work = work + Work(
            tokens=suffix_tokens,
            pairs=anchor_prefix * suffix_tokens + suffix_triangles,
            kv_written=suffix_tokens,
            kv_read=anchor_prefix,
        )
    return work


@dataclass
class QueryInputs:
    plan: object
    scans: list
    filters: dict
    joins: list
    aliases: dict
    survivors: dict
    work: Work
    resident: set
    filter_stages: list
    filter_evaluations: int
    post_filter_counts: dict


def prepare_query(query, model: ModelSpec, chunk_tokens: int,
                  plan=None) -> QueryInputs:
    scans, filters, joins = collect_operators(query.logical)
    if isinstance(plan, Refusal):
        raise ValueError(f"query was refused: {plan.reasons}")
    aliases = {}
    for scan_node in scans:
        key = f"{scan_node.provider}.{scan_node.column}"
        ids = list(lengths[key])
        aliases[scan_node.alias] = {
            "column": key,
            "ids": ids,
            "tokens": [lengths[key][doc_id] for doc_id in ids],
        }
    if plan is None:
        rule = (query.order if query.order is not None else
                default_order_rule(filters, joins)[0])
        filter_orders = {
            alias: order_filters_indexed(
                predicates, rule,
                prefix_tokens=(PRE + (
                    sum(aliases[alias]["tokens"])
                    / len(aliases[alias]["tokens"])
                    if aliases[alias]["tokens"] else 0)),
                model=model, device=H100_SXM,
                chunk_tokens=chunk_tokens)
            for alias, predicates in filters.items()
        }
    else:
        filter_orders = {
            node.alias: [stage.written_pos for stage in node.stages]
            for node in plan.graph.nodes_by_type("quail.packed_filter")
        }
    survivors = {
        alias: list(range(len(data["ids"])))
        for alias, data in aliases.items()}
    work = Work()
    resident = set()
    filter_stages = []
    filter_evaluations = 0

    for alias, order in filter_orders.items():
        live = survivors[alias]
        for stage_index, written_pos in enumerate(order):
            predicate = filters[alias][written_pos]
            code = prompt_code(predicate.prompt)
            qtokens = predicate.prompt.tail_tokens
            operation = scan if stage_index == 0 else ask
            for row in live:
                prefix = PRE + aliases[alias]["tokens"][row]
                work = work + operation(prefix, qtokens)
            passed = [
                row for row in live
                if prompt_answer(predicate.prompt, {alias: row}, aliases)
            ]
            filter_evaluations += len(live)
            filter_stages.append({
                "alias": alias,
                "code": code,
                "question_tokens": qtokens,
                "provided_selectivity": predicate.selectivity,
                "evaluated": len(live),
                "passed": len(passed),
                "selectivity": (round(len(passed) / len(live), 6)
                                if live else 0.0),
            })
            live = passed
        survivors[alias] = live
        resident.add(alias)

    post_filter_counts = {
        alias: len(rows) for alias, rows in survivors.items()}
    return QueryInputs(
        plan=plan,
        scans=scans,
        filters=filters,
        joins=joins,
        aliases=aliases,
        survivors=survivors,
        work=work,
        resident=resident,
        filter_stages=filter_stages,
        filter_evaluations=filter_evaluations,
        post_filter_counts=post_filter_counts,
    )


def simulate_production_planner(query, model: ModelSpec,
                                chunk_tokens: int):
    """Optional diagnostic for the production planner.

    The simulation uses exact saved answers after every join group.
    At group boundaries it applies the arena's page limit and the
    same prefix tokens per page victim order. It does not reproduce temporary
    overlap between packed GPU chunks.

    The SoL output does not call this function.
    """
    prepared = prepare_query(query, model, chunk_tokens, query.plan())
    plan = prepared.plan
    scans = prepared.scans
    joins = prepared.joins
    aliases = prepared.aliases
    survivors = prepared.survivors
    work = prepared.work
    # the engine retains KV where the plan says so: survivors of
    # keep_kv filter chains, then gate survivors of anchors a later
    # group re-uses
    keep_aliases = {
        node.alias
        for node in plan.graph.nodes_by_type("quail.packed_filter")
        if node.keep_kv
    }
    resident_rows = {alias: (set(rows) if alias in keep_aliases
                             else set())
                     for alias, rows in survivors.items()}
    filter_stages = prepared.filter_stages
    filter_evaluations = prepared.filter_evaluations
    post_filter_counts = prepared.post_filter_counts
    finished_full = []
    join_stages = []
    join_pair_evaluations = 0
    single_join_options = {}

    all_specs = join_specs(joins)
    markers = [dict(semantics=j.semantics, written_pos=i)
               for i, j in enumerate(joins)]
    remaining = set(range(len(joins)))
    already_joined = set()
    search_runs = []
    search_sequence = []

    def possible_anchors(indices):
        out = set()
        for index in indices:
            spec = all_specs[index]
            if spec["semantics"] == "full" and spec.get("anchor_free"):
                out.update(spec["aliases"])
            else:
                out.add(spec["anchor"])
        return out

    def next_group():
        specs = [all_specs[i] for i in sorted(remaining)]
        involved = sorted({a for spec in specs
                           for a in spec["aliases"]})
        found = search_joins(
            specs,
            {a: float(len(survivors[a])) for a in involved},
            {a: summarize_alias(
                (aliases[a]["tokens"][row] for row in survivors[a]),
                resident_flags=(row in resident_rows[a]
                                for row in survivors[a]))
             for a in involved},
            {},
            PRE, chunk_tokens, model, H100_SXM,
            fixed_order=(plan.settings["order_rule"] == "as_written"),
            already_joined=already_joined)
        if found is not None:
            search_runs.append(found)
            nodes = runtime_join_steps(found["seq"], markers)
            return next(node for node in nodes
                        if node["op"] == "AnchoredJoin")

        ordered = sorted(remaining)
        first = ordered[0]
        anchor = all_specs[first]["anchor"]
        group = [first]
        if all_specs[first]["semantics"] == "full":
            for index in ordered[1:]:
                spec = all_specs[index]
                if spec["semantics"] != "full" \
                        or spec["anchor"] != anchor:
                    break
                group.append(index)
        return dict(op="AnchoredJoin", anchor=anchor,
                    stage_idxs=tuple(group))

    while remaining:
        node = next_group()
        search_sequence.extend((index, node["anchor"])
                               for index in node["stage_idxs"])
        stage_defs = [joins[i] for i in node["stage_idxs"]]
        anchor = node["anchor"]
        remaining.difference_update(node["stage_idxs"])
        future_anchors = possible_anchors(remaining)
        group_semantics = stage_defs[-1].semantics
        for stage_index, (join_index, join) in enumerate(
                zip(node["stage_idxs"], stage_defs)):
            prompt = join.predicate
            code = prompt_code(prompt)
            stage_aliases = [arg.alias for arg in prompt.args]
            partners = [alias for alias in stage_aliases if alias != anchor]
            anchor_rows = list(survivors[anchor])
            partner_rows = list(itertools.product(
                *[survivors[alias] for alias in partners]))
            if stage_index > 0:
                stage_resident = set(anchor_rows)
            else:
                stage_resident = resident_rows[anchor] & set(anchor_rows)

            if len(joins) == 1:
                single_join_options = {
                    candidate: join_stage_work(
                        candidate,
                        [alias for alias in stage_aliases
                         if alias != candidate],
                        aliases, survivors, prompt,
                        resident_rows[candidate]
                        & set(survivors[candidate])).tokens
                    for candidate in stage_aliases}

            stage_work = join_stage_work(
                anchor, partners, aliases, survivors, prompt,
                stage_resident)
            work = work + stage_work
            rows = {}
            matched = []
            passing_pairs = 0
            for local_anchor, anchor_row in enumerate(anchor_rows):
                answers = []
                for partner_row in partner_rows:
                    assignment = {anchor: anchor_row}
                    assignment.update(zip(partners, partner_row))
                    answer = prompt_answer(prompt, assignment, aliases)
                    answers.append(answer)
                    passing_pairs += int(answer)
                rows[local_anchor] = answers
                if any(answers):
                    matched.append(anchor_row)
            evaluated = len(anchor_rows) * len(partner_rows)
            join_pair_evaluations += evaluated
            stage_out = {
                "anchor": anchor,
                "partners": partners,
                "anchor_index": anchor_rows,
                "partner_index": partner_rows,
                "rows": rows,
            }
            if join.semantics == "full":
                finished_full.append(stage_out)
            last = stage_index == len(stage_defs) - 1
            if last and group_semantics == "anti":
                survivors[anchor] = [row for row in anchor_rows
                                     if row not in set(matched)]
            else:
                survivors[anchor] = matched
            join_stages.append({
                "code": code,
                "anchor": anchor,
                "partners": partners,
                "evaluated_pairs": evaluated,
                "passing_pairs": passing_pairs,
                "fresh_tokens": stage_work.tokens,
            })
        if anchor in future_anchors:
            resident_rows[anchor] = set(survivors[anchor])
        else:
            resident_rows[anchor] = set()
        thin_survivors(finished_full, survivors)
        for alias, rows in survivors.items():
            resident_rows[alias] &= set(rows)
            if alias not in future_anchors:
                resident_rows[alias].clear()
        resident_rows = fit_resident_documents(
            resident_rows,
            {alias: aliases[alias]["tokens"] for alias in aliases},
            PRE, plan.settings["admission_tokens"],
            budgets.PAGE_TOKENS)
        for join_index in node["stage_idxs"]:
            already_joined.update(all_specs[join_index]["aliases"])

    first_alias = scans[0].alias
    input_document_rows = sum(
        len(aliases[scan_node.alias]["ids"]) for scan_node in scans)
    held_alias = (join_stages[0]["anchor"] if join_stages else first_alias)
    anchor = None
    both = {}
    if len(joins) == 1:
        stage_aliases = [arg.alias for arg in joins[0].predicate.args]
        anchor = ("left" if join_stages[0]["anchor"] == stage_aliases[0]
                  else "right")
        both = {
            "left": single_join_options[stage_aliases[0]],
            "right": single_join_options[stage_aliases[1]],
        }
    elif joins:
        anchor = ",".join(stage["anchor"] for stage in join_stages)

    return {
        "work": work,
        "document_column": aliases[first_alias]["column"],
        "partner_column": (aliases[scans[1].alias]["column"]
                           if len(scans) > 1 else None),
        "documents": len(aliases[first_alias]["ids"]),
        "documents_after_filters": post_filter_counts[first_alias],
        "input_document_rows": input_document_rows,
        "filter_evaluations": filter_evaluations,
        "join_pair_evaluations": join_pair_evaluations,
        "filter_stages": filter_stages,
        "join_stages": join_stages,
        "anchor": anchor,
        "anchor_tokens_both_ways": both,
        "held_column": aliases[held_alias]["column"],
        "runtime_search": (None if not search_runs else dict(
            states=sum(run["states"] for run in search_runs),
            generated=sum(run["generated"] for run in search_runs),
            replans=len(search_runs),
            sequence=[list(step) for step in search_sequence])),
    }


def simulate_optimal_left_deep(query, model: ModelSpec, chunk_tokens: int):
    """Find the best exact left deep join plan for this model."""

    prepared = prepare_query(query, model, chunk_tokens)
    scans = prepared.scans
    joins = prepared.joins
    aliases = prepared.aliases
    base_rows = {
        alias: tuple(rows) for alias, rows in prepared.survivors.items()}
    alias_order = tuple(scan.alias for scan in scans)

    edge_relations = []
    edge_aliases = []
    for edge_index, join in enumerate(joins):
        stage_aliases = tuple(arg.alias for arg in join.predicate.args)
        if join.semantics != "full":
            raise NotImplementedError(
                "optimal SoL join search requires full join semantics")
        if len(stage_aliases) != 2:
            raise NotImplementedError(
                "optimal SoL join search requires binary predicates")
        left, right = stage_aliases
        passing = frozenset(
            (left_row, right_row)
            for left_row in base_rows[left]
            for right_row in base_rows[right]
            if prompt_answer(
                join.predicate,
                {left: left_row, right: right_row},
                aliases,
            )
        )
        edge_relations.append(PairRelation(left, right, passing))
        edge_aliases.append(frozenset((left, right)))

    logical_cache = {}

    def live_for(active_edges: frozenset[int]):
        if active_edges not in logical_cache:
            logical_cache[active_edges] = exact_live_rows(
                base_rows,
                tuple(edge_relations[index]
                      for index in sorted(active_edges)),
            )
        return logical_cache[active_edges]

    def active_inside(relations: frozenset[str]) -> frozenset[int]:
        return frozenset(
            index for index, endpoints in enumerate(edge_aliases)
            if endpoints <= relations
        )

    def anchor_fits(prompt, anchor, stage_aliases, live) -> bool:
        labels_by_alias, tail = prompt_token_counts(prompt)
        anchor_max = max(
            (aliases[anchor]["tokens"][row] for row in live[anchor]),
            default=0,
        )
        partners = [alias for alias in stage_aliases if alias != anchor]
        prefix = (PRE + anchor_max
                  + labels_by_alias[anchor]["frame"])
        if not live[anchor] or any(not live[partner] for partner in partners):
            return prefix <= chunk_tokens
        return (
            prefix
            + tail
            + sum(
                labels_by_alias[partner]["label"]
                + max(
                    (aliases[partner]["tokens"][row]
                     for row in live[partner]),
                    default=0,
                )
                for partner in partners
            )
            <= chunk_tokens
        )

    extension_cache = {}

    def extend(relations: frozenset[str], cached: frozenset[str], added: str):
        cache_key = (relations, cached, added)
        if cache_key in extension_cache:
            return extension_cache[cache_key]
        crossing = tuple(
            index for index, endpoints in enumerate(edge_aliases)
            if added in endpoints and endpoints & relations
        )
        if not crossing:
            extension_cache[cache_key] = ()
            return ()

        initial_edges = active_inside(relations)
        extensions = []

        def visit_edges(order, position, active_edges, live, live_cache,
                        work, steps):
            if position == len(order):
                extensions.append(Extension(
                    work=work,
                    state_property=live_cache,
                    steps=tuple(steps),
                ))
                return

            edge_index = order[position]
            join = joins[edge_index]
            relation = edge_relations[edge_index]
            stage_aliases = tuple(arg.alias for arg in join.predicate.args)
            for anchor in stage_aliases:
                if not anchor_fits(
                        join.predicate, anchor, stage_aliases, live):
                    continue
                partners = [alias for alias in stage_aliases
                            if alias != anchor]
                stage_work = join_stage_work(
                    anchor,
                    partners,
                    aliases,
                    live,
                    join.predicate,
                    live[anchor] if anchor in live_cache else (),
                )
                next_edges = active_edges | {edge_index}
                next_live = live_for(next_edges)
                next_cache = live_cache | {anchor}
                evaluated = math.prod(
                    len(live[alias]) for alias in stage_aliases)
                passing = sum(
                    1
                    for left_row in live[relation.left]
                    for right_row in live[relation.right]
                    if (left_row, right_row) in relation.pairs
                )
                step = {
                    "code": prompt_code(join.predicate),
                    "added_alias": added,
                    "anchor": anchor,
                    "partners": partners,
                    "evaluated_pairs": evaluated,
                    "passing_pairs": passing,
                    "fresh_tokens": stage_work.tokens,
                    "cached_prefixes_after": sorted(next_cache),
                }
                visit_edges(
                    order,
                    position + 1,
                    next_edges,
                    next_live,
                    next_cache,
                    work + stage_work,
                    steps + [step],
                )

        for order in itertools.permutations(crossing):
            visit_edges(
                order,
                0,
                initial_edges,
                live_for(initial_edges),
                cached,
                Work(),
                [],
            )
        extension_cache[cache_key] = tuple(extensions)
        return extension_cache[cache_key]

    search = optimize_left_deep(
        alias_order,
        frozenset(prepared.resident),
        prepared.work,
        extend,
    )
    if not search.candidates:
        raise ValueError("query join graph has no connected left deep plan")

    def candidate_seconds(candidate):
        return speed_of_light(
            candidate.work, model, H100_SXM, chunk_tokens).seconds

    best = min(
        search.candidates,
        key=lambda candidate: (
            candidate_seconds(candidate),
            candidate.work.tokens,
            candidate.work.pairs,
            candidate.work.kv_written,
            candidate.work.kv_read,
            candidate.relation_order,
        ),
    )

    first_alias = scans[0].alias
    join_stages = list(best.steps)
    input_document_rows = sum(
        len(aliases[scan.alias]["ids"]) for scan in scans)
    held_alias = join_stages[0]["anchor"] if join_stages else first_alias
    anchor = None
    both = {}
    if len(joins) == 1:
        stage_aliases = [arg.alias for arg in joins[0].predicate.args]
        anchor = ("left" if join_stages[0]["anchor"] == stage_aliases[0]
                  else "right")
        start_live = live_for(frozenset())
        both = {
            side: join_stage_work(
                candidate,
                [alias for alias in stage_aliases if alias != candidate],
                aliases,
                start_live,
                joins[0].predicate,
                start_live[candidate]
                if candidate in prepared.resident else (),
            ).tokens
            for side, candidate in zip(("left", "right"), stage_aliases)
            if anchor_fits(
                joins[0].predicate, candidate, stage_aliases, start_live)
        }
    elif joins:
        anchor = ",".join(stage["anchor"] for stage in join_stages)

    return {
        "work": best.work,
        "document_column": aliases[first_alias]["column"],
        "partner_column": (aliases[scans[1].alias]["column"]
                           if len(scans) > 1 else None),
        "documents": len(aliases[first_alias]["ids"]),
        "documents_after_filters": prepared.post_filter_counts[first_alias],
        "input_document_rows": input_document_rows,
        "filter_evaluations": prepared.filter_evaluations,
        "join_pair_evaluations": sum(
            stage["evaluated_pairs"] for stage in join_stages),
        "filter_stages": prepared.filter_stages,
        "join_stages": join_stages,
        "anchor": anchor,
        "anchor_tokens_both_ways": both,
        "held_column": aliases[held_alias]["column"],
        "optimizer": {
            "plan_space": "all feasible left deep plans",
            "relation_order": list(best.relation_order),
            "cached_prefixes": sorted(best.state_property),
            "persistent_kv_capacity": "unlimited",
            "gpu_count": 1,
            "dp_states": search.state_count,
            "dp_records_generated": search.generated_count,
            "dp_final_records": len(search.candidates),
        },
    }


# ================================================================
# PART 3: every query, on both models
# ================================================================

query_defs_by_model = {}
for model in MODELS:
    session = quail.Session(
        EngineConfig(gpus=1, model=model.name),
        tokenizer=encode)
    Q.register_sets(session, W / "data" / TAG)
    query_defs_by_model[model.name] = Q.queries(session)

query_ids = list(query_defs_by_model[MODELS[0].name])
if any(list(query_defs_by_model[model.name]) != query_ids for model in MODELS):
    raise ValueError("4B and 32B query definitions do not have the same ids")


def add_sol_metrics(simulated, model: ModelSpec):
    """Add the one H100! time, cost, and throughput to a simulation."""

    simulated = dict(simulated)
    work = simulated.pop("work")
    held = simulated["held_column"]
    s = speed_of_light(work, model, H100_SXM, CHUNK[model.name])
    attn_proj = s.component("attn_proj")
    mlp = s.component("mlp")
    attention = s.component("attention")
    dense_compute = attn_proj.compute_seconds + mlp.compute_seconds
    dense_memory = attn_proj.memory_seconds + mlp.memory_seconds
    dense_seconds = attn_proj.seconds + mlp.seconds
    return {
        **simulated,
        "chunk_tokens": CHUNK[model.name],
        "tuples": simulated["join_pair_evaluations"],
        "tokens": work.tokens,
        "pairs": work.pairs,
        "kv_written": work.kv_written,
        "kv_read": work.kv_read,
        "held_mean_doc_tokens": (
            sum(lengths[held].values()) / len(lengths[held])),
        "passes": s.passes,
        "bytes_moved": s.bytes_moved,
        "components": [
            {
                "name": component.name,
                "precision": component.precision,
                "flops": component.flops,
                "bytes_moved": component.bytes_moved,
                "t_compute": component.compute_seconds,
                "t_memory": component.memory_seconds,
                "seconds": component.seconds,
                "bound_by": component.bound_by,
            }
            for component in s.components
        ],
        "t_dense": dense_compute,
        "t_dense_memory": dense_memory,
        "t_dense_roofline": dense_seconds,
        "t_attention": attention.compute_seconds,
        "t_attention_memory": attention.memory_seconds,
        "t_attention_roofline": attention.seconds,
        "t_compute": s.compute,
        "t_memory": s.memory,
        "sol_s": s.seconds,
        "bound_by": s.bound_by,
        "cost_usd_per_query_at_sol": (
            s.seconds * H100_USD_PER_HOUR / 3600),
        "documents_per_second_at_sol": (
            simulated["input_document_rows"] / s.seconds
            if not simulated["join_stages"] and s.seconds else None),
        "document_pairs_per_second_at_sol": (
            simulated["join_pair_evaluations"] / s.seconds
            if simulated["join_stages"] and s.seconds else None),
    }


rows = {}
query_inputs = {}
# queries over a corpus this estimate does not tokenize (the agent
# traces) are recorded as skipped rather than estimated
skipped = {}
for qid in query_ids:
    descriptions = {
        query_defs_by_model[model.name][qid][0] for model in MODELS}
    if len(descriptions) != 1:
        raise ValueError(f"model query descriptions differ for {qid}")
    probe_scans, _, _ = collect_operators(
        query_defs_by_model[MODELS[0].name][qid][1]().logical)
    missing = sorted(
        {f"{scan.provider}.{scan.column}" for scan in probe_scans}
        - set(lengths))
    if missing:
        skipped[qid] = f"no modeled corpus for {', '.join(missing)}"
        print(f"{qid}: skipped, {skipped[qid]}")
        continue
    rows[qid] = {"description": descriptions.pop(), "models": {}}
    query_inputs[qid] = {}
    for model in MODELS:
        _, build = query_defs_by_model[model.name][qid]
        optimal = add_sol_metrics(
            simulate_optimal_left_deep(
                build(), model, CHUNK[model.name]),
            model,
        )
        rows[qid]["models"][model.name] = optimal
        query_inputs[qid][model.name] = {
            "optimal_left_deep": {
                "filters": optimal["filter_stages"],
                "joins": optimal["join_stages"],
                "optimizer": optimal["optimizer"],
            },
        }

hdr = (f"{'query':7} {'4B tokens':>11} {'4B anchor':>15} {'4B SoL':>9} "
       f"{'32B tokens':>11} {'32B anchor':>15} {'32B SoL':>9} "
       f"{'32B/4B':>7}")
print(hdr)
print("-" * len(hdr))
for qid, r in rows.items():
    a, b = r["models"]["qwen3-4b-fp8"], r["models"]["qwen3-32b-fp8"]
    print(f"{qid:7} {a['tokens']:>11,.0f} {str(a['anchor'] or '-'):>15} "
          f"{a['sol_s']:>9.3f} {b['tokens']:>11,.0f} "
          f"{str(b['anchor'] or '-'):>15} {b['sol_s']:>9.3f} "
          f"{b['sol_s'] / a['sol_s']:>7.2f}")


json.dump({
    "what": f"Speed of light for {len(rows)} QUAIL-B queries at "
            f"sf={SF:g}, on "
            "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100! request each. "
            "Every feasible left deep order and anchor choice is considered. "
            "No measured or fitted constant is used.",
    "method": "plans/sol_model.md, computed by reports/make_sol_quailb.py",
    "scale_factor": SF,
    "query_count": len(rows),
    "skipped": skipped,
    "corpus_id": CORPUS_ID,
    "collection_id": COLLECTION_ID,
    "pricing": {
        "gpu": "H100!",
        "h100_usd_per_hour": H100_USD_PER_HOUR,
        "price_source": H100_PRICE_SOURCE,
        "method": "SoL seconds multiplied by the H100! price per second",
    },
    "metric_definitions": {
        "cost_usd_per_query_at_sol": (
            "GPU cost at SoL; SoL seconds times the H100! price"),
        "documents_per_second_at_sol": (
            "filter only throughput at SoL; input document rows divided "
            "by SoL seconds"),
        "document_pairs_per_second_at_sol": (
            "join throughput at SoL; evaluated pairs summed across join "
            "stages and divided by SoL seconds"),
    },
    "optimizer": {
        "gpu_count_per_model": 1,
        "filter_order": (
            "by_cost from fixed benchmark selectivity estimates"),
        "filter_selectivity_sources": {
            "collection": Q.SELECTIVITY_ESTIMATE_COLLECTION,
            "corpus": Q.SELECTIVITY_ESTIMATE_CORPUS,
            "scale_factor": Q.SELECTIVITY_ESTIMATE_SCALE_FACTOR,
        },
        "plan_space": "all feasible left deep plans",
        "dp_state": "joined alias set and cached prefix alias set",
        "work_frontier": (
            "keep every record not larger in all four work categories"),
        "survivors": "exact ground truth survivors",
        "persistent_kv_capacity": "unlimited",
        "cached_values": "document prefixes used by filters or as anchors",
        "streamed_partner_kv": "not reusable",
        "validation": "unit tests compare DP with complete enumeration",
    },
    "sources": {
        "corpora": f"/results/quailb_data/{TAG}, seed 20260818",
        "labels": "/results/ground_truth/quailb/schema_v1/label_sets on "
                  "quail-results, qwen3-32b-fp8 answering",
        "tokenizer": "Qwen/Qwen3-4B-FP8, shared by every Qwen3 model"},
    "chunk_tokens": CHUNK,
    "measured_inputs": {
        "preamble_tokens": PRE,
        "document_lengths": {
            k: dict(sorted(collections.Counter(v.values()).items()))
            for k, v in lengths.items()},
        "filter_question_tokens": question,
        "join_prompt_tokens": join_prompt,
        "query_stages_by_model": query_inputs},
    "queries": rows,
}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}\n"
      "put it on the volume:\n"
      f"  modal volume put quail-results {OUT} /sol/{OUT.name}")
