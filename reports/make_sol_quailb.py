r"""Ideal SoL estimate for every QUAIL-B query.

The estimate is Quail's own, `quail.speed_of_light_estimate`. It prices
three model components on one H100! request:

  1. attention projections, with their weight bytes and fp8 FLOPs
  2. the MLP, with its weight bytes and fp8 FLOPs
  3. attention, with its KV bytes and bf16 pair FLOPs

No measured or fitted performance constant appears in the calculation. The
estimate omits runtime overhead and uses ideal query-wide packing and unlimited
document prefix KV. Its join search is restricted to eager binary full left
deep plans. It is therefore an optimistic comparison point for that modeled
execution, not the exact minimum for every possible execution. The dollar
metric uses Modal's published H100! price.

The equations are in docs/content/docs/architecture/sol-model.mdx. This
script only supplies what the benchmark has: the corpora, the queries,
and the saved labels that give the exact survivors at every stage. The
survivors, the work counting, the left deep search, and the component
pricing are the estimator's.

Two estimates come out of one run. `per_document` computes each alias's
document once and reuses it only across that document's own questions.
The main estimate also credits a prefix another document already
computed: with unlimited KV every distinct prefix in the corpus is
computed once.

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
        $G/collections/gt_8d0030fc5f187e81480fb859d7a2cd69/manifest.json \
        $W/collection_manifest.json
    uv run --with transformers --with pyarrow \
        python reports/make_sol_quailb.py $W $SF --queries BIO-1,BIO-2,BIO-3
    modal volume put quail-results $W/sol_quailb_sf${SF}_BIO-1_BIO-2_BIO-3.json \
        /sol/sol_quailb_sf${SF}_BIO-1_BIO-2_BIO-3.json

The scale factor defaults to 0.1. The run rejects references made with a
different prompt format or scale factor.

The saved estimates must be regenerated when a query definition changes.
Use Quail revision 370c81fcad30ef9906e3684ad2e667c830bf29a4.
Pass --queries FEV-9 to recalculate only that query.
On a mounted results volume, --root /results reads the saved benchmark
directly and --collection selects its reference labels. The output filename then
includes the selected query IDs so the full suite file is not overwritten.
"""

import argparse
import collections
import hashlib
import json
from functools import cache
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

import quail
from quail.bench.quailb import (
    answer_oracle,
    canonical_templates,
    queries,
)
from quail.planner.plan import EngineConfig
from quail.planner.prefixes import shared_prefix_tokens
from quail.specs import (
    H100_PRICE_SOURCE,
    H100_USD_PER_HOUR,
    QWEN3_4B_FP8,
    QWEN3_32B_FP8,
)
from quail_b import data, prompts
from quail_b.benchmark import load_benchmark, select_queries
from quail_b.labels import GroundTruthCollection, PredicateLabels
from quail_b.predicates import PREDICATE_BY_KEY, predicate_payload
from quail_b.queries import (
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    get_query,
)
from quail_b.rendering import PROMPT_FORMAT, SHARED_PRE

if quail.SHARED_PRE != SHARED_PRE:
    raise SystemExit("Use the Quail revision documented in this script.")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("workdir")
parser.add_argument("scale_factor", nargs="?", type=float, default=0.1)
parser.add_argument("--queries", help="Comma-separated query IDs")
parser.add_argument("--root", help="Saved benchmark root, such as /results")
parser.add_argument("--collection", help="Reference collection id")
args = parser.parse_args()
W = Path(args.workdir)
W.mkdir(parents=True, exist_ok=True)
SF = args.scale_factor
# "%g" so 0.1 stays "0.1" and 0.01 stays "0.01", matching the volume's
# own directory names
TAG = f"sf{SF:g}"
selection = "_" + args.queries.replace(",", "_") if args.queries else ""
OUT = W / f"sol_quailb_{TAG}{selection}.json"
MODELS = [QWEN3_4B_FP8, QWEN3_32B_FP8]

# check the workdir holds this scale factor before tokenizing anything:
# both of these otherwise surface much later as a missing parquet file
if not args.root and not (W / "data" / TAG).is_dir():
    raise SystemExit(f"no corpus at {W / 'data' / TAG}: pull "
                     f"/quailb_data/{TAG} off the volume")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")


@cache
def encode(text):
    return tuple(tok(text, add_special_tokens=False)["input_ids"])


# ================================================================
# PART 1: the labels, as the exact answer to every prompt
# ================================================================

def load_collection(workdir: Path, collection: dict) -> GroundTruthCollection:
    """Read the pulled label sets of one collection."""
    active = dict(collection["label_sets"])
    predicates = {}
    for manifest_path in (workdir / "allabels" / "label_sets").glob(
            "*/*/*/manifest.json"):
        label_set_id = manifest_path.parent.name
        if label_set_id not in active.values():
            continue
        manifest = json.load(open(manifest_path))
        rows = pq.read_table(
            manifest_path.parent / "labels.parquet",
            columns=["left_id", "right_id", "answer"]).to_pylist()
        key = manifest["predicate"]["key"]
        predicates[key] = PredicateLabels(
            key=key,
            label_set_id=label_set_id,
            predicate=manifest["predicate"],
            answers={
                (str(row["left_id"]),
                 None if row["right_id"] is None else str(row["right_id"])):
                bool(row["answer"])
                for row in rows},
            source_rows=manifest.get("source_rows", {}),
            predicate_payload=manifest.get("predicate_payload"),
        )
    if len(predicates) != len(active):
        raise ValueError(
            f"loaded {len(predicates)} active predicates, expected "
            f"{len(active)}")
    return GroundTruthCollection(
        collection_id=collection["collection_id"],
        corpus_id=collection["corpus_id"],
        scale_factor=float(collection["scale_factor"]),
        reference_model=collection.get("summary", {}).get("model"),
        predicates=predicates,
    )


if args.root:
    benchmark = load_benchmark(
        args.queries.split(",") if args.queries else None, scale_factor=SF,
        root=args.root, collection_id=args.collection)
    truth = benchmark.ground_truth
    corpus_rows = benchmark.tables
else:
    collection = json.loads((W / "collection_manifest.json").read_text())
    if collection["scale_factor"] != SF:
        raise ValueError("the reference collection has a different scale factor")
    truth = load_collection(W, collection)
    corpus_rows = data.read_corpus(W / "data" / TAG)
    corpus = data.corpus_identity(
        corpus_rows, SF, data.DATA_SEED, data.SOURCE_REVISIONS)
    if corpus["corpus_id"] != truth.corpus_id:
        raise ValueError("the corpus and reference labels have different ids")
    selected = select_queries(
        args.queries.split(",") if args.queries else None, scale_factor=SF)
    for query in selected:
        for operator in query._info.operators:
            labels = truth.predicates[truth.key_for_template(operator.prompt)]
            spec = PREDICATE_BY_KEY[labels.key]
            if ("qwen3_32b" in spec.source_policy
                    and labels.predicate_payload != predicate_payload(spec)):
                raise ValueError(f"{spec.key}: reference prompt format differs")
answer = answer_oracle(truth, corpus_rows)

# ================================================================
# PART 2: the queries, on one session with the shared Qwen3 tokenizer
# ================================================================

session = quail.Session(
    EngineConfig(gpus=1, model=QWEN3_4B_FP8.name, device="h100-sxm"),
    tokenizer=encode)
for name, table in corpus_rows.items():
    session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
query_defs = queries(session)
query_ids = list(query_defs)
if args.queries:
    selected = args.queries.split(",")
    if set(selected) - set(query_ids):
        raise ValueError(f"unknown queries: {set(selected) - set(query_ids)}")
    query_ids = [query for query in query_ids if query in selected]

# 2. prompt lengths, for the record
FILTER_TEMPLATES = {c: getattr(prompts, c) for c in (
    "F1", "F4", "F5", "SERIOUS_ADVERSE_EVENT", "F11", "F12", "F13",
    "LEP1", "LEP2", "LEP3", "LEP4", "LEP5", "LEPS1")}
JOIN_TEMPLATES = {c: getattr(prompts, c) for c in (
    "DISCUSS_ASPECT", "ASPECT_SENTIMENT", "REACTION",
    "SUPPORT", "REFUTE", "LEPJOIN")}
col_ref = (quail.ColumnRef("x", "t", "c"),)
question = {c: quail.bind_prompt(t, col_ref, encode).tail_tokens
            for c, t in FILTER_TEMPLATES.items()}
join_refs = (quail.ColumnRef("left", "left_table", "text"),
             quail.ColumnRef("right", "right_table", "text"))
join_prompt = {}
for code, template in JOIN_TEMPLATES.items():
    prompt = quail.bind_join_prompt(template, join_refs, encode)
    join_prompt[code] = {
        "left_frame": prompt.labels[0][2],
        "right_frame": prompt.labels[1][2],
        "left_label": prompt.labels[0][1],
        "right_label": prompt.labels[1][1],
        "tail": prompt.tail_tokens,
    }
PRE = len(encode(quail.SHARED_PRE))


# ================================================================
# PART 3: every query, on both models
# ================================================================

stores = {}     # "table.column" -> the session's token store


def with_codes(stages, truth: GroundTruthCollection) -> list[dict]:
    """Name each stage's predicate by its ground truth key.

    A stage records the template as Quail binds it; the labels are
    keyed by the template as written.
    """
    written = canonical_templates(truth)
    return [
        {**stage, "code": truth.key_for_template(
            written.get(stage["template"], stage["template"]))}
        for stage in stages
    ]


def record(estimate: quail.SpeedOfLightEstimate) -> dict:
    """Return one estimate in the saved file's layout."""
    data = estimate.as_dict()
    latency = estimate.latency
    attn_proj = latency.component("attn_proj")
    mlp = latency.component("mlp")
    attention = latency.component("attention")
    scans = list(estimate.alias_columns)
    first_alias = scans[0]
    join_stages = with_codes(estimate.join_stages, truth)
    held_alias = join_stages[0]["anchor"] if join_stages else first_alias
    anchor = None
    if len(join_stages) == 1:
        stage = join_stages[0]
        anchor = "left" if stage["anchor"] == stage["aliases"][0] else "right"
    elif join_stages:
        anchor = ",".join(stage["anchor"] for stage in join_stages)
    held_tokens = stores[estimate.alias_columns[held_alias]].lengths
    return {
        "chunk_tokens": estimate.chunk_tokens,
        "document_column": estimate.alias_columns[first_alias],
        "partner_column": (estimate.alias_columns[scans[1]]
                           if len(scans) > 1 else None),
        "documents": estimate.documents_by_alias[first_alias],
        "documents_after_filters": estimate.post_filter_counts[first_alias],
        "input_document_rows": estimate.input_document_rows,
        "filter_evaluations": estimate.filter_evaluations,
        "join_pair_evaluations": estimate.join_pair_evaluations,
        "tuples": estimate.join_pair_evaluations,
        "filter_stages": with_codes(estimate.filter_stages, truth),
        "join_stages": join_stages,
        "anchor": anchor,
        "held_column": estimate.alias_columns[held_alias],
        "held_mean_doc_tokens": sum(held_tokens) / len(held_tokens),
        "tokens": data["tokens"],
        "pairs": data["pairs"],
        "kv_written": data["kv_written"],
        "kv_read": data["kv_read"],
        "passes": data["passes"],
        "bytes_moved": data["bytes_moved"],
        "components": data["components"],
        "t_dense": attn_proj.compute_seconds + mlp.compute_seconds,
        "t_dense_memory": attn_proj.memory_seconds + mlp.memory_seconds,
        "t_dense_roofline": attn_proj.seconds + mlp.seconds,
        "t_attention": attention.compute_seconds,
        "t_attention_memory": attention.memory_seconds,
        "t_attention_roofline": attention.seconds,
        "t_compute": data["t_compute"],
        "t_memory": data["t_memory"],
        "sol_s": data["sol_s"],
        "bound_by": data["bound_by"],
        "cost_usd_per_query_at_sol": estimate.usd_per_query,
        "documents_per_second_at_sol": (
            estimate.input_document_rows / estimate.seconds
            if not join_stages and estimate.seconds else None),
        "document_pairs_per_second_at_sol": (
            estimate.join_pair_evaluations / estimate.seconds
            if join_stages and estimate.seconds else None),
        "optimizer": {
            "plan_space": data["assumptions"]["plan_space"],
            "relation_order": data["relation_order"],
            "cached_prefixes": data["cached_prefixes"],
            "persistent_kv_capacity": (
                data["assumptions"]["persistent_kv_capacity"]),
            "gpu_count": data["assumptions"]["gpu_count"],
            **data["search"],
        },
    }


rows = {}
query_inputs = {}
assumptions = {}
for qid in query_ids:
    description, build = query_defs[qid]
    probe = build()
    scans = probe.logical.operators().scans
    for scan in scans:
        stores.setdefault(
            f"{scan.provider}.{scan.column}", probe.token_inputs()[scan.alias])
    rows[qid] = {
        "description": description,
        "plan_sha256": hashlib.sha256(get_query(qid).plan_bytes).hexdigest(),
        "prompt_format": PROMPT_FORMAT,
        "alias_columns": {
            scan.alias: f"{scan.provider}.{scan.column}" for scan in scans},
        "models": {},
    }
    query_inputs[qid] = {}
    for model in MODELS:
        per_document = record(quail.speed_of_light_estimate(
            build(), answer, model=model,
            credit_shared_prefixes=False))
        estimate = quail.speed_of_light_estimate(
            build(), answer, model=model)
        assumptions.setdefault(model.name, estimate.assumptions())
        optimal = record(estimate)
        optimal["shared_prefix_tokens_credited"] = (
            per_document["tokens"] - optimal["tokens"])
        optimal["per_document"] = {
            key: per_document[key]
            for key in ("sol_s", "tokens", "pairs", "kv_written",
                        "kv_read", "passes", "bound_by",
                        "cost_usd_per_query_at_sol",
                        "documents_per_second_at_sol",
                        "document_pairs_per_second_at_sol",
                        "t_compute", "t_memory", "anchor")
        }
        rows[qid]["models"][model.name] = optimal
        query_inputs[qid][model.name] = {
            "optimal_left_deep": {
                "filters": optimal["filter_stages"],
                "joins": optimal["join_stages"],
                "optimizer": optimal["optimizer"],
            },
        }

hdr = (f"{'query':7} {'4B tokens':>11} {'4B anchor':>15} {'4B SoL':>9} "
       f"{'4B per-doc':>10} {'32B SoL':>9} {'32B per-doc':>11} {'32B/4B':>7}")
print(hdr)
print("-" * len(hdr))
for qid, row in rows.items():
    a = row["models"]["qwen3-4b-fp8"]
    b = row["models"]["qwen3-32b-fp8"]
    print(f"{qid:7} {a['tokens']:>11,.0f} {str(a['anchor'] or '-'):>15} "
          f"{a['sol_s']:>9.3f} {a['per_document']['sol_s']:>10.3f} "
          f"{b['sol_s']:>9.3f} {b['per_document']['sol_s']:>11.3f} "
          f"{b['sol_s'] / a['sol_s']:>7.2f}")

for key, store in stores.items():
    lengths = list(store.lengths)
    credited = shared_prefix_tokens(store)
    print(f"{key:32} {len(lengths):>6} docs  {sum(lengths):>10,} tokens  "
          f"mean {sum(lengths) / len(lengths):>8.1f}  shared prefix "
          f"{credited:>10,} ({credited / sum(lengths):.1%})", flush=True)

json.dump({
    "what": f"SoL for {len(rows)} QUAIL-B queries at "
            f"sf={SF:g}, on "
            "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100! request each. "
            "Every feasible left deep order and anchor choice is considered. "
            "No measured or fitted constant is used.",
    "method": "docs/content/docs/architecture/sol-model.mdx, "
              "quail.speed_of_light_estimate called by "
              "reports/make_sol_quailb.py",
    "scale_factor": SF,
    "query_count": len(rows),
    "corpora": {
        key: {
            "documents": len(store.lengths),
            "tokens": sum(store.lengths),
            "shared_prefix_tokens": shared_prefix_tokens(store),
        }
        for key, store in stores.items()
    },
    "estimates": {
        "sol_s": "each distinct token prefix in the query's scanned "
                 "documents computed once, across documents and across "
                 "aliases of one column",
        "per_document.sol_s": "each alias's document computed once, "
                              "reused only across its own questions",
    },
    "corpus_id": truth.corpus_id,
    "collection_id": truth.collection_id,
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
            "collection": SELECTIVITY_ESTIMATE_COLLECTION,
            "corpus": SELECTIVITY_ESTIMATE_CORPUS,
            "scale_factor": SELECTIVITY_ESTIMATE_SCALE_FACTOR,
        },
        "plan_space": "all feasible left deep plans",
        "dp_state": "joined alias set and cached prefix alias set",
        "work_frontier": (
            "keep every record not larger in all four work categories"),
        "survivors": "exact ground truth survivors",
        "persistent_kv_capacity": "unlimited",
        "cached_values": "document prefixes used by filters or as anchors",
        "cross_alias_prefix_reuse": (
            "in the distinct prefix estimate an anchor row whose column "
            "another alias filtered, or whose row is live under another "
            "cached alias of the column, pays the frame only"),
        "streamed_partner_kv": "not reusable",
        "validation": "unit tests compare DP with complete enumeration",
        "estimator_assumptions": assumptions,
    },
    "sources": {
        "corpora": f"/results/quailb_data/{TAG}, seed {data.DATA_SEED}",
        "labels": "/results/ground_truth/quailb/schema_v1/label_sets on "
                  "quail-results, qwen3-32b-fp8 answering",
        "tokenizer": "Qwen/Qwen3-4B-FP8, shared by every Qwen3 model"},
    "chunk_tokens": {
        model.name: rows[query_ids[0]]["models"][model.name]["chunk_tokens"]
        for model in MODELS},
    "measured_inputs": {
        "preamble_tokens": PRE,
        "document_lengths": {
            key: dict(sorted(collections.Counter(
                int(length) for length in store.lengths).items()))
            for key, store in stores.items()},
        "filter_question_tokens": question,
        "join_prompt_tokens": join_prompt,
        "query_stages_by_model": query_inputs},
    "queries": rows,
}, open(OUT, "w"), indent=1)
session.close()
print(f"\nwrote {OUT}\n"
      "put it on the volume:\n"
      f"  modal volume put quail-results {OUT} /sol/{OUT.name}")
