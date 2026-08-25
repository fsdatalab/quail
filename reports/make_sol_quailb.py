"""Speed of light for every QUAIL-B query, on 4B and on 32B.

One script, one output: results/sol_quailb_sf0.1.json holds the
measured inputs and the bound computed from them.

It measures exactly three things and computes everything from them:

  1. document lengths   - each corpus tokenized with the Qwen3
                          tokenizer (4B and 32B share it)
  2. prompt lengths     - the shared preamble, each filter's
                          question, each join's label, naming line
                          and question
  3. selectivities      - from the QUAIL-B ground truth labels, per
                          stage and conditional on the stages before

The arithmetic is quail/sol.py; the equations are
plans/sol_model.md; the report is reports/2026-08-25-sol-quailb.md.

Needs two pulls off the quail-results volume first, since the
corpora and the per-document labels are raw data that stays there:

    W=<workdir>
    modal volume get quail-results /quailb_data/sf0.1 $W/data/
    modal volume get quail-results \
        /ground_truth/quailb/schema_v1/label_sets $W/allabels/
    uv run --with transformers --with pyarrow \
        python reports/make_sol_quailb.py $W
"""
import collections
import glob
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from quail.bench import quailb as Q
from quail.logical import (ColumnRef, SHARED_PRE, bind_prompt,
                           join_anchor_note, join_label,
                           render_join_question)
from quail.sol import (Work, cheaper_anchor, filter_chain, seconds,
                       survivors)
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8

W = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")
encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]
length = lambda t: len(encode(t))
PRE = length(SHARED_PRE)   # the engine preamble, 2 tokens

COLUMNS = {
    "reviews.body": ("reviews", "body"),
    "aspects.aspect": ("aspects", "aspect"),
    "reports.report": ("reports", "report"),
    "terms.term": ("terms", "term"),
    "claims.claim": ("claims", "claim"),
    "evidence.text": ("evidence", "text"),
    "citations.destination_context": ("citations", "destination_context"),
    "citations.passage_text": ("citations", "passage_text"),
}

# 1. document lengths -------------------------------------------------
lengths = {}        # column -> {doc id: token count}
for key, (table, col) in COLUMNS.items():
    t = pq.read_table(W / "data" / "sf0.1" / f"{table}.parquet",
                      columns=["id", col])
    lengths[key] = {i: length(x) for i, x in
                    zip(t.column("id").to_pylist(), t.column(col).to_pylist())}
    v = lengths[key].values()
    print(f"{key:32} {len(v):>6} docs  {sum(v):>10,} tokens  "
          f"mean {sum(v) / len(v):>8.1f}", flush=True)

# 2. prompt lengths ---------------------------------------------------
FILTER_TEMPLATES = {c: getattr(Q, c) for c in (
    "F1", "F4", "F5", "F7", "F8", "F9", "F11", "F12", "F13",
    "LEP1", "LEP2", "LEP3", "LEP4", "LEP5", "LEPS1")}
JOIN_TEMPLATES = {"DISCUSS_ASPECT": Q.DISCUSS_ASPECT, "REACTION": Q.REACTION,
                  "SUPPORT": Q.SUPPORT, "LEPJOIN": Q.LEPJOIN}
col_ref = (ColumnRef("x", "t", "c"),)
question = {c: bind_prompt(t, col_ref, encode).tail_tokens
            for c, t in FILTER_TEMPLATES.items()}
join_prompt = {c: {"question": length(render_join_question(t)),
                   "partner_label": length(join_label(1)),
                   "anchor_note": length(join_anchor_note(0))}
               for c, t in JOIN_TEMPLATES.items()}

# 3. labels -----------------------------------------------------------
labels = {}
for m in glob.glob(str(W / "allabels/label_sets/*/*/*/manifest.json")):
    meta = json.load(open(m))["predicate"]
    if meta["kind"] != "filter":
        continue
    rows = pq.read_table(Path(m).parent / "labels.parquet",
                         columns=["left_id", "answer"]).to_pylist()
    labels[meta["legacy_code"]] = {r["left_id"]: r["answer"] for r in rows}

# (document column, filter codes, (partner column, partner filters, join))
QUERIES = {
 "IMDB-1": ("reviews.body", ["F1"], None),
 "IMDB-2": ("reviews.body", [], ("aspects.aspect", [], "DISCUSS_ASPECT")),
 "IMDB-3": ("reviews.body", ["F1"], ("aspects.aspect", [], "DISCUSS_ASPECT")),
 "IMDB-4": ("reviews.body", ["F1", "F4"],
            ("aspects.aspect", [], "DISCUSS_ASPECT")),
 "IMDB-5": ("reviews.body", ["F1", "F4", "F5"],
            ("aspects.aspect", [], "DISCUSS_ASPECT")),
 "IMDB-6": ("reviews.body", ["F1", "F4"], None),
 "IMDB-7": ("reviews.body", ["F1", "F4", "F5"], None),
 "BIO-1": ("reports.report", ["F7"], None),
 "BIO-2": ("reports.report", [], ("terms.term", [], "REACTION")),
 "BIO-3": ("reports.report", ["F7"], ("terms.term", [], "REACTION")),
 "BIO-4": ("reports.report", ["F7", "F8"], ("terms.term", [], "REACTION")),
 "BIO-5": ("reports.report", ["F7", "F8", "F9"],
           ("terms.term", [], "REACTION")),
 "FEV-1": ("claims.claim", ["F11"], None),
 "FEV-2": ("claims.claim", [], ("evidence.text", [], "SUPPORT")),
 "FEV-3": ("claims.claim", ["F11"], ("evidence.text", [], "SUPPORT")),
 "FEV-4": ("claims.claim", ["F11", "F12"], ("evidence.text", [], "SUPPORT")),
 "FEV-5": ("claims.claim", ["F11"], ("evidence.text", ["F13"], "SUPPORT")),
 "FEV-6": ("claims.claim", ["F11", "F12"],
           ("evidence.text", ["F13"], "SUPPORT")),
 "LEP-1": ("citations.destination_context", ["LEP1"], None),
 "LEP-2": ("citations.destination_context", [],
           ("citations.passage_text", [], "LEPJOIN")),
 "LEP-3": ("citations.destination_context", ["LEP1"],
           ("citations.passage_text", [], "LEPJOIN")),
 "LEP-4": ("citations.destination_context", ["LEP1", "LEP2"],
           ("citations.passage_text", [], "LEPJOIN")),
 "LEP-5": ("citations.destination_context", ["LEP1", "LEP2", "LEP3"],
           ("citations.passage_text", [], "LEPJOIN")),
 "LEP-6": ("citations.destination_context",
           ["LEP1", "LEP2", "LEP3", "LEP4", "LEP5"],
           ("citations.passage_text", [], "LEPJOIN")),
 "LEP-7": ("citations.destination_context", ["LEP1", "LEP2"],
           ("citations.passage_text", ["LEPS1"], "LEPJOIN")),
 "LEP-8": ("citations.destination_context",
           ["LEP1", "LEP2", "LEP3", "LEP4", "LEP5"], None),
}


def stage_selectivities(column, codes):
    """Each stage's selectivity conditional on the stages before it:
    how many of the documents that reach this filter pass it."""
    live = set(lengths[column])
    out = []
    for c in codes:
        passed = {i for i in live if labels[c][i]}
        out.append({"code": c, "question_tokens": question[c],
                    "evaluated": len(live),
                    "selectivity": round(len(passed) / len(live), 6)
                                   if live else 0.0})
        live = passed
    return out


queries = {}
for qid, (column, codes, join) in QUERIES.items():
    rec = {"document_column": column, "filters": stage_selectivities(column,
                                                                     codes)}
    if join:
        partner_column, partner_codes, join_code = join
        rec["partner_column"] = partner_column
        rec["partner_filters"] = stage_selectivities(partner_column,
                                                     partner_codes)
        rec["join"] = {"code": join_code, **join_prompt[join_code]}
    queries[qid] = rec


# The batch size each model's forward pass runs at: (2^31 - 1) over
# the widest projection, the fused kernels' 32-bit offset limit. An
# input to the bound, not something it derives.
CHUNK = {"qwen3-4b-fp8": 110_376, "qwen3-32b-fp8": 41_943}
MODELS = [QWEN3_4B_FP8, QWEN3_32B_FP8]


def doc_lengths(column):
    """One token count per document in that column."""
    return list(lengths[column].values())


def after(lengths, stages):
    """The documents left once every stage in `stages` has run."""
    for st in stages:
        lengths = survivors(lengths, st["selectivity"])
    return lengths


def query_work(rec):
    """One query: its filter chain, its partner's filter chain if it
    has one, and its join."""
    docs = doc_lengths(rec["document_column"])
    stages = rec["filters"]
    work = filter_chain(docs, PRE, [s["question_tokens"] for s in stages],
                        [s["selectivity"] for s in stages])
    if "join" not in rec:
        return work, None, {}
    partners = doc_lengths(rec["partner_column"])
    pstages = rec["partner_filters"]
    work = work + filter_chain(
        partners, PRE, [s["question_tokens"] for s in pstages],
        [s["selectivity"] for s in pstages])
    j = rec["join"]
    jwork, anchor, both = cheaper_anchor(
        after(docs, stages), after(partners, pstages),
        preamble=PRE, note=j["anchor_note"], label=j["partner_label"],
        question=j["question"],
        left_resident=bool(stages), right_resident=bool(pstages))
    return work + jwork, anchor, both


rows = {}
for qid, rec in queries.items():
    work, anchor, both = query_work(rec)
    docs = doc_lengths(rec["document_column"])
    live = after(docs, rec["filters"])
    rows[qid] = {
        "documents": len(docs),
        "documents_after_filters": len(live),
        "tuples": (len(live) * len(after(doc_lengths(rec["partner_column"]),
                                         rec["partner_filters"]))
                   if "join" in rec else 0),
        "anchor": anchor, "anchor_tokens_both_ways": both,
        "tokens": work.tokens, "pairs": work.pairs,
        "kv_written": work.kv_written, "kv_read": work.kv_read,
        "models": {}}
    for model in MODELS:
        s = seconds(work, model, H100_SXM, CHUNK[model.name])
        rows[qid]["models"][model.name] = {
            "passes": s.passes, "bytes_moved": s.bytes_moved,
            "t_dense": s.dense, "t_attention": s.attention,
            "t_compute": s.compute, "t_memory": s.memory,
            "sol_s": s.sol, "bound_by": s.bound_by}

hdr = (f"{'query':7} {'tokens':>11} {'pairs':>15} {'tuples':>8} "
       f"{'anchor':>7}  {'4B SoL':>9} {'att%':>5}  {'32B SoL':>9} "
       f"{'att%':>5} {'32B/4B':>7}")
print(hdr)
print("-" * len(hdr))
for qid, r in rows.items():
    a, b = r["models"]["qwen3-4b-fp8"], r["models"]["qwen3-32b-fp8"]
    print(f"{qid:7} {r['tokens']:>11,.0f} {r['pairs']:>15,.0f} "
          f"{r['tuples']:>8,} {str(r['anchor'] or '-'):>7}  "
          f"{a['sol_s']:>9.3f} {100 * a['t_attention'] / a['t_compute']:>5.1f}"
          f"  {b['sol_s']:>9.3f} "
          f"{100 * b['t_attention'] / b['t_compute']:>5.1f} "
          f"{b['sol_s'] / a['sol_s']:>7.2f}")


json.dump({
    "what": "Speed of light for all 26 QUAIL-B queries at sf=0.1, on "
            "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100 each. A floor on "
            "wall time: no measured or fitted constant is used.",
    "method": "plans/sol_model.md, computed by quail/sol.py",
    "scale_factor": 0.1,
    "sources": {
        "corpora": "/quailb_data/sf0.1 on quail-results, seed 20260818",
        "labels": "/ground_truth/quailb/schema_v1/label_sets on "
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
        "selectivities": {qid: {
            "filters": rec["filters"],
            "partner_filters": rec.get("partner_filters", [])}
            for qid, rec in queries.items()}},
    "queries": rows,
}, open(ROOT / "results" / "sol_quailb_sf0.1.json", "w"), indent=1)
print("\nwrote results/sol_quailb_sf0.1.json")
