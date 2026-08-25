"""Speed of light for every QUAIL-B query at sf=0.1, on 4B and 32B.

Writes results/sol_quailb_sf0.1.json. The report is
reports/2026-08-25-sol-quailb.md; the method is plans/sol_model.md.

Needs two pulls off the quail-results volume first, into <workdir>:

    modal volume get quail-results /quailb_data/sf0.1 <workdir>/data/
    modal volume get quail-results \
        /ground_truth/quailb/schema_v1/label_sets <workdir>/allabels/
    modal volume get quail-results sol_check_sf0.1_4b.json \
        <workdir>/sol_check.json

then:

    uv run --with transformers --with pyarrow \
        python reports/make_sol_quailb.py <workdir>

Corpora and labels are raw experiment data and stay on the volume.
Only the aggregates this writes are committed.
"""
import glob
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from quail.logical import (ColumnRef, SHARED_PRE, bind_prompt,
                           join_anchor_note, join_label,
                           render_join_question)
from quail.bench import quailb as Q
from quail.sol import (Corpus, FilterStage, JoinSide, JoinStage, Workload,
                       bound, filter_chain_workload, join_workload)
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8

S = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
from quail.logical import (bind_prompt, ColumnRef, SHARED_PRE, join_label,
                           join_anchor_note, render_join_question)
from quail.bench import quailb as Q

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")
enc = lambda t: tk(t, add_special_tokens=False)["input_ids"]
tlen = lambda t: len(enc(t))

# --- corpora -------------------------------------------------------
COLS = {
    "reviews.body": ("reviews", "id", "body"),
    "aspects.aspect": ("aspects", "id", "aspect"),
    "reports.report": ("reports", "id", "report"),
    "terms.term": ("terms", "id", "term"),
    "claims.claim": ("claims", "id", "claim"),
    "evidence.text": ("evidence", "id", "text"),
    "citations.destination_context": ("citations", "id",
                                      "destination_context"),
    "citations.passage_text": ("citations", "id", "passage_text"),
}
lens = {}       # "table.col" -> {id: token count}, source order kept
for key, (tbl, idc, col) in COLS.items():
    t = pq.read_table(S / "data" / "sf0.1" / f"{tbl}.parquet",
                      columns=[idc, col])
    ids = t.column(idc).to_pylist()
    txt = t.column(col).to_pylist()
    lens[key] = dict(zip(ids, (tlen(x) for x in txt)))
    print(f"{key:32} {len(ids):>6} rows, "
          f"{sum(lens[key].values()):>10,} tokens", flush=True)

PRE = tlen(SHARED_PRE)

# --- prompts -------------------------------------------------------
FILTERS = {c: getattr(Q, c) for c in
           ("F1", "F4", "F5", "F7", "F8", "F9", "F11", "F12", "F13",
            "LEP1", "LEP2", "LEP3", "LEP4", "LEP5", "LEPS1")}
JOINS = {"DISCUSS_ASPECT": Q.DISCUSS_ASPECT, "REACTION": Q.REACTION,
         "SUPPORT": Q.SUPPORT, "LEPJOIN": Q.LEPJOIN}
one = (ColumnRef("x", "t", "c"),)
two = (ColumnRef("x", "t", "c"), ColumnRef("y", "u", "d"))
prompts = {c: bind_prompt(t, one, enc).tail_tokens
           for c, t in FILTERS.items()}
joins = {c: {"tail": tlen(render_join_question(t)),
             "partner_label": tlen(join_label(1)),
             "anchor_note": tlen(join_anchor_note(0))}
         for c, t in JOINS.items()}

# --- labels --------------------------------------------------------
labels = {}     # legacy code -> {left_id: bool} for filters
for m in glob.glob(str(S / "allabels/label_sets/*/*/*/manifest.json")):
    d = json.load(open(m))
    p = d["predicate"]
    if p["kind"] != "filter":
        continue
    rows = pq.read_table(Path(m).parent / "labels.parquet",
                         columns=["left_id", "answer"]).to_pylist()
    labels[p["legacy_code"]] = {r["left_id"]: r["answer"] for r in rows}
    print(f"labels {p['legacy_code']:>6} {len(rows):>6}", flush=True)


def moments(col, ids, add=0):
    v = [lens[col][i] + add for i in ids]
    return {"n": len(v), "total": sum(v),
            "total_sq": sum(x * x for x in v)}


def chain(col, codes):
    """Per stage: the live set entering it, counted, not scaled."""
    live = list(lens[col])
    out = []
    for c in codes:
        out.append({"code": c, "question_tokens": prompts[c],
                    "live": moments(col, live, PRE)})
        live = [i for i in live if labels[c][i]]
    return out, live


# --- per-query stage inputs ---------------------------------------
# (doc column, filter codes, [(partner column, partner filters, join code)])
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

out = {}
for qid, (col, codes, join) in QUERIES.items():
    stages, live = chain(col, codes)
    rec = {"doc_column": col, "filters": stages,
           "survivors_after_filters": len(live)}
    if join:
        pcol, pfilters, jcode = join
        pstages, plive = chain(pcol, pfilters)
        jt = joins[jcode]
        rec["partner_filters"] = pstages
        # Raw survivor moments for BOTH sides. Either side can anchor:
        # the planner keeps the KV of whichever side is cheaper to
        # hold and streams the other. A side that already ran a filter
        # has its prefixes resident, so anchoring there computes only
        # the naming line.
        rec["join"] = {
            "code": jcode, "tokens": jt,
            "left": {"column": col, "moments": moments(col, live),
                     "had_filters": bool(codes)},
            "right": {"column": pcol, "moments": moments(pcol, plive),
                      "had_filters": bool(pfilters)},
            "tuples": len(live) * len(plive),
        }
    out[qid] = rec


# The batch size each model's forward pass runs at: (2^31-1) // the
# widest projection, the fused kernels' 32-bit offset limit.
CHUNK = {"qwen3-4b-fp8": 110_376, "qwen3-32b-fp8": 41_943}
MODELS = [QWEN3_4B_FP8, QWEN3_32B_FP8]


def chain_workload(stages, carry=False):
    """stages: the per-stage live records from the extraction."""
    if not stages:
        return Workload(0.0, 0.0, 0.0)
    c = Corpus(stages[0]["live"]["n"], float(stages[0]["live"]["total"]),
               float(stages[0]["live"]["total_sq"]))
    if c.n_docs == 0:
        return Workload(0.0, 0.0, 0.0)
    fs = []
    for i, st in enumerate(stages):
        nxt = stages[i + 1]["live"] if i + 1 < len(stages) else None
        if nxt is None:
            fs.append(FilterStage(st["question_tokens"]))
        else:
            fs.append(FilterStage(st["question_tokens"],
                                  surviving_docs=nxt["n"],
                                  surviving_prefix=float(nxt["total"])))
    return filter_chain_workload(c, fs, carry_question_kv=carry)


def _side(m, add):
    """Shift a raw length distribution by a constant, keeping both
    moments: (x + a) sums to total + n*a and squares to
    total_sq + 2a*total + n*a^2."""
    n, t, t2 = m["n"], float(m["total"]), float(m["total_sq"])
    return JoinSide(n, t + n * add, t2 + 2 * add * t + n * add * add)


def join_both_ways(j, pre):
    """The two orientations of one join. The planner keeps whichever
    side is cheaper to hold and streams the other, so the bound has to
    make the same choice: anchoring the long side costs one prefix per
    document, anchoring the short side costs one full copy of every
    long document per tuple."""
    tok = j["tokens"]
    out = {}
    for name, anchor, partner in (("left", j["left"], j["right"]),
                                  ("right", j["right"], j["left"])):
        if not anchor["moments"]["n"] or not partner["moments"]["n"]:
            out[name] = Workload(0.0, 0.0, 0.0)
            continue
        out[name] = join_workload(JoinStage(
            anchor=_side(anchor["moments"], pre),
            suffix=_side(partner["moments"],
                         tok["partner_label"] + tok["tail"]),
            note_tokens=tok["anchor_note"],
            # a side that ran a filter already holds its prefixes
            opens_anchor=not anchor["had_filters"]))
    return out


def query_workload(rec, pre, anchor=None):
    w = chain_workload(rec["filters"])
    if "join" in rec:
        w = w + chain_workload(rec.get("partner_filters", []))
        both = join_both_ways(rec["join"], pre)
        pick = anchor or min(both, key=lambda k: both[k].tokens)
        w = w + both[pick]
        return w, pick, {k: v.tokens for k, v in both.items()}
    return w, None, {}


rows = {}
for qid, rec in out.items():
    w, anchor, both = query_workload(rec, PRE)
    rows[qid] = {"tokens": w.tokens, "pairs": w.pairs,
                 "kv_read_tokens": w.kv_read_tokens,
                 "docs_in": (rec["filters"][0]["live"]["n"] if rec["filters"]
                             else rec["join"]["left"]["moments"]["n"]),
                 "survivors_after_filters": rec["survivors_after_filters"],
                 "tuples": rec.get("join", {}).get("tuples", 0),
                 "anchor": anchor, "anchor_tokens_both_ways": both,
                 "models": {}}
    for m in MODELS:
        b = bound(m, H100_SXM, w, CHUNK[m.name])
        rows[qid]["models"][m.name] = {
            "passes": b.passes, "bytes_moved": b.bytes_moved,
            "t_dense": b.t_dense, "t_attention": b.t_attention,
            "t_compute": b.t_compute, "t_memory": b.t_memory,
            "sol_s": b.seconds, "bound_by": b.bound_by}


# --- validation against the engine's own recorded token counts -----
chk = json.load(open(S / "sol_check.json"))
val = {}
for q in chk["passes"]["cold"]["queries"]:
    qid = q["query"]
    if qid not in rows:
        continue
    a = rows[qid]["models"]["qwen3-4b-fp8"]["sol_s"]
    val[qid] = {"engine_fresh_tokens": q["fresh_tokens"],
                "sol_tokens": round(rows[qid]["tokens"]),
                "ratio": round(rows[qid]["tokens"] / q["fresh_tokens"], 4),
                "engine_wall_s": q["wall_s"],
                "sol_s_4b": round(a, 4),
                "fraction_of_sol": round(a / q["wall_s"], 4)}

json.dump({
 "what": "Speed-of-light bound for all 26 QUAIL-B queries at sf=0.1, on "
         "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100 each. Aggregates only.",
 "method": "plans/sol_model.md; computed by quail/sol.py",
 "inputs": {
   "corpora": "/quailb_data/sf0.1 on quail-results (seed 20260818)",
   "labels": "/ground_truth/quailb/schema_v1/label_sets on quail-results, "
             "qwen3-32b-fp8. Survivor counts and token masses are counted "
             "per document, never scaled by a selectivity.",
   "tokenizer": "Qwen/Qwen3-4B-FP8, shared by every Qwen3 model",
   "chunk_tokens": CHUNK,
   "preamble_tokens": PRE,
   "filter_question_tokens": prompts,
   "join_prompt_tokens": joins,
   "columns": {k: {"n": len(v), "total": sum(v.values()),
                   "total_sq": sum(x * x for x in v.values())}
               for k, v in lens.items()}},
 "validation": {
   "what": "Token counts against the engine's own fresh_tokens from "
           "/results/sol_check_sf0.1_4b.json on quail-results. IMDB-1, "
           "IMDB-2 and BIO-2 do not depend on any model's answers, so "
           "they must match exactly. IMDB-5 and FEV-5 do: that run was "
           "4B answering, these numbers are the 32B ground truth.",
   "queries": val},
 "queries": rows,
}, open(ROOT / "results" / "sol_quailb_sf0.1.json", "w"), indent=1)
print("wrote results/sol_quailb_sf0.1.json")
