"""QUAIL-B: twenty-six queries over four document sets.

Real filter and join predicates over real, unpadded, un-concatenated
text, rather than planted flags. Selectivity hints are omitted
throughout, so the planner falls back to `as_written` ordering
instead of guessing (`planner/decide.py`).

Build the data and run:

    mkdir -p results/benchmark
    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb.log"
    uv run python -m quail.bench.quailb --sf 0.1 \
        --model qwen3-4b-fp8 --gpus 1 \
        --prediction "State the expected runtime and accuracy here" \
        2>&1 | tee "$run_log"

Omitting `--only` runs all twenty-six queries.

The command reads the matching ground truth from the `quail-results`
Modal volume. It writes the aggregate JSON summary and Markdown report
under `results/benchmark/`. It writes the PNG plot under
`reports/plots/benchmark/`. Raw returned rows and model answers stay on
the Modal volume. The aggregate JSON is also saved on that volume.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_SEED = 20260818
CACHE_SCHEMA_VERSION = 2

# Exact source snapshots for the benchmark corpus.  The row selection below
# is deterministic only when the upstream revisions are fixed as well as the
# sampling seed.
SOURCE_REVISIONS = {
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "BioDEX/BioDEX-Reactions":
        "01a5dacdabd144a120af04931a11a99febd48432",
    # FEVER's parquet files live on its conversion ref, so this is the
    # resolved commit for refs/convert/parquet rather than the default branch.
    "fever/fever": "5f577157472532aa1d9924d2df63aac44f70cf2b",
    "rmahari/LePaRD": "0194f95c3091acceab3b887c9b09ef432cf84052",
}

# Base document counts at sf=1. Only these three scale with sf; the
# partner tables (aspects, terms) are fixed vocabulary and evidence is
# bounded by whichever claims get sampled (see the FEVER open item in
# query-design.md).
SETS = {
    "reviews": 50_000,
    "reports": 2_000,
    "claims": 1_000,
    "citations": 2_000,
}

ASPECTS = ["the acting", "the plot", "the directing", "the cinematography",
           "the soundtrack", "the pacing", "the ending", "the dialogue",
           "the special effects", "the character development",
           "the screenplay", "the editing"]


def _n_docs(name, sf):
    return max(8, int(SETS[name] * sf))


# ------------------------------------------------------- set builders

def _imdb_pool():
    from huggingface_hub import hf_hub_download
    texts = []
    for split in ("train", "test"):
        f = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["stanfordnlp/imdb"])
        texts += pq.read_table(f, columns=["text"]).column(
            "text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    return texts


def _biodex_rows(n):
    """Real BioDEX rows, unpadded, un-concatenated: (text, reactions)
    per row. `reactions` seeds the `terms` table."""
    from datasets import load_dataset
    ds = load_dataset(
        "BioDEX/BioDEX-Reactions", split="train", streaming=True,
        revision=SOURCE_REVISIONS["BioDEX/BioDEX-Reactions"])
    rows = []
    for row in ds:
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        reactions = [t.strip() for t in
                     str(row.get("reactions", "")).split(",") if t.strip()]
        if len(text) >= 200 and reactions:
            rows.append((text, reactions))
        if len(rows) >= n:
            break
    return rows


def _fever_data(n_claims):
    """FEVER claims (SUPPORTS/REFUTES only - those carry a real
    annotated evidence page) and the small pool of Wikipedia pages
    those claims actually reference. The evidence pool is bounded by
    the sampled claims - a full join against all 5M wiki pages is not
    the query under test (see the FEVER open item in
    query-design.md)."""
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("fever/fever", "v1.0/labelled_dev/0000.parquet",
                        repo_type="dataset",
                        revision=SOURCE_REVISIONS["fever/fever"])
    rows = pq.read_table(f).to_pylist()
    seen, claims = set(), []
    for r in rows:
        if (r["id"] in seen or r["label"] not in ("SUPPORTS", "REFUTES")
                or not r["evidence_wiki_url"]):
            continue
        seen.add(r["id"])
        claims.append(r)
        if len(claims) >= n_claims:
            break
    pages_needed = {r["evidence_wiki_url"] for r in claims}
    page_text = {}
    for shard in range(10):
        if len(page_text) >= len(pages_needed):
            break
        fw = hf_hub_download(
            "fever/fever",
            f"wiki_pages/partial-wikipedia_pages/{shard:04d}.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["fever/fever"])
        t = pq.read_table(fw, columns=["id", "text"])
        for pid, txt in zip(t.column("id").to_pylist(),
                            t.column("text").to_pylist()):
            if pid in pages_needed and pid not in page_text:
                page_text[pid] = txt
    return claims, page_text


def _lepard_rows(n):
    """Real LePaRD citation events, unpadded: (destination_context,
    canonical passage text, passage_id) per row, one row per distinct
    dest_id (a citing case can quote several passages; one keeps the
    sample from packing near-duplicate excerpts under different
    passage_ids).

    Passage text comes from passage_dict.json's canonical entry for
    that passage_id, not the CSV's own `quote` column - `quote` is
    often just the isolated cited clause, sometimes with OCR noise
    and no surrounding grammar (e.g. "foster[s] an excessive
    government entanglement with religion."); passage_dict gives the
    fuller paragraph it was drawn from - real numbers, measured on
    this file: destination_context averages 709 chars, raw `quote`
    132, the canonical passage 242 - long enough to actually read."""
    import json as _json

    from huggingface_hub import hf_hub_download
    import pandas as pd

    csv_path = hf_hub_download("rmahari/LePaRD", "top_10000_data.csv.gz",
                               repo_type="dataset",
                               revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    dict_path = hf_hub_download("rmahari/LePaRD", "passage_dict.json",
                                repo_type="dataset",
                                revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    passages = _json.load(open(dict_path))["data"]

    rows, seen_dest = [], set()
    cols = ["dest_id", "destination_context", "passage_id"]
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=50_000):
        for r in chunk.itertuples(index=False):
            if r.dest_id in seen_dest:
                continue
            text = passages.get(r.passage_id)
            ctx = str(r.destination_context)
            if not text or len(ctx) < 50:
                continue
            seen_dest.add(r.dest_id)
            rows.append((ctx, str(text).strip(), r.passage_id))
            if len(rows) >= n:
                return rows
    return rows


def _vocab_table(rows, idx, cap=None):
    """A frequency-sorted, deduplicated vocabulary column from one
    field across sampled rows (the `terms` table, from `reactions`)."""
    freq = {}
    for row in rows:
        for t in row[idx]:
            freq[t] = freq.get(t, 0) + 1
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    return vocab[:cap] if cap else vocab


def _build_citations(d, sf, force=False):
    """citations.parquet, idempotent: real LePaRD citation events,
    self-joined against itself - one table, one row per citing case,
    its own excerpt (destination_context) and its own actually-cited
    passage (passage_dict.json text) on the same row. A join query
    reads this same table under two aliases with two different
    columns - see LEP-2 in queries() below.

    Called from both branches of build_sets (cache hit and full
    build) rather than gated behind the one whole-directory DONE
    marker, so adding a table later backfills existing sf caches
    instead of silently no-op'ing against a stale marker - exactly
    what broke the first time this table was added."""
    path = d / "citations.parquet"
    if path.exists() and not force:
        return
    n = _n_docs("citations", sf)
    lep = _lepard_rows(n)
    pq.write_table(pa.table({
        "id": [f"lp{i}" for i in range(len(lep))],
        "destination_context": [r[0] for r in lep],
        "passage_text": [r[1] for r in lep],
        "passage_id": [r[2] for r in lep],
    }), path)


def build_sets(data_dir, sf, lf=1):
    """All seven tables as parquet files, cached by sf.

    lf (load factor) is accepted but unused: documents here are real
    and unpadded, so there's nothing to scale. Kept in the signature
    so callers don't have to change when it's wired back up."""
    d = Path(data_dir) / f"sf{sf}"
    marker = d / "DONE"
    if marker.exists():
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "data_seed": DATA_SEED,
            "scale_factor": sf,
            "source_revisions": SOURCE_REVISIONS,
        }
        try:
            current = json.loads(marker.read_text())
        except json.JSONDecodeError:
            current = None
        if current == expected:
            _build_citations(d, sf)
            return d
    d.mkdir(parents=True, exist_ok=True)

    def write(name, ids, col_name, values):
        pq.write_table(pa.table({"id": ids, col_name: values}),
                       d / f"{name}.parquet")

    # reviews: real IMDB text, one row = one review, unpadded
    n = _n_docs("reviews", sf)
    imdb = _imdb_pool()
    write("reviews", [f"rv{i}" for i in range(n)], "body", imdb[:n])
    write("aspects", [f"as{i}" for i in range(len(ASPECTS))],
          "aspect", ASPECTS)

    # reports: real BioDEX text, one row = one report, unpadded
    n = _n_docs("reports", sf)
    bio = _biodex_rows(n)
    pq.write_table(pa.table({
        "id": [f"rp{i}" for i in range(len(bio))],
        "report": [t for t, _ in bio],
        "reactions": [r for _, r in bio],
    }), d / "reports.parquet")
    terms = _vocab_table(bio, 1, cap=2_560)
    write("terms", [f"tm{i}" for i in range(len(terms))], "term", terms)

    # claims + evidence: real FEVER claims and only the Wikipedia
    # pages those claims reference
    n = _n_docs("claims", sf)
    claims, page_text = _fever_data(n)
    pq.write_table(pa.table({
        "id": [f"cl{i}" for i in range(len(claims))],
        "claim": [c["claim"] for c in claims],
        "label": [c["label"] for c in claims],
        "evidence_wiki_url": [c["evidence_wiki_url"] for c in claims],
    }), d / "claims.parquet")
    ev_ids = list(page_text.keys())
    pq.write_table(pa.table({
        "id": ev_ids,
        "text": [page_text[p] for p in ev_ids],
    }), d / "evidence.parquet")

    _build_citations(d, sf, force=True)

    marker.write_text(json.dumps({
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "data_seed": DATA_SEED,
        "scale_factor": sf,
        "source_revisions": SOURCE_REVISIONS,
    }, indent=2, sort_keys=True))
    return d


def register_sets(sess, data_dir):
    from quail.catalog import DocumentProvider
    for name in ("reviews", "aspects", "reports", "terms",
                "claims", "evidence", "citations"):
        sess.register(name, DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col="id"))


# ---------------------------------------------------------- predicates
#
# Same shape as the REACTION/SUPPORT templates this replaces: the
# instruction text is written before {0}, and the engine relocates it
# to just after the document (`split_frame` in `quail/logical.py`) so
# it's paid once per anchor's kept KV, not once per pair. Frame sits
# right before the candidate/question, which is where the 4B model is
# sensitive to wording - neutral, "judge strictly" phrasing throughout,
# the fix already validated on the original REACTION predicate (see
# query-design.md).
#
# None of these have been through the judge pass. Wording may need to
# change once that pass runs and some predicate misses the 90%
# agreement floor or clusters selectivity with another predicate.

F1 = ("Judge strictly from the review above whether it mentions at "
      "least one positive aspect of the movie.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions at least one positive aspect "
      "of the movie, FALSE otherwise.\nANSWER=")

F4 = ("Judge strictly from the review above whether it discusses the "
      "ending of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
      "review discusses the ending of the movie, FALSE otherwise.\n"
      "ANSWER=")

F5 = ("Judge strictly from the review above whether it mentions any "
      "specific actor or actress by name.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions a specific actor or actress "
      "by name, FALSE otherwise.\nANSWER=")

DISCUSS_ASPECT = ("Candidate movie aspects follow, one at a time. For "
                   "each, judge strictly from the review above whether "
                   "it discusses that aspect of the movie.\n\n{0}\n\n"
                   "ASPECT: {1}\nInstruction: answer TRUE if the review "
                   "above discusses this aspect, FALSE otherwise.\n"
                   "ANSWER=")

F7 = ("Judge strictly from the report above whether it describes a "
      "case involving a female patient.\n\n{0}\n\nInstruction: answer "
      "TRUE if the report describes a case involving a female patient, "
      "FALSE otherwise.\nANSWER=")

F8 = ("Judge strictly from the report above whether it describes "
      "combination drug therapy.\n\n{0}\n\nInstruction: answer TRUE if "
      "the report describes combination drug therapy, FALSE otherwise.\n"
      "ANSWER=")

F9 = ("Judge strictly from the report above whether it describes a "
      "serious or life-threatening adverse event.\n\n{0}\n\n"
      "Instruction: answer TRUE if the report describes a serious or "
      "life-threatening adverse event, FALSE otherwise.\nANSWER=")

# Unchanged from the earlier design: after-document frame, neutral
# wording. An earlier version with similar framing before the report
# measured selectivity 0.76 (biased toward YES); this version measured
# 0.287.
REACTION = ("Candidate medical reaction terms follow, one at a time. "
            "For each, judge strictly from the report above whether it "
            "describes that reaction as something the patient "
            "experienced.\n\n{0}\n\nCANDIDATE REACTION: {1}\n"
            "Instruction: answer TRUE if the report above describes "
            "this reaction, FALSE otherwise.\nANSWER=")

F11 = ("Judge strictly from the claim above whether it asserts "
       "something about a person, rather than an organization, place, "
       "or event.\n\n{0}\n\nInstruction: answer TRUE if the claim "
       "asserts something about a person, FALSE otherwise.\nANSWER=")

F12 = ("Judge strictly from the claim above whether it contains a "
       "specific date or year.\n\n{0}\n\nInstruction: answer TRUE if "
       "the claim contains a specific date or year, FALSE otherwise.\n"
       "ANSWER=")

F14 = ("Judge strictly from the claim above whether it references a "
       "specific place (a city, country, or other named location).\n\n"
       "{0}\n\nInstruction: answer TRUE if the claim references a "
       "specific place, FALSE otherwise.\nANSWER=")

# Unchanged from the earlier design: same after-document fix as
# REACTION, for the same reason.
SUPPORT = ("Wikipedia passages follow, one at a time. For each, judge "
           "strictly from the claim above whether it supports that "
           "claim.\n\n{0}\n\nPASSAGE:\n{1}\nInstruction: answer TRUE if "
           "the passage above supports this claim, FALSE otherwise.\n"
           "ANSWER=")

# FEV-5 only: filters the evidence side of a join, not just the
# anchor. Mirrors F11's "about a person" judgment so the join pairs
# claim and passage on the same axis, independently filtered.
F13 = ("Judge strictly from the Wikipedia passage above whether it "
       "primarily describes a specific person (their life, actions, "
       "or role), rather than an organization, place, or event.\n\n"
       "{0}\n\nInstruction: answer TRUE if the passage primarily "
       "describes a specific person, FALSE otherwise.\nANSWER=")

# LePaRD predicates: "excerpt" for destination_context throughout,
# to avoid colliding with this dataset's own use of "passage" for
# the quoted/cited text.
LEP1 = ("Judge strictly from the excerpt above whether it argues that "
        "the cited case's reasoning does not apply here.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt argues the cited "
        "case's reasoning does not apply here, FALSE otherwise.\n"
        "ANSWER=")

LEP2 = ("Judge strictly from the excerpt above whether it discusses a "
        "procedural or jurisdictional issue.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt discusses a procedural or "
        "jurisdictional issue, FALSE otherwise.\nANSWER=")

LEP3 = ("Judge strictly from the excerpt above whether it treats the "
        "cited passage as binding precedent.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt treats the cited passage as "
        "binding precedent, FALSE otherwise.\nANSWER=")

LEP4 = ("Judge strictly from the excerpt above whether it cites the "
        "passage to support a conclusion about a party's liability or "
        "guilt.\n\n{0}\n\nInstruction: answer TRUE if the excerpt "
        "cites the passage to support a conclusion about a party's "
        "liability or guilt, FALSE otherwise.\nANSWER=")

LEP5 = ("Judge strictly from the excerpt above whether it acknowledges "
        "disagreement between courts on the issue.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt acknowledges "
        "disagreement between courts on the issue, FALSE otherwise.\n"
        "ANSWER=")

# LEP-7 only: filters the passage side of the self-join, not just the
# excerpt (anchor) side.
LEPS1 = ("Judge strictly from the passage above whether it states a "
         "general legal rule.\n\n{0}\n\nInstruction: answer TRUE if "
         "the passage states a general legal rule, FALSE otherwise.\n"
         "ANSWER=")

# The LEP-2..LEP-7 join predicate: real ground truth exists for this
# one (passage_id, from the dataset itself, not a judge pass) - see
# judge_pass.py's LEP probe.
LEPJOIN = ("Judge strictly from the excerpt above whether the passage "
           "below is the one being cited.\n\n{0}\n\nPASSAGE:\n{1}\n"
           "Instruction: answer TRUE if the passage below is the one "
           "being cited in the excerpt above, FALSE otherwise.\n"
           "ANSWER=")


# ---------------------------------------------------------- queries

def queries(sess):
    """id -> (description, callable() -> Query). Fresh Query objects
    per call so each pass re-plans."""
    import quail

    def make(doc_table, doc_alias, doc_col, filters, joins, select):
        """filters: list of prompt templates applied in order to the
        base table. joins: list of (partner_table, partner_alias,
        partner_col, prompt_template[, partner_filters]) applied in
        order, each dependent on whatever survived the stages before
        it; partner_filters push filters onto the partner side before
        the join (the FEV-5/6 and LEP-7 two-sided shape)."""
        def build():
            qy = sess.docs(doc_table).alias(doc_alias)
            for tmpl in filters:
                qy = qy.ai_filter(
                    quail.prompt(tmpl, quail.col(f"{doc_alias}.{doc_col}")))
            for partner, palias, pcol, tmpl, *rest in joins:
                pq = sess.docs(partner).alias(palias)
                for pf in (rest[0] if rest else ()):
                    pq = pq.ai_filter(
                        quail.prompt(pf, quail.col(f"{palias}.{pcol}")))
                qy = qy.ai_join(
                    pq,
                    quail.prompt(tmpl, quail.col(f"{doc_alias}.{doc_col}"),
                                quail.col(f"{palias}.{pcol}")))
            return qy.select(*select)
        return build

    q = {}

    # IMDB: filter alone, join alone, then filter-chain depth 1/2/3
    # feeding the one join (reviews x aspects).
    q["IMDB-1"] = ("filter: F1 (at least one positive aspect)", make(
        "reviews", "r", "body", [F1], [], ["r.id"]))
    q["IMDB-2"] = ("join: J1 (reviews x aspects)", make(
        "reviews", "r", "body", [],
        [("aspects", "a", "aspect", DISCUSS_ASPECT)], ["r.id", "a.id"]))
    q["IMDB-3"] = ("F1 -> J1, dependent", make(
        "reviews", "r", "body", [F1],
        [("aspects", "a", "aspect", DISCUSS_ASPECT)], ["r.id", "a.id"]))
    q["IMDB-4"] = ("F1 -> F4 -> J1, 2 filters then 1 join", make(
        "reviews", "r", "body", [F1, F4],
        [("aspects", "a", "aspect", DISCUSS_ASPECT)], ["r.id", "a.id"]))
    q["IMDB-5"] = ("F1 -> F4 -> F5 -> J1, 3 filters then 1 join", make(
        "reviews", "r", "body", [F1, F4, F5],
        [("aspects", "a", "aspect", DISCUSS_ASPECT)], ["r.id", "a.id"]))
    # filter chains with no join: IMDB-4 and IMDB-5 without their join,
    # so the difference is the join's cost.
    q["IMDB-6"] = ("F1 -> F4, 2 filters, no join", make(
        "reviews", "r", "body", [F1, F4], [], ["r.id"]))
    q["IMDB-7"] = ("F1 -> F4 -> F5, 3 filters, no join", make(
        "reviews", "r", "body", [F1, F4, F5], [], ["r.id"]))

    # BioDEX: same five shapes (reports x terms).
    q["BIO-1"] = ("filter: F7 (female patient)", make(
        "reports", "r", "report", [F7], [], ["r.id"]))
    q["BIO-2"] = ("join: J1 (reports x terms)", make(
        "reports", "r", "report", [],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))
    q["BIO-3"] = ("F7 -> J1, dependent", make(
        "reports", "r", "report", [F7],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))
    q["BIO-4"] = ("F7 -> F8 -> J1, 2 filters then 1 join", make(
        "reports", "r", "report", [F7, F8],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))
    q["BIO-5"] = ("F7 -> F8 -> F9 -> J1, 3 filters then 1 join", make(
        "reports", "r", "report", [F7, F8, F9],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))

    # FEVER: filter alone, join alone, then a filter chain to depth 2
    # only (depth 3 hit 0 rows - see the module docstring), plus the
    # two-sided FEV-5/FEV-6 pushdown, the shape that actually tests
    # joins under independent filtering on both sides.
    q["FEV-1"] = ("filter: F11 (about a person)", make(
        "claims", "c", "claim", [F11], [], ["c.id"]))
    q["FEV-2"] = ("join: J3 (claims x evidence)", make(
        "claims", "c", "claim", [],
        [("evidence", "e", "text", SUPPORT)], ["c.id", "e.id"]))
    q["FEV-3"] = ("F11 -> J3, dependent", make(
        "claims", "c", "claim", [F11],
        [("evidence", "e", "text", SUPPORT)], ["c.id", "e.id"]))
    q["FEV-4"] = ("F11 -> F12 -> J3, 2 filters then 1 join", make(
        "claims", "c", "claim", [F11, F12],
        [("evidence", "e", "text", SUPPORT)], ["c.id", "e.id"]))

    q["FEV-5"] = ("2F + 1J: two-sided pushdown - F11 on claims, F13 on "
                  "evidence, each filtered before J3", make(
        "claims", "c", "claim", [F11],
        [("evidence", "e", "text", SUPPORT, [F13])], ["c.id", "e.id"]))
    q["FEV-6"] = ("3F + 1J: two-sided pushdown, deeper - F11 -> F12 on "
                  "claims, F13 on evidence, each filtered before J3",
                  make(
        "claims", "c", "claim", [F11, F12],
        [("evidence", "e", "text", SUPPORT, [F13])], ["c.id", "e.id"]))

    # LePaRD: one table, `citations`, self-joined - the anchor alias
    # ("d") reads destination_context, the partner alias ("s") reads
    # passage_text, both from the same registered provider. Same
    # five-shape pattern as IMDB/BioDEX, plus a 5-filter chain
    # (LEP-6) and the two-sided pushdown (LEP-7, matching FEV-5/6).
    q["LEP-1"] = ("filter: LEP1 (reasoning does not apply)", make(
        "citations", "d", "destination_context", [LEP1], [], ["d.id"]))
    q["LEP-2"] = ("join: self-join (citations x citations)", make(
        "citations", "d", "destination_context", [],
        [("citations", "s", "passage_text", LEPJOIN)], ["d.id", "s.id"]))
    q["LEP-3"] = ("LEP1 -> join, dependent", make(
        "citations", "d", "destination_context", [LEP1],
        [("citations", "s", "passage_text", LEPJOIN)], ["d.id", "s.id"]))
    q["LEP-4"] = ("LEP1 -> LEP2 -> join, 2 filters then 1 join", make(
        "citations", "d", "destination_context", [LEP1, LEP2],
        [("citations", "s", "passage_text", LEPJOIN)], ["d.id", "s.id"]))
    q["LEP-5"] = ("LEP1 -> LEP2 -> LEP3 -> join, 3 filters then 1 join",
                  make("citations", "d", "destination_context",
                      [LEP1, LEP2, LEP3],
                      [("citations", "s", "passage_text", LEPJOIN)],
                      ["d.id", "s.id"]))
    q["LEP-6"] = ("LEP1..LEP5 -> join, 5 filters then 1 join", make(
        "citations", "d", "destination_context",
        [LEP1, LEP2, LEP3, LEP4, LEP5],
        [("citations", "s", "passage_text", LEPJOIN)], ["d.id", "s.id"]))

    q["LEP-7"] = ("2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
                  "LEPS1 on passages, each filtered before the self-join",
                  make(
        "citations", "d", "destination_context", [LEP1, LEP2],
        [("citations", "s", "passage_text", LEPJOIN, [LEPS1])],
        ["d.id", "s.id"]))
    # LEP-6 without its join: the deepest filter chain in the suite,
    # five stages of KV reuse with no join work mixed in.
    q["LEP-8"] = ("LEP1..LEP5, 5 filters, no join", make(
        "citations", "d", "destination_context",
        [LEP1, LEP2, LEP3, LEP4, LEP5], [], ["d.id"]))

    return q


# ----------------------------------------------------------- driver

def _artifact_stem(started, sf, lf, model):
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-quailb-sf{sf}-lf{lf}-{model}"


def run_suite(data_dir, sf=0.1, lf=1, gpus=1, only=None,
              out_path=None, cpu_memory_gb=80, model="qwen3-4b-fp8",
              accuracy=True, ground_truth_collection=None,
              h100_usd_per_hour=3.9492, ground_truth_files=None,
              prediction=None, artifact_stem=None):
    """cpu_memory_gb defaults to what the 96 GB worker container
    holds: a 64 GB store (8 slabs). The corpus KV usually exceeds it,
    so the length threshold keeps the longest documents - partial
    restores are the capacity arithmetic working, not a bug."""
    import quail
    from quail.bench.evaluate import (
        BenchmarkEvaluator,
        H100_PRICE_SOURCE,
        ModalVolumeFiles,
        add_query_metrics,
        corpus_identity,
        load_ground_truth,
        read_corpus,
        summarize_queries,
    )
    from quail.planner.plan import EngineConfig

    d = build_sets(data_dir, sf, lf)
    corpus_rows = read_corpus(d)
    corpus = corpus_identity(
        corpus_rows, sf, DATA_SEED, SOURCE_REVISIONS)
    evaluator = None
    truth = None
    result_files = ground_truth_files or ModalVolumeFiles()
    if accuracy:
        truth = load_ground_truth(
            result_files, scale_factor=sf, corpus_id=corpus["corpus_id"],
            collection_id=ground_truth_collection)
        if truth.corpus_id != corpus["corpus_id"]:
            raise ValueError(
                f"benchmark corpus {corpus['corpus_id']} does not match "
                f"ground truth {truth.corpus_id}")
        evaluator = BenchmarkEvaluator(truth, corpus_rows)
    sess = quail.Session(EngineConfig(gpus=gpus, model=model,
                                      cpu_memory_gb=cpu_memory_gb))
    register_sets(sess, d)
    qdefs = queries(sess)
    ids = [i for i in qdefs if only is None or i in only]
    started = datetime.now(timezone.utc)
    artifact_stem = artifact_stem or _artifact_stem(
        started, sf, lf, model)
    run_id = (f"qb_{started.strftime('%Y%m%dT%H%M%SZ')}_"
              f"{uuid.uuid4().hex[:8]}")
    raw_root = f"benchmarks/quailb/runs/{run_id}"
    aggregate_volume_path = f"{raw_root}/{artifact_stem}.json"
    suite = dict(
        run_id=run_id,
        artifact_stem=artifact_stem,
        started_at=started.isoformat(),
        prediction=prediction,
        sf=sf, lf=lf, gpus=gpus, model=model,
        corpus_id=corpus["corpus_id"],
        raw_volume_path=f"/results/{raw_root}",
        aggregate_volume_path=f"/results/{aggregate_volume_path}",
        pricing=dict(
            gpu="H100",
            h100_usd_per_hour=h100_usd_per_hour,
            gpu_count=gpus,
            price_source=H100_PRICE_SOURCE,
            method=("query runtime in hours multiplied by the H100 hourly "
                    "price and GPU count"),
        ),
        metric_definitions=dict(
            runtime_s=("GPU worker query runtime; corpus construction, "
                       "ground truth loading, and local evaluation are "
                       "excluded"),
            runtime_with_boot_s=("GPU worker query runtime plus model load "
                                 "and warmup; ground truth loading is "
                                 "excluded"),
            pass_wall_s=("host time for the query loop; ground truth "
                         "loading is excluded"),
            tokens_processed=("sum of fresh tokens sent through model "
                              "forward calls; tokens read from KV are not "
                              "counted again"),
            input_document_rows=("sum of input table rows for every query "
                                 "alias; a self join counts the table once "
                                 "per alias"),
            documents_per_second=("input_document_rows divided by query "
                                  "runtime_s"),
            inference_cost_per_token_usd=("inference_cost_usd divided by "
                                          "tokens_processed"),
            answer_accuracy=("agreement with saved labels on model calls "
                             "that the query evaluated"),
            output_accuracy=("precision, recall, and F1 for final returned "
                             "rows against rows derived from saved labels"),
        ),
        ground_truth=(
            dict(collection_id=truth.collection_id,
                 reference_model=truth.reference_model)
            if truth else None),
        passes={})
    try:
        for pass_name in ("cold", "warm"):
            sess.set_store(pass_name == "warm")
            if pass_name == "cold":
                sess.flush_store()
            rows = []
            t_pass = time.time()
            for qid in ids:
                desc, build = qdefs[qid]
                print(f"[quailb] {pass_name} {qid}: {desc}",
                      flush=True)
                try:
                    query = build()
                    res = query.run()
                    row = dict(query=qid, desc=desc,
                               wall_s=res.report["wall_s"],
                               boot_s=res.report["boot_s"],
                               boot_kind=res.report.get("boot_kind"),
                               boot=res.report.get("boot"),
                               fresh_tokens=res.report["fresh_tokens"],
                               rows=len(res.rows),
                               peak_gib=res.report.get("peak_gib"),
                               stages=res.report["stages"],
                               store=res.report.get("store"))
                    if evaluator is not None:
                        evaluation = evaluator.evaluate(query, res)
                        add_query_metrics(
                            row, evaluation, h100_usd_per_hour, gpus)
                        raw_path = f"{raw_root}/{pass_name}/{qid}.json"
                        result_files.write_json(raw_path, {
                            "run_id": run_id,
                            "pass": pass_name,
                            "query": qid,
                            "description": desc,
                            "columns": res.columns,
                            "rows": res.rows,
                            "answer_rows": res.answer_rows,
                            "engine_report": res.report,
                            "accuracy": row["accuracy"],
                        })
                        row["raw_volume_path"] = f"/results/{raw_path}"
                except Exception as e:            # noqa: BLE001
                    row = dict(query=qid, desc=desc,
                               error=f"{type(e).__name__}: {e}")
                rows.append(row)
                print(f"[quailb] {row}", flush=True)
            passed = dict(
                queries=rows,
                pass_wall_s=round(time.time() - t_pass, 1))
            if evaluator is not None:
                passed["summary"] = summarize_queries(
                    rows, h100_usd_per_hour, gpus)
            suite["passes"][pass_name] = passed
    finally:
        sess.close()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(suite, f, indent=2)
        print(f"[quailb] saved {out_path}", flush=True)
    result_files.write_json(aggregate_volume_path, suite)
    print(
        f"[quailb] saved /results/{aggregate_volume_path} on quail-results",
        flush=True)
    return suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1)
    ap.add_argument("--lf", type=int, default=1,
                    help="load factor, unused for now (see build_sets)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--data-dir", default="results/quailb_data")
    ap.add_argument("--only", default=None,
                    help=("comma-separated query ids; default runs all "
                          "queries"))
    ap.add_argument(
        "--out", default=None,
        help=("JSON path; default uses a UTC timestamp under "
              "results/benchmark/"))
    ap.add_argument("--model", default="qwen3-4b-fp8",
                    help="registered ModelSpec name, see quail.specs.MODELS")
    ap.add_argument(
        "--accuracy", action=argparse.BooleanOptionalAction, default=True,
        help="compare answers and output rows with the Modal ground truth")
    ap.add_argument("--ground-truth-collection", default=None,
                    help="collection id; default is the one matching the corpus")
    ap.add_argument("--h100-usd-per-hour", type=float, default=3.9492,
                    help="H100 price used for query cost estimates")
    ap.add_argument("--prediction", default=None,
                    help="prediction stated before this benchmark run")
    ap.add_argument(
        "--report", action=argparse.BooleanOptionalAction, default=True,
        help=("write a Markdown report under results/benchmark/ and a PNG "
              "plot under reports/plots/benchmark/"))
    ap.add_argument("--report-path", default=None,
                    help="Markdown path; default uses the run UTC timestamp")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    started = datetime.now(timezone.utc)
    artifact_stem = _artifact_stem(
        started, args.sf, args.lf, args.model)
    out = args.out or f"results/benchmark/{artifact_stem}.json"
    suite = run_suite(
        args.data_dir, sf=args.sf, lf=args.lf, gpus=args.gpus,
        only=only, out_path=out, model=args.model,
        accuracy=args.accuracy,
        ground_truth_collection=args.ground_truth_collection,
        h100_usd_per_hour=args.h100_usd_per_hour,
        prediction=args.prediction,
        artifact_stem=artifact_stem)
    if args.report:
        if not args.accuracy:
            raise ValueError("the evaluation report requires accuracy")
        report_path = args.report_path or (
            f"results/benchmark/{suite['artifact_stem']}.md")
        script = Path(__file__).resolve().parents[1] / "reports" \
            / "make_quailb_eval_plots.py"
        subprocess.run(
            [sys.executable, str(script), "--input", out,
             "--report", report_path], check=True)


if __name__ == "__main__":
    main()
