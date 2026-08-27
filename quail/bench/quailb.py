"""QUAIL-B: thirty-five queries over four document sets (IMDB, BioDEX,
FEVER, LePaRD).

    uv run python -m quail.bench.quailb --sf 0.1 --model qwen3-4b-fp8 --gpus 1
"""

import argparse
import hashlib
import heapq
import json
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_SEED = 20260818
CACHE_SCHEMA_VERSION = 5
LEPARD_POSITIVE_PAIRS = 5_000

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
# bounded by whichever claims get sampled. LePaRD scales sampled positive
# citation pairs before its two document tables are deduplicated.
SETS = {
    "reviews": 50_000,
    "reports": 2_000,
    "claims": 1_000,
}

ASPECTS = ["the acting", "the plot", "the directing", "the cinematography",
           "the soundtrack", "the pacing", "the ending", "the dialogue",
           "the special effects", "the character development",
           "the screenplay", "the editing"]

QUERY_ORDER = (
    *(f"IMDB-{i}" for i in range(1, 11)),
    *(f"BIO-{i}" for i in range(1, 9)),
    *(f"FEV-{i}" for i in range(1, 10)),
    *(f"LEP-{i}" for i in range(1, 9)),
)


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


def _n_docs(name, sf):
    return max(8, int(SETS[name] * sf))


def _n_lepard_pairs(sf):
    return max(8, int(LEPARD_POSITIVE_PAIRS * sf))


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
    """FEVER claims (SUPPORTS/REFUTES only) and the Wikipedia pages they
    reference. The evidence pool is bounded by the sampled claims."""
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


def _lepard_pair_priority(dest_id, passage_id):
    value = f"{DATA_SEED}\0{dest_id}\0{passage_id}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=16).digest(),
                          "big")


def _sample_lepard_pairs(rows, passages, n):
    """Select a stable random sample of distinct known citation pairs."""
    passages = {str(key): str(value).strip()
                for key, value in passages.items() if value}
    selected = []
    selected_rows = {}
    for dest_id, destination_context, passage_id in rows:
        dest_id = str(dest_id)
        passage_id = str(passage_id)
        key = (dest_id, passage_id)
        passage_text = passages.get(passage_id)
        context = str(destination_context).strip()
        if not passage_text or len(context) < 50:
            continue
        if key in selected_rows:
            prior_context, _passage_text = selected_rows[key]
            if (len(context), context) > (len(prior_context), prior_context):
                selected_rows[key] = (context, passage_text)
            continue
        priority = _lepard_pair_priority(dest_id, passage_id)
        item = (-priority, dest_id, passage_id)
        if len(selected) < n:
            heapq.heappush(selected, item)
            selected_rows[key] = (context, passage_text)
            continue
        if priority >= -selected[0][0]:
            continue
        removed = heapq.heapreplace(selected, item)
        del selected_rows[(removed[1], removed[2])]
        selected_rows[key] = (context, passage_text)
    pairs = []
    for _priority, dest_id, passage_id in sorted(
            selected, key=lambda item: (-item[0], item[1], item[2])):
        context, passage_text = selected_rows[(dest_id, passage_id)]
        pairs.append((dest_id, passage_id, context, passage_text))
    return pairs


def _lepard_documents(pairs):
    """Deduplicate the two document columns after sampling pairs."""
    contexts = {}
    passages = {}
    for _dest_id, passage_id, context, passage_text in pairs:
        contexts.setdefault(context, set()).add(passage_id)
        passages.setdefault(passage_text, set()).add(passage_id)
    context_rows = [{
        "id": f"lc{i}",
        "destination_context": context,
        "cited_passage_ids": sorted(passage_ids),
    } for i, (context, passage_ids) in enumerate(contexts.items())]
    passage_rows = [{
        "id": f"lp{i}",
        "passage_text": passage_text,
        "passage_ids": sorted(passage_ids),
    } for i, (passage_text, passage_ids) in enumerate(passages.items())]
    return context_rows, passage_rows


def _lepard_rows(n):
    """Read LePaRD and sample known positive citation pairs."""
    import json as _json

    import pandas as pd
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download("rmahari/LePaRD", "top_10000_data.csv.gz",
                               repo_type="dataset",
                               revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    dict_path = hf_hub_download("rmahari/LePaRD", "passage_dict.json",
                                repo_type="dataset",
                                revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    with open(dict_path) as source:
        passages = _json.load(source)["data"]

    cols = ["dest_id", "destination_context", "passage_id"]
    chunks = pd.read_csv(
        csv_path, usecols=cols, chunksize=50_000,
        dtype={"dest_id": "string", "destination_context": "string",
               "passage_id": "string"})
    rows = (row for chunk in chunks
            for row in chunk.loc[:, cols].itertuples(index=False, name=None))
    return _sample_lepard_pairs(rows, passages, n)


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


def _build_lepard(d, sf, force=False):
    """Build the two deduplicated LePaRD document tables."""
    context_path = d / "citation_contexts.parquet"
    passage_path = d / "citation_passages.parquet"
    if context_path.exists() and passage_path.exists() and not force:
        return
    expected_pairs = _n_lepard_pairs(sf)
    pairs = _lepard_rows(expected_pairs)
    if len(pairs) != expected_pairs:
        raise ValueError(
            f"LePaRD provided {len(pairs)} valid citation pairs, expected "
            f"{expected_pairs}")
    contexts, passages = _lepard_documents(pairs)
    context_schema = pa.schema([
        ("id", pa.string()),
        ("destination_context", pa.string()),
        ("cited_passage_ids", pa.list_(pa.string())),
    ])
    passage_schema = pa.schema([
        ("id", pa.string()),
        ("passage_text", pa.string()),
        ("passage_ids", pa.list_(pa.string())),
    ])
    pq.write_table(pa.Table.from_pylist(contexts, schema=context_schema),
                   context_path)
    pq.write_table(pa.Table.from_pylist(passages, schema=passage_schema),
                   passage_path)


def build_sets(data_dir, sf, lf=1):
    """All eight tables as parquet files, cached by sf.

    lf (load factor) is accepted but unused: documents here are real
    and unpadded, so there's nothing to scale. Kept in the signature
    so callers don't have to change when it's wired back up."""
    d = Path(data_dir) / f"sf{sf}"
    marker = d / "DONE"
    if marker.exists():
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "data_seed": DATA_SEED,
            "lepard_positive_pairs": LEPARD_POSITIVE_PAIRS,
            "scale_factor": sf,
            "source_revisions": SOURCE_REVISIONS,
        }
        try:
            current = json.loads(marker.read_text())
        except json.JSONDecodeError:
            current = None
        if current == expected:
            _build_lepard(d, sf)
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

    _build_lepard(d, sf, force=True)

    marker.write_text(json.dumps({
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "data_seed": DATA_SEED,
        "lepard_positive_pairs": LEPARD_POSITIVE_PAIRS,
        "scale_factor": sf,
        "source_revisions": SOURCE_REVISIONS,
    }, indent=2, sort_keys=True))
    return d


def register_sets(sess, data_dir):
    from quail.catalog import DocumentProvider
    for name in ("reviews", "aspects", "reports", "terms",
                "claims", "evidence", "citation_contexts",
                "citation_passages"):
        sess.register(name, DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col="id"))


# ---------------------------------------------------------- predicates

F1 = ("Judge strictly from the review above whether it mentions at "
      "least one positive aspect of the movie.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions at least one positive aspect "
      "of the movie, FALSE otherwise.")

F4 = ("Judge strictly from the review above whether it discusses the "
      "ending of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
      "review discusses the ending of the movie, FALSE otherwise.")

F5 = ("Judge strictly from the review above whether it mentions any "
      "specific actor or actress by name.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions a specific actor or actress "
      "by name, FALSE otherwise.")

DISCUSS_ASPECT = ("Does the review in DOCUMENT {0} discuss the movie "
                  "aspect in DOCUMENT {1}?")


# IMDB-8 only: a second question over the same aspects table, joined
# under a second alias (a2) - a 2-join star, both joins anchored on
# reviews so the second stage runs over whatever DISCUSS_ASPECT
# already kept.
ASPECT_SENTIMENT = ("Does the review in DOCUMENT {0} express positive "
                    "sentiment about the movie aspect in DOCUMENT {1}?")

F7 = ("Judge strictly from the report above whether it describes a "
      "case involving a female patient.\n\n{0}\n\nInstruction: answer "
      "TRUE if the report describes a case involving a female patient, "
      "FALSE otherwise.")

F8 = ("Judge strictly from the report above whether it describes "
      "combination drug therapy.\n\n{0}\n\nInstruction: answer TRUE if "
      "the report describes combination drug therapy, FALSE otherwise.")

F9 = ("Judge strictly from the report above whether it describes a "
      "serious or life-threatening adverse event.\n\n{0}\n\n"
      "Instruction: answer TRUE if the report describes a serious or "
      "life-threatening adverse event, FALSE otherwise.")

REACTION = ("Does the medical report in DOCUMENT {0} describe the "
            "reaction in DOCUMENT {1} as something the patient "
            "experienced?")

# BIO-6 only: a second question over the same terms table, joined
# under alias m2, so the star shape has two distinct stages.
REACTION_SEVERE = ("Does the medical report in DOCUMENT {0} describe "
                   "the reaction in DOCUMENT {1} as serious or life "
                   "threatening for the patient?")

F11 = ("Judge strictly from the claim above whether it asserts "
       "something about a person, rather than an organization, place, "
       "or event.\n\n{0}\n\nInstruction: answer TRUE if the claim "
       "asserts something about a person, FALSE otherwise.")

F12 = ("Judge strictly from the claim above whether it contains a "
       "specific date or year.\n\n{0}\n\nInstruction: answer TRUE if "
       "the claim contains a specific date or year, FALSE otherwise.")

F14 = ("Judge strictly from the claim above whether it references a "
       "specific place (a city, country, or other named location).\n\n"
       "{0}\n\nInstruction: answer TRUE if the claim references a "
       "specific place, FALSE otherwise.")

SUPPORT = ("Does the Wikipedia passage in DOCUMENT {1} support the "
           "claim in DOCUMENT {0}?")

# FEV-7 only: a second question over the same evidence table, joined
# under alias e2.
REFUTE = ("Does the Wikipedia passage in DOCUMENT {1} refute or "
          "contradict the claim in DOCUMENT {0}?")

# FEV-5 only: filters the evidence side of a join, not just the
# anchor. Mirrors F11's "about a person" judgment so the join pairs
# claim and passage on the same axis, independently filtered.
F13 = ("Judge strictly from the Wikipedia passage above whether it "
       "primarily describes a specific person (their life, actions, "
       "or role), rather than an organization, place, or event.\n\n"
       "{0}\n\nInstruction: answer TRUE if the passage primarily "
       "describes a specific person, FALSE otherwise.")

# LePaRD predicates: "excerpt" for destination_context throughout,
# to avoid colliding with this dataset's own use of "passage" for
# the quoted/cited text.
LEP1 = ("Judge strictly from the excerpt above whether it argues that "
        "the cited case's reasoning does not apply here.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt argues the cited "
        "case's reasoning does not apply here, FALSE otherwise.")

LEP2 = ("Judge strictly from the excerpt above whether it discusses a "
        "procedural or jurisdictional issue.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt discusses a procedural or "
        "jurisdictional issue, FALSE otherwise.")

LEP3 = ("Judge strictly from the excerpt above whether it treats the "
        "cited passage as binding precedent.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt treats the cited passage as "
        "binding precedent, FALSE otherwise.")

LEP4 = ("Judge strictly from the excerpt above whether it cites the "
        "passage to support a conclusion about a party's liability or "
        "guilt.\n\n{0}\n\nInstruction: answer TRUE if the excerpt "
        "cites the passage to support a conclusion about a party's "
        "liability or guilt, FALSE otherwise.")

LEP5 = ("Judge strictly from the excerpt above whether it acknowledges "
        "disagreement between courts on the issue.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt acknowledges "
        "disagreement between courts on the issue, FALSE otherwise.")

# LEP-7 only: filters the passage side of the self-join, not just the
# excerpt (anchor) side.
LEPS1 = ("Judge strictly from the passage above whether it states a "
         "general legal rule.\n\n{0}\n\nInstruction: answer TRUE if "
         "the passage states a general legal rule, FALSE otherwise.")

# The LEP-2..LEP-7 join predicate. Ground truth comes from the
# dataset's own passage_id, not a judge pass.
LEPJOIN = ("Is the passage in DOCUMENT {1} cited by the legal excerpt "
           "in DOCUMENT {0}?")


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

    # IMDB-9/IMDB-10: 3-join chain r1-a1-r2-a2. Two reviews that
    # discuss the same aspect; what other aspect does the second
    # review feel positively about?
    def imdb9():
        r1 = sess.docs("reviews").alias("r1")
        a1 = sess.docs("aspects").alias("a1")
        r2 = sess.docs("reviews").alias("r2")
        a2 = sess.docs("aspects").alias("a2")
        return (r1
                .ai_join(a1, quail.prompt(DISCUSS_ASPECT,
                                          quail.col("r1.body"),
                                          quail.col("a1.aspect")))
                .ai_join(r2, quail.prompt(DISCUSS_ASPECT,
                                          quail.col("r2.body"),
                                          quail.col("a1.aspect")))
                .ai_join(a2, quail.prompt(ASPECT_SENTIMENT,
                                          quail.col("r2.body"),
                                          quail.col("a2.aspect")))
                .select("r1.id", "a1.id", "r2.id", "a2.id"))
    q["IMDB-9"] = ("3J chain r1-a1-r2-a2: two reviews discuss the "
                   "same aspect, second review positive about another",
                   imdb9)

    def imdb10():
        r1 = sess.docs("reviews").alias("r1").ai_filter(
            quail.prompt(F1, quail.col("r1.body")))
        a1 = sess.docs("aspects").alias("a1")
        r2 = sess.docs("reviews").alias("r2")
        a2 = sess.docs("aspects").alias("a2")
        return (r1
                .ai_join(a1, quail.prompt(DISCUSS_ASPECT,
                                          quail.col("r1.body"),
                                          quail.col("a1.aspect")))
                .ai_join(r2, quail.prompt(DISCUSS_ASPECT,
                                          quail.col("r2.body"),
                                          quail.col("a1.aspect")))
                .ai_join(a2, quail.prompt(ASPECT_SENTIMENT,
                                          quail.col("r2.body"),
                                          quail.col("a2.aspect")))
                .select("r1.id", "a1.id", "r2.id", "a2.id"))
    q["IMDB-10"] = ("F1 -> 3J chain r1-a1-r2-a2", imdb10)

    # IMDB-8: star shape - A joins B and A joins C, same anchor
    # throughout, barrier between stages but no anchor switch.
    q["IMDB-8"] = ("2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
                   "(ASPECT_SENTIMENT), reviews x aspects x aspects",
                   make("reviews", "r", "body", [],
                       [("aspects", "a", "aspect", DISCUSS_ASPECT),
                        ("aspects", "a2", "aspect", ASPECT_SENTIMENT)],
                       ["r.id", "a.id", "a2.id"]))

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

    # BIO-7/BIO-8: 3-join chain r1-m1-r2-m2. Report r1 experienced
    # reaction m1; reaction m1 was severe in report r2; report r2
    # also experienced a different reaction m2.
    def bio7():
        r1 = sess.docs("reports").alias("r1")
        m1 = sess.docs("terms").alias("m1")
        r2 = sess.docs("reports").alias("r2")
        m2 = sess.docs("terms").alias("m2")
        return (r1
                .ai_join(m1, quail.prompt(REACTION,
                                          quail.col("r1.report"),
                                          quail.col("m1.term")))
                .ai_join(r2, quail.prompt(REACTION_SEVERE,
                                          quail.col("r2.report"),
                                          quail.col("m1.term")))
                .ai_join(m2, quail.prompt(REACTION,
                                          quail.col("r2.report"),
                                          quail.col("m2.term")))
                .select("r1.id", "m1.id", "r2.id", "m2.id"))
    q["BIO-7"] = ("3J chain r1-m1-r2-m2: shared reaction, severe in "
                  "second report, second report has another reaction",
                  bio7)

    def bio8():
        r1 = sess.docs("reports").alias("r1").ai_filter(
            quail.prompt(F7, quail.col("r1.report")))
        m1 = sess.docs("terms").alias("m1")
        r2 = sess.docs("reports").alias("r2")
        m2 = sess.docs("terms").alias("m2")
        return (r1
                .ai_join(m1, quail.prompt(REACTION,
                                          quail.col("r1.report"),
                                          quail.col("m1.term")))
                .ai_join(r2, quail.prompt(REACTION_SEVERE,
                                          quail.col("r2.report"),
                                          quail.col("m1.term")))
                .ai_join(m2, quail.prompt(REACTION,
                                          quail.col("r2.report"),
                                          quail.col("m2.term")))
                .select("r1.id", "m1.id", "r2.id", "m2.id"))
    q["BIO-8"] = ("F7 -> 3J chain r1-m1-r2-m2", bio8)

    # BIO-6: star shape, both joins anchored on reports, both over
    # the full terms table under two aliases. IMDB-8's counterpart.
    q["BIO-6"] = ("2J, same anchor: J1 (REACTION) -> J2 "
                 "(REACTION_SEVERE), reports x terms x terms", make(
        "reports", "r", "report", [],
        [("terms", "m", "term", REACTION),
         ("terms", "m2", "term", REACTION_SEVERE)],
        ["r.id", "m.id", "m2.id"]))

    # FEVER: filter alone, join alone, filter chain to depth 2 only
    # (depth 3 yields 0 rows), plus two-sided FEV-5/FEV-6 pushdown.
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

    # FEV-8/FEV-9: 3-join chain c1-e1-c2-e2. Evidence e1 supports
    # claim c1 but refutes claim c2; claim c2 is supported by
    # different evidence e2.
    def fev8():
        c1 = sess.docs("claims").alias("c1")
        e1 = sess.docs("evidence").alias("e1")
        c2 = sess.docs("claims").alias("c2")
        e2 = sess.docs("evidence").alias("e2")
        return (c1
                .ai_join(e1, quail.prompt(SUPPORT,
                                          quail.col("c1.claim"),
                                          quail.col("e1.text")))
                .ai_join(c2, quail.prompt(REFUTE,
                                          quail.col("c2.claim"),
                                          quail.col("e1.text")))
                .ai_join(e2, quail.prompt(SUPPORT,
                                          quail.col("c2.claim"),
                                          quail.col("e2.text")))
                .select("c1.id", "e1.id", "c2.id", "e2.id"))
    q["FEV-8"] = ("3J chain c1-e1-c2-e2: evidence supports c1 but "
                  "refutes c2, c2 supported by different evidence",
                  fev8)

    def fev9():
        c1 = sess.docs("claims").alias("c1").ai_filter(
            quail.prompt(F11, quail.col("c1.claim")))
        e1 = sess.docs("evidence").alias("e1")
        c2 = sess.docs("claims").alias("c2")
        e2 = sess.docs("evidence").alias("e2")
        return (c1
                .ai_join(e1, quail.prompt(SUPPORT,
                                          quail.col("c1.claim"),
                                          quail.col("e1.text")))
                .ai_join(c2, quail.prompt(REFUTE,
                                          quail.col("c2.claim"),
                                          quail.col("e1.text")))
                .ai_join(e2, quail.prompt(SUPPORT,
                                          quail.col("c2.claim"),
                                          quail.col("e2.text")))
                .select("c1.id", "e1.id", "c2.id", "e2.id"))
    q["FEV-9"] = ("F11 -> 3J chain c1-e1-c2-e2", fev9)

    # FEV-7: star shape, both joins anchored on claims.
    q["FEV-7"] = ("2J, same anchor: J1 (SUPPORT) -> J2 (REFUTE), "
                 "claims x evidence x evidence", make(
        "claims", "c", "claim", [],
        [("evidence", "e", "text", SUPPORT),
         ("evidence", "e2", "text", REFUTE)],
        ["c.id", "e.id", "e2.id"]))

    # LePaRD uses two deduplicated projections of sampled citation pairs.
    q["LEP-1"] = ("filter: LEP1 (reasoning does not apply)", make(
        "citation_contexts", "d", "destination_context", [LEP1], [],
        ["d.id"]))
    q["LEP-2"] = ("join: citation contexts x cited passages", make(
        "citation_contexts", "d", "destination_context", [],
        [("citation_passages", "s", "passage_text", LEPJOIN)],
        ["d.id", "s.id"]))
    q["LEP-3"] = ("LEP1 -> join, dependent", make(
        "citation_contexts", "d", "destination_context", [LEP1],
        [("citation_passages", "s", "passage_text", LEPJOIN)],
        ["d.id", "s.id"]))
    q["LEP-4"] = ("LEP1 -> LEP2 -> join, 2 filters then 1 join", make(
        "citation_contexts", "d", "destination_context", [LEP1, LEP2],
        [("citation_passages", "s", "passage_text", LEPJOIN)],
        ["d.id", "s.id"]))
    q["LEP-5"] = ("LEP1 -> LEP2 -> LEP3 -> join, 3 filters then 1 join",
                  make("citation_contexts", "d", "destination_context",
                      [LEP1, LEP2, LEP3],
                      [("citation_passages", "s", "passage_text", LEPJOIN)],
                      ["d.id", "s.id"]))
    q["LEP-6"] = ("LEP1..LEP5 -> join, 5 filters then 1 join", make(
        "citation_contexts", "d", "destination_context",
        [LEP1, LEP2, LEP3, LEP4, LEP5],
        [("citation_passages", "s", "passage_text", LEPJOIN)],
        ["d.id", "s.id"]))

    q["LEP-7"] = ("2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
                  "LEPS1 on passages, each filtered before the join", make(
        "citation_contexts", "d", "destination_context", [LEP1, LEP2],
        [("citation_passages", "s", "passage_text", LEPJOIN, [LEPS1])],
        ["d.id", "s.id"]))
    # LEP-6 without its join: the deepest filter chain in the suite,
    # five stages of KV reuse with no join work mixed in.
    q["LEP-8"] = ("LEP1..LEP5, 5 filters, no join", make(
        "citation_contexts", "d", "destination_context",
        [LEP1, LEP2, LEP3, LEP4, LEP5], [], ["d.id"]))

    return q


# ----------------------------------------------------------- driver

def _artifact_stem(started, sf, lf, model):
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-quailb-sf{sf}-lf{lf}-{model}"


def run_suite(data_dir, sf=0.1, lf=1, gpus=1, only=None,
              out_path=None, model="qwen3-4b-fp8",
              accuracy=True, ground_truth_collection=None,
              ground_truth_workload=None,
              h100_usd_per_hour=3.9492, ground_truth_files=None,
              prediction=None, artifact_stem=None, execute=None):
    """Run all (or selected) QUAIL-B queries through the engine."""
    import quail
    from quail.bench.evaluate import (
        H100_PRICE_SOURCE,
        BenchmarkEvaluator,
        ModalVolumeFiles,
        add_query_metrics,
        corpus_identity,
        load_ground_truth,
        load_ground_truth_workload,
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
        if ground_truth_workload:
            truth = load_ground_truth_workload(
                result_files,
                scale_factor=sf,
                corpus_id=corpus["corpus_id"],
                corpus_full_hash=corpus["corpus_full_hash"],
                workload=ground_truth_workload,
            )
        else:
            truth = load_ground_truth(
                result_files, scale_factor=sf,
                corpus_id=corpus["corpus_id"],
                collection_id=ground_truth_collection)
        if truth.corpus_id != corpus["corpus_id"]:
            raise ValueError(
                f"benchmark corpus {corpus['corpus_id']} does not match "
                f"ground truth {truth.corpus_id}")
        evaluator = BenchmarkEvaluator(truth, corpus_rows)
    sess = quail.Session(EngineConfig(gpus=gpus, model=model))
    register_sets(sess, d)
    qdefs = queries(sess)
    if only is None:
        ids = list(qdefs)
    elif isinstance(only, (set, frozenset)):
        ids = [query_id for query_id in qdefs if query_id in only]
    else:
        ids = [query_id for query_id in only if query_id in qdefs]
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
            gpu="H100!",
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
        pass_name = "single"
        rows = []
        t_pass = time.time()
        for qid in ids:
            desc, build = qdefs[qid]
            print(f"[quailb] {qid}: {desc}", flush=True)
            try:
                query = build()
                res = query.run(_execute=execute)
                result_rows = res.count()
                row = dict(query=qid, desc=desc,
                           wall_s=res.report["wall_s"],
                           boot_s=res.report["boot_s"],
                           boot_kind=res.report.get("boot_kind"),
                           boot=res.report.get("boot"),
                           fresh_tokens=res.report["fresh_tokens"],
                           rows=result_rows,
                           peak_gib=res.report.get("peak_gib"),
                           stages=res.report["stages"])
                if evaluator is not None:
                    evaluation = evaluator.evaluate(query, res)
                    add_query_metrics(
                        row, evaluation, h100_usd_per_hour, gpus)
                    raw_path = f"{raw_root}/{pass_name}/{qid}.json"
                    answer_paths = {"filters": {}, "joins": {}}
                    for (alias, written_pos), table in \
                            res.answer_tables["filters"].items():
                        path = (
                            f"{raw_root}/{pass_name}/{qid}/answers/"
                            f"filter-{alias}-{written_pos}.parquet")
                        result_files.write_parquet(path, table)
                        answer_paths["filters"][
                            f"{alias}:{written_pos}"] = f"/results/{path}"
                    for written_pos, table in \
                            res.answer_tables["joins"].items():
                        path = (
                            f"{raw_root}/{pass_name}/{qid}/answers/"
                            f"join-{written_pos}.parquet")
                        result_files.write_parquet(path, table)
                        answer_paths["joins"][str(written_pos)] = \
                            f"/results/{path}"
                    result_files.write_json(raw_path, {
                        "run_id": run_id,
                        "pass": pass_name,
                        "query": qid,
                        "description": desc,
                        "columns": res.columns,
                        "result": {
                            "schema": str(res.schema),
                            "rows": result_rows,
                            "materialized": False,
                        },
                        "answer_tables": answer_paths,
                        "engine_report": res.report,
                        "accuracy": row["accuracy"],
                    })
                    row["raw_volume_path"] = f"/results/{raw_path}"
            except Exception as e:            # noqa: BLE001
                row = dict(query=qid, desc=desc,
                           error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc())
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
