"""QUAIL-B: thirty-two queries over five document sets.

The default sets are IMDB, BioDEX, FEVER, LePaRD, and SWE-Next agent
trace snapshots. Two optional PrivacyPolicies queries are also available.

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
CACHE_SCHEMA_VERSION = 9
LEPARD_POSITIVE_PAIRS = 5_000
AGENT_TRACE_DOCUMENTS = 17_718
AGENT_TRACE_TURN_INTERVAL = 5
AGENT_TRACE_MAX_TOKENS = 24_000
AGENT_TRACE_TOKENIZER = "Qwen/Qwen3-4B-FP8"
AGENT_TRACE_TOKENIZER_REVISION = (
    "96b30dc13593a244a5e59e84687309f53c375cfa"
)

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
    "TIGER-Lab/SWE-Next-SFT-Trajectories":
        "e378a60ddd7050fe9519a31a4d41d4872eeec6ac",
    "mukund/PrivacyPolicies":
        "8fd6abfc7ca99d1f95c7f3f3a5dd5ea0cf9b7deb",
}

# Base document counts at sf=1. LePaRD scales sampled citation pairs
# before it deduplicates the two document tables.
SETS = {
    "reviews": 50_000,
    "reports": 5_000,
    "claims": 5_000,
    "agent_traces": AGENT_TRACE_DOCUMENTS,
    "policies": 1_000_000,
}

ASPECTS = ["the acting", "the plot", "the directing", "the cinematography",
           "the soundtrack", "the pacing", "the ending", "the dialogue",
           "the special effects", "the character development",
           "the screenplay", "the editing"]

QUERY_ORDER = (
    *(f"IMDB-{i}" for i in range(1, 11)),
    *(f"BIO-{i}" for i in range(1, 4)),
    *(f"FEV-{i}" for i in range(1, 10)),
    *(f"LEP-{i}" for i in range(1, 9)),
    *(f"AGENT-{i}" for i in range(1, 3)),
)

QUERY_FAMILY_WORKLOADS = {
    "IMDB": "imdb",
    "BIO": "biodex",
    "FEV": "fever",
    "LEP": "lepard",
    "AGENT": "agent",
}


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


def _n_docs(name, sf):
    return max(8, int(SETS[name] * sf))


def _n_lepard_pairs(sf):
    return max(8, int(LEPARD_POSITIVE_PAIRS * sf))


def _n_agent_documents(sf):
    return min(
        AGENT_TRACE_DOCUMENTS,
        max(8, round(AGENT_TRACE_DOCUMENTS * sf)),
    )


def _agent_message_text(message) -> str:
    """Render one agent message in the stored trace format."""
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(
            content, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False)
    return f"[{str(message.get('role', '')).upper()}]\n{content}"


def _agent_snapshot_boundaries(messages) -> tuple[str, list[tuple[int, int]]]:
    """Render one trace and return every fifth completed turn."""
    targets = {}
    assistant_turn = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn % AGENT_TRACE_TURN_INTERVAL:
            continue
        end = index + 1
        if end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        targets[end] = assistant_turn

    pieces = []
    boundaries = []
    length = 0
    for message_number, message in enumerate(messages, start=1):
        piece = _agent_message_text(message)
        if pieces:
            length += 2
        pieces.append(piece)
        length += len(piece)
        if message_number in targets:
            boundaries.append((targets[message_number], length))
    return "\n\n".join(pieces), boundaries


def _agent_has_issue(messages) -> bool:
    """Return whether the trace contains a nonempty user issue."""
    return any(
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def _agent_trace_rows(trace_index, messages, tokenizer) -> list[dict]:
    """Build the eligible snapshots for one SWE-Next trace."""
    if not _agent_has_issue(messages):
        return []
    text, boundaries = _agent_snapshot_boundaries(messages)
    rows = []
    for turn_index, end in boundaries:
        snapshot = text[:end]
        token_count = len(tokenizer.encode(
            snapshot, add_special_tokens=False))
        if token_count > AGENT_TRACE_MAX_TOKENS:
            continue
        rows.append({
            "id": f"at{trace_index:04d}-t{turn_index:03d}",
            "trace": snapshot,
            "trajectory_id": f"at{trace_index:04d}",
            "turn_index": turn_index,
            "token_count": token_count,
        })
    return rows


def _agent_rows(n):
    """Read SWE-Next and select a nested sample of trace snapshots."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = load_dataset(
        "TIGER-Lab/SWE-Next-SFT-Trajectories",
        split="train",
        revision=SOURCE_REVISIONS["TIGER-Lab/SWE-Next-SFT-Trajectories"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER,
        revision=AGENT_TRACE_TOKENIZER_REVISION,
    )
    order = np.random.default_rng(DATA_SEED).permutation(len(source))
    rows = []
    for trace_index in order:
        rows.extend(_agent_trace_rows(
            int(trace_index), source[int(trace_index)]["messages"],
            tokenizer))
        if len(rows) >= n:
            return rows[:n]
    raise ValueError(
        f"SWE-Next produced {len(rows)} eligible snapshots, expected {n}")


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
    seen, claims = set(), []
    for split in ("v1.0/train/0000.parquet",
                  "v1.0/labelled_dev/0000.parquet"):
        f = hf_hub_download("fever/fever", split,
                            repo_type="dataset",
                            revision=SOURCE_REVISIONS["fever/fever"])
        rows = pq.read_table(f).to_pylist()
        for r in rows:
            if (r["id"] in seen
                    or r["label"] not in ("SUPPORTS", "REFUTES")
                    or not r["evidence_wiki_url"]):
                continue
            seen.add(r["id"])
            claims.append(r)
            if len(claims) >= n_claims:
                break
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
    return int.from_bytes(
        hashlib.blake2b(value, digest_size=16).digest(), "big")


def _sample_lepard_pairs(rows, passages, n):
    """Select a stable random sample of distinct known citation pairs."""
    passages = {
        str(key): str(value).strip()
        for key, value in passages.items()
        if value
    }
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
            prior_context, _ = selected_rows[key]
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
    for _, dest_id, passage_id in sorted(
            selected, key=lambda item: (-item[0], item[1], item[2])):
        context, passage_text = selected_rows[(dest_id, passage_id)]
        pairs.append((dest_id, passage_id, context, passage_text))
    return pairs


def _lepard_documents(pairs):
    """Deduplicate each document column after sampling citation pairs."""
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

    from huggingface_hub import hf_hub_download
    import pandas as pd

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
        csv_path,
        usecols=cols,
        chunksize=50_000,
        dtype={
            "dest_id": "string",
            "destination_context": "string",
            "passage_id": "string",
        },
    )
    rows = (
        row
        for chunk in chunks
        for row in chunk.loc[:, cols].itertuples(index=False, name=None)
    )
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
    pq.write_table(
        pa.Table.from_pylist(contexts, schema=context_schema), context_path)
    pq.write_table(
        pa.Table.from_pylist(passages, schema=passage_schema), passage_path)


def _build_policies(d, sf, force=False):
    """Build policies.parquet and scenarios.parquet, idempotent.

    The PrivacyPolicies corpus is ~1M documents and 48 GiB, so it may
    not be available on every machine. Skips silently when the source
    dataset is not installed.
    """
    pol_path = d / "policies.parquet"
    scen_path = d / "scenarios.parquet"
    if pol_path.exists() and scen_path.exists() and not force:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    n = _n_docs("policies", sf)
    try:
        f = hf_hub_download(
            "mukund/PrivacyPolicies",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["mukund/PrivacyPolicies"])
    except Exception:
        return
    t = pq.read_table(f, columns=["text"])
    texts = t.column("text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    texts = texts[:n]
    pq.write_table(pa.table({
        "id": [f"pp{i}" for i in range(len(texts))],
        "policy_text": texts,
    }), pol_path)
    pq.write_table(pa.table({
        "id": [f"sc{i}" for i in range(len(SCENARIOS))],
        "scenario": SCENARIOS,
    }), scen_path)


def _build_agent_traces(d, sf, force=False):
    """Build the SWE-Next cumulative trace snapshots."""
    path = d / "agent_traces.parquet"
    if path.exists() and not force:
        return
    rows = _agent_rows(_n_agent_documents(sf))
    schema = pa.schema([
        ("id", pa.string()),
        ("trace", pa.string()),
        ("trajectory_id", pa.string()),
        ("turn_index", pa.int32()),
        ("token_count", pa.int32()),
    ])
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=False,
    )


def build_sets(data_dir, sf, lf=1):
    """Build the benchmark tables as Parquet files, cached by sf.

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
            _build_agent_traces(d, sf)
            return d
        base_sources = {
            name: revision for name, revision in SOURCE_REVISIONS.items()
            if name != "TIGER-Lab/SWE-Next-SFT-Trajectories"
        }
        same_sources = (
            current
            and current.get("data_seed") == DATA_SEED
            and current.get("scale_factor") == sf
            and all(
                current.get("source_revisions", {}).get(name) == revision
                for name, revision in base_sources.items()
            )
        )
        other_tables = (
            "reviews", "aspects", "reports", "terms", "claims", "evidence"
        )
        if same_sources and all((d / f"{name}.parquet").exists()
                                for name in other_tables):
            _build_lepard(d, sf, force=True)
            _build_agent_traces(d, sf, force=True)
            marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
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
    terms = _vocab_table(bio, 1)
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
    _build_agent_traces(d, sf, force=True)

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
                "citation_passages", "agent_traces"):
        sess.register(name, DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col="id"))


def register_privacy_sets(sess, data_dir):
    """Register policies and scenarios tables for PRIV queries.

    Separate from register_sets so the privacy policy queries do not
    run in the default benchmark suite.
    """
    from quail.catalog import DocumentProvider
    d = Path(data_dir)
    for name in ("policies", "scenarios"):
        sess.register(name, DocumentProvider.from_parquet(
            str(d / f"{name}.parquet"), id_col="id"))


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

REACTION = ("Does the medical report in DOCUMENT {0} describe the "
            "reaction in DOCUMENT {1} as something the patient "
            "experienced?")

AGENT_RECOVERED = (
    "Judge strictly from the agent trace above whether the agent recovered "
    "after pursuing an approach that did not work. Recovery means the agent "
    "recognized or moved past the unsuccessful approach and then made useful "
    "progress with a different or corrected approach.\n\n{0}\n\n"
    "Instruction: answer TRUE if the trace shows the agent recovering after "
    "an unsuccessful approach, FALSE otherwise."
)

AGENT_IMPLEMENTED_FIX = (
    "Judge strictly from the agent trace above whether, by the end of the "
    "trace, the agent has implemented a plausible fix that directly addresses "
    "the reported issue. A fix must include a code or configuration change "
    "whose purpose is to correct the issue. Inspection, reproduction, tests "
    "without a fix, and unrelated edits do not count.\n\n{0}\n\nInstruction: "
    "answer TRUE if the agent has implemented a plausible fix that directly "
    "addresses the reported issue. Answer FALSE otherwise."
)

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

REFUTE = ("Does the Wikipedia passage in DOCUMENT {1} refute or "
          "contradict the claim in DOCUMENT {0}?")

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


# PrivacyPolicies predicates: user-language questions about user
# outcomes, not legal language about company practices. The vocabulary
# mismatch means keyword search, regex, and embeddings cannot solve
# these.
P_MSG = ("Judge strictly from the policy above whether, if a user sent "
         "a private message through this service, an employee of the "
         "company could read it.\n\n{0}\n\nInstruction: answer TRUE if "
         "an employee could read the user's private messages, FALSE "
         "otherwise.")

P_LOC = ("Judge strictly from the policy above whether this service "
         "would keep track of the user's physical location, even when "
         "the user is not actively using the service.\n\n{0}\n\n"
         "Instruction: answer TRUE if the service would track the "
         "user's location while they are not using it, FALSE otherwise.")

SCENARIO_MATCH = ("Based on the privacy policy in DOCUMENT {0}, could "
                  "the situation described in DOCUMENT {1} happen to a "
                  "user of this service?")

SCENARIOS = [
    # marketing
    "You stop using the app, but months later you start getting ads "
    "from companies you have never heard of, based on things you "
    "searched for while you were still using it.",
    "You buy a product once, and then you keep getting emails and push "
    "notifications about similar products, even after you unsubscribe "
    "from the mailing list.",
    "You notice that the ads you see on other websites change right "
    "after you browse this service, as if your activity here followed "
    "you around the internet.",
    "You create an account just to try the free version, and within a "
    "week you start getting phone calls from salespeople who know your "
    "name and what features you looked at.",
    "You fill out a survey on the app, and later a completely different "
    "company contacts you about the exact topics you mentioned in "
    "your answers.",
    "You use the app for a few weeks, then delete it, but you keep "
    "seeing ads for it on social media that reference things you did "
    "inside the app.",
    "You sign up with a throwaway email, but the app somehow starts "
    "showing you ads related to purchases you made with your main "
    "email at other stores.",
    "You mention a product in a chat on the service, and within hours "
    "you see targeted ads for that exact product on other platforms.",
    "You opt out of marketing emails, but the company still sends you "
    "promotional messages disguised as account updates or security "
    "alerts.",
    "You notice the app suggesting friends who are customers of a "
    "partner company, even though you never shared your contacts.",
    # third-party sharing
    "A data broker contacts you with an offer, and when you ask how "
    "they got your information, they name this service as the source.",
    "You apply for a loan and the lender already has a profile of "
    "your spending habits, built from data this service shared with "
    "a financial analytics company.",
    "Your health insurance premium goes up, and when you investigate "
    "you find the insurer received wellness data that you entered "
    "into this app.",
    "A background check company has records of your activity on this "
    "service, even though you never gave them permission to access it.",
    "You discover that a political campaign has your personal details "
    "and browsing habits, traced back to a data-sharing agreement "
    "with this service.",
    "Your employer uses a workplace analytics tool that has data about "
    "your personal usage of this service, shared without your knowledge.",
    "A research firm publishes a study that includes aggregated data "
    "about users like you, and you can identify yourself from the "
    "details even though names were removed.",
    "You find your profile information listed on a people-search "
    "website, and the data matches exactly what you entered into "
    "this service.",
    "A retailer you have never visited sends you a coupon by mail, "
    "using your home address and product preferences from this app.",
    "You learn that a foreign government obtained your account data "
    "through a third party that this service shared it with.",
    # law enforcement
    "Police show up with a warrant for records of your activity on "
    "the service, and the company hands over six months of your chat "
    "history without telling you first.",
    "A government agency requests your location data from the past "
    "year, and the company provides it without requiring a court "
    "order.",
    "You are involved in a lawsuit, and the opposing side introduces "
    "your private messages from this service as evidence, obtained "
    "through a subpoena the company complied with.",
    "An immigration agency accesses your travel-related searches and "
    "account activity through a bulk data request to the company.",
    "You find out that the company gave law enforcement real-time "
    "access to your location for an investigation you were never "
    "told about.",
    "A tax authority receives your transaction records from this "
    "service as part of a compliance program the company participates "
    "in voluntarily.",
    "You are detained at a border crossing, and the officers already "
    "have a printout of your recent activity on this service.",
    "A local police department uses facial recognition to match a "
    "photo from your profile on this service to surveillance footage.",
    "Your account is flagged and frozen after the company runs an "
    "automated scan and reports your content to a government agency.",
    "A foreign court orders the company to hand over your data, and "
    "the company complies even though you live in a different country.",
    # data retention
    "You delete your account, but a year later you discover the "
    "company still has your photos stored on its servers.",
    "You request a copy of your data and find that the company kept "
    "records of searches you made five years ago, long after you "
    "stopped using the service.",
    "You close your account and later reopen one with the same email, "
    "and all your old preferences and history are still there.",
    "You ask the company to delete your data, they confirm it is "
    "done, but a data breach months later reveals your old records "
    "were still in their backup systems.",
    "You find out the company keeps a permanent record of every "
    "version of your profile, including photos and bios you changed "
    "years ago.",
    "You cancel your subscription, but the company continues to "
    "store and analyze your usage patterns for its own research.",
    "Your messages to other users remain visible to those users "
    "even after you delete your account, with your name still "
    "attached.",
    "You discover that the company retains your payment information "
    "indefinitely, even after you remove your credit card from the "
    "account settings.",
    "You move to a country with stricter data laws and request "
    "deletion, but the company says your data is stored in a "
    "jurisdiction where they are not required to delete it.",
    "You stop paying for the premium tier, but the company keeps "
    "all the data you uploaded during your subscription period "
    "without any stated expiration date.",
    # tracking
    "You use the app only at home, but it builds a detailed map of "
    "every store and restaurant you visit, using your phone's "
    "location in the background.",
    "You turn off location services for the app, but it still "
    "figures out where you are by scanning nearby Wi-Fi networks "
    "and Bluetooth devices.",
    "You browse the service on your laptop, and later when you open "
    "the app on your phone, it knows exactly which pages you visited "
    "on the laptop.",
    "You visit a physical store, and the app sends you a notification "
    "about a sale at that store moments later, even though you never "
    "searched for it.",
    "You notice the app has a record of how long you spend on each "
    "screen, how fast you scroll, and exactly where you tap.",
    "You clear your browser cookies, but the service still recognizes "
    "you the next time you visit, using device fingerprinting or "
    "other tracking methods.",
    "You use a VPN to hide your location, but the app still shows "
    "you local content, suggesting it has another way to determine "
    "where you are.",
    "You create a second account under a different name, but the "
    "service links it to your original account within days.",
    "You lend your phone to a friend, and the app records their "
    "usage pattern as yours, mixing their browsing into your profile.",
    "You find out the app tracks which other apps are installed on "
    "your phone and uses that information to build a profile of "
    "your interests.",
    # content and communications
    "You send a private photo to one person through the service, "
    "and later find it was scanned and flagged by the company's "
    "automated content review system.",
    "You write a private note in the app that you never share, and "
    "later the company uses the text to train a language model.",
    "You have a private video call on the service, and you later "
    "discover the company recorded and stored a transcript of the "
    "conversation.",
    "You upload a document to the service for personal storage, and "
    "the company uses its contents to improve its search algorithm.",
    "You send an encrypted message, but the company can still read "
    "it because the encryption keys are stored on the company's "
    "servers.",
    "You post something to a small private group, and the company's "
    "moderation system shares it with an external review team in "
    "another country.",
    "You draft a message but never send it, and later discover the "
    "company saved the draft and analyzed its contents.",
    "You share a voice message with a friend, and the company "
    "converts it to text and adds it to your advertising profile.",
    "You delete a post you made, but the company keeps a copy and "
    "continues to use it for content recommendations.",
    "You set your profile to private, but the company still allows "
    "search engines to index your profile photo and display name.",
    # AI and automated decisions
    "You apply for a service upgrade, and an algorithm denies your "
    "request based on your usage patterns, with no explanation and "
    "no way to appeal.",
    "The app automatically adjusts the prices you see based on how "
    "much it predicts you are willing to pay, without telling you.",
    "You are banned from the platform by an automated system that "
    "flagged your content, and no human ever reviews your appeal.",
    "The service uses your photos to train a facial recognition "
    "model, and that model is later sold to a company you have "
    "never interacted with.",
    "An algorithm decides which customer service tier you belong to, "
    "so your support tickets are deprioritized compared to users the "
    "system considers more valuable.",
    "You are shown a different version of the terms of service than "
    "other users, tailored by an algorithm based on your likelihood "
    "of reading the full text.",
    "The service uses your data to build a creditworthiness score "
    "that other companies can purchase and use in their own lending "
    "decisions.",
    "An automated system flags your account as suspicious based on "
    "your browsing patterns, and your access is restricted without "
    "any notification.",
    "The app uses your purchase history to predict your political "
    "views and sells that prediction to a data analytics firm.",
    "You receive different search results than other users because "
    "an algorithm decided what it thinks you want to see, without "
    "telling you it is personalizing.",
    # security and breaches
    "Your password is leaked in a data breach, and you find out "
    "about it from a news article before the company ever contacts "
    "you.",
    "The company suffers a breach that exposes your home address, "
    "phone number, and payment history, and offers you only one "
    "year of credit monitoring.",
    "Your biometric data, like a fingerprint or face scan, is stolen "
    "in a breach, and unlike a password, you cannot change it.",
    "You learn that an employee at the company accessed your account "
    "and read your private messages out of personal curiosity.",
    "The company stores your password in a way that allows anyone who "
    "breaks into their database to read it directly.",
    "A contractor working for the company downloads a database backup "
    "containing your data and takes it with them when they leave.",
    "Your account is taken over by someone who called the company's "
    "support line and convinced them to reset your password.",
    "The company shares your data with a partner whose security "
    "practices are weaker, and that partner gets breached.",
    "You discover that the company has no way to tell you which "
    "employees accessed your data or when.",
    "A security researcher publicly discloses a vulnerability that "
    "exposed your data, and the company had known about it for "
    "months without fixing it.",
    # children and family
    "Your thirteen-year-old child signs up for the service by "
    "entering a fake birth date, and the company collects and sells "
    "their data just like an adult's.",
    "You share a family account with your children, and the company "
    "builds advertising profiles for each family member, including "
    "the minors.",
    "Your child's school requires this service for homework, and "
    "the company uses the child's usage data for purposes beyond "
    "education.",
    "You give the app permission to access your contacts, and it "
    "starts sending messages to your children's phone numbers "
    "inviting them to join.",
    "You find out the company kept detailed records of your child's "
    "online activity from when they were ten years old, and those "
    "records are still accessible years later.",
    "The service recommends content to your teenager based on a "
    "profile built from data collected before they were old enough "
    "to consent.",
    "Your family's smart home device shares your children's voice "
    "recordings with this service, which uses them for product "
    "development.",
    "You set up parental controls, but the company's data collection "
    "practices apply the same way to your child's account as to "
    "yours.",
    "A classmate's parent uses the app to look up information about "
    "your child, and the service provides it because your child's "
    "profile is not fully private by default.",
    "You discover that the company used your child's data to train "
    "an AI model, even though your child's account was flagged as "
    "belonging to a minor.",
    # financial and sensitive data
    "You link your bank account to the service for payments, and the "
    "company uses your transaction history to build a spending profile "
    "that it shares with advertisers.",
    "You enter your Social Security number for identity verification, "
    "and the company stores it indefinitely, even after verification "
    "is complete.",
    "The service infers your income level from your usage patterns "
    "and uses it to decide which subscription plans to show you.",
    "You authorize a one-time payment, but the company stores your "
    "full credit card details and later charges you for a renewal "
    "you did not agree to.",
    "Your medical information, entered into a wellness feature of "
    "the app, is shared with an insurance company as part of a data "
    "partnership.",
    "The service tracks which financial articles you read and sells "
    "that behavioral data to investment firms.",
    "You discover that the company has been collecting information "
    "about your race, religion, or sexual orientation from your "
    "profile and activity, and using it for ad targeting.",
    "You apply for a job through the service, and the employer sees "
    "a risk score calculated from your financial data on the "
    "platform.",
    "You connect a fitness tracker to the app, and it shares your "
    "health metrics with third parties without a separate consent "
    "step.",
    "The service combines your purchase history with public records "
    "to estimate your net worth, and makes that estimate available "
    "to its business partners.",
]
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


# ---------------------------------------------------------- queries

def queries(sess):
    """id -> (description, callable() -> Query). Fresh Query objects
    per call so each pass re-plans."""
    import quail

    def add_filter(query, template, column):
        return query.ai_filter(
            quail.prompt(template, column),
            selectivity=FILTER_SELECTIVITY_ESTIMATES.get(template))

    def add_join(query, partner, template, left, right):
        return query.ai_join(
            partner,
            quail.prompt(template, left, right),
            selectivity=JOIN_SELECTIVITY_ESTIMATES.get(template))

    def make(doc_table, doc_alias, doc_col, filters, joins, select):
        """filters: list of prompt templates applied in order to the
        base table. joins: list of (partner_table, partner_alias,
        partner_col, prompt_template[, partner_filters]) applied in
        order, each dependent on whatever survived the stages before
        it; partner_filters push filters onto the partner side before
        the join (the FEV-5/6 and LEP-7 two-sided shape)."""
        filter_templates = list(filters)
        for _partner, _alias, _column, _join, *rest in joins:
            filter_templates.extend(rest[0] if rest else ())
        has_estimates = (
            all(template in FILTER_SELECTIVITY_ESTIMATES
                for template in filter_templates)
            and all(join[3] in JOIN_SELECTIVITY_ESTIMATES for join in joins)
        )

        def build():
            qy = sess.docs(doc_table).alias(doc_alias)
            for tmpl in filters:
                qy = add_filter(
                    qy, tmpl, quail.col(f"{doc_alias}.{doc_col}"))
            for partner, palias, pcol, tmpl, *rest in joins:
                pq = sess.docs(partner).alias(palias)
                for pf in (rest[0] if rest else ()):
                    pq = add_filter(
                        pq, pf, quail.col(f"{palias}.{pcol}"))
                qy = add_join(
                    qy, pq, tmpl, quail.col(f"{doc_alias}.{doc_col}"),
                    quail.col(f"{palias}.{pcol}"))
            order = "by_cost" if has_estimates else "as_written"
            return qy.select(*select, order=order)
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
        qy = add_join(r1, a1, DISCUSS_ASPECT,
                      quail.col("r1.body"), quail.col("a1.aspect"))
        qy = add_join(qy, r2, DISCUSS_ASPECT,
                      quail.col("r2.body"), quail.col("a1.aspect"))
        qy = add_join(qy, a2, ASPECT_SENTIMENT,
                      quail.col("r2.body"), quail.col("a2.aspect"))
        return qy.select(
            "r1.id", "a1.id", "r2.id", "a2.id", order="by_cost")
    q["IMDB-9"] = ("3J chain r1-a1-r2-a2: two reviews discuss the "
                   "same aspect, second review positive about another",
                   imdb9)

    def imdb10():
        r1 = add_filter(
            sess.docs("reviews").alias("r1"), F1,
            quail.col("r1.body"))
        a1 = sess.docs("aspects").alias("a1")
        r2 = sess.docs("reviews").alias("r2")
        a2 = sess.docs("aspects").alias("a2")
        qy = add_join(r1, a1, DISCUSS_ASPECT,
                      quail.col("r1.body"), quail.col("a1.aspect"))
        qy = add_join(qy, r2, DISCUSS_ASPECT,
                      quail.col("r2.body"), quail.col("a1.aspect"))
        qy = add_join(qy, a2, ASPECT_SENTIMENT,
                      quail.col("r2.body"), quail.col("a2.aspect"))
        return qy.select(
            "r1.id", "a1.id", "r2.id", "a2.id", order="by_cost")
    q["IMDB-10"] = ("F1 -> 3J chain r1-a1-r2-a2", imdb10)

    # IMDB-8: star shape - A joins B and A joins C, same anchor
    # throughout, barrier between stages but no anchor switch.
    q["IMDB-8"] = ("2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
                   "(ASPECT_SENTIMENT), reviews x aspects x aspects",
                   make("reviews", "r", "body", [],
                       [("aspects", "a", "aspect", DISCUSS_ASPECT),
                        ("aspects", "a2", "aspect", ASPECT_SENTIMENT)],
                       ["r.id", "a.id", "a2.id"]))

    # BioDEX: filter, join, filter->join on long medical reports.
    # Deeper chains and multi-join shapes (star, 3J) are covered by
    # the IMDB queries; BioDEX adds long-document behavior.
    q["BIO-1"] = ("filter: F7 (female patient)", make(
        "reports", "r", "report", [F7], [], ["r.id"]))
    q["BIO-2"] = ("join: J1 (reports x terms)", make(
        "reports", "r", "report", [],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))
    q["BIO-3"] = ("F7 -> J1, dependent", make(
        "reports", "r", "report", [F7],
        [("terms", "m", "term", REACTION)], ["r.id", "m.id"]))

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
        qy = add_join(c1, e1, SUPPORT,
                      quail.col("c1.claim"), quail.col("e1.text"))
        qy = add_join(qy, c2, REFUTE,
                      quail.col("c2.claim"), quail.col("e1.text"))
        qy = add_join(qy, e2, SUPPORT,
                      quail.col("c2.claim"), quail.col("e2.text"))
        return qy.select(
            "c1.id", "e1.id", "c2.id", "e2.id", order="by_cost")
    q["FEV-8"] = ("3J chain c1-e1-c2-e2: evidence supports c1 but "
                  "refutes c2, c2 supported by different evidence",
                  fev8)

    def fev9():
        c1 = add_filter(
            sess.docs("claims").alias("c1"), F11,
            quail.col("c1.claim"))
        e1 = add_filter(
            sess.docs("evidence").alias("e1"), F13,
            quail.col("e1.text"))
        c2 = add_filter(
            sess.docs("claims").alias("c2"), F11,
            quail.col("c2.claim"))
        e2 = add_filter(
            sess.docs("evidence").alias("e2"), F13,
            quail.col("e2.text"))
        qy = add_join(c1, e1, SUPPORT,
                      quail.col("c1.claim"), quail.col("e1.text"))
        qy = add_join(qy, c2, REFUTE,
                      quail.col("c2.claim"), quail.col("e1.text"))
        qy = add_join(qy, e2, SUPPORT,
                      quail.col("c2.claim"), quail.col("e2.text"))
        return qy.select(
            "c1.id", "e1.id", "c2.id", "e2.id", order="by_cost")
    q["FEV-9"] = ("4F + 3J: F11 on c1 and c2, F13 on e1 and e2, "
                  "then the c1-e1-c2-e2 join chain", fev9)

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

    # SWE-Next: semantic filters over cumulative agent trace snapshots.
    q["AGENT-1"] = ("filter: recovered after an unsuccessful approach", make(
        "agent_traces", "t", "trace", [AGENT_RECOVERED], [], ["t.id"]))
    q["AGENT-2"] = ("filter: implemented a plausible fix", make(
        "agent_traces", "t", "trace", [AGENT_IMPLEMENTED_FIX], [],
        ["t.id"]))

    # PrivacyPolicies: conditional on the corpus being available.
    if "policies" in sess.catalog:
        q["PRIV-1"] = ("2 filters: P_MSG + P_LOC", make(
            "policies", "p", "policy_text", [P_MSG, P_LOC], [],
            ["p.id"]))
        q["PRIV-2"] = ("2 filters + 1 join: P_MSG + P_LOC -> scenarios",
                       make(
            "policies", "p", "policy_text", [P_MSG, P_LOC],
            [("scenarios", "s", "scenario", SCENARIO_MATCH)],
            ["p.id", "s.id"]))

    return q


# ----------------------------------------------------------- driver

def _artifact_stem(started, sf, lf, model, backend="quail"):
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-quailb-sf{sf}-lf{lf}-{model}-{backend}"


def run_suite(data_dir, sf=0.1, lf=1, gpus=1, only=None,
              out_path=None, model="qwen3-4b-fp8",
              backend="quail",
              accuracy=True, ground_truth_collection=None,
              ground_truth_workload=None,
              h100_usd_per_hour=3.9492, ground_truth_files=None,
              prediction=None, artifact_stem=None,
              compute_provider=None):
    """Run all (or selected) QUAIL-B queries through the engine."""
    import quail
    from quail.bench.evaluate import (
        H100_PRICE_SOURCE,
        BenchmarkEvaluator,
        ModalVolumeFiles,
        add_prefix_metrics,
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
    sess = quail.Session(EngineConfig(
        gpus=gpus,
        model=model,
        backend=backend,
    ), compute_provider=compute_provider)
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
        started, sf, lf, model, backend)
    run_id = (f"qb_{started.strftime('%Y%m%dT%H%M%SZ')}_"
              f"{uuid.uuid4().hex[:8]}")
    raw_root = f"benchmarks/quailb/runs/{run_id}"
    aggregate_volume_path = f"{raw_root}/{artifact_stem}.json"
    suite = dict(
        run_id=run_id,
        artifact_stem=artifact_stem,
        started_at=started.isoformat(),
        prediction=prediction,
        sf=sf, lf=lf, gpus=gpus, model=model, backend=backend,
        corpus_id=corpus["corpus_id"],
        selectivity_estimates=dict(
            source_collection=SELECTIVITY_ESTIMATE_COLLECTION,
            source_corpus=SELECTIVITY_ESTIMATE_CORPUS,
            source_scale_factor=SELECTIVITY_ESTIMATE_SCALE_FACTOR,
            method=("TRUE labels divided by all labels, fixed across "
                    "scale factors")),
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
            regret_tokens=("per document KV regret: fresh tokens spent "
                           "recomputing a document's own prefix after an "
                           "earlier request of the query computed it"),
            shared_prefix_tokens=("tokens of the scanned documents that "
                                  "are a prefix another scanned document "
                                  "also has, across aliases of one column "
                                  "as well as within one; an execution "
                                  "that computes each distinct prefix "
                                  "once never computes them"),
            cross_row_cached_tokens=("cached tokens inside a document's "
                                     "own tokens that another document's "
                                     "request computed; cached preamble, "
                                     "label, question, or block rounding "
                                     "tokens do not count; null when the "
                                     "run did not record it"),
            regret_distinct_tokens=("distinct prefix KV regret: "
                                    "regret_tokens plus "
                                    "shared_prefix_tokens minus "
                                    "cross_row_cached_tokens"),
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
                 reference_model=truth.reference_model,
                 workload=ground_truth_workload)
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
                res = query.run()
                result_rows = res.count()
                row = dict(query=qid, desc=desc,
                           backend=backend,
                           wall_s=res.report["wall_s"],
                           boot_s=res.report["boot_s"],
                           boot_kind=res.report.get("boot_kind"),
                           boot=res.report.get("boot"),
                           fresh_tokens=res.report["fresh_tokens"],
                           cached_tokens=res.report.get("cached_tokens"),
                           regret_tokens=res.report.get("regret_tokens"),
                           rows=result_rows,
                           peak_gib=res.report.get("peak_gib"),
                           stages=res.report["stages"],
                           backend_metrics=res.report.get(
                               "backend_metrics"
                           ))
                add_prefix_metrics(row, query, res.report)
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
        "--backend",
        default="quail",
        choices=(
            "quail",
            "stock_vllm",
            "pipelined_vllm",
            "pipelined_sglang",
        ),
    )
    ap.add_argument(
        "--accuracy", action=argparse.BooleanOptionalAction, default=True,
        help="compare answers and output rows with the Modal ground truth")
    ap.add_argument("--ground-truth-collection", default=None,
                    help="collection id; default is the one matching the corpus")
    ap.add_argument("--ground-truth-workload", default=None,
                    help="load labels for one workload from the current corpus")
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
        started, args.sf, args.lf, args.model, args.backend)
    out = args.out or f"results/benchmark/{artifact_stem}.json"
    suite = run_suite(
        args.data_dir, sf=args.sf, lf=args.lf, gpus=args.gpus,
        only=only, out_path=out, model=args.model,
        backend=args.backend,
        accuracy=args.accuracy,
        ground_truth_collection=args.ground_truth_collection,
        ground_truth_workload=args.ground_truth_workload,
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
