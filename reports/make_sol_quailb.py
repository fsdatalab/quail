"""Speed of light for every QUAIL-B query, on Qwen3-4B and Qwen3-32B.

The least time each query can take on one H100. Three things cost
time and nothing else is counted:

  1. the dense projections - 2 FLOPs per parameter per token
  2. attention - 4 * n_q * d_head FLOPs per scored (query, key) pair,
     per layer
  3. moving bytes - the weights once per forward pass, KV once per
     token written, and KV again wherever a later stage reads it back

No measured or fitted constant appears anywhere, which is what makes
the answer a floor: a run can approach it and can never beat it.
The equations are plans/sol_model.md; this file is them, plus the
measurement of the three inputs they need.

KV reuse, the part that has to be right
---------------------------------------
Every document has a PREFIX: the shared preamble plus the document
text. Its KV is computed once and stays resident. Anything attached
after it is a SUFFIX - a filter's question, or a join's partner
block plus question - and suffix KV is computed, used once, and
dropped (`executor/pack.py`: suffix KV is never cached).

So a document is read once however many questions get asked about
it. Three operations follow:

    scan()      compute a prefix and its first suffix, from nothing
    ask()       reuse a resident prefix, attach one more suffix
    stream()    reuse a resident prefix, attach many suffixes (a join)

`ask` and `stream` never charge for the document again; `scan` is
the only one that does.

Running it
----------
The corpora and the per-document ground-truth labels are raw data
and live on the quail-results volume, so pull them first. The
answers go back to the volume too:

    W=<workdir>; SF=0.1; C=<corpus id for that scale factor>
    G=/ground_truth/quailb/schema_v1
    mkdir -p $W/data $W/allabels
    modal volume get quail-results /quailb_data/sf$SF $W/data/
    modal volume get quail-results $G/label_sets $W/allabels/
    modal volume get quail-results $G/corpora/$C/active_collection.json \
        $W/active_collection.json
    GT=$(grep -o 'gt_[0-9a-f]*' $W/active_collection.json)
    modal volume get quail-results $G/collections/$GT/manifest.json \
        $W/collection_manifest.json
    uv run --with transformers --with pyarrow \
        python reports/make_sol_quailb.py $W $SF
    modal volume put quail-results $W/sol_quailb_sf$SF.json \
        /sol/sol_quailb_sf$SF.json

The scale factor defaults to 0.1. Each one has its own corpus and its
own label collection, so `modal volume ls quail-results $G/corpora`
gives the corpus ids; the run stops if the collection you pulled is
for a different scale factor than the one you asked for.

The report is reports/2026-08-26-sol-quailb.md.
"""
import collections
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from quail.bench import quailb as Q
from quail.logical import (ColumnRef, SHARED_PRE, bind_prompt,
                           join_anchor_note, join_label,
                           render_join_question)
from quail.specs import (H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8, DeviceSpec,
                         ModelSpec)

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
encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]
length = lambda t: len(encode(t))
PRE = length(SHARED_PRE)   # the engine preamble, 2 tokens

# The batch size each model's forward pass runs at: (2^31 - 1) over
# the widest projection, the fused kernels' 32-bit offset limit. It
# decides how often the weights are re-read, and it is a planner
# choice rather than a hardware property, so it is stated here.
CHUNK = {"qwen3-4b-fp8": 110_376, "qwen3-32b-fp8": 41_943}
MODELS = [QWEN3_4B_FP8, QWEN3_32B_FP8]


# ================================================================
# PART 1: the bound
# ================================================================

# The model, counted rather than looked up


def dense_params(model: ModelSpec) -> int:
    """Parameters every token passes through.

    Counted rather than quoted: `ModelSpec.params` is rounded to
    3.6e9 against Qwen3-4B's real 3,633,511,936, and that 0.93%
    lands straight on the largest term of the bound.

    Assumes the Qwen3 block: q/k/v/o projections with no bias, a
    gated MLP, two RMS norms per layer, and q/k head norms.
    Embeddings and the lm_head are left out: a token touches one
    embedding row rather than doing 2 FLOPs per parameter, and a
    filter reads logits at one position per evaluation.
    """
    h, dh = model.hidden, model.d_head
    attn = h * model.n_q * dh + 2 * h * model.n_kv * dh + model.n_q * dh * h
    mlp = 3 * h * model.intermediate
    norms = 2 * h + 2 * dh
    return (attn + mlp + norms) * model.layers + h


def flops_per_pair(model: ModelSpec) -> int:
    """Attention FLOPs for one (query token, key token) pair in one
    layer. The QK dot product runs over d_head dimensions, so
    2 * d_head; multiplying the weight into V costs another
    2 * d_head. Times n_q heads. At 4B: 4 * 32 * 128 = 16,384."""
    return 4 * model.n_q * model.d_head


def kv_bytes_per_token(model: ModelSpec) -> float:
    """One token's KV: a key and a value, per layer, per KV head.
    At 4B: 2 * 36 * 8 * 128 * 2 bytes = 147,456."""
    return model.kappa


# ---------------------------------------------------------------- 2
# What the GPU is asked to do


def triangle(n: float) -> float:
    """A causal sequence attending to itself: token 1 sees 1 key,
    token 2 sees 2, and so on. 1 + 2 + ... + n."""
    return n * (n + 1) / 2


@dataclass(frozen=True)
class Work:
    """Four counts. No seconds and no hardware in here."""
    tokens: float = 0.0       # tokens pushed through the forward pass
    pairs: float = 0.0        # scored (query, key) pairs, per layer
    kv_written: float = 0.0   # KV rows written
    kv_read: float = 0.0      # KV rows read back out of the arena

    def __add__(self, o: "Work") -> "Work":
        return Work(self.tokens + o.tokens, self.pairs + o.pairs,
                    self.kv_written + o.kv_written,
                    self.kv_read + o.kv_read)

    def __mul__(self, k: float) -> "Work":
        """The same work done k times."""
        return Work(self.tokens * k, self.pairs * k,
                    self.kv_written * k, self.kv_read * k)


def scan(prefix: float, suffix: float) -> Work:
    """Compute one document from nothing: [prefix | suffix] as one
    causal sequence. Every token attends to itself and everything
    before it, so the pairs are one triangle over the whole length.

    This is the only operation that pays for the document text.
    """
    n = prefix + suffix
    return Work(tokens=n, pairs=triangle(n), kv_written=n, kv_read=0.0)


def ask(prefix: float, suffix: float) -> Work:
    """Attach one more suffix to a prefix already in the arena.

    Only the suffix is computed. Each of its tokens attends to the
    whole resident prefix - a rectangle, `suffix * prefix` - and to
    itself and the suffix tokens before it - a triangle. The prefix
    is read back out of the arena once.

    The document is not recomputed and does not appear in `tokens`.
    That is KV rewind.
    """
    return Work(tokens=suffix,
                pairs=suffix * prefix + triangle(suffix),
                kv_written=suffix,
                kv_read=prefix)


def stream(prefix: float, suffixes) -> Work:
    """One resident prefix, many suffixes: a join anchor and its
    tuples.

    Suffixes are atomic and never attend to each other
    (`executor/pack.py`), so each is its own rectangle over the
    prefix plus its own triangle - exactly `ask`, repeated. The
    difference is the arena: the prefix is read back once for the
    whole stream, not once per tuple, because the tuples run
    consecutively against it.
    """
    tokens = pairs = 0.0
    for u in suffixes:
        tokens += u
        pairs += u * prefix + triangle(u)
    return Work(tokens=tokens, pairs=pairs, kv_written=tokens,
                kv_read=prefix)


# ---------------------------------------------------------------- 3
# Whole queries


def survivors(lengths, selectivity: float):
    """Which documents pass a filter.

    A selectivity says how many survive, not which. This keeps an
    evenly spaced slice of the length-sorted list, so the survivors
    carry the same length distribution as the pool they came from.

    A predicate correlated with document length breaks that: its
    survivors are longer than the slice, and the bound undercounts
    their token mass.
    """
    keep = round(len(lengths) * selectivity)
    if keep <= 0:
        return []
    if keep >= len(lengths):
        return list(lengths)
    order = sorted(lengths)
    step = len(order) / keep
    return [order[min(len(order) - 1, int(i * step))] for i in range(keep)]


def filter_chain(doc_tokens, preamble: int, questions, selectivities):
    """A chain of filters over one document set.

    The first stage scans every document. Every later stage rewinds
    to the end of the document and asks its own question of whatever
    survived so far.

    doc_tokens: one token count per document.
    questions: one question length per stage.
    selectivities: one per stage, the fraction of the documents
    entering that stage that pass it. The last one is never used.
    """
    live = list(doc_tokens)
    work = Work()
    for i, q in enumerate(questions):
        step = scan if i == 0 else ask
        for d in live:
            work = work + step(preamble + d, q)
        if i + 1 < len(questions):
            live = survivors(live, selectivities[i])
    return work


def join(anchor_tokens, partner_tokens, preamble: int, note: int,
         label: int, question: int, anchor_resident: bool) -> Work:
    """Every live anchor against every live partner.

    One side anchors: its KV is held and every tuple attends to it.
    The other streams: a copy of each of its documents rides in every
    tuple's suffix, alongside that block's label and the question.

    `note` is the anchor naming line, written into each anchor's kept
    KV once for this stage. `anchor_resident` is True when a filter
    on the anchor side already computed the prefixes, which is every
    query here with a filter before its join; then the join adds the
    naming line and the tuples, and nothing else.
    """
    suffixes = [label + p + question for p in partner_tokens]
    work = Work()
    for a in anchor_tokens:
        prefix = preamble + a
        if anchor_resident:
            work = work + ask(prefix, note)
        else:
            work = work + scan(prefix, note)
        work = work + stream(prefix + note, suffixes)
    return work


def cheaper_anchor(left, right, preamble, note, label, question,
                   left_resident, right_resident):
    """Both orientations of a join, and the one the engine would run.

    The planner keeps whichever side is cheaper to hold and streams
    the other, so the bound has to make the same choice or it is not
    a bound on what runs. Anchoring the long side costs one prefix
    per document; anchoring the short side copies every long document
    into every tuple. On FEVER, where claims average 11 tokens and
    evidence 370, that is a 5.5x difference.

    Returns (work, "left" | "right", {orientation: tokens}).
    """
    if not left or not right:
        return Work(), "left", {"left": 0.0, "right": 0.0}
    a = join(left, right, preamble, note, label, question, left_resident)
    b = join(right, left, preamble, note, label, question, right_resident)
    pick = "left" if a.tokens <= b.tokens else "right"
    return ({"left": a, "right": b}[pick], pick,
            {"left": a.tokens, "right": b.tokens})


# ---------------------------------------------------------------- 4
# Seconds


@dataclass(frozen=True)
class Seconds:
    """The bound, with every term it was built from."""
    work: Work
    passes: int
    bytes_moved: float
    dense: float
    attention: float
    compute: float
    memory: float

    @property
    def sol(self) -> float:
        return max(self.compute, self.memory)

    @property
    def bound_by(self) -> str:
        return "compute" if self.compute >= self.memory else "memory"

    def explain(self) -> str:
        w = self.work
        return "\n".join([
            f"tokens         {w.tokens:>18,.0f}",
            f"pairs          {w.pairs:>18,.0f}",
            f"kv written     {w.kv_written:>18,.0f}",
            f"kv read        {w.kv_read:>18,.0f}",
            f"forward passes {self.passes:>18,d}",
            f"bytes moved    {self.bytes_moved:>18,.0f}",
            f"T_dense        {self.dense:>18.4f} s",
            f"T_attention    {self.attention:>18.4f} s",
            f"T_compute      {self.compute:>18.4f} s",
            f"T_memory       {self.memory:>18.4f} s",
            f"SoL            {self.sol:>18.4f} s ({self.bound_by} bound)",
        ])


def seconds(work: Work, model: ModelSpec, device: DeviceSpec,
            chunk_tokens: int) -> Seconds:
    """Turn the four counts into a floor on wall time.

    `chunk_tokens` is the batch size the forward pass runs at. It
    decides how many times the weights are re-read, and what the
    engine picks for it is a planner decision, so it is an input
    here with no default.

    Compute and memory are combined with max, not added: the
    arithmetic units and the memory system run at once and a floor
    may assume they overlap perfectly. Inside compute the two terms
    are added, because the dense and attention kernels are separate
    launches on the same SMs.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    dense = 2.0 * dense_params(model) * work.tokens / device.peak_flops
    # attention runs in bf16 (FlashAttention-3 over bf16 KV), so it
    # prices against the bf16 peak, not the fp8 one
    attention = (flops_per_pair(model) * work.pairs * model.layers
                 / device.attn_flops)
    passes = math.ceil(work.tokens / chunk_tokens) if work.tokens else 0
    moved = (model.W_mem * passes
             + kv_bytes_per_token(model) * (work.kv_written + work.kv_read))
    return Seconds(work=work, passes=passes, bytes_moved=moved,
                   dense=dense, attention=attention,
                   compute=dense + attention,
                   memory=moved / device.hbm_bw)


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
    "citations.destination_context": ("citations", "destination_context"),
    "citations.passage_text": ("citations", "passage_text"),
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
# A predicate keeps one label set per template it has been judged
# under, so the volume holds several at once. The corpus's
# active_collection.json names the current collection, and that
# collection names one label set per predicate; anything else is a
# superseded run and must not be read.
ACTIVE = set(COLLECTION["label_sets"].values())
labels = {}
for m in glob.glob(str(W / "allabels/label_sets/*/*/*/manifest.json")):
    if Path(m).parent.name not in ACTIVE:
        continue
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
for qid, (column, codes, joined) in QUERIES.items():
    rec = {"document_column": column, "filters": stage_selectivities(column,
                                                                     codes)}
    if joined:
        partner_column, partner_codes, join_code = joined
        rec["partner_column"] = partner_column
        rec["partner_filters"] = stage_selectivities(partner_column,
                                                     partner_codes)
        rec["join"] = {"code": join_code, **join_prompt[join_code]}
    queries[qid] = rec


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


# ================================================================
# PART 3: every query, on both models
# ================================================================

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
        "document_column": rec["document_column"],
        "partner_column": rec.get("partner_column"),
        "anchor": anchor, "anchor_tokens_both_ways": both,
        # mean length of the documents whose KV is held: the context
        # every question and every tuple attends over, so the thing
        # that decides how much of the compute is attention
        "held_column": (rec["document_column"] if anchor != "right"
                        else rec["partner_column"]),
        "tokens": work.tokens, "pairs": work.pairs,
        "kv_written": work.kv_written, "kv_read": work.kv_read,
        "models": {}}
    held = rows[qid]["held_column"]
    rows[qid]["held_mean_doc_tokens"] = (
        sum(lengths[held].values()) / len(lengths[held]))
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
    "what": f"Speed of light for all 26 QUAIL-B queries at sf={SF:g}, on "
            "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100 each. A floor on "
            "wall time: no measured or fitted constant is used.",
    "method": "plans/sol_model.md, computed by reports/make_sol_quailb.py",
    "scale_factor": SF,
    "corpus_id": CORPUS_ID,
    "collection_id": COLLECTION_ID,
    "sources": {
        "corpora": f"/quailb_data/{TAG} on quail-results, seed 20260818",
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
}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}\n"
      "put it on the volume:\n"
      f"  modal volume put quail-results {OUT} /sol/{OUT.name}")
