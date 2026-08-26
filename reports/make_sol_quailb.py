"""Speed of light for every QUAIL-B query, on Qwen3-4B and Qwen3-32B.

The least time each query can take on one H100! request. Three things cost
time and nothing else is counted:

  1. the dense projections - 2 FLOPs per parameter per token
  2. attention - 4 * n_q * d_head FLOPs per scored (query, key) pair,
     per layer
  3. moving bytes - the weights once per forward pass, KV once per
     token written, and KV again wherever a later stage reads it back

No measured or fitted performance constant appears in the time bound,
which is what makes the answer a floor: a run can approach it and can
never beat it. The dollar metric uses Modal's published H100! price.
The equations are plans/sol_model.md; this file is them, plus the
measurement of the three inputs they need.

KV reuse, the part that has to be right
---------------------------------------
Every document has a PREFIX: the shared preamble plus the document
text. Its KV is computed once and stays resident. A filter attaches
its question. A join attaches one anchor frame, then many tuple
suffixes. Each tuple suffix contains the partner label, partner
document, and answer cue. Suffix KV is computed, used once, and
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
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

import quail
from quail.bench import quailb as Q
from quail.bench.evaluate import H100_PRICE_SOURCE, H100_USD_PER_HOUR
from quail.logical import (ColumnRef, SHARED_PRE, bind_join_prompt,
                           bind_prompt)
from quail.planner.decide import _collect
from quail.planner.plan import EngineConfig, Refusal
from quail.runtime.coordinator import thin_survivors
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


# ---------------------------------------------------------------- 4
# Speed of light


@dataclass(frozen=True)
class SpeedOfLight:
    """The bound, with every term it was built from."""
    work: Work
    passes: int
    bytes_moved: float
    dense: float
    attention: float
    compute: float
    memory: float

    @property
    def seconds(self) -> float:
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
            f"SoL            {self.seconds:>18.4f} s ({self.bound_by} bound)",
        ])


def speed_of_light(work: Work, model: ModelSpec, device: DeviceSpec,
                   chunk_tokens: int) -> SpeedOfLight:
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
    return SpeedOfLight(work=work, passes=passes, bytes_moved=moved,
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
    "severe_terms.term": ("severe_terms", "term"),
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
JOIN_TEMPLATES = {c: getattr(Q, c) for c in (
    "DISCUSS_ASPECT", "ASPECT_SENTIMENT", "ASPECT_RELATED", "REACTION",
    "REACTION_SEVERE", "SUPPORT", "REFUTE", "LEPJOIN")}
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
# A predicate keeps one label set per template it has been judged
# under, so the volume holds several at once. The corpus's
# active_collection.json names the current collection, and that
# collection names one label set per predicate; anything else is a
# superseded run and must not be read.
ACTIVE = set(COLLECTION["label_sets"].values())
filter_answers = {}
join_answers = {}
predicate_meta = {}
template_codes = {}
for m in glob.glob(str(W / "allabels/label_sets/*/*/*/manifest.json")):
    if Path(m).parent.name not in ACTIVE:
        continue
    meta = json.load(open(m))["predicate"]
    rows = pq.read_table(Path(m).parent / "labels.parquet",
                         columns=["left_id", "right_id",
                                  "answer"]).to_pylist()
    code = meta["legacy_code"]
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
                    resident: bool) -> Work:
    labels_by_alias, tail = prompt_token_counts(prompt)
    partner_rows = list(itertools.product(
        *[survivors[alias] for alias in partners]))
    suffixes = [
        tail + sum(labels_by_alias[alias]["label"]
                   + aliases[alias]["tokens"][row]
                   for alias, row in zip(partners, partner_row))
        for partner_row in partner_rows
    ]
    frame = labels_by_alias[anchor]["frame"]
    work = Work()
    for row in survivors[anchor]:
        prefix = PRE + aliases[anchor]["tokens"][row]
        work = work + (ask(prefix, frame) if resident
                       else scan(prefix, frame))
        work = work + stream(prefix + frame, suffixes)
    return work


def runtime_anchor(join, compiled_anchor, aliases, survivors):
    prompt = join.predicate
    labels_by_alias, tail = prompt_token_counts(prompt)
    candidates = [arg.alias for arg in prompt.args]
    counts = {alias: len(survivors[alias]) for alias in candidates}
    means = {
        alias: (sum(aliases[alias]["tokens"][row]
                    for row in survivors[alias]) / counts[alias])
        if counts[alias] else 0.0
        for alias in candidates}
    maxes = {
        alias: max((aliases[alias]["tokens"][row]
                    for row in survivors[alias]), default=0)
        for alias in candidates}

    def need(anchor):
        return (PRE + maxes[anchor] + labels_by_alias[anchor]["frame"]
                + tail
                + sum(labels_by_alias[alias]["label"] + maxes[alias]
                      for alias in candidates if alias != anchor))

    def total(anchor):
        tuples = math.prod(counts.values())
        per_tuple = tail + sum(
            labels_by_alias[alias]["label"] + means[alias]
            for alias in candidates if alias != anchor)
        return (counts[anchor]
                * (means[anchor] + PRE
                   + labels_by_alias[anchor]["frame"])
                + tuples * per_tuple)

    chunk = min(CHUNK.values())
    feasible = [alias for alias in candidates
                if alias == compiled_anchor or need(alias) <= chunk]
    return min(feasible, key=total)


def simulate_query(query):
    plan = query.plan()
    if isinstance(plan, Refusal):
        raise ValueError(f"SoL query was refused: {plan.reasons}")
    scans, filters, joins = _collect(query.logical)
    aliases = {}
    for scan_node in scans:
        key = f"{scan_node.provider}.{scan_node.column}"
        ids = list(lengths[key])
        aliases[scan_node.alias] = {
            "column": key,
            "ids": ids,
            "tokens": [lengths[key][doc_id] for doc_id in ids],
        }
    survivors = {
        alias: list(range(len(data["ids"])))
        for alias, data in aliases.items()}
    work = Work()
    resident = set()
    filter_stages = []
    filter_evaluations = 0

    for node in plan.nodes_by_op("FilterChain"):
        alias = node["alias"]
        live = survivors[alias]
        for stage_index, stage in enumerate(node["stages"]):
            predicate = filters[alias][stage["written_pos"]]
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
    finished_full = []
    join_stages = []
    join_pair_evaluations = 0
    single_join_options = {}

    for node in plan.nodes:
        if node["op"] == "Barrier":
            thin_survivors(finished_full, survivors)
            continue
        if node["op"] != "JoinGroup":
            continue
        stage_defs = [joins[stage["written_pos"]]
                      for stage in node["stages"]]
        anchor = node["anchor"]
        if len(stage_defs) == 1 and stage_defs[0].anchor is None:
            anchor = runtime_anchor(
                stage_defs[0], anchor, aliases, survivors)

        for stage_index, join in enumerate(stage_defs):
            prompt = join.predicate
            code = prompt_code(prompt)
            stage_aliases = [arg.alias for arg in prompt.args]
            partners = [alias for alias in stage_aliases if alias != anchor]
            anchor_rows = list(survivors[anchor])
            partner_rows = list(itertools.product(
                *[survivors[alias] for alias in partners]))
            anchor_resident = anchor in resident or stage_index > 0

            if len(joins) == 1:
                single_join_options = {
                    candidate: join_stage_work(
                        candidate,
                        [alias for alias in stage_aliases
                         if alias != candidate],
                        aliases, survivors, prompt,
                        candidate in resident).tokens
                    for candidate in stage_aliases}

            stage_work = join_stage_work(
                anchor, partners, aliases, survivors, prompt,
                anchor_resident)
            work = work + stage_work
            rows = {}
            kept = []
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
                    kept.append(anchor_row)
            evaluated = len(anchor_rows) * len(partner_rows)
            join_pair_evaluations += evaluated
            stage_out = {
                "anchor": anchor,
                "partners": partners,
                "anchor_index": anchor_rows,
                "partner_index": partner_rows,
                "rows": rows,
            }
            finished_full.append(stage_out)
            survivors[anchor] = kept
            resident.add(anchor)
            join_stages.append({
                "code": code,
                "anchor": anchor,
                "partners": partners,
                "evaluated_pairs": evaluated,
                "passing_pairs": passing_pairs,
                "fresh_tokens": stage_work.tokens,
            })

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
    }


# ================================================================
# PART 3: every query, on both models
# ================================================================

session = quail.Session(
    EngineConfig(gpus=1, cpu_memory_gb=80, model="qwen3-4b-fp8"),
    tokenizer=encode)
Q.register_sets(session, W / "data" / TAG)
query_defs = Q.queries(session)
rows = {}
query_inputs = {}
for qid, (description, build) in query_defs.items():
    simulated = simulate_query(build())
    work = simulated.pop("work")
    rows[qid] = {
        "description": description,
        **simulated,
        "tuples": simulated["join_pair_evaluations"],
        "tokens": work.tokens, "pairs": work.pairs,
        "kv_written": work.kv_written, "kv_read": work.kv_read,
        "models": {}}
    query_inputs[qid] = {
        "filters": simulated["filter_stages"],
        "joins": simulated["join_stages"],
    }
    held = simulated["held_column"]
    rows[qid]["held_mean_doc_tokens"] = (
        sum(lengths[held].values()) / len(lengths[held]))
    for model in MODELS:
        s = speed_of_light(work, model, H100_SXM, CHUNK[model.name])
        rows[qid]["models"][model.name] = {
            "passes": s.passes, "bytes_moved": s.bytes_moved,
            "t_dense": s.dense, "t_attention": s.attention,
            "t_compute": s.compute, "t_memory": s.memory,
            "sol_s": s.seconds, "bound_by": s.bound_by,
            "cost_usd_per_query": (
                s.seconds * H100_USD_PER_HOUR / 3600),
            "documents_per_second": (
                simulated["input_document_rows"] / s.seconds
                if not simulated["join_stages"] and s.seconds else None),
            "document_pairs_per_second": (
                simulated["join_pair_evaluations"] / s.seconds
                if simulated["join_stages"] and s.seconds else None),
        }

hdr = (f"{'query':7} {'tokens':>11} {'pairs':>15} {'tuples':>10} "
       f"{'anchor':>15}  {'4B SoL':>9} {'att%':>5}  {'32B SoL':>9} "
       f"{'att%':>5} {'32B/4B':>7}")
print(hdr)
print("-" * len(hdr))
for qid, r in rows.items():
    a, b = r["models"]["qwen3-4b-fp8"], r["models"]["qwen3-32b-fp8"]
    print(f"{qid:7} {r['tokens']:>11,.0f} {r['pairs']:>15,.0f} "
          f"{r['tuples']:>10,} {str(r['anchor'] or '-'):>15}  "
          f"{a['sol_s']:>9.3f} {100 * a['t_attention'] / a['t_compute']:>5.1f}"
          f"  {b['sol_s']:>9.3f} "
          f"{100 * b['t_attention'] / b['t_compute']:>5.1f} "
          f"{b['sol_s'] / a['sol_s']:>7.2f}")


json.dump({
    "what": f"Speed of light for all {len(rows)} QUAIL-B queries at "
            f"sf={SF:g}, on "
            "Qwen3-4B-fp8 and Qwen3-32B-fp8, one H100! request each. "
            "A floor on "
            "wall time: no measured or fitted constant is used.",
    "method": "plans/sol_model.md, computed by reports/make_sol_quailb.py",
    "scale_factor": SF,
    "query_count": len(rows),
    "corpus_id": CORPUS_ID,
    "collection_id": COLLECTION_ID,
    "pricing": {
        "gpu": "H100!",
        "h100_usd_per_hour": H100_USD_PER_HOUR,
        "price_source": H100_PRICE_SOURCE,
        "method": "SoL seconds multiplied by the H100! price per second",
    },
    "metric_definitions": {
        "cost_usd_per_query": (
            "lower bound on GPU cost; SoL seconds times the H100! price"),
        "documents_per_second": (
            "upper bound for filter only queries; input document rows "
            "divided by SoL seconds"),
        "document_pairs_per_second": (
            "upper bound for join queries; evaluated pairs summed across "
            "join stages and divided by SoL seconds"),
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
        "query_stages": query_inputs},
    "queries": rows,
}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}\n"
      "put it on the volume:\n"
      f"  modal volume put quail-results {OUT} /sol/{OUT.name}")
