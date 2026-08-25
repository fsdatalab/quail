"""Speed of light: the least time one query can take on one GPU.

Read this file top to bottom. It is meant to be checked by hand.

Three things cost time and nothing else is counted:

  1. the dense projections - 2 FLOPs per parameter per token
  2. attention - 4 * n_q * d_head FLOPs per scored (query, key) pair,
     per layer
  3. moving bytes - the weights once per forward pass, KV once per
     token written, and KV again wherever a later stage reads it back

Everything below is arithmetic over four counts and two spec structs.
No measured constant and no fitted efficiency factor appears anywhere,
which is what makes the answer a floor rather than a prediction: a run
can approach it and can never beat it.


KV reuse, the part that has to be right
---------------------------------------
Every document has a PREFIX: the engine's shared preamble plus the
document text. Its KV is computed once and stays resident in the
arena. Anything attached after it is a SUFFIX - a filter's question,
or a join's partner block plus question. Suffix KV is computed, used
for that one evaluation, and thrown away (`executor/pack.py`: suffix
KV is never cached).

So a document is read once no matter how many questions get asked
about it. A second filter does not rescan the document; it rewinds to
the end of the prefix and attaches a new suffix. That is why a
five-filter chain costs barely more than a one-filter chain, and it
is the single fact this file exists to price correctly.

Three operations cover every query in QUAIL-B:

    scan()      compute a prefix and its first suffix, from nothing
    ask()       reuse a resident prefix, attach one more suffix
    stream()    reuse a resident prefix, attach many suffixes (a join)

`ask` and `stream` never charge for the document again. `scan` is the
only one that does.
"""

import math
from dataclasses import dataclass

from quail.specs import DeviceSpec, ModelSpec


# ---------------------------------------------------------------- 1
# The model, counted rather than looked up


def dense_params(model: ModelSpec) -> int:
    """Parameters every token passes through.

    Counted from the model dimensions, not read off
    `ModelSpec.params`, which is rounded: 3.6e9 where Qwen3-4B's real
    non-embedding count is 3,633,511,936. That 0.93% lands straight
    on the largest term of the bound.

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

    That is an assumption, and it is visible here rather than hidden
    in a scaling factor. It is also wrong in a known direction: the
    QUAIL-B predicates prefer long documents, so the real survivors
    carry more tokens than this (up to 31% more after three filters).
    It changes the bound by under 0.1%, because a later stage adds
    only about 50 tokens per surviving document while the first scan
    already paid for every prefix.
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
