"""Speed of light: a floor on the wall time of one query.

SoL prices only the work the hardware cannot avoid - dense
projection FLOPs, attention pair FLOPs, weight reads, KV traffic -
and drops every source of loss. A run can approach it and never beat
it, so the gap between a measured wall and SoL is the whole of what
the engine could still win.

This module derives everything from the two spec structs and the
query, and borrows nothing from the planner. It does not import
`planner/budgets.py`, it reads no calibration constant, and no
efficiency factor appears anywhere in it. That is what makes it a
bound rather than an estimate: every number in it is a datasheet
figure, a model dimension, or a count of work the query cannot
avoid. Nothing in the planner reads it either.

The formulas and their assumptions are `plans/sol_model.md`. This
module implements the filter-chain case (sections 4-6). Joins are
section 8 of that document and are a proposal, not implemented here.

The split is deliberate: `filter_chain_workload` is query-shape
arithmetic with no hardware in it, and `bound` is hardware
arithmetic with no query shape in it. Each is checkable on its own.
"""

import math
from dataclasses import dataclass

from quail.specs import DeviceSpec, ModelSpec


def dense_params(model: ModelSpec) -> int:
    """Parameters every token passes through, counted from the model
    dimensions rather than read off `ModelSpec.params`.

    `params` is a rounded stand-in - 3.6e9 where Qwen3-4B's real
    non-embedding count is 3,633,511,936, which is 0.93% higher and
    moves T_dense by the same 0.93%. A bound cannot round its
    largest term, so it counts instead.

    The count assumes the Qwen3 block: q/k/v/o projections with no
    bias, a gated MLP (gate, up, down over `intermediate`), two RMS
    norms per layer, and q/k head norms. Embeddings are excluded on
    purpose - a token touches its own row and nothing else, so they
    are not 2 FLOPs per parameter per token. The lm_head is excluded
    for the same reason plus a second one: a filter reads logits at
    one position per evaluation, not at every token.
    """
    h, dh = model.hidden, model.d_head
    attn = h * model.n_q * dh + 2 * h * model.n_kv * dh + model.n_q * dh * h
    mlp = 3 * h * model.intermediate
    norms = 2 * h + 2 * dh          # 2 RMS norms, q norm, k norm
    return (attn + mlp + norms) * model.layers + h


@dataclass(frozen=True)
class Corpus:
    """The base document set, reduced to what the bound needs.

    A prefix is one document's retained span: the engine's shared
    preamble plus the document text. Its KV is computed once and
    survives every rewind, so it is the unit both the pair count and
    the KV read-back count are written in.

    The pair count is quadratic in length, so a total is not enough -
    the second moment has to come from the length distribution and
    cannot be recovered from the mean.
    """
    n_docs: int
    sum_prefix: float       # sum_i b_i
    sum_prefix_sq: float    # sum_i b_i^2

    @classmethod
    def from_doc_tokens(cls, doc_tokens, preamble_tokens: int = 0):
        b = [int(d) + preamble_tokens for d in doc_tokens]
        return cls(n_docs=len(b), sum_prefix=float(sum(b)),
                   sum_prefix_sq=float(sum(x * x for x in b)))


@dataclass(frozen=True)
class FilterStage:
    """One filter in the chain.

    `question_tokens` is everything the stage appends per live
    document: the engine's task instruction, the user's question, and
    the answer cue. `selectivity` is the fraction of the documents
    entering this stage that pass it.
    """
    question_tokens: int
    selectivity: float = 1.0


@dataclass(frozen=True)
class Workload:
    """What the query asks of the hardware, with no hardware in it.

    `tokens` is new tokens pushed through the forward pass.
    `pairs` is scored (query token, key token) pairs, per layer.
    `kv_read_tokens` is prefix tokens read back out of the arena by
    stages after the first; the write side is one write per new
    token and is already `tokens`.
    """
    tokens: float
    pairs: float
    kv_read_tokens: float
    per_stage: tuple = ()    # one (tokens, pairs, kv_read) per stage


@dataclass(frozen=True)
class SoLBound:
    """Seconds, plus every intermediate the arithmetic passed
    through, so each line can be checked against the derivation."""
    workload: Workload
    passes: int
    bytes_moved: float
    t_dense: float
    t_attention: float
    t_compute: float
    t_memory: float

    @property
    def seconds(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound_by(self) -> str:
        return "compute" if self.t_compute >= self.t_memory else "memory"

    def explain(self) -> str:
        w = self.workload
        return "\n".join([
            f"tokens          {w.tokens:>18,.0f}",
            f"attention pairs {w.pairs:>18,.0f}",
            f"kv read tokens  {w.kv_read_tokens:>18,.0f}",
            f"forward passes  {self.passes:>18,d}",
            f"bytes moved     {self.bytes_moved:>18,.0f}",
            f"T_dense         {self.t_dense:>18.4f} s",
            f"T_attention     {self.t_attention:>18.4f} s",
            f"T_compute       {self.t_compute:>18.4f} s",
            f"T_memory        {self.t_memory:>18.4f} s",
            f"SoL             {self.seconds:>18.4f} s "
            f"({self.bound_by} bound)",
        ])


def filter_chain_workload(corpus: Corpus, stages,
                          carry_question_kv: bool = False) -> Workload:
    """Tokens, pairs and KV read-backs for a chain of filters over
    one document set. Sections 4 and 5 of `plans/sol_model.md`.

    Stage 1 computes each document's whole sequence
    [preamble | document | question 1] from nothing, so it pays a
    full causal triangle. Every later stage rewinds to the end of the
    document and computes only its own question tokens: those attend
    to the retained prefix (a rectangle) and to each other (a small
    triangle).

    Selectivity thins the live set between stages. The surviving
    documents are taken to be a uniform random sample of the ones
    that entered, so a stage's surviving token mass is its survival
    factor times the whole corpus. Nothing enforces that - a filter
    that prefers long documents breaks it - and it is the one
    assumption here that the length distribution can violate.

    carry_question_kv counts each stage's question tokens as part of
    the prefix that later stages read back and attend over. The
    engine does not keep them (`executor/pack.py`: suffix KV is never
    cached), so False is what the engine does; True reproduces a hand
    derivation that folded question 1 into the document length.
    """
    stages = tuple(stages)
    if not stages:
        raise ValueError("a filter chain needs at least one stage")
    n, B = corpus.n_docs, corpus.sum_prefix
    tokens = pairs = kv_read = 0.0
    per_stage = []
    survival = 1.0          # fraction of the corpus entering this stage
    carried = 0             # question tokens folded into the prefix
    for s, st in enumerate(stages):
        q = st.question_tokens
        if s == 0:
            # full causal triangle over l_i = b_i + q, summed over
            # documents: sum l_i(l_i+1)/2 needs both moments.
            sum_l = B + n * q
            sum_l_sq = (corpus.sum_prefix_sq + 2 * q * B + n * q * q)
            st_tokens = sum_l
            st_pairs = (sum_l_sq + sum_l) / 2
            st_read = 0.0
        else:
            prefix = survival * (B + n * carried)
            st_tokens = survival * n * q
            st_pairs = q * prefix + survival * n * q * (q + 1) / 2
            st_read = prefix
        tokens += st_tokens
        pairs += st_pairs
        kv_read += st_read
        per_stage.append((st_tokens, st_pairs, st_read))
        survival *= st.selectivity
        if carry_question_kv:
            carried += q
    return Workload(tokens=tokens, pairs=pairs, kv_read_tokens=kv_read,
                    per_stage=tuple(per_stage))


def bound(model: ModelSpec, device: DeviceSpec, workload: Workload,
          chunk_tokens: int) -> SoLBound:
    """Seconds for a workload on one (model, device). Section 6 of
    `plans/sol_model.md`.

    Compute and memory are combined with max, not added: the two
    engines run at once and a floor may assume they overlap
    perfectly. Within compute the dense and attention terms are
    added, because they are separate kernels on the same SMs.

    chunk_tokens is the batch size the forward pass runs at, and it
    is an input with no default. It sets how many times the weights
    are re-read, so a wrong value moves the answer, and the value the
    engine happens to use is a planner decision this bound must not
    reach into. State it at the call site.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    t_dense = (2.0 * dense_params(model) * workload.tokens
               / device.peak_flops)
    pair_flops = 4.0 * model.n_q * model.d_head
    t_attention = (pair_flops * workload.pairs * model.layers
                   / device.attn_flops)
    passes = math.ceil(workload.tokens / chunk_tokens)
    moved = (model.W_mem * passes
             + model.kappa * (workload.tokens + workload.kv_read_tokens))
    return SoLBound(workload=workload, passes=passes, bytes_moved=moved,
                    t_dense=t_dense, t_attention=t_attention,
                    t_compute=t_dense + t_attention,
                    t_memory=moved / device.hbm_bw)


def filter_chain_sol(model: ModelSpec, device: DeviceSpec,
                     corpus: Corpus, stages, chunk_tokens: int,
                     carry_question_kv: bool = False) -> SoLBound:
    """The two halves together, for the common case."""
    return bound(model, device,
                 filter_chain_workload(corpus, stages, carry_question_kv),
                 chunk_tokens)
