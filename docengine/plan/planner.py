"""The declarative layer: describe the query, get a physical plan.

plan_query takes what the user knows (filters, corpus statistics,
model, device, GPU count, whether a KV store exists) and returns a
Plan choosing everything the measurements showed to matter:

- evaluation mode: chain (one living request per document) against
  separate requests; chain wins every measured regime with two or
  more filters, and single-filter queries have nothing to chain
- GPU layout: split the corpus across single-GPU workers (documents
  are independent, so every floor divides by the worker count) and
  split the model only when its weights do not fit one card
- admission budget: how many tokens of documents may be resident at
  once, kept under the KV pool so the strict memory invariant can
  never force an eviction
- pinning: only for the request path, and only when the corpus fits
  the pool - the 32B measurement showed pins losing to churn under
  overflow (54.5s against 48.1 naive), so overflow turns them off
- access: restore KV from a store instead of recomputing when the
  store's bandwidth beats the rate at which prefill would recreate
  them (kappa times the prefill rate, the measured break-even that
  lost two to one at 4B and wins several-fold at 32B)
- decode budget per stage: one token for models that keep the
  one-token answer contract; a decisive-token window for models that
  restate or chatter (the 32B probe)

A configuration that cannot run is refused, not planned around: per
PAPER.md Algorithm 2 (lines 3, 8, 16) plan_query returns a Refusal
naming the violated constraint when the weights need more cards than
exist, when the weights leave no KV pool, or when the pool cannot
hold the saturation working set. E12 exercises the gate.

Every rule cites a banked measurement or the calibrated model; the
tests assert the planner reproduces each measured winner. The
predicted makespan is Algorithm 1, priced by docengine/plan/cost.py,
where every calibrated constant lives.
"""

from dataclasses import dataclass, field

from docengine.configs import DeviceConfig, ModelConfig

from .cost import (BOOT_POOL_FRACTION, POOL_HEADROOM, predict_makespan,
                   t_in)


@dataclass(frozen=True)
class CorpusStats:
    n_docs: int
    total_tokens: int
    max_doc_tokens: int

    @property
    def mean_doc_tokens(self) -> float:
        return self.total_tokens / max(1, self.n_docs)


@dataclass(frozen=True)
class StoreSpec:
    """A persisted-KV store: measured read bandwidth, bytes/s."""
    read_bw: float
    warm: bool = False        # KV for this corpus already saved


@dataclass(frozen=True)
class Plan:
    mode: str                 # "chain" | "requests"
    workers: int              # data-parallel single-model workers
    tensor_parallel: int      # GPUs per worker (model split)
    shards: tuple             # doc-id tuples, one per worker
    budget_tokens: int        # resident-document admission cap
    pin: bool                 # request path only
    access: str               # "read" | "restore"
    stage_token_window: int   # decode budget per filter stage
    predicted_makespan_s: float
    remarks: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Refusal:
    """Algorithm 2's answer when the configuration is infeasible: the
    violated constraint, reported instead of a run (PAPER.md line 16)."""
    reasons: tuple            # human-readable, one per violated gate
    constraint: str           # "weights_need_more_cards" |
    #                           "pool_exhausted" | "pool_under_working_set"
    needed: float             # what the configuration requires
    available: float          # what the device offers, same unit
    unit: str                 # "cards" | "bytes" | "tokens"


def _pool_tokens(model: ModelConfig, device: DeviceConfig, tp: int) -> int:
    """KV-pool tokens per worker: free memory after weights, spread
    over the tensor-parallel group (each card holds 1/tp of weights
    and 1/tp of every token's KV). May be negative or zero: the
    refusal gates need the sign, so it is not clamped here."""
    free = device.M * BOOT_POOL_FRACTION * tp - model.W_mem
    return int(free / model.kappa)


def _balanced_shards(doc_tokens, workers):
    """Greedy balance by token count: makespan is the slowest shard."""
    order = sorted(range(len(doc_tokens)), key=lambda i: -doc_tokens[i])
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(i)
        loads[w] += doc_tokens[i]
    return tuple(tuple(sorted(s)) for s in shards), max(loads)


def plan_query(n_filters, doc_tokens, model: ModelConfig,
               device: DeviceConfig, gpus: int = 1,
               question_tokens: int = 46, preamble_tokens: int = 33,
               selectivity: float = 1.0, store: StoreSpec = None,
               one_token_answers: bool = True,
               saturation_width_docs: int = 1) -> "Plan | Refusal":
    """Plan the scan, or refuse it naming the violated constraint.

    saturation_width_docs is W* of PAPER.md Algorithm 2 line 8: the
    pool left after weights must hold W* times the per-document
    working set (largest document plus its first question; W* * f in
    the paper's bytes form) or the configuration is refused. The
    default 1 is the honest minimum - any execution needs one
    resident document - until W* is measured per device. E12
    exercises this gate.
    """
    corpus = CorpusStats(n_docs=len(doc_tokens),
                         total_tokens=int(sum(doc_tokens)),
                         max_doc_tokens=int(max(doc_tokens)))
    remarks = []

    # GPU layout: split the model only when one card cannot hold it;
    # otherwise every card is an independent worker on its own shard.
    tp = 1
    while model.W_mem > device.M * BOOT_POOL_FRACTION * tp:
        tp *= 2
    if tp > gpus:
        return Refusal(
            reasons=(f"weights need {tp} cards, {gpus} available",),
            constraint="weights_need_more_cards",
            needed=tp, available=gpus, unit="cards")
    if tp > 1:
        remarks.append(f"model needs {tp} cards; workers are {tp}-card")
    workers = max(1, gpus // tp)
    shards, worst_load = _balanced_shards(doc_tokens, workers)

    pool = _pool_tokens(model, device, tp)
    if pool <= 0:
        provisioned = device.M * BOOT_POOL_FRACTION * tp
        return Refusal(
            reasons=(f"no KV pool left after weights: {model.W_mem / 1e9:.1f}"
                     f" GB of weights against {provisioned / 1e9:.1f} GB"
                     f" provisioned on the {tp}-card group",),
            constraint="pool_exhausted",
            needed=model.W_mem, available=provisioned, unit="bytes")
    working_set = (corpus.max_doc_tokens + question_tokens) \
        * saturation_width_docs
    if pool < working_set:
        return Refusal(
            reasons=(f"pool holds {pool} tokens; saturation width "
                     f"{saturation_width_docs} needs {working_set} "
                     f"(largest document {corpus.max_doc_tokens} + first "
                     f"question {question_tokens} tokens)",),
            constraint="pool_under_working_set",
            needed=working_set, available=pool, unit="tokens")

    budget = int(min(pool * POOL_HEADROOM,
                     max(corpus.max_doc_tokens + question_tokens,
                         worst_load)))
    overflow = worst_load > pool
    if overflow:
        remarks.append("shard overflows the KV pool; pins off")

    mode = "chain" if n_filters >= 2 else "requests"
    # A pin keeps KV for a future consumer. One filter has no
    # future consumer, and chain mode keeps documents resident by
    # construction, so pins apply only to a multi-filter request plan
    # (kept for when chain mode is unavailable) and never under
    # overflow, where the 32B measurement showed them losing to churn.
    pin = mode == "requests" and n_filters >= 2 and not overflow

    access = "read"
    read_s = t_in(model, device, worst_load)
    if store is not None and store.warm:
        restore_s = t_in(model, device, worst_load, access="restore",
                         store_read_bw=store.read_bw)
        if restore_s < read_s:
            access = "restore"
            remarks.append("store beats recompute at the break-even")

    predicted = predict_makespan(
        model, device, n_docs=corpus.n_docs, workers=workers,
        shard_tokens=worst_load, n_filters=n_filters,
        question_tokens=question_tokens, preamble_tokens=preamble_tokens,
        selectivity=selectivity, access=access,
        store_read_bw=store.read_bw if access == "restore" else None)

    return Plan(mode=mode, workers=workers, tensor_parallel=tp,
                shards=shards, budget_tokens=budget, pin=pin,
                access=access,
                stage_token_window=1 if one_token_answers else 6,
                predicted_makespan_s=round(predicted, 1),
                remarks=tuple(remarks))
