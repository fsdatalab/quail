"""The declarative layer: describe the filter query, get a physical plan.

plan_query takes what the user knows (filter count, corpus statistics,
model, device, GPU count, whether a KV store exists) and returns a
Plan choosing everything the measurements showed to matter:

- executor: chain (one living request per document, rewound between
  filters) against separate requests; chain wins with two or more
  filters, and a single-filter query has nothing to chain
- GPU layout: split the corpus across single-GPU workers (documents
  are independent, so every floor divides by the worker count) and
  split the model only when its weights do not fit one card
- admission budget: how many tokens of documents may be resident at
  once, kept under the KV pool so the pool never overflows and the
  prefix cache is never forced to evict. This is a TOKEN budget, not
  a request count: a request count cannot see document length, and
  the same 4,096-request cap that fits short documents overflows the
  pool on long ones (measured: reads 2.40x against 1.23x, wall 80.1s
  against 42.9s at 10k documents on the 4B model)
- access: restore KV from a store instead of recomputing when the
  store's bandwidth beats the rate at which prefill would recreate it
  (kappa times the prefill rate, the measured break-even)
- decode budget per stage: one token for models that keep the
  one-token answer contract; a decisive-token window for models that
  restate or chatter

A configuration that cannot run is refused, not planned around:
plan_query returns a Refusal naming the violated constraint when the
weights need more cards than exist, when the weights leave no KV
pool, or when the pool cannot hold the saturation working set.

Every calibrated constant lives in quail/plan/cost.py.
"""

from dataclasses import dataclass, field

from quail.configs import DeviceConfig, ModelConfig

from .cost import (ACT_BYTES_PER_HIDDEN, BOOT_POOL_FRACTION,
                   ENGINE_SEQS_MAX, POOL_HEADROOM, STEP_POOL_FRACTION,
                   STEP_TOKENS_MAX, STEP_TOKENS_MIN, predict_makespan,
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
    mode: str                 # executor internals: "chain" | "requests"
    workers: int              # data-parallel single-model workers
    tensor_parallel: int      # GPUs per worker (model split)
    shards: tuple             # doc-id tuples, one per worker
    budget_tokens: int        # resident-document admission cap
    access: str               # "read" | "restore" | "spill": how the
    #                           document KV is supplied. read =
    #                           compute from text; restore = load
    #                           from a warm store written at ingest;
    #                           spill = the pool cannot hold the
    #                           working set, so the tiering store
    #                           absorbs the overflow and reloads on
    #                           demand (only plannable with a store)
    stage_token_window: int   # decode budget per filter stage
    predicted_makespan_s: float
    operator: str = ""        # the public operator name:
    #                           pipelined_filter (chain mode, gated,
    #                           rewound between stages) | requests
    engine_max_seqs: int = 0  # boot the engine's max_num_seqs at least
    #                           this high: the worst-case admitted
    #                           document count, one live sequence each
    engine_step_tokens: int = 0  # boot max_num_batched_tokens here: the
    #                              largest step budget whose activation
    #                              reservation stays under
    #                              STEP_POOL_FRACTION of the KV pool,
    #                              clamped to the tested range and never
    #                              below the sequence cap (the engine
    #                              requires the step budget to cover it)
    remarks: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Refusal:
    """The answer when the configuration is infeasible: the violated
    constraint, reported instead of a run."""
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
    """Plan the filter scan, or refuse it naming the violated
    constraint.

    The query is a chain of gated filters: a document that fails a
    predicate skips the rest. `selectivity` is the per-stage pass rate
    (a scalar, or one rate per stage), used to price the survivors.

    saturation_width_docs is the pool floor: the memory left after
    weights must hold that many per-document working sets (largest
    document plus its first question) or the configuration is refused.
    The default 1 is the honest minimum - any execution needs one
    resident document.
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
        if store is None:
            return Refusal(
                reasons=(f"pool holds {pool} tokens; saturation width "
                         f"{saturation_width_docs} needs {working_set} "
                         f"(largest document {corpus.max_doc_tokens} + "
                         f"first question {question_tokens} tokens)",),
                constraint="pool_under_working_set",
                needed=working_set, available=pool, unit="tokens")
        # a store turns the refusal into a plan: the tiering
        # connector spills overflow KV and reloads it on demand.
        # Priced pessimistically: every pooled token beyond capacity
        # moves out and back once at store bandwidth.
        remarks.append(
            f"pool holds {pool} tokens against a {working_set}-token "
            "working set: the tiering store spills the overflow; "
            "expect the spill-bound rate, not the read floor")

    budget = int(min(pool * POOL_HEADROOM,
                     max(corpus.max_doc_tokens + question_tokens,
                         worst_load)))
    overflow = worst_load > pool
    if overflow:
        remarks.append("shard overflows the KV pool; pins off")

    # one filter has nothing to chain; two or more run in chain mode,
    # where the document's KV belongs to a living request and cannot
    # be evicted between stages
    mode = "requests" if n_filters < 2 else "chain"

    access = "read"
    read_s = t_in(model, device, worst_load)
    if store is not None and store.warm:
        restore_s = t_in(model, device, worst_load, access="restore",
                         store_read_bw=store.read_bw)
        if restore_s < read_s:
            access = "restore"
            remarks.append("store beats recompute at the break-even")
    spill_s = 0.0
    if pool < working_set and store is not None:
        access = "spill"
        # pessimistic spill traffic: every token past the pool moves
        # out and back once at store bandwidth (StoreSpec carries the
        # measured read rate; writes are priced at the same rate)
        overflow_tokens = max(0, worst_load - pool)
        spill_s = 2 * overflow_tokens * model.kappa / store.read_bw

    operator = "requests" if mode == "requests" else "pipelined_filter"

    # per-filter selectivities when given: a skewed chain (0.9, 0.9,
    # 0.2, ...) has its survivor cliff where the selective filter
    # sits, which a mean smears away
    try:
        sels = [float(x) for x in selectivity]
        selectivity = sum(sels) / len(sels)   # scalar for pricing
    except TypeError:
        pass

    predicted = predict_makespan(
        model, device, n_docs=corpus.n_docs, workers=workers,
        shard_tokens=worst_load, n_filters=n_filters,
        question_tokens=question_tokens, preamble_tokens=preamble_tokens,
        selectivity=selectivity,
        access=access if access != "spill" else "read",
        store_read_bw=store.read_bw if access == "restore" else None)
    predicted += spill_s

    # the sequence cap must never bind before the token budget: size
    # it from the worst case, not a margin. Admission is
    # budget-limited and the smallest documents pack densest, so the
    # bound is exact at the longest ascending prefix that fits under
    # the budget - one tiny outlier no longer inflates the bound the
    # way dividing by the single smallest document did. Gated filters
    # hold one live sequence per document. The small constant covers
    # the one-step overlap between a finishing document's retirement
    # and the next admission.
    admitted_worst, running = 0, 0
    for t in sorted(doc_tokens):
        running += int(t)
        if running > budget:
            break
        admitted_worst += 1
    admitted_worst = max(1, admitted_worst)
    engine_max_seqs = min(admitted_worst + 16, ENGINE_SEQS_MAX)

    # the step budget (max_num_batched_tokens), derived: the dense
    # projections are compute-bound past ~400 tokens per step, so a
    # bigger budget buys only amortization of the per-step host cost
    # (the measured b of T_step = a*B + b) - take it only where the
    # activation reservation it forces at boot stays a rounding error
    # against the KV pool. Fat pools pack large steps (~25k at 4B on
    # the H100); thin pools keep the floor rather than trade pool for
    # a small wall gain.
    act_bytes = ACT_BYTES_PER_HIDDEN * model.h
    step_tokens = int(min(
        STEP_TOKENS_MAX,
        max(STEP_TOKENS_MIN,
            STEP_POOL_FRACTION * pool * model.kappa / act_bytes)))
    engine_step_tokens = max(step_tokens, engine_max_seqs)

    return Plan(mode=mode, workers=workers, tensor_parallel=tp,
                shards=shards, budget_tokens=budget,
                access=access,
                stage_token_window=1 if one_token_answers else 6,
                predicted_makespan_s=round(predicted, 1),
                operator=operator,
                engine_max_seqs=engine_max_seqs,
                engine_step_tokens=engine_step_tokens,
                remarks=tuple(remarks))
