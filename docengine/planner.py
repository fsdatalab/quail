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

Every rule cites a banked measurement or the calibrated model; the
tests assert the planner reproduces each measured winner.
"""

from dataclasses import dataclass, field

from .configs import DeviceConfig, ModelConfig

# Calibrated constants from the measured layer.
PHI = 80_000 / 275_000        # serving rate over spec ceiling, 4B anchor
ENGINE_OVERHEAD_S = 3.2       # per-query software residue at 10k docs
BOOT_POOL_FRACTION = 0.92     # gpu_memory_utilization we ship
POOL_HEADROOM = 0.80          # budget stays under the pool by this


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
    warm: bool = False        # notes for this corpus already saved


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


def _pool_tokens(model: ModelConfig, device: DeviceConfig, tp: int) -> int:
    """KV-pool tokens per worker: free memory after weights, spread
    over the tensor-parallel group (each card holds 1/tp of weights
    and 1/tp of every token's KV)."""
    free = device.M * BOOT_POOL_FRACTION * tp - model.W_mem
    return max(0, int(free / model.kappa))


def _prefill_rate(model: ModelConfig) -> float:
    """Serving-rate estimate, tokens/s: the calibrated fraction of the
    spec compute ceiling 2P FLOPs per token. Anchors: 80,000 measured
    at 4B (the ratio PHI), 10,800 measured at 32B (predicted 9,200,
    fifteen percent under)."""
    return PHI * 1.979e15 / (2 * model.P)


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
               one_token_answers: bool = True) -> Plan:
    corpus = CorpusStats(n_docs=len(doc_tokens),
                         total_tokens=int(sum(doc_tokens)),
                         max_doc_tokens=int(max(doc_tokens)))
    remarks = []

    # GPU layout: split the model only when one card cannot hold it;
    # otherwise every card is an independent worker on its own shard.
    tp = 1
    while model.W_mem > device.M * BOOT_POOL_FRACTION * tp:
        tp *= 2
    if tp > 1:
        remarks.append(f"model needs {tp} cards; workers are {tp}-card")
    workers = max(1, gpus // tp)
    shards, worst_load = _balanced_shards(doc_tokens, workers)

    pool = _pool_tokens(model, device, tp)
    budget = int(min(pool * POOL_HEADROOM,
                     max(corpus.max_doc_tokens + question_tokens,
                         worst_load)))
    overflow = worst_load > pool
    if overflow:
        remarks.append("shard overflows the KV pool; pins off")

    mode = "chain" if n_filters >= 2 else "requests"
    # A pin keeps notes for a future consumer. One filter has no
    # future consumer, and chain mode keeps documents resident by
    # construction, so pins apply only to a multi-filter request plan
    # (kept for when chain mode is unavailable) and never under
    # overflow, where the 32B measurement showed them losing to churn.
    pin = mode == "requests" and n_filters >= 2 and not overflow

    rate = _prefill_rate(model)
    access = "read"
    read_s = worst_load / rate
    if store is not None and store.warm:
        restore_s = worst_load * model.kappa / store.read_bw
        if restore_s < read_s:
            access = "restore"
            read_s = restore_s
            remarks.append("store beats recompute at the break-even")

    # Question work per shard: stage one reads its whole question,
    # later stages only the tail past the shared preamble, and the
    # documents reaching stage j thin as selectivity**(j-1) (all-pass
    # bound when selectivity is unknown).
    tail = max(1, question_tokens - preamble_tokens)
    reach = sum(selectivity ** j for j in range(1, n_filters))
    q_s = (corpus.n_docs / workers) * (
        question_tokens + reach * tail) / rate
    predicted = read_s + q_s + ENGINE_OVERHEAD_S

    return Plan(mode=mode, workers=workers, tensor_parallel=tp,
                shards=shards, budget_tokens=budget, pin=pin,
                access=access,
                stage_token_window=1 if one_token_answers else 6,
                predicted_makespan_s=round(predicted, 1),
                remarks=tuple(remarks))
