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
from docengine.engineext import chainlogic

from .cost import (ACT_BYTES_PER_HIDDEN, BOOT_POOL_FRACTION,
                   ENGINE_SEQS_MAX, FORK_SEQ_S, POOL_HEADROOM,
                   ROUND_TOKENS, STEP_POOL_FRACTION, STEP_TOKENS_MAX,
                   STEP_TOKENS_MIN, decode_rate, predict_makespan, t_in)


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
    mode: str                 # executor internals: "chain" | "spec" |
    #                           "requests"
    workers: int              # data-parallel single-model workers
    tensor_parallel: int      # GPUs per worker (model split)
    shards: tuple             # doc-id tuples, one per worker
    budget_tokens: int        # resident-document admission cap
    pin: bool                 # request path only
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
    #                           pipelined_filter | hybrid_filter |
    #                           pipelined_map | hybrid_map | requests.
    #                           filter = gated; map = every prompt on
    #                           every document. hybrid forks the
    #                           remaining prompts at the switch stage;
    #                           one rule places the switch for both
    #                           semantics (chainlogic.hybrid_switch_stage)
    spec_after_stage: int = 0  # hybrid: gate through this stage, then
    #                            fork the remaining filters on the
    #                            survivors (0 = never switch)
    engine_max_seqs: int = 0  # boot the engine's max_num_seqs at least
    #                           this high: the worst-case admitted
    #                           document count (budget over the
    #                           smallest document) times the
    #                           live-sequence multiplier (n_filters
    #                           for spec, 1 otherwise)
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
               saturation_width_docs: int = 1,
               gated: bool = True,
               policy: str = None,
               gen_tokens: int = 0) -> "Plan | Refusal":
    """Plan the scan, or refuse it naming the violated constraint.

    `gated` is query semantics, not an estimate: True for a filter
    (a failed predicate skips the rest), False for a map - binary
    classification, every prompt on every document. The four
    operators come from crossing that semantics with one switch rule
    (chainlogic.hybrid_switch_stage): pipelined_filter and
    hybrid_filter for gated queries, pipelined_map and hybrid_map
    for maps. The rule forks the remaining prompts at the first
    stage whose remaining work underfills a round; for maps the
    document count never shrinks, so the rule says fork immediately
    (small corpus) or never (large corpus, where forking measured a
    tie and costs per-sibling CPU - the underfilled-round cell).
    Map plans are priced at selectivity 1 and hold every model to
    the one-token answer contract by constraining the sampler
    (allowed_token_ids = the yes and no ids), so no decode steps
    exist in them.

    `policy` overrides the rule for testing: "hybrid" turns the fork
    on even where the rule says never; "pipelined" turns it off.
    Semantics stay with `gated` either way.

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

    assert policy in (None, "pipelined", "hybrid"), policy
    if n_filters < 2:
        mode = "requests"
    else:
        mode = "chain" if gated else "spec"
    if mode == "spec" and not one_token_answers:
        remarks.append(
            "spec constrains sampling to the yes/no token ids: one "
            "token per stage by construction; the free-running model "
            "would chatter, so accuracy is judged against the "
            "constrained contract")
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
    spill_s = 0.0
    if pool < working_set and store is not None:
        access = "spill"
        # pessimistic spill traffic: every token past the pool moves
        # out and back once at store bandwidth (StoreSpec carries the
        # measured read rate; writes are priced at the same rate)
        overflow_tokens = max(0, worst_load - pool)
        spill_s = 2 * overflow_tokens * model.kappa / store.read_bw

    # one switch rule for both semantics: fork the remaining prompts
    # at the first stage whose remaining work no longer fills a round
    # (underfilled rounds cost full round time, so forking into them
    # is free - the ledger's underfilled-round cell). For filters the
    # survivors shrink by selectivity, so the switch lands mid-chain;
    # for maps the count is constant, so the rule says fork at stage
    # one (small corpus) or never (large corpus - measured tie, and
    # the fork's per-sibling CPU decides against it).
    spec_after = 0
    if n_filters >= 2:
        # per-filter selectivities when given: a skewed chain (0.9,
        # 0.9, 0.2, ...) has its survivor cliff where the selective
        # filter sits, which a mean smears away
        try:
            sels = [float(x) for x in selectivity]
        except TypeError:
            sels = [float(selectivity)] * n_filters
        rule_sels = sels if gated else [1.0] * n_filters
        spec_after = chainlogic.hybrid_switch_stage(
            corpus.n_docs / workers, rule_sels,
            max(1, question_tokens - preamble_tokens), ROUND_TOKENS)
        selectivity = sum(sels) / len(sels)   # scalar for pricing
    if policy == "pipelined":
        spec_after = 0
    elif policy == "hybrid":
        spec_after = spec_after or 1
    if access != "read" and spec_after:
        # the store's tiering connector runs on the stock scheduler
        # (as banked: the persist milestone); forks need the
        # plan-owned scheduler, and the two are not yet reconciled
        spec_after = 0
        remarks.append(
            "store access disables forks: the tiering connector runs "
            "on the stock scheduler and plan-owned memory has not "
            "been reconciled with it (persist milestone one)")
    if mode == "requests":
        operator = "requests"
    elif gated:
        operator = "hybrid_filter" if spec_after else "pipelined_filter"
    else:
        operator = "hybrid_map" if spec_after else "pipelined_map"

    predicted = predict_makespan(
        model, device, n_docs=corpus.n_docs, workers=workers,
        shard_tokens=worst_load, n_filters=n_filters,
        question_tokens=question_tokens, preamble_tokens=preamble_tokens,
        selectivity=selectivity if mode != "spec" else 1.0,
        access=access if access != "spill" else "read",
        store_read_bw=store.read_bw if access == "restore" else None)
    predicted += spill_s
    if gen_tokens:
        # generative maps: decode priced by the calibrated decode
        # model (cost.decode_rate: step time from device physics at
        # the re-anchored PHI plus the measured per-step software
        # floor), at the mean context - each generated token's
        # attention reads the whole context, so the rate falls with
        # document length. Width is what admission actually allows:
        # the budget holds documents at their worst-case charge (body
        # plus every prompt plus the full cap per prompt), each live
        # document runs all its prompts, and the engine's sequence
        # cap bounds the total. Pessimistic: assumes every prompt
        # generates its full cap.
        gen_ctx = int(corpus.mean_doc_tokens + question_tokens
                      + gen_tokens / 2)
        per_doc_charge = corpus.mean_doc_tokens \
            + n_filters * (question_tokens + gen_tokens)
        docs_live = max(1.0, budget / max(1.0, per_doc_charge))
        concurrent = int(max(1, min(
            corpus.n_docs * n_filters / workers,
            docs_live * n_filters,
            ENGINE_SEQS_MAX)))
        rate = decode_rate(model, device, concurrent, gen_ctx)
        predicted += (corpus.n_docs * n_filters * gen_tokens
                      / rate / workers)
        remarks.append(
            f"generation priced at {rate:,.0f} tokens/second by the "
            f"calibrated decode model (~{gen_ctx}-token contexts, "
            f"{concurrent} concurrent from the admission arithmetic), "
            "assuming the full output cap per prompt")
    if operator == "hybrid_map":
        # forking runs one extra sequence per remaining prompt. Each
        # costs scheduler CPU (about 70 microseconds, measured: the
        # ledger's fork section, spec_smoke 2026-08-05) plus one
        # partial-block KV copy, which moves at memory bandwidth, not
        # at the compute rate - a copy is not a kernel. The
        # hybrid_filter's forked tail is priced by the same terms
        # over its expected survivors.
        extra = corpus.n_docs * (n_filters - 1) / workers
        predicted += extra * FORK_SEQ_S
        predicted += extra * (16 * model.kappa) / device.BW
    elif operator == "hybrid_filter":
        surviving = corpus.n_docs / workers
        for s_j in [selectivity] * spec_after:
            surviving *= s_j
        extra = surviving * (n_filters - spec_after)
        predicted += extra * FORK_SEQ_S
        predicted += extra * (16 * model.kappa) / device.BW

    # the sequence cap must never bind before the token budget: size
    # it from the worst case, not a margin. Admission is
    # budget-limited and the smallest documents pack densest, so the
    # bound is exact at the smallest document size; speculation
    # multiplies each admitted document by its live siblings. The
    # small constant covers the one-step overlap between a finishing
    # document's retirement and the next admission. The cap prices a
    # bookkeeping table (bytes per row), so binding is expensive and
    # generosity is not.

    # exact under any size skew: the most documents that can be
    # resident together is the longest ascending prefix that fits
    # under the budget - one tiny outlier no longer inflates the
    # bound the way dividing by the single smallest document did
    admitted_worst, running = 0, 0
    for t in sorted(doc_tokens):
        running += int(t)
        if running > budget:
            break
        admitted_worst += 1
    admitted_worst = max(1, admitted_worst)
    # every map runs all its prompts as live sequences at once (the
    # hybrids by forking, pipelined_map by per-pair requests), so only
    # the gated filters hold one live sequence per document
    live_mult = (1 if operator in ("pipelined_filter", "requests")
                 else n_filters)
    engine_max_seqs = min(admitted_worst * live_mult + 16,
                          ENGINE_SEQS_MAX)

    # the step budget (max_num_batched_tokens), derived: steps are
    # compute-bound past ~105 tokens, so a bigger budget buys only
    # amortization of per-step CPU and the weight pass (measured 0.8
    # percent across the sweep's 8k-32k cells) - take it only where
    # the activation reservation it forces at boot stays a rounding
    # error against the KV pool. Fat pools pack large steps (~25k at
    # 4B on the H100); thin pools keep the floor rather than trade
    # pool for under a percent of wall (~2k at 32B on the L40S).
    act_bytes = ACT_BYTES_PER_HIDDEN * model.h
    step_tokens = int(min(
        STEP_TOKENS_MAX,
        max(STEP_TOKENS_MIN,
            STEP_POOL_FRACTION * pool * model.kappa / act_bytes)))
    engine_step_tokens = max(step_tokens, engine_max_seqs)

    if operator in ("hybrid_filter", "hybrid_map"):
        remarks.append(
            f"boot the engine with max_num_seqs >= {engine_max_seqs} "
            f"and max_num_batched_tokens >= {engine_step_tokens}: "
            "speculation multiplies live sequences by the filter "
            "count, a low cap binds before the token budget does, and "
            "the engine requires the round budget to cover the cap")

    return Plan(mode=mode, workers=workers, tensor_parallel=tp,
                shards=shards, budget_tokens=budget, pin=pin,
                access=access,
                stage_token_window=(
                    1 if one_token_answers or mode == "spec" else 6),
                predicted_makespan_s=round(predicted, 1),
                operator=operator,
                spec_after_stage=spec_after,
                engine_max_seqs=engine_max_seqs,
                engine_step_tokens=engine_step_tokens,
                remarks=tuple(remarks))
