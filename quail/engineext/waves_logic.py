"""Pure decision logic for wave-phased KV movement.

Waves are the synchronous alternative to vLLM's per-request async
loads. Each scheduling pass, the driver may pick ONE batch of
admitted documents whose KV sits in the host store, allocate their
GPU blocks, and hand the worker a single copy list; the worker runs
it on the transfer stream and records a CUDA event; the first step
that reads any of a wave's blocks makes the compute stream wait on
that wave's event. Ordering is CUDA stream ordering end to end - no
callbacks, no request parking, no polling.

Spill under a choked pool is the same machinery pointed backwards:
documents were already written through to the store at first
prefill, so parking a cohort is freeing its GPU blocks, and bringing
it back is a wave reload before its next visit.

This module must import no vLLM or torch - same rule as chainlogic.
The driver (waves.py) applies these decisions to engine state;
tests/test_waves_logic.py covers them with no engine installed.
"""


def plan_wave(candidates, wave_tokens, in_flight, max_in_flight=2):
    """Pick the next wave: document ids in admission order until the
    wave budget fills. candidates = [(doc_id, tokens)] for admitted
    documents with full store hits and no wave assigned yet. Returns
    [] while max_in_flight waves are outstanding - the channel runs
    ahead of compute by a bounded number of waves, never unbounded."""
    if in_flight >= max_in_flight:
        return []
    out, used = [], 0
    for doc_id, tokens in candidates:
        if out and used + tokens > wave_tokens:
            break
        out.append(doc_id)
        used += tokens
    return out


def waves_to_gate(scheduled_wave_ids, already_gated):
    """Wave ids the compute stream must wait on this step: every wave
    holding a document scheduled now, minus waves gated before. One
    event wait per wave, ever - later steps on the same stream
    inherit the ordering, so a second wait would be dead code."""
    return sorted(set(scheduled_wave_ids) - set(already_gated))


def full_blocks(doc_tokens, block_size):
    """Blocks a wave loads for one document: the full blocks only.
    The partial tail block is left uncached and its tail tokens
    recompute (at most block_size - 1 of them), because a partially
    filled block cannot serve a prefix-cache hit."""
    return doc_tokens // block_size


def plan_rotation(cohorts, resident_capacity_tokens):
    """Round-robin under a choked pool: keep the cohorts visited
    soonest, park the tail. cohorts = [(cohort_id, tokens,
    next_visit)] with next_visit comparable (lower = sooner). Parking
    is free at the memory level - every prefilled document is already
    written through to the store - so the only cost of a parked
    cohort is its wave reload before the next visit.

    Chain mode makes this the provably optimal eviction: the
    scheduler knows every document's next visit exactly, so the
    furthest-future cohorts are the correct victims, not a guess."""
    keep, park, used = [], [], 0
    for cid, tokens, _visit in sorted(cohorts, key=lambda c: (c[2], c[0])):
        if used + tokens <= resident_capacity_tokens:
            keep.append(cid)
            used += tokens
        else:
            park.append(cid)
    return keep, park
