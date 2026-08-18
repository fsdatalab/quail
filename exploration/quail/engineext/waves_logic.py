"""Pure decision logic for wave-phased KV movement.

Waves are the synchronous alternative to vLLM's per-request async
loads: batches of stored documents whose KV is copied host-to-GPU on
the transfer stream, gated by one CUDA event per wave. The driver
(waves.py) applies these decisions to engine state.

Territory between waves and the vendor's per-request load path is
enforced by the claim mechanism in offload.py, not here: a claimed
document's store lookup answers no-hit, so the vendor can never open
a duplicate per-request load for bytes a wave is already copying.
What lives here is the packing decision only.

This module must import no vLLM or torch - same rule as chainlogic.
tests/test_waves_logic.py covers it with no engine installed.
"""


def plan_waves(candidates, wave_tokens, budget_tokens):
    """Chunk candidates into waves, in the order given (the driver
    passes the vendor's own queue order, so registrations land just
    ahead of the vendor's sweep). Each wave packs documents until
    wave_tokens; planning stops once the total pinned tokens would
    pass budget_tokens (the pool headroom above the vendor's floor).
    Returns a list of waves, each a list of doc ids. A document
    larger than wave_tokens gets its own wave rather than becoming
    unreachable."""
    waves, cur, cur_used, total = [], [], 0, 0
    for doc_id, tokens in candidates:
        if total + tokens > budget_tokens:
            break
        if cur and cur_used + tokens > wave_tokens:
            waves.append(cur)
            cur, cur_used = [], 0
        cur.append(doc_id)
        cur_used += tokens
        total += tokens
    if cur:
        waves.append(cur)
    return waves
