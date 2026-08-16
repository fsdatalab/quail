"""Pure decision logic for coalescing offload load jobs.

The measured problem (results/engine/persist_xfer_trace.jsonl.gz): the
stock connector submits one transfer job per request, ~16 MB each, and
strict job ordering prices every job at ~1.7 ms of overhead against
~0.6 ms of copying, so the channel runs a quarter of the wall. The
scheduler already hands the worker every load job created in one
engine step as a single dict; merging that dict into one transfer
pays the overhead once per step instead of once per request.

This module must import no vLLM, torch, or numpy - same rule as
chainlogic.py. The adapter (offload.py) cannot be imported without
vLLM 0.26.0, so the merge decisions are computed here on plain ints
and lists and covered by tests/test_offload_logic.py without an
engine.
"""


def mergeable(n_groups, src_blocks_per_chunk):
    """A load job can join a merged transfer only when its descriptor
    arithmetic is offset-free: one KV group, and CPU chunks the same
    size as GPU blocks. With src_blocks_per_chunk == 1 the handler's
    partial-first-chunk skip (block_index % blocks_per_chunk) is zero
    for every job, so concatenated block lists mean exactly what the
    jobs meant separately. Anything else is submitted unmerged."""
    return n_groups == 1 and src_blocks_per_chunk == 1


def plan_merge(jobs, src_blocks_per_chunk):
    """Split one step's load jobs into a merged batch and a remainder.

    `jobs` is a list of (job_id, n_src_blocks, n_dst_blocks, n_groups).
    Returns (merged_ids, single_ids): merged_ids in submission order
    when two or more jobs qualify, else empty and everything single.
    A lone qualifying job gains nothing from merging and stays single,
    so the merged path never fires without an actual batch."""
    ok, single = [], []
    for job_id, n_src, n_dst, n_groups in jobs:
        if n_src == n_dst and mergeable(n_groups, src_blocks_per_chunk):
            ok.append(job_id)
        else:
            single.append(job_id)
    if len(ok) < 2:
        return [], [j for j, *_ in jobs]
    return ok, single


def split_result(total_time, block_counts):
    """Share one merged transfer's measured duration across its
    constituent jobs, proportional to block count. Only the totals are
    physical; the per-job splits keep downstream per-job accounting
    additive (bytes / time still sums to the measured whole)."""
    total_blocks = sum(block_counts)
    if total_blocks <= 0:
        return [0.0] * len(block_counts)
    return [total_time * n / total_blocks for n in block_counts]


def synthetic_ids():
    """Job ids for merged submissions: negative and descending, so
    they can never collide with the scheduler's non-negative ids and
    never enter its per-request job bookkeeping by accident."""
    n = 0
    while True:
        n -= 1
        yield n
