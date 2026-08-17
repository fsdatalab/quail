"""Runner slot truth for the admission gate.

The model runner keeps a fixed table of per-request slots
(RequestStates); admitting more concurrent requests than it has
slots crashes it ("No free indices"). The admission decision lives
in the scheduler, so the scheduler needs the free-slot count - but
with overlapped scheduling the runner applies each batch on a
worker thread while the scheduler is already planning the next
one, so any count the scheduler samples can be stale by the
batches still in flight. Margin-based corrections (subtract the
last N batches' admissions) were tried and leaked: they subtract a
window, not the actual in-flight set.

This module closes the gap exactly. The runner side - installed by
wrapping GPUModelRunner.add_requests, which the engine calls once
per applied batch, removals already applied - publishes one atomic
snapshot after applying each batch:

    snap = (free slots now, slots taken since this engine booted)

The scheduler counts slots taken in the batches it has emitted
(quail scheduler.py, _de_emitted_new). emitted - applied is then
exactly the consumption still in flight, so the gate may admit up
to

    snap.free - (emitted - snap.applied)

new sessions and the runner cannot run out. Two properties make
this exact rather than heuristic:

  which requests take a slot   only scheduled_new_reqs consume
      free_indices, and a session continuation (a rewound chain
      stage arrives as a "new" request whose id is already in the
      runner's table) is remove-then-add, net zero. Both sides
      exclude it by the same membership rule: the runner checks
      its table, the scheduler checks its rewound set. Resumption
      after preemption freed its slot at preemption, so both
      sides count it again.
  frees are never guessed      a slot freed by an in-flight batch
      is simply absent from the snapshot until the runner's next
      publish. That error direction admits less for one step,
      never more.

Both sides run in the engine-core process (the worker is
in-process under the single-GPU executor; overlapped scheduling
adds a worker thread, not a process), so the snapshot is one tuple
assignment under the GIL: a reader sees the old pair or the new
pair, never a mix. The applied counter lives on the runner's own
RequestStates object, so a fresh engine starts a fresh count;
QuailScheduler.__init__ calls BOARD.reset() so a stale snapshot
from a dead engine in the same process can never be read against
a new scheduler's emitted count.

QUAIL_SLOTDUMP=1 additionally prints who holds every slot whenever
free slots run low - the diagnostic that localized every slot
defect so far.
"""

import json
import os


class _Board:
    def __init__(self):
        self.snap = None   # (free, applied), None until a batch applies

    def reset(self):
        self.snap = None


BOARD = _Board()
_installed = False


def install():
    """Wrap GPUModelRunner.add_requests to publish the snapshot.
    Idempotent. Called from QuailScheduler.__init__, which runs in
    the engine-core process after the executor (and so the runner
    module) exists and before any batch executes."""
    global _installed
    if _installed:
        return
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    orig = GPUModelRunner.add_requests
    dump = os.environ.get("QUAIL_SLOTDUMP", "0") == "1"

    def add_requests(self, scheduler_output):
        st = self.req_states
        taken = sum(1 for r in scheduler_output.scheduled_new_reqs
                    if r.req_id not in st.req_id_to_index)
        if dump and taken and len(st.free_indices) < taken + 8:
            ids = list(st.req_id_to_index)
            from collections import Counter
            pref = Counter(i.split("|")[0][:24] for i in ids)
            print("[quail-slotdump] " + json.dumps(dict(
                need=taken, free=len(st.free_indices),
                held=len(ids), prefixes=dict(pref),
                adding=[r.req_id
                        for r in scheduler_output.scheduled_new_reqs][:12],
                sample=ids[:24])), flush=True)
        out = orig(self, scheduler_output)
        st._quail_applied = getattr(st, "_quail_applied", 0) + taken
        BOARD.snap = (len(st.free_indices), st._quail_applied)
        return out

    GPUModelRunner.add_requests = add_requests
    _installed = True
