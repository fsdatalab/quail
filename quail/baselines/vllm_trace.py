"""A record of vLLM's scheduler steps, for measuring preemption."""

from __future__ import annotations

import json
import os
import time

STEP_TRACE_ENV = "QUAIL_VLLM_STEP_TRACE"


def install_step_trace() -> None:
    """Append one JSON line per vLLM scheduler step to a file.

    vLLM loads this as a general plugin in every process it starts. It
    does nothing unless ``QUAIL_VLLM_STEP_TRACE`` names the file. Each
    line holds the step's wall-clock time ``t``, the tokens it
    schedules, the tokens of those it recomputes because their request
    was preempted, the requests preempted since the previous step, the
    requests running and waiting, and the fraction of KV blocks in use.
    """
    path = os.environ.get(STEP_TRACE_ENV)
    if not path:
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "quail_step_trace", None) == path:
        return
    schedule = Scheduler.schedule
    preempt = Scheduler._preempt_request
    # request id -> KV positions it had computed when last preempted
    computed_at_preemption = {}
    counts = {"preempted": 0}

    def traced_preempt(self, request, *args, **kwargs):
        computed_at_preemption[request.request_id] = max(
            computed_at_preemption.get(request.request_id, 0),
            request.num_computed_tokens)
        counts["preempted"] += 1
        return preempt(self, request, *args, **kwargs)

    def traced_schedule(self, *args, **kwargs):
        output = schedule(self, *args, **kwargs)
        scheduled = output.num_scheduled_tokens
        recomputed = 0
        for request_id in [request_id for request_id in computed_at_preemption
                           if request_id in scheduled]:
            after = self.requests[request_id].num_computed_tokens
            before = after - scheduled[request_id]
            reached = computed_at_preemption[request_id]
            recomputed += max(0, min(after, reached) - before)
            if after >= reached:
                del computed_at_preemption[request_id]
        with open(path, "a") as trace:
            trace.write(json.dumps({
                "t": round(time.time(), 6),
                "scheduled": output.total_num_scheduled_tokens,
                "recomputed": recomputed,
                "preempted": counts["preempted"],
                "running": len(self.running),
                "waiting": len(self.waiting),
                "kv_usage": round(self.kv_cache_manager.usage, 4),
            }) + "\n")
        counts["preempted"] = 0
        return output

    Scheduler.schedule = traced_schedule
    Scheduler._preempt_request = traced_preempt
    Scheduler.quail_step_trace = path
