"""Query-aware scheduler extension for vLLM v1.

Subclasses vLLM's scheduler so the query plan, not the engine's
heuristics, governs memory. Live documents' prefix blocks are pinned
by holding an extra block-pool reference, dead documents' blocks are
stripped and freed the instant the plan learns of the death, and every
block a planned request leaves behind that is not pinned has its cache
entry removed on free. Everything uses the block pool's public
interface; vendor code is not modified.

Chain mode is the reason this class exists: one living request runs a
document's whole filter chain. The client registers the question token
lists and the yes/no token ids once; after that each stage's answer is
judged here, the request is rewound to the document boundary (question
and answer KV erased, document KV kept), and the next question is
appended through the engine's own session mechanism. The document is
prefilled exactly once no matter how many filters it survives.

Single tenant mode (env QUAIL_SINGLE_TENANT, default on) is the
shipping configuration: requests that do not carry a plan tag are
refused, so unplanned memory cannot exist, and the engine's
keep-the-most-recent eviction rule becomes unreachable by
construction. If it ever fires anyway, that means memory escaped the
plan's accounting, and the scheduler raises instead of silently
falling back to heuristic behavior. Set the env var to 0 for shared
card experiments, where foreign traffic is legitimate and the recency
rule is its default policy.

The scheduler lives in the engine core process, so client directives
ride inside request ids. Protocol, fields separated by "|" ("de1" is
the wire-format version tag, not a product name):

    de1|p<tokens>|d<doc>|u<count>|r<doc,doc,...>|<suffix>

  p<tokens>  pin the document prefix covering the first <tokens> tokens
             when this request's blocks are freed
  d<doc>     document key that owns the pin
  u<count>   the pin expects <count> release mentions before it frees
             (shared scans: one per query reading the document);
             missing means one, the single-query behavior
  r<...>     release one mention of each listed document key's pin;
             a pin frees when its mention count reaches zero.
             "*" frees every pin outright regardless of counts
  c          a chain request: this request runs a document's whole
             filter chain through in-engine rewinds
  reg        a registration request: its prompt carries the question
             token lists, its id the gate token ids
  Q<qid>     on registration and chain requests: the query these
             questions or this chain belong to; missing means the
             default query id "0", the single-query protocol

Every decision - id parsing, pin refcounts, rewind arithmetic, the
answer gate, the strict-mode predicate - is computed in chainlogic.py,
which imports no vLLM and is unit-tested without an engine
(tests/test_engineext_logic.py). This class only applies those
decisions to vLLM state.
"""

import os
from collections import deque
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus, StreamingUpdate

from . import chainlogic
from .chainlogic import parse_qid, parse_tag  # noqa: F401  module API


class QuailScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._de_pins = chainlogic.PinLedger()  # doc key -> blocks, uses
        self._de_rid_tag = {}       # request id -> parsed tag (once)
        self._de_pinned_ids = set()  # block ids currently pinned
        self._de_intent = {}        # request id -> (pin_tokens, doc, uses)
        self._de_queries = {}       # query id -> dict(qs, qc, yes, no)
        self._de_chain = {}         # request id -> dict(stage, d, qid)
        # a rewound session waiting mid-chain still holds its
        # model-runner slot, so living sessions (running + rewound)
        # must never exceed the slots. Two rules enforce it. Rewound
        # sessions are prepended to the waiting queue: their KV is
        # resident and finishing them frees pool and slots, so they
        # must never starve behind fresh documents the pool cannot
        # even admit (measured: running drained to zero with every
        # slot parked rewound, then one floored admission crashed the
        # runner). And fresh documents are held out of the queue
        # until there is slot slack for them; shrinking the vendor's
        # running bound instead would throttle the rewound sessions
        # themselves, which need no new slot.
        self._de_base_max_reqs = self.max_num_running_reqs
        self._de_requeued = set()
        self._de_held = deque()
        # slot accounting trace (env QUAIL_SLOTSTATS): one line per
        # 100 steps naming every population that can hold or shadow a
        # model-runner slot, for localizing 'No free indices'
        self._de_slotstats = os.environ.get(
            "QUAIL_SLOTSTATS", "0") == "1"
        self._de_sched_i = 0
        self._de_strict = os.environ.get(
            "QUAIL_SINGLE_TENANT", "1") == "1"
        print(f"[quail-sched] init: strict {self._de_strict}, overlapped "
              f"scheduling {self.scheduler_config.async_scheduling}",
              flush=True)
        self._de_plan_evicting = False
        # Core-process profiling: this object lives in the engine core,
        # the one process the client-side profiler cannot see, so the
        # profiler must be started from here.
        self._de_prof = os.environ.get("QUAIL_PROFILE", "0") == "1"
        self._de_prof_cleared = False
        if self._de_prof:
            import yappi
            yappi.set_clock_type("cpu")
            yappi.start()
        # Step recorder: near-zero overhead, for finding where wall
        # time goes when CPU profiling says CPU is not the problem.
        self._de_steps = ([] if os.environ.get(
            "QUAIL_STEPSTATS", "0") == "1" else None)
        # Step trace: one JSONL record per step (tokens, sequences,
        # prefill/decode split, unique documents, pool occupancy,
        # packing CPU), for the utilization figure. Appended in
        # batches so a run that never drains still leaves at most
        # one batch unwritten.
        self._de_trace = os.environ.get("QUAIL_STEPTRACE") or None
        self._de_trace_buf = []
        # Calibration extensions, both default off. QUAIL_STEPSHAPES
        # records each scheduled request's [new, cached] token pair.
        # QUAIL_STEPTRACE_FLUSH writes every record as soon as its
        # timing lands - the 256-record batching would otherwise keep
        # a short calibration run's records (1-3 per cell) unreadable
        # until drain.
        self._de_shapes = os.environ.get("QUAIL_STEPSHAPES", "0") == "1"
        self._de_flush_every = os.environ.get(
            "QUAIL_STEPTRACE_FLUSH", "0") == "1"
        self._de_sched_exit = None
        if self._de_trace:
            tp = self.kv_cache_manager.block_pool
            if not (hasattr(tp, "get_num_free_blocks")
                    and hasattr(tp, "num_gpu_blocks")):
                # vendor seam moved; trace degrades to off, run intact
                print("[quail-trace] block pool lacks occupancy "
                      "accessors; step trace disabled", flush=True)
                self._de_trace = None
            else:
                open(self._de_trace, "w").close()
                print(f"[quail-trace] step trace -> {self._de_trace}",
                      flush=True)
        self._de_stats = dict(pinned=0, released=0, blocks=0,
                              foreign_rejected=0, heuristic_evictions=0)
        # wave pre-loading (QUAIL_WAVES=1): synchronous KV pre-load
        # from the host store, decisions in waves_logic, engine
        # surgery in waves.py; off by default and inert when off
        from .waves import WaveDriver
        self._de_waves = WaveDriver(self)

        pool = self.kv_cache_manager.block_pool
        orig_evict = pool._maybe_evict_cached_block

        def guarded(block):
            evicted = orig_evict(block)
            if chainlogic.is_heuristic_eviction(
                    evicted, self._de_plan_evicting):
                self._de_stats["heuristic_evictions"] += 1
            if chainlogic.strict_violation(
                    evicted, self._de_plan_evicting, self._de_strict):
                raise RuntimeError(chainlogic.STRICT_VIOLATION)
            return evicted

        pool._maybe_evict_cached_block = guarded

    def _de_gate_fresh(self):
        """Slot gate: fresh documents may sit in the waiting queue
        only while there is model-runner slot slack for them. A
        request with computed tokens or a rewound continuation
        already holds its slot and passes untouched; a preempted
        request lost its slot at preemption and re-enters as fresh.
        Held documents wait aside in arrival order and are released
        as sessions finish and free slots."""
        base = self._de_base_max_reqs
        slack = base - len(self.running) - len(self._de_requeued)
        keep = []
        while self.waiting:
            r = self.waiting.pop_request()
            fresh = (r.num_computed_tokens == 0
                     and r.request_id not in self._de_requeued)
            if fresh and slack <= 0:
                self._de_held.append(r)
            else:
                keep.append(r)
                if fresh:
                    slack -= 1
        for r in keep:
            self.waiting.add_request(r)
        while self._de_held and slack > 0:
            self.waiting.add_request(self._de_held.popleft())
            slack -= 1

    def schedule(self, *args, **kwargs):
        t0 = time.monotonic()
        self._de_gate_fresh()
        self._de_sched_i += 1
        if self._de_slotstats and (
                self._de_sched_i % 100 == 0
                or len(self.running) >= (3 * self.max_num_running_reqs) // 4):
            print(f"[quail-slots] step {self._de_sched_i}: "
                  f"running {len(self.running)} "
                  f"requeued {len(self._de_requeued)} "
                  f"bound {self.max_num_running_reqs} "
                  f"base {self._de_base_max_reqs} "
                  f"waiting {len(self.waiting)} "
                  f"skipped {len(self.skipped_waiting)} "
                  f"stream {self.num_waiting_for_streaming_input} "
                  f"finished_pending {len(self.finished_req_ids)} "
                  f"preempt {getattr(self, 'num_preempted_reqs', '?')}",
                  flush=True)
        self._de_waves.before()
        out = super().schedule(*args, **kwargs)
        self._de_requeued.difference_update(out.num_scheduled_tokens)
        self._de_waves.after(out.num_scheduled_tokens.keys())
        if self._de_steps is not None:
            self._de_steps.append((time.monotonic(),
                                   out.total_num_scheduled_tokens))
        if self._de_trace:
            now = time.monotonic()
            pool = self.kv_cache_manager.block_pool
            toks = dict(out.num_scheduled_tokens)
            # generating = holds sampled output already; this makes
            # the trace's prefill/decode split exact (a one-token
            # final prefill chunk is prefill, not decode)
            decoding = {rid for rid in toks
                        if (req := self.requests.get(rid)) is not None
                        and req._output_token_ids}
            shapes = None
            if self._de_shapes:
                # schedule() has already advanced each request's
                # computed-token count by its scheduled share, so the
                # cached context is recoverable here and nowhere later
                shapes = [chainlogic.request_shape(
                              c, req.num_computed_tokens)
                          for rid, c in toks.items()
                          if (req := self.requests.get(rid)) is not None]
            rec = chainlogic.step_record(
                now, now - t0, toks,
                [t[1] if (t := self._de_tag(rid)) else None
                 for rid in toks],
                pool.get_num_free_blocks(), pool.num_gpu_blocks,
                self.block_size, decoding=decoding, shapes=shapes)
            rec["waiting"] = len(self.waiting)
            rec["running"] = len(self.running)
            # flush the full batch before appending, never after: the
            # newest record must stay in the buffer so the step's
            # execution timing can still be attached to it
            if len(self._de_trace_buf) >= 256:
                self._de_trace_flush()
            self._de_trace_buf.append(rec)
            self._de_sched_exit = now
        return out

    def update_from_output(self, *args, **kwargs):
        """The engine calls this after the step's model execution, so
        the span from schedule exit to here is the execution window
        (in the synchronous engine: the GPU wait). Attach it and the
        output-processing time to the step's trace record."""
        if not self._de_trace:
            return super().update_from_output(*args, **kwargs)
        t_in = time.monotonic()
        out = super().update_from_output(*args, **kwargs)
        t_out = time.monotonic()
        if self._de_trace_buf and self._de_sched_exit is not None:
            chainlogic.attach_step_timing(
                self._de_trace_buf[-1],
                t_in - self._de_sched_exit, t_out - t_in)
            self._de_sched_exit = None
        if self._de_flush_every:
            self._de_trace_flush()
        return out

    def reset_prefix_cache(self, *args, **kwargs):
        # the wave driver's block references would make the reset
        # refuse; a reset is a quiesce point, so drain them first
        self._de_waves.drain()
        return super().reset_prefix_cache(*args, **kwargs)

    def _de_trace_flush(self):
        if not self._de_trace_buf:
            return
        import json
        with open(self._de_trace, "a") as f:
            for rec in self._de_trace_buf:
                f.write(json.dumps(rec) + "\n")
        self._de_trace_buf = []

    def _de_dump_steps(self, label):
        if self._de_trace:
            self._de_trace_flush()
        steps = self._de_steps
        if not steps:
            return
        self._de_steps = []
        times = [t for t, _tok in steps]
        toks = [tok for _t, tok in steps]
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        gaps = sorted((times[i + 1] - times[i], times[i] - times[0])
                      for i in range(len(times) - 1))[-5:]
        small = sum(1 for tok in toks if tok < 2048)
        print(f"[quail-steps] {label}: {len(steps)} steps over {span:.2f}s, "
              f"{sum(toks)} tokens, mean {sum(toks) / len(toks):.0f} "
              f"tokens/step, {small} steps under 2048 tokens", flush=True)
        for gap, at in reversed(gaps):
            print(f"[quail-steps] {label} gap {gap * 1000:.0f}ms at "
                  f"t={at:.2f}s", flush=True)

    def _de_dump_profile(self, label):
        """Print the core process's CPU table since the last dump (or
        since the first request), then restart the counters."""
        self._de_dump_steps(label)
        if not self._de_prof:
            return
        import yappi
        yappi.stop()
        rows = [(st.tsub, st.ncall,
                 f"{st.module.split('/')[-1]}:{st.name}")
                for st in yappi.get_func_stats()]
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows) or 1.0
        print(f"[quail-prof] {label}: core CPU {total:.1f}s", flush=True)
        for tsub, ncall, fn in rows[:25]:
            print(f"[quail-prof] {label} {100 * tsub / total:5.1f}% "
                  f"{tsub:7.2f}s {ncall:>9} calls  {fn[:80]}", flush=True)
        yappi.clear_stats()
        yappi.start()

    def _de_evict(self, block_ids):
        """Plan-initiated cache-entry removal (not a heuristic event)."""
        if not block_ids:
            return
        pool = self.kv_cache_manager.block_pool
        self._de_plan_evicting = True
        try:
            pool.evict_blocks(block_ids)
        finally:
            self._de_plan_evicting = False

    # ---- chain mode ---------------------------------------------------

    def _de_register(self, request):
        qs = chainlogic.parse_registration(list(request.prompt_token_ids))
        # The questions' shared preamble survives every rewind; see
        # chainlogic.rewind_target for the cost of erasing it.
        qc = chainlogic.shared_preamble(qs)
        yes, no = chainlogic.parse_yes_no(request.request_id)
        qid = chainlogic.parse_qid(request.request_id)
        self._de_queries[qid] = dict(qs=qs, qc=qc, yes=yes, no=no)
        self._de_stats["registered"] = sum(
            len(q["qs"]) for q in self._de_queries.values())
        print(f"[quail-sched] chain registered: query {qid}, {len(qs)} "
              f"questions, lengths {[len(q) for q in qs]}, shared "
              f"preamble {qc} tokens, yes ids "
              f"{sorted(yes)}, no ids {sorted(no)}", flush=True)
        # Let the registration request finish normally: one step of a
        # tiny prompt. Aborting it here left the client waiting out a
        # 30-second timeout, because an abort before scheduling never
        # sends the client anything.
        super().add_request(request)

    def _de_rewind(self, request, d):
        """Erase everything past the boundary d: the token record, the
        block hashes, and the KV blocks. The KV below d stays
        untouched."""
        del request._all_token_ids[d:]
        del request.prompt_token_ids[d:]
        request._output_token_ids.clear()
        request.num_prompt_tokens = d
        request.num_computed_tokens = min(request.num_computed_tokens, d)
        del request.block_hashes[chainlogic.full_blocks(d, self.block_size):]
        mgr = self.kv_cache_manager.coordinator.single_type_managers[0]
        blocks = mgr.req_to_blocks.get(request.request_id)
        keep = chainlogic.retained_blocks(d, self.block_size)
        if blocks and len(blocks) > keep:
            tail = blocks[keep:]
            del blocks[keep:]
            self._de_evict({b.block_id for b in tail if not b.is_null})
            self.kv_cache_manager.block_pool.free_blocks(reversed(tail))
        if (blocks and chainlogic.partial_boundary(d, self.block_size)
                and len(blocks) >= keep):
            # the boundary block keeps positions past d that the next
            # question overwrites; its cache entry describes the old
            # content and must not survive to serve a hit
            b = blocks[keep - 1]
            if not b.is_null:
                self._de_evict({b.block_id})
        self._de_stats["rewinds"] = self._de_stats.get("rewinds", 0) + 1

    def _update_request_with_output(self, request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids)
        st = self._de_chain.get(request.request_id)
        if st is None:
            return new_token_ids, stopped
        q = self._de_queries[st["qid"]]
        out = request._output_token_ids
        tok = out[-1] if out else None
        decision = chainlogic.gate_decision(tok, stopped, q["yes"], q["no"])
        if not stopped and decision != chainlogic.KEEP_DECODING:
            # Decisive-token mode (a no set is registered): the stage
            # ends at the first token that decides the answer; the
            # trailing chatter never decodes.
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            stopped = True
        if stopped:
            st["advance"] = chainlogic.stage_advances(
                decision, st["stage"], len(q["qs"]))
            if st["advance"]:
                # The stop path reads a finish reason from the status
                # right after this returns and sends it to the client,
                # which would end the client's stream mid-chain. Erase
                # the stop status before it is read; the final stage
                # keeps its real finish and closes the stream.
                request.status = RequestStatus.RUNNING
        return new_token_ids, stopped

    def _handle_stopped_request(self, request):
        st = self._de_chain.get(request.request_id)
        if st is None:
            return super()._handle_stopped_request(request)
        self._de_stats["chain_stops"] = (
            self._de_stats.get("chain_stops", 0) + 1)
        if not st.get("advance"):
            self._de_requeued.discard(request.request_id)
            del self._de_chain[request.request_id]
            self._de_stats["chain_done"] = (
                self._de_stats.get("chain_done", 0) + 1)
            if not self._de_chain:
                print("[quail-sched] chains drained: stats "
                      f"{self._de_stats}", flush=True)
                self._de_dump_profile("chain-mode")
            return True
        st["advance"] = False
        q = self._de_queries[st["qid"]]
        self._de_rewind(request, chainlogic.rewind_target(st["d"], q["qc"]))
        st["stage"] += 1
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=chainlogic.continuation_tail(
                q["qs"][st["stage"] - 1], q["qc"]),
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params)
        self._update_request_as_session(request, update)
        # to the FRONT of the queue: the session's KV is resident and
        # finishing it frees pool and slots, so it must never wait
        # behind fresh documents the pool cannot admit
        self.waiting.prepend_request(request)
        self._de_requeued.add(request.request_id)
        return False

    def add_request(self, request):
        if self._de_prof and not self._de_prof_cleared:
            # drop engine-boot CPU so the first dump covers requests only
            import yappi
            yappi.clear_stats()
            self._de_prof_cleared = True
        rid = request.request_id
        # parse the id ONCE per request. The step trace and the free
        # path used to re-parse every scheduled request every step -
        # 1.85 million parses in 900 steps, 2.6 percent of core CPU
        self._de_rid_tag[rid] = chainlogic.parse_tag(rid)
        if chainlogic.is_registration(rid):
            self._de_register(request)
            return
        if chainlogic.is_chain(rid):
            qid = chainlogic.parse_qid(rid)
            q = self._de_queries.get(qid)
            assert q, f"chain request before query {qid}'s registration"
            self._de_chain[rid] = dict(
                stage=1, qid=qid,
                d=chainlogic.document_boundary(
                    request.num_prompt_tokens, len(q["qs"][0])))
            request.priority = 0
            # Shared scans pin the document under its chains: the pin
            # (taken by whichever of the document's chains frees first)
            # keeps the document's cache entries alive for sibling
            # queries' chains that have not computed yet, and releases
            # ride on later chain submissions like in request mode.
            tag = self._de_rid_tag[rid]
            if tag is not None:
                pin, doc, rel, uses = tag
                if rel:
                    self._de_release(rel)
                if chainlogic.new_pin_intent(pin, doc, self._de_pins):
                    self._de_intent[rid] = (pin, doc, uses)
            super().add_request(request)
            return
        tag = self._de_rid_tag[rid]
        if tag is None and self._de_strict:
            # a single tenant appliance serves only planned work
            super().add_request(request)
            self._de_stats["foreign_rejected"] += 1
            self.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
            return
        # Ordering is the plan's decision, not the client's: rank is
        # assigned here from the verified tag, and whatever priority a
        # client requested is overridden. Consumers of resident KV run
        # first, new document reads next, unplanned traffic (only
        # possible outside strict mode) last.
        request.priority = chainlogic.plan_priority(tag)
        if tag is not None:
            pin, doc, rel, uses = tag
            if rel:
                self._de_release(rel)
            if chainlogic.new_pin_intent(pin, doc, self._de_pins):
                self._de_intent[rid] = (pin, doc, uses)
        super().add_request(request)

    def _de_release(self, docs):
        for _key, blocks in self._de_pins.to_free(docs):
            if not blocks:
                continue
            # dead by the plan's decree: strip cache entries, then
            # drop the pin references; hashless blocks join the head
            # of the free queue and are reusable immediately
            self._de_evict({b.block_id for b in blocks})
            self._de_pinned_ids.difference_update(
                b.block_id for b in blocks)
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
            self._de_stats["released"] += 1
        if chainlogic.is_release_all(docs):
            print(f"[quail-sched] release-all: stats {self._de_stats}",
                  flush=True)
            self._de_dump_profile("request-mode")

    def _de_tag(self, rid):
        """The parsed tag for a request id, from the once-per-request
        cache; falls back to parsing for ids that never passed
        add_request."""
        try:
            return self._de_rid_tag[rid]
        except KeyError:
            tag = chainlogic.parse_tag(rid)
            self._de_rid_tag[rid] = tag
            return tag

    def _free_request_blocks(self, request):
        if self._de_tag(request.request_id) is not None:
            groups = self.kv_cache_manager.coordinator.get_blocks(
                request.request_id)
            blocks = groups[0] if groups else []
            intent = self._de_intent.pop(request.request_id, None)
            if chainlogic.pin_ready(intent, request.num_computed_tokens,
                                    self._de_pins):
                pin_tokens, doc, uses = intent
                keep = [b for b in blocks[:chainlogic.full_blocks(
                            pin_tokens, self.block_size)]
                        if not b.is_null]
                if keep:
                    self.kv_cache_manager.block_pool.touch(keep)
                    self._de_pins.add(doc, keep, uses)
                    self._de_pinned_ids.update(b.block_id for b in keep)
                    self._de_stats["pinned"] += 1
                    self._de_stats["blocks"] += len(keep)
            # the plan owns its memory: whatever this request leaves
            # behind unpinned has no future consumer, so strip its
            # cache entries rather than leave them to the recency rule.
            # Blocks other live requests still hold are not dead, so
            # only the last holder strips them (ref_cnt <= 1).
            tail = {b.block_id for b in blocks
                    if not b.is_null
                    and b.block_id not in self._de_pinned_ids
                    and getattr(b, "ref_cnt", 1) <= 1}
            self._de_evict(tail)
        super()._free_request_blocks(request)
        # the freed request's tag is never read again
        self._de_rid_tag.pop(request.request_id, None)
