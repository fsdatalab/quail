"""Query-aware scheduler extension for vLLM v1 (phase C).

Subclasses vLLM's scheduler so the query plan, not the engine's
heuristics, governs memory. Live documents' prefix blocks are pinned
by holding an extra block-pool reference, dead documents' blocks are
stripped and freed the instant the plan learns of the death, and every
block a planned request leaves behind that is not pinned has its cache
entry removed on free. Everything uses the block pool's public
interface; vendor code is not modified.

Single tenant mode (env DOCENGINE_SINGLE_TENANT, default on) is the
shipping configuration: requests that do not carry a plan tag are
refused, so unplanned memory cannot exist, and the engine's
keep-the-most-recent eviction rule becomes unreachable by
construction. If it ever fires anyway, that means memory escaped the
plan's accounting, and the scheduler raises instead of silently
falling back to heuristic behavior. Set the env var to 0 for shared
card experiments, where foreign traffic is legitimate and the recency
rule is its default policy.

The scheduler lives in the engine core process, so client directives
ride inside request ids. Protocol, fields separated by "|":

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

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus, StreamingUpdate

from . import chainlogic
from .chainlogic import parse_qid, parse_tag  # noqa: F401  module API


class DocEngineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._de_pins = chainlogic.PinLedger()  # doc key -> blocks, uses
        self._de_pinned_ids = set()  # block ids currently pinned
        self._de_intent = {}        # request id -> (pin_tokens, doc, uses)
        self._de_queries = {}       # query id -> dict(qs, qc, yes, no)
        self._de_chain = {}         # request id -> dict(stage, d, qid)
        self._de_strict = os.environ.get(
            "DOCENGINE_SINGLE_TENANT", "1") == "1"
        print(f"[de-sched] init: strict {self._de_strict}, overlapped "
              f"scheduling {self.scheduler_config.async_scheduling}",
              flush=True)
        self._de_plan_evicting = False
        # Core-process profiling: this object lives in the engine core,
        # the one process the client-side profiler cannot see, so the
        # profiler must be started from here.
        self._de_prof = os.environ.get("DOCENGINE_PROFILE", "0") == "1"
        self._de_prof_cleared = False
        if self._de_prof:
            import yappi
            yappi.set_clock_type("cpu")
            yappi.start()
        # Step recorder: near-zero overhead, for finding where wall
        # time goes when CPU profiling says CPU is not the problem.
        self._de_steps = ([] if os.environ.get(
            "DOCENGINE_STEPSTATS", "0") == "1" else None)
        self._de_stats = dict(pinned=0, released=0, blocks=0,
                              foreign_rejected=0, heuristic_evictions=0)

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

    def schedule(self, *args, **kwargs):
        out = super().schedule(*args, **kwargs)
        if self._de_steps is not None:
            import time
            self._de_steps.append((time.monotonic(),
                                   out.total_num_scheduled_tokens))
        return out

    def _de_dump_steps(self, label):
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
        print(f"[de-steps] {label}: {len(steps)} steps over {span:.2f}s, "
              f"{sum(toks)} tokens, mean {sum(toks) / len(toks):.0f} "
              f"tokens/step, {small} steps under 2048 tokens", flush=True)
        for gap, at in reversed(gaps):
            print(f"[de-steps] {label} gap {gap * 1000:.0f}ms at "
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
        print(f"[de-prof] {label}: core CPU {total:.1f}s", flush=True)
        for tsub, ncall, fn in rows[:25]:
            print(f"[de-prof] {label} {100 * tsub / total:5.1f}% "
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

    # ---- chain mode: one living request runs a document's whole filter
    # chain. The client registers the question token lists and the yes
    # token ids once; after that, each stage's answer is judged here,
    # the request is rewound to the document boundary (question and
    # answer KV erased, document KV kept), and the next question is
    # appended through the engine's own session mechanism.

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
        print(f"[de-sched] chain registered: query {qid}, {len(qs)} "
              f"questions, lengths {[len(q) for q in qs]}, shared "
              f"preamble {self._de_queries[qid]['qc']} tokens, yes ids "
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
            del self._de_chain[request.request_id]
            self._de_stats["chain_done"] = (
                self._de_stats.get("chain_done", 0) + 1)
            if not self._de_chain:
                print(f"[de-sched] chains drained: stats {self._de_stats}",
                      flush=True)
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
        self._enqueue_waiting_request(request)
        return False

    def add_request(self, request):
        if self._de_prof and not self._de_prof_cleared:
            # drop engine-boot CPU so the first dump covers requests only
            import yappi
            yappi.clear_stats()
            self._de_prof_cleared = True
        rid = request.request_id
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
            tag = chainlogic.parse_tag(rid)
            if tag is not None:
                pin, doc, rel, uses = tag
                if rel:
                    self._de_release(rel)
                if chainlogic.new_pin_intent(pin, doc, self._de_pins):
                    self._de_intent[rid] = (pin, doc, uses)
            super().add_request(request)
            return
        tag = chainlogic.parse_tag(request.request_id)
        if tag is None and self._de_strict:
            # a single tenant appliance serves only planned work
            super().add_request(request)
            self._de_stats["foreign_rejected"] += 1
            self.finish_requests(request.request_id,
                                 RequestStatus.FINISHED_ABORTED)
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
                self._de_intent[request.request_id] = (pin, doc, uses)
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
            print(f"[de-sched] release-all: stats {self._de_stats}",
                  flush=True)
            self._de_dump_profile("request-mode")

    def _free_request_blocks(self, request):
        if chainlogic.parse_tag(request.request_id) is not None:
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
            # cache entries rather than leave them to the recency rule
            tail = {b.block_id for b in blocks
                    if not b.is_null
                    and b.block_id not in self._de_pinned_ids}
            self._de_evict(tail)
        super()._free_request_blocks(request)
