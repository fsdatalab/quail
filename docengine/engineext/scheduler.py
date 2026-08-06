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
  s          on chain requests: speculative chain - advance through
             every stage regardless of the gate, so the query gets
             all answers from the one request (classifier sets)

Every decision - id parsing, pin refcounts, rewind arithmetic, the
answer gate, the strict-mode predicate - is computed in chainlogic.py,
which imports no vLLM and is unit-tested without an engine
(tests/test_engineext_logic.py). This class only applies those
decisions to vLLM state.
"""

import os
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus, StreamingUpdate

try:
    # forked speculation fabricates the parent's final output itself;
    # without this type the fork path disables and spec chains run
    # their stages sequentially (correct, one step per stage)
    from vllm.v1.engine import EngineCoreOutput
except ImportError:
    EngineCoreOutput = None

from . import chainlogic
from .chainlogic import parse_qid, parse_tag  # noqa: F401  module API

try:
    # the fork connector copies each sibling's partial boundary block
    # instead of recomputing it; without the module the registry is a
    # plain dict nobody reads and siblings recompute (correct, slower)
    from .forkconnector import FORK_PARTIALS
except Exception:
    FORK_PARTIALS = {}


class DocEngineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._de_pins = chainlogic.PinLedger()  # doc key -> blocks, uses
        self._de_pinned_ids = set()  # block ids currently pinned
        self._de_intent = {}        # request id -> (pin_tokens, doc, uses)
        self._de_queries = {}       # query id -> dict(qs, qc, yes, no)
        self._de_chain = {}         # request id -> dict(stage, d, qid)
        self._de_parked = {}        # parent rid -> (request, ForkBoard)
        self._de_forks = {}         # sibling rid -> (parent rid, stage)
        self._de_sib_done = set()   # sibling rids to suppress this step
        self._de_fork_ready = []    # parents whose boards filled this step
        self._de_fork_debug = 3     # print the first few boards verbatim
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
        # Step trace: one JSONL record per step (tokens, sequences,
        # prefill/decode split, unique documents, pool occupancy,
        # packing CPU), for the utilization figure. Appended in
        # batches so a run that never drains still leaves at most
        # one batch unwritten.
        self._de_trace = os.environ.get("DOCENGINE_STEPTRACE") or None
        self._de_trace_buf = []
        if self._de_trace:
            tp = self.kv_cache_manager.block_pool
            if not (hasattr(tp, "get_num_free_blocks")
                    and hasattr(tp, "num_gpu_blocks")):
                # vendor seam moved; trace degrades to off, run intact
                print("[de-trace] block pool lacks occupancy "
                      "accessors; step trace disabled", flush=True)
                self._de_trace = None
            else:
                open(self._de_trace, "w").close()
                print(f"[de-trace] step trace -> {self._de_trace}",
                      flush=True)
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
        t0 = time.monotonic()
        out = super().schedule(*args, **kwargs)
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
            self._de_trace_buf.append(chainlogic.step_record(
                now, now - t0, toks,
                [chainlogic.doc_key(rid) for rid in toks],
                pool.get_num_free_blocks(), pool.num_gpu_blocks,
                self.block_size, decoding=decoding))
            if len(self._de_trace_buf) >= 256:
                self._de_trace_flush()
        return out

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
        self._de_queries[qid] = dict(
            qs=qs, qc=qc, yes=yes, no=no,
            sep=chainlogic.parse_sep(request.request_id))
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

    # ---- forked speculation: the parent request samples stage one,
    # parks, and the scheduler fabricates one sibling sequence per
    # remaining question. The siblings share the document's cached
    # blocks, so they all fit in the next engine step; each samples
    # its one answer from its own prefill. The last answer finishes
    # the parent's stream with the answers in stage order.

    def _de_make_sibling(self, parent, doc_ids, question, stage):
        # build the sibling the way vLLM builds every real request:
        # through the constructor, so every engine-facing field gets
        # its correct initial state. A shallow copy of the parent
        # produced siblings whose attention never saw the reused
        # document blocks.
        prompt = list(doc_ids) + list(question)
        sib = type(parent)(
            request_id=chainlogic.fork_rid(parent.request_id, stage),
            prompt_token_ids=prompt,
            sampling_params=parent.sampling_params,
            pooling_params=None,
            client_index=parent.client_index,
            arrival_time=parent.arrival_time,
            lora_request=parent.lora_request,
            cache_salt=getattr(parent, "cache_salt", None),
            priority=0,
            block_hasher=None)
        # seed the hash record from the parent instead of re-hashing
        # the ~340 document tokens per sibling: block hashes are
        # content hashes, and the from-scratch recomputation was
        # measured identical (it hit the parent's cache entries).
        # Only the question tail gets hashed fresh.
        sib._block_hasher = parent._block_hasher
        sib.block_hashes = list(parent.block_hashes[
            :chainlogic.full_blocks(len(doc_ids), self.block_size)])
        if hasattr(sib, "update_block_hashes"):
            sib.update_block_hashes()
        return sib

    def _de_fork(self, request, st):
        q = self._de_queries[st["qid"]]
        n = len(q["qs"])
        first = st["stage"] + 1   # fork everything past the current stage
        doc_ids = list(request._all_token_ids[:st["d"]])
        sibs = [(self._de_make_sibling(request, doc_ids,
                                       q["qs"][stage - 1], stage), stage)
                for stage in range(first, n + 1)]
        # the shared prefix ends mid-block for most documents; hand
        # the connector each sibling's partial-block copy job (source:
        # the parent's block holding the boundary). Degrades to
        # recompute when the registry has no reader.
        shared = chainlogic.rewind_target(st["d"], q["qc"])
        r = shared % self.block_size
        if r:
            mgr = self.kv_cache_manager.coordinator.single_type_managers[0]
            pb = mgr.req_to_blocks.get(request.request_id)
            k = chainlogic.full_blocks(shared, self.block_size)
            if pb and len(pb) > k:
                for sib, _stage in sibs:
                    FORK_PARTIALS[sib.request_id] = (pb[k].block_id,
                                                     shared)
        # park before enqueueing: a sibling can finish on the very
        # next step and must find the board
        self._de_parked[request.request_id] = (
            request, chainlogic.ForkBoard(range(first, n + 1)))
        for sib, stage in sibs:
            self._de_forks[sib.request_id] = (request.request_id, stage)
            super().add_request(sib)
        self._de_stats["forks"] = (
            self._de_stats.get("forks", 0) + len(sibs))

    def _de_unpark_sequential(self, request, st):
        """Fork finalization failed: resume the parked parent on the
        sequential path from stage one, recomputing the remaining
        stages the proven way."""
        q = self._de_queries[st["qid"]]
        self._de_rewind(request, chainlogic.rewind_target(st["d"],
                                                          q["qc"]))
        st["stage"] = 2
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=chainlogic.continuation_tail(
                q["qs"][1], q["qc"]),
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params)
        self._update_request_as_session(request, update)
        self._enqueue_waiting_request(request)

    def update_from_output(self, scheduler_output, model_runner_output):
        out = super().update_from_output(scheduler_output,
                                         model_runner_output)
        if not (self._de_fork_ready or self._de_sib_done):
            return out
        groups = list(out.values()) if hasattr(out, "values") else [out]
        ready, self._de_fork_ready = self._de_fork_ready, []
        for parent_rid in ready:
            request, board = self._de_parked.pop(parent_rid)
            st = self._de_chain.get(parent_rid)
            if self._de_fork_debug > 0:
                self._de_fork_debug -= 1
                a1 = list(request._output_token_ids)
                print(f"[de-fork] {parent_rid}: stage1 sampled {a1}, "
                      f"board {dict(sorted(board.got.items()))}, "
                      f"parent d={self._de_chain.get(parent_rid, {}).get('d')}, "
                      f"sibling (cached, computed, total)="
                      f"{dict(sorted(board.diag.items()))}", flush=True)
            try:
                q_sep = None
                if st is not None:
                    q_sep = self._de_queries[st["qid"]].get("sep")
                toks = chainlogic.splice(board.in_stage_order(), q_sep)
                reason = RequestStatus.get_finished_reason(
                    RequestStatus.FINISHED_LENGTH_CAPPED)
                eco = EngineCoreOutput(request_id=parent_rid,
                                       new_token_ids=toks,
                                       finish_reason=reason)
                self._de_chain.pop(parent_rid, None)
                # the parked parent sits in no queue, so free it
                # through the internal path; finish_requests would
                # try to remove it from running and fail
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._free_request(request)
                groups[0].outputs.append(eco)
                self._de_stats["fork_docs"] = (
                    self._de_stats.get("fork_docs", 0) + 1)
            except Exception as e:
                # correctness over speed: recompute sequentially
                self._de_stats["fork_fallbacks"] = (
                    self._de_stats.get("fork_fallbacks", 0) + 1)
                print(f"[de-sched] fork finalize fallback: "
                      f"{type(e).__name__}: {e}", flush=True)
                if st is not None:
                    self._de_unpark_sequential(request, st)
        if ready and not self._de_parked and not self._de_forks:
            print(f"[de-sched] forks drained: stats {self._de_stats}",
                  flush=True)
            self._de_dump_profile("spec-fork")
        if self._de_sib_done:
            drop = self._de_sib_done
            self._de_sib_done = set()
            for g in groups:
                g.outputs = [o for o in g.outputs
                             if o.request_id not in drop]
        return out

    def _update_request_with_output(self, request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids)
        fk = self._de_forks.get(request.request_id)
        if fk is not None:
            if stopped:
                del self._de_forks[request.request_id]
                FORK_PARTIALS.pop(request.request_id, None)  # backstop
                self._de_sib_done.add(request.request_id)
                parent_rid, stage = fk
                parked = self._de_parked.get(parent_rid)
                out = request._output_token_ids
                if parked is not None:
                    if self._de_fork_debug > 0:
                        mgr = (self.kv_cache_manager.coordinator
                               .single_type_managers[0])
                        sb = mgr.req_to_blocks.get(request.request_id)
                        pb = mgr.req_to_blocks.get(parent_rid)
                        print(f"[de-fork-blocks] stage {stage}: sibling "
                              f"{[b.block_id for b in (sb or [])[:6]]} "
                              f"parent "
                              f"{[b.block_id for b in (pb or [])[:6]]}",
                              flush=True)
                    parked[1].diag[stage] = (
                        getattr(request, "num_cached_tokens", -1),
                        request.num_computed_tokens,
                        len(request._all_token_ids))
                    if parked[1].record(stage,
                                        list(out) if out else [-1]):
                        self._de_fork_ready.append(parent_rid)
            return new_token_ids, stopped
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
            st["advance"] = (
                chainlogic.spec_advances(st["stage"], len(q["qs"]))
                if st.get("spec")
                else chainlogic.stage_advances(
                    decision, st["stage"], len(q["qs"])))
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
        # the sq part on the request id forces the sequential path per
        # request (the control for fork validation); the env var is
        # the boot-time master switch, read in the core process
        forks_on = (EngineCoreOutput is not None and not st.get("seq")
                    and os.environ.get("DOCENGINE_FORKS", "1") == "1")
        # two fork triggers: a speculative chain forks everything
        # after its first answer; a hybrid gated chain forks the
        # remaining filters once it passes its switch stage, where
        # the survivors are too few to fill rounds with gated work
        at_fork = ((st.get("spec") and st["stage"] == 1)
                   or (st.get("switch") is not None
                       and st["stage"] == st["switch"]))
        if at_fork and forks_on:
            # forked speculation: park the parent and let fabricated
            # siblings answer stages 2..n in the next engine step; on
            # any failure fall through to the sequential advance
            try:
                self._de_fork(request, st)
                return False
            except Exception as e:
                self._de_stats["fork_fallbacks"] = (
                    self._de_stats.get("fork_fallbacks", 0) + 1)
                print(f"[de-sched] fork fallback: {type(e).__name__}: "
                      f"{e}", flush=True)
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
                stage=1, qid=qid, spec=chainlogic.is_spec(rid),
                seq=chainlogic.is_seq(rid),
                switch=chainlogic.parse_switch(rid),
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
