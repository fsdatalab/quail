"""Quail offloading connector: one transfer per step, not per request.

Boot with

    KVTransferConfig(
        kv_connector="QuailOffloadingConnector",
        kv_connector_module_path="quail.engineext.offload",
        kv_role="kv_both", ...)

Identical to vLLM's OffloadingConnector except on the worker side:
the stock worker submits each request's load as its own transfer job,
and the handler's strict job ordering prices every job at ~1.7 ms of
overhead against ~0.6 ms of copying (measured,
results/engine/persist_xfer_trace.jsonl.gz). This subclass merges all
load jobs the scheduler created in one engine step into a single
batched transfer, then fans the finished transfer back out into
per-job results so scheduler-side bookkeeping and stats see exactly
the jobs it created.

Merging is only descriptor concatenation when the offload chunking is
trivial (one KV group, CPU chunk == GPU block); the decision lives in
offload_logic.mergeable. Jobs that do not qualify are submitted
unmerged, so this class degrades to stock behavior rather than
guessing at offset arithmetic.

Depends on vLLM 0.26.0 internals, pinned in the image for the same
reason the scheduler subclass pins it: OffloadingConnectorWorker and
its handlers are not public API.
"""

from dataclasses import dataclass, field
from itertools import count

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    TransferResult,
    make_offload_key,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from .offload_logic import plan_merge, split_result, synthetic_ids


@dataclass
class QuailOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """The stock metadata plus the wave channel. Wave jobs fill
    pre-registered prefix-cache blocks and belong to no request;
    gate_waves names the waves whose blocks a request scheduled THIS
    step may read, so the worker makes the compute stream wait on
    each wave's CUDA event exactly once. All ordering is stream
    ordering - no request ever parks waiting for a transfer."""
    wave_of_job: dict = field(default_factory=dict)   # job id -> wave id
    gate_waves: list = field(default_factory=list)


class QuailOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    """Adds the wave channel on the scheduler side. The QuailScheduler
    lives in the same process and calls queue_wave/queue_gates
    directly; the queued work rides the step's connector metadata to
    the worker. Wave jobs are tracked here only to release their CPU
    block pins when the worker reports the copy done."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wave_jobs = []            # (wave_id, job_id, TransferJob)
        self._wave_keys = {}            # job_id -> store keys to release
        self._job_wave = {}             # job id -> wave id, until done
        self._gate_queue = []
        # far below the coalescing worker's synthetic range, so the
        # three id spaces (scheduler jobs >= 0, merges < 0, waves
        # <= -10^9) can never collide
        self._wave_ids = count(start=-(10 ** 9), step=-1)
        # request ids the wave driver owns: their store lookup answers
        # no-hit, so the vendor never opens a per-request async load
        # for a document a wave is already copying. The driver adds on
        # claim and removes at registration.
        self.wave_claimed = set()
        # wave ids whose copy the worker reported finished; the driver
        # drains this to register waves only after their bytes landed
        self.finished_waves = set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # a deferred external load charges nothing against the step
        # budget (vendor scheduler: load_kv_async -> num_new_tokens =
        # 0), so the vendor would sweep the whole queue in one step
        # and duplicate every wave's bytes through per-request loads -
        # measured 1.6x load duplication. A claimed document reads as
        # cache-cold instead: at worst the vendor prefills it, which
        # charges full tokens and is self-limiting at one step budget.
        if request.request_id in self.wave_claimed:
            return 0, False
        return super().get_num_new_matched_tokens(
            request, num_computed_tokens)

    def wave_keys_for(self, request):
        """The request's store keys at chunk granularity, or None if
        any chunk misses (a partial hit is not worth a wave). Only
        full blocks are considered; the tail recomputes."""
        n_full = len(request.block_hashes)
        if not n_full:
            return None
        keys = [make_offload_key(h, 0)
                for h in request.block_hashes]
        for key in keys:
            if self.manager.lookup(key, None) != LookupResult.HIT:
                return None
        return keys

    def prepare_wave(self, keys):
        """Pin the CPU blocks for a wave and return the source spec."""
        return self.manager.prepare_load(keys, None)

    def queue_wave(self, wave_id, keys, src_spec, dst_spec):
        job_id = next(self._wave_ids)
        self._wave_jobs.append((wave_id, job_id, TransferJob(
            req_id=f"quail-wave-{wave_id}", src_spec=src_spec,
            dst_spec=dst_spec)))
        self._wave_keys[job_id] = list(keys)
        self._job_wave[job_id] = wave_id
        return job_id

    def queue_gates(self, wave_ids):
        self._gate_queue.extend(wave_ids)

    def build_connector_meta(self, *args, **kwargs):
        meta = super().build_connector_meta(*args, **kwargs)
        wave_of_job = {}
        for wave_id, job_id, job in self._wave_jobs:
            meta.load_jobs[job_id] = job
            wave_of_job[job_id] = wave_id
        out = QuailOffloadingConnectorMetadata(
            load_jobs=meta.load_jobs, store_jobs=meta.store_jobs,
            jobs_to_flush=meta.jobs_to_flush,
            wave_of_job=wave_of_job, gate_waves=list(self._gate_queue))
        self._wave_jobs = []
        self._gate_queue = []
        return out

    def update_connector_output(self, connector_output):
        meta = connector_output.kv_connector_worker_meta
        if (isinstance(meta, OffloadingWorkerMetadata)
                and meta.completed_jobs):
            for job_id in [j for j in meta.completed_jobs
                           if j in self._wave_keys]:
                keys = self._wave_keys.pop(job_id)
                self.finished_waves.add(self._job_wave.pop(job_id))
                try:
                    self.manager.complete_load(keys, None)
                except Exception as e:
                    print(f"[quail-waves] complete_load failed: {e}",
                          flush=True)
                del meta.completed_jobs[job_id]
        super().update_connector_output(connector_output)


class QuailOffloadingConnectorWorker(OffloadingConnectorWorker):
    def __init__(self, spec, kv_cache_config):
        super().__init__(spec, kv_cache_config)
        # synthetic merged-job id -> list of (job_id, n_blocks)
        self._merged: dict[int, list[tuple[int, int]]] = {}
        self._synthetic = synthetic_ids()
        self._wave_events = {}          # wave id -> CUDA end event
        self._wave_job_ids = set()      # job ids that were waves
        self._gated = set()             # waves already waited on
        self._wave_handler_obj = None   # dedicated wave transfer chain

    def _wave_handler(self):
        """A dedicated transfer chain for waves, built lazily from the
        load handler's own tensors. Transfers within one handler are
        strictly ordered; giving waves their own keeps a wave's gate
        from waiting behind unrelated on-demand loads."""
        if self._wave_handler_obj is None:
            from vllm.v1.kv_offload.cpu.gpu_worker import (
                SingleDirectionOffloadingHandler)
            load = self.worker._load_handler
            self._wave_handler_obj = SingleDirectionOffloadingHandler(
                gpu_tensors=load.dst_tensors,
                cpu_tensors=load.src_tensors,
                blocks_per_chunk=load.src_blocks_per_chunk,
                kv_cache_groups_data_refs=load.kv_cache_groups_data_refs,
                gpu_to_cpu=False)
        return self._wave_handler_obj

    def apply_gates(self, wave_ids):
        """Make the compute stream wait on each named wave's event,
        once. A wait on an already-signaled event is free; on a
        pending one it orders the step after the copy - which is the
        entire synchronization story of the wave design."""
        import torch
        for wave_id in wave_ids:
            if wave_id in self._gated:
                continue
            self._gated.add(wave_id)
            ev = self._wave_events.pop(wave_id, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)

    def _src_blocks_per_chunk(self) -> int:
        # the load handler's CPU-side chunking; 1 means CPU chunks and
        # GPU blocks are the same size and merges are offset-free
        return self.worker._load_handler.src_blocks_per_chunk

    def _check_pinned(self):
        """Pinned host memory is the only intended tier: the measured
        rates are 55 GB/s raw and 38 GB/s through batched transfers,
        against ~11 GB/s unpinned. A failed pin still works, so it
        must be loud - the restore break-even (7.1 GB/s at 4B) is
        priced on the pinned rate."""
        try:
            cpu = self.worker._load_handler.src_tensors
            pinned = bool(cpu) and all(t.is_pinned() for t in cpu)
        except Exception:
            return
        if not pinned:
            print("[quail-offload] WARNING: host KV pool is NOT pinned; "
                  "transfers fall to unpinned DMA (~11 GB/s measured "
                  "against 55 pinned) - reprice restore before trusting "
                  "the plan", flush=True)

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        self._check_pinned()

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        super().register_cross_layers_kv_cache(kv_cache, attn_backend)
        self._check_pinned()

    def shutdown(self):
        if self._wave_handler_obj is not None:
            self._wave_handler_obj.shutdown()
            self._wave_handler_obj = None
        super().shutdown()

    def start_kv_transfers(self, metadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert self.worker.submit_store(job_id, src_spec, dst_spec)
        self._unsubmitted_store_jobs.clear()

        # wave jobs first: each is already one batched copy list, and
        # it runs on the wave handler's OWN chain, so a wave's gate
        # event waits only on its own copies - waves sharing the
        # on-demand chain leaked ~1.9 s of transfer backlog into
        # compute (measured). The transfer's end event becomes the
        # wave's gate event.
        wave_of_job = getattr(metadata, "wave_of_job", None) or {}
        if wave_of_job:
            handler = self._wave_handler()
            for job_id in list(wave_of_job):
                entry = metadata.load_jobs.pop(job_id, None)
                if entry is None:
                    continue
                self._wave_job_ids.add(job_id)
                assert handler.transfer_async(
                    job_id, entry.src_spec, entry.dst_spec)
                try:
                    ev = handler._transfers[-1].end_event
                except Exception:
                    ev = None
                if ev is not None:
                    self._wave_events[wave_of_job[job_id]] = ev

        load_items = metadata.load_jobs
        if len(load_items) < 2:
            for job_id, entry in load_items.items():
                self._load_jobs[job_id] = entry.req_id
                assert self.worker.submit_load(
                    job_id, entry.src_spec, entry.dst_spec)
            return

        shapes = []
        for job_id, entry in load_items.items():
            dst = entry.dst_spec
            assert isinstance(dst, GPULoadStoreSpec)
            shapes.append((job_id, len(entry.src_spec.block_ids),
                           len(dst.block_ids), len(dst.group_sizes)))
        merged_ids, single_ids = plan_merge(
            shapes, self._src_blocks_per_chunk())

        for job_id in single_ids:
            entry = load_items[job_id]
            self._load_jobs[job_id] = entry.req_id
            assert self.worker.submit_load(
                job_id, entry.src_spec, entry.dst_spec)

        if not merged_ids:
            return
        src_blocks, dst_blocks, parts = [], [], []
        src_cls = type(load_items[merged_ids[0]].src_spec)
        for job_id in merged_ids:
            entry = load_items[job_id]
            self._load_jobs[job_id] = entry.req_id
            n = len(entry.src_spec.block_ids)
            src_blocks.extend(int(b) for b in entry.src_spec.block_ids)
            dst_blocks.extend(int(b) for b in entry.dst_spec.block_ids)
            parts.append((job_id, n))
        merged_id = next(self._synthetic)
        self._merged[merged_id] = parts
        # block_indices is only read for partial-first-chunk skips,
        # which mergeable() has ruled out; zero is the aligned value
        merged_dst = GPULoadStoreSpec(
            block_ids=dst_blocks,
            group_sizes=[len(dst_blocks)],
            block_indices=[0])
        assert self.worker.submit_load(
            merged_id, src_cls(src_blocks), merged_dst)

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        results = list(self.worker.get_finished())
        if self._wave_handler_obj is not None:
            results.extend(self._wave_handler_obj.get_finished())
        finished_recving: set[str] = set()
        for result in self._expand(results):
            job_id = result.job_id
            assert result.success
            is_load = (job_id in self._load_jobs
                       or job_id in self._wave_job_ids)
            self._wave_job_ids.discard(job_id)
            if (result.transfer_time is not None
                    and result.transfer_size is not None):
                stats = (self._connector_worker_meta.transfer_stats.load
                         if is_load else
                         self._connector_worker_meta.transfer_stats.store)
                stats.record(result.transfer_size, result.transfer_time)
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def _expand(self, results):
        """Fan a merged transfer's result out into its constituents so
        every downstream consumer sees the scheduler's own job ids."""
        out = []
        for result in results:
            parts = self._merged.pop(result.job_id, None)
            if parts is None:
                out.append(result)
                continue
            total_blocks = sum(n for _, n in parts)
            times = split_result(result.transfer_time or 0.0,
                                 [n for _, n in parts])
            for (job_id, n), t in zip(parts, times):
                size = ((result.transfer_size or 0) * n // total_blocks
                        if total_blocks else 0)
                out.append(TransferResult(
                    job_id=job_id, success=result.success,
                    transfer_size=size, transfer_time=t))
        return out


class QuailOffloadingConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            # rebuild the worker as the coalescing subclass; __init__
            # only stores references, so the swap has no side effects
            self.connector_worker = QuailOffloadingConnectorWorker(
                self.connector_worker.spec, kv_cache_config)
        if self.connector_scheduler is not None:
            # rebuild the scheduler side with the wave channel; the
            # parent's instance holds no state yet at this point
            offloading_config = build_offloading_config(
                vllm_config, kv_cache_config)
            spec = OffloadingSpecFactory.create_spec(offloading_config)
            self.connector_scheduler = QuailOffloadingConnectorScheduler(
                spec, vllm_config, kv_cache_config)

    def start_load_kv(self, forward_context, **kwargs):
        gates = getattr(self._connector_metadata, "gate_waves", None)
        if gates:
            assert self.connector_worker is not None
            self.connector_worker.apply_gates(gates)
        super().start_load_kv(forward_context, **kwargs)
