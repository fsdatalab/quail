"""Greedy generation for many documents at once over the KV arena.

run_stages answers a fixed list of stages whose request tokens are
known before the run. Generation is different: each document's next
request is the token the model just chose, for up to thousands of
rounds. GreedyGenerator schedules that work with the same pieces the
stage scheduler uses: the arena's pages and state slots, pack_chunk,
and the pipeline's forward pass.

Each step decodes one token for every document in flight and, when
lanes are free, prefills the prompts of newly admitted documents. A
prompt's last token is its one suffix, so the pass that prefills it
also gives its first output token. A decode row appends the last
chosen token to the document's KV and, on a model with linear-attention
layers, steps the state kept in its slot in place. The decode rows
replay a CUDA graph captured for their batch size, and a step's prompts
prefill in a pass of their own: a pass mixing the two runs every
decode row eagerly, which on LightOnOCR-3-0.8B cost more than it saved
(ParseBench run 20261012T015625Z-1df818). Without graphs, one pass
packs both.

A document leaves at a stop token, its output limit, or its context,
and its lane takes the next document.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Generation:
    """One document to generate from.

    Attributes:
        key: Arena key, unique among the documents in flight.
        prompt: The prompt's token ids.
        rope: Rotary positions with one row per axis, for a model whose
            rotary reads several; None counts up from 0 on every axis.
        rope_offset: Added to a generated row's sequence index to give
            its position on every axis.
        image: Optional callable returning the prompt rows that are not
            token embeddings and their embeddings on the current stream.
    """

    key: Any
    prompt: np.ndarray
    rope: np.ndarray | None = None
    rope_offset: int = 0
    image: Callable | None = None


@dataclass
class _Document:
    """A document in flight: its lane, KV rows, and the tokens chosen so far."""

    generation: Generation
    lane: int
    capacity: int
    rows: np.ndarray
    slot: int = 0
    tokens: list = field(default_factory=list)
    launched: int = 0    # decode steps launched, read or not
    finished: bool = False

    @property
    def fed(self) -> int:
        """KV rows written: the prompt and every chosen token but the last."""
        return len(self.generation.prompt) + len(self.tokens) - 1


def admission_ready(free: int, decoding: int, mean_prompt: float,
                    budget: int, exhausted: bool) -> bool:
    """Whether a step should prefill now rather than decode alone.

    A step's token budget holds its decode rows and its prefills. With
    documents decoding, a step admits once the free lanes can take a
    full budget of prompts, since a step that prefills cannot replay a
    decode graph and pays its packing on the host; with none decoding,
    or with no more documents coming, it admits whatever it can.

    Args:
        free: Free lanes.
        decoding: Documents decoding.
        mean_prompt: Mean prompt length so far.
        budget: The step's token budget.
        exhausted: Whether every document has been taken from the input.
    """
    if not free:
        return False
    if not decoding or exhausted:
        return True
    return free * mean_prompt >= budget - decoding


class _Admission:
    """The documents waiting to start, taken as the admission rule allows."""

    def __init__(self, pending, generator):
        self.pending = pending
        self.generator = generator
        self.waiting = None
        self.exhausted = False
        self.prompt_tokens = 0
        self.prompt_count = 0

    @property
    def done(self) -> bool:
        """Whether every document has been taken."""
        return self.exhausted and self.waiting is None

    def take(self, free: int, decoding: int) -> list:
        """Reserve and return (Generation, capacity) pairs for this step."""
        import time

        generator = self.generator
        mean = self.prompt_tokens / self.prompt_count if self.prompt_count else 0.0
        if not admission_ready(free, decoding, mean, generator.budget,
                               self.exhausted):
            return []
        admit = []
        room = generator.budget - decoding
        while len(admit) < free and not self.exhausted:
            if self.waiting is None:
                started = time.perf_counter()
                self.waiting = next(self.pending, None)
                generator.stats["wait_s"] += time.perf_counter() - started
                if self.waiting is None:
                    self.exhausted = True
                    break
            size = len(self.waiting.prompt)
            if admit and size > room:
                break
            capacity = generator._reserve(self.waiting)
            if capacity is None:
                # the arena is full: the prompt waits for documents to
                # finish and free their pages
                break
            admit.append((self.waiting, capacity))
            generator.stats["prefilled_tokens"] += size
            self.prompt_tokens += size
            self.prompt_count += 1
            room -= size
            self.waiting = None
        return admit


class GreedyGenerator:
    """Generate greedily for documents streamed through a fixed set of lanes.

    Args:
        torch: The torch module.
        arena: KVArena holding the documents' KV and state slots.
        pipeline: The model's ModelPipeline.
        head: The output head, (vocabulary, hidden), for greedy logits.
        lanes: Documents in flight at most.
        budget: Rows one step packs at most, decode rows included.
        max_new_tokens: Tokens a document generates at most.
        context_tokens: Prompt and generated tokens a document holds at most.
        stops: Token ids that end a document's generation.
        graphs: Whether decode-only steps replay captured CUDA graphs.
        profile_mixed: Which mixed step, counted from 1, to run under
            torch.profiler, listing its kernels in stats; 0 for none.
    """

    def __init__(self, torch, arena, pipeline, head, *, lanes: int, budget: int,
                 max_new_tokens: int, context_tokens: int, stops,
                 graphs: bool = True, profile_mixed: int = 0):
        self.torch = torch
        self.arena = arena
        self.pipeline = pipeline
        self.head = head
        self.lanes = lanes
        self.budget = budget
        self.max_new_tokens = max_new_tokens
        self.context_tokens = context_tokens
        self.stops = frozenset(stops)
        self.graphs = graphs
        self.profile_mixed = profile_mixed
        self.width = arena.pages_needed(context_tokens)
        self.staging = None
        if str(arena.device).startswith("cuda"):
            from quail.backends.quail.executor.chunk import InputStaging

            self.staging = InputStaging(torch)
        self.stats = {}
        if graphs:
            self._capture()

    # ---- the run -------------------------------------------------------

    def run(self, generations):
        """Generate for every document; yields (key, tokens) as each ends."""
        self.stats = dict(wait_s=0.0, mixed_s=0.0, mixed_steps=0,
                          mixed_pack_s=0.0, prefilled_tokens=0, decode_s=0.0,
                          decode_steps=0, decode_rows=0, decode_host_s=0.0,
                          decode_wait_s=0.0, discarded_rows=0)
        admission = _Admission(iter(generations), self)
        try:
            if self.graphs:
                yield from self._run_graphs(admission)
            else:
                yield from self._run_eager(admission)
        finally:
            # a run its consumer stopped early still holds lanes
            self._free_documents()

    def _free_documents(self) -> None:
        """Free every document's pages and slot; the graphs' scratch lane stays."""
        scratch = getattr(self, "_scratch", None)
        held = [key for key in self.arena.resident_keys() if key != scratch]
        if held and self.graphs:
            # a launched step may still write the lanes
            self.torch.cuda.synchronize()
        for key in held:
            self.arena.free_key(key)

    def _run_eager(self, admission):
        """One pass per step, packing the decode rows with the admitted prompts."""
        import time

        stats = self.stats
        active, free = [], list(range(self.lanes - 1, -1, -1))
        while True:
            admit = admission.take(len(free), len(active))
            started = time.perf_counter()
            if admit:
                tokens = self._mixed(active, admit, free)
                stats["mixed_s"] += time.perf_counter() - started
                stats["mixed_steps"] += 1
            elif active:
                tokens = self._mixed(active, [], free)
                stats["decode_s"] += time.perf_counter() - started
                stats["decode_steps"] += 1
                stats["decode_rows"] += len(active)
            elif admission.done:
                return
            else:
                raise RuntimeError("the arena cannot hold one prompt")
            still = []
            for document, token in zip(active, tokens):
                document.tokens.append(token)
                if self._done(document):
                    self._release(document, free)
                    yield document.generation.key, document.tokens
                else:
                    still.append(document)
            active = still

    def _run_graphs(self, admission):
        """Prefill admitted prompts in their own pass and decode by graph, ahead.

        Each decode step launches before the host reads the step
        before it: the graph reads each lane's last token from the
        GPU, so the host only counts rows to lay out the next step,
        and reads the chosen tokens one step late. A document that
        stopped in the step being read has already run one more
        step, whose token is discarded.
        """
        import time

        stats = self.stats
        active, free = [], list(range(self.lanes - 1, -1, -1))
        flight = None    # the launched step not read yet
        while True:
            admit = admission.take(len(free), len(active))
            if admit:
                started = time.perf_counter()
                admitted = []
                first = self._mixed(admitted, admit, free)
                stats["mixed_s"] += time.perf_counter() - started
                stats["mixed_steps"] += 1
                for document, token in zip(admitted, first):
                    document.tokens.append(token)
                    if self._done(document):
                        self._release(document, free)
                        yield document.generation.key, document.tokens
                    else:
                        active.append(document)
            launch = [document for document in active if self._launchable(document)]
            started = time.perf_counter()
            launched = self._launch(launch) if launch else None
            stats["decode_host_s"] += time.perf_counter() - started
            if flight is not None:
                started = time.perf_counter()
                tokens = self._read(flight)
                stats["decode_wait_s"] += time.perf_counter() - started
                for document, token in zip(flight[0], tokens):
                    if document.finished:
                        stats["discarded_rows"] += 1
                        continue
                    document.tokens.append(token)
                    if self._done(document):
                        document.finished = True
                        self._release(document, free)
                        yield document.generation.key, document.tokens
                active = [document for document in active if not document.finished]
            flight = launched
            if launched is None and not active:
                if admission.done:
                    return
                if not admit:
                    raise RuntimeError("the arena cannot hold one prompt")

    def _reserve(self, generation):
        """Reserve the prompt's pages and slot; returns its capacity, or None."""
        f = len(generation.prompt) - 1
        capacity = min(len(generation.prompt) + self.max_new_tokens,
                       self.context_tokens)
        if self.arena.activate(generation.key, f, capacity_tokens=capacity,
                               base_tokens=f, slots=1) is None:
            return None
        return capacity

    def _done(self, document) -> bool:
        return (document.tokens[-1] in self.stops
                or len(document.tokens) >= self.max_new_tokens
                or document.fed + 1 >= document.capacity)

    def _launchable(self, document) -> bool:
        """Whether one more decode step stays within the document's limits.

        Every launched step writes one row and yields one token, so
        the tokens after it count one per launched step and one from
        the prompt's pass.
        """
        tokens = 1 + document.launched + 1
        prompt = len(document.generation.prompt)
        return (tokens <= self.max_new_tokens
                and prompt + tokens <= document.capacity)

    def _release(self, document, free):
        self.arena.free_key(document.generation.key)
        free.append(document.lane)

    # ---- a step that packs prompts --------------------------------------

    def _mixed(self, active, admit, free) -> list:
        """Decode every document in flight and prefill the admitted ones.

        Args:
            active: The documents in flight; the admitted ones join it
                after them, in admission order.
            admit: (Generation, capacity) pairs whose pages are reserved.
            free: The free lanes.

        Returns:
            One token per document of the new `active`, in its order.
        """
        import time

        from quail.backends.quail.executor.chunk import pack_chunk
        from quail.backends.quail.executor.loop import _forward

        torch = self.torch
        arena = self.arena
        started = time.perf_counter()
        groups, rope = [], []
        for document in active:
            f = document.fed
            # decode graphs step the kept state without telling the
            # arena; its record moves to the state's position here
            arena.claim_state(document.generation.key, f, "kept")
            groups.append(dict(
                key=document.generation.key, prefix=None, f=f,
                suffixes=[[document.tokens[-1]]], write_suffix_tokens=1,
                save_at=(f + 1,), decode=True))
            rope.append(np.full((3, 1), f + document.generation.rope_offset,
                                dtype=np.int64))
        for generation, _ in admit:
            prompt = np.asarray(generation.prompt, dtype=np.int64)
            f = len(prompt) - 1
            groups.append(dict(
                key=generation.key, prefix=prompt[:-1], start=0, f=f,
                suffixes=[prompt[-1:]], write_suffix_tokens=1, single=True,
                save_at=(f + 1,)))
            rope.append(generation.rope if generation.rope is not None
                        else np.broadcast_to(np.arange(f + 1), (3, f + 1)))
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                           staging=self.staging)
        chunk.rope_positions = torch.as_tensor(
            np.concatenate(rope, axis=1)).to(arena.device, non_blocking=True)
        if any(generation.image is not None for generation, _ in admit):
            chunk.inputs_embeds = self._embeds(chunk, active, admit)
        self.stats["mixed_pack_s"] += time.perf_counter() - started
        timed = str(arena.device).startswith("cuda")
        if timed:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True)
            begin.record()
        lanes = [free.pop() for _ in admit]
        profiling = self.stats["mixed_steps"] + 1 == self.profile_mixed
        if profiling:
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
            profile_started = time.perf_counter()
        with torch.inference_mode():
            hidden = _forward(self.pipeline, arena, chunk)
            tokens = (hidden @ self.head.t()).argmax(dim=-1)
            if self.graphs and admit:
                # the decode graphs read each lane's last token here
                self._last.index_copy_(
                    0, torch.as_tensor(lanes, device=arena.device),
                    tokens[len(tokens) - len(admit):])
        if timed:
            end.record()
        tokens = tokens.tolist()
        if profiling:
            wall_ms = (time.perf_counter() - profile_started) * 1e3
            profiler.__exit__(None, None, None)
            self.stats["mixed_profile"] = dict(
                _kernels(profiler, len(groups), chunk.tokens), wall_ms=wall_ms)
        if timed:
            self.stats["mixed_gpu_s"] = (self.stats.get("mixed_gpu_s", 0.0)
                                         + begin.elapsed_time(end) / 1e3)
        for (generation, capacity), lane in zip(admit, lanes):
            key = generation.key
            document = _Document(
                generation, lane, capacity,
                arena.capacity_rows(key).numpy(),
                slot=arena.state_slot_at(key, len(generation.prompt)) or 0)
            active.append(document)
            if self.graphs:
                self._write_lane(document)
        return tokens

    def _embeds(self, chunk, active, admitted):
        """Token embeddings with the admitted prompts' image rows written over."""
        torch = self.torch
        with torch.inference_mode():
            embeds = self.pipeline.embed(chunk.input_ids)
            row = len(active)
            for generation, _ in admitted:
                if generation.image is not None:
                    rows, values = generation.image()
                    index = torch.as_tensor(rows + row, device=self.arena.device)
                    embeds.index_copy_(0, index, values.to(embeds.dtype))
                row += len(generation.prompt)
        return embeds

    # ---- a step that only decodes: CUDA graphs --------------------------

    def _capture(self):
        """Capture the decode step as a CUDA graph per batch size.

        The graphs read static buffers: each lane's block table,
        written once when a document takes the lane, and each step's
        tokens, positions, KV rows, lengths, lanes, and slots. Padding
        rows read a scratch lane and slot 0, which the decode kernels
        skip.
        """
        torch = self.torch
        arena = self.arena
        n = self.lanes
        self._scratch = ("generate", "scratch")
        if arena.alloc(self._scratch, 1, capacity_tokens=arena.page_tokens) is None:
            raise RuntimeError("no arena page for the decode graphs' scratch lane")
        scratch_row = int(arena.capacity_rows(self._scratch)[0])
        self._table = torch.zeros((n + 1, self.width), dtype=torch.int32,
                                  device="cuda")
        self._table[n, 0] = arena.owned_pages(self._scratch)[0]
        self._pad = dict(positions=0, dst=scratch_row, used=1, lanes=n, slots=0)
        dtypes = dict(positions=torch.int64, dst=torch.int64, used=torch.int32,
                      lanes=torch.int64, slots=torch.int32)
        # two sets of host buffers: a launch fills one while the copies
        # of the launch before it may still be pending from the other
        self._host = [{}, {}]
        self._device = {}
        for name, dtype in dtypes.items():
            width = 3 * n if name == "positions" else n
            for host in self._host:
                host[name] = torch.full((width,), self._pad[name], dtype=dtype,
                                        pin_memory=True)
            self._device[name] = self._host[0][name].to("cuda")
        self._chosen = [torch.zeros(n, dtype=torch.int64, pin_memory=True)
                        for _ in range(2)]
        self._parity = 0
        # each lane's last chosen token, which its next decode row feeds
        self._last = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
        self._out = torch.zeros(n, dtype=torch.int64, device="cuda")
        self._cu = torch.arange(n + 1, dtype=torch.int32, device="cuda")
        self._index = torch.arange(n, device="cuda")
        self._sizes = sorted({min(1 << k, n) for k in range(n.bit_length() + 1)})
        pool = torch.cuda.graph_pool_handle()
        self._graphs = {}
        with torch.inference_mode():
            for size in reversed(self._sizes):
                # the eager run compiles and tunes every kernel first
                self._decode_rows(size)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._decode_rows(size)
                self._graphs[size] = graph
        torch.cuda.synchronize()

    def _decode_rows(self, size: int):
        """One decode step over the first `size` rows of the static buffers."""
        from types import SimpleNamespace

        d = self._device
        meta = dict(
            layer=0,
            unified=dict(
                cu_q=self._cu[:size + 1], max_q=1, src=self._index[:size],
                dst=d["dst"][:size], tail_src=None, used=d["used"][:size],
                max_used=self.width * self.arena.page_tokens,
                table=self._table.index_select(0, d["lanes"][:size])),
            state=dict(waves=[], decode=dict(rows=None, slots=d["slots"][:size])))
        lanes = d["lanes"][:size]
        ids = self._last.index_select(0, lanes)
        chunk = SimpleNamespace(
            input_ids=ids, positions=ids,
            rope_positions=d["positions"][:3 * size].view(3, size),
            meta=meta, final_indices=self._index[:size])
        hidden = self.pipeline.forward_chunk(chunk)
        chosen = (hidden @ self.head.t()).argmax(dim=-1)
        self._out[:size].copy_(chosen)
        self._last.index_copy_(0, lanes, chosen)

    def _write_lane(self, document):
        """Write the document's pages into its lane's block table row."""
        pages = self.arena.owned_pages(document.generation.key)
        row = self.torch.zeros(self.width, dtype=self.torch.int32)
        row[:len(pages)] = self.torch.as_tensor(pages, dtype=self.torch.int32)
        self._table[document.lane].copy_(row)

    def _launch(self, documents):
        """Lay out and replay one decode step without waiting for its tokens.

        Returns what _read needs: the documents in row order, their
        count, the event after the tokens' copy to the host, and the
        host buffer they land in.
        """
        torch = self.torch
        n = len(documents)
        size = next(s for s in self._sizes if s >= n)
        host = self._host[self._parity]
        rows = [len(d.generation.prompt) + d.launched for d in documents]
        values = dict(
            dst=[int(d.rows[row]) for d, row in zip(documents, rows)],
            used=[row + 1 for row in rows],
            lanes=[d.lane for d in documents],
            slots=[d.slot for d in documents])
        for name, value in values.items():
            buffer = host[name].numpy()
            buffer[:n] = value
            buffer[n:size] = self._pad[name]
            self._device[name][:size].copy_(host[name][:size], non_blocking=True)
        positions = host["positions"].numpy()[:3 * size].reshape(3, size)
        positions[:, :n] = [row + d.generation.rope_offset
                            for row, d in zip(rows, documents)]
        positions[:, n:] = 0
        self._device["positions"][:3 * size].copy_(
            host["positions"][:3 * size], non_blocking=True)
        chosen = self._chosen[self._parity]
        with torch.inference_mode():
            self._graphs[size].replay()
            chosen[:n].copy_(self._out[:n], non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        for document in documents:
            document.launched += 1
        self.stats["decode_steps"] += 1
        self.stats["decode_rows"] += n
        flight = (documents, n, done, chosen)
        self._parity ^= 1
        return flight

    def _read(self, flight) -> list:
        """The tokens a launched decode step chose, in its row order."""
        _, n, done, chosen = flight
        done.synchronize()
        return chosen[:n].tolist()

def _kernels(profiler, groups: int, rows: int) -> dict:
    """A profiled step's GPU time by kernel and CPU time by op, largest first."""
    events = [(getattr(e, "self_device_time_total", 0), e.count, e.key)
              for e in profiler.key_averages()]
    events = sorted((e for e in events if e[0] > 0), reverse=True)
    cpu = sorted(((e.self_cpu_time_total, e.count, e.key)
                  for e in profiler.key_averages()), reverse=True)
    return dict(groups=groups, rows=rows,
                gpu_ms=sum(e[0] for e in events) / 1e3,
                kernels=[f"{us / 1e3:8.2f} ms {count:5}x {name[:90]}"
                         for us, count, name in events[:25]],
                cpu=[f"{us / 1e3:8.2f} ms {count:5}x {name[:90]}"
                     for us, count, name in cpu[:25]])
