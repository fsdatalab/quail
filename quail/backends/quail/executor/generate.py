"""Greedy generation for many documents at once over the KV arena.

run_stages answers a fixed list of stages whose request tokens are
known before the run. Generation is different: each document's next
request is the token the model just chose, for up to thousands of
rounds. GreedyGenerator schedules that work with the same pieces the
stage scheduler uses: the arena's pages and state slots, pack_chunk,
and the pipeline's forward pass.

Each step packs one decode row for every document in flight and, when
lanes are free, the prompts of newly admitted documents. A prompt's
last token is its one suffix, so the step that prefills it also gives
its first output token. A decode row appends the last chosen token to
the document's KV and, on a model with linear-attention layers, steps
the state kept in its slot in place. A step that admits nobody has the
same shape as the last, so it replays a CUDA graph captured for its
batch size.

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
    """

    def __init__(self, torch, arena, pipeline, head, *, lanes: int, budget: int,
                 max_new_tokens: int, context_tokens: int, stops,
                 graphs: bool = True):
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
        import time

        stats = self.stats = dict(wait_s=0.0, mixed_s=0.0, mixed_steps=0,
                                  mixed_pack_s=0.0, prefilled_tokens=0,
                                  decode_s=0.0, decode_steps=0, decode_rows=0,
                                  decode_host_s=0.0)
        pending = iter(generations)
        waiting = None
        exhausted = False
        active = []
        free = list(range(self.lanes - 1, -1, -1))
        prompt_tokens = prompt_count = 0
        while True:
            admit = []
            room = self.budget - len(active)
            mean = prompt_tokens / prompt_count if prompt_count else 0.0
            if admission_ready(len(free), len(active), mean, self.budget,
                               exhausted):
                while len(admit) < len(free) and not exhausted:
                    if waiting is None:
                        started = time.perf_counter()
                        waiting = next(pending, None)
                        stats["wait_s"] += time.perf_counter() - started
                        if waiting is None:
                            exhausted = True
                            break
                    size = len(waiting.prompt)
                    if admit and size > room:
                        break
                    capacity = self._reserve(waiting)
                    if capacity is None:
                        # the arena is full: the prompt waits for lanes
                        # to finish and free their pages
                        break
                    admit.append((waiting, capacity))
                    prompt_tokens += size
                    prompt_count += 1
                    room -= size
                    waiting = None
            started = time.perf_counter()
            if admit:
                tokens = self._mixed(active, admit, free)
                stats["mixed_s"] += time.perf_counter() - started
                stats["mixed_steps"] += 1
                stats["prefilled_tokens"] += sum(len(g.prompt) for g, _ in admit)
            elif active:
                tokens = (self._replay(active) if self.graphs
                          else self._mixed(active, [], free))
                stats["decode_s"] += time.perf_counter() - started
                stats["decode_steps"] += 1
                stats["decode_rows"] += len(active)
            elif exhausted and waiting is None:
                return
            else:
                # the arena could not take the next prompt and nothing is
                # in flight to free room
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
        with torch.inference_mode():
            hidden = _forward(self.pipeline, arena, chunk)
            tokens = (hidden @ self.head.t()).argmax(dim=-1).tolist()
        for generation, capacity in admit:
            key = generation.key
            document = _Document(
                generation, free.pop(), capacity,
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
        self._pad = dict(ids=0, positions=0, dst=scratch_row, used=1, lanes=n,
                         slots=0)
        dtypes = dict(ids=torch.int64, positions=torch.int64, dst=torch.int64,
                      used=torch.int32, lanes=torch.int64, slots=torch.int32)
        self._host, self._device = {}, {}
        for name, dtype in dtypes.items():
            width = 3 * n if name == "positions" else n
            self._host[name] = torch.full((width,), self._pad[name], dtype=dtype,
                                          pin_memory=True)
            self._device[name] = self._host[name].to("cuda")
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
        chunk = SimpleNamespace(
            input_ids=d["ids"][:size], positions=d["ids"][:size],
            rope_positions=d["positions"][:3 * size].view(3, size),
            meta=meta, final_indices=self._index[:size])
        hidden = self.pipeline.forward_chunk(chunk)
        self._out[:size].copy_((hidden @ self.head.t()).argmax(dim=-1))

    def _write_lane(self, document):
        """Write the document's pages into its lane's block table row."""
        pages = self.arena.owned_pages(document.generation.key)
        row = self.torch.zeros(self.width, dtype=self.torch.int32)
        row[:len(pages)] = self.torch.as_tensor(pages, dtype=self.torch.int32)
        self._table[document.lane].copy_(row)

    def _replay(self, active) -> list:
        """One token for every document in flight, by replaying a graph."""
        import time

        torch = self.torch
        started = time.perf_counter()
        n = len(active)
        size = next(s for s in self._sizes if s >= n)
        fed = [document.fed for document in active]
        values = dict(
            ids=[document.tokens[-1] for document in active],
            dst=[int(document.rows[f]) for document, f in zip(active, fed)],
            used=[f + 1 for f in fed],
            lanes=[document.lane for document in active],
            slots=[document.slot for document in active])
        for name, value in values.items():
            host = self._host[name].numpy()
            host[:n] = value
            host[n:size] = self._pad[name]
            self._device[name][:size].copy_(self._host[name][:size],
                                            non_blocking=True)
        positions = self._host["positions"].numpy()[:3 * size].reshape(3, size)
        positions[:, :n] = [f + document.generation.rope_offset
                            for f, document in zip(fed, active)]
        positions[:, n:] = 0
        self._device["positions"][:3 * size].copy_(
            self._host["positions"][:3 * size], non_blocking=True)
        self.stats["decode_host_s"] += time.perf_counter() - started
        with torch.inference_mode():
            self._graphs[size].replay()
            return self._out[:n].tolist()
