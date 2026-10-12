"""Greedy page transcription with a Qwen3.5-family OCR model.

PageParser loads an OCR checkpoint such as LightOnOCR-3-0.8B through
vLLM's loader, runs its vision tower over page images, and transcribes
each page with greedy decoding on Quail's executor: the Qwen3.5
pipeline's forward pass, the engine's paged attention for the
full-attention layers, and the arena's state pool for the Gated
DeltaNet layers.

Every row of the batch owns a fixed block of pages, sized for the
longest prompt plus the output limit, and one state slot. A step
prefills newly admitted pages up to a token budget, then decodes one
token for every page in flight; a page leaves the batch at its end of
text or its output limit, and the next page takes its row.

prepare_page runs on the CPU, in worker processes: the checkpoint's
image processor gives the prompt's token ids and the patches, which
travel as uint8 and are normalized on the GPU.
"""

from dataclasses import dataclass, field


def prepare_page(processor, prompt: str, image) -> dict:
    """Token ids, uint8 patches, and patch grid for one page image.

    Args:
        processor: The checkpoint's AutoProcessor.
        prompt: The chat turn with one image placeholder.
        image: A PIL image.

    Returns:
        input_ids (int32 array), patches (uint8 array of patches by
        pixels), and grid (time, height, width in patches).

    Raises:
        ValueError: The processor's normalized pixels are not an exact
            image of uint8 values, so uint8 would lose precision.
    """
    import numpy as np

    out = processor(text=[prompt], images=[image], return_tensors="np")
    mean, std = _normalization(processor)
    raw = (out["pixel_values"] * std + mean) * 255.0
    rounded = np.rint(raw)
    if np.abs(raw - rounded).max() > 1e-2:
        raise ValueError("the processor's pixels are not uint8 values")
    return dict(input_ids=out["input_ids"][0].astype(np.int32),
                patches=rounded.astype(np.uint8),
                grid=tuple(int(x) for x in out["image_grid_thw"][0]))


_WORKER = {}


def init_worker(model_path: str, prompt: str) -> None:
    """Load the image processor once in a page-preparing worker process."""
    import torch
    from transformers import AutoProcessor

    # the workers already run in parallel; one thread each avoids
    # oversubscribing the CPUs
    torch.set_num_threads(1)
    _WORKER["processor"] = AutoProcessor.from_pretrained(model_path)
    _WORKER["prompt"] = prompt


def prepare_png(png: bytes) -> dict:
    """prepare_page on PNG bytes, in a worker set up by init_worker."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    return prepare_page(_WORKER["processor"], _WORKER["prompt"], image)


def prepared_pages(items, model_path: str, prompt: str, *, workers: int,
                   window: int):
    """Yield (key, prepared page) in order, preparing ahead in worker processes.

    Args:
        items: Iterable of (key, PNG bytes).
        model_path: The checkpoint directory the processor loads from.
        prompt: The chat turn with one image placeholder.
        workers: Worker processes.
        window: Pages prepared ahead of the consumer at most.
    """
    import multiprocessing
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    # spawn: the parent has CUDA initialized
    with ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=(model_path, prompt)) as pool:
        ahead = deque()
        for key, png in items:
            ahead.append((key, pool.submit(prepare_png, png)))
            if len(ahead) >= window:
                key, future = ahead.popleft()
                yield key, future.result()
        while ahead:
            key, future = ahead.popleft()
            yield key, future.result()


def _normalization(processor) -> tuple[float, float]:
    """The image processor's one mean and one standard deviation."""
    images = processor.image_processor
    means, stds = set(images.image_mean), set(images.image_std)
    if len(means) != 1 or len(stds) != 1:
        raise ValueError("per-channel normalization is not supported")
    return means.pop(), stds.pop()


def mrope_positions(input_ids, grid, image_token_id: int, merge: int):
    """Rotary positions (time, row, column) for a prompt with one image.

    Text before the image counts up on all three axes; the image's
    tokens take its row and column offsets past that text; text after
    the image resumes one past the largest image position, as vLLM's
    Qwen3-VL positions do.

    Returns:
        A (3, tokens) int64 array and the decode offset: a token at
        sequence index i takes position i + offset on all three axes.
    """
    import numpy as np

    ids = np.asarray(input_ids)
    where = np.flatnonzero(ids == image_token_id)
    _, height, width = grid
    rows, cols = height // merge, width // merge
    if where.size != rows * cols or (where.size and
                                     where[-1] - where[0] + 1 != where.size):
        raise ValueError("the prompt needs one contiguous image of the grid's size")
    start = int(where[0])
    tail = ids.size - start - where.size
    row, col = np.indices((rows, cols)).reshape(2, -1)
    image = np.stack([np.zeros_like(row), row, col]) + start
    resume = start + max(rows, cols)
    positions = np.concatenate([
        np.broadcast_to(np.arange(start), (3, start)),
        image,
        np.broadcast_to(np.arange(tail) + resume, (3, tail))], axis=1)
    return positions.astype(np.int64), resume + tail - ids.size


@dataclass
class _Chunk:
    """The rows one forward pass reads, in the shape the pipeline takes."""

    input_ids: object
    positions: object
    meta: dict
    final_indices: object
    inputs_embeds: object = None


@dataclass
class _Row:
    """One page in flight: its key, row, and the tokens so far."""

    key: object
    row: int
    prompt_tokens: int
    offset: int
    tokens: list = field(default_factory=list)


class PageParser:
    """Transcribe page images greedily on Quail's executor.

    Args:
        spec: The OCR model's ModelSpec (arch "qwen3_5").
        rows: Pages in flight at once.
        max_output_tokens: The output limit per page.
        context_tokens: The longest prompt plus output a row holds.
        prefill_tokens: The prompt tokens one step prefills at most.
    """

    def __init__(self, spec, *, rows: int, max_output_tokens: int,
                 context_tokens: int, prefill_tokens: int):
        import torch
        from transformers import AutoProcessor, GenerationConfig

        from quail.backends.quail.executor.arena import KVArena
        from quail.backends.quail.executor.model import (
            full_output_head,
            load_model,
            resolve_model_path,
        )
        from quail.backends.quail.executor.models import build_pipeline
        from quail.cost.budgets import PAGE_TOKENS

        self.torch = torch
        path = resolve_model_path(spec.hf_name, spec.revision)
        self.processor = AutoProcessor.from_pretrained(path)
        tokenizer = self.processor.tokenizer
        stops = {tokenizer.eos_token_id}
        generation = GenerationConfig.from_pretrained(path).eos_token_id
        stops.update(generation if isinstance(generation, list) else [generation])
        self.stops = frozenset(s for s in stops if s is not None)
        self.model = load_model(spec.hf_name, spec.revision,
                                answer_token_ids=tuple(sorted(self.stops)),
                                max_batched_tokens=prefill_tokens)
        self.head = full_output_head(self.model)
        self.visual = self.model.visual
        self.merge = self.visual.spatial_merge_size
        self.image_token_id = self.model.config.image_token_id
        self.mean, self.std = _normalization(self.processor)
        self.rows = rows
        self.max_output_tokens = max_output_tokens
        self.context_tokens = context_tokens
        self.prefill_tokens = prefill_tokens
        self.row_pages = -(-context_tokens // PAGE_TOKENS)
        self.row_capacity = self.row_pages * PAGE_TOKENS
        # one row past the last is scratch: a decode graph's padded rows
        # write their KV there and read state slot 0, which the decode
        # kernels skip
        self.arena = KVArena(
            n_layers=spec.layers, n_pages=(rows + 1) * self.row_pages,
            page_tokens=PAGE_TOKENS, n_kv=spec.n_kv, d_head=spec.d_head,
            dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
            state_layers=spec.linear_layer_set, n_state_slots=rows + 1,
            state_shape=spec.state_shape, conv_shape=spec.conv_shape)
        self.pipeline = build_pipeline(spec, self.model, self.arena)
        self.table = torch.arange((rows + 1) * self.row_pages, dtype=torch.int32,
                                  device="cuda").view(rows + 1, self.row_pages)
        self._capture_decode()

    def _capture_decode(self):
        """Capture the decode step as a CUDA graph per batch size.

        The graphs read static input buffers; a step fills the first
        rows of the smallest graph that holds its pages and pads the
        rest onto the scratch row.
        """
        torch = self.torch
        n = self.rows
        self._pad = dict(ids=0, positions=0, dst=n * self.row_capacity, used=1,
                         rows=n, slots=0)
        dtypes = dict(ids=torch.int64, positions=torch.int64, dst=torch.int64,
                      used=torch.int32, rows=torch.int64, slots=torch.int32)
        sizes = dict(positions=3 * n)
        self._host, self._device = {}, {}
        for name, dtype in dtypes.items():
            width = sizes.get(name, n)
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
        d = self._device
        meta = dict(
            unified=dict(
                cu_q=self._cu[:size + 1], max_q=1, src=self._index[:size],
                dst=d["dst"][:size], tail_src=None, used=d["used"][:size],
                max_used=self.row_capacity,
                table=self.table.index_select(0, d["rows"][:size])),
            decode_slots=d["slots"][:size])
        chunk = _Chunk(d["ids"][:size], d["positions"][:3 * size].view(3, size),
                       meta, self._index[:size])
        hidden = self.pipeline.forward_chunk(chunk)
        self._out[:size].copy_((hidden @ self.head.t()).argmax(dim=-1))

    def prompt(self) -> str:
        """The transcription prompt: one image, no text, thinking off."""
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def parse(self, pages):
        """Transcribe pages; yields (key, token ids, prompt tokens) as each ends.

        Args:
            pages: Iterable of (key, prepare_page output) pairs.
        """
        pending = iter(pages)
        waiting, active = None, {}
        free = list(range(self.rows - 1, -1, -1))
        exhausted = False
        while True:
            admit = []
            budget = self.prefill_tokens
            while free and not exhausted:
                if waiting is None:
                    waiting = next(pending, None)
                    if waiting is None:
                        exhausted = True
                        break
                size = waiting[1]["input_ids"].size
                if admit and size > budget:
                    break
                admit.append(waiting)
                budget -= size
                waiting = None
            if admit:
                rows = [free.pop() for _ in admit]
                with self.torch.inference_mode():
                    first = self._prefill(admit, rows)
                for (key, page), row, token in zip(admit, rows, first):
                    state = _Row(key, row, page["input_ids"].size, page["offset"],
                                 [token])
                    if self._done(state):
                        free.append(row)
                        yield key, state.tokens, state.prompt_tokens
                    else:
                        active[row] = state
            if not active:
                if exhausted and waiting is None:
                    return
                continue
            states = list(active.values())
            with self.torch.inference_mode():
                tokens = self._decode(states)
            for state, token in zip(states, tokens):
                state.tokens.append(token)
                if self._done(state):
                    del active[state.row]
                    free.append(state.row)
                    yield state.key, state.tokens, state.prompt_tokens

    def _done(self, state) -> bool:
        return (state.tokens[-1] in self.stops
                or len(state.tokens) >= self.max_output_tokens
                or state.prompt_tokens + len(state.tokens) >= self.context_tokens)

    def _tensor(self, values, dtype):
        torch = self.torch
        return torch.as_tensor(values, dtype=dtype).pin_memory().to(
            "cuda", non_blocking=True)

    def _images(self, pages):
        """Run the vision tower over the pages' patches; one tensor per page."""
        import numpy as np
        from vllm.config import set_current_vllm_config

        torch = self.torch
        patches = self._tensor(np.concatenate([p["patches"] for p in pages]),
                               torch.uint8)
        pixels = ((patches.to(torch.float32) / 255.0 - self.mean) / self.std)
        grid = torch.tensor([p["grid"] for p in pages], dtype=torch.int64)
        with set_current_vllm_config(self.model.quail_vllm_config):
            embeds = self.visual(pixels.to(self.visual.dtype), grid_thw=grid)
        sizes = (grid.prod(-1) // self.merge // self.merge).tolist()
        return embeds.split(sizes)

    def _prefill(self, admitted, rows) -> list:
        """Prefill new pages into their rows; returns each page's first token."""
        import numpy as np

        torch = self.torch
        pages = [page for _, page in admitted]
        ids = np.concatenate([p["input_ids"] for p in pages])
        lengths = [p["input_ids"].size for p in pages]
        positions = []
        for page in pages:
            pos, page["offset"] = mrope_positions(
                page["input_ids"], page["grid"], self.image_token_id, self.merge)
            positions.append(pos)
        input_ids = self._tensor(ids, torch.int64)
        embeds = self.pipeline.embed(input_ids)
        images = torch.cat(self._images(pages))
        embeds[input_ids == self.image_token_id] = images.to(embeds.dtype)
        cu = np.concatenate([[0], np.cumsum(lengths)])
        dst = np.concatenate([row * self.row_capacity + np.arange(n)
                              for row, n in zip(rows, lengths)])
        cu_q = self._tensor(cu, torch.int32)
        slots = self._tensor([row + 1 for row in rows], torch.int64)
        k = len(pages)
        meta = dict(unified=dict(
            cu_q=cu_q, max_q=max(lengths),
            src=torch.arange(ids.size, device="cuda"),
            dst=self._tensor(dst, torch.int64), tail_src=None,
            used=self._tensor(lengths, torch.int32), max_used=max(lengths),
            table=self.table.index_select(0, self._tensor(rows, torch.int64))),
            state=dict(waves=[dict(
                rows=None, cu=cu_q, n=k,
                init=torch.zeros(k, dtype=torch.int64, device="cuda"),
                has_init=torch.zeros(k, dtype=torch.bool, device="cuda"),
                save_index=torch.arange(k, device="cuda"), save_slots=slots)]))
        chunk = _Chunk(input_ids, self._tensor(np.concatenate(positions, axis=1),
                                               torch.int64),
                       meta, self._tensor(cu[1:] - 1, torch.int64), embeds)
        return self._next_tokens(chunk)

    def _decode(self, states) -> list:
        """One token for every page in flight, by replaying a captured graph."""
        n = len(states)
        size = next(s for s in self._sizes if s >= n)
        lengths = [s.prompt_tokens + len(s.tokens) - 1 for s in states]
        values = dict(
            ids=[s.tokens[-1] for s in states],
            dst=[s.row * self.row_capacity + length
                 for s, length in zip(states, lengths)],
            used=[length + 1 for length in lengths],
            rows=[s.row for s in states],
            slots=[s.row + 1 for s in states])
        for name, value in values.items():
            host = self._host[name].numpy()
            host[:n] = value
            host[n:size] = self._pad[name]
            self._device[name][:size].copy_(self._host[name][:size],
                                            non_blocking=True)
        positions = self._host["positions"].numpy()[:3 * size].reshape(3, size)
        positions[:, :n] = [length + s.offset for length, s in zip(lengths, states)]
        positions[:, n:] = 0
        self._device["positions"][:3 * size].copy_(
            self._host["positions"][:3 * size], non_blocking=True)
        self._graphs[size].replay()
        return self._out[:n].tolist()

    def _next_tokens(self, chunk) -> list:
        """The greedy next token after each final row of the chunk."""
        hidden = self.pipeline.forward_chunk(chunk)
        return (hidden @ self.head.t()).argmax(dim=-1).tolist()
