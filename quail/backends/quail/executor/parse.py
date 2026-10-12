"""Greedy page transcription with a Qwen3.5-family OCR model.

PageParser loads an OCR checkpoint such as LightOnOCR-3-0.8B through
vLLM's loader, runs its vision tower over page images, and transcribes
each page with greedy decoding on Quail's executor: GreedyGenerator
(executor/generate.py) schedules the steps over the arena, each page's
prompt carries its image rows and Qwen3-VL's three-axis rotary
positions, and the Qwen3.5 pipeline runs the forward pass.

prepare_page runs on the CPU, in worker processes: the checkpoint's
image processor gives the prompt's token ids and the patches, which
travel as uint8 and are normalized on the GPU.
"""

from functools import partial

import numpy as np


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


def _ready() -> bool:
    return "processor" in _WORKER


class PagePreparer:
    """Prepare pages in worker processes, ahead of the parser.

    The workers start, and load the image processor, when the preparer
    is made, so their startup can overlap loading the model.

    Args:
        model_path: The checkpoint directory the processor loads from.
        prompt: The chat turn with one image placeholder.
        workers: Worker processes.
    """

    def __init__(self, model_path: str, prompt: str, *, workers: int):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # spawn: the parent has, or will have, CUDA initialized
        self.pool = ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=(model_path, prompt))
        self.started = [self.pool.submit(_ready) for _ in range(workers)]

    def pages(self, items, *, window: int):
        """Yield (key, prepared page) in order, at most `window` pages ahead.

        Args:
            items: Iterable of (key, PNG bytes).
            window: Pages prepared ahead of the consumer at most.
        """
        from collections import deque

        ahead = deque()
        for key, png in items:
            ahead.append((key, self.pool.submit(prepare_png, png)))
            if len(ahead) >= window:
                key, future = ahead.popleft()
                yield key, future.result()
        while ahead:
            key, future = ahead.popleft()
            yield key, future.result()

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


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


class PageParser:
    """Transcribe page images greedily on Quail's executor.

    Args:
        spec: The OCR model's ModelSpec (arch "qwen3_5").
        lanes: Pages in flight at once.
        max_output_tokens: The output limit per page.
        context_tokens: The longest prompt plus output a page holds.
        budget: Rows one step packs at most, decode rows included.
        encode_ahead: Pages whose image rows a thread computes ahead,
            on its own CUDA stream, so the vision tower runs beside the
            decode steps; 0 runs the tower in the step that prefills.
        encode_batch: Pages per vision tower call in that thread.
        profile_mixed: Passed to GreedyGenerator.
    """

    def __init__(self, spec, *, lanes: int, max_output_tokens: int,
                 context_tokens: int, budget: int, encode_ahead: int = 0,
                 encode_batch: int = 4, profile_mixed: int = 0):
        import torch
        from transformers import AutoProcessor, GenerationConfig

        from quail.backends.quail.executor.arena import KVArena
        from quail.backends.quail.executor.generate import GreedyGenerator
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
        stops = frozenset(s for s in stops if s is not None)
        self.model = load_model(spec.hf_name, spec.revision,
                                answer_token_ids=tuple(sorted(stops)),
                                max_batched_tokens=budget)
        self.visual = self.model.visual
        self.merge = self.visual.spatial_merge_size
        self.image_token_id = self.model.config.image_token_id
        self.mean, self.std = _normalization(self.processor)
        self.encode_ahead = encode_ahead
        self.encode_batch = encode_batch
        # every lane can hold a whole context, and one page is the
        # decode graphs' scratch lane
        lane_pages = -(-context_tokens // PAGE_TOKENS)
        self.arena = KVArena(
            n_layers=spec.layers, n_pages=lanes * lane_pages + 1,
            page_tokens=PAGE_TOKENS, n_kv=spec.n_kv, d_head=spec.d_head,
            dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
            state_layers=spec.linear_layer_set, n_state_slots=lanes + 1,
            state_shape=spec.state_shape, conv_shape=spec.conv_shape)
        self.pipeline = build_pipeline(spec, self.model, self.arena)
        self.generator = GreedyGenerator(
            torch, self.arena, self.pipeline, full_output_head(self.model),
            lanes=lanes, budget=budget, max_new_tokens=max_output_tokens,
            context_tokens=context_tokens, stops=stops,
            profile_mixed=profile_mixed)

    @property
    def stats(self) -> dict:
        """The last run's step counts and times (see GreedyGenerator.run)."""
        return self.generator.stats

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
        from quail.backends.quail.executor.generate import Generation

        if self.encode_ahead:
            pages = self._encoded(pages)
        prompts = {}

        def generations():
            for key, page in pages:
                rope, offset = mrope_positions(
                    page["input_ids"], page["grid"], self.image_token_id,
                    self.merge)
                prompts[key] = page["input_ids"].size
                yield Generation(key, page["input_ids"].astype(np.int64), rope,
                                 offset, image=partial(self._image_rows, page))

        for key, tokens in self.generator.run(generations()):
            yield key, tokens, prompts.pop(key)

    def _encoded(self, pages):
        """Yield pages with their image rows, computed ahead in a thread.

        The thread runs the vision tower on its own CUDA stream and
        records an event after each group; the step that prefills a
        page waits on the event before it reads the rows.
        """
        import queue
        import threading

        torch = self.torch
        stream = torch.cuda.Stream()
        groups = queue.Queue(maxsize=max(1, self.encode_ahead // self.encode_batch))
        failure = []

        def encode(group):
            embeds = self._images([page for _, page in group])
            done = torch.cuda.Event()
            done.record(stream)
            for (_, page), rows in zip(group, embeds):
                page["image_embeds"], page["image_done"] = rows, done
            groups.put(group)

        def run():
            try:
                with torch.cuda.stream(stream), torch.inference_mode():
                    group = []
                    for item in pages:
                        group.append(item)
                        if len(group) == self.encode_batch:
                            encode(group)
                            group = []
                    if group:
                        encode(group)
            except BaseException as error:
                failure.append(error)
            finally:
                groups.put(None)

        threading.Thread(target=run, daemon=True).start()
        while (group := groups.get()) is not None:
            yield from group
        if failure:
            raise failure[0]

    def _image_rows(self, page):
        """The page's image rows in its prompt and their embeddings, on this stream."""
        torch = self.torch
        rows = np.flatnonzero(page["input_ids"] == self.image_token_id)
        if "image_embeds" not in page:
            with torch.inference_mode():
                return rows, self._images([page])[0]
        import time

        stream = torch.cuda.current_stream()
        done = page.pop("image_done")
        started = time.perf_counter()
        # waiting on the host shows how long the vision thread kept the
        # step waiting
        done.synchronize()
        stats = self.generator.stats
        stats["image_wait_s"] = (stats.get("image_wait_s", 0.0)
                                 + time.perf_counter() - started)
        stream.wait_event(done)
        embeds = page.pop("image_embeds")
        # the rows were allocated on the encoding stream
        embeds.record_stream(stream)
        return rows, embeds

    def _images(self, pages):
        """Run the vision tower over the pages' patches; one tensor per page."""
        from vllm.config import set_current_vllm_config

        torch = self.torch
        patches = torch.as_tensor(
            np.concatenate([p["patches"] for p in pages])).pin_memory().to(
                "cuda", non_blocking=True)
        pixels = ((patches.to(torch.float32) / 255.0 - self.mean) / self.std)
        grid = torch.tensor([p["grid"] for p in pages], dtype=torch.int64)
        with set_current_vllm_config(self.model.quail_vllm_config):
            embeds = self.visual(pixels.to(self.visual.dtype), grid_thw=grid)
        sizes = (grid.prod(-1) // self.merge // self.merge).tolist()
        return embeds.split(sizes)
