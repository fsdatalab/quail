"""DiffusionGemma's denoising steps, which AI.CLASSIFY decodes answers with.

vLLM 0.26 (vllm/model_executor/models/diffusion_gemma.py) generates
with DiffusionGemma this way, and Quail follows it:

- The prompt runs causally once and its KV stays. The answer is a
  canvas of N rows placed after the prompt. Its first input is N token
  ids drawn uniformly from the vocabulary.
- Each denoising step runs the canvas rows alone. They read the
  prompt's KV and every canvas row, with no causal mask; a sliding
  layer's window reaches both ways. A row's input is its token's
  embedding times the normalizer plus the self-conditioning signal,
  through a weightless RMS norm (DiffusionGemmaSelfConditioning). The
  signal is the self-conditioning MLP of the soft embedding: the
  previous step's probabilities times the embedding, times the
  normalizer. The first step's soft embedding is zero, so its signal
  is zero.
- Every canvas row is read through the output head. The logits are
  capped (tanh at 30) and divided by the step's temperature, which
  falls linearly from t_max at the first step towards t_min at the
  last. Their softmax gives the soft embedding for the next step, each
  row's entropy, and each row's argmax.
- The rows kept are those of lowest entropy, while the summed entropy
  of the kept rows below the highest kept one is at most
  entropy_bound. A kept row's next input is the token the step chose;
  every other row gets a fresh random token.
- The canvas is done when its argmax tokens equal the previous step's
  (stability_threshold steps in a row) and its mean row entropy is
  below confidence_threshold, or after max_steps steps. The answer is
  that step's argmax tokens, cut at the first stop token and
  detokenized without special tokens.

Two parts are not reproduced exactly. vLLM draws a kept row's token by
Gumbel sampling at the step's temperature and draws the random tokens
from the process-wide generator, so its answers depend on the seed and
on what else is batched. Quail takes the argmax for a kept row, the
temperature-zero draw, and draws every random token from a generator
seeded by the document's row, so an answer is the same in any batch.
The temperature still scales the probabilities that give the entropy
and the self-conditioning input, as in vLLM.
"""

import numpy as np

# Every random canvas token comes from this seed: a filter's fixed
# canvas, and a classified document's canvas with the document's row.
CANVAS_SEED = 0


def check_denoising(settings, generation_config: dict,
                    logit_softcap: float) -> None:
    """Check the spec's denoising settings against the loaded checkpoint.

    Args:
        settings: The ModelSpec's Denoising.
        generation_config: The checkpoint's generation_config.json values
            that differ from the defaults.
        logit_softcap: The checkpoint's final_logit_softcapping.

    Raises:
        ValueError: A value differs.
    """
    loaded = {
        "max_denoising_steps": generation_config.get("max_denoising_steps"),
        "t_min": generation_config.get("t_min"),
        "t_max": generation_config.get("t_max"),
        "entropy_bound": (generation_config.get("sampler_config") or {}).get(
            "entropy_bound"),
        "confidence_threshold": generation_config.get("confidence_threshold"),
        "stability_threshold": generation_config.get("stability_threshold"),
        "final_logit_softcapping": logit_softcap,
    }
    spec = {
        "max_denoising_steps": settings.max_steps,
        "t_min": settings.t_min,
        "t_max": settings.t_max,
        "entropy_bound": settings.entropy_bound,
        "confidence_threshold": settings.confidence_threshold,
        "stability_threshold": settings.stability_threshold,
        "final_logit_softcapping": settings.logit_softcap,
    }
    differ = [name for name in spec if loaded[name] != spec[name]]
    stops = generation_config.get("eos_token_id")
    if stops is not None:
        stops = {stops} if isinstance(stops, int) else set(stops)
        if stops != set(settings.stop_token_ids):
            differ.append("eos_token_id")
    if differ:
        raise ValueError(
            f"the checkpoint's denoising settings {differ} differ from the "
            f"model spec's")


def accepted_rows(entropy, bound: float) -> np.ndarray:
    """Which canvas rows a step keeps, by vLLM's entropy bound.

    Rows are taken in order of rising entropy while the entropy summed
    over the rows taken, less the largest of them, is at most bound.
    """
    entropy = np.asarray(entropy, dtype=np.float32)
    order = np.argsort(entropy, kind="stable")
    ranked = entropy[order]
    keep = np.empty(len(entropy), dtype=bool)
    keep[order] = np.cumsum(ranked) - np.maximum.accumulate(ranked) <= bound
    return keep


class DocumentCanvas:
    """One document's canvas between its denoising steps, on the host.

    Args:
        settings: The model's Denoising.
        vocab: Vocabulary rows the random tokens are drawn from.
        seed: The document's generator seed.
        conditioning_row: The first of the document's rows in the
            ConditioningRows.

    Attributes:
        canvas: The token ids the next step packs.
        steps: Steps whose readout was applied.
        tokens: The answer's token ids once the canvas is done, else
            None.
    """

    def __init__(self, settings, vocab: int, seed, conditioning_row: int):
        self.settings = settings
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)
        self.canvas = self.rng.integers(0, vocab, settings.canvas_rows)
        self.conditioning_row = conditioning_row
        self.history = []
        self.steps = 0
        self.tokens = None

    def update(self, tokens, entropy) -> bool:
        """Apply one step's argmax tokens and row entropies.

        Returns:
            Whether the document takes another step.
        """
        settings = self.settings
        tokens = np.asarray(tokens, dtype=np.int64)
        self.steps += 1
        kept = settings.stability_threshold + 1
        self.history = (self.history + [tokens])[-kept:]
        stable = len(self.history) == kept and all(
            np.array_equal(self.history[0], earlier)
            for earlier in self.history[1:])
        confident = float(np.mean(entropy, dtype=np.float32)) < \
            settings.confidence_threshold
        if (stable and confident) or self.steps >= settings.max_steps:
            self.tokens = tokens
            return False
        noise = self.rng.integers(0, self.vocab, settings.canvas_rows)
        self.canvas = np.where(
            accepted_rows(entropy, settings.entropy_bound), tokens, noise)
        return True


def answer_text(tokenizer, tokens, stop_token_ids) -> str:
    """The answer's text: tokens before the first stop, special tokens skipped."""
    stops = set(stop_token_ids)
    tokens = [int(token) for token in tokens]
    end = next((i for i, token in enumerate(tokens) if token in stops),
               len(tokens))
    return tokenizer.decode(tokens[:end], skip_special_tokens=True)


class ConditioningRows:
    """Every running document's self-conditioning input, on the GPU.

    A document takes one block of canvas_rows rows at its first step,
    zeroed, and gives it back after its last. A step's readout writes
    the next step's input into the rows its canvas read. The block
    count doubles when every block is taken.

    Args:
        torch: The torch module.
        width: The model's hidden size.
        canvas_rows: Rows per block.
        dtype: The embedding's dtype.
        device: Where the rows live.
        blocks: Blocks to start with.
    """

    def __init__(self, torch, width, canvas_rows, dtype, device, blocks=64):
        self.torch = torch
        self.canvas_rows = canvas_rows
        self.rows = torch.zeros(blocks * canvas_rows, width, dtype=dtype,
                                device=device)
        self.free = list(range(blocks - 1, -1, -1))

    def take(self) -> int:
        """A zeroed block for one document; returns its first row."""
        if not self.free:
            blocks = self.rows.shape[0] // self.canvas_rows
            grown = self.torch.zeros(
                2 * self.rows.shape[0], self.rows.shape[1],
                dtype=self.rows.dtype, device=self.rows.device)
            grown[:self.rows.shape[0]].copy_(self.rows)
            self.rows = grown
            self.free = list(range(2 * blocks - 1, blocks - 1, -1))
        first = self.free.pop() * self.canvas_rows
        self.rows[first:first + self.canvas_rows].zero_()
        return first

    def release(self, first: int) -> None:
        """Give back the block starting at row ``first``."""
        self.free.append(first // self.canvas_rows)


def denoise_rows(torch, F, normed, head, normalizer, temperature,
                 logit_softcap, block_rows):
    """Read canvas rows through the output head as vLLM's sampler reads them.

    Args:
        torch: The torch module.
        F: torch.nn.functional.
        normed: Final-normed hidden rows, (rows, hidden).
        head: The output head, tied to the embedding, (vocab, hidden).
        normalizer: The embedding's scale.
        temperature: Per row, its step's temperature, float32.
        logit_softcap: The tanh cap on the logits.
        block_rows: Rows per head block.

    Returns:
        (tokens, entropy, soft): per row the argmax token, the entropy
        of the tempered distribution, and the soft embedding the next
        step's self-conditioning reads.
    """
    n = normed.shape[0]
    device = normed.device
    tokens = torch.empty(n, dtype=torch.int64, device=device)
    entropy = torch.empty(n, dtype=torch.float32, device=device)
    soft = torch.empty(n, head.shape[1], dtype=head.dtype, device=device)
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        logits = F.linear(normed[start:end].to(head.dtype), head).float()
        logits.div_(logit_softcap).tanh_().mul_(logit_softcap)
        logits.div_(temperature[start:end, None])
        tokens[start:end] = logits.argmax(dim=1)
        logprobs = torch.log_softmax(logits, dim=1)
        del logits
        probs = logprobs.exp()
        entropy[start:end] = -logprobs.mul_(probs).sum(dim=1)
        del logprobs
        soft[start:end] = (probs.to(head.dtype) @ head) * normalizer
    return tokens, entropy, soft


class AsyncCanvasReadout:
    """Non-blocking readout of one denoising step for every canvas in a chunk.

    Each answer is one canvas: its rows' argmax tokens and entropies.
    The next step's self-conditioning input stays on the GPU, written
    into the rows each canvas read; only the tokens and entropies are
    copied to the host.

    Args:
        torch: The torch module.
        F: torch.nn.functional.
        head: The output head, tied to the embedding.
        normalizer: The embedding's scale.
        settings: The model's Denoising.
        conditioning: The run's ConditioningRows.
    """

    # 256 rows of 262,144 float32 logits are 256 MiB; a block holds
    # about three such matrices at once
    BLOCK_ROWS = 256
    reads_chunk = True

    def __init__(self, torch, F, head, normalizer, settings, conditioning):
        self.torch = torch
        self.F = F
        self.head = head
        self.normalizer = normalizer
        self.settings = settings
        self.conditioning = conditioning
        rows = settings.canvas_rows
        self.dtype = np.dtype([("tokens", np.int64, (rows,)),
                               ("entropy", np.float32, (rows,))])

    def submit(self, normed, *, chunk, stages):
        """Read one chunk's canvases; ``stages`` gives each canvas's step."""
        torch = self.torch
        rows = self.settings.canvas_rows
        if normed.shape[0] != rows * len(stages):
            raise ValueError(
                f"{normed.shape[0]} rows read for {len(stages)} canvases "
                f"of {rows} rows")
        on_gpu = normed.is_cuda
        temperature = torch.tensor(
            np.repeat([self.settings.temperature(step) for step in stages],
                      rows),
            dtype=torch.float32, pin_memory=on_gpu).to(
                normed.device, non_blocking=on_gpu)
        tokens, entropy, soft = denoise_rows(
            torch, self.F, normed, self.head, self.normalizer, temperature,
            self.settings.logit_softcap, self.BLOCK_ROWS)
        self.conditioning.rows.index_copy_(
            0, chunk.meta["canvas"]["conditioning_rows"], soft)
        host_tokens = torch.empty(tokens.shape, dtype=tokens.dtype,
                                  pin_memory=on_gpu)
        host_entropy = torch.empty(entropy.shape, dtype=entropy.dtype,
                                   pin_memory=on_gpu)
        host_tokens.copy_(tokens, non_blocking=on_gpu)
        host_entropy.copy_(entropy, non_blocking=on_gpu)
        event = None
        if on_gpu:
            event = torch.cuda.Event()
            event.record()
        return event, host_tokens, host_entropy, len(stages)

    def result(self, handle):
        """One record per canvas: its tokens and row entropies."""
        event, tokens, entropy, count = handle
        if event is not None:
            event.synchronize()
        out = np.empty(count, dtype=self.dtype)
        out["tokens"] = tokens.numpy().reshape(count, -1)
        out["entropy"] = entropy.numpy().reshape(count, -1)
        return out
