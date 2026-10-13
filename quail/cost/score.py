"""Work and time estimates for scores over prepared prompt and document lengths."""

from dataclasses import dataclass

from quail.cost.sol import speed_of_light
from quail.cost.work import Work, triangle
from quail.specs import DeviceSpec, ModelSpec


def score_fixed_tokens(token_lengths, canvas_tokens: int, draws: int) -> int:
    """Return the tokens a score adds to each input besides its documents.

    Args:
        token_lengths: Token counts of the prompt pieces around the documents.
        canvas_tokens: The model's canvas rows per answer; 0 without one.
        draws: The noise draws averaged per answer.

    Returns:
        The prompt pieces, the canvas, and each later draw's cue and canvas.
    """
    return (sum(token_lengths) + canvas_tokens
            + (draws - 1) * (1 + canvas_tokens))


def score_work(count, mean_tokens, fixed_tokens, *, variance=0.0,
                prefix_tokens=0.0, prefix_variance=0.0, groups=0.0,
                shared_tokens=0, copies=1) -> Work:
    if count <= 0:
        return Work()
    length = mean_tokens + fixed_tokens
    work = Work(
        tokens=length,
        pairs=triangle(length) + variance / 2,
        kv_written=length,
        state_written=1.0,
    ) * count
    reused = max(0.0, count - groups)
    work += Work(
        tokens=-prefix_tokens,
        pairs=-triangle(prefix_tokens) - prefix_variance / 2,
        kv_written=-prefix_tokens,
        kv_read=prefix_tokens,
        state_written=-1.0,
        state_read=1.0,
    ) * reused
    shared_reuses = max(0.0, groups - copies)
    return work + Work(
        tokens=-shared_tokens,
        pairs=-triangle(shared_tokens),
        kv_written=-shared_tokens,
        kv_read=shared_tokens,
        state_written=-1.0,
        state_read=1.0,
    ) * shared_reuses


@dataclass(frozen=True)
class ScoreCost:
    """Prepared numeric inputs for one score's cost at any survivor count."""

    token_lengths: tuple[int, ...]
    summaries: tuple
    mean_tokens: float
    model: ModelSpec
    device: DeviceSpec
    chunk_tokens: int
    capacity: int
    workers: int
    draws: int

    def estimate(self, expected_inputs, *, prefix_groups=None) -> tuple[Work, float]:
        """Return model work and seconds for the expected inputs."""
        summaries = self.summaries
        canvas = self.model.canvas_tokens
        # a one-table score on a canvas model may average noise draws, each
        # the cue and its canvas after the document's KV; every draw is priced
        draws = self.draws
        fixed_tokens = score_fixed_tokens(self.token_lengths, canvas, draws)
        shared = self.token_lengths[0]
        prefix = float(shared)
        prefix_variance = 0.0
        groups = min(float(self.workers), expected_inputs)
        capacity = self.capacity
        if len(summaries) == 2:
            prefix += summaries[0].mean + self.token_lengths[1]
            prefix_variance = summaries[0].variance
            groups = min(
                expected_inputs,
                summaries[0].count if prefix_groups is None else prefix_groups,
            )
            longest = sum(summary.maximum for summary in summaries) + fixed_tokens
            if longest > capacity:
                prefix = float(shared)
                prefix_variance = 0.0
                groups = min(float(self.workers), expected_inputs)
        work = score_work(
            expected_inputs, self.mean_tokens, fixed_tokens,
            variance=sum(summary.variance for summary in summaries),
            prefix_tokens=prefix, prefix_variance=prefix_variance,
            groups=groups, shared_tokens=shared, copies=self.workers,
        )
        estimate = speed_of_light(
            work * (1 / max(1, min(self.workers, expected_inputs))),
            self.model, self.device, self.chunk_tokens,
        ).seconds
        return work, estimate

    def seconds(self, count: float) -> float:
        """Return the estimated seconds at one survivor count."""
        return self.estimate(count)[1]
