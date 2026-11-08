"""Time remaining for a running query, corrected by how fast it has run.

The prediction prices every finished and unfinished batch in predicted
seconds. While the query runs, the observed speed is elapsed seconds
over the predicted seconds of the finished work. Early on the estimate
trusts the prediction; as finished work grows it trusts the observed
speed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

# standard normal quantile of the 80th percentile
Z_UPPER = 0.8416


@dataclass(frozen=True)
class RemainingEstimate:
    """A time-remaining estimate.

    Attributes:
        seconds: The median seconds until the query finishes.
        upper_seconds: Seconds within which the query finishes about 80%
            of the time.
        speed: Estimated wall seconds per predicted second from here on.
        basis: "predicted" before any work finishes, "blended" while the
            prediction still outweighs the observed speed, then "observed".
    """

    seconds: float
    upper_seconds: float
    speed: float
    basis: str


class RemainingTime:
    """Estimate seconds remaining from predicted and observed progress.

    Args:
        total_seconds: Predicted seconds of all the query's work.
        prior_fraction: The share of total_seconds of finished work at
            which the observed speed and the prediction weigh equally.
        prior_spread: Relative uncertainty of the prediction, as the
            standard deviation of the log of the speed.
        spread_floor: The least relative uncertainty once the observed
            speed dominates, for changes of speed between stages.
        window: The most recent updates whose speeds measure the spread.
    """

    def __init__(self, total_seconds: float, *, prior_fraction: float = 0.05,
                 prior_spread: float = 0.5, spread_floor: float = 0.02,
                 window: int = 8):
        if total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        self.total_seconds = total_seconds
        self.prior_fraction = prior_fraction
        self.prior_spread = prior_spread
        self.spread_floor = spread_floor
        self._steps = deque(maxlen=window)
        self._elapsed = 0.0
        self._done = 0.0

    def update(self, elapsed_seconds: float, done_seconds: float,
               total_seconds: float | None = None) -> RemainingEstimate:
        """Record progress and return the estimate.

        Args:
            elapsed_seconds: Wall seconds since the first pass started.
            done_seconds: Predicted seconds of the work finished so far.
            total_seconds: Predicted seconds of all the work, when it has
                changed, such as after survivors differ from the plan.
        """
        if total_seconds is not None:
            if total_seconds <= 0:
                raise ValueError("total_seconds must be positive")
            self.total_seconds = total_seconds
        done = min(done_seconds, self.total_seconds)
        step_done, step_elapsed = done - self._done, elapsed_seconds - self._elapsed
        if step_done > 0 and step_elapsed > 0:
            self._steps.append((step_elapsed, step_done))
        self._elapsed, self._done = elapsed_seconds, done
        if done <= 0 or elapsed_seconds <= 0:
            return self._estimate(1.0, self.prior_spread, self.total_seconds,
                                  "predicted")
        weight = done / (done + self.prior_fraction * self.total_seconds)
        overall = elapsed_seconds / done
        recent = self._recent_speed() or overall
        speed = (math.sqrt(overall * recent)) ** weight
        drift = abs(math.log(recent / overall))
        spread = self.prior_spread * (1 - weight) ** 2 + weight * max(
            self._standard_error() + drift, self.spread_floor)
        basis = "blended" if weight < 0.5 else "observed"
        return self._estimate(speed, spread, self.total_seconds - done, basis)

    def _standard_error(self) -> float:
        """The standard error of the mean log speed over the window."""
        speeds = [math.log(e / d) for e, d in self._steps]
        if len(speeds) < 2:
            return 0.0
        mean = sum(speeds) / len(speeds)
        variance = sum((value - mean) ** 2 for value in speeds) / (len(speeds) - 1)
        return math.sqrt(variance / len(speeds))

    def _recent_speed(self) -> float:
        """Wall seconds per predicted second over the window, or 0 if empty."""
        done = sum(d for _, d in self._steps)
        return sum(e for e, _ in self._steps) / done if done else 0.0

    def _estimate(self, speed: float, spread: float, remaining_seconds: float,
                  basis: str) -> RemainingEstimate:
        seconds = remaining_seconds * speed
        return RemainingEstimate(
            seconds=seconds,
            upper_seconds=seconds * math.exp(Z_UPPER * spread),
            speed=speed, basis=basis)
