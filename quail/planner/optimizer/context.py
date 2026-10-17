"""Configuration, input statistics, and memoized prompts available during planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from quail.planner.prompt_lowering import LoweredPrompt, lower_prompt
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class PlanningContext:
    """Inputs shared by optimizer rules and backend planners.

    Tree rewrites that do not estimate costs may omit model and device.
    Each candidate has its own settings. Memoized values are shared within
    one search and must be keyed by all inputs that affect the result.
    """

    model: ModelSpec | None = None
    device: DeviceSpec | None = None
    gpus: int = 1
    document_tokens: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "quail"
    order: str | None = None
    # the most noise draws a diffusion model averages per letters read
    # and per one-table AI.SCORE
    canvas_draws: int = 4
    # an attention path forced for every filter and join; None lets
    # the planner choose per node
    attention: str | None = None
    tokenizer: Callable[[str], Any] | None = None
    # join written position -> its equality pairs as a fraction of
    # the cross product; joins without conditions are absent
    pair_fractions: Mapping[int, float] = field(default_factory=dict)
    # alias -> the fraction of its documents the regular predicates are
    # expected to keep; aliases without tests are absent
    scan_fractions: Mapping[str, float] = field(default_factory=dict)
    # the logical plan the graph was planned from; None when the rules
    # run again over a finished plan
    logical_plan: Any = None
    catalog: Any = None
    engine_config: Any = None
    # the plan's settings as the rules see them; a rule adds what the
    # executor needs for its decision, such as the KV retention
    # schedule, and the planner puts the result on the plan
    settings: dict = field(default_factory=dict, compare=False, repr=False)
    # results computed once per query, such as the plan's statistics,
    # shared with the tree rewrites that planned it
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def lowered(self, prompt) -> LoweredPrompt:
        """Return the tokenized prompt, reusing an earlier result when available.

        Raises:
            ValueError: The context has no tokenizer.
        """
        key = ("lowered", prompt)
        if key not in self.memo:
            if self.tokenizer is None:
                raise ValueError("planning needs a tokenizer to lower prompts")
            self.memo[key] = lower_prompt(prompt, self.tokenizer)
        return self.memo[key]
