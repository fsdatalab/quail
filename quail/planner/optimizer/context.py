"""Configuration, input statistics, and memoized prompts available during planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from quail.planner.prompt_lowering import LoweredPrompt, lower_prompt
from quail.specs import DeviceSpec, ModelSpec


def lowered_prompt(context, prompt) -> LoweredPrompt:
    """Lower a prompt through a context's tokenizer, memoized on the context.

    Raises:
        ValueError: The context has no tokenizer.
    """
    key = ("lowered", prompt)
    if key not in context.memo:
        if context.tokenizer is None:
            raise ValueError("planning needs a tokenizer to lower prompts")
        context.memo[key] = lower_prompt(prompt, context.tokenizer)
    return context.memo[key]


@dataclass(frozen=True)
class PlanningContext:
    """Inputs available to physical planners."""

    model: ModelSpec
    device: DeviceSpec
    gpu_count: int
    document_tokens: Mapping[str, Any]
    backend: str
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
        """Return the prompt lowered with the session tokenizer, once per query."""
        return lowered_prompt(self, prompt)


@dataclass(frozen=True)
class LogicalPlanningContext:
    """Session values and statistics available to rules that rewrite query trees.

    Document token counts are exact when the column is tokenized, else
    estimated from a sample. A cost-based rule prices a predicate
    written without a selectivity with the default selectivity.

    Attributes:
        catalog: The session catalog.
        engine_config: The session's EngineConfig.
        model: Model spec; None when no cost-based rule will run.
        device: Device spec; None when no cost-based rule will run.
        gpu_count: GPU count; one model copy runs per GPU.
        document_tokens: alias -> per-document token counts.
        backend: Registered model backend name.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        canvas_draws: Maximum diffusion draws per classification.
        attention: An attention path forced for every filter and join.
        tokenizer: Callable (text -> token list) for prompts.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep.
        memo: Results a rule computed for one plan root, so a later
            pass or another rule pricing the same root reuses them.
    """

    catalog: Any
    engine_config: Any
    model: Any = None
    device: Any = None
    gpu_count: int = 1
    document_tokens: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "quail"
    order: str | None = None
    canvas_draws: int = 4
    attention: str | None = None
    tokenizer: Callable[[str], Any] | None = None
    pair_fractions: Mapping[int, float] = field(default_factory=dict)
    scan_fractions: Mapping[str, float] = field(default_factory=dict)
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def lowered(self, prompt):
        """Return the prompt lowered with the session tokenizer, once per query."""
        return lowered_prompt(self, prompt)

    def physical_context(self, logical_plan=None) -> PlanningContext:
        """Return the physical planning context with the same inputs."""
        return PlanningContext(
            model=self.model, device=self.device, gpu_count=self.gpu_count,
            document_tokens=self.document_tokens, backend=self.backend,
            order=self.order, canvas_draws=self.canvas_draws,
            attention=self.attention, tokenizer=self.tokenizer,
            pair_fractions=dict(self.pair_fractions),
            scan_fractions=dict(self.scan_fractions),
            logical_plan=logical_plan, memo=self.memo,
            catalog=self.catalog, engine_config=self.engine_config)

    @classmethod
    def from_physical(cls, context: PlanningContext) -> LogicalPlanningContext:
        """Return a query tree context with the current search inputs."""
        return cls(
            catalog=context.catalog, engine_config=context.engine_config,
            model=context.model, device=context.device, gpu_count=context.gpu_count,
            document_tokens=context.document_tokens, backend=context.backend,
            order=context.order, canvas_draws=context.canvas_draws,
            attention=context.attention, tokenizer=context.tokenizer,
            pair_fractions=context.pair_fractions,
            scan_fractions=context.scan_fractions, memo=context.memo)
