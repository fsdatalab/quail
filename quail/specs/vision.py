"""Image geometry: render size and soft token count per image.

The Gemma 4 image processor resizes an image so that it keeps its
aspect ratio, has at most ``budget * pool**2`` patches of
``patch`` pixels, and has both sides divisible by ``pool * patch``.
Each ``pool x pool`` block of patches becomes one soft token, so the
soft token count of an image depends on its shape and is usually
below the budget. These functions repeat that arithmetic on the CPU
so the planner knows every row's exact token layout before a page is
rendered, and so a page can be rendered at the processor's target
size directly. The Phase 0 probe checked them against the
checkpoint's processor at every budget
(/results/ablations/pdf_probe_vision.json).
"""

from __future__ import annotations

import math

from quail.specs.base import ModelSpec


def render_size(spec: ModelSpec, width: float, height: float,
                budget: int) -> tuple[int, int]:
    """The processor's resize target for one image, as (height, width) pixels.

    Args:
        spec: A model with image support (patch and pool sizes set).
        width: Source width in any unit; only the aspect ratio matters.
        height: Source height in the same unit.
        budget: The soft token budget, one of spec.image_token_budgets.

    Raises:
        ValueError: The model has no image geometry, or the size is empty.
    """
    if not spec.image_patch_pixels or not spec.image_pool_kernel:
        raise ValueError(f"model {spec.name!r} has no image geometry")
    if width <= 0 or height <= 0:
        raise ValueError(f"an image needs a positive size, got {width}x{height}")
    patch, pool = spec.image_patch_pixels, spec.image_pool_kernel
    max_patches = budget * pool * pool
    factor = math.sqrt(max_patches * patch * patch / (width * height))
    side = pool * patch
    target_h = int(math.floor(factor * height / side)) * side
    target_w = int(math.floor(factor * width / side)) * side
    # a very tall or very wide image rounds one side to zero; the
    # processor then gives that side one block and bounds the other
    longest = (max_patches // (pool * pool)) * side
    if target_h == 0 and target_w == 0:
        raise ValueError(f"an image of {width}x{height} resizes to nothing")
    if target_h == 0:
        target_h = side
        target_w = min(int(math.floor(width / height)) * side, longest)
    elif target_w == 0:
        target_w = side
        target_h = min(int(math.floor(height / width)) * side, longest)
    return target_h, target_w


def soft_tokens(spec: ModelSpec, width: float, height: float,
                budget: int) -> int:
    """How many soft tokens one image of this shape becomes."""
    target_h, target_w = render_size(spec, width, height, budget)
    patch, pool = spec.image_patch_pixels, spec.image_pool_kernel
    return (target_h // patch) * (target_w // patch) // (pool * pool)


def image_prefix_tokens(spec: ModelSpec, width: float, height: float,
                        budget: int) -> int:
    """Prompt tokens one image occupies: its soft tokens and their frame."""
    return soft_tokens(spec, width, height, budget) + spec.image_frame_tokens


def resolve_image_tokens(spec: ModelSpec, requested: int | None) -> int:
    """The soft token budget a session uses, or 0 for a text-only model.

    Args:
        spec: The session's model.
        requested: EngineConfig.image_tokens; None takes the model's
            default.

    Raises:
        ValueError: The model takes no images but a budget was asked
            for, or the budget is not one the model's processor accepts.
    """
    if "image" not in spec.input_modalities:
        if requested is not None:
            raise ValueError(
                f"model {spec.name!r} takes text only; image_tokens does "
                f"not apply")
        return 0
    budget = spec.default_image_tokens if requested is None else requested
    if budget not in spec.image_token_budgets:
        raise ValueError(
            f"model {spec.name!r} accepts image_tokens in "
            f"{spec.image_token_budgets}, got {budget}")
    return budget
