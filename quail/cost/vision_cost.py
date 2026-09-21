"""Vision tower components for the roofline calculation.

The tower is a ViT in front of the decoder. One page is a patch
projection, then every layer's full attention and gated MLP, then a
linear map of the pooled soft tokens into the decoder width. Norms,
RoPE, and the pooling adds are omitted, the same way the decoder
omits its norms. Attention does not cross pages: `image_pairs` is the
sum of each page's patches squared.

Tower weights are read once for the query. The runtime may call the
tower more than once to stay inside its activation budget; those
extra reads are overhead this estimate leaves out, as it leaves out
kernel gaps. K and V of a patch are written once per layer and priced
at the tower's weight dtype. The read of that KV inside the same
attention is not charged again.
"""

from __future__ import annotations

from collections.abc import Sequence

from quail.cost.roofline import CostComponent
from quail.cost.work import Work
from quail.specs import ModelSpec
from quail.specs.vision import soft_tokens

# RGB patches. The tower's patch projection takes 3 * patch * patch inputs.
_PATCH_CHANNELS = 3


def image_work(page_sizes: Sequence[tuple[float, float]], model: ModelSpec,
               budget: int) -> Work:
    """Return tower work for these pages, each embedded once.

    Args:
        page_sizes: One (width, height) per page, in any unit. Only the
            aspect ratio matters. An empty list is no image work.
        model: The model whose image geometry and tower are priced.
        budget: Soft token budget, one of the model's image budgets.

    Raises:
        ValueError: Pages were given but the model has no vision tower
            or no image geometry.
    """
    if not page_sizes:
        return Work()
    tower = model.vision_tower
    pool = model.image_pool_kernel
    if tower is None or model.image_patch_pixels <= 0 or pool <= 0:
        raise ValueError(
            f"model {model.name!r} has no vision tower for these pages")
    if budget <= 0:
        raise ValueError("image work needs a positive soft token budget")
    patches = []
    softs = []
    for width, height in page_sizes:
        soft = soft_tokens(model, width, height, budget)
        patches.append(soft * pool * pool)
        softs.append(soft)
    return Work(
        image_patches=float(sum(patches)),
        image_pairs=float(sum(count * count for count in patches)),
        image_soft_tokens=float(sum(softs)),
    )


def vision_components(work: Work, model: ModelSpec) -> tuple[CostComponent, ...]:
    """Build the tower components for one work record.

    Returns an empty tuple when the record has no image work, so a
    text query's component list is unchanged.

    Raises:
        ValueError: The record has image work the model cannot price,
            or a count is negative.
    """
    patches = work.image_patches
    pairs = work.image_pairs
    soft = work.image_soft_tokens
    if patches == 0 and pairs == 0 and soft == 0:
        return ()
    if patches <= 0 or pairs <= 0 or soft <= 0:
        raise ValueError(
            "image work needs positive patch, pair, and soft token counts, "
            f"got {patches}, {pairs}, {soft}")
    tower = model.vision_tower
    if tower is None or model.image_patch_pixels <= 0:
        raise ValueError(
            f"model {model.name!r} has image work but no vision tower")
    hidden = tower.hidden
    layers = tower.layers
    width = tower.weight_bytes
    precision = tower.weight_precision
    patch_in = _PATCH_CHANNELS * model.image_patch_pixels ** 2
    embed_params = patch_in * hidden
    position_params = 2 * tower.position_rows * hidden
    attn_params = (hidden * tower.heads * tower.head_dim
                   + 2 * hidden * tower.kv_heads * tower.head_dim
                   + tower.heads * tower.head_dim * hidden) * layers
    mlp_params = 3 * hidden * tower.intermediate * layers
    project_params = hidden * model.hidden
    # 4 * heads * head_dim per pair: QK and the attention-value product,
    # two FLOPs per multiply, on every layer.
    attention_flops = 4 * tower.heads * tower.head_dim * layers * pairs
    kv_bytes = (2 * tower.kv_heads * tower.head_dim * width * layers * patches)
    return (
        CostComponent("vision_embed", 2.0 * embed_params * patches,
                      (embed_params + position_params) * width, precision),
        CostComponent("vision_attn_proj", 2.0 * attn_params * patches,
                      attn_params * width, precision),
        CostComponent("vision_mlp", 2.0 * mlp_params * patches,
                      mlp_params * width, precision),
        CostComponent("vision_attention", attention_flops, kv_bytes, precision),
        CostComponent("vision_project", 2.0 * project_params * soft,
                      project_params * width, precision),
    )
