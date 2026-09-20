"""Decoder components used by the roofline calculation.

Any pre-norm decoder with grouped-query attention and a gated MLP
is priced from its ModelSpec shape; nothing here is Qwen3-specific.
A mixture-of-experts model names its experts on the spec: FLOPs
follow the params one token multiplies, weight bytes follow the
params a full chunk reads.
"""

from __future__ import annotations

from quail.cost.roofline import CostComponent
from quail.cost.work import Work
from quail.specs import ModelSpec


def attention_projection_params(model: ModelSpec) -> int:
    """Return Q, K, V, and output projection parameters over the layers."""
    h = model.hidden
    return sum(h * model.n_q * head + 2 * h * n_kv * head
               + model.n_q * head * h
               for n_kv, head in model.kv_shapes)


def mlp_params(model: ModelSpec) -> int:
    """Return the MLP parameters one token multiplies."""
    return _mlp_params(model, model.experts_active)


def mlp_weight_params(model: ModelSpec) -> int:
    """Return the MLP parameters a full chunk reads: every expert."""
    return _mlp_params(model, model.experts)


def _mlp_params(model: ModelSpec, experts: int) -> int:
    """The dense MLP plus this many experts, over the layers."""
    per_layer = 3 * model.hidden * (
        model.intermediate + experts * model.expert_intermediate)
    return per_layer * model.layers


def dense_params(model: ModelSpec) -> int:
    """Return parameters used by the modeled dense components."""
    return attention_projection_params(model) + mlp_params(model)


def flops_per_pair(model: ModelSpec) -> int:
    """Return attention FLOPs per query and key pair in one layer."""
    return 4 * model.n_q * model.d_head


def kv_bytes_per_token(model: ModelSpec) -> float:
    """Return bytes in one token's KV across every layer."""
    return model.kappa


def dense_decoder_components(work: Work, model: ModelSpec,
                             passes: float) -> tuple[CostComponent, ...]:
    """Build the modeled decoder components for one work record."""
    if passes < 0:
        raise ValueError("passes must be nonnegative")
    attn_proj = attention_projection_params(model)
    mlp = mlp_params(model)
    return (
        CostComponent(
            name="attn_proj",
            flops=2.0 * attn_proj * work.tokens,
            bytes_moved=attn_proj * model.w_bytes * passes,
            precision=model.weight_precision,
        ),
        CostComponent(
            name="mlp",
            flops=2.0 * mlp * work.tokens,
            bytes_moved=mlp_weight_params(model) * model.w_bytes * passes,
            precision=model.weight_precision,
        ),
        CostComponent(
            name="attention",
            flops=(flops_per_pair(model) * work.pairs * model.layers),
            bytes_moved=(
                kv_bytes_per_token(model)
                * (work.kv_written + work.kv_read)),
            precision=model.attention_precision,
        ),
    )
