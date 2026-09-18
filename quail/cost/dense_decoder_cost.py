"""Decoder components used by the roofline calculation.

Any pre-norm decoder with grouped-query attention and a gated MLP
is priced from its ModelSpec shape; nothing here is Qwen3-specific.
A mixture-of-experts model gives its per-layer attention and MLP
parameter counts on the spec: FLOPs follow the params one token
multiplies, weight bytes follow the params a full chunk reads.
"""

from __future__ import annotations

from quail.cost.roofline import CostComponent
from quail.cost.work import Work
from quail.specs import ModelSpec


def attention_projection_params(model: ModelSpec) -> int:
    """Return Q, K, V, and output projection parameters."""
    if model.attn_params_per_layer:
        return model.attn_params_per_layer * model.layers
    h = model.hidden
    head = model.d_head
    per_layer = (
        h * model.n_q * head
        + 2 * h * model.n_kv * head
        + model.n_q * head * h
    )
    return per_layer * model.layers


def mlp_params(model: ModelSpec) -> int:
    """Return the MLP parameters one token multiplies."""
    if model.mlp_active_params_per_layer:
        return model.mlp_active_params_per_layer * model.layers
    return 3 * model.hidden * model.intermediate * model.layers


def mlp_weight_params(model: ModelSpec) -> int:
    """Return the MLP parameters a full chunk reads: every expert."""
    if model.mlp_total_params_per_layer:
        return model.mlp_total_params_per_layer * model.layers
    return mlp_params(model)


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
