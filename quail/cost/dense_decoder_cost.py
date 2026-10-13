"""Decoder components used by the roofline calculation.

Any pre-norm decoder with grouped-query attention and a gated MLP
is priced from its ModelSpec shape; nothing here is Qwen3-specific.
A mixture-of-experts model names its experts on the spec: FLOPs
follow the params one token multiplies, weight bytes follow the
params a full chunk reads.
"""

from __future__ import annotations

from quail.cost.roofline import CostComponent, component_latencies
from quail.cost.work import Work
from quail.specs import AttentionRates, DeviceSpec, ModelSpec


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


def attention_pair_flops(model: ModelSpec) -> tuple[int, int]:
    """Return FLOPs per pair across full layers and sliding layers."""
    sliding = model.sliding_layer_set
    full_flops = sliding_flops = 0
    for layer, (_, head) in enumerate(model.kv_shapes):
        if layer in sliding:
            sliding_flops += 4 * model.n_q * head
        else:
            full_flops += 4 * model.n_q * head
    return full_flops, sliding_flops


def attention_rates(model: ModelSpec,
                    device: DeviceSpec | None) -> AttentionRates | None:
    """Return the device's measured attention rates for the model, or None.

    Only a model whose layers all share one head layout has them.
    """
    if device is None or model.sliding_layer_set:
        return None
    if any(shape != (model.n_kv, model.d_head) for shape in model.kv_shapes):
        return None
    return device.attention_rates_for(model.n_q, model.n_kv, model.d_head)


def measured_attention_seconds(work: Work, model: ModelSpec,
                               rates: AttentionRates) -> float:
    """Return attention compute time for the work at measured rates.

    The causal pairs price at the rate of their pair-weighted document
    length. The read pairs price at the rate of their mean query rows
    per sequence and their row-weighted context.
    """
    full_flops, _ = attention_pair_flops(model)
    causal_pairs = max(0.0, work.pairs - work.read_pairs)
    seconds = 0.0
    if causal_pairs > 0 and work.tokens > 0:
        length = max(1.0, 2.0 * causal_pairs / work.tokens - 1.0)
        seconds += full_flops * causal_pairs / rates.causal_rate(length)
    if work.read_pairs > 0 and work.read_rows > 0 and work.reads > 0:
        rows = work.read_rows / work.reads
        context = work.read_pairs / work.read_rows
        seconds += full_flops * work.read_pairs / rates.read_rate(rows, context)
    return seconds


def dense_decoder_components(work: Work, model: ModelSpec, passes: float,
                             device: DeviceSpec | None = None,
                             ) -> tuple[CostComponent, ...]:
    """Build the modeled decoder components for one work record.

    Args:
        work: The work record.
        model: The model that does it.
        passes: Forward passes, each reading the weights once.
        device: The device. When it has measured attention rates for
            the model's head layout, the attention component carries its
            compute time at those rates.

    Returns:
        The attn_proj, mlp, and attention components.

    Raises:
        ValueError: If passes is negative.
    """
    if passes < 0:
        raise ValueError("passes must be nonnegative")
    attn_proj = attention_projection_params(model)
    mlp = mlp_params(model)
    full_flops, sliding_flops = attention_pair_flops(model)
    rates = attention_rates(model, device)
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
            flops=full_flops * work.pairs + sliding_flops * work.sliding_pairs,
            bytes_moved=(
                model.kappa_full * (work.kv_written + work.kv_read)
                + model.kappa_sliding * (work.kv_written + work.sliding_kv_read)),
            precision=model.attention_precision,
            compute_seconds=(measured_attention_seconds(work, model, rates)
                             if rates is not None else None),
        ),
    )


def _readout(model, rows: float, passes: float, head_rows: int) -> CostComponent:
    """Price the answer readout: the decision head, or output-head rows.

    A decision model passes each option row and the prompt's last row
    through its fp32 head: four projections of the hidden size to the
    head width per row, and the head's weights once per pass. Any other
    model multiplies each answer row by head_rows rows of the bf16
    output head, the whole vocabulary when head_rows is 0, read once per
    pass.
    """
    if model.role == "decision":
        width, hidden = model.decision_head_dim, model.hidden
        return CostComponent(
            name="readout", flops=4.0 * width * hidden * rows,
            bytes_moved=4.0 * (4 * width * hidden + 2 * width + 4 * hidden) * passes,
            precision="fp32")
    head = 2.0 * model.hidden * (head_rows or model.vocab)
    return CostComponent(name="readout", flops=head * rows,
                         bytes_moved=head * passes, precision="bf16")


def decoder_seconds(work: Work, rows: float, passes: float, model, device,
             head_rows: int = 0) -> float:
    """Price work, answer rows, and forward passes with the component roofline."""
    components = dense_decoder_components(work, model, passes=passes,
                                          device=device)
    components += (_readout(model, rows, passes, head_rows),)
    return sum(component.seconds
               for component in component_latencies(components, device))
