//! Decoder components priced by the roofline.

use crate::cost::quantity::Quantity;
use crate::cost::roofline::CostComponent;
use crate::cost::specs::{ModelSpec, Precision};
use crate::cost::work::Work;

/// vLLM's `FLA_CHUNK_SIZE`: tokens per chunk of the chunked gated delta rule.
pub const GDN_CHUNK_TOKENS: i64 = 64;

/// Q, K, V, and output projection parameters over the layers.
pub fn attention_projection_params(model: &ModelSpec) -> i64 {
    let h = model.hidden;
    let query_width = if model.attn_output_gate { 2 } else { 1 };
    model
        .kv_shapes()
        .iter()
        .map(|(heads, head)| {
            h * query_width * model.num_q_heads * head
                + 2 * h * heads * head
                + model.num_q_heads * head * h
        })
        .sum()
}

/// Gated DeltaNet input and output projection parameters.
pub fn gdn_projection_params(model: &ModelSpec) -> i64 {
    let key_width = model.gdn_key_heads * model.gdn_key_dim;
    let value_width = model.gdn_value_heads * model.gdn_value_dim;
    let per_layer = model.hidden * (2 * key_width + 2 * value_width + 2 * model.gdn_value_heads)
        + value_width * model.hidden;
    per_layer * model.gdn_layers().len() as i64
}

/// Short convolution input and output projection parameters.
pub fn short_conv_projection_params(model: &ModelSpec) -> i64 {
    4 * model.hidden * model.hidden * model.short_conv_layers().len() as i64
}

/// FLOPs per token of the chunked gated delta rule, over all layers.
pub fn gdn_core_flops_per_token(model: &ModelSpec) -> f64 {
    let c = GDN_CHUNK_TOKENS;
    let (k, v) = (model.gdn_key_dim, model.gdn_value_dim);
    let per_head = (6 * c * k + 4 * c * v + 6 * k * v) as f64 + (2 * c * c) as f64 / 3.0;
    per_head * model.gdn_value_heads as f64 * model.gdn_layers().len() as f64
}

/// FLOPs per token of the depthwise convolutions, over all layers.
pub fn conv_flops_per_token(model: &ModelSpec) -> f64 {
    let gdn = model.gdn_layers().len() as i64 * model.conv_channels();
    let short = model.short_conv_layers().len() as i64 * model.hidden;
    2.0 * model.conv_width as f64 * (gdn + short) as f64
}

fn mlp_params_with(model: &ModelSpec, experts: i64) -> i64 {
    3 * model.hidden * (model.intermediate() + experts * model.expert_intermediate) * model.layers
}

/// The MLP parameters one token multiplies.
pub fn mlp_params(model: &ModelSpec) -> i64 {
    mlp_params_with(model, model.experts_active)
}

/// The MLP parameters a full chunk reads: every expert.
pub fn mlp_weight_params(model: &ModelSpec) -> i64 {
    mlp_params_with(model, model.experts)
}

/// Parameters used by the modeled dense components.
pub fn dense_params(model: &ModelSpec) -> i64 {
    attention_projection_params(model)
        + gdn_projection_params(model)
        + short_conv_projection_params(model)
        + mlp_params(model)
}

/// FLOPs per attention pair across the full layers and the sliding layers.
pub fn attention_pair_flops(model: &ModelSpec) -> (i64, i64) {
    let sliding = model.sliding_layer_set();
    let (mut full, mut sliding_flops) = (0, 0);
    for (layer, (_, head)) in model.kv_shapes().iter().enumerate() {
        if sliding.contains(&(layer as i64)) {
            sliding_flops += 4 * model.num_q_heads * head;
        } else {
            full += 4 * model.num_q_heads * head;
        }
    }
    (full, sliding_flops)
}

/// The modeled decoder components, with a readout when `rows` is set.
///
/// # Panics
///
/// `passes` is negative, or `work.sae_encoded` is an array.
pub fn dense_decoder_components(
    work: &Work,
    model: &ModelSpec,
    passes: f64,
    rows: f64,
    head_rows: i64,
) -> Vec<CostComponent> {
    assert!(passes >= 0.0, "passes must be nonnegative");
    let separate_gdn = if model.gdn_weight_precision.is_some() {
        gdn_projection_params(model)
    } else {
        0
    };
    let attn_proj = attention_projection_params(model)
        + gdn_projection_params(model)
        + short_conv_projection_params(model)
        - separate_gdn;
    let mlp = mlp_params(model);
    let (full_flops, sliding_flops) = attention_pair_flops(model);
    let mut components = vec![
        CostComponent {
            name: "attn_proj",
            flops: (2.0 * attn_proj as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar(attn_proj as f64 * model.w_bytes * passes),
            precision: model.weight_precision,
            gemm: true,
        },
        CostComponent {
            name: "mlp",
            flops: (2.0 * mlp as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar(mlp_weight_params(model) as f64 * model.w_bytes * passes),
            precision: model.weight_precision,
            gemm: true,
        },
        CostComponent {
            name: "attention",
            flops: &(full_flops as f64 * &work.pairs)
                + &(sliding_flops as f64 * &work.sliding_pairs),
            bytes_moved: &(model.kappa_full() * &(&work.kv_written + &work.kv_read))
                + &(model.kappa_sliding() * &(&work.kv_written + &work.sliding_kv_read)),
            precision: model.attention_precision,
            gemm: false,
        },
    ];
    if model.experts != 0 {
        let router = model.layers * model.hidden * model.experts;
        components.push(CostComponent {
            name: "router",
            flops: (2.0 * router as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar((2 * router) as f64 * passes),
            precision: Precision::Bf16,
            gemm: true,
        });
    }
    if let Some(precision) = model.gdn_weight_precision
        && separate_gdn != 0
    {
        components.push(CostComponent {
            name: "gdn_proj",
            flops: (2.0 * separate_gdn as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar((separate_gdn * precision.bytes()) as f64 * passes),
            precision,
            gemm: true,
        });
    }
    if !model.gdn_layers().is_empty() {
        components.push(CostComponent {
            name: "gdn_core",
            flops: gdn_core_flops_per_token(model) * &work.tokens,
            bytes_moved: Quantity::Scalar(0.0),
            precision: Precision::Bf16,
            gemm: false,
        });
    }
    if model.checkpoint_bytes() != 0 {
        components.push(CostComponent {
            name: "state",
            flops: Quantity::Scalar(0.0),
            bytes_moved: model.checkpoint_bytes() as f64
                * &(&work.state_written + &work.state_read),
            precision: Precision::Bf16,
            gemm: false,
        });
    }
    if !model.gdn_layers().is_empty() || !model.short_conv_layers().is_empty() {
        components.push(CostComponent {
            name: "conv",
            flops: conv_flops_per_token(model) * &work.tokens,
            bytes_moved: Quantity::Scalar(0.0),
            precision: Precision::Fp32,
            gemm: false,
        });
    }
    if let Some(sae) = &model.sae
        && work.sae_encoded.is_nonzero()
    {
        let encode = 2.0 * model.hidden as f64 * sae.num_features as f64;
        components.push(CostComponent {
            name: "sae_encode",
            flops: encode * &work.sae_encoded,
            bytes_moved: &Quantity::Scalar(encode * passes)
                + &(&work.sae_encoded * (2.0 * model.hidden as f64 + 8.0 * sae.keep as f64)),
            precision: Precision::Bf16,
            gemm: true,
        });
    }
    if rows != 0.0 {
        components.push(readout(model, rows, passes, head_rows));
    }
    components
}

/// The answer readout: the decision head, or the output head's rows.
fn readout(model: &ModelSpec, rows: f64, passes: f64, head_rows: i64) -> CostComponent {
    if model.role == "decision" {
        let (width, hidden) = (model.decision_head_dim, model.hidden);
        return CostComponent {
            name: "readout",
            flops: Quantity::Scalar(4.0 * width as f64 * hidden as f64 * rows),
            bytes_moved: Quantity::Scalar(
                4.0 * (4 * width * hidden + 2 * width + 4 * hidden) as f64 * passes,
            ),
            precision: Precision::Fp32,
            gemm: false,
        };
    }
    let head_rows = if head_rows != 0 {
        head_rows
    } else {
        model.vocab
    };
    let head = 2.0 * model.hidden as f64 * head_rows as f64;
    CostComponent {
        name: "readout",
        flops: Quantity::Scalar(head * rows),
        bytes_moved: Quantity::Scalar(head * passes),
        precision: Precision::Bf16,
        gemm: true,
    }
}
