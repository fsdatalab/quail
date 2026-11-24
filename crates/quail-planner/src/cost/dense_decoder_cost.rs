use crate::cost::work::{Quantity, Work};
use crate::specs::{DeviceSpec, ModelSpec, Precision};

pub const GDN_CHUNK_TOKENS: i64 = 64;

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

pub fn gdn_projection_params(model: &ModelSpec) -> i64 {
    let key_width = model.gdn_key_heads * model.gdn_key_dim;
    let value_width = model.gdn_value_heads * model.gdn_value_dim;
    let per_layer = model.hidden * (2 * key_width + 2 * value_width + 2 * model.gdn_value_heads)
        + value_width * model.hidden;
    per_layer * model.gdn_layers().len() as i64
}

pub fn short_conv_projection_params(model: &ModelSpec) -> i64 {
    4 * model.hidden * model.hidden * model.short_conv_layers().len() as i64
}

pub fn gdn_core_flops_per_token(model: &ModelSpec) -> f64 {
    let c = GDN_CHUNK_TOKENS;
    let (k, v) = (model.gdn_key_dim, model.gdn_value_dim);
    let per_head = (6 * c * k + 4 * c * v + 6 * k * v) as f64 + (2 * c * c) as f64 / 3.0;
    per_head * model.gdn_value_heads as f64 * model.gdn_layers().len() as f64
}

pub fn conv_flops_per_token(model: &ModelSpec) -> f64 {
    let gdn = model.gdn_layers().len() as i64 * model.conv_channels();
    let short = model.short_conv_layers().len() as i64 * model.hidden;
    let attention = model.attention_conv_layers().len() as i64 * model.attention_conv_channels;
    2.0 * model.conv_width as f64 * (gdn + short + attention) as f64
}

fn mlp_params_with(model: &ModelSpec, experts: i64, layers: i64) -> i64 {
    3 * model.hidden * (model.intermediate() + experts * model.expert_intermediate) * layers
}

pub fn mlp_params(model: &ModelSpec) -> i64 {
    mlp_params_with(model, model.experts_active, model.layers)
}

pub fn mlp_weight_params(model: &ModelSpec) -> i64 {
    mlp_params_with(model, model.experts, model.weight_layer_count())
}

pub fn dense_params(model: &ModelSpec) -> i64 {
    attention_projection_params(model)
        + gdn_projection_params(model)
        + short_conv_projection_params(model)
        + model.extra_projection_params
        + mlp_params(model)
}

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

pub fn dense_decoder_components(work: &Work, model: &ModelSpec, passes: f64) -> Vec<CostComponent> {
    assert!(passes >= 0.0, "passes must be nonnegative");
    let separate_gdn = if model.gdn_has_own_precision() {
        gdn_projection_params(model)
    } else {
        0
    };
    let attn_proj = attention_projection_params(model)
        + gdn_projection_params(model)
        + short_conv_projection_params(model)
        + model.extra_projection_params
        - separate_gdn;
    let weight_share = model.weight_layer_count() as f64 / model.layers as f64;
    let fp8_share = model.mlp_fp8_layers as f64 / model.layers as f64;
    let fp8_weight_share = model.mlp_fp8_layers as f64 / model.weight_layer_count() as f64;
    let mlp = mlp_params(model) as f64;
    let mlp_weights = mlp_weight_params(model) as f64;
    let (full_flops, sliding_flops) = attention_pair_flops(model);
    let mut components = vec![
        CostComponent {
            name: "attn_proj",
            flops: (2.0 * attn_proj as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar(attn_proj as f64 * weight_share * model.w_bytes * passes),
            precision: model.weight_precision,
            gemm: true,
        },
        CostComponent {
            name: "mlp",
            flops: (2.0 * mlp * (1.0 - fp8_share)) * &work.tokens,
            bytes_moved: Quantity::Scalar(
                mlp_weights * (1.0 - fp8_weight_share) * model.mlp_weight_bytes() * passes,
            ),
            precision: model.mlp_precision(),
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
    if model.mlp_fp8_layers != 0 {
        components.push(CostComponent {
            name: "mlp_fp8",
            flops: (2.0 * mlp * fp8_share) * &work.tokens,
            bytes_moved: Quantity::Scalar(
                mlp_weights * fp8_weight_share * Precision::Fp8.bytes() * passes,
            ),
            precision: Precision::Fp8,
            gemm: true,
        });
    }
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
    if separate_gdn != 0 {
        let precision = model.gdn_precision();
        components.push(CostComponent {
            name: "gdn_proj",
            flops: (2.0 * separate_gdn as f64) * &work.tokens,
            bytes_moved: Quantity::Scalar(separate_gdn as f64 * precision.bytes() * passes),
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
    if !model.gdn_layers().is_empty()
        || !model.short_conv_layers().is_empty()
        || !model.attention_conv_layers().is_empty()
    {
        components.push(CostComponent {
            name: "conv",
            flops: conv_flops_per_token(model) * &work.tokens,
            bytes_moved: Quantity::Scalar(0.0),
            precision: Precision::Fp32,
            gemm: false,
        });
    }
    if let Some((num_features, keep)) = model.sae_shape()
        && work.sae_encoded.is_nonzero()
    {
        let encode = 2.0 * model.hidden as f64 * num_features as f64;
        components.push(CostComponent {
            name: "sae_encode",
            flops: encode * &work.sae_encoded,
            bytes_moved: &Quantity::Scalar(encode * passes)
                + &(&work.sae_encoded * (2.0 * model.hidden as f64 + 8.0 * keep as f64)),
            precision: Precision::Bf16,
            gemm: true,
        });
    }
    let rows = work.readout_rows.scalar().expect("summed work");
    if rows != 0.0 {
        let head_rows = work.readout_head_rows.scalar().expect("summed work") / rows;
        components.push(readout(model, rows, passes, head_rows));
    }
    components
}

fn readout(model: &ModelSpec, rows: f64, passes: f64, head_rows: f64) -> CostComponent {
    if model.reads_letter_logits() {
        let codes = (crate::constants::MAX_LETTERS as i64 * model.hidden) as f64;
        return CostComponent {
            name: "readout",
            flops: Quantity::Scalar(2.0 * codes * rows),
            bytes_moved: Quantity::Scalar(4.0 * codes * passes),
            precision: Precision::Fp32,
            gemm: false,
        };
    }
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
    let head = 2.0 * model.hidden as f64 * head_rows;
    CostComponent {
        name: "readout",
        flops: Quantity::Scalar(head * rows),
        bytes_moved: Quantity::Scalar(head * passes),
        precision: Precision::Bf16,
        gemm: true,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CostComponent {
    pub name: &'static str,
    pub flops: Quantity,
    pub bytes_moved: Quantity,
    pub precision: Precision,
    pub gemm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComponentLatency {
    pub component: CostComponent,
    pub compute_seconds: Quantity,
    pub memory_seconds: Quantity,
}

impl ComponentLatency {
    pub fn seconds(&self) -> f64 {
        let compute = self.compute_seconds.scalar().expect("scalar compute time");
        let memory = self.memory_seconds.scalar().expect("scalar memory time");
        if memory > compute { memory } else { compute }
    }
}

pub fn component_latency(component: CostComponent, device: &DeviceSpec) -> ComponentLatency {
    let bandwidth = if component.gemm {
        device.gemm_flops(component.precision)
    } else {
        device.arithmetic_bandwidth(component.precision)
    };
    ComponentLatency {
        compute_seconds: &component.flops / bandwidth,
        memory_seconds: &component.bytes_moved / device.hbm_bw,
        component,
    }
}

pub fn component_latencies(
    components: Vec<CostComponent>,
    device: &DeviceSpec,
) -> Vec<ComponentLatency> {
    components
        .into_iter()
        .map(|component| component_latency(component, device))
        .collect()
}
