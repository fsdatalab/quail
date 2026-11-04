//! Model and device specs, read from the TOML files in `quail/specs/data`.
//!
//! The fields and methods mirror `quail/specs/base.py`. Integer
//! arithmetic stays integral and converts to `f64` where Python's
//! `int * float` does, so results equal Python's.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Bytes per element of a Gated DeltaNet layer's recurrent state.
pub const GDN_STATE_ELEMENT_BYTES: i64 = 4;
/// Bytes per element of a convolution's kept input columns.
pub const CONV_STATE_ELEMENT_BYTES: i64 = 2;
/// Reserved activation bytes per hidden unit of a batched token.
pub const ACT_BYTES_PER_HIDDEN: i64 = 32;

/// Token layout for a diffusion model's classification answer.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerCanvas {
    pub rows: i64,
    pub turn_close_id: i64,
    pub pad_id: i64,
}

/// A sparse autoencoder read beside a model, for `AI.SAE_FEATURES`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaeSpec {
    pub hf_name: String,
    pub revision: String,
    pub file: String,
    pub num_features: i64,
    pub keep: i64,
}

fn text(value: &str) -> String {
    value.to_string()
}
fn fp8() -> String {
    text("fp8")
}
fn bf16() -> String {
    text("bf16")
}
fn qwen3() -> String {
    text("qwen3")
}
fn generative() -> String {
    text("generative")
}
fn causal() -> String {
    text("causal")
}
fn raw_v1() -> String {
    text("raw-v1")
}
fn ai_if() -> String {
    text("ai-if")
}
fn two() -> f64 {
    2.0
}
fn one() -> f64 {
    1.0
}
fn sixteen() -> i64 {
    16
}

/// One model's shape, weights, prompt format, and measured constants.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub name: String,
    pub params: f64,
    pub layers: i64,
    pub hidden: i64,
    pub num_q_heads: i64,
    pub num_kv_heads: i64,
    pub d_head: i64,
    pub ffn_width: i64,
    pub w_bytes: f64,
    #[serde(default)]
    pub hf_name: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default = "two")]
    pub kv_bytes: f64,
    #[serde(default)]
    pub w_mem_bytes: f64,
    #[serde(default)]
    pub vocab: i64,
    #[serde(default)]
    pub tied_head: bool,
    #[serde(default = "fp8")]
    pub weight_precision: String,
    #[serde(default = "bf16")]
    pub attention_precision: String,
    #[serde(default = "qwen3")]
    pub arch: String,
    #[serde(default = "generative")]
    pub role: String,
    #[serde(default)]
    pub decision_head_dim: i64,
    #[serde(default)]
    pub full_num_kv_heads: i64,
    #[serde(default)]
    pub full_d_head: i64,
    #[serde(default)]
    pub sliding_window: i64,
    #[serde(default)]
    pub experts: i64,
    #[serde(default)]
    pub experts_active: i64,
    #[serde(default)]
    pub expert_intermediate: i64,
    #[serde(default)]
    pub canvas_tokens: i64,
    #[serde(default)]
    pub canvas_answer_row: i64,
    #[serde(default)]
    pub answer_canvas: Option<AnswerCanvas>,
    #[serde(default)]
    pub turn_prefix: String,
    #[serde(default)]
    pub turn_suffix: String,
    #[serde(default)]
    pub bos_text: String,
    #[serde(default = "one")]
    pub answer_temperature: f64,
    #[serde(default)]
    pub read_found: Vec<(f64, f64)>,
    #[serde(default)]
    pub read_span_tokens: i64,
    #[serde(default)]
    pub set_size: i64,
    #[serde(default = "causal")]
    pub attention_mask: String,
    #[serde(default)]
    pub answer_spellings: Vec<Vec<String>>,
    #[serde(default = "raw_v1")]
    pub prompt_format: String,
    #[serde(default = "ai_if")]
    pub prompt_layout: String,
    #[serde(default)]
    pub chunk_cap_tokens: i64,
    #[serde(default)]
    pub moe_backend: Option<String>,
    #[serde(default)]
    pub vllm_reserve_bytes: f64,
    #[serde(default)]
    pub layer_kinds: Vec<String>,
    #[serde(default)]
    pub vision_tower: bool,
    #[serde(default)]
    pub max_num_seqs: i64,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub gdn_key_heads: i64,
    #[serde(default)]
    pub gdn_value_heads: i64,
    #[serde(default)]
    pub gdn_key_dim: i64,
    #[serde(default)]
    pub gdn_value_dim: i64,
    #[serde(default)]
    pub gdn_weight_precision: Option<String>,
    #[serde(default)]
    pub conv_width: i64,
    #[serde(default)]
    pub checkpoint_layers: i64,
    #[serde(default)]
    pub sae: Option<SaeSpec>,
}

impl ModelSpec {
    /// The kind of one layer.
    pub fn layer_kind(&self, layer: i64) -> &str {
        self.layer_kinds
            .get(layer as usize)
            .map(String::as_str)
            .unwrap_or("full_attention")
    }

    /// Whether the layer stores KV.
    pub fn keeps_kv(&self, layer: i64) -> bool {
        matches!(
            self.layer_kind(layer),
            "full_attention" | "sliding_attention"
        )
    }

    /// Whether a document's KV can be kept and read again.
    pub fn reuses_kv(&self) -> bool {
        self.attention_mask != "set"
    }

    fn layers_of(&self, kinds: &[&str]) -> Vec<i64> {
        (0..self.layers)
            .filter(|layer| kinds.contains(&self.layer_kind(*layer)))
            .collect()
    }

    /// The layers that store KV.
    pub fn attention_layers(&self) -> Vec<i64> {
        (0..self.layers)
            .filter(|layer| self.keeps_kv(*layer))
            .collect()
    }

    /// The layers that keep a Gated DeltaNet state.
    pub fn gdn_layers(&self) -> Vec<i64> {
        self.layers_of(&["gated_delta_net"])
    }

    /// The layers that keep a short convolution's input columns.
    pub fn short_conv_layers(&self) -> Vec<i64> {
        self.layers_of(&["short_conv"])
    }

    /// The layers that keep state between a document's tokens.
    pub fn state_layers(&self) -> Vec<i64> {
        self.layers_of(&["gated_delta_net", "short_conv"])
    }

    /// Channels of the convolution input a state layer keeps.
    pub fn conv_state_channels(&self) -> i64 {
        if self.gdn_layers().is_empty() {
            self.hidden
        } else {
            self.conv_channels()
        }
    }

    /// Channels of a Gated DeltaNet layer's convolution.
    pub fn conv_channels(&self) -> i64 {
        2 * self.gdn_key_heads * self.gdn_key_dim + self.gdn_value_heads * self.gdn_value_dim
    }

    /// Bytes of one layer's recurrent state.
    pub fn gdn_state_bytes(&self) -> i64 {
        self.gdn_value_heads * self.gdn_value_dim * self.gdn_key_dim * GDN_STATE_ELEMENT_BYTES
    }

    /// Bytes of one layer's convolution input columns.
    pub fn conv_state_bytes(&self) -> i64 {
        self.conv_channels() * (self.conv_width - 1) * CONV_STATE_ELEMENT_BYTES
    }

    /// Bytes of the state a sequence leaves in every state layer.
    pub fn checkpoint_bytes(&self) -> i64 {
        self.gdn_layers().len() as i64 * (self.gdn_state_bytes() + self.conv_state_bytes())
            + self.short_conv_layers().len() as i64
                * self.hidden
                * (self.conv_width - 1)
                * CONV_STATE_ELEMENT_BYTES
    }

    /// Whether the layer keeps every token with the full KV geometry.
    pub fn is_full_layer(&self, layer: i64) -> bool {
        self.layer_kind(layer) == "full_attention"
    }

    /// Per layer, the (KV heads, head dim) its KV stores.
    pub fn kv_shapes(&self) -> Vec<(i64, i64)> {
        let full_heads = if self.full_num_kv_heads != 0 {
            self.full_num_kv_heads
        } else {
            self.num_kv_heads
        };
        let full_head = if self.full_d_head != 0 {
            self.full_d_head
        } else {
            self.d_head
        };
        (0..self.layers)
            .map(|layer| {
                if !self.keeps_kv(layer) {
                    (0, 0)
                } else if self.is_full_layer(layer) {
                    (full_heads, full_head)
                } else {
                    (self.num_kv_heads, self.d_head)
                }
            })
            .collect()
    }

    /// Output columns of the widest dense projection in any layer.
    pub fn widest_projection(&self) -> i64 {
        let query_width = if self.attn_output_gate { 2 } else { 1 };
        let attention = self
            .kv_shapes()
            .iter()
            .map(|(heads, head)| (query_width * self.num_q_heads + 2 * heads) * head)
            .max()
            .unwrap_or(0);
        let gdn = 2 * self.gdn_key_heads * self.gdn_key_dim
            + 2 * self.gdn_value_heads * self.gdn_value_dim;
        self.ffn_width.max(attention).max(gdn)
    }

    /// KV elements per token: twice the sum of KV heads times head dim.
    pub fn kv_elements_per_token(&self) -> i64 {
        2 * self
            .kv_shapes()
            .iter()
            .map(|(heads, head)| heads * head)
            .sum::<i64>()
    }

    /// KV bytes per cached token, summed over the layers.
    pub fn kappa(&self) -> f64 {
        self.kv_elements_per_token() as f64 * self.kv_bytes
    }

    /// The layers that keep only the last `sliding_window` tokens of KV.
    pub fn sliding_layer_set(&self) -> Vec<i64> {
        if self.sliding_window == 0 {
            return Vec::new();
        }
        self.layers_of(&["sliding_attention"])
    }

    /// KV bytes per token on the sliding layers alone.
    pub fn kappa_sliding(&self) -> f64 {
        let sliding = self.sliding_layer_set();
        let elements: i64 = self
            .kv_shapes()
            .iter()
            .enumerate()
            .filter(|(layer, _)| sliding.contains(&(*layer as i64)))
            .map(|(_, (heads, head))| heads * head)
            .sum();
        2.0 * self.kv_bytes * elements as f64
    }

    /// KV bytes per token on the layers that keep every token.
    pub fn kappa_full(&self) -> f64 {
        self.kappa() - self.kappa_sliding()
    }

    /// Weight bytes as loaded.
    #[allow(non_snake_case)]
    pub fn W_mem(&self) -> f64 {
        if self.w_mem_bytes != 0.0 {
            self.w_mem_bytes
        } else {
            self.params * self.w_bytes
        }
    }

    /// Untied output-head bytes, or zero for tied weights.
    pub fn head_mem_bytes(&self) -> f64 {
        if self.tied_head {
            0.0
        } else {
            (self.vocab * self.hidden) as f64 * 2.0
        }
    }

    /// Model weight bytes resident on the GPU.
    #[allow(non_snake_case)]
    pub fn W_resident(&self) -> f64 {
        self.W_mem()
    }

    /// Reserved activation bytes per batched token.
    pub fn act_per_token(&self) -> f64 {
        let dense = ACT_BYTES_PER_HIDDEN * self.hidden;
        let expert = 2
            * self.experts_active
            * (self.hidden.max(self.expert_intermediate)
                + self.hidden.max(2 * self.expert_intermediate)
                + self.expert_intermediate);
        let routing = 4 * self.experts + 16 * self.experts_active;
        (dense + expert + routing) as f64
    }

    /// MLP intermediate width; gate_up packs two of them.
    pub fn intermediate(&self) -> i64 {
        self.ffn_width / 2
    }
}

/// One GPU's memory, bandwidth, and compute rates.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceSpec {
    pub name: String,
    pub mem_bytes: f64,
    pub hbm_bw: f64,
    pub fp8_flops: f64,
    pub bf16_flops: f64,
    pub fp32_flops: f64,
    #[serde(default)]
    pub usd_per_hour: f64,
    #[serde(default)]
    pub price_source: String,
    #[serde(default)]
    pub cuda_name: String,
    #[serde(default)]
    pub modal_gpu: String,
    #[serde(default = "sixteen")]
    pub page_tokens: i64,
    #[serde(default)]
    pub mamf_fp8_flops: f64,
    #[serde(default)]
    pub mamf_bf16_flops: f64,
}

impl DeviceSpec {
    /// Arithmetic throughput for one component precision.
    ///
    /// # Panics
    ///
    /// The precision is not fp8, bf16, or fp32.
    pub fn arithmetic_bandwidth(&self, precision: &str) -> f64 {
        match precision {
            "fp8" => self.fp8_flops,
            "bf16" => self.bf16_flops,
            "fp32" => self.fp32_flops,
            other => panic!("unsupported precision: {other}"),
        }
    }

    /// Measured GEMM throughput, or the specification rate.
    pub fn gemm_flops(&self, precision: &str) -> f64 {
        let measured = match precision {
            "fp8" => self.mamf_fp8_flops,
            "bf16" => self.mamf_bf16_flops,
            _ => 0.0,
        };
        if measured != 0.0 {
            measured
        } else {
            self.arithmetic_bandwidth(precision)
        }
    }
}

/// An error reading a spec directory.
#[derive(Debug)]
pub struct SpecError(pub String);

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpecError {}

fn read_tables(directory: &Path) -> Result<BTreeMap<String, toml::Table>, SpecError> {
    let mut tables = BTreeMap::new();
    let entries = fs::read_dir(directory)
        .map_err(|error| SpecError(format!("{}: {error}", directory.display())))?;
    for entry in entries {
        let path = entry.map_err(|error| SpecError(error.to_string()))?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| SpecError(format!("{}: {error}", path.display())))?;
        let table: toml::Table = text
            .parse()
            .map_err(|error| SpecError(format!("{}: {error}", path.display())))?;
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("");
        if table.get("name").and_then(toml::Value::as_str) != Some(stem) {
            return Err(SpecError(format!("{} names another spec", path.display())));
        }
        tables.insert(stem.to_string(), table);
    }
    Ok(tables)
}

fn merged(
    name: &str,
    tables: &BTreeMap<String, toml::Table>,
    seen: &mut Vec<String>,
) -> Result<toml::Table, SpecError> {
    if seen.iter().any(|visited| visited == name) {
        return Err(SpecError(format!("model bases form a cycle at {name}")));
    }
    let table = tables
        .get(name)
        .ok_or_else(|| SpecError(format!("no model spec named {name}")))?;
    let mut entries = table.clone();
    let Some(base) = entries.remove("base") else {
        return Ok(entries);
    };
    let base = base
        .as_str()
        .ok_or_else(|| SpecError(format!("{name}: base is not a name")))?
        .to_string();
    seen.push(name.to_string());
    let mut values = merged(&base, tables, seen)?;
    values.extend(entries);
    Ok(values)
}

/// Read every model spec in a directory, by name.
pub fn read_models(directory: &Path) -> Result<BTreeMap<String, ModelSpec>, SpecError> {
    let tables = read_tables(directory)?;
    tables
        .keys()
        .map(|name| {
            let values = merged(name, &tables, &mut Vec::new())?;
            let spec = ModelSpec::deserialize(values)
                .map_err(|error| SpecError(format!("{name}: {error}")))?;
            Ok((name.clone(), spec))
        })
        .collect()
}

/// Read every device spec in a directory, by name.
pub fn read_devices(directory: &Path) -> Result<BTreeMap<String, DeviceSpec>, SpecError> {
    read_tables(directory)?
        .into_iter()
        .map(|(name, table)| {
            let spec = DeviceSpec::deserialize(table)
                .map_err(|error| SpecError(format!("{name}: {error}")))?;
            Ok((name, spec))
        })
        .collect()
}
