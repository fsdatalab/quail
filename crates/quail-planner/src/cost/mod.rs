//! Quail's model and device specs and its cost model.
//!
//! The modules with a Python counterpart compute the same numbers
//! in the same order, so results equal Python's:
//!
//! | Module | Python |
//! | --- | --- |
//! | `specs` | `quail/specs/base.py`, `quail/specs/loading.py` |
//! | `quantity` | numpy and Python summation and broadcasting |
//! | `work` | `quail/common/work.py` |
//! | `budgets` | `quail/common/budgets.py` |
//! | `answers` | `quail/common/answers.py` |
//! | `draft_policy` | `quail/common/draft_policy.py` |
//! | `retention` | `quail/common/retention.py` |
//! | `decode` | `quail/common/decode.py`, the planner's part |
//! | `roofline`, `dense_decoder_cost`, `sol`, `statistics`, `filters`, `packing`, `score`, `sae_features`, `classify`, `joins`, `extract` | none; Rust only |
//! | `labels` | `quail/labels.py`, the trie the costs read |

pub mod answers;
pub mod budgets;
pub mod classify;
pub mod decode;
pub mod dense_decoder_cost;
pub mod draft_policy;
pub mod extract;
pub mod filters;
pub mod joins;
pub mod labels;
pub mod packing;
pub mod quantity;
pub mod retention;
pub mod roofline;
pub mod sae_features;
pub mod score;
pub mod sol;
pub mod specs;
pub mod statistics;
pub mod work;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::cost::specs::{ModelSpec, Precision, read_devices, read_models};

    fn data(kind: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../quail/specs/data")
            .join(kind)
    }

    #[test]
    fn every_built_in_spec_reads() {
        let models = read_models(&data("models")).unwrap();
        let devices = read_devices(&data("devices")).unwrap();
        assert_eq!(models.len(), 15);
        assert_eq!(devices.len(), 4);
        for model in models.values() {
            let json = serde_json::to_value(model).unwrap();
            assert_eq!(serde_json::from_value::<ModelSpec>(json).unwrap(), *model);
            let toml = toml::to_string(model).unwrap();
            assert_eq!(toml::from_str::<ModelSpec>(&toml).unwrap(), *model);
        }
        let variant = &models["qwen3.5-4b-fp8"];
        assert_eq!(variant.layers, models["qwen3.5-4b-bf16"].layers);
        assert_eq!(variant.w_bytes, 1.0);
        assert_eq!(variant.weight_precision, Precision::Fp8);
        assert_eq!(variant.attention_precision, Precision::Bf16);
        assert_eq!(variant.gdn_weight_precision, Some(Precision::Bf16));
    }

    #[test]
    fn malformed_model_precisions_are_deserialization_errors() {
        let source = std::fs::read_to_string(data("models").join("qwen3-4b-fp8.toml")).unwrap();
        let model: ModelSpec = toml::from_str(&source).unwrap();
        for field in [
            "weight_precision",
            "attention_precision",
            "gdn_weight_precision",
        ] {
            for malformed in ["\"fp16\"", "\"FP8\"", "\"\"", "8"] {
                let invalid = format!("{source}\n{field} = {malformed}\n");
                let error = toml::from_str::<ModelSpec>(&invalid)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(field), "{error}");

                let mut json = serde_json::to_value(&model).unwrap();
                json[field] = serde_json::from_str(malformed).unwrap();
                assert!(serde_json::from_value::<ModelSpec>(json).is_err());
            }
        }
    }

    #[test]
    fn precision_wire_strings_and_device_rates_are_preserved() {
        let devices = read_devices(&data("devices")).unwrap();
        for (precision, wire, bytes) in [
            (Precision::Fp8, "fp8", 1),
            (Precision::Bf16, "bf16", 2),
            (Precision::Fp32, "fp32", 4),
        ] {
            assert_eq!(serde_json::to_value(precision).unwrap(), wire);
            assert_eq!(
                serde_json::from_value::<Precision>(serde_json::json!(wire)).unwrap(),
                precision,
            );
            assert_eq!(precision.bytes(), bytes);
            for device in devices.values() {
                let (peak, measured) = match precision {
                    Precision::Fp8 => (device.fp8_flops, device.mamf_fp8_flops),
                    Precision::Bf16 => (device.bf16_flops, device.mamf_bf16_flops),
                    Precision::Fp32 => (device.fp32_flops, 0.0),
                };
                assert_eq!(device.arithmetic_bandwidth(precision), peak);
                assert_eq!(
                    device.gemm_flops(precision),
                    if measured != 0.0 { measured } else { peak },
                );
            }
        }
    }
}
