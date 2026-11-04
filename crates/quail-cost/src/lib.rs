//! Quail's model and device specs and its cost model, in Rust.
//!
//! The modules with a Python counterpart compute the same numbers
//! in the same order, so results equal Python's:
//!
//! | Module | Python |
//! | --- | --- |
//! | `specs` | `quail/specs/base.py`, `quail/specs/loading.py` |
//! | `quantity` | numpy and Python summation and broadcasting |
//! | `work` | `quail/common/work.py` |
//! | `roofline` | `quail/common/roofline.py` |
//! | `dense_decoder_cost` | `quail/common/dense_decoder_cost.py` |
//! | `sol` | `quail/common/sol.py` |
//! | `budgets` | `quail/common/budgets.py` |
//! | `answers` | `quail/common/answers.py` |
//! | `draft_policy` | `quail/common/draft_policy.py` |
//! | `retention` | `quail/common/retention.py` |
//! | `decode` | `quail/common/decode.py`, the planner's part |
//! | `statistics`, `filters`, `packing`, `score`, `sae_features`, `classify`, `joins`, `extract` | none; Rust only |
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

    use crate::specs::{read_devices, read_models};

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
        let variant = &models["qwen3.5-4b-fp8"];
        assert_eq!(variant.layers, models["qwen3.5-4b-bf16"].layers);
        assert_eq!(variant.w_bytes, 1.0);
    }
}
