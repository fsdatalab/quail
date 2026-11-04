//! Quail's model and device specs and its cost model, in Rust.
//!
//! Each module mirrors one Python module and computes the same
//! numbers in the same order, so results equal Python's:
//!
//! | Module | Python |
//! | --- | --- |
//! | `specs` | `quail/specs/base.py`, `quail/specs/loading.py` |
//! | `quantity` | numpy and Python summation and broadcasting |
//! | `work` | `quail/common/work.py` |
//! | `roofline` | `quail/common/roofline.py` |
//! | `dense_decoder_cost` | `quail/common/dense_decoder_cost.py` |
//! | `sol` | `quail/common/sol.py` |

pub mod dense_decoder_cost;
pub mod quantity;
pub mod roofline;
pub mod sol;
pub mod specs;
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
