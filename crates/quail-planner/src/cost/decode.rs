//! Decode pass time from the specs alone, as `quail/common/decode.py`
//! prices it for the planner.
//!
//! The executor's per-row admission methods (`row_work`,
//! `replay_seconds`, `gain`) stay in Python with the executor.

use crate::cost::dense_decoder_cost::dense_decoder_components;
use crate::cost::quantity::python_sum;
use crate::cost::retention::{Coefficients, coefficients};
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::work::Work;

/// Prices decode passes and prefix recomputation in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeCost {
    pub recompute_cost: Coefficients,
    pub weight_seconds: f64,
    pub kv_seconds: f64,
    pub sliding_kv_seconds: f64,
    pub state_seconds: f64,
}

impl DecodeCost {
    /// Coefficients from the model's shapes and the device's peak rates.
    pub fn from_specs(model: &ModelSpec, device: &DeviceSpec) -> Self {
        let weight_bytes = python_sum(
            dense_decoder_components(&Work::default(), model, 1.0)
                .iter()
                .map(|component| component.bytes_moved.total()),
        );
        Self {
            recompute_cost: coefficients(model, device, true),
            weight_seconds: weight_bytes / device.hbm_bw,
            kv_seconds: model.kappa_full() / device.hbm_bw,
            sliding_kv_seconds: model.kappa_sliding() / device.hbm_bw,
            state_seconds: model.checkpoint_bytes() as f64 / device.hbm_bw,
        }
    }

    /// The fewest seconds `passes` passes doing `work` in all take.
    ///
    /// # Panics
    ///
    /// A count in `work` is an array.
    pub fn lower_bound_seconds(&self, work: &Work, passes: f64) -> f64 {
        let value =
            |quantity: &crate::cost::quantity::Quantity| quantity.scalar().expect("summed work");
        let policy = &self.recompute_cost;
        let memory = passes * self.weight_seconds
            + self.kv_seconds * value(&work.kv_read)
            + self.sliding_kv_seconds * value(&work.sliding_kv_read)
            + self.state_seconds * (value(&work.state_read) + value(&work.state_written));
        let compute = policy.linear_seconds * value(&work.tokens)
            + policy.pair_seconds * value(&work.pairs)
            + policy.sliding_pair_seconds * value(&work.sliding_pairs);
        if compute > memory { compute } else { memory }
    }
}
