//! Work and time of an `AI.SAE_FEATURES` call.

use crate::cost::budgets::parallel_copies;
use crate::cost::quantity::Quantity;
use crate::cost::score::{python_min, score_work};
use crate::cost::sol::speed_of_light;
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::statistics::AliasStats;
use crate::cost::work::Work;

/// Encoded rows, including a shared max hook in a mixed pipeline.
pub fn sae_encoded_tokens(
    pooling: &str,
    expected_rows: f64,
    prompt_tokens: f64,
    max_pooling: bool,
) -> f64 {
    (if pooling == "last" {
        expected_rows
    } else {
        0.0
    }) + (if pooling == "max" || max_pooling {
        prompt_tokens
    } else {
        0.0
    })
}

/// Prepared inputs for one feature call's cost at any count.
#[derive(Debug, Clone)]
pub struct SaeFeaturesCost {
    /// Token counts of the head and the question tail.
    pub token_lengths: Vec<i64>,
    pub summary: AliasStats,
    pub pooling: String,
    pub model: ModelSpec,
    pub device: DeviceSpec,
    pub chunk_tokens: i64,
    pub gpus: i64,
    pub max_pooling: bool,
}

impl SaeFeaturesCost {
    /// Model work and seconds for the expected documents.
    pub fn estimate(&self, expected_rows: f64, resident: bool) -> (Work, f64) {
        let fixed: i64 = self.token_lengths.iter().sum();
        let mean = self.summary.mean();
        let shared = self.token_lengths[0];
        let prefix = shared as f64 + if resident { mean } else { 0.0 };
        let groups = if resident {
            0.0
        } else {
            python_min(self.gpus as f64, expected_rows)
        };
        let mut work = score_work(
            expected_rows,
            mean,
            fixed,
            if resident {
                0.0
            } else {
                self.summary.variance()
            },
            prefix,
            0.0,
            groups,
            shared,
            self.gpus,
        );
        let tokens_per_row = if resident {
            self.token_lengths[1] as f64
        } else {
            fixed as f64 + mean
        };
        work.sae_encoded = Quantity::Scalar(sae_encoded_tokens(
            &self.pooling,
            expected_rows,
            expected_rows * tokens_per_row,
            self.max_pooling,
        ));
        let copies = parallel_copies(
            self.gpus,
            python_min(expected_rows, self.summary.count as f64),
        );
        let seconds = speed_of_light(
            &(&work * (1.0 / copies)),
            &self.model,
            &self.device,
            self.chunk_tokens,
            0,
        )
        .seconds();
        (work, seconds)
    }
}
