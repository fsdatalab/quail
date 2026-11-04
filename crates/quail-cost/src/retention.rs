//! The priority and price of document KV kept between operators, as
//! `quail/common/retention.py` computes them.

use crate::budgets::{checkpoint_pages, execution_pages, floor_divide_int};
use crate::dense_decoder_cost::attention_pair_flops;
use crate::quantity::Quantity;
use crate::sol::compute_seconds;
use crate::specs::{DeviceSpec, ModelSpec};
use crate::work::{Work, triangle};

/// A model's ideal prefix computation coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients {
    pub linear_seconds: f64,
    pub pair_seconds: f64,
    pub sliding_pair_seconds: f64,
    pub sliding_window: i64,
    /// Pages of the state a hybrid model's document keeps beside its
    /// KV; None for a model without state layers.
    pub checkpoint_pages: Option<f64>,
}

impl Coefficients {
    /// Ideal compute time to rebuild a prefix of this length.
    pub fn prefill_seconds(&self, tokens: f64) -> f64 {
        let tokens = Quantity::Scalar(tokens);
        let pairs = triangle(&tokens, 0).total();
        let sliding_pairs = triangle(&tokens, self.sliding_window).total();
        self.linear_seconds * tokens.total()
            + self.pair_seconds * pairs
            + self.sliding_pair_seconds * sliding_pairs
    }
}

/// Retained pages, after two execution chunks are reserved.
pub fn retention_pages(arena_tokens: i64, chunk_tokens: i64, page_tokens: i64) -> i64 {
    0_i64.max(
        floor_divide_int(arena_tokens, page_tokens) - execution_pages(chunk_tokens, page_tokens),
    )
}

/// The model's ideal prefix computation coefficients.
pub fn coefficients(model: &ModelSpec, device: &DeviceSpec, peak: bool) -> Coefficients {
    let (full, sliding) = attention_pair_flops(model);
    let bandwidth = device.arithmetic_bandwidth(&model.attention_precision);
    let one_token = Work {
        tokens: Quantity::Scalar(1.0),
        ..Work::default()
    };
    Coefficients {
        linear_seconds: compute_seconds(&one_token, model, device, peak),
        pair_seconds: full as f64 / bandwidth,
        sliding_pair_seconds: sliding as f64 / bandwidth,
        sliding_window: model.sliding_window,
        checkpoint_pages: if model.state_layers().is_empty() {
            None
        } else {
            Some(checkpoint_pages(model, device.page_tokens))
        },
    }
}
