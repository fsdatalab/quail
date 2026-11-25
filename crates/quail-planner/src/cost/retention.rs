use crate::budgets::{checkpoint_pages, execution_pages, floor_divide_int};
use crate::cost::dense_decoder_cost::attention_pair_flops;
use crate::cost::sol::compute_seconds;
use crate::cost::work::Quantity;
use crate::cost::work::Work;
use crate::specs::{DeviceSpec, ModelSpec};

#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients {
    pub linear_seconds: f64,
    pub pair_seconds: f64,
    pub sliding_pair_seconds: f64,
    pub sliding_window: i64,
    pub checkpoint_pages: Option<f64>,
}

pub fn retention_pages(arena_tokens: i64, chunk_tokens: i64, page_tokens: i64) -> i64 {
    0_i64.max(
        floor_divide_int(arena_tokens, page_tokens) - execution_pages(chunk_tokens, page_tokens),
    )
}

pub fn coefficients(model: &ModelSpec, device: &DeviceSpec, peak: bool) -> Coefficients {
    let (full, sliding) = attention_pair_flops(model);
    let bandwidth = device.arithmetic_bandwidth(model.attention_precision);
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
