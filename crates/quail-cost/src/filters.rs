//! The cost of one boolean filter, as `quail/cost/filters.py` prices it.

use crate::quantity::Quantity;
use crate::sol::unrounded_seconds;
use crate::specs::{DeviceSpec, ModelSpec};
use crate::work::{ask, packed, scan};

/// Ideal time for one filter evaluation over a document of the mean
/// length; the first filter computes the document, a later one reads
/// its KV. A set-attention model reads the document again every pass.
pub fn filter_cost(
    question_tokens: i64,
    mean_tokens: f64,
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    first: bool,
) -> f64 {
    let mean = Quantity::Scalar(mean_tokens);
    let question = Quantity::Scalar(question_tokens as f64);
    let work = if model.attention_mask == "set" {
        packed(&mean, &[question_tokens as f64], model.sliding_window)
    } else if first {
        scan(&mean, &question, model.sliding_window)
    } else {
        ask(&mean, &question, model.sliding_window)
    };
    unrounded_seconds(&work, model, device, chunk_tokens, 0.0, 0)
}
