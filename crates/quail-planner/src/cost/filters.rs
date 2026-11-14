//! The work of one boolean filter.

use crate::cost::quantity::Quantity;
use crate::cost::specs::ModelSpec;
use crate::cost::work::{Work, ask, packed, scan};

/// The work of one filter evaluation over a document of the mean
/// length; the first filter computes the document, a later one reads
/// its KV. A set-attention model reads the document again every pass.
pub fn filter_work(question_tokens: i64, mean_tokens: f64, model: &ModelSpec, first: bool) -> Work {
    let mean = Quantity::Scalar(mean_tokens);
    let question = Quantity::Scalar(question_tokens as f64);
    if model.attention_mask == "set" {
        packed(&mean, &[question_tokens as f64], model.sliding_window)
    } else if first {
        scan(&mean, &question, model.sliding_window)
    } else {
        ask(&mean, &question, model.sliding_window)
    }
}
