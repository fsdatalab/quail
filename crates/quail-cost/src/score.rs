//! Work and time of an `AI.SCORE` call, as `quail/cost/score.py`
//! estimates them.

use crate::budgets::parallel_copies;
use crate::quantity::{Quantity, python_sum};
use crate::sol::speed_of_light;
use crate::specs::{DeviceSpec, ModelSpec};
use crate::statistics::AliasStats;
use crate::work::{Work, packed, triangle};

/// The tokens a score adds to each input besides its documents.
pub fn score_fixed_tokens(token_lengths: &[i64], canvas_tokens: i64, draws: i64) -> i64 {
    token_lengths.iter().sum::<i64>() + canvas_tokens + (draws - 1) * (1 + canvas_tokens)
}

/// Python's `max(0.0, value)`.
fn at_least_zero(value: f64) -> f64 {
    if value > 0.0 { value } else { 0.0 }
}

/// Python's `min(a, b)`: `b` only when it is smaller.
pub fn python_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

fn tri(tokens: f64) -> f64 {
    triangle(&Quantity::Scalar(tokens), 0).total()
}

/// The work of scoring `count` inputs, with a reused prefix.
#[allow(clippy::too_many_arguments)]
pub fn score_work(
    count: f64,
    mean_tokens: f64,
    fixed_tokens: i64,
    variance: f64,
    prefix_tokens: f64,
    prefix_variance: f64,
    groups: f64,
    shared_tokens: i64,
    copies: i64,
) -> Work {
    if count <= 0.0 {
        return Work::default();
    }
    let scalar = Quantity::Scalar;
    let length = mean_tokens + fixed_tokens as f64;
    let work = &Work {
        tokens: scalar(length),
        pairs: scalar(tri(length) + variance / 2.0),
        kv_written: scalar(length),
        state_written: scalar(1.0),
        ..Work::default()
    } * count;
    let reused = at_least_zero(count - groups);
    let work = &work
        + &(&Work {
            tokens: scalar(-prefix_tokens),
            pairs: scalar(-tri(prefix_tokens) - prefix_variance / 2.0),
            kv_written: scalar(-prefix_tokens),
            kv_read: scalar(prefix_tokens),
            state_written: scalar(-1.0),
            state_read: scalar(1.0),
            ..Work::default()
        } * reused);
    let shared = shared_tokens as f64;
    let shared_reuses = at_least_zero(groups - copies as f64);
    &work
        + &(&Work {
            tokens: scalar(-shared),
            pairs: scalar(-tri(shared)),
            kv_written: scalar(-shared),
            kv_read: scalar(shared),
            state_written: scalar(-1.0),
            state_read: scalar(1.0),
            ..Work::default()
        } * shared_reuses)
}

/// Prepared inputs for one score's cost at any survivor count.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreCost {
    pub token_lengths: Vec<i64>,
    pub summaries: Vec<AliasStats>,
    pub mean_tokens: f64,
    pub model: ModelSpec,
    pub device: DeviceSpec,
    pub chunk_tokens: i64,
    pub capacity: i64,
    pub gpus: i64,
    pub draws: i64,
}

impl ScoreCost {
    /// Model work and seconds for the expected inputs.
    pub fn estimate(&self, expected_rows: f64, prefix_groups: Option<f64>) -> (Work, f64) {
        let fixed = score_fixed_tokens(&self.token_lengths, self.model.canvas_tokens, self.draws);
        let shared = self.token_lengths[0];
        let mut prefix = shared as f64;
        let mut prefix_variance = 0.0;
        let mut groups = python_min(self.gpus as f64, expected_rows);
        let first = &self.summaries[0];
        if self.summaries.len() == 2 {
            prefix += first.mean() + self.token_lengths[1] as f64;
            prefix_variance = first.variance();
            groups = python_min(expected_rows, prefix_groups.unwrap_or(first.count as f64));
            let longest = self
                .summaries
                .iter()
                .map(|summary| summary.maximum)
                .sum::<i64>()
                + fixed;
            if longest > self.capacity {
                prefix = shared as f64;
                prefix_variance = 0.0;
                groups = python_min(self.gpus as f64, expected_rows);
            }
        }
        let variance = python_sum(self.summaries.iter().map(AliasStats::variance));
        let copies = parallel_copies(self.gpus, python_min(expected_rows, first.count as f64));
        let work = if self.model.attention_mask == "set" {
            let question = *self.token_lengths.last().expect("a question");
            let state = self.mean_tokens + fixed as f64 - question as f64;
            let work = &packed(
                &Quantity::Scalar(state),
                &[question as f64],
                self.model.sliding_window,
            ) * expected_rows;
            &work
                + &Work {
                    pairs: Quantity::Scalar(variance * expected_rows),
                    ..Work::default()
                }
        } else {
            score_work(
                expected_rows,
                self.mean_tokens,
                fixed,
                variance,
                prefix,
                prefix_variance,
                groups,
                shared,
                self.gpus,
            )
        };
        let seconds = speed_of_light(
            &(&work * (1.0 / copies)),
            &self.model,
            &self.device,
            self.chunk_tokens,
            0.0,
            0,
        )
        .seconds();
        (work, seconds)
    }
}
