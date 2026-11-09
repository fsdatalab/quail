//! Pass groups of a set-attention chain.

use crate::cost::quantity::Quantity;
use crate::cost::sol::unrounded_seconds;
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::work::packed;

/// Split an ordered chain of questions into consecutive pass groups,
/// the cheapest split by dynamic programming.
///
/// Returns each question's pass group, numbered from 0 in chain order.
///
/// # Errors
///
/// Inputs have inconsistent lengths, invalid counts or selectivities,
/// a cost is not finite, or a question with the longest state does not
/// fit one chunk.
#[allow(clippy::too_many_arguments)]
pub fn choose_pass_groups(
    question_tokens: &[i64],
    selectivities: &[f64],
    rows: f64,
    state_tokens: f64,
    longest_state: i64,
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    set_size: usize,
    header: i64,
) -> Result<Vec<usize>, String> {
    if question_tokens.len() != selectivities.len() {
        return Err("one selectivity is required per question".into());
    }
    for (name, value) in [("rows", rows), ("state_tokens", state_tokens)] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and nonnegative"));
        }
    }
    if selectivities
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("selectivities must be finite and between 0 and 1".into());
    }
    if chunk_tokens <= 0 || set_size == 0 {
        return Err("chunk_tokens and set_size must be positive".into());
    }
    if longest_state < 0 || header < 0 || question_tokens.iter().any(|tokens| *tokens < 0) {
        return Err("question_tokens, longest_state and header must be nonnegative".into());
    }
    let m = question_tokens.len();
    let mut reach = vec![rows];
    for selectivity in selectivities {
        let last = *reach.last().expect("one reach");
        reach.push(last * selectivity);
    }
    let mut best = vec![f64::INFINITY; m + 1];
    best[0] = 0.0;
    let mut start = vec![0_usize; m + 1];
    let state = Quantity::Scalar(state_tokens);
    for b in 1..=m {
        for a in (1..=b).rev() {
            // Compare in a wider integer type before adding headers: even
            // individually valid i64 token counts can overflow a chunk sum.
            let tokens = i128::from(longest_state)
                + question_tokens[a - 1..b]
                    .iter()
                    .map(|tokens| i128::from(*tokens))
                    .sum::<i128>()
                + (b - a) as i128 * i128::from(header);
            if b - a + 1 > set_size || tokens > i128::from(chunk_tokens) {
                break;
            }
            let mut questions = vec![question_tokens[a - 1] as f64];
            questions.extend(
                question_tokens[a..b]
                    .iter()
                    .map(|tokens| (tokens + header) as f64),
            );
            let work = &packed(&state, &questions, model.sliding_window) * reach[a - 1];
            let cost = best[a - 1] + unrounded_seconds(&work, model, device, chunk_tokens, 0.0, 0);
            if !cost.is_finite() {
                return Err("pass-group cost must be finite".into());
            }
            if cost < best[b] {
                best[b] = cost;
                start[b] = a;
            }
        }
        if best[b] == f64::INFINITY {
            return Err(format!(
                "question {b} does not fit one {chunk_tokens}-token chunk with the longest document"
            ));
        }
    }
    let mut bounds = Vec::new();
    let mut b = m;
    while b != 0 {
        bounds.push((start[b], b));
        b = start[b] - 1;
    }
    let mut groups = vec![0_usize; m];
    for (index, (a, b)) in bounds.iter().rev().enumerate() {
        for group in groups.iter_mut().take(*b).skip(a - 1) {
            *group = index;
        }
    }
    Ok(groups)
}
