//! Pass groups of a set-attention chain.

use crate::quantity::Quantity;
use crate::sol::unrounded_seconds;
use crate::specs::{DeviceSpec, ModelSpec};
use crate::work::packed;

/// Split an ordered chain of questions into consecutive pass groups,
/// the cheapest split by dynamic programming.
///
/// Returns each question's pass group, numbered from 0 in chain order.
///
/// # Errors
///
/// A question with the longest state does not fit one chunk.
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
            let mut group = vec![question_tokens[a - 1]];
            group.extend(question_tokens[a..b].iter().map(|tokens| tokens + header));
            if group.len() > set_size || longest_state + group.iter().sum::<i64>() > chunk_tokens {
                break;
            }
            let questions: Vec<f64> = group.iter().map(|tokens| *tokens as f64).collect();
            let work = &packed(&state, &questions, model.sliding_window) * reach[a - 1];
            let cost = best[a - 1] + unrounded_seconds(&work, model, device, chunk_tokens, 0.0, 0);
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
