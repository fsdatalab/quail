//! How long written answers run, as `quail/common/answers.py` estimates
//! it: survival curves over answer tokens.

/// Answers the prior counts as, until ended answers outweigh it.
pub const PRIOR_WEIGHT: f64 = 20.0;

/// S(t) for t = 0..=limit under geometric lengths with the given mean.
pub fn prior_survival(expected_tokens: f64, limit: usize) -> Vec<f64> {
    let mean = if expected_tokens > 1.0 {
        expected_tokens
    } else {
        1.0
    };
    let keep = 1.0 - 1.0 / mean;
    let mut curve: Vec<f64> = (0..=limit).map(|t| keep.powf(t as f64)).collect();
    curve[limit] = 0.0;
    curve
}

/// Kaplan-Meier curves, one per row, starting from each row's prior.
///
/// `counts` holds the answers observed last at each token, ended or
/// censored; `events` the answers that ended at each token; `prior`
/// each row's prior curve; `limits` each row's answer token limit.
pub fn survival_rows(
    counts: &[Vec<i64>],
    events: &[Vec<i64>],
    prior: &[Vec<f64>],
    limits: &[usize],
) -> Vec<Vec<f64>> {
    counts
        .iter()
        .zip(events)
        .zip(prior)
        .zip(limits)
        .map(|(((counts, events), prior), limit)| {
            let width = counts.len();
            // risk[j - 1] for j = 1..width: answers observed at j or later
            let mut at_or_after = vec![0_i64; width];
            let mut running = 0_i64;
            for j in (0..width).rev() {
                running += counts[j];
                at_or_after[j] = running;
            }
            let mut survival = vec![0.0; width];
            survival[0] = 1.0;
            let mut product = 1.0;
            for j in 1..width {
                let risk = at_or_after[j] as f64 + PRIOR_WEIGHT * prior[j - 1];
                let ended = events[j] as f64 + PRIOR_WEIGHT * (prior[j - 1] - prior[j]);
                let hazard = if risk > 0.0 { ended / risk } else { 0.0 };
                product *= 1.0 - hazard;
                survival[j] = product;
            }
            for value in survival.iter_mut().skip(*limit) {
                *value = 0.0;
            }
            survival
        })
        .collect()
}

/// S(t), the chance an answer runs past t tokens, t = 0..=limit.
///
/// `lengths` holds each answer's length, -1 while it is being written;
/// `written` the tokens each answer has written.
pub fn answer_survival(
    lengths: &[i64],
    written: &[i64],
    expected_tokens: f64,
    limit: usize,
    prior: Option<Vec<f64>>,
) -> Vec<f64> {
    let mut counts = vec![0_i64; limit + 1];
    let mut events = vec![0_i64; limit + 1];
    for (length, so_far) in lengths.iter().zip(written) {
        let observed = if *length >= 0 { *length } else { *so_far };
        if (observed as usize) <= limit {
            counts[observed as usize] += 1;
        }
        if *length >= 0 && (*length as usize) <= limit {
            events[*length as usize] += 1;
        }
    }
    let prior = prior.unwrap_or_else(|| prior_survival(expected_tokens, limit));
    survival_rows(&[counts], &[events], &[prior], &[limit])
        .pop()
        .expect("one row")
}

/// The expected tokens left after t tokens, t = 0..=limit.
pub fn remaining_tokens(survival: &[f64]) -> Vec<f64> {
    let width = survival.len();
    let mut tail = vec![0.0; width];
    let mut running = 0.0;
    for j in (0..width).rev() {
        running += survival[j];
        tail[j] = running;
    }
    tail.iter()
        .zip(survival)
        .map(|(tail, alive)| if *alive > 0.0 { tail / alive } else { 0.0 })
        .collect()
}
