pub const PRIOR_WEIGHT: f64 = 20.0;

pub fn keep(expected_tokens: f64) -> f64 {
    let mean = if expected_tokens > 1.0 {
        expected_tokens
    } else {
        1.0
    };
    1.0 - 1.0 / mean
}

pub fn prior_survival(expected_tokens: f64, limit: usize) -> Vec<f64> {
    let keep = keep(expected_tokens);
    let mut curve: Vec<f64> = (0..=limit).map(|t| keep.powf(t as f64)).collect();
    curve[limit] = 0.0;
    curve
}

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
