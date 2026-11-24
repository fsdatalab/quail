use crate::cost::answers::{prior_survival, survival_rows};

pub const SOURCE_PRIOR_WEIGHT: i64 = 8;
pub const LIVE_BUCKETS: [i64; 11] = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

pub fn live_bucket(num_streams: i64) -> usize {
    let point = (num_streams as f64).ln_1p();
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (index, bucket) in LIVE_BUCKETS.iter().enumerate() {
        let distance = (point - (*bucket as f64).ln_1p()).abs();
        if distance < best_distance {
            best = index;
            best_distance = distance;
        }
    }
    best
}

pub fn source_odds(num_streams: i64, trials: &[i64], accepts: &[i64]) -> f64 {
    if num_streams <= 1 {
        return 1.0;
    }
    let bucket = live_bucket(num_streams);
    (SOURCE_PRIOR_WEIGHT as f64 / LIVE_BUCKETS[bucket] as f64 + accepts[bucket] as f64)
        / (SOURCE_PRIOR_WEIGHT + trials[bucket]) as f64
}

pub fn draft_gains(
    cap: usize,
    limit: usize,
    survival: &[f64],
    written: &[i64],
    num_streams: &[i64],
    trials: &[i64],
    accepts: &[i64],
) -> Vec<Vec<f64>> {
    written
        .iter()
        .zip(num_streams)
        .map(|(written, streams)| {
            let at = |j: usize| survival[(*written as usize + j).min(limit)];
            let base = at(0);
            let odds = source_odds(*streams, trials, accepts);
            (1..=cap)
                .map(|j| {
                    if *written as usize + j >= limit {
                        0.0
                    } else if base > 0.0 {
                        odds * at(j) / base
                    } else {
                        odds
                    }
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn draft_lengths(
    cap: usize,
    limit: usize,
    survival: &[f64],
    written: &[i64],
    num_streams: &[i64],
    rho: &[f64],
    trials: &[i64],
    accepts: &[i64],
) -> Vec<usize> {
    let gains = draft_gains(cap, limit, survival, written, num_streams, trials, accepts);
    written
        .iter()
        .zip(rho)
        .zip(&gains)
        .map(|((written, rho), gains)| {
            let mut writes = 1.0;
            let mut best = 0;
            let mut best_cost = f64::INFINITY;
            for j in 0..=cap {
                if j > 0 {
                    writes += gains[j - 1];
                }
                let cost = if *written as usize + j >= limit {
                    f64::INFINITY
                } else {
                    ((1 + j) as f64 + rho) / writes
                };
                if cost < best_cost {
                    best = j;
                    best_cost = cost;
                }
            }
            best
        })
        .collect()
}

pub fn fit_drafts(gains: &[Vec<f64>], drafts: &[i64], room: i64) -> Vec<i64> {
    if drafts.iter().sum::<i64>() <= room {
        return drafts.to_vec();
    }
    if room <= 0 {
        return vec![0; drafts.len()];
    }
    let kept: Vec<Vec<f64>> = gains
        .iter()
        .zip(drafts)
        .map(|(row, draft)| {
            row.iter()
                .enumerate()
                .map(|(j, gain)| {
                    if (j as i64) < *draft {
                        *gain
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect();
    let mut sorted: Vec<f64> = kept.iter().flatten().copied().collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold = sorted[room as usize - 1];
    let above: Vec<i64> = kept
        .iter()
        .map(|row| row.iter().filter(|gain| **gain > threshold).count() as i64)
        .collect();
    let ties: Vec<i64> = kept
        .iter()
        .map(|row| row.iter().filter(|gain| **gain == threshold).count() as i64)
        .collect();
    let mut left = room - above.iter().sum::<i64>();
    above
        .iter()
        .zip(&ties)
        .map(|(above, ties)| {
            let taken = left.clamp(0, *ties);
            left -= ties;
            above + taken
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct DraftPolicy {
    pub cap: usize,
    pub max_answer_tokens: usize,
    pub prior: Vec<f64>,
    pub lengths: Vec<i64>,
    pub trials: Vec<i64>,
    pub accepts: Vec<i64>,
}

impl DraftPolicy {
    pub fn new(cap: usize, max_answer_tokens: usize, expected_answer_tokens: f64) -> Self {
        Self {
            cap,
            max_answer_tokens,
            prior: prior_survival(expected_answer_tokens, max_answer_tokens),
            lengths: vec![0; max_answer_tokens + 1],
            trials: vec![0; LIVE_BUCKETS.len()],
            accepts: vec![0; LIVE_BUCKETS.len()],
        }
    }

    pub fn curve(&self) -> Vec<f64> {
        survival_rows(
            std::slice::from_ref(&self.lengths),
            std::slice::from_ref(&self.lengths),
            std::slice::from_ref(&self.prior),
            &[self.max_answer_tokens],
        )
        .pop()
        .expect("one row")
    }
}
