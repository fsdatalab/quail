//! The draft length of each extraction round, as
//! `quail/common/draft_policy.py` chooses it.

use crate::answers::{prior_survival, survival_rows};

/// Drafted rounds the seed of the source odds counts as.
pub const SOURCE_PRIOR_WEIGHT: i64 = 8;
/// Ranges of live positions the source odds are learned per.
pub const LIVE_BUCKETS: [i64; 11] = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

/// The range of live positions a count falls in, nearest on a log scale.
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

/// P(the position followed is the answer's source) for one round.
pub fn source_odds(num_streams: i64, trials: &[i64], accepts: &[i64]) -> f64 {
    if num_streams <= 1 {
        return 1.0;
    }
    let bucket = live_bucket(num_streams);
    (SOURCE_PRIOR_WEIGHT as f64 / LIVE_BUCKETS[bucket] as f64 + accepts[bucket] as f64)
        / (SOURCE_PRIOR_WEIGHT + trials[bucket]) as f64
}

/// Each answer's expected written tokens from each draft token, one row
/// of `cap` per answer.
///
/// Column j - 1 is draft token j's: the chance the answer goes on past
/// j more tokens times the chance the position followed is its source.
/// It never rises with j. Draft tokens at or past the limit write none.
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

/// Each answer's draft length: the least cost per written token.
///
/// `survival` is the field's S(t), t = 0..=limit; `written`,
/// `num_streams`, and `rho` hold one value per answer.
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
            // the chosen token, then each draft token while the answer goes
            // on and the position followed is its source
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

/// The draft lengths that keep the most expected writes in `room` rows.
///
/// A pass holds at most `room` draft rows. Since an answer's gains never
/// rise along its draft, keeping the `room` largest gains keeps a prefix
/// of every draft; ties go to the earlier answers.
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

/// Chooses each round's draft length for one question.
#[derive(Debug, Clone)]
pub struct DraftPolicy {
    pub cap: usize,
    pub max_answer_tokens: usize,
    pub prior: Vec<f64>,
    /// Settled answers by length.
    pub lengths: Vec<i64>,
    /// Drafted rounds per range of live positions.
    pub trials: Vec<i64>,
    /// Of those, the rounds whose first draft token was written.
    pub accepts: Vec<i64>,
}

impl DraftPolicy {
    /// A policy with no rounds seen, starting from the planner's guess
    /// at the mean answer length.
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

    /// The answer survival curve of the settled answers.
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
