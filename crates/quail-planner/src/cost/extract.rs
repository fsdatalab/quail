//! Extraction costs.
//!
//! Each document's work is a record of scalars; the table's work sums
//! them as numpy sums an array.

use std::collections::BTreeMap;

use crate::cost::budgets::parallel_copies;
use crate::cost::draft_policy::{DraftPolicy, draft_lengths};
use crate::cost::quantity::{Quantity, numpy_sum};
use crate::cost::sol::unrounded_seconds;
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::work::{Work, ask, scan, triangle};

/// A larger table is priced from this many evenly spaced documents.
pub const SAMPLE_DOCUMENTS: usize = 1024;

fn scalar(value: f64) -> Quantity {
    Quantity::Scalar(value)
}

fn value(quantity: &Quantity) -> f64 {
    quantity.scalar().expect("one document's count")
}

fn tri(tokens: f64, window: i64) -> f64 {
    value(&triangle(&scalar(tokens), window))
}

/// The keep chance of `ExtractAssumptions`: 1 - 1/max(1, answer_tokens).
fn keep(answer_tokens: i64) -> f64 {
    1.0 - 1.0 / (answer_tokens as f64).max(1.0)
}

/// `ExtractAssumptions.survival`: the chance an answer has more than
/// `tokens` tokens after its first.
pub fn survival(answer_tokens: i64, tokens: f64) -> f64 {
    if answer_tokens == 0 {
        0.0
    } else {
        keep(answer_tokens).powf(tokens)
    }
}

/// Rounds an answer may take at the limit: ceil((limit - 1) / per_round), at least 0.
fn most_rounds(max_answer_tokens: i64, per_round: i64) -> i64 {
    0.max(-(-(max_answer_tokens - 1)).div_euclid(per_round))
}

/// `ExtractAssumptions.rounds`: the expected rounds an answer takes.
pub fn expected_rounds(answer_tokens: i64, max_answer_tokens: i64, per_round: i64) -> f64 {
    let most = most_rounds(max_answer_tokens, per_round);
    let keep = keep(answer_tokens);
    if answer_tokens == 0 {
        0.0
    } else if keep == 0.0 {
        if most > 0 { 1.0 } else { 0.0 }
    } else {
        let log_q = per_round as f64 * keep.ln();
        (most as f64 * log_q).exp_m1() / log_q.exp_m1()
    }
}

/// Each round length's chance to run each round, and sums over rounds.
struct RoundChances {
    num_rounds: i64,
    /// round tokens -> chance of round r, r = 0..=num_rounds.
    chance: BTreeMap<i64, Vec<f64>>,
    /// round tokens -> sum of chance over rounds below k, k = 0..=num_rounds + 1.
    sum_chance: BTreeMap<i64, Vec<f64>>,
    /// round tokens -> sum of chance times r over rounds below k.
    sum_chance_round: BTreeMap<i64, Vec<f64>>,
}

impl RoundChances {
    fn new(answer_tokens: i64, max_answer_tokens: i64, round_tokens: &[i64]) -> Self {
        let mut lengths: Vec<i64> = round_tokens.to_vec();
        lengths.sort_unstable();
        lengths.dedup();
        let most: Vec<i64> = lengths
            .iter()
            .map(|length| {
                if answer_tokens == 0 {
                    0
                } else {
                    most_rounds(max_answer_tokens, *length)
                }
            })
            .collect();
        let num_rounds = most.iter().copied().max().unwrap_or(0);
        let keep = keep(answer_tokens);
        let (mut chance, mut sum_chance, mut sum_chance_round) =
            (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        for (length, most) in lengths.iter().zip(&most) {
            let row: Vec<f64> = (0..=num_rounds)
                .map(|r| {
                    if r < *most {
                        keep.powf((length * r) as f64)
                    } else {
                        0.0
                    }
                })
                .collect();
            let (mut sums, mut round_sums) = (vec![0.0], vec![0.0]);
            let (mut running, mut running_round) = (0.0, 0.0);
            for (r, chance) in row.iter().enumerate() {
                running += chance;
                running_round += chance * r as f64;
                sums.push(running);
                round_sums.push(running_round);
            }
            chance.insert(*length, row);
            sum_chance.insert(*length, sums);
            sum_chance_round.insert(*length, round_sums);
        }
        Self {
            num_rounds,
            chance,
            sum_chance,
            sum_chance_round,
        }
    }

    fn first(&self, round_tokens: i64) -> f64 {
        self.chance[&round_tokens][0]
    }

    /// Sums of chance and of chance times j over rounds offset + j, j < count.
    fn sums(&self, round_tokens: i64, offset: i64, count: i64) -> (f64, f64) {
        let (sums, round_sums) = (
            &self.sum_chance[&round_tokens],
            &self.sum_chance_round[&round_tokens],
        );
        let (stop, offset_index) = ((offset + count) as usize, offset as usize);
        let chance = sums[stop] - sums[offset_index];
        let chance_round = round_sums[stop] - round_sums[offset_index];
        (chance, chance_round - offset as f64 * chance)
    }

    /// The expected sum of ask(context + j step, multiple step) over rounds offset + j.
    fn asks(
        &self,
        round_tokens: i64,
        context: f64,
        multiple: i64,
        offset: i64,
        window: i64,
    ) -> Work {
        let step = round_tokens as f64;
        let suffix = multiple as f64 * step;
        let count = (self.num_rounds - offset).max(0);
        let (chance, chance_round) = self.sums(round_tokens, offset, count);
        let kv_read = context * chance + step * chance_round;
        let (mut sliding_pairs, mut sliding_kv_read) = (0.0, 0.0);
        if window != 0 {
            let w = window as f64;
            let clip = |x: f64| x.clamp(0.0, count as f64) as i64;
            let below = clip(((w - context - suffix) / step).floor() + 1.0);
            let above = clip(((w - context) / step).ceil());
            let (below_chance, below_round) = self.sums(round_tokens, offset, below);
            let (above_chance, _) = self.sums(round_tokens, offset, above);
            sliding_pairs = suffix * (context * below_chance + step * below_round)
                + tri(suffix, 0) * below_chance
                + w * suffix * (chance - above_chance);
            for i in 0..multiple {
                let j = below + i;
                let start = context + j as f64 * step;
                let crossing = if j < above {
                    self.chance[&round_tokens][(offset + j).min(self.num_rounds) as usize]
                } else {
                    0.0
                };
                sliding_pairs += crossing * (tri(start + suffix, window) - tri(start, window));
            }
            let reach = clip(((w - 1.0 - context) / step).ceil());
            let (reach_chance, reach_round) = self.sums(round_tokens, offset, reach);
            sliding_kv_read =
                context * reach_chance + step * reach_round + (w - 1.0) * (chance - reach_chance);
        }
        Work {
            tokens: scalar(suffix * chance),
            pairs: scalar(suffix * kv_read + tri(suffix, 0) * chance),
            kv_written: scalar(suffix * chance),
            kv_read: scalar(kv_read),
            sliding_pairs: scalar(sliding_pairs),
            sliding_kv_read: scalar(sliding_kv_read),
            state_read: scalar(chance),
            ..Work::default()
        }
    }
}

/// The extraction settings every document shares.
#[derive(Debug, Clone)]
pub struct ExtractInputs {
    pub head_tokens: i64,
    pub tail_tokens: i64,
    pub window: i64,
    pub resident: bool,
    pub max_answer_tokens: i64,
    pub draft_tokens: i64,
    /// "fixed" or "adaptive".
    pub draft_policy: String,
    pub round_cost_per_token: f64,
    /// The assumed mean answer length.
    pub answer_tokens: i64,
    pub recurrent: bool,
}

/// Each document's work, fed tokens, and readout rows.
///
/// `shared` holds one borrowed prefix length per document, or is empty
/// when no document borrows one.
///
/// # Errors
///
/// The draft policy is neither "fixed" nor "adaptive".
pub fn document_work(
    lengths: &[f64],
    shared: &[f64],
    inputs: &ExtractInputs,
) -> Result<Vec<(Work, f64, f64)>, String> {
    let (head, tail, window) = (
        inputs.head_tokens as f64,
        inputs.tail_tokens as f64,
        inputs.window,
    );
    let contexts: Vec<f64> = lengths.iter().map(|length| head + length + tail).collect();
    let drafts: Vec<i64> = match inputs.draft_policy.as_str() {
        "adaptive" => {
            let policy = DraftPolicy::new(
                inputs.draft_tokens as usize,
                inputs.max_answer_tokens as usize,
                inputs.answer_tokens as f64,
            );
            let curve = policy.curve();
            let ones = vec![1_i64; contexts.len()];
            let rho: Vec<f64> = contexts
                .iter()
                .map(|context| inputs.round_cost_per_token * context)
                .collect();
            draft_lengths(
                policy.cap,
                policy.max_answer_tokens,
                &curve,
                &ones,
                &ones,
                &rho,
                &policy.trials,
                &policy.accepts,
            )
            .into_iter()
            .map(|draft| draft as i64)
            .collect()
        }
        "fixed" => vec![inputs.draft_tokens; contexts.len()],
        other => return Err(format!("unknown draft policy {other:?}")),
    };
    let round_tokens: Vec<i64> = drafts.iter().map(|draft| 1 + draft).collect();
    let chances = RoundChances::new(
        inputs.answer_tokens,
        inputs.max_answer_tokens,
        &round_tokens,
    );
    Ok(lengths
        .iter()
        .enumerate()
        .map(|(index, length)| {
            let prefix = head + length;
            let context = contexts[index];
            let borrowed = if inputs.resident {
                0.0
            } else {
                shared.get(index).copied().unwrap_or(0.0)
            };
            let mut work = if inputs.resident {
                ask(&scalar(prefix), &scalar(tail), window)
            } else if borrowed > 0.0 {
                ask(
                    &scalar(borrowed + head),
                    &scalar(context - borrowed - head),
                    window,
                )
            } else {
                scan(&scalar(prefix), &scalar(tail), window)
            };
            if inputs.recurrent {
                work.state_read = scalar(1.0 + if borrowed > 0.0 { 1.0 } else { 0.0 });
                work.state_written = scalar(1.0 + if inputs.resident { 0.0 } else { 1.0 });
            }
            let step = round_tokens[index];
            let rounds = expected_rounds(inputs.answer_tokens, inputs.max_answer_tokens, step);
            let mut fed = tail + rounds * step as f64;
            if inputs.recurrent {
                let mut first = ask(&scalar(context), &scalar(step as f64), window);
                first.state_read = scalar(1.0);
                first.state_written = scalar(0.0);
                work = &work + &(&first * chances.first(step));
                let mut replays = chances.asks(step, context, 2, 1, window);
                let read = value(&replays.state_read);
                replays.state_read = scalar(2.0 * read);
                replays.state_written = scalar(read);
                work = &work + &replays;
                fed += step as f64 * read;
            } else {
                let mut steps = chances.asks(step, context, 1, 0, window);
                steps.state_written = steps.state_read.clone();
                work = &work + &steps;
            }
            (work, fed, 1.0 + rounds * step as f64)
        })
        .collect())
}

fn column(records: &[(Work, f64, f64)], field: impl Fn(&Work) -> &Quantity) -> f64 {
    numpy_sum(
        &records
            .iter()
            .map(|(work, _, _)| value(field(work)))
            .collect::<Vec<_>>(),
    )
}

/// The work, fed tokens, and readout rows of every document.
///
/// A table of more than `SAMPLE_DOCUMENTS` documents sums evenly
/// spaced documents and scales the sum to the table.
///
/// # Errors
///
/// The draft policy is unknown.
pub fn table_work(
    lengths: &[f64],
    shared: &[f64],
    inputs: &ExtractInputs,
) -> Result<(Work, f64, f64), String> {
    let n = lengths.len();
    let indices: Vec<usize> = if n <= SAMPLE_DOCUMENTS {
        (0..n).collect()
    } else {
        (0..SAMPLE_DOCUMENTS)
            .map(|i| i * (n - 1) / (SAMPLE_DOCUMENTS - 1))
            .collect()
    };
    let sampled: Vec<f64> = indices.iter().map(|i| lengths[*i]).collect();
    let sampled_shared: Vec<f64> = if shared.is_empty() {
        Vec::new()
    } else {
        indices.iter().map(|i| shared[*i]).collect()
    };
    let records = document_work(&sampled, &sampled_shared, inputs)?;
    let mut work = Work {
        tokens: scalar(column(&records, |w| &w.tokens)),
        pairs: scalar(column(&records, |w| &w.pairs)),
        kv_written: scalar(column(&records, |w| &w.kv_written)),
        kv_read: scalar(column(&records, |w| &w.kv_read)),
        sliding_pairs: scalar(column(&records, |w| &w.sliding_pairs)),
        sliding_kv_read: scalar(column(&records, |w| &w.sliding_kv_read)),
        state_written: scalar(column(&records, |w| &w.state_written)),
        state_read: scalar(column(&records, |w| &w.state_read)),
        sae_encoded: scalar(column(&records, |w| &w.sae_encoded)),
    };
    let mut fed = numpy_sum(&records.iter().map(|r| r.1).collect::<Vec<_>>());
    let mut rows = numpy_sum(&records.iter().map(|r| r.2).collect::<Vec<_>>());
    if n > SAMPLE_DOCUMENTS {
        let scale = n as f64 / SAMPLE_DOCUMENTS as f64;
        work = &work * scale;
        fed *= scale;
        rows *= scale;
    }
    Ok((work, fed, rows))
}

/// Estimated extraction time, work, and token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractCost {
    pub seconds: f64,
    pub passes: f64,
    pub work: Work,
    pub suffix_tokens: f64,
    pub rows: f64,
}

/// The cost of one extraction over a table.
///
/// The window and recurrence in `inputs` come from the model.
///
/// # Errors
///
/// The draft policy is unknown.
#[allow(clippy::too_many_arguments)]
pub fn estimate(
    expected_rows: f64,
    lengths: &[f64],
    shared: &[f64],
    inputs: &ExtractInputs,
    chunk: i64,
    model: &ModelSpec,
    device: &DeviceSpec,
    gpus: i64,
) -> Result<ExtractCost, String> {
    let n = lengths.len();
    if expected_rows <= 0.0 || n == 0 {
        return Ok(ExtractCost {
            seconds: 0.0,
            passes: 0.0,
            work: Work::default(),
            suffix_tokens: 0.0,
            rows: 0.0,
        });
    }
    let inputs = ExtractInputs {
        window: model.sliding_window,
        recurrent: !model.state_layers().is_empty(),
        ..inputs.clone()
    };
    let (work, fed, rows) = table_work(lengths, shared, &inputs)?;
    let scale = expected_rows / n as f64;
    let total = &work * scale;
    let copies = parallel_copies(gpus, expected_rows.min(n as f64));
    let seconds = unrounded_seconds(
        &(&total * (1.0 / copies)),
        model,
        device,
        chunk,
        rows * scale / copies,
        0,
    );
    let passes = value(&total.tokens) / chunk as f64;
    Ok(ExtractCost {
        seconds,
        passes,
        work: total,
        suffix_tokens: fed * scale,
        rows: rows * scale,
    })
}
