//! Classification costs.

use std::collections::BTreeSet;

use crate::cost::budgets::parallel_copies;
use crate::cost::labels::{
    DECODE_SCORING, LETTERS_SCORING, TREE_SCORING, decode_round_tokens, fed_nodes, read_nodes,
};
use crate::cost::quantity::Quantity;
use crate::cost::score::python_min;
use crate::cost::sol::unrounded_seconds;
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::work::{Work, ask, decode, stream};

/// Estimated classification time, work, and token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifyCost {
    pub seconds: f64,
    pub passes: f64,
    pub work: Work,
    pub suffix_tokens: f64,
    pub rounds: usize,
}

/// The request lengths of a scoring rule, the answer cue included.
///
/// # Errors
///
/// The scoring rule is unknown, or a label has no tokens.
pub fn suffix_lengths(
    scoring: &str,
    labels: &[Vec<i64>],
    canvas_rows: i64,
    draws: i64,
    starts: &[Vec<bool>],
) -> Result<Vec<i64>, String> {
    match scoring {
        LETTERS_SCORING => Ok(vec![
            1 + canvas_rows;
            if canvas_rows != 0 { draws as usize } else { 1 }
        ]),
        TREE_SCORING => Ok(vec![fed_nodes(&read_nodes(labels, starts)?).len() as i64]),
        DECODE_SCORING => {
            let fed = decode_round_tokens(labels)?;
            let rounds = fed.iter().map(Vec::len).max().unwrap_or(0);
            Ok((0..rounds)
                .map(|round| {
                    fed.iter()
                        .filter(|counts| counts.len() > round)
                        .map(|counts| counts[round])
                        .max()
                        .expect("a label reaches the round")
                })
                .collect())
        }
        other => Err(format!("unknown label scoring rule '{other}'")),
    }
}

fn filled(like: &Quantity, value: f64) -> Quantity {
    like.map(|_| value)
}

fn length(values: &Quantity) -> usize {
    match values {
        Quantity::Array(values) => values.len(),
        Quantity::Scalar(_) => 1,
    }
}

/// Every document's prefix-and-tail work: scan, or ask on resident KV.
fn first_work(
    prefix: &Quantity,
    shared: &Quantity,
    tail: i64,
    resident: bool,
    window: i64,
) -> Work {
    let documents = length(prefix) as f64;
    if resident {
        if tail == 0 {
            return Work::default();
        }
        let mut work = ask(prefix, &filled(prefix, tail as f64), window).total();
        work.state_written = Quantity::Scalar(documents);
        return work;
    }
    let fresh = &(prefix - shared) + tail as f64;
    let mut work = ask(shared, &fresh, window).total();
    let has_tail = if tail != 0 { 1.0 } else { 0.0 };
    let borrowed = match shared {
        Quantity::Array(values) => values.iter().filter(|value| **value > 0.0).count(),
        Quantity::Scalar(value) => usize::from(*value > 0.0),
    } as f64;
    work.state_written = Quantity::Scalar(documents * (1.0 + has_tail));
    work.state_read = Quantity::Scalar(borrowed + documents * has_tail);
    work
}

/// The second attention read of a multi-row answer canvas.
fn canvas_work(work: Work, canvas_rows: i64) -> Work {
    if canvas_rows > 1 {
        let reread = Work {
            pairs: work.pairs.clone(),
            kv_read: work.kv_read.clone(),
            sliding_pairs: work.sliding_pairs.clone(),
            sliding_kv_read: work.sliding_kv_read.clone(),
            ..Work::default()
        };
        return &work + &reread;
    }
    work
}

fn stream_work(context: &Quantity, suffixes: &[i64], window: i64, canvas_rows: i64) -> Work {
    let fed: Vec<Quantity> = suffixes
        .iter()
        .map(|suffix| filled(context, *suffix as f64))
        .collect();
    canvas_work(stream(context, &fed, window).total(), canvas_rows)
}

#[allow(clippy::too_many_arguments)]
fn rounds_work(
    prefix: &Quantity,
    shared: &Quantity,
    frame_tokens: i64,
    runs: &[i64],
    resident: bool,
    window: i64,
    canvas_rows: i64,
) -> Work {
    let first_tail = frame_tokens + runs[0];
    let work = first_work(prefix, shared, first_tail, resident, window);
    let context = prefix + first_tail as f64;
    let later: Vec<f64> = runs[1..].iter().map(|run| *run as f64).collect();
    let rounds = decode(&context, &later, window).total();
    &work + &canvas_work(rounds, canvas_rows)
}

/// The inputs of one classification estimate besides its suffixes.
#[derive(Debug, Clone)]
pub struct ClassifyInputs<'a> {
    pub head_tokens: i64,
    pub frame_tokens: i64,
    pub expected_rows: f64,
    pub lengths: &'a [f64],
    pub shared: &'a [f64],
    pub chunk: i64,
    pub model: &'a ModelSpec,
    pub device: &'a DeviceSpec,
    pub resident: bool,
    pub gpus: i64,
}

/// Expected classification work and its roofline time.
pub fn estimate_suffixes(
    inputs: &ClassifyInputs,
    suffixes: &[i64],
    canvas_rows: i64,
    runs: Option<&[Vec<i64>]>,
    answer_rows: Option<f64>,
    head_rows: i64,
) -> ClassifyCost {
    let n = inputs.lengths.len();
    let has_requests = !suffixes.is_empty() || runs.is_some_and(|runs| !runs.is_empty());
    if inputs.expected_rows <= 0.0 || n == 0 || !has_requests {
        return ClassifyCost {
            seconds: 0.0,
            passes: 0.0,
            work: Work::default(),
            suffix_tokens: 0.0,
            rounds: 0,
        };
    }
    let window = inputs.model.sliding_window;
    let head = inputs.head_tokens as f64;
    let prefix = Quantity::Array(inputs.lengths.iter().map(|length| length + head).collect());
    let shared = Quantity::Array(if inputs.shared.is_empty() {
        vec![0.0; n]
    } else {
        inputs
            .shared
            .iter()
            .map(|tokens| if *tokens > 0.0 { tokens + head } else { 0.0 })
            .collect()
    });
    let (work, suffix_tokens, mut rows, rounds) = match runs {
        Some(runs) => {
            let mut counted: Vec<(&Vec<i64>, usize)> = Vec::new();
            for fed in runs {
                match counted.iter_mut().find(|(seen, _)| *seen == fed) {
                    Some((_, count)) => *count += 1,
                    None => counted.push((fed, 1)),
                }
            }
            let mut work = Work::default();
            let (mut suffix_tokens, mut rows) = (0.0, 0.0);
            for (fed, count) in counted {
                let share = count as f64 / runs.len() as f64;
                let label = rounds_work(
                    &prefix,
                    &shared,
                    inputs.frame_tokens,
                    fed,
                    inputs.resident,
                    window,
                    canvas_rows,
                );
                work = &work + &(&label * share);
                suffix_tokens += fed.iter().sum::<i64>() as f64 * share;
                rows += fed.len() as f64 * share;
            }
            let rounds = runs.iter().map(Vec::len).max().unwrap_or(0);
            (work, suffix_tokens, rows, rounds)
        }
        None => {
            let work = &first_work(
                &prefix,
                &shared,
                inputs.frame_tokens,
                inputs.resident,
                window,
            ) + &stream_work(
                &(&prefix + inputs.frame_tokens as f64),
                suffixes,
                window,
                canvas_rows,
            );
            let fed: i64 = suffixes.iter().sum();
            let rows = if canvas_rows != 0 { canvas_rows } else { fed };
            (work, fed as f64, rows as f64, 1)
        }
    };
    if let Some(answer_rows) = answer_rows {
        rows = answer_rows;
    }
    let expected_rows = inputs.expected_rows;
    let total = (&work * (expected_rows / n as f64)).with_readout(
        rows * expected_rows,
        head_rows,
        inputs.model.vocab,
    );
    let copies = parallel_copies(inputs.gpus, python_min(expected_rows, n as f64));
    let seconds = unrounded_seconds(
        &(&total * (1.0 / copies)),
        inputs.model,
        inputs.device,
        inputs.chunk,
    );
    let passes = total.tokens.total() / inputs.chunk as f64;
    ClassifyCost {
        seconds,
        passes,
        work: total,
        suffix_tokens: suffix_tokens * expected_rows,
        rounds,
    }
}

/// The cost of one classification scoring rule.
///
/// # Errors
///
/// The scoring rule is unknown, or a label has no tokens.
pub fn estimate(
    inputs: &ClassifyInputs,
    scoring: &str,
    labels: &[Vec<i64>],
    draws: i64,
    starts: &[Vec<bool>],
) -> Result<ClassifyCost, String> {
    let model = inputs.model;
    let canvas_rows = model
        .answer_canvas
        .as_ref()
        .map(|canvas| canvas.rows)
        .unwrap_or(0);
    let suffixes = suffix_lengths(scoring, labels, canvas_rows, draws, starts)?;
    let mut runs = None;
    let mut answer_rows = None;
    if scoring == DECODE_SCORING {
        runs = Some(decode_round_tokens(labels)?);
    } else if scoring == TREE_SCORING {
        answer_rows = Some(read_nodes(labels, starts)?.len() as f64);
    }
    let targets: BTreeSet<i64> = labels.iter().flatten().copied().collect();
    let head_rows = if labels.iter().all(|ids| ids.len() == 1) {
        targets.len() as i64
    } else {
        0
    };
    Ok(estimate_suffixes(
        inputs,
        &suffixes,
        if scoring == LETTERS_SCORING {
            canvas_rows
        } else {
            model.canvas_tokens
        },
        runs.as_deref(),
        answer_rows,
        head_rows,
    ))
}
