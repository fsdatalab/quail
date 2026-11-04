//! Model work of a join at given row counts and KV residency.

use std::collections::BTreeMap;

use crate::quantity::{Quantity, python_sum};
use crate::statistics::{AliasStats, int_true_divide};
use crate::work::{Work, triangle};

/// One join with the token counts the cost model prices it by.
#[derive(Debug, Clone, PartialEq)]
pub struct PricedJoin {
    /// The join's position among the query's joins.
    pub written_position: usize,
    /// The join's tables in placeholder order.
    pub aliases: Vec<String>,
    /// The table whose document KV the join reuses, or None when the
    /// search chooses it.
    pub anchor: Option<String>,
    /// Whether the search may choose the anchor.
    pub anchor_free: bool,
    /// "full" for a join returning pairs, else the gate's kind.
    pub semantics: String,
    /// The fraction of pairs expected to pass, or None.
    pub selectivity: Option<f64>,
    /// Equality conditions: (left alias, left column, right alias, right column).
    pub on: Vec<(String, String, String, String)>,
    /// The fraction of the cross product the equality conditions keep.
    pub pair_fraction: f64,
    /// A score join's own pricing; None for a join the model answers.
    pub score: Option<crate::score::ScoreCost>,
    /// alias -> tokens of the frame after that alias's document when it
    /// anchors.
    pub frame_tokens: BTreeMap<String, i64>,
    /// alias -> tokens of its label block.
    pub label_tokens: BTreeMap<String, i64>,
    /// Tokens of the question tail after every pair.
    pub tail_tokens: i64,
}

fn tri(tokens: f64) -> f64 {
    triangle(&Quantity::Scalar(tokens), 0).total()
}

fn tri_window(tokens: f64, window: i64) -> f64 {
    triangle(&Quantity::Scalar(tokens), window).total()
}

fn scalar(value: f64) -> Quantity {
    Quantity::Scalar(value)
}

/// The tokens of the longest input one join orientation makes.
///
/// `partners` holds each partner's label tokens and longest document.
pub fn longest_join_input(
    pre: i64,
    anchor_max: i64,
    frame: i64,
    tail: i64,
    partners: &[(i64, i64)],
) -> i64 {
    pre + anchor_max
        + frame
        + tail
        + partners
            .iter()
            .map(|(label, longest)| label + longest)
            .sum::<i64>()
}

/// Work of streaming every partner suffix past one resident anchor.
pub fn partner_stream_work(anchor_prefix: f64, suffixes: &[f64], window: i64) -> Work {
    let tokens = python_sum(suffixes.iter().copied());
    let sliding_pairs = if window == 0 {
        0.0
    } else if anchor_prefix >= (window - 1) as f64 {
        window as f64 * tokens
    } else {
        python_sum(suffixes.iter().map(|suffix| {
            tri_window(anchor_prefix + suffix, window) - tri_window(anchor_prefix, window)
        }))
    };
    let cap = (window - 1) as f64;
    Work {
        tokens: scalar(tokens),
        pairs: scalar(anchor_prefix * tokens + python_sum(suffixes.iter().map(|s| tri(*s)))),
        kv_written: scalar(tokens),
        kv_read: scalar(anchor_prefix),
        sliding_pairs: scalar(sliding_pairs),
        sliding_kv_read: scalar(if window == 0 {
            0.0
        } else if cap < anchor_prefix {
            cap
        } else {
            anchor_prefix
        }),
        ..Work::default()
    }
}

fn check_window(stats: &AliasStats, window: i64) -> Result<(), String> {
    if window != stats.window {
        return Err("length summaries must use the model attention window".to_string());
    }
    Ok(())
}

/// Expected work of one join stage at the current live counts.
///
/// # Errors
///
/// The anchor's length summary uses another attention window.
#[allow(clippy::too_many_arguments)]
pub fn stage_work(
    spec: &PricedJoin,
    anchor: &str,
    n: f64,
    tuples: f64,
    lengths: &BTreeMap<String, AliasStats>,
    pre: i64,
    resident: bool,
    window: i64,
) -> Result<Work, String> {
    let stats = &lengths[anchor];
    check_window(stats, window)?;
    if stats.count == 0 || n <= 0.0 {
        return Ok(Work::default());
    }
    let partners: Vec<&String> = spec
        .aliases
        .iter()
        .filter(|alias| *alias != anchor)
        .collect();
    let later_partners = python_sum(
        partners
            .iter()
            .skip(1)
            .map(|p| spec.label_tokens[*p] as f64 + lengths[*p].mean()),
    );
    let first_mean = partners.first().map(|p| lengths[*p].mean()).unwrap_or(0.0);
    let u = spec.tail_tokens as f64 + later_partners + first_mean;
    let frame =
        spec.frame_tokens[anchor] + partners.first().map(|p| spec.label_tokens[*p]).unwrap_or(0);
    let per_anchor = tuples / n;
    let fraction = n / stats.count as f64;

    let count = stats.count as i128;
    let (pre, frame_wide, total) = (pre as i128, frame as i128, stats.total as i128);
    let prefix_sum = total + pre * count;
    let prefix_squared = stats.squared + 2 * pre * total + pre * pre * count;
    let window_pairs = |offset: f64| stats.window_pairs(offset);
    let start = if resident {
        Work {
            tokens: scalar((count * frame_wide) as f64),
            pairs: scalar((frame_wide * prefix_sum) as f64 + count as f64 * tri(frame as f64)),
            kv_written: scalar((count * frame_wide) as f64),
            kv_read: scalar(prefix_sum as f64),
            sliding_pairs: scalar(if window != 0 {
                window_pairs((pre + frame_wide) as f64) - window_pairs(pre as f64)
            } else {
                0.0
            }),
            sliding_kv_read: scalar(if window != 0 {
                stats.window_reads(pre as f64)
            } else {
                0.0
            }),
            state_read: scalar(count as f64),
            state_written: scalar(count as f64),
            ..Work::default()
        }
    } else {
        let scan_sum = prefix_sum + count * frame_wide;
        let scan_squared =
            prefix_squared + 2 * frame_wide * prefix_sum + count * frame_wide * frame_wide;
        Work {
            tokens: scalar(scan_sum as f64),
            pairs: scalar(int_true_divide(scan_squared + scan_sum, 2)),
            kv_written: scalar(scan_sum as f64),
            sliding_pairs: scalar(if window != 0 {
                window_pairs((pre + frame_wide) as f64)
            } else {
                0.0
            }),
            state_written: scalar(count as f64),
            ..Work::default()
        }
    };
    let read = (prefix_sum + count * frame_wide) as f64;
    let stream = Work {
        tokens: scalar(count as f64 * per_anchor * u),
        pairs: scalar(per_anchor * (u * read + count as f64 * tri(u))),
        kv_written: scalar(count as f64 * per_anchor * u),
        kv_read: scalar(read),
        sliding_pairs: scalar(if window != 0 {
            let offset = (pre + frame_wide) as f64;
            per_anchor * (window_pairs(offset + u) - window_pairs(offset))
        } else {
            0.0
        }),
        sliding_kv_read: scalar(if window != 0 {
            stats.window_reads((pre + frame_wide) as f64)
        } else {
            0.0
        }),
        state_read: scalar(count as f64 * per_anchor),
        ..Work::default()
    };
    Ok(&(&start + &stream) * fraction)
}

/// Expected work of one join under set attention.
///
/// # Errors
///
/// The anchor's length summary uses another attention window.
#[allow(clippy::too_many_arguments)]
pub fn packed_join_work(
    spec: &PricedJoin,
    anchor: &str,
    n: f64,
    tuples: f64,
    lengths: &BTreeMap<String, AliasStats>,
    pre: i64,
    window: i64,
    set_size: i64,
) -> Result<Work, String> {
    let stats = &lengths[anchor];
    check_window(stats, window)?;
    if stats.count == 0 || n <= 0.0 {
        return Ok(Work::default());
    }
    let question = spec.tail_tokens as f64
        + python_sum(
            spec.aliases
                .iter()
                .filter(|alias| *alias != anchor)
                .map(|p| spec.label_tokens[p] as f64 + lengths[p].mean()),
        );
    let head = (pre + spec.label_tokens[anchor] + spec.frame_tokens[anchor]) as i128;
    let per_anchor = tuples / n;
    let passes = 1_i64.max((per_anchor / set_size as f64).ceil() as i64) as f64;
    let asked = per_anchor / passes;
    let questions = asked * question;
    let count = stats.count as i128;
    let total = stats.total as i128;
    let state_sum = total + head * count;
    let state_squared = stats.squared + 2 * head * total + head * head * count;
    let tokens = state_sum as f64 + count as f64 * questions;
    let pass_work = Work {
        tokens: scalar(tokens),
        pairs: scalar(
            state_squared as f64
                + 2.0 * questions * state_sum as f64
                + count as f64 * asked * question * question,
        ),
        kv_written: scalar(tokens),
        kv_read: scalar(if asked <= 1.0 { tokens } else { 2.0 * tokens }),
        sliding_pairs: scalar(if window != 0 {
            stats.window_pairs(head as f64 + questions)
        } else {
            0.0
        }),
        ..Work::default()
    };
    Ok(&pass_work * (passes * n / count as f64))
}
