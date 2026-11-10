//! Survivor and pair counts, and the arena's size.

use std::collections::BTreeMap;

use crate::cost::budgets::arena_pages;
use crate::cost::joins::PricedJoin;
use crate::cost::statistics::AliasStats;

use crate::context::PlanningContext;
use crate::error::CompileError;

/// The planner's selectivity when the query gives none.
pub const DEFAULT_SELECTIVITY: f64 = 0.2;

/// The selectivity the planner uses: the given one or the default.
pub fn effective_selectivity(selectivity: Option<f64>) -> f64 {
    selectivity.unwrap_or(DEFAULT_SELECTIVITY)
}

/// The fraction of the rows an AI filter reads that the filters after
/// it still work on: every row for a filter that returns its answer; for
/// a term of an OR that asks only the rows no earlier term passed, the
/// rows it does not pass; every row for any other term of an OR; else
/// its selectivity.
pub fn row_selectivity(predicate: &crate::physical::FilterPredicate) -> f64 {
    let passing = effective_selectivity(predicate.selectivity);
    if predicate.answer.is_some() {
        1.0
    } else if predicate.short_circuit {
        1.0 - passing
    } else if predicate.disjunction.is_some() {
        1.0
    } else {
        passing
    }
}

/// The fraction of the rows a model node reads that it asks about: the
/// guard's estimated selectivity, the default selectivity for a guard
/// without an estimate, and every row without a guard.
pub fn guarded_fraction(
    guard: Option<&crate::physical::ValueExpression>,
    selectivity: Option<f64>,
) -> f64 {
    match guard {
        Some(_) => effective_selectivity(selectivity),
        None => 1.0,
    }
}

/// The fraction of the rows an AI filter reads that it asks about.
pub fn predicate_asked_fraction(predicate: &crate::physical::FilterPredicate) -> f64 {
    guarded_fraction(predicate.guard.as_ref(), predicate.guard_selectivity)
}

/// Expected documents with at least one matching tuple.
pub fn surviving_documents(num_documents: f64, num_partners: f64, selectivity: Option<f64>) -> f64 {
    let selectivity = effective_selectivity(selectivity);
    let probability = if selectivity >= 1.0 {
        if num_partners > 0.0 { 1.0 } else { 0.0 }
    } else {
        -(num_partners * (-selectivity).ln_1p()).exp_m1()
    };
    num_documents * probability
}

/// Update live document counts after one join's selectivity.
pub fn thin(expected_rows: &mut BTreeMap<String, f64>, spec: &PricedJoin) {
    let selectivity = effective_selectivity(spec.selectivity) * spec.pair_fraction;
    let others = |rows: &BTreeMap<String, f64>, alias: &str| {
        let mut out = 1.0;
        for other in &spec.aliases {
            if other != alias {
                out *= rows[other];
            }
        }
        out
    };
    if spec.semantics == "full" {
        let updated: Vec<(String, f64)> = spec
            .aliases
            .iter()
            .map(|alias| {
                let partners = others(expected_rows, alias);
                (
                    alias.clone(),
                    surviving_documents(expected_rows[alias], partners, Some(selectivity)),
                )
            })
            .collect();
        expected_rows.extend(updated);
    } else {
        let anchor = spec.anchor.clone().expect("a gate's anchor");
        let matched = surviving_documents(
            expected_rows[&anchor],
            others(expected_rows, &anchor),
            Some(selectivity),
        );
        let rows = expected_rows[&anchor];
        expected_rows.insert(
            anchor,
            if spec.semantics == "exists" {
                matched
            } else {
                rows - matched
            },
        );
    }
}

/// Expected tuples of one join.
pub fn cross_tuples(spec: &PricedJoin, expected_rows: &BTreeMap<String, f64>) -> f64 {
    let mut tuples = 1.0;
    for alias in &spec.aliases {
        tuples *= expected_rows[alias];
    }
    tuples * spec.pair_fraction
}

/// Expected pairs and first documents with at least one pair.
pub fn pair_counts(
    expected_rows: &BTreeMap<String, f64>,
    aliases: &[String],
    fraction: f64,
) -> (f64, f64) {
    let (left, right) = (expected_rows[&aliases[0]], expected_rows[&aliases[1]]);
    (
        left * right * fraction,
        surviving_documents(left, right, Some(fraction)),
    )
}

/// The tokens of document KV the query's arena holds at once.
///
/// # Errors
///
/// The arena does not fit the device.
pub fn arena_capacity(context: &PlanningContext, chunk: i64) -> Result<i64, CompileError> {
    let window = context.model.sliding_window;
    let mut longest: Option<f64> = None;
    for lengths in context.document_tokens.values() {
        let mean = AliasStats::summarize(lengths, window).mean();
        if longest.is_none_or(|found| mean > found) {
            longest = Some(mean);
        }
    }
    let (full, _) = arena_pages(&context.model, &context.device, Some(chunk), longest)
        .map_err(CompileError::new)?;
    Ok(full * context.device.page_tokens)
}
