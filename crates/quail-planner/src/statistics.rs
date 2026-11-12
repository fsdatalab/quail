//! Survivor and pair counts, and the arena's size.

use crate::constants::DEFAULT_SELECTIVITY;
use std::collections::BTreeMap;

use crate::cost::budgets::arena_pages;
use crate::cost::joins::PricedJoin;
use crate::cost::statistics::AliasStats;

use crate::context::PlanningContext;
use crate::error::CompileError;

/// The selectivity the planner uses: the given one or the default.
pub fn effective_selectivity(selectivity: Option<f64>) -> f64 {
    selectivity.unwrap_or(DEFAULT_SELECTIVITY)
}

/// The fraction of the rows an AI filter reads that it passes: every row
/// for a filter that returns its answer, else its selectivity.
pub fn row_selectivity(predicate: &crate::physical::FilterPredicate) -> f64 {
    if predicate.answer.is_some() {
        1.0
    } else {
        effective_selectivity(predicate.selectivity)
    }
}

/// An AI filter's place in an OR of several `AI.IF` calls.
#[derive(Debug, Clone, PartialEq)]
pub struct OrTerm {
    /// The OR's name.
    pub disjunction: String,
    /// Whether the filter asks only the rows no earlier term passed.
    pub short_circuit: bool,
}

impl OrTerm {
    /// The place of `predicate` in an OR, when it is a term of one.
    pub fn of(predicate: &crate::physical::FilterPredicate) -> Option<Self> {
        Some(Self {
            disjunction: predicate.disjunction.clone()?,
            short_circuit: predicate.short_circuit,
        })
    }
}

/// The rows of one table that reach each of its steps, as the planner
/// estimates them while the steps run in order.
///
/// A step that is no term of an OR passes its selectivity of the rows.
/// The terms of an OR pass `1 - (1 - p_1)...(1 - p_k)` of them together,
/// for terms that pass `p_i` each, independently (Selinger et al., 1979).
/// The rows leave at the OR's last term, in whatever order the terms
/// run and whatever other steps run between them; until then the flow
/// keeps the fraction of the rows no term has passed, which is all a
/// short-circuit term asks about.
#[derive(Debug, Clone)]
pub struct RowFlow {
    rows: f64,
    /// By OR, the fraction of the rows no term has passed yet.
    undecided: BTreeMap<String, f64>,
}

impl RowFlow {
    /// The flow of a table with `rows` rows, before any step.
    pub fn new(rows: f64) -> Self {
        Self {
            rows,
            undecided: BTreeMap::new(),
        }
    }

    /// The rows that reach the next step.
    pub fn rows(&self) -> f64 {
        self.rows
    }

    /// This flow with `rows` reaching the next step, for a caller that
    /// reads them from the step's input.
    pub fn with_rows(mut self, rows: f64) -> Self {
        self.rows = rows;
        self
    }

    /// The rows the next step asks the model about, when it is `term`
    /// of an OR or no term of one.
    pub fn asked(&self, term: Option<&OrTerm>) -> f64 {
        match term {
            Some(term) if term.short_circuit => {
                self.rows * self.undecided.get(&term.disjunction).unwrap_or(&1.0)
            }
            _ => self.rows,
        }
    }

    /// Move past a step that passes `selectivity` of the rows it asks.
    ///
    /// `last` says whether the step is the last term of its OR to run.
    pub fn pass(&mut self, selectivity: f64, term: Option<&OrTerm>, last: bool) {
        let Some(term) = term else {
            self.rows *= selectivity;
            return;
        };
        let undecided =
            self.undecided.remove(&term.disjunction).unwrap_or(1.0) * (1.0 - selectivity);
        if last {
            self.rows *= 1.0 - undecided;
        } else {
            self.undecided.insert(term.disjunction.clone(), undecided);
        }
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

#[cfg(test)]
mod row_flow_tests {
    use super::{OrTerm, RowFlow};

    fn term() -> OrTerm {
        OrTerm {
            disjunction: "or".into(),
            short_circuit: true,
        }
    }

    #[test]
    fn an_or_passes_the_rows_any_term_passes_whatever_runs_between_its_terms() {
        // (a OR b) AND c over 100 rows, run as a, c, b: a and b pass 20%
        // each and c keeps half
        let mut flow = RowFlow::new(100.0);
        assert_eq!(flow.asked(Some(&term())), 100.0);
        flow.pass(0.2, Some(&term()), false);
        assert_eq!(flow.rows(), 100.0);
        flow.pass(0.5, None, false);
        // b asks the 40 of the 50 rows left that a did not pass
        assert!((flow.asked(Some(&term())) - 40.0).abs() < 1e-9);
        flow.pass(0.2, Some(&term()), true);
        // 10 rows a passed and 8 rows b passed
        assert!((flow.rows() - 18.0).abs() < 1e-9);
    }

    #[test]
    fn the_rows_an_or_passes_do_not_depend_on_the_order_of_its_terms() {
        let run = |first: f64, second: f64| {
            let mut flow = RowFlow::new(100.0);
            flow.pass(first, Some(&term()), false);
            flow.pass(second, Some(&term()), true);
            flow.rows()
        };
        // 1 - 0.9 * 0.5
        assert!((run(0.1, 0.5) - 55.0).abs() < 1e-9);
        assert!((run(0.5, 0.1) - 55.0).abs() < 1e-9);
    }

    #[test]
    fn a_term_that_does_not_short_circuit_asks_every_row() {
        let every = OrTerm {
            short_circuit: false,
            ..term()
        };
        let mut flow = RowFlow::new(100.0);
        flow.pass(0.2, Some(&every), false);
        assert_eq!(flow.asked(Some(&every)), 100.0);
    }
}
