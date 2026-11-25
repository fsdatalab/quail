use crate::operators::effective_selectivity;
use std::collections::BTreeMap;

use crate::cost::joins::PricedJoin;

pub fn row_selectivity(predicate: &crate::physical::FilterPredicate) -> f64 {
    if predicate.answer.is_some() {
        1.0
    } else {
        effective_selectivity(predicate.selectivity)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrTerm {
    pub disjunction: String,
    pub short_circuit: bool,
}

impl OrTerm {
    pub fn of(predicate: &crate::physical::FilterPredicate) -> Option<Self> {
        Some(Self {
            disjunction: predicate.disjunction.clone()?,
            short_circuit: predicate.short_circuit,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RowFlow {
    rows: f64,
    undecided: BTreeMap<String, f64>,
}

impl RowFlow {
    pub fn new(rows: f64) -> Self {
        Self {
            rows,
            undecided: BTreeMap::new(),
        }
    }

    pub fn rows(&self) -> f64 {
        self.rows
    }

    pub fn with_rows(mut self, rows: f64) -> Self {
        self.rows = rows;
        self
    }

    pub fn asked(&self, term: Option<&OrTerm>) -> f64 {
        match term {
            Some(term) if term.short_circuit => {
                self.rows * self.undecided.get(&term.disjunction).unwrap_or(&1.0)
            }
            _ => self.rows,
        }
    }

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

pub fn surviving_documents(num_documents: f64, num_partners: f64, selectivity: Option<f64>) -> f64 {
    let selectivity = effective_selectivity(selectivity);
    let probability = if selectivity >= 1.0 {
        if num_partners > 0.0 { 1.0 } else { 0.0 }
    } else {
        -(num_partners * (-selectivity).ln_1p()).exp_m1()
    };
    num_documents * probability
}

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

pub fn cross_tuples(spec: &PricedJoin, expected_rows: &BTreeMap<String, f64>) -> f64 {
    let mut tuples = 1.0;
    for alias in &spec.aliases {
        tuples *= expected_rows[alias];
    }
    tuples * spec.pair_fraction
}

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
        let mut flow = RowFlow::new(100.0);
        assert_eq!(flow.asked(Some(&term())), 100.0);
        flow.pass(0.2, Some(&term()), false);
        assert_eq!(flow.rows(), 100.0);
        flow.pass(0.5, None, false);
        assert!((flow.asked(Some(&term())) - 40.0).abs() < 1e-9);
        flow.pass(0.2, Some(&term()), true);
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
