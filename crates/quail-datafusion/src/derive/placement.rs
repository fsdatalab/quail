//! Place conditions where their columns exist: at a table's scan, after
//! the column operator whose answer they read, or on the result rows.

use super::assemble::Assembly;
use super::binding::{Binder, check_labels};
use super::relation::{JoinSpec, Relation, check_calls, check_joins};
use super::walk::Walk;
use super::{ByAlias, lookup, note, push};
use crate::SPAN_SUFFIX;
use quail_planner::error::CompileError;
use quail_planner::expressions::{aliased_call, column_name, column_references, condition_columns};
use quail_planner::operators::{ColumnTest, Operators, SemanticColumn};
use quail_planner::physical::{ModelCall, ValueExpression};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// A condition that asks no model, with its HINT's selectivity and, for
/// a test of a label, the labels it accepts.
#[derive(Clone)]
pub(super) struct Term {
    pub(super) condition: ValueExpression,
    selectivity: Option<f64>,
    labels: Option<Value>,
}

/// The conditions of a relation, by where they run.
#[derive(Default)]
pub(super) struct PlacedConditions {
    pub(super) scans: ByAlias<ValueExpression>,
    pub(super) computed: ByAlias<Term>,
    pub(super) rows: Vec<ValueExpression>,
}

/// Place each condition where its columns exist: at a table's scan,
/// after the operator of one table that computes the columns it reads,
/// or on the result rows.
///
/// `computed` maps (alias, column) to the kind of the operator that
/// computes it.
fn place_conditions(
    conditions: Vec<Term>,
    computed: &BTreeMap<(String, String), &str>,
) -> Result<PlacedConditions, CompileError> {
    let mut placed = PlacedConditions::default();
    for term in conditions {
        let part = &term.condition;
        let references = column_references(part);
        let aliases: BTreeSet<&String> = references.iter().map(|found| &found.alias).collect();
        let owners: BTreeSet<Option<(&str, &String)>> = references
            .iter()
            .map(|found| {
                computed
                    .get(&(found.alias.clone(), found.column.clone()))
                    .map(|kind| (*kind, &found.column))
            })
            .collect();
        let kinds: BTreeSet<&str> = owners.iter().flatten().map(|(kind, _)| *kind).collect();
        let one_table = aliases.len() == 1;
        if one_table && owners.len() == 1 && owners.contains(&None) {
            if term.selectivity.is_some() {
                return Err(CompileError::unsupported(
                    "DataFusion must evaluate this hinted test of source columns.",
                ));
            }
            push(
                &mut placed.scans,
                aliases.first().expect("one alias"),
                part.clone(),
            );
        } else if one_table && kinds.len() == 1 {
            let alias = (*aliases.first().expect("one alias")).clone();
            push(&mut placed.computed, &alias, term);
        } else {
            if kinds.contains("label") && owners.iter().flatten().count() > 1 {
                return Err(CompileError::unsupported(
                    "DataFusion must evaluate this test of different model results.",
                ));
            }
            if term.selectivity.is_some() {
                return Err(CompileError::unsupported(
                    "DataFusion must evaluate this hinted test of result rows.",
                ));
            }
            placed.rows.push(term.condition);
        }
    }
    Ok(placed)
}

impl<'w, 'a> Assembly<'w, 'a> {
    /// Name each column operator and place the relation's conditions.
    ///
    /// `columns` is the SELECT list.
    ///
    /// # Errors
    ///
    /// A term does not fit the query's tables, or the query has no AI
    /// predicate where it needs one.
    pub(super) fn new(
        walk: &'w Walk<'a>,
        binder: &'w mut Binder,
        relation: Relation,
        columns: &[ValueExpression],
    ) -> Result<Self, CompileError> {
        let Relation {
            filters,
            joins,
            label_tests,
            mut conditions,
            equalities,
            tables,
            ..
        } = relation;
        let projected: Vec<(&ModelCall, String)> = columns
            .iter()
            .filter_map(|column| Some((aliased_call(column, &[])?, column_name(column, true))))
            .collect();
        let specs: Vec<&JoinSpec> = joins.values().collect();
        check_joins(&equalities, &tables, &specs, &projected)?;
        check_calls(&filters, &specs, &projected)?;
        // the columns each table's column operators add
        let mut names: Vec<(ModelCall, String)> = Vec::new();
        let mut computed: BTreeMap<(String, String), &str> = BTreeMap::new();
        for found in &binder.answers {
            if !["label", "extract", "features", "map"].contains(&found.kind) {
                continue;
            }
            for (call, name) in &found.calls {
                names.push((call.clone(), name.clone()));
                if let Some(alias) = &found.alias {
                    computed.insert((alias.clone(), name.clone()), found.kind);
                    if found.kind == "extract" {
                        let span = format!("{name}{SPAN_SUFFIX}");
                        computed.insert((alias.clone(), span), "extract");
                    }
                }
            }
        }
        let mut terms: Vec<Term> = conditions
            .drain(..)
            .map(|(condition, selectivity)| Term {
                condition,
                selectivity,
                labels: None,
            })
            .collect();
        for (alias, found) in &label_tests {
            for (name, labels, selectivity) in found {
                let (call, _) = names
                    .iter()
                    .find(|(_, found)| found == name)
                    .expect("a named label");
                check_labels(name, labels, call)?;
                terms.push(Term {
                    condition: binder.label_condition(alias, name, labels)?,
                    selectivity: *selectivity,
                    labels: Some(labels.clone()),
                });
            }
        }
        let conditions = place_conditions(terms, &computed)?;
        Ok(Self {
            walk,
            binder,
            filters,
            joins,
            conditions,
            found: Operators::default(),
            placed: Vec::new(),
            noted: Vec::new(),
        })
    }
    /// Each table's tests of answers, as its operators' column filters:
    /// a test follows the last column operator whose answer it reads.
    pub(super) fn place_tests(&mut self) -> Option<()> {
        for scan in &self.found.scans {
            if !lookup(&self.conditions.computed, &scan.alias).is_empty() {
                note(&mut self.noted, &scan.alias);
            }
        }
        for alias in &self.noted {
            let computed = lookup(&self.conditions.computed, alias);
            let mut tests: Vec<(usize, &SemanticColumn)> = Vec::new();
            for (index, term) in computed.iter().enumerate() {
                let read = condition_columns(&term.condition);
                let producer = lookup(&self.placed, alias)
                    .iter()
                    .rev()
                    .map(|(kind, position)| self.column(kind, *position))
                    .find(|column| {
                        column
                            .added_columns()
                            .iter()
                            .any(|name| read.contains(name))
                    })?;
                tests.push((index, producer));
            }
            if tests.is_empty() {
                continue;
            }
            tests.sort_by_key(|(_, producer)| producer.call.kind != "label");
            let asks = lookup(&self.found.filters, alias).len();
            let column_tests = tests
                .iter()
                .enumerate()
                .map(|(order, (index, producer))| ColumnTest {
                    name: producer.name.clone(),
                    call: producer.call.clone(),
                    condition: computed[*index].condition.clone(),
                    labels: computed[*index].labels.clone(),
                    selectivity: computed[*index].selectivity,
                    position: asks + order,
                    before_joins: true,
                })
                .collect();
            self.found
                .column_filters
                .push((alias.clone(), column_tests));
        }
        Some(())
    }
}
