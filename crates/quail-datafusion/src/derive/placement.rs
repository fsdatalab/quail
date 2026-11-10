//! Name column operators and place conditions where their columns exist.
//! Put tests of answers on each table's operator stack.

use super::assemble::{Assembly, StackItem};
use super::binding::{Binder, check_labels};
use super::relation::{JoinSpec, Relation, check_calls, check_joins};
use super::walk::Walk;
use super::{ByAlias, lookup, note, push};
use crate::SPAN_SUFFIX;
use quail_planner::error::CompileError;
use quail_planner::expressions::{
    aliased_call, call_aliases, column_name, column_references, condition_columns,
};
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
        let tested = &binder.tested;
        // a tested label takes SELECT's name, a test's, or one of its own
        let mut names: Vec<(ModelCall, String)> = Vec::new();
        for (alias, found) in &label_tests {
            let asks = lookup(&filters, alias).len();
            for (index, (call, _, _)) in found.iter().enumerate() {
                let name = projected
                    .iter()
                    .find(|(owner, _)| *owner == call)
                    .map(|(_, name)| name)
                    .or_else(|| {
                        tested
                            .iter()
                            .rev()
                            .find(|(owner, _)| owner == call)
                            .map(|(_, name)| name)
                    })
                    .cloned()
                    .unwrap_or_else(|| format!("__label_{alias}_{}", asks + index));
                names.push((call.clone(), name));
            }
        }
        names.extend(
            projected
                .iter()
                .map(|(call, name)| ((*call).clone(), name.clone())),
        );
        names.extend(tested.iter().cloned());
        let named = |call: &ModelCall| -> &String {
            let (_, name) = names
                .iter()
                .find(|(found, _)| found == call)
                .expect("a named call");
            name
        };
        // the columns each table's column operators add
        let mut computed: BTreeMap<(String, String), &str> = BTreeMap::new();
        for (call, _) in &names {
            let read = call_aliases(call);
            if read.len() == 1
                && ["label", "extract", "features", "map"].contains(&call.kind.as_str())
            {
                let name = named(call);
                computed.insert((read[0].clone(), name.clone()), &call.kind);
                if call.kind == "extract" {
                    computed.insert((read[0].clone(), format!("{name}{SPAN_SUFFIX}")), "extract");
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
            for (call, labels, selectivity) in found {
                check_labels(named(call), labels, call)?;
                terms.push(Term {
                    condition: binder.label_condition(alias, named(call), labels)?,
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
            names,
            conditions,
            found: Operators::default(),
            stacks: Vec::new(),
            noted: Vec::new(),
            join_on: BTreeMap::new(),
            columns_at: BTreeMap::new(),
        })
    }
    /// Each table's tests of answers, on its stack, and the operators'
    /// column filters.
    pub(super) fn place_tests(&mut self) -> Option<()> {
        let mut stacks = std::mem::take(&mut self.stacks);
        for (alias, stack) in &mut stacks {
            for (index, term) in lookup(&self.conditions.computed, alias).iter().enumerate() {
                let position = self.test_position(stack, &term.condition);
                stack.insert(position, StackItem::Test(index));
                note(&mut self.noted, alias);
            }
        }
        for alias in &self.noted {
            let stack = lookup(&stacks, alias);
            let computed = lookup(&self.conditions.computed, alias);
            let mut tests: Vec<(usize, &SemanticColumn)> = Vec::new();
            for (at, item) in stack.iter().enumerate() {
                let StackItem::Test(index) = item else {
                    continue;
                };
                let read = condition_columns(&computed[*index].condition);
                let producer = stack[..at].iter().rev().find_map(|below| match below {
                    StackItem::Column(kind, position) => {
                        let column = self.column(kind, *position);
                        let adds = column.added_columns();
                        adds.iter()
                            .any(|name| read.contains(name))
                            .then_some(column)
                    }
                    _ => None,
                })?;
                tests.push((*index, producer));
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
        self.stacks = stacks;
        Some(())
    }

    /// Where on a stack a test of answers goes: right above the lowest
    /// operator after which every answer column it reads is added, and
    /// above the tests placed before it; a test that reads no answer
    /// column goes right above the first extraction, SAE feature, or map.
    fn test_position(&self, stack: &[StackItem], condition: &ValueExpression) -> usize {
        let reads: BTreeSet<String> = condition_columns(condition)
            .into_iter()
            .filter(|column| self.stack_columns(stack).contains(column))
            .collect();
        let mut top = stack.len();
        while let Some(StackItem::Column(..)) = top.checked_sub(1).map(|at| &stack[at]) {
            let below = &stack[..top - 1];
            let descends = if reads.is_empty() {
                self.has_answer_column(below)
            } else {
                reads.is_subset(&self.stack_columns(below))
            };
            if !descends {
                break;
            }
            top -= 1;
        }
        top
    }

    /// The answer columns the column operators of a stack add, down to its
    /// AI filters.
    fn stack_columns(&self, stack: &[StackItem]) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        for item in stack.iter().rev() {
            match item {
                StackItem::Column(kind, position) => {
                    found.extend(self.column(kind, *position).added_columns());
                }
                StackItem::Test(_) => {}
                StackItem::Filters => break,
            }
        }
        found
    }

    /// Whether the top of a stack, or an operator below it, is an
    /// extraction, SAE feature, or map column.
    fn has_answer_column(&self, stack: &[StackItem]) -> bool {
        for item in stack.iter().rev() {
            match item {
                StackItem::Column(kind, _) if *kind != "label" => return true,
                StackItem::Filters => return false,
                _ => {}
            }
        }
        false
    }
}
