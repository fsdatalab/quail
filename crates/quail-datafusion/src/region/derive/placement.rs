use super::{ByAlias, lookup, push};
use quail_planner::error::CompileError;
use quail_planner::expressions::{column_references, condition_columns};
use quail_planner::operators::{ColumnTest, SemanticColumn};
use quail_planner::physical::ValueExpression;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(super) struct Term {
    condition: ValueExpression,
    selectivity: Option<f64>,
    labels: Option<Value>,
}

impl Term {
    pub(super) fn new(
        condition: ValueExpression,
        selectivity: Option<f64>,
        labels: Option<Value>,
    ) -> Self {
        Self {
            condition,
            selectivity,
            labels,
        }
    }
}

#[derive(Default)]
pub(super) struct PlacedConditions {
    scans: ByAlias<ValueExpression>,
    computed: ByAlias<Term>,
    rows: Vec<ValueExpression>,
}

impl PlacedConditions {
    pub(super) fn place(
        conditions: Vec<Term>,
        computed: &BTreeMap<(String, String), &str>,
    ) -> Result<Self, CompileError> {
        let mut placed = Self::default();
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

    pub(super) fn at_scan(&self, alias: &str) -> Vec<ValueExpression> {
        lookup(&self.scans, alias).to_vec()
    }

    pub(super) fn tests_answers(&self, alias: &str) -> bool {
        !lookup(&self.computed, alias).is_empty()
    }

    pub(super) fn column_tests(
        &self,
        alias: &str,
        producers: &[&SemanticColumn],
        position: usize,
    ) -> Option<Vec<ColumnTest>> {
        let computed = lookup(&self.computed, alias);
        let mut tests: Vec<(&Term, &SemanticColumn)> = Vec::new();
        for term in computed {
            let read = condition_columns(&term.condition);
            let producer = producers.iter().rev().find(|column| {
                column
                    .added_columns()
                    .iter()
                    .any(|name| read.contains(name))
            })?;
            tests.push((term, producer));
        }
        tests.sort_by_key(|(_, producer)| producer.call.kind != "label");
        Some(
            tests
                .into_iter()
                .enumerate()
                .map(|(order, (term, producer))| ColumnTest {
                    name: producer.name.clone(),
                    call: producer.call.clone(),
                    condition: term.condition.clone(),
                    labels: term.labels.clone(),
                    selectivity: term.selectivity,
                    position: position + order,
                    before_joins: true,
                })
                .collect(),
        )
    }

    pub(super) fn row_test(&self) -> Option<&ValueExpression> {
        self.rows.first()
    }
}
