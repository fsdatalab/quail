//! What the executor's caller reads from a region's logical plan, so that
//! no logical plan crosses into Python.

use std::collections::BTreeMap;

use quail_physical::{Projected, SortExpression, ValueExpression};
use serde_json::{Value, json};

use crate::LogicalPlan;
use crate::costs::join_conditions;
use crate::expressions::{argument_alias, call_aliases, column_references, tests};
use crate::graph::filter_stop_keys;
use crate::logical::LogicalNode;
use crate::nodes::{asked_call, column_call};
use crate::operators::{operators, projection};
use crate::plan_statistics::plan_prompts;

/// A region's facts as JSON:
///
/// - `asks_model`: whether any operator asks the model;
/// - `scans`: each scan's alias, provider, document column, value
///   columns, and the column names of its table's schema;
/// - `value_columns`: alias -> the columns the executor reads by
///   document id: join equalities, regular predicates, tests of labels
///   and answers on the table's own columns, and DISTINCT stop keys;
/// - `relational_columns`: the source columns a region that asks no
///   model reads: its projection, sort keys, and predicates;
/// - `stop_keys`: alias -> its DISTINCT stop key;
/// - `join_conditions`: for each join with equalities, its written
///   position, its two aliases, and each equality's sides in that
///   alias order;
/// - `prompts`: each table's AI.IF prompts in written order, each
///   join's prompt by written position, and each classification's
///   prompt and aliases by its output name.
pub fn region_facts(plan: &LogicalPlan) -> Value {
    let found = operators(plan);
    let mut value_columns: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut add = |alias: &str, column: &str| {
        let columns = value_columns.entry(alias.to_string()).or_default();
        if !columns.iter().any(|known| known == column) {
            columns.push(column.to_string());
        }
    };
    for join in &found.joins {
        for condition in join_conditions(join) {
            for reference in column_references(&condition.left)
                .into_iter()
                .chain(column_references(&condition.right))
            {
                add(&reference.alias, &reference.column);
            }
        }
    }
    let schemas: BTreeMap<&str, Vec<String>> = found
        .scans
        .iter()
        .map(|scan| {
            (
                scan.alias.as_str(),
                scan.schema
                    .as_ref()
                    .map(|schema| schema.names())
                    .unwrap_or_default(),
            )
        })
        .collect();
    for scan in &found.scans {
        for predicate in &scan.predicates {
            for test in tests(predicate) {
                for reference in column_references(&test.expression) {
                    add(&scan.alias, &reference.column);
                }
            }
        }
    }
    for (alias, filters) in &found.column_filters {
        let columns = schemas.get(alias.as_str()).cloned().unwrap_or_default();
        for filter in filters {
            for test in tests(&filter.condition) {
                for reference in column_references(&test.expression) {
                    if reference.alias == *alias && columns.contains(&reference.column) {
                        add(alias, &reference.column);
                    }
                }
            }
        }
    }
    for (alias, key) in filter_stop_keys(plan) {
        for column in key {
            add(&alias, &column);
        }
    }
    let scans: Vec<Value> = found
        .scans
        .iter()
        .map(|scan| {
            json!({
                "alias": scan.alias, "provider": scan.provider,
                "column": scan.column, "columns": scan.columns,
                "schema": schemas.get(scan.alias.as_str()),
                "schema_ipc": scan.schema.as_ref().map(|schema| schema.ipc.clone()),
            })
        })
        .collect();
    let mut oriented = Vec::new();
    for (position, join) in found.joins.iter().enumerate() {
        let conditions = join_conditions(join);
        let Some(first) = conditions.first() else {
            continue;
        };
        let left_alias = argument_alias(&first.left);
        let right_alias = argument_alias(&first.right);
        let pairs: Vec<Value> = conditions
            .iter()
            .map(|condition| {
                if argument_alias(&condition.left) == left_alias {
                    json!([condition.left, condition.right])
                } else {
                    json!([condition.right, condition.left])
                }
            })
            .collect();
        oriented.push(json!({
            "position": position, "left_alias": left_alias,
            "right_alias": right_alias, "pairs": pairs,
        }));
    }
    let filters: BTreeMap<&str, Vec<Value>> = found
        .filters
        .iter()
        .map(|(alias, predicates)| {
            let prompts = predicates
                .iter()
                .map(|predicate| json!(asked_call(&predicate.expression).prompt))
                .collect();
            (alias.as_str(), prompts)
        })
        .collect();
    let joins: Vec<Value> = found
        .joins
        .iter()
        .map(|join| json!(asked_call(&join.predicate).prompt))
        .collect();
    let classifies: BTreeMap<&str, Value> = found
        .classifies
        .iter()
        .map(|node| {
            let call = column_call(node);
            (
                node.name.as_str(),
                json!({"prompt": call.prompt, "aliases": call_aliases(call)}),
            )
        })
        .collect();
    json!({
        "prompts": {"filters": filters, "joins": joins, "classifies": classifies},
        "asks_model": !plan_prompts(&found).is_empty(),
        "stop_keys": filter_stop_keys(plan),
        "join_conditions": oriented,
        "scans": scans,
        "value_columns": value_columns,
        "relational_columns": relational_columns(plan),
    })
}

/// The source columns a region that asks no model reads, in first-read
/// order: its projection, its sort keys, and its scans' predicates.
fn relational_columns(plan: &LogicalPlan) -> Vec<String> {
    let mut expressions: Vec<&ValueExpression> = Vec::new();
    if let LogicalNode::Project(project) = projection(plan) {
        expressions.extend(project.columns.iter().filter_map(|column| match column {
            Projected::Expression(expression) => Some(expression),
            Projected::Name(_) => None,
        }));
    }
    if let LogicalNode::Result(result) = &plan.root {
        expressions.extend(result.order.iter().map(|key| {
            let SortExpression::Value(value) = &key.expression;
            value
        }));
    }
    let found = operators(plan);
    for scan in &found.scans {
        for predicate in &scan.predicates {
            expressions.extend(tests(predicate).into_iter().map(|test| &test.expression));
        }
    }
    let mut columns: Vec<String> = Vec::new();
    for expression in expressions {
        for reference in column_references(expression) {
            if !columns.contains(&reference.column) {
                columns.push(reference.column);
            }
        }
    }
    columns
}
