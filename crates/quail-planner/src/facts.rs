//! What the executor's caller reads from a region's operators, so that
//! no plan crosses into Python.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::expressions::asked_call;
use crate::expressions::{argument_alias, call_aliases, column_name, column_references};
use crate::operators::{Operators, ScanColumn};
use crate::plan_statistics::plan_prompts;

/// A region's facts as JSON:
///
/// - `asks_model`: whether any operator asks the model;
/// - `scans`: each scan's alias, provider, document column and its
///   name without the alias, value columns, and the column names of
///   its table's schema;
/// - `value_columns`: alias -> the columns the executor reads by
///   document id: join equalities, regular predicates, tests of labels
///   and answers on the table's own columns, DISTINCT stop keys, and
///   the guards of column operators and AI filters and the
///   alternatives of AI filters;
/// - `stop_keys`: alias -> its DISTINCT stop key;
/// - `join_conditions`: for each join with equalities, its written
///   position, its two aliases, and each equality's sides in that
///   alias order;
/// - `prompts`: each table's AI.IF prompts in written order, each
///   join's prompt by written position, and each classification's
///   prompt and aliases by its output name.
pub fn region_facts(found: &Operators) -> Value {
    let mut value_columns: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut add = |alias: &str, column: &str| {
        let columns = value_columns.entry(alias.to_string()).or_default();
        if !columns.iter().any(|known| known == column) {
            columns.push(column.to_string());
        }
    };
    for join in &found.joins {
        for condition in &join.on {
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
            for reference in column_references(predicate) {
                add(&scan.alias, &reference.column);
            }
        }
    }
    for (alias, filters) in &found.column_filters {
        let columns = schemas.get(alias.as_str()).cloned().unwrap_or_default();
        for filter in filters {
            for reference in column_references(&filter.condition) {
                if reference.alias == *alias && columns.contains(&reference.column) {
                    add(alias, &reference.column);
                }
            }
        }
    }
    for (alias, key) in &found.stop_keys {
        for column in key {
            add(alias, column);
        }
    }
    let answers = found.filter_answers();
    let filter_guards = found.filters.iter().flat_map(|(_, predicates)| {
        predicates
            .iter()
            .flat_map(|predicate| predicate.guard.iter().chain(&predicate.alternative))
    });
    for guard in found
        .columns()
        .filter_map(|column| column.guard.as_ref())
        .chain(filter_guards)
    {
        // a guard may read an earlier AI filter's answer, which no
        // table holds
        for reference in column_references(guard) {
            if !answers.contains(&(reference.alias.as_str(), reference.column.as_str())) {
                add(&reference.alias, &reference.column);
            }
        }
    }
    let scans: Vec<Value> = found
        .scans
        .iter()
        .map(|scan| {
            json!({
                "alias": scan.alias, "provider": scan.provider,
                "column": scan.column, "columns": scan.columns,
                "column_name": match &scan.column {
                    ScanColumn::Unnamed(name) => name.clone(),
                    ScanColumn::Expression(column) => column_name(column, false),
                },
                "schema": schemas.get(scan.alias.as_str()),
                "schema_ipc": scan.schema.as_ref().map(|schema| schema.ipc.clone()),
            })
        })
        .collect();
    let mut oriented = Vec::new();
    for (position, join) in found.joins.iter().enumerate() {
        let conditions = &join.on;
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
        .map(|column| {
            let call = &column.call;
            (
                column.name.as_str(),
                json!({"prompt": call.prompt, "aliases": call_aliases(call)}),
            )
        })
        .collect();
    json!({
        "prompts": {"filters": filters, "joins": joins, "classifies": classifies},
        "asks_model": !plan_prompts(found).is_empty(),
        "stop_keys": found.stop_keys,
        "num_key_survivors": found.num_key_survivors,
        "join_conditions": oriented,
        "scans": scans,
        "value_columns": value_columns,
    })
}
