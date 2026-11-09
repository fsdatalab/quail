//! Bind SELECT items to output columns and extraction structs.
//! Check output names and reject repeated AI expressions.

use super::calls::{arguments, is_call};
use super::conditions::has_ai_call;
use super::{Binder, contains, unsupported};
use crate::functions::{ai_function_name, column_call_kind, function_name};
use datafusion::logical_expr::Expr;
use quail_planner::error::CompileError;
use quail_planner::expressions::{aliased_call, call_aliases, column_name, repr_list, repr_str};
use quail_planner::physical::{Alias, ExtractStruct, ValueExpression};

/// Each output column name of a SELECT list and its expression, but a
/// column the list passes on under its own name.
pub(super) fn select_outputs(expressions: &[Expr]) -> Vec<(String, Expr)> {
    let mut outputs: Vec<(String, Expr)> = Vec::new();
    for item in expressions {
        let (inner, _) = unalias(item);
        let name = output_name(item);
        let self_reference = matches!(inner, Expr::Column(column) if column.name == name);
        if !self_reference && !outputs.iter().any(|(found, _)| *found == name) {
            outputs.push((name, inner.clone()));
        }
    }
    outputs
}

/// Each column-operator call a SELECT list returns, by its output name.
pub(super) fn selected_calls(expressions: &[Expr]) -> Vec<(String, Expr)> {
    expressions
        .iter()
        .filter_map(|expr| {
            let (inner, _) = unalias(expr);
            column_call_kind(inner).map(|_| (output_name(expr), inner.clone()))
        })
        .collect()
}

fn output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias(alias) => alias.name.clone(),
        Expr::Column(column) => column.name.clone(),
        other => other.schema_name().to_string(),
    }
}

/// An expression without the AS names on it, and the name the query
/// wrote for it.
///
/// DataFusion names `COUNT(*)` itself, as `count(Int64(1)) AS count(*)`,
/// and a call written by a dotted name, as `ai_map(...) AS AI.MAP(...)`;
/// those names are not ones the query wrote.
fn unalias(expr: &Expr) -> (&Expr, Option<&str>) {
    let mut names = Vec::new();
    let mut node = expr;
    while let Expr::Alias(alias) = node {
        names.push(alias.name.as_str());
        node = &alias.expr;
    }
    let automatic = names.last().is_some_and(|name| {
        *name == "count(*)"
            || name.split_once('(').is_some_and(|(function, _)| {
                function.contains('.')
                    && ai_function_name(function).is_some()
                    && ai_function_name(function) == function_name(node)
            })
    });
    let written = if automatic {
        names.len() > 1
    } else {
        !names.is_empty()
    };
    (
        node,
        if written {
            names.first().copied()
        } else {
            None
        },
    )
}

/// A bound SELECT list.
pub(crate) struct Selected {
    pub(crate) columns: Vec<ValueExpression>,
    pub(crate) structs: Vec<ExtractStruct>,
    /// The structs whose spans the SELECT list returns.
    pub(crate) spans: Vec<String>,
    /// The output column names, in SELECT order.
    pub(crate) output: Vec<String>,
}

impl Binder {
    pub(crate) fn validate_query(
        &self,
        columns: &[ValueExpression],
        structs: &[ExtractStruct],
    ) -> Result<(), CompileError> {
        let is_alias = |name: &str| self.provider(name).is_some();
        for found in structs {
            if is_alias(&found.name) {
                return unsupported(
                    "DataFusion must assign this output name because it also names a table.",
                );
            }
        }
        for column in columns {
            let name = column_name(column, true);
            if let ValueExpression::Alias(_) = column {
                if is_alias(&name) {
                    return unsupported(
                        "DataFusion must assign this output name because it also names a table.",
                    );
                }
                if self
                    .tables
                    .iter()
                    .any(|(_, _, schema)| schema.field_with_name(&name).is_ok())
                {
                    return unsupported(
                        "DataFusion must assign this output name because it also names an input column.",
                    );
                }
            }
        }
        let mut prompts_seen: Vec<(quail_planner::physical::Prompt, String)> = Vec::new();
        for column in columns {
            let Some(call) = aliased_call(column, &[]) else {
                continue;
            };
            let name = column_name(column, true);
            if let Some((_, first)) = prompts_seen
                .iter()
                .find(|(prompt, _)| *prompt == *call.prompt)
            {
                return unsupported(&format!(
                    "DataFusion must project the repeated model result as {} and {}.",
                    repr_str(first),
                    repr_str(&name)
                ));
            }
            prompts_seen.push(((*call.prompt).clone(), name));
        }
        Ok(())
    }

    fn output_of(&self, expr: &Expr) -> Option<Expr> {
        let Expr::Column(column) = expr else {
            return None;
        };
        self.outputs
            .iter()
            .find(|(name, _)| *name == column.name)
            .map(|(_, found)| found.clone())
    }

    /// Bind a SELECT list.
    ///
    /// # Errors
    ///
    /// An item does not bind, or two output columns share a name.
    pub(crate) fn bind_select(&mut self, projection: &[Expr]) -> Result<Selected, CompileError> {
        let mut selected = Selected {
            columns: Vec::new(),
            structs: Vec::new(),
            spans: Vec::new(),
            output: Vec::new(),
        };
        let mut extracts: Vec<(String, Expr)> = Vec::new();
        let mut spans: Vec<(String, Expr)> = Vec::new();
        for item in projection {
            let (node, alias) = unalias(item);
            let node = self.output_of(node).unwrap_or_else(|| node.clone());
            let automatic = output_name(item);
            let alias = alias.or_else(|| crate::is_ai_call(&node).then_some(automatic.as_str()));
            if is_call(&node, "ai_extract_span") {
                let Some(alias) = alias else {
                    return unsupported("AI.EXTRACT_SPAN needs an AS name in SELECT.");
                };
                spans.push((alias.to_string(), node));
                continue;
            }
            if is_call(&node, "ai_extract")
                && let Some(alias) = alias
            {
                extracts.push((alias.to_string(), node.clone()));
            }
            for column in self.bind_select_item(&node, alias, &mut selected.structs)? {
                selected.output.push(column_name(&column, true));
                if !selected.columns.contains(&column) {
                    selected.columns.push(column);
                }
            }
        }
        for (alias, span) in spans {
            let paired = extracts
                .iter()
                .find(|(_, extract)| arguments(extract) == arguments(&span));
            let Some((name, _)) = paired else {
                return unsupported(&format!(
                    "The extraction for span output {alias} must run in a separate model region."
                ));
            };
            selected.spans.push((*name).to_string());
        }
        let mut unique = selected.output.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != selected.output.len() {
            return unsupported(&format!(
                "DataFusion must assign the repeated output names {}.",
                repr_list(&selected.output)
            ));
        }
        Ok(selected)
    }

    fn bind_select_item(
        &mut self,
        node: &Expr,
        alias: Option<&str>,
        structs: &mut Vec<ExtractStruct>,
    ) -> Result<Vec<ValueExpression>, CompileError> {
        let named = |expression: ValueExpression,
                     function: &str|
         -> Result<Vec<ValueExpression>, CompileError> {
            let Some(alias) = alias else {
                return unsupported(&format!("{function} needs an AS name in SELECT."));
            };
            Ok(vec![ValueExpression::Alias(Alias {
                expression: Box::new(expression),
                name: alias.to_string(),
            })])
        };
        if is_call(node, "ai_score") {
            let call = self.bind_ai_value(node)?;
            if !(1..=2).contains(&call_aliases(&call).len()) {
                return unsupported("This score must run over a materialized input relation.");
            }
            return named(ValueExpression::ModelCall(call), "AI.SCORE");
        }
        if is_call(node, "ai_classify") {
            let call = self.bind_ai_value(node)?;
            return named(ValueExpression::ModelCall(call), "AI.CLASSIFY");
        }
        if is_call(node, "ai_extract") {
            let Some(alias) = alias else {
                return unsupported("AI.EXTRACT needs an AS name in SELECT.");
            };
            let (found, members) = self.bind_ai_extract(node, alias)?;
            structs.push(found);
            return Ok(members);
        }
        if is_call(node, "ai_sae_features") {
            let call = self.bind_ai_sae_features(node)?;
            return named(ValueExpression::ModelCall(call), "AI.SAE_FEATURES");
        }
        if is_call(node, "ai_map") {
            let call = self.bind_ai_map(node)?;
            return named(ValueExpression::ModelCall(call), "AI.MAP");
        }
        if has_ai_call(node) || contains(node, |inner| column_call_kind(inner).is_some()) {
            return unsupported("A model call cannot be nested in this SELECT expression.");
        }
        let value = self.bind_value(node)?;
        Ok(vec![match alias {
            Some(alias) => ValueExpression::Alias(Alias {
                expression: Box::new(value),
                name: alias.to_string(),
            }),
            None => value,
        }])
    }
}
