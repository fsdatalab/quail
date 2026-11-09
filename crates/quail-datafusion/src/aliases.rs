//! Rewrites of `SELECT` lists and AI function arguments.

use std::collections::HashSet;
use std::ops::ControlFlow;

use sqlparser::ast::{
    Expr, FunctionArg, FunctionArgExpr, FunctionArgumentList, FunctionArguments, Ident, ObjectName,
    OrderByExpr, OrderByKind, Query, Select, SelectItem, SetExpr, Statement, Value,
    visit_expressions_mut,
};

use crate::functions::ai_function_name;

/// Order a `SELECT DISTINCT` by the position of each `SELECT` column its
/// `ORDER BY` repeats.
///
/// DataFusion refuses `SELECT DISTINCT r.details.kind ... ORDER BY
/// r.details.kind`: it compares the sort key with the select list after
/// naming the struct field access differently. `ORDER BY 1` names the
/// same column.
pub fn distinct_order_ordinals(statement: &mut Statement) {
    for_each_select(statement, &mut |select, order_by| {
        let (Some(_), Some(order_by)) = (&select.distinct, order_by) else {
            return;
        };
        for key in order_by.iter_mut() {
            let position = select.projection.iter().position(|item| match item {
                SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                    *expr == key.expr
                }
                _ => false,
            });
            if let Some(position) = position
                && !matches!(key.expr, Expr::Identifier(_))
            {
                key.expr = Expr::value(Value::Number((position + 1).to_string(), false));
            }
        }
    });
}

/// Write each named argument of an AI function, `name => value`, as a
/// trailing `named_struct('name', value)` argument.
///
/// DataFusion's UDFs take positional arguments only. For example,
/// `ai_sae_features(PROMPT(...), pooling => 'last')` becomes
/// `ai_sae_features(PROMPT(...), named_struct('pooling', 'last'))`,
/// which the Rust binder reads as the option.
pub fn positional_ai_arguments(statement: &mut Statement) {
    let _ = visit_expressions_mut(statement, |expr| {
        if let Expr::Function(call) = expr
            && ai_function_name(&call.name.to_string()).is_some()
        {
            let template = call.clone();
            let FunctionArguments::List(list) = &mut call.args else {
                return ControlFlow::Continue(());
            };
            for argument in &mut list.args {
                let (name, value) = match argument {
                    FunctionArg::Named {
                        name,
                        arg: FunctionArgExpr::Expr(value),
                        ..
                    } => (name.value.clone(), value.clone()),
                    FunctionArg::ExprNamed {
                        name: Expr::Identifier(name),
                        arg: FunctionArgExpr::Expr(value),
                        ..
                    } => (name.value.clone(), value.clone()),
                    _ => continue,
                };
                let mut named = template.clone();
                named.name = ObjectName::from(vec![Ident::new("named_struct")]);
                named.args = FunctionArguments::List(FunctionArgumentList {
                    duplicate_treatment: None,
                    args: vec![
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::value(
                            Value::SingleQuotedString(name),
                        ))),
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(value)),
                    ],
                    clauses: Vec::new(),
                });
                *argument = FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Function(named)));
            }
        }
        ControlFlow::<()>::Continue(())
    });
}

/// The suffix of the column that holds an extraction's answer
/// positions, as in Quail's `SPAN_SUFFIX`.
pub const SPAN_SUFFIX: &str = "_span";

/// After each `ai_extract(...) AS name` in `SELECT`, add
/// `ai_extract_span(...) AS name_span`, unless `SELECT` names it.
///
/// Quail returns each extraction's answer positions beside its
/// answers, so a query may select or test `lease_span` without
/// writing it.
pub fn add_extract_spans(statement: &mut Statement) {
    for_each_select(statement, &mut |select, _| {
        let named: HashSet<String> = select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(key(alias)),
                _ => None,
            })
            .collect();
        let mut projection = Vec::with_capacity(select.projection.len());
        for item in select.projection.drain(..) {
            let span = match &item {
                SelectItem::ExprWithAlias {
                    expr: Expr::Function(call),
                    alias,
                } if ai_function_name(&call.name.to_string()) == Some("ai_extract")
                    && !named.contains(&format!("{}{SPAN_SUFFIX}", key(alias))) =>
                {
                    let mut span_call = call.clone();
                    span_call.name = ObjectName::from(vec![Ident::new("ai_extract_span")]);
                    Some(SelectItem::ExprWithAlias {
                        expr: Expr::Function(span_call),
                        alias: Ident::new(format!("{}{SPAN_SUFFIX}", alias.value)),
                    })
                }
                _ => None,
            };
            projection.push(item);
            projection.extend(span);
        }
        select.projection = projection;
    });
}

/// Call `visit` on every `SELECT` in a statement, with the `ORDER BY`
/// of the query that holds it when that query's body is the `SELECT`.
fn for_each_select(
    statement: &mut Statement,
    visit: &mut dyn FnMut(&mut Select, Option<&mut Vec<OrderByExpr>>),
) {
    if let Statement::Query(query) = statement {
        visit_query(query, visit);
    }
}

fn visit_query(
    query: &mut Query,
    visit: &mut dyn FnMut(&mut Select, Option<&mut Vec<OrderByExpr>>),
) {
    if let Some(with) = &mut query.with {
        for table in &mut with.cte_tables {
            visit_query(&mut table.query, visit);
        }
    }
    let order_by = query
        .order_by
        .as_mut()
        .and_then(|order_by| match &mut order_by.kind {
            OrderByKind::Expressions(expressions) => Some(expressions),
            OrderByKind::All(_) => None,
        });
    match query.body.as_mut() {
        SetExpr::Select(select) => visit(select, order_by),
        body => visit_set_expr(body, visit),
    }
}

fn visit_set_expr(
    body: &mut SetExpr,
    visit: &mut dyn FnMut(&mut Select, Option<&mut Vec<OrderByExpr>>),
) {
    match body {
        SetExpr::Select(select) => visit(select, None),
        SetExpr::Query(query) => visit_query(query, visit),
        SetExpr::SetOperation { left, right, .. } => {
            visit_set_expr(left, visit);
            visit_set_expr(right, visit);
        }
        _ => {}
    }
}

/// Names compare as DataFusion compares unquoted identifiers.
fn key(name: &Ident) -> String {
    if name.quote_style.is_some() {
        name.value.clone()
    } else {
        name.value.to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use sqlparser::dialect::SnowflakeDialect;
    use sqlparser::parser::Parser;

    use super::*;

    #[test]
    fn an_extraction_gets_its_answer_positions_column() {
        let mut statements = Parser::parse_sql(
            &SnowflakeDialect {},
            "SELECT ai_extract(c.text, {'end': 'When?'}) AS lease FROM contracts c",
        )
        .unwrap();
        add_extract_spans(&mut statements[0]);
        assert_eq!(
            statements[0].to_string(),
            "SELECT ai_extract(c.text, {'end': 'When?'}) AS lease, \
             ai_extract_span(c.text, {'end': 'When?'}) AS lease_span FROM contracts c"
        );
    }
}
