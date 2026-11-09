//! Rewrites of `SELECT` lists and AI function arguments.

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

/// The suffix of the column that holds an extraction's answer
/// positions, `AI.EXTRACT_SPAN`, as in Quail's `SPAN_SUFFIX`.
pub const SPAN_SUFFIX: &str = "_span";

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
