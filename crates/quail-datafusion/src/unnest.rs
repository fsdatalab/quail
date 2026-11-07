//! `UNNEST` in `FROM`, written so DataFusion can bind the query.
//!
//! `FROM reviews r, UNNEST(AI.SAE_FEATURES(...)) AS f` names one list
//! `f`, whose elements the query reads as `f.feature` and `f.strength`;
//! `UNNEST(a, b) AS u(x, y)` zips two lists named `x` and `y`. DataFusion
//! cannot unnest a list another table in the same `FROM` computes, but a
//! model region runs in Quail, so DataFusion only needs the query's
//! types. [`take_unnests`] moves each `UNNEST` out of `FROM`: a
//! `quail_unnest(...)` test in `WHERE` records the lists, and
//! `quail_unnest_field(<list>, 'f', 'feature')` stands for each field
//! the query reads. When the region is claimed, both become DataFusion's
//! `Unnest` of the list column the region returns, and `get_field` of
//! its elements, above the region (see `lift.rs`).

use std::ops::ControlFlow;

use sqlparser::ast::{
    BinaryOperator, Expr, FunctionArg, FunctionArgExpr, FunctionArgumentList, FunctionArguments,
    Ident, JoinConstraint, JoinOperator, ObjectName, Select, Statement, TableFactor, Value,
    visit_expressions_mut,
};

/// The `WHERE` test that records one `UNNEST`'s lists:
/// `quail_unnest(<outer>, '<name>', <list>, ...)`.
pub const UNNEST_TEST: &str = "quail_unnest";
/// A field of an unnested list's element:
/// `quail_unnest_field(<list>, '<name>', '<field>')`.
pub const UNNEST_FIELD: &str = "quail_unnest_field";

/// One `UNNEST` taken out of `FROM`.
struct Taken {
    /// The table alias of `UNNEST(a, b) AS u(x, y)`, if it names columns.
    table: Option<String>,
    names: Vec<String>,
    lists: Vec<Expr>,
    outer: bool,
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(sqlparser::ast::Function {
        name: ObjectName::from(vec![Ident::new(name)]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|arg| FunctionArg::Unnamed(FunctionArgExpr::Expr(arg)))
                .collect(),
            clauses: Vec::new(),
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: Vec::new(),
    })
}

fn text(value: &str) -> Expr {
    Expr::value(Value::SingleQuotedString(value.to_string()))
}

/// The lists and alias of an `UNNEST` table factor, if it is one.
fn unnest_of(factor: &TableFactor) -> Option<(Vec<Expr>, Option<sqlparser::ast::TableAlias>)> {
    match factor {
        TableFactor::UNNEST {
            alias,
            array_exprs,
            with_offset: false,
            with_ordinality: false,
            ..
        } => Some((array_exprs.clone(), alias.clone())),
        TableFactor::Table {
            name,
            alias,
            args: Some(args),
            ..
        } if name.to_string().eq_ignore_ascii_case("unnest") => {
            let lists = args
                .args
                .iter()
                .map(|arg| match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Some(expr.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<Expr>>>()?;
            Some((lists, alias.clone()))
        }
        _ => None,
    }
}

fn taken(
    lists: Vec<Expr>,
    alias: Option<sqlparser::ast::TableAlias>,
    outer: bool,
) -> Option<Taken> {
    let alias = alias?;
    let (table, names) = if alias.columns.is_empty() {
        (None, vec![alias.name.value.clone()])
    } else {
        (
            Some(alias.name.value.clone()),
            alias
                .columns
                .iter()
                .map(|column| column.name.value.clone())
                .collect(),
        )
    };
    (names.len() == lists.len()).then_some(Taken {
        table,
        names,
        lists,
        outer,
    })
}

fn take_from_select(select: &mut Select) -> Vec<Taken> {
    let mut found = Vec::new();
    // a comma join: its own FROM item
    let mut kept = Vec::new();
    for (position, item) in select.from.drain(..).enumerate() {
        if position > 0
            && item.joins.is_empty()
            && let Some((lists, alias)) = unnest_of(&item.relation)
            && let Some(unnest) = taken(lists, alias, false)
        {
            found.push(unnest);
            continue;
        }
        kept.push(item);
    }
    select.from = kept;
    for item in &mut select.from {
        let mut joins = Vec::new();
        for join in item.joins.drain(..) {
            let outer = match &join.join_operator {
                JoinOperator::CrossJoin(JoinConstraint::None) => Some(false),
                JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
                    match constraint {
                        JoinConstraint::None => Some(true),
                        JoinConstraint::On(Expr::Value(value))
                            if value.value == Value::Boolean(true) =>
                        {
                            Some(true)
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(outer) = outer
                && let Some((lists, alias)) = unnest_of(&join.relation)
                && let Some(unnest) = taken(lists, alias, outer)
            {
                found.push(unnest);
                continue;
            }
            joins.push(join);
        }
        item.joins = joins;
    }
    found
}

/// The unnested list and field a reference names, if it names one.
fn field_of(parts: &[Ident], unnests: &[Taken]) -> Option<Expr> {
    for unnest in unnests {
        for (name, list) in unnest.names.iter().zip(&unnest.lists) {
            let field = match parts {
                [first, field] if first.value == *name => field,
                [table, first, field]
                    if Some(&table.value) == unnest.table.as_ref() && first.value == *name =>
                {
                    field
                }
                _ => continue,
            };
            return Some(call(
                UNNEST_FIELD,
                vec![list.clone(), text(name), text(&field.value)],
            ));
        }
    }
    None
}

/// Move each `UNNEST` of `FROM` into a `WHERE` test and its field
/// references into calls DataFusion types, as the module explains.
pub fn take_unnests(statement: &mut Statement) {
    let Statement::Query(query) = statement else {
        return;
    };
    let sqlparser::ast::SetExpr::Select(select) = query.body.as_mut() else {
        return;
    };
    let unnests = take_from_select(select);
    if unnests.is_empty() {
        return;
    }
    let rewrite = |expr: &mut Expr| {
        if let Expr::CompoundIdentifier(parts) = expr
            && let Some(field) = field_of(parts, &unnests)
        {
            *expr = field;
        }
        ControlFlow::<()>::Continue(())
    };
    let _ = visit_expressions_mut(select.as_mut(), rewrite);
    if let Some(order_by) = &mut query.order_by {
        let _ = visit_expressions_mut(order_by, rewrite);
    }
    for unnest in &unnests {
        let mut args = vec![Expr::value(Value::Boolean(unnest.outer))];
        for (name, list) in unnest.names.iter().zip(&unnest.lists) {
            args.push(text(name));
            args.push(list.clone());
        }
        let test = call(UNNEST_TEST, args);
        select.selection = Some(match select.selection.take() {
            None => test,
            Some(condition) => Expr::BinaryOp {
                left: Box::new(condition),
                op: BinaryOperator::And,
                right: Box::new(test),
            },
        });
    }
}
