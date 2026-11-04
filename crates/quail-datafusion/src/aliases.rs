//! Move between `SELECT` aliases and the calls they name.

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use sqlparser::ast::{
    AccessExpr, Expr, FunctionArg, FunctionArgExpr, FunctionArgumentList, FunctionArguments,
    GroupByExpr, Ident, ObjectName, OrderByExpr, OrderByKind, Query, Select, SelectItem, SetExpr,
    Statement, Subscript, TableFactor, Value, visit_expressions_mut,
};

/// Replace each `SELECT` alias named in a `WHERE` clause with the
/// expression it names.
///
/// For example, with `AI.CLASSIFY(...) AS sentiment` in `SELECT`,
/// `WHERE sentiment IN ('negative')` becomes `WHERE AI.CLASSIFY(...)
/// IN ('negative')`, and `WHERE lease.end_date LIKE '%2027%'` becomes
/// `WHERE AI.EXTRACT(...)['end_date'] LIKE '%2027%'`. Quail's planner
/// reads a `WHERE` call equal to a `SELECT` call as that call's
/// answer, so the model runs once. A name that is also a table alias
/// in `FROM` is left alone.
pub fn inline_select_aliases(statement: &mut Statement) {
    for_each_select(statement, &mut |select, _| inline_in_select(select));
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
            && call
                .name
                .to_string()
                .to_ascii_lowercase()
                .starts_with("ai_")
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
                } if call.name.to_string().eq_ignore_ascii_case("ai_extract")
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

/// Remove each `SELECT` item that calls `function`.
pub fn remove_select_calls(statement: &mut Statement, function: &str) {
    for_each_select(statement, &mut |select, _| {
        select.projection.retain(|item| {
            let expr = match item {
                SelectItem::ExprWithAlias { expr, .. } | SelectItem::UnnamedExpr(expr) => expr,
                _ => return true,
            };
            !matches!(expr, Expr::Function(call) if call.name.to_string() == function)
        });
    });
}

/// Replace each `WHERE`, `GROUP BY`, `HAVING`, or `ORDER BY`
/// expression that equals an aliased `SELECT` call with the alias.
///
/// This undoes [`inline_select_aliases`] for Quail's SQL front end,
/// which finds an AI call's answer by its alias: `WHERE
/// AI.CLASSIFY(...) IN ('negative')` becomes `WHERE sentiment IN
/// ('negative')`, and `AI.EXTRACT(...)['end_date']` becomes
/// `lease.end_date`.
pub fn restore_select_aliases(statement: &mut Statement) {
    for_each_select(statement, &mut |select, order_by| {
        let aliases: Vec<(Expr, Ident)> = select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias {
                    expr: expr @ Expr::Function(_),
                    alias,
                } => Some((expr.clone(), alias.clone())),
                _ => None,
            })
            .collect();
        if aliases.is_empty() {
            return;
        }
        let mut targets: Vec<&mut Expr> = Vec::new();
        targets.extend(select.selection.as_mut());
        targets.extend(select.having.as_mut());
        if let GroupByExpr::Expressions(expressions, _) = &mut select.group_by {
            targets.extend(expressions.iter_mut());
        }
        targets.extend(order_by.into_iter().flatten().map(|item| &mut item.expr));
        for target in targets {
            let _ = visit_expressions_mut(target, |expr| {
                if let Some(replacement) = alias_for(expr, &aliases) {
                    *expr = replacement;
                }
                ControlFlow::<()>::Continue(())
            });
        }
    });
}

/// The alias reference for an expression that is an aliased call or
/// a field of one.
fn alias_for(expr: &Expr, aliases: &[(Expr, Ident)]) -> Option<Expr> {
    if let Some((_, alias)) = aliases.iter().find(|(call, _)| call == expr) {
        return Some(Expr::Identifier(alias.clone()));
    }
    let Expr::CompoundFieldAccess { root, access_chain } = expr else {
        return None;
    };
    let Expr::Identifier(alias) = root.as_ref() else {
        return None;
    };
    let mut parts = vec![alias.clone()];
    for access in access_chain {
        let AccessExpr::Subscript(Subscript::Index {
            index: Expr::Value(value),
        }) = access
        else {
            return None;
        };
        let Value::SingleQuotedString(field) = &value.value else {
            return None;
        };
        parts.push(Ident::new(field.clone()));
    }
    Some(Expr::CompoundIdentifier(parts))
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

fn inline_in_select(select: &mut Select) {
    let aliases: HashMap<String, Expr> = select
        .projection
        .iter()
        .filter_map(|item| match item {
            SelectItem::ExprWithAlias { expr, alias } => Some((key(alias), expr.clone())),
            _ => None,
        })
        .collect();
    if aliases.is_empty() {
        return;
    }
    let tables = table_names(select);
    // a window function in SELECT may read another SELECT column by name
    for item in &mut select.projection {
        let own = match item {
            SelectItem::ExprWithAlias { alias, .. } => Some(key(alias)),
            _ => None,
        };
        let others: HashMap<String, Expr> = aliases
            .iter()
            .filter(|(name, _)| Some(*name) != own.as_ref())
            .map(|(name, expr)| (name.clone(), expr.clone()))
            .collect();
        let expr = match item {
            SelectItem::ExprWithAlias { expr, .. } | SelectItem::UnnamedExpr(expr) => expr,
            _ => continue,
        };
        let _ = visit_expressions_mut(expr, |inner| {
            if let Expr::Function(call) = inner
                && call.over.is_some()
            {
                let _ = visit_expressions_mut(call, |reference| {
                    if let Some(replacement) = resolve(reference, &others, &tables) {
                        *reference = replacement;
                    }
                    ControlFlow::<()>::Continue(())
                });
            }
            ControlFlow::<()>::Continue(())
        });
    }
    let Some(selection) = &mut select.selection else {
        return;
    };
    let _ = visit_expressions_mut(selection, |expr| {
        if let Some(replacement) = resolve(expr, &aliases, &tables) {
            *expr = replacement;
        }
        ControlFlow::<()>::Continue(())
    });
}

/// The expression an alias reference stands for, if `expr` is one.
fn resolve(expr: &Expr, aliases: &HashMap<String, Expr>, tables: &HashSet<String>) -> Option<Expr> {
    match expr {
        Expr::Identifier(name) => aliases.get(&key(name)).cloned(),
        Expr::CompoundIdentifier(parts) => {
            let (head, fields) = parts.split_first()?;
            if tables.contains(&key(head)) {
                return None;
            }
            let root = aliases.get(&key(head))?.clone();
            let access_chain = fields
                .iter()
                .map(|field| {
                    AccessExpr::Subscript(Subscript::Index {
                        index: Expr::value(Value::SingleQuotedString(field.value.clone())),
                    })
                })
                .collect();
            Some(Expr::CompoundFieldAccess {
                root: Box::new(root),
                access_chain,
            })
        }
        _ => None,
    }
}

/// The names `FROM` gives its tables: each alias, or the table name.
fn table_names(select: &Select) -> HashSet<String> {
    let mut names = HashSet::new();
    let factors = select.from.iter().flat_map(|table| {
        std::iter::once(&table.relation).chain(table.joins.iter().map(|join| &join.relation))
    });
    for factor in factors {
        if let TableFactor::Table { name, alias, .. } = factor {
            match alias {
                Some(alias) => names.insert(key(&alias.name)),
                None => names.insert(name.to_string().to_lowercase()),
            };
        }
    }
    names
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

    fn inlined(sql: &str) -> String {
        let mut statements = Parser::parse_sql(&SnowflakeDialect {}, sql).unwrap();
        inline_select_aliases(&mut statements[0]);
        statements[0].to_string()
    }

    #[test]
    fn a_label_alias_in_where_becomes_its_call() {
        let sql = "SELECT ai_classify(r.body, ['a', 'b']) AS label FROM reviews r \
            WHERE label IN ('a')";
        assert_eq!(
            inlined(sql),
            "SELECT ai_classify(r.body, ['a', 'b']) AS label FROM reviews r \
             WHERE ai_classify(r.body, ['a', 'b']) IN ('a')"
        );
    }

    #[test]
    fn an_answer_field_in_where_becomes_a_field_of_the_call() {
        let sql = "SELECT ai_extract(c.text, {'end': 'When?'}) AS lease FROM contracts c \
            WHERE lease.end LIKE '%2027%'";
        assert!(
            inlined(sql)
                .ends_with("WHERE ai_extract(c.text, {'end': 'When?'})['end'] LIKE '%2027%'"),
            "{}",
            inlined(sql)
        );
    }

    #[test]
    fn a_table_alias_that_matches_a_select_alias_is_kept() {
        let sql = "SELECT r.body AS r FROM reviews r WHERE r.id = 'x'";
        assert!(
            inlined(sql).ends_with("WHERE r.id = 'x'"),
            "{}",
            inlined(sql)
        );
    }

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
