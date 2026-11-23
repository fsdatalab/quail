use datafusion::common::Column;
use datafusion::logical_expr::expr::BinaryExpr;
use datafusion::logical_expr::{Expr, Operator};

pub(super) fn is_count_star(expr: &Expr) -> bool {
    let expr = match expr {
        Expr::Alias(alias) => alias.expr.as_ref(),
        other => other,
    };
    let Expr::AggregateFunction(function) = expr else {
        return false;
    };
    let params = &function.params;
    function.func.name() == "count"
        && !params.distinct
        && params.filter.is_none()
        && params.order_by.is_empty()
        && matches!(params.args.as_slice(), [Expr::Literal(value, _)] if !value.is_null())
}

pub(super) fn at_least(predicate: &Expr, count: &Column) -> Option<usize> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = predicate else {
        return None;
    };
    let is_count = |expr: &Expr| matches!(expr, Expr::Column(column) if column == count);
    let integer = |expr: &Expr| match expr {
        Expr::Literal(value, _) => value.to_string().parse::<i64>().ok(),
        _ => None,
    };
    let bound = if is_count(left) {
        match op {
            Operator::GtEq => integer(right)?,
            Operator::Gt => integer(right)? + 1,
            _ => return None,
        }
    } else if is_count(right) {
        match op {
            Operator::LtEq => integer(left)?,
            Operator::Lt => integer(left)? + 1,
            _ => return None,
        }
    } else {
        return None;
    };
    usize::try_from(bound).ok().filter(|bound| *bound > 0)
}
