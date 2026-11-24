use datafusion::logical_expr::Expr;

pub(crate) fn precondition_selectivity(
    precondition: &Expr,
    answers: &[(String, Option<f64>)],
) -> f64 {
    use datafusion::logical_expr::{BinaryExpr, Operator};
    use quail_planner::operators::effective_selectivity;
    match precondition {
        Expr::IsNotTrue(inner) | Expr::Not(inner) => 1.0 - precondition_selectivity(inner, answers),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::And => {
            precondition_selectivity(left, answers) * precondition_selectivity(right, answers)
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::Or => {
            1.0 - (1.0 - precondition_selectivity(left, answers))
                * (1.0 - precondition_selectivity(right, answers))
        }
        Expr::Column(column) if column.relation.is_none() => effective_selectivity(
            answers
                .iter()
                .find(|(name, _)| *name == column.name)
                .and_then(|(_, selectivity)| *selectivity),
        ),
        Expr::ScalarFunction(call) if call.name() == "hint" => {
            effective_selectivity(call.args.get(1).and_then(hinted_selectivity))
        }
        _ => effective_selectivity(None),
    }
}

fn hinted_selectivity(options: &Expr) -> Option<f64> {
    let Expr::ScalarFunction(call) = options else {
        return None;
    };
    if call.name() != "named_struct" {
        return None;
    }
    call.args.chunks(2).find_map(|pair| match pair {
        [Expr::Literal(key, _), Expr::Literal(value, _)]
            if key.try_as_str().flatten() == Some("selectivity") =>
        {
            value.to_string().parse().ok()
        }
        _ => None,
    })
}
