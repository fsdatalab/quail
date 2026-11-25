use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DFSchema, Result, ScalarValue};
use datafusion::functions::core::expr_fn::get_field;
use datafusion::logical_expr::planner::{ExprPlanner, PlannerResult, RawBinaryExpr};
use datafusion::logical_expr::{Expr, ExprSchemable};
use sqlparser::ast::BinaryOperator;

#[derive(Debug, Default)]
pub struct StructFieldAccess;

impl ExprPlanner for StructFieldAccess {
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        schema: &DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>> {
        let path = match (&expr.op, &expr.right) {
            (BinaryOperator::Custom(op), Expr::Literal(ScalarValue::Utf8(Some(path)), _))
                if op == ":" =>
            {
                path_keys(path)
            }
            _ => None,
        };
        let Some(keys) = path else {
            return Ok(PlannerResult::Original(expr));
        };
        if !matches!(
            expr.left.get_type(schema)?,
            DataType::Struct(_) | DataType::Map(..)
        ) {
            return Ok(PlannerResult::Original(expr));
        }
        let field = keys.into_iter().fold(expr.left, get_field);
        Ok(PlannerResult::Planned(field))
    }
}

fn path_keys(path: &str) -> Option<Vec<String>> {
    let mut keys = Vec::new();
    let mut rest = path;
    while !rest.is_empty() {
        let inner = rest.strip_prefix("['")?;
        let end = inner.find('\'')?;
        let key = &inner[..end];
        rest = inner[end + 1..].strip_prefix(']')?;
        keys.push(key.to_string());
    }
    (!keys.is_empty()).then_some(keys)
}

#[cfg(test)]
mod tests {
    use super::path_keys;

    #[test]
    fn a_path_names_each_field_in_order() {
        assert_eq!(path_keys("['ends']"), Some(vec!["ends".into()]));
        assert_eq!(
            path_keys("['ends']['start']"),
            Some(vec!["ends".into(), "start".into()])
        );
        assert_eq!(path_keys("ends.start"), None);
        assert_eq!(path_keys("[\"ends\"]"), None);
        assert_eq!(path_keys("[0]"), None);
        assert_eq!(path_keys("['it''s']"), None);
    }
}
