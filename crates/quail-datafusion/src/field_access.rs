//! `value['field']` on a struct in the Snowflake dialect.
//!
//! sqlparser reads a bracket after a value in Snowflake SQL, such as
//! `AI.EXTRACT(c.text, {...})['ends']`, as Snowflake's path access into
//! a semi-structured value, and DataFusion's SQL planner hands it to
//! each `ExprPlanner` as the `:` operator between the value and the path
//! as text. [`StructFieldAccess`] plans it as `get_field` when the value
//! is a struct or a map, as the bracket means in DataFusion's own
//! dialect; any other value is left to the next planner.

use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DFSchema, Result, ScalarValue};
use datafusion::functions::core::expr_fn::get_field;
use datafusion::logical_expr::planner::{ExprPlanner, PlannerResult, RawBinaryExpr};
use datafusion::logical_expr::{Expr, ExprSchemable};
use sqlparser::ast::BinaryOperator;

/// Plans Snowflake's `value['field']['inner']` on a struct as nested
/// `get_field` calls.
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
        let field = keys
            .into_iter()
            .fold(expr.left, |value, key| get_field(value, key));
        Ok(PlannerResult::Planned(field))
    }
}

/// The keys of a path written as single-quoted brackets, such as
/// `['ends']['start']`, or None for any other path, such as `[0]`,
/// `["ends"]`, or `ends.start`, which the next planner reads.
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
