//! Regular expression tests as each SQL dialect means them.
//!
//! DataFusion's `regexp_like` finds a match anywhere in the value, as
//! BigQuery's `REGEXP_CONTAINS` does. Snowflake's `REGEXP_LIKE` and
//! `RLIKE` match the whole value. [`regexp_functions`] registers
//! `regexp_like` under the names `rlike` and `regexp_contains` too, and
//! [`SnowflakeRegexpLike`] anchors the pattern of each call at both ends
//! of the value when the session reads Snowflake SQL.
//!
//! For example, in Snowflake SQL `REGEXP_LIKE(r.lang, 'e.')` tests
//! `regexp_like(r.lang, '\A(?:e.)\z')`: it keeps `en` and rejects `eng`.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchema, Result, ScalarValue};
use datafusion::config::{ConfigOptions, Dialect};
use datafusion::functions::regex::regexp_like;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::FunctionRewrite;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator, ScalarUDF, lit};

/// What a pattern is wrapped in to match the whole value. `\A` and `\z`
/// match only at the value's ends, whatever the `m` flag says.
const ANCHOR_START: &str = "\\A(?:";
const ANCHOR_END: &str = ")\\z";

/// DataFusion's `regexp_like`, also named `rlike` and `regexp_contains`.
pub(crate) fn regexp_functions() -> Arc<ScalarUDF> {
    Arc::new(
        regexp_like()
            .as_ref()
            .clone()
            .with_aliases(["rlike", "regexp_contains"]),
    )
}

/// Anchors the pattern of each `regexp_like` call at both ends of the
/// value when the session's SQL dialect is Snowflake.
#[derive(Debug, Default)]
pub struct SnowflakeRegexpLike;

impl FunctionRewrite for SnowflakeRegexpLike {
    fn name(&self) -> &str {
        "snowflake_regexp_like"
    }

    fn rewrite(
        &self,
        expr: Expr,
        _schema: &DFSchema,
        config: &ConfigOptions,
    ) -> Result<Transformed<Expr>> {
        if !matches!(config.sql_parser.dialect, Dialect::Snowflake) {
            return Ok(Transformed::no(expr));
        }
        let Expr::ScalarFunction(ScalarFunction { func, mut args }) = expr else {
            return Ok(Transformed::no(expr));
        };
        // the analyzer may run twice: an anchored pattern stays as it is
        if func.name() != "regexp_like" || args.len() < 2 || is_anchored(&args[1]) {
            return Ok(Transformed::no(Expr::ScalarFunction(ScalarFunction {
                func,
                args,
            })));
        }
        args[1] = anchored(args[1].clone());
        Ok(Transformed::yes(Expr::ScalarFunction(ScalarFunction {
            func,
            args,
        })))
    }
}

/// `pattern` anchored at both ends of the value: a string literal
/// rewritten, any other pattern concatenated, so a null stays null.
fn anchored(pattern: Expr) -> Expr {
    if let Expr::Literal(ScalarValue::Utf8(Some(text)), _) = &pattern {
        return lit(format!("{ANCHOR_START}{text}{ANCHOR_END}"));
    }
    let concat = |left: Expr, right: Expr| {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            Operator::StringConcat,
            Box::new(right),
        ))
    };
    concat(concat(lit(ANCHOR_START), pattern), lit(ANCHOR_END))
}

/// Whether [`anchored`] wrote `pattern`.
fn is_anchored(pattern: &Expr) -> bool {
    match pattern {
        Expr::Literal(ScalarValue::Utf8(Some(text)), _) => {
            text.starts_with(ANCHOR_START) && text.ends_with(ANCHOR_END)
        }
        Expr::BinaryExpr(concat) => *concat.right == lit(ANCHOR_END),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::col;

    use super::*;

    fn rewritten(dialect: Dialect, pattern: Expr) -> String {
        let mut config = ConfigOptions::new();
        config.sql_parser.dialect = dialect;
        let call = regexp_functions().call(vec![col("a"), pattern]);
        SnowflakeRegexpLike
            .rewrite(call, &DFSchema::empty(), &config)
            .unwrap()
            .data
            .to_string()
    }

    #[test]
    fn a_snowflake_regexp_matches_the_whole_value() {
        assert_eq!(
            rewritten(Dialect::Snowflake, lit("e.")),
            r#"regexp_like(a, Utf8("\A(?:e.)\z"))"#
        );
        assert_eq!(
            rewritten(Dialect::Snowflake, col("b")),
            r#"regexp_like(a, Utf8("\A(?:") || b || Utf8(")\z"))"#
        );
    }

    #[test]
    fn another_dialects_regexp_matches_anywhere() {
        assert_eq!(
            rewritten(Dialect::BigQuery, lit("e.")),
            r#"regexp_like(a, Utf8("e."))"#
        );
    }
}
