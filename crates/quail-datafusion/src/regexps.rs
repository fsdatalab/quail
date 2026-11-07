//! Regular expression tests as each SQL dialect means them.
//!
//! DataFusion's `regexp_like` finds a match anywhere in the value. In
//! Snowflake, `REGEXP_LIKE`, `RLIKE`, and the `RLIKE` and `REGEXP`
//! operators match the whole value, so [`anchor_snowflake_regexps`]
//! anchors each pattern at both ends. BigQuery's `REGEXP_CONTAINS` finds
//! a match anywhere, as `regexp_like` does, so
//! [`bigquery_regexp_contains`] only renames it.
//!
//! For example, in Snowflake SQL `REGEXP_LIKE(r.lang, 'e.')` becomes
//! `regexp_like(r.lang, '\A(?:e.)\z')`: it keeps `en` and rejects `eng`.

use std::ops::ControlFlow;

use sqlparser::ast::{
    BinaryOperator, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, Ident, ObjectName, Statement, UnaryOperator, Value, visit_expressions_mut,
};

/// What a pattern is wrapped in to match the whole value. `\A` and `\z`
/// match only at the value's ends, whatever the `m` flag says.
const ANCHOR_START: &str = "\\A(?:";
const ANCHOR_END: &str = ")\\z";

/// Whether `function` is named `name`, unqualified, in any case.
fn named(function: &Function, name: &str) -> bool {
    matches!(&function.name.0[..], [part] if part.to_string().eq_ignore_ascii_case(name))
}

/// The second argument of a call written with unnamed arguments.
fn pattern_argument(function: &mut Function) -> Option<&mut Expr> {
    let FunctionArguments::List(list) = &mut function.args else {
        return None;
    };
    match list.args.get_mut(1) {
        Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(pattern))) => Some(pattern),
        _ => None,
    }
}

/// `pattern` anchored at both ends of the value: a string literal
/// rewritten, any other pattern concatenated, so a null stays null.
fn anchored(pattern: Expr) -> Expr {
    if let Expr::Value(value) = &pattern
        && let Value::SingleQuotedString(text) = &value.value
    {
        return Expr::value(Value::SingleQuotedString(format!(
            "{ANCHOR_START}{text}{ANCHOR_END}"
        )));
    }
    let concat = |left: Expr, right: Expr| Expr::BinaryOp {
        left: Box::new(left),
        op: BinaryOperator::StringConcat,
        right: Box::new(right),
    };
    Expr::Nested(Box::new(concat(
        concat(
            Expr::value(Value::SingleQuotedString(ANCHOR_START.to_string())),
            pattern,
        ),
        Expr::value(Value::SingleQuotedString(ANCHOR_END.to_string())),
    )))
}

/// The call `regexp_like(value, pattern)`.
fn regexp_like(value: Expr, pattern: Expr) -> Expr {
    let argument = |expr: Expr| FunctionArg::Unnamed(FunctionArgExpr::Expr(expr));
    Expr::Function(Function {
        name: ObjectName::from(vec![Ident::new("regexp_like")]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![argument(value), argument(pattern)],
            clauses: Vec::new(),
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: Vec::new(),
    })
}

/// Anchor the pattern of each `REGEXP_LIKE` and `RLIKE` call and each
/// `RLIKE` and `REGEXP` test, which match the whole value in Snowflake,
/// and write each as DataFusion's `regexp_like`.
pub fn anchor_snowflake_regexps(statement: &mut Statement) {
    let _ = visit_expressions_mut(statement, |expr| {
        match expr {
            Expr::Function(function)
                if named(function, "regexp_like") || named(function, "rlike") =>
            {
                if let Some(pattern) = pattern_argument(function) {
                    *pattern = anchored(pattern.clone());
                    function.name = ObjectName::from(vec![Ident::new("regexp_like")]);
                }
            }
            Expr::RLike {
                negated,
                expr: value,
                pattern,
                ..
            } => {
                let test = regexp_like(value.as_ref().clone(), anchored(pattern.as_ref().clone()));
                *expr = if *negated {
                    Expr::UnaryOp {
                        op: UnaryOperator::Not,
                        expr: Box::new(test),
                    }
                } else {
                    test
                };
            }
            _ => {}
        }
        ControlFlow::<()>::Continue(())
    });
}

/// Write each `REGEXP_CONTAINS` call, which finds a match anywhere in
/// BigQuery, as DataFusion's `regexp_like`, which does too.
pub fn bigquery_regexp_contains(statement: &mut Statement) {
    let _ = visit_expressions_mut(statement, |expr| {
        if let Expr::Function(function) = expr
            && named(function, "regexp_contains")
        {
            function.name = ObjectName::from(vec![Ident::new("regexp_like")]);
        }
        ControlFlow::<()>::Continue(())
    });
}

#[cfg(test)]
mod tests {
    use datafusion::sql::parser::DFParserBuilder;
    use sqlparser::dialect::{BigQueryDialect, SnowflakeDialect};

    use super::*;

    fn rewritten(sql: &str, snowflake: bool) -> String {
        let dialect: Box<dyn sqlparser::dialect::Dialect> = if snowflake {
            Box::new(SnowflakeDialect {})
        } else {
            Box::new(BigQueryDialect {})
        };
        let mut statements = DFParserBuilder::new(sql)
            .with_dialect(dialect.as_ref())
            .build()
            .unwrap()
            .parse_statements()
            .unwrap();
        let datafusion::sql::parser::Statement::Statement(statement) =
            statements.front_mut().unwrap()
        else {
            panic!("a SQL statement")
        };
        if snowflake {
            anchor_snowflake_regexps(statement);
        } else {
            bigquery_regexp_contains(statement);
        }
        statement.to_string()
    }

    #[test]
    fn a_snowflake_regexp_matches_the_whole_value() {
        assert_eq!(
            rewritten("SELECT a FROM t WHERE REGEXP_LIKE(a, 'e.', 'i')", true),
            r"SELECT a FROM t WHERE regexp_like(a, '\A(?:e.)\z', 'i')"
        );
        assert_eq!(
            rewritten("SELECT a FROM t WHERE a NOT RLIKE b", true),
            r"SELECT a FROM t WHERE NOT regexp_like(a, ('\A(?:' || b || ')\z'))"
        );
    }

    #[test]
    fn a_bigquery_regexp_contains_matches_anywhere() {
        assert_eq!(
            rewritten("SELECT a FROM t WHERE REGEXP_CONTAINS(a, 'e')", false),
            "SELECT a FROM t WHERE regexp_like(a, 'e')"
        );
    }
}
