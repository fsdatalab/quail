//! Print a model region as SQL that Quail's SQL front end reads.

use std::ops::ControlFlow;

use datafusion::arrow::array::{Array, ListArray, StructArray};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::sql::sqlparser::ast;
use datafusion::sql::unparser::Unparser;
use datafusion::sql::unparser::dialect::Dialect;
use sqlparser::ast::visit_expressions_mut;
use sqlparser::dialect::SnowflakeDialect;
use sqlparser::parser::Parser;

use crate::aliases::{remove_select_calls, restore_select_aliases};
use crate::region::ModelRegion;

/// How an extraction's answer positions print before `region_sql`
/// removes them; Quail's front end adds the column itself.
const EXTRACT_SPAN: &str = "AI_EXTRACT_SPAN";

/// Quail's name for each function, by its DataFusion name.
const QUAIL_NAMES: [(&str, &str); 9] = [
    ("ai_filter", "AI_FILTER"),
    ("ai_score", "AI.SCORE"),
    ("ai_classify", "AI.CLASSIFY"),
    ("ai_extract", "AI.EXTRACT"),
    ("ai_extract_span", EXTRACT_SPAN),
    ("ai_map", "AI.MAP"),
    ("ai_sae_features", "AI.SAE_FEATURES"),
    ("prompt", "PROMPT"),
    ("hint", "HINT"),
];

/// The unparser dialect for Quail's SQL front end (Snowflake syntax).
struct QuailDialect;

impl Dialect for QuailDialect {
    fn identifier_quote_style(&self, identifier: &str) -> Option<char> {
        let mut chars = identifier.chars();
        let simple = chars
            .next()
            .is_some_and(|first| first.is_ascii_lowercase() || first == '_')
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if simple { None } else { Some('"') }
    }

    fn scalar_function_to_sql_overrides(
        &self,
        unparser: &Unparser,
        func_name: &str,
        args: &[Expr],
    ) -> Result<Option<ast::Expr>> {
        if func_name == "get_field" {
            return field_access_sql(unparser, args);
        }
        let Some((_, quail_name)) = QUAIL_NAMES.iter().find(|(name, _)| *name == func_name) else {
            return Ok(None);
        };
        let arguments = args
            .iter()
            .map(|arg| argument_sql(unparser, arg))
            .collect::<Result<Vec<_>>>()?;
        parse_expr(&format!("{quail_name}({})", arguments.join(", "))).map(Some)
    }
}

/// Print `get_field(root, 'a', 'b')` as `root['a']['b']` when the root
/// is a Quail function, so its answer prints as Quail's front end
/// reads it; any other root prints as DataFusion prints it.
///
/// Nested accesses, `get_field(get_field(root, 'a'), 'b')`, print as one chain.
fn field_access_sql(unparser: &Unparser, args: &[Expr]) -> Result<Option<ast::Expr>> {
    let Some((mut root, fields)) = args.split_first() else {
        return Ok(None);
    };
    let mut chain: Vec<&Expr> = fields.iter().rev().collect();
    while let Expr::ScalarFunction(inner) = root
        && inner.name() == "get_field"
        && let Some((inner_root, inner_fields)) = inner.args.split_first()
    {
        chain.extend(inner_fields.iter().rev());
        root = inner_root;
    }
    let Expr::ScalarFunction(call) = root else {
        return Ok(None);
    };
    if !QUAIL_NAMES.iter().any(|(name, _)| *name == call.name()) {
        return Ok(None);
    }
    let mut access_chain = Vec::with_capacity(chain.len());
    for field in chain.into_iter().rev() {
        access_chain.push(ast::AccessExpr::Subscript(ast::Subscript::Index {
            index: parse_expr(&argument_sql(unparser, field)?)?,
        }));
    }
    Ok(Some(ast::Expr::CompoundFieldAccess {
        root: Box::new(parse_expr(&argument_sql(unparser, root)?)?),
        access_chain,
    }))
}

/// Print one argument of a Quail function.
///
/// Literal structs and lists print in Quail's syntax: a struct with
/// named fields as `{'name': value}`, a struct with positional fields
/// (`c0`, `c1`, ...) as a tuple, and a list as `[a, b]`. The
/// constructors DataFusion uses before folding them print the same.
fn argument_sql(unparser: &Unparser, arg: &Expr) -> Result<String> {
    match arg {
        Expr::Literal(value, _) => literal_sql(value),
        Expr::ScalarFunction(call) => {
            let parts = call
                .args
                .iter()
                .map(|inner| argument_sql(unparser, inner))
                .collect::<Result<Vec<_>>>()?;
            match call.name() {
                "make_array" => Ok(format!("[{}]", parts.join(", "))),
                "struct" => Ok(format!("({})", parts.join(", "))),
                "named_struct" => Ok(format!(
                    "{{{}}}",
                    parts
                        .chunks(2)
                        .map(|pair| pair.join(": "))
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
                _ => Ok(unparser.expr_to_sql(arg)?.to_string()),
            }
        }
        _ => Ok(unparser.expr_to_sql(arg)?.to_string()),
    }
}

/// Print a literal in Quail's syntax.
fn literal_sql(value: &ScalarValue) -> Result<String> {
    match value {
        ScalarValue::Utf8(Some(text))
        | ScalarValue::LargeUtf8(Some(text))
        | ScalarValue::Utf8View(Some(text)) => {
            Ok(ast::Value::SingleQuotedString(text.clone()).to_string())
        }
        ScalarValue::Struct(array) => struct_sql(array),
        ScalarValue::List(array) => list_sql(array),
        other => Ok(other.to_string()),
    }
}

fn struct_sql(array: &StructArray) -> Result<String> {
    let names: Vec<&str> = array
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    let values = (0..array.num_columns())
        .map(|index| literal_sql(&ScalarValue::try_from_array(array.column(index), 0)?))
        .collect::<Result<Vec<_>>>()?;
    let positional = names
        .iter()
        .enumerate()
        .all(|(index, name)| *name == format!("c{index}"));
    if positional {
        return Ok(format!("({})", values.join(", ")));
    }
    let pairs: Vec<String> = names
        .iter()
        .zip(values)
        .map(|(name, value)| {
            format!(
                "{}: {value}",
                ast::Value::SingleQuotedString(name.to_string())
            )
        })
        .collect();
    Ok(format!("{{{}}}", pairs.join(", ")))
}

fn list_sql(array: &ListArray) -> Result<String> {
    let values = array.value(0);
    let items = (0..values.len())
        .map(|index| literal_sql(&ScalarValue::try_from_array(&values, index)?))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("[{}]", items.join(", ")))
}

fn parse_expr(text: &str) -> Result<ast::Expr> {
    Parser::new(&SnowflakeDialect {})
        .try_with_sql(text)
        .and_then(|mut parser| parser.parse_expr())
        .map_err(|error| DataFusionError::Plan(format!("cannot print {text}: {error}")))
}

/// Print a model region as one SQL statement for Quail's front end.
///
/// AI calls print with Quail's names, such as `AI.CLASSIFY`, and a
/// `WHERE` call that equals an aliased `SELECT` call prints as the
/// alias, as Quail's front end expects. Each input the region reads
/// prints as the query DataFusion computes it with.
pub fn region_sql(region: &ModelRegion) -> Result<String> {
    let dialect = QuailDialect;
    let bound = region
        .bound
        .clone()
        .transform_up(|node| {
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            let name = scan.table_name.table();
            let Some(index) = region.input_names.iter().position(|found| found == name) else {
                return Ok(Transformed::no(node));
            };
            let input = &region.inputs[index];
            // an input holding another model region keeps its name
            let regions = input.exists(|inner| {
                Ok(matches!(inner, LogicalPlan::Extension(extension)
                    if extension.node.as_any().downcast_ref::<ModelRegion>().is_some()))
            })?;
            if regions {
                return Ok(Transformed::no(node));
            }
            Ok(Transformed::yes(crate::claim::lift_filters_above_aliases(
                input,
            )?))
        })?
        .data;
    let mut statement = Unparser::new(&dialect).plan_to_sql(&bound)?;
    restore_select_aliases(&mut statement);
    remove_select_calls(&mut statement, EXTRACT_SPAN);
    let count_all = parse_expr("COUNT(*)")?;
    let _ = visit_expressions_mut(&mut statement, |expr| {
        if is_count_one(expr) {
            *expr = count_all.clone();
        }
        ControlFlow::<()>::Continue(())
    });
    Ok(statement.to_string())
}

/// Whether an expression is `count(1)`, DataFusion's form of `COUNT(*)`.
fn is_count_one(expr: &ast::Expr) -> bool {
    let ast::Expr::Function(function) = expr else {
        return false;
    };
    let ast::FunctionArguments::List(arguments) = &function.args else {
        return false;
    };
    function.name.to_string().eq_ignore_ascii_case("count")
        && matches!(
            arguments.args.as_slice(),
            [ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(ast::Expr::Value(value)))]
                if value.value == ast::Value::Number("1".to_string(), false)
        )
}
