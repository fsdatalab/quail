use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::expr::ScalarFunction;
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::repr_str;

use super::function_name;

#[derive(Default, Clone)]
pub(crate) struct Options {
    pub(crate) selectivity: Option<f64>,
    pub(crate) anchor: Option<String>,
}

pub(crate) fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(value, _) => value.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

fn numeric_literal(expr: &Expr, to: &DataType) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) if value.data_type().is_numeric() => value.cast_to(to).ok(),
        Expr::Negative(inner) => numeric_literal(inner, to)?.arithmetic_negate().ok(),
        _ => None,
    }
}

pub(crate) fn number_of(expr: &Expr) -> Option<f64> {
    match numeric_literal(expr, &DataType::Float64)? {
        ScalarValue::Float64(value) => value,
        _ => None,
    }
}

pub(crate) fn object_fields(expr: &Expr) -> Option<Vec<(String, &Expr)>> {
    let Expr::ScalarFunction(ScalarFunction { args, .. }) = expr else {
        return None;
    };
    if function_name(expr) != Some("named_struct") {
        return None;
    }
    args.chunks(2)
        .map(|pair| Some((string_literal(&pair[0])?, &pair[1])))
        .collect()
}

fn sorted_keys(keys: &[&str]) -> String {
    let mut keys: Vec<String> = keys.iter().map(|key| repr_str(key)).collect();
    keys.sort();
    format!("[{}]", keys.join(", "))
}

pub(crate) fn parse_options(
    node: Option<&Expr>,
    allowed: &[&str],
) -> Result<Options, CompileError> {
    let Some(node) = node else {
        return Ok(Options::default());
    };
    let Some(fields) = object_fields(node) else {
        return compile_error("AI options must be an object such as {'selectivity': 0.3}.");
    };
    let mut options = Options::default();
    for (key, value) in fields {
        if !allowed.contains(&key.as_str()) {
            return compile_error(format!(
                "Unknown option {} must be replaced with one of {}.",
                repr_str(&key),
                sorted_keys(allowed)
            ));
        }
        if key == "selectivity" {
            match (string_literal(value), number_of(value)) {
                (None, Some(number)) => options.selectivity = Some(number),
                _ => return compile_error("The selectivity option must be a number."),
            }
        } else {
            match string_literal(value) {
                Some(text) => options.anchor = Some(text),
                None => return compile_error("The anchor option must be a table alias string."),
            }
        }
    }
    Ok(options)
}

pub(crate) fn unwrap_hint(term: &Expr) -> Result<(Expr, Option<f64>), CompileError> {
    let Expr::ScalarFunction(ScalarFunction { args, .. }) = term else {
        return Ok((term.clone(), None));
    };
    if function_name(term) != Some("hint") {
        return Ok((term.clone(), None));
    }
    if args.len() != 2 {
        return compile_error("HINT takes a test and an option object like {'selectivity': 0.4}");
    }
    let options = parse_options(Some(&args[1]), &["selectivity"])?;
    Ok((args[0].clone(), options.selectivity))
}

pub(crate) fn unfolded(expr: Expr) -> Expr {
    expr.transform_up(|inner| Ok(unfold(inner)))
        .map(|done| done.data)
        .expect("unfolding a literal does not fail")
}

fn unfold(expr: Expr) -> Transformed<Expr> {
    use datafusion::arrow::array::Array;
    use datafusion::logical_expr::expr::ScalarFunction as Call;
    let Expr::Literal(value, _) = &expr else {
        return Transformed::no(expr);
    };
    let element = |array: &dyn Array, index: usize| {
        ScalarValue::try_from_array(array, index)
            .map(|value| unfold(Expr::Literal(value, None)).data)
    };
    match value {
        ScalarValue::Struct(array) if array.len() == 1 => {
            let fields = array.fields();
            let positional = fields
                .iter()
                .enumerate()
                .all(|(index, field)| *field.name() == format!("c{index}"));
            let mut args = Vec::new();
            for (index, field) in fields.iter().enumerate() {
                let Ok(value) = element(array.column(index).as_ref(), 0) else {
                    return Transformed::no(expr);
                };
                if !positional {
                    args.push(Expr::Literal(
                        ScalarValue::Utf8(Some(field.name().clone())),
                        None,
                    ));
                }
                args.push(value);
            }
            let function = if positional {
                datafusion::functions::core::r#struct()
            } else {
                datafusion::functions::core::named_struct()
            };
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(function, args)))
        }
        ScalarValue::List(array) if array.len() == 1 => {
            let values = array.value(0);
            let mut args = Vec::new();
            for index in 0..values.len() {
                let Ok(value) = element(values.as_ref(), index) else {
                    return Transformed::no(expr);
                };
                args.push(value);
            }
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(
                datafusion::functions_nested::make_array::make_array_udf(),
                args,
            )))
        }
        _ => Transformed::no(expr),
    }
}
