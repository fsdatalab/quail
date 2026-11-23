use std::ops::ControlFlow;

use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    BinaryOperator, DataType, ExactNumberInfo, Expr, Function, FunctionArg, FunctionArgExpr,
    FunctionArgumentList, FunctionArguments, Ident, ObjectName, ObjectNamePart, Value,
    visit_expressions_mut,
};

const AI_CALLS: [&str; 6] = [
    "AI.IF",
    "AI.SCORE",
    "AI.CLASSIFY",
    "AI_FILTER",
    "AI_SCORE",
    "AI_CLASSIFY",
];
const SERVICE_ARGUMENTS: [&str; 4] = ["connection_id", "endpoint", "model_params", "request_type"];
const POSITIONAL_ARGUMENTS: [&str; 2] = ["prompt", "categories"];

pub(crate) fn rewrite(statement: &mut Statement) {
    if let Statement::Statement(inner) = statement {
        let _ = visit_expressions_mut(inner.as_mut(), |expr| {
            rewrite_expr(expr);
            ControlFlow::<()>::Continue(())
        });
    }
}

fn rewrite_expr(expr: &mut Expr) {
    match expr {
        Expr::Cast { data_type, .. } if matches!(data_type, DataType::Float64) => {
            *data_type = DataType::Double(ExactNumberInfo::None);
        }
        Expr::Function(function) if AI_CALLS.contains(&call_name(&function.name).as_str()) => {
            rewrite_ai_call(function);
        }
        _ => {}
    }
}

fn call_name(name: &ObjectName) -> String {
    name.0
        .iter()
        .filter_map(|part| match part {
            ObjectNamePart::Identifier(ident) => Some(ident.value.to_uppercase()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn rewrite_ai_call(function: &mut Function) {
    let FunctionArguments::List(list) = &mut function.args else {
        return;
    };
    list.args.retain(|arg| {
        !matches!(arg, FunctionArg::Named { name, .. }
            if SERVICE_ARGUMENTS.contains(&name.value.to_lowercase().as_str()))
    });
    for arg in &mut list.args {
        if let FunctionArg::Named {
            name, arg: value, ..
        } = arg
            && POSITIONAL_ARGUMENTS.contains(&name.value.to_lowercase().as_str())
        {
            *arg = FunctionArg::Unnamed(value.clone());
        }
    }
    if let Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(first))) = list.args.first_mut()
        && let Some(prompt) = prompt_call(first)
    {
        *first = prompt;
    }
}

fn prompt_call(expr: &Expr) -> Option<Expr> {
    let parts = match expr {
        Expr::Tuple(items) | Expr::Struct { values: items, .. } => items.clone(),
        Expr::BinaryOp {
            op: BinaryOperator::StringConcat,
            ..
        } => concatenated(expr),
        Expr::Function(function) if call_name(&function.name) == "FORMAT" => formatted(function)?,
        _ => return None,
    };
    let mut template = String::new();
    let mut values = Vec::new();
    for part in parts {
        match literal(&part) {
            Some(text) if names_a_placeholder(&text) => return None,
            Some(text) => template.push_str(&text),
            None => {
                template.push_str(&format!("{{{}}}", values.len()));
                values.push(part);
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    let args = std::iter::once(Expr::value(Value::SingleQuotedString(template)))
        .chain(values)
        .map(|value| FunctionArg::Unnamed(FunctionArgExpr::Expr(value)))
        .collect();
    Some(Expr::Function(Function {
        name: ObjectName::from(vec![Ident::new("PROMPT")]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args,
            clauses: Vec::new(),
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: Vec::new(),
    }))
}

fn concatenated(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::StringConcat,
            right,
        } => {
            let mut parts = concatenated(left);
            parts.extend(concatenated(right));
            parts
        }
        Expr::Nested(inner) => concatenated(inner),
        other => vec![other.clone()],
    }
}

fn formatted(function: &Function) -> Option<Vec<Expr>> {
    let FunctionArguments::List(list) = &function.args else {
        return None;
    };
    let mut args = list.args.iter().map(|arg| match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Some(expr.clone()),
        _ => None,
    });
    let pattern = literal(&args.next()??)?;
    let values: Vec<Expr> = args.collect::<Option<_>>()?;
    let pieces: Vec<&str> = pattern.split("%s").collect();
    if pieces.len() != values.len() + 1 || pieces.iter().any(|piece| piece.contains('%')) {
        return None;
    }
    let mut parts = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        parts.push(Expr::value(Value::SingleQuotedString(piece.to_string())));
        if let Some(value) = values.get(index) {
            parts.push(value.clone());
        }
    }
    Some(parts)
}

fn names_a_placeholder(text: &str) -> bool {
    text.match_indices('{').any(|(index, _)| {
        let digits = text[index + 1..]
            .chars()
            .take_while(char::is_ascii_digit)
            .count();
        digits > 0 && text[index + 1 + digits..].starts_with('}')
    })
}

fn literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Value(value) => value.clone().into_string(),
        _ => None,
    }
}
