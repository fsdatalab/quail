use super::{Binder, contains, unsupported};
use crate::functions::{ai_function_name, function_name};
use datafusion::logical_expr::Expr;
use quail_planner::constants::SPAN_SUFFIX;
use quail_planner::error::CompileError;
use quail_planner::expressions::{aliased_call, call_aliases, column_name, repr_list, repr_str};
use quail_planner::physical::{Alias, ExtractStruct, ValueExpression};

pub(crate) fn output_name(expr: &Expr) -> String {
    expr.qualified_name().1
}

fn unalias(expr: &Expr) -> (&Expr, Option<&str>) {
    let mut names = Vec::new();
    let mut node = expr;
    while let Expr::Alias(alias) = node {
        names.push(alias.name.as_str());
        node = &alias.expr;
    }
    let automatic = names.last().is_some_and(|name| {
        *name == "count(*)"
            || name.split_once('(').is_some_and(|(function, _)| {
                function.contains('.')
                    && ai_function_name(function).is_some()
                    && ai_function_name(function) == function_name(node)
            })
    });
    let written = if automatic {
        names.len() > 1
    } else {
        !names.is_empty()
    };
    (
        node,
        if written {
            names.first().copied()
        } else {
            None
        },
    )
}

pub(crate) struct Selected {
    pub(crate) columns: Vec<ValueExpression>,
    pub(crate) structs: Vec<ExtractStruct>,
    pub(crate) spans: Vec<String>,
    pub(crate) output: Vec<String>,
}

impl Binder {
    pub(crate) fn validate_query(
        &self,
        columns: &[ValueExpression],
        structs: &[ExtractStruct],
    ) -> Result<(), CompileError> {
        let is_alias = |name: &str| self.provider(name).is_some();
        for found in structs {
            if is_alias(&found.name) {
                return unsupported(
                    "DataFusion must assign this output name because it also names a table.",
                );
            }
        }
        for column in columns {
            let name = column_name(column, true);
            if let ValueExpression::Alias(_) = column {
                if is_alias(&name) {
                    return unsupported(
                        "DataFusion must assign this output name because it also names a table.",
                    );
                }
                if self
                    .tables
                    .iter()
                    .any(|(_, _, schema)| schema.field_with_name(&name).is_ok())
                {
                    return unsupported(
                        "DataFusion must assign this output name because it also names an input column.",
                    );
                }
            }
        }
        let mut prompts_seen: Vec<(quail_planner::physical::Prompt, String)> = Vec::new();
        for column in columns {
            let Some(call) = aliased_call(column, &[]) else {
                continue;
            };
            let name = column_name(column, true);
            if let Some((_, first)) = prompts_seen
                .iter()
                .find(|(prompt, _)| *prompt == *call.prompt)
            {
                return unsupported(&format!(
                    "DataFusion must project the repeated model result as {} and {}.",
                    repr_str(first),
                    repr_str(&name)
                ));
            }
            prompts_seen.push(((*call.prompt).clone(), name));
        }
        Ok(())
    }

    pub(crate) fn bind_select(&mut self, projection: &[Expr]) -> Result<Selected, CompileError> {
        let mut selected = Selected {
            columns: Vec::new(),
            structs: Vec::new(),
            spans: Vec::new(),
            output: Vec::new(),
        };
        let mut extracts: Vec<(String, String)> = Vec::new();
        let mut spans: Vec<(String, String)> = Vec::new();
        for item in projection {
            let (node, alias) = unalias(item);
            let name = alias.map_or_else(|| output_name(item), str::to_string);
            if let Expr::Column(column) = node
                && column.relation.is_none()
                && let Some(answer) = column.name.strip_suffix(SPAN_SUFFIX)
                && self
                    .answer(answer)
                    .is_some_and(|found| found.kind == "extract")
            {
                spans.push((name, answer.to_string()));
                continue;
            }
            for column in self.bind_select_item(node, &name, alias.is_some(), &mut selected)? {
                selected.output.push(column_name(&column, true));
                if !selected.columns.contains(&column) {
                    selected.columns.push(column);
                }
            }
            if let Expr::Column(column) = node
                && column.relation.is_none()
                && self
                    .answer(&column.name)
                    .is_some_and(|found| found.kind == "extract")
            {
                extracts.push((name, column.name.clone()));
            }
        }
        for (alias, span) in spans {
            let Some((name, _)) = extracts.iter().find(|(_, answer)| *answer == span) else {
                return unsupported(&format!(
                    "The extraction for span output {alias} must run in a separate model region."
                ));
            };
            selected.spans.push(name.clone());
        }
        let mut unique = selected.output.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != selected.output.len() {
            return unsupported(&format!(
                "DataFusion must assign the repeated output names {}.",
                repr_list(&selected.output)
            ));
        }
        Ok(selected)
    }

    fn bind_select_item(
        &mut self,
        node: &Expr,
        name: &str,
        aliased: bool,
        selected: &mut Selected,
    ) -> Result<Vec<ValueExpression>, CompileError> {
        let named = |expression: ValueExpression| {
            ValueExpression::Alias(Alias {
                expression: Box::new(expression),
                name: name.to_string(),
            })
        };
        let answer = match node {
            Expr::Column(column) if column.relation.is_none() => self.answer(&column.name),
            _ => None,
        };
        match answer {
            Some(found) if found.kind == "extract" => {
                selected.structs.push(ExtractStruct {
                    name: name.to_string(),
                    fields: found.fields.clone(),
                });
                return Ok(found
                    .calls
                    .iter()
                    .zip(&found.fields)
                    .map(|((call, _), field)| {
                        ValueExpression::Alias(Alias {
                            expression: Box::new(ValueExpression::ModelCall(call.clone())),
                            name: format!("{name}.{field}"),
                        })
                    })
                    .collect());
            }
            Some(found) if found.kind != "boolean" => {
                let call = found.calls[0].0.clone();
                if found.kind == "score" && !(1..=2).contains(&call_aliases(&call).len()) {
                    return unsupported("This score must run over a materialized input relation.");
                }
                return Ok(vec![named(ValueExpression::ModelCall(call))]);
            }
            _ => {}
        }
        let nested = contains(node, |inner| match inner {
            Expr::Column(column) if column.relation.is_none() => {
                let answer = column
                    .name
                    .strip_suffix(SPAN_SUFFIX)
                    .unwrap_or(&column.name);
                self.answer(answer)
                    .is_some_and(|found| found.kind != "boolean")
            }
            _ => false,
        });
        if nested {
            return unsupported("A model call cannot be nested in this SELECT expression.");
        }
        let read = self.answer_refs(node)?;
        let value = self.bind_value(&read)?;
        let renamed = aliased || read != *node && column_name(&value, true) != name;
        Ok(vec![if renamed { named(value) } else { value }])
    }
}
