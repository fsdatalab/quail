//! Bind conditions and join equalities.
//! Read each answer a condition tests as its operator's column.

use quail_planner::constants::SPAN_SUFFIX;
use std::collections::BTreeSet;

use super::calls::{arguments, is_call, parse_options, string_literal};
use super::{AnswerColumn, Binder, contains, unsupported};
use crate::functions::{sae_features_type, span_type};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::{column_alias_of, repr_list, repr_str};
use quail_planner::physical::{Equality, ModelCall, ValueExpression};
use serde_json::Value;

/// `term`, negated when `negate`, with each NOT moved down to the tests
/// below AND and OR by De Morgan's laws, so the parts of an AND, such as
/// `NOT (a OR b)`, are placed apart.
pub(crate) fn negated_inward(term: Expr, negate: bool) -> Expr {
    match term {
        Expr::Not(inner) => negated_inward(*inner, !negate),
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::And | Operator::Or) =>
        {
            let op = match (op, negate) {
                (Operator::And, true) => Operator::Or,
                (Operator::Or, true) => Operator::And,
                (op, _) => op,
            };
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(negated_inward(*left, negate)),
                op,
                Box::new(negated_inward(*right, negate)),
            ))
        }
        Expr::IsNull(inner) if negate => Expr::IsNotNull(inner),
        Expr::IsNotNull(inner) if negate => Expr::IsNull(inner),
        other if negate => Expr::Not(Box::new(other)),
        other => other,
    }
}

impl Binder {
    /// Bind `left = right`, an ON equality of the join of `joined` and a
    /// table already in the query.
    pub(crate) fn bind_equality(
        &mut self,
        term: &Expr,
        joined: &str,
    ) -> Result<Equality, CompileError> {
        let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) = term
        else {
            return unsupported("DataFusion must evaluate this join condition.");
        };
        let equality = Equality {
            left: self.bind_value(left)?,
            right: self.bind_value(right)?,
        };
        let sides = [
            column_alias_of(&equality.left),
            column_alias_of(&equality.right),
        ];
        if !sides.contains(&joined.to_string()) || sides[0] == sides[1] {
            return unsupported(
                "DataFusion must evaluate a join condition that does not connect its inputs.",
            );
        }
        Ok(equality)
    }

    /// The test a `HINT` wraps and the selectivity it gives; `term` and
    /// None without one.
    pub(crate) fn unwrap_hint(&mut self, term: &Expr) -> Result<(Expr, Option<f64>), CompileError> {
        if !is_call(term, "hint") {
            return Ok((term.clone(), None));
        }
        let args = arguments(term);
        if args.len() != 2 {
            return compile_error(
                "HINT takes a test and an option object like {'selectivity': 0.4}",
            );
        }
        let options = parse_options(Some(&args[1]), &["selectivity"])?;
        Ok((args[0].clone(), options.selectivity))
    }

    // ------------------------------------------------------------ conditions

    /// Bind a term that asks no model as a condition, each answer it
    /// tests read as the answer's column.
    pub(crate) fn bind_condition(&mut self, term: &Expr) -> Result<ValueExpression, CompileError> {
        let read = self.answer_refs(term)?;
        // a test that still holds a call is one DataFusion evaluates
        // above the region
        if contains(&read, |inner| {
            crate::is_ai_call(inner) || is_call(inner, "prompt")
        }) {
            return unsupported("DataFusion must evaluate this expression of a model result.");
        }
        self.computed(&read)
    }

    /// The columns of the `AiNode` that adds the column `name`.
    pub(crate) fn answer(&self, name: &str) -> Option<&AnswerColumn> {
        self.answers.iter().rev().find(|found| found.answer == name)
    }

    /// Record the columns an `AiNode` adds, which conditions may read.
    pub(crate) fn add_answer(&mut self, found: AnswerColumn) {
        if let Some(alias) = &found.alias {
            let data_type = match found.kind {
                "features" => sae_features_type(),
                "boolean" => DataType::Boolean,
                "score" => DataType::Float64,
                _ => DataType::Utf8,
            };
            let mut names: Vec<String> = found.calls.iter().map(|(_, name)| name.clone()).collect();
            if found.kind == "boolean" {
                names.push(found.name.clone());
            }
            for name in names {
                if found.kind == "extract" {
                    let span = format!("{name}{SPAN_SUFFIX}");
                    self.answer_types.push((alias.clone(), span, span_type()));
                }
                self.answer_types
                    .push((alias.clone(), name, data_type.clone()));
            }
        }
        self.answers.push(found);
    }

    /// `term` with each answer column it reads as the column of the
    /// operator that computes it: a label, a map's text, an AI filter's
    /// answer, or a field of an extraction or of its span.
    ///
    /// # Errors
    ///
    /// The term reads an answer no single table's operator computes, a
    /// whole extraction, or an SAE feature list other than for NULL, or
    /// it compares a label with a text its classification lacks.
    pub(crate) fn answer_refs(&self, term: &Expr) -> Result<Expr, CompileError> {
        let answer_of = |expr: &Expr| match expr {
            Expr::Column(column) if column.relation.is_none() => self.answer(&column.name),
            _ => None,
        };
        // each label a test compares a classification with
        let mut compared: Vec<(&AnswerColumn, Value)> = Vec::new();
        let _ = term.apply(|node| {
            match node {
                Expr::BinaryExpr(BinaryExpr {
                    left,
                    op: Operator::Eq,
                    right,
                }) => {
                    for (column, other) in [(left, right), (right, left)] {
                        if let Some(found) = answer_of(column).filter(|found| found.kind == "label")
                            && let Some(text) = string_literal(other)
                        {
                            compared.push((found, Value::from(text)));
                        }
                    }
                }
                Expr::InList(InList { expr, list, .. }) => {
                    if let Some(found) = answer_of(expr).filter(|found| found.kind == "label") {
                        let labels = list.iter().filter_map(string_literal).map(Value::from);
                        compared.push((found, Value::Array(labels.collect())));
                    }
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        });
        for (found, labels) in &compared {
            check_labels(&found.name, labels, &found.calls[0].0)?;
        }
        let column_of = |found: &AnswerColumn| match &found.alias {
            Some(alias) => Ok(Expr::Column(Column::new(Some(alias.clone()), &found.name))),
            None => unsupported(
                "DataFusion must evaluate this test of a model result over joined rows.",
            ),
        };
        let mut failed: Option<CompileError> = None;
        let read = term
            .clone()
            .transform_down(|node| {
                if failed.is_some() {
                    return Ok(Transformed::no(node));
                }
                let found = match &node {
                    // AI.SAE_FEATURES returns a list, which a test reads only here
                    Expr::IsNull(inner) | Expr::IsNotNull(inner)
                        if answer_of(inner).is_some_and(|found| found.kind == "features") =>
                    {
                        column_of(answer_of(inner).expect("an answer")).map(|column| {
                            let column = Box::new(column);
                            Some(match &node {
                                Expr::IsNull(_) => Expr::IsNull(column),
                                _ => Expr::IsNotNull(column),
                            })
                        })
                    }
                    _ => match answer_of(&node) {
                        Some(found) => match found.kind {
                            "extract" => unsupported(
                                "DataFusion must evaluate this test of an extraction result.",
                            ),
                            "features" => unsupported(
                                "DataFusion must evaluate this expression of an AI.SAE_FEATURES list.",
                            ),
                            "score" => unsupported(
                                "DataFusion must evaluate this expression of a model result.",
                            ),
                            _ => column_of(found).map(Some),
                        },
                        None => self.extract_column(&node),
                    },
                };
                match found {
                    Ok(Some(column)) => Ok(Transformed::new(column, true, TreeNodeRecursion::Jump)),
                    Ok(None) => Ok(Transformed::no(node)),
                    Err(error) => {
                        failed = Some(error);
                        Ok(Transformed::no(node))
                    }
                }
            })
            .map_err(|error| CompileError::new(error.to_string()))?
            .data;
        match failed {
            Some(error) => Err(error),
            None => Ok(read),
        }
    }

    /// The column of an extraction's field, or the field of its span, a
    /// field path reads: `get_field(answer, field)`, or
    /// `get_field(answer's span, field, start|end)`; None for another
    /// expression.
    fn extract_column(&self, node: &Expr) -> Result<Option<Expr>, CompileError> {
        let mut path = Vec::new();
        let mut inner = node;
        while is_get_field(inner) {
            let args = arguments(inner);
            let mut fields: Vec<String> = args[1..]
                .iter()
                .map(|field| string_literal(field).unwrap_or_default())
                .collect();
            fields.extend(path);
            path = fields;
            inner = &args[0];
        }
        let Expr::Column(column) = inner else {
            return Ok(None);
        };
        if column.relation.is_some() || path.is_empty() {
            return Ok(None);
        }
        let (answer, span) = match column.name.strip_suffix(SPAN_SUFFIX) {
            Some(answer) => (answer, true),
            None => (column.name.as_str(), false),
        };
        let Some(found) = self.answer(answer).filter(|found| found.kind == "extract") else {
            return Ok(None);
        };
        let name = &found.name;
        let field = path[0].clone();
        let rest = &path[1..];
        if span && !(rest == ["start"] || rest == ["end"]) {
            return unsupported("DataFusion must evaluate this expression of an extraction span.");
        }
        if !span && !rest.is_empty() {
            return compile_error(format!("{name}.{field} reads a field under an answer"));
        }
        if !found.fields.contains(&field) {
            return compile_error(format!(
                "AI.EXTRACT {} has no field {}; its fields are {}",
                repr_str(name),
                repr_str(&field),
                repr_list(&found.fields)
            ));
        }
        let Some(alias) = found.alias.clone() else {
            return unsupported(
                "DataFusion must evaluate this condition over the extraction result.",
            );
        };
        let column = format!("{name}.{field}{}", if span { SPAN_SUFFIX } else { "" });
        let mut read = Expr::Column(Column::new(Some(alias), column));
        for part in rest {
            read = datafusion::functions::core::expr_fn::get_field(read, part.as_str());
        }
        Ok(Some(read))
    }

    /// The condition `alias.column IN (labels)` on a label's column.
    pub(crate) fn label_condition(
        &mut self,
        alias: &str,
        column: &str,
        labels: &Value,
    ) -> Result<ValueExpression, CompileError> {
        let list = labels
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|label| Expr::Literal(ScalarValue::Utf8(Some(label.to_string())), None))
            .collect();
        self.answer_types
            .push((alias.to_string(), column.to_string(), DataType::Utf8));
        let test = Expr::InList(InList::new(
            Box::new(Expr::Column(Column::new(Some(alias), column))),
            list,
            false,
        ));
        self.computed(&test)
    }
}

fn is_get_field(expr: &Expr) -> bool {
    is_call(expr, "get_field")
}

/// Refuse a test of the label `name` that accepts a label its
/// classification `call` lacks.
pub(crate) fn check_labels(
    name: &str,
    accepted: &Value,
    call: &ModelCall,
) -> Result<(), CompileError> {
    let values = match accepted {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    let labels: Vec<Value> = call.labels.as_array().cloned().unwrap_or_default();
    let unknown: BTreeSet<String> = values
        .iter()
        .filter(|value| !labels.contains(value))
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    let unknown: Vec<String> = unknown.into_iter().collect();
    let written: Vec<String> = labels
        .iter()
        .filter_map(|label| label.as_str().map(str::to_string))
        .collect();
    // a call the query did not name has a name of Quail's
    let subject = if name.starts_with("__quail") {
        "an AI.CLASSIFY call".to_string()
    } else {
        repr_str(name)
    };
    compile_error(format!(
        "the filter on {subject} lists {}, which are not among its labels {}",
        repr_list(&unknown),
        repr_list(&written)
    ))
}
