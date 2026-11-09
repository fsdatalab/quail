//! Bind WHERE conditions and join equalities.
//! Replace tested AI calls with their answer columns and check labels.

use std::collections::BTreeSet;

use super::calls::{arguments, is_call, parse_options, score_comparison, string_literal};
use super::{Binder, Hinted, contains, unsupported};
use crate::functions::{column_call_kind, sae_features_type, span_type};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::{
    SPAN_SUFFIX, aliased_call, call_aliases, column_alias_of, column_name, repr_list, repr_str,
};
use quail_planner::physical::{Equality, ModelCall, ValueExpression};
use serde_json::Value;

/// Whether a term asks the model, counting AI.EXTRACT only bare.
pub(super) fn has_ai_call(expr: &Expr) -> bool {
    has_ai_call_under(expr, false)
}

fn has_ai_call_under(expr: &Expr, in_field: bool) -> bool {
    match expr {
        Expr::Exists(_) => true,
        Expr::ScalarFunction(call) => {
            let name = call.name();
            if name == "ai_filter" || name == "ai_score" {
                return true;
            }
            if (name == "ai_extract" || name == "ai_extract_span") && !in_field {
                return true;
            }
            let field = name == "get_field";
            call.args
                .iter()
                .enumerate()
                .any(|(index, arg)| has_ai_call_under(arg, field && index == 0))
        }
        other => {
            let mut any = false;
            let _ = other.apply_children(|child| {
                any = has_ai_call_under(child, false);
                Ok(if any {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                })
            });
            any
        }
    }
}

/// `term`, negated when `negate`, with each NOT moved down to the tests
/// below AND and OR by De Morgan's laws, so the parts of an AND, such as
/// `NOT (a OR b)`, are placed apart.
fn negated_inward(term: Expr, negate: bool) -> Expr {
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

    // ------------------------------------------------------------ WHERE

    /// Bind a WHERE term that asks no model: its conditions, one per
    /// part of an AND unless a HINT estimates the whole term, and the
    /// selectivity the HINT gives; or None for a term that asks one.
    pub(crate) fn bind_plain_where_term(
        &mut self,
        term: &Expr,
    ) -> Result<Option<Vec<Hinted>>, CompileError> {
        let (term, hint) = self.unwrap_hint(term)?;
        if has_ai_call(&term) || label_test(&term).is_some() {
            return Ok(None);
        }
        let term = negated_inward(term, false);
        let parts = match hint {
            Some(_) => vec![&term],
            None => split_conjunction(&term),
        };
        let mut bound = Vec::new();
        for part in parts {
            bound.push((self.bind_condition(part)?, hint));
        }
        Ok(Some(bound))
    }

    /// Refuse a WHERE term that is no plain condition and no test of a
    /// label, with the message that names what is wrong with it.
    ///
    /// # Errors
    ///
    /// The term is outside Quail's language, or its AI filter or score
    /// does not bind.
    pub(crate) fn refuse_where_term(&mut self, term: &Expr) -> Result<(), CompileError> {
        let (term, _) = self.unwrap_hint(term)?;
        if matches!(
            &term,
            Expr::BinaryExpr(BinaryExpr {
                op: Operator::Or,
                ..
            })
        ) {
            // an OR of AI.IF calls and plain tests is an AiNode's filter
            return unsupported(&format!("DataFusion must evaluate the disjunction {term}."));
        }
        let (node, anti) = match &term {
            Expr::Not(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        if matches!(node, Expr::Exists(_)) {
            return Ok(());
        }
        if anti {
            return unsupported("DataFusion must evaluate NOT around this model call.");
        }
        if contains(&term, |inner| is_call(inner, "ai_classify")) {
            return unsupported("DataFusion must evaluate this test of an AI.CLASSIFY result.");
        }
        if contains(&term, |inner| is_call(inner, "ai_extract")) {
            return unsupported("DataFusion must evaluate this test of an extraction result.");
        }
        if score_comparison(&term).is_some() {
            self.bind_ai_score_test(&term, &["selectivity"], None)?;
        } else {
            if !is_call(&term, "ai_filter") {
                return unsupported("DataFusion must evaluate this expression of a model result.");
            }
            self.bind_ai_filter_call(&term, &["selectivity", "anchor"], None)?;
        }
        Ok(())
    }

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
        let read = self.answer_columns(term)?;
        self.computed(&read)
    }

    /// `term` with each column-operator call it tests as the column of
    /// the call's answer: an AI.CLASSIFY, AI.MAP, or AI.SAE_FEATURES
    /// call, or a field of an AI.EXTRACT call or its span.
    fn answer_columns(&mut self, term: &Expr) -> Result<Expr, CompileError> {
        // each label a test compares a classification with
        let mut compared: Vec<(Expr, Value)> = Vec::new();
        let _ = term.apply(|node| {
            match node {
                Expr::BinaryExpr(BinaryExpr {
                    left,
                    op: Operator::Eq,
                    right,
                }) => {
                    for (call, other) in [(left, right), (right, left)] {
                        if column_call_kind(call) == Some("label")
                            && let Some(text) = string_literal(other)
                        {
                            compared.push((call.as_ref().clone(), Value::from(text)));
                        }
                    }
                }
                Expr::InList(InList { expr, list, .. })
                    if column_call_kind(expr) == Some("label") =>
                {
                    let labels = list.iter().filter_map(string_literal).map(Value::from);
                    compared.push((expr.as_ref().clone(), Value::Array(labels.collect())));
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        });
        for (node, labels) in &compared {
            let call = self.bind_column_call(node, "label")?;
            let name = self.answer_name(node, &call)?;
            check_labels(&name, labels, &call)?;
        }
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
                        if column_call_kind(inner) == Some("features") =>
                    {
                        self.answer_column(inner, "features").map(|column| {
                            let column = Box::new(column);
                            Some(match &node {
                                Expr::IsNull(_) => Expr::IsNull(column),
                                _ => Expr::IsNotNull(column),
                            })
                        })
                    }
                    _ => match column_call_kind(&node) {
                        Some("extract") => unsupported(
                            "DataFusion must evaluate this test of an extraction result.",
                        ),
                        Some("features") => unsupported(
                            "DataFusion must evaluate this expression of an AI.SAE_FEATURES list.",
                        ),
                        Some(kind) => self.answer_column(&node, kind).map(Some),
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

    /// The answer column of a column-operator call a test reads: the
    /// SELECT list's name for it, or a name of its own.
    fn answer_column(&mut self, node: &Expr, kind: &str) -> Result<Expr, CompileError> {
        let call = self.bind_column_call(node, kind)?;
        let name = self.answer_name(node, &call)?;
        let alias = call_aliases(&call)[0].clone();
        let data_type = match kind {
            "features" => sae_features_type(),
            _ => DataType::Utf8,
        };
        self.answer_types
            .push((alias.clone(), name.clone(), data_type));
        Ok(Expr::Column(Column::new(Some(alias), name)))
    }

    /// Record that a condition may read `alias.column`, an answer of
    /// type `data_type` that no table holds.
    pub(crate) fn note_answer_type(&mut self, alias: &str, column: &str, data_type: DataType) {
        let entry = (alias.to_string(), column.to_string(), data_type);
        if !self.answer_types.contains(&entry) {
            self.answer_types.push(entry);
        }
    }

    /// The name of the answer column of `call`, the call `node` makes.
    fn answer_name(&mut self, node: &Expr, call: &ModelCall) -> Result<String, CompileError> {
        let aliases = call_aliases(call);
        if aliases.len() != 1 {
            return unsupported(
                "DataFusion must evaluate this test of a model result over joined rows.",
            );
        }
        let alias = aliases[0].clone();
        let name = self
            .selected
            .iter()
            .find(|(_, selected)| selected == node)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| format!("__column_{alias}_{}", self.tested.len()));
        Ok(match self.tested.iter().find(|(found, _)| found == call) {
            Some((_, existing)) => existing.clone(),
            None => {
                self.tested.push((call.clone(), name.clone()));
                name
            }
        })
    }

    /// The column of the AI.EXTRACT answer, or the field of its span, a
    /// field path reads: `get_field(ai_extract(...), field)`, or
    /// `get_field(ai_extract_span(...), field, start|end)`; None for
    /// another expression.
    fn extract_column(&mut self, node: &Expr) -> Result<Option<Expr>, CompileError> {
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
        let span = is_call(inner, "ai_extract_span");
        if !(span || is_call(inner, "ai_extract")) || path.is_empty() {
            return Ok(None);
        }
        let Some((name, call_node)) = self
            .selected
            .iter()
            .find(|(_, selected)| {
                is_call(selected, "ai_extract") && arguments(selected) == arguments(inner)
            })
            .map(|(name, selected)| (name.clone(), selected.clone()))
        else {
            return unsupported(
                "DataFusion must evaluate this condition over the extraction result.",
            );
        };
        let field = path[0].clone();
        let rest = &path[1..];
        if span && !(rest == ["start"] || rest == ["end"]) {
            return unsupported("DataFusion must evaluate this expression of an extraction span.");
        }
        if !span && !rest.is_empty() {
            return compile_error(format!("{name}.{field} reads a field under an answer"));
        }
        let (found, members) = self.bind_ai_extract(&call_node, &name)?;
        if !found.fields.contains(&field) {
            return compile_error(format!(
                "AI.EXTRACT {} has no field {}; its fields are {}",
                repr_str(&name),
                repr_str(&field),
                repr_list(&found.fields)
            ));
        }
        let member = members
            .iter()
            .find(|member| column_name(member, true) == format!("{name}.{field}"))
            .expect("a member per field");
        let alias = call_aliases(aliased_call(member, &[]).expect("an extraction"))[0].clone();
        let column = format!("{name}.{field}{}", if span { SPAN_SUFFIX } else { "" });
        let data_type = if span { span_type() } else { DataType::Utf8 };
        self.answer_types
            .push((alias.clone(), column.clone(), data_type));
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

/// The AI.CLASSIFY call and tested values of an = or IN test, or None.
pub(crate) fn label_test(expr: &Expr) -> Option<(&Expr, Vec<&Expr>)> {
    match expr {
        Expr::InList(InList {
            expr,
            list,
            negated: false,
        }) if is_call(expr, "ai_classify") => Some((expr, list.iter().collect())),
        // DataFusion writes a short IN list as an OR of equalities
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Or,
            right,
        }) => {
            let (call, mut labels) = label_test(left)?;
            let (other, more) = label_test(right)?;
            if call != other {
                return None;
            }
            labels.extend(more);
            Some((call, labels))
        }
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) => {
            if is_call(left, "ai_classify") {
                Some((left, vec![right]))
            } else if is_call(right, "ai_classify") {
                Some((right, vec![left]))
            } else {
                None
            }
        }
        _ => None,
    }
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
    compile_error(format!(
        "the filter on {} lists {}, which are not among its labels {}",
        repr_str(name),
        repr_list(&unknown),
        repr_list(&written)
    ))
}
