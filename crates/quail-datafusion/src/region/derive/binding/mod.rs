mod calls;
mod conditions;
mod select;

use crate::functions::function_name;
use crate::functions::options::string_literal;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::Column;
use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::Expr;
use quail_planner::constants::SPAN_SUFFIX;
use quail_planner::error::CompileError;
use quail_planner::expressions::repr_str;
use quail_planner::operators::{Scan, ScanColumn, Schema};
use quail_planner::physical::{ColumnRef, GetField, ModelCall, ValueExpression};
use quail_planner::prompts::Turn;
use std::collections::BTreeMap;

pub(crate) use conditions::{check_labels, negated_inward};
pub(crate) use select::{Selected, output_name};

pub(crate) fn unsupported<T>(message: &str) -> Result<T, CompileError> {
    Err(CompileError::unsupported(message))
}

pub(crate) type ComputedBy = BTreeMap<(String, String), &'static str>;

fn contains(expr: &Expr, found: impl Fn(&Expr) -> bool) -> bool {
    expr.exists(|inner| Ok(found(inner))).unwrap_or(false)
}

#[derive(Debug, Clone)]
pub(crate) struct AnswerColumn {
    pub(crate) answer: String,
    pub(crate) name: String,
    pub(crate) kind: &'static str,
    pub(crate) alias: Option<String>,
    pub(crate) calls: Vec<(ModelCall, String)>,
    pub(crate) fields: Vec<String>,
}

pub(crate) struct Binder {
    turn: Turn,
    layout: String,
    tables: Vec<(String, String, SchemaRef)>,
    document_columns: Vec<(String, ValueExpression)>,
    answers: Vec<AnswerColumn>,
    answer_types: Vec<(String, String, DataType)>,
}

impl Binder {
    pub(crate) fn new(turn: &Turn, layout: &str) -> Self {
        Self {
            turn: turn.clone(),
            layout: layout.to_string(),
            tables: Vec::new(),
            document_columns: Vec::new(),
            answers: Vec::new(),
            answer_types: Vec::new(),
        }
    }

    pub(crate) fn column_operators(&self) -> (Vec<(ModelCall, String)>, ComputedBy) {
        let mut names: Vec<(ModelCall, String)> = Vec::new();
        let mut computed: ComputedBy = BTreeMap::new();
        for found in &self.answers {
            if !["label", "extract", "features", "map"].contains(&found.kind) {
                continue;
            }
            for (call, name) in &found.calls {
                names.push((call.clone(), name.clone()));
                if let Some(alias) = &found.alias {
                    computed.insert((alias.clone(), name.clone()), found.kind);
                    if found.kind == "extract" {
                        let span = format!("{name}{SPAN_SUFFIX}");
                        computed.insert((alias.clone(), span), "extract");
                    }
                }
            }
        }
        (names, computed)
    }

    fn provider(&self, alias: &str) -> Option<&str> {
        self.tables
            .iter()
            .find(|(found, _, _)| found == alias)
            .map(|(_, provider, _)| provider.as_str())
    }

    pub(crate) fn add_table(
        &mut self,
        alias: &str,
        provider: &str,
        schema: SchemaRef,
    ) -> Result<(), CompileError> {
        if self.provider(alias).is_some() {
            return unsupported("DataFusion must keep repeated table aliases in separate scopes.");
        }
        self.tables
            .push((alias.to_string(), provider.to_string(), schema));
        Ok(())
    }

    fn resolve_column(&self, column: &Column) -> Result<ColumnRef, CompileError> {
        let alias = match &column.relation {
            Some(relation) => relation.table().to_string(),
            None => {
                let owners: Vec<&(String, String, SchemaRef)> = self
                    .tables
                    .iter()
                    .filter(|(_, _, schema)| schema.field_with_name(&column.name).is_ok())
                    .collect();
                if owners.len() != 1 {
                    return unsupported(&format!(
                        "DataFusion must resolve column {} outside this model region.",
                        repr_str(&column.name)
                    ));
                }
                owners[0].0.clone()
            }
        };
        let Some(provider) = self.provider(&alias) else {
            return unsupported(&format!(
                "DataFusion must materialize table alias {}.",
                repr_str(&alias)
            ));
        };
        Ok(ColumnRef {
            alias,
            provider: provider.to_string(),
            column: column.name.clone(),
        })
    }

    fn bind_value(&mut self, expr: &Expr) -> Result<ValueExpression, CompileError> {
        match expr {
            Expr::Column(column) => {
                return self.resolve_column(column).map(ValueExpression::ColumnRef);
            }
            Expr::OuterReferenceColumn(_, column) => {
                return self.resolve_column(column).map(ValueExpression::ColumnRef);
            }
            Expr::Alias(alias) => return self.bind_value(&alias.expr),
            Expr::ScalarFunction(call)
                if call.name() == "get_field"
                    && call.args[1..]
                        .iter()
                        .all(|field| string_literal(field).is_some()) =>
            {
                let input = self.bind_value(&call.args[0])?;
                if matches!(
                    input,
                    ValueExpression::ColumnRef(_) | ValueExpression::GetField(_)
                ) {
                    let mut value = input;
                    for field in &call.args[1..] {
                        value = ValueExpression::GetField(GetField {
                            input: Box::new(value),
                            field: string_literal(field).expect("a field name"),
                        });
                    }
                    return Ok(value);
                }
            }
            _ => {}
        }
        if contains(expr, |inner| {
            function_name(inner).is_some_and(|name| name.starts_with("ai_") || name == "prompt")
        }) {
            return unsupported(
                "DataFusion must evaluate this expression outside the model region.",
            );
        }
        self.computed(expr)
    }

    fn computed(&self, expr: &Expr) -> Result<ValueExpression, CompileError> {
        if contains(expr, |inner| {
            matches!(
                inner,
                Expr::AggregateFunction(_)
                    | Expr::WindowFunction(_)
                    | Expr::ScalarSubquery(_)
                    | Expr::InSubquery(_)
                    | Expr::Exists(_)
                    | Expr::Placeholder(_)
            )
        }) {
            return unsupported(
                "DataFusion must evaluate this expression outside the model region.",
            );
        }
        crate::region::derive::expressions::expression(expr, |column| {
            let reference = self.resolve_column(column)?;
            let data_type = self.column_type(&reference)?;
            Ok((reference, data_type))
        })
        .map(ValueExpression::Value)
    }

    fn column_type(&self, reference: &ColumnRef) -> Result<DataType, CompileError> {
        if let Some((_, _, data_type)) = self
            .answer_types
            .iter()
            .find(|(alias, column, _)| *alias == reference.alias && *column == reference.column)
        {
            return Ok(data_type.clone());
        }
        self.tables
            .iter()
            .find(|(alias, _, _)| *alias == reference.alias)
            .and_then(|(_, _, schema)| schema.field_with_name(&reference.column).ok())
            .map(|field| field.data_type().clone())
            .ok_or_else(|| {
                CompileError::unsupported(format!(
                    "DataFusion must resolve the type of column {} outside the model region.",
                    repr_str(&reference.name(true))
                ))
            })
    }

    pub(crate) fn scan(&self, alias: &str) -> Scan {
        let (_, provider, schema) = self
            .tables
            .iter()
            .find(|(found, _, _)| found == alias)
            .expect("a table in scope");
        Scan {
            column: match self
                .document_columns
                .iter()
                .find(|(found, _)| found == alias)
            {
                Some((_, column)) => ScanColumn::Expression(column.clone()),
                None => ScanColumn::Unnamed(String::new()),
            },
            schema: Some(schema_of(schema)),
            provider: provider.clone(),
            alias: alias.to_string(),
            columns: Vec::new(),
            predicates: Vec::new(),
        }
    }
}

fn schema_of(schema: &SchemaRef) -> Schema {
    use base64::Engine;
    use datafusion::arrow::ipc::writer::{
        DictionaryTracker, IpcDataGenerator, IpcWriteOptions, write_message,
    };
    let options = IpcWriteOptions::default();
    let mut tracker = DictionaryTracker::new(false);
    let encoded = IpcDataGenerator::default().schema_to_bytes_with_dictionary_tracker(
        schema,
        &mut tracker,
        &options,
    );
    let mut bytes = Vec::new();
    write_message(&mut bytes, encoded, &options).expect("an IPC schema message");
    Schema {
        ipc: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}
