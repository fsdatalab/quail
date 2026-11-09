//! The values and conditions of a model region's operators as DataFusion
//! expressions.
//!
//! The binder reads each value or condition the executor computes, such
//! as a scan's predicate or a test of a label, as an [`Expression`]: the
//! DataFusion expression, encoded with datafusion-proto, with each
//! column read by its position and its Arrow type. A process that reads
//! a plan builds each one once as a [`PhysicalValue`], DataFusion's
//! physical expression over those types, and evaluates it over the
//! executor's rows as often as it needs.
//!
//! For example, `r.stars > 3` becomes an `Expression` that reads column
//! `#0` where `columns` is `[r.stars]` and `types` is `[Int64]`, with the
//! text `{0} > 3`.

use std::str::FromStr;
use std::sync::{Arc, OnceLock};

use base64::Engine;
use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, StructArray, make_array,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::logical_expr::{BinaryExpr, Cast, Expr, Like, Operator, TryCast};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::prelude::SessionContext;
use datafusion_proto::bytes::Serializeable;
use quail_planner::error::CompileError;
use quail_planner::expressions::{repr_float, repr_str};
use quail_planner::physical::{ColumnRef, Expression, ValueExpression};

/// Where a column's position starts and ends in an expression's text
/// before braces are escaped.
const OPEN: char = '\u{1}';
const CLOSE: char = '\u{2}';

/// The name an expression's `index`th column has in its encoded form.
fn position_name(index: usize) -> String {
    format!("#{index}")
}

/// The DataFusion session whose functions decode and build
/// expressions: DataFusion's own and Quail's. Quail's executor decodes a
/// region's expressions outside the session that planned them, so an
/// expression that calls a host's own function does not build.
fn session() -> &'static SessionContext {
    static SESSION: OnceLock<SessionContext> = OnceLock::new();
    SESSION.get_or_init(SessionContext::new)
}

/// `data_type` with every nested field nullable and without metadata,
/// as an expression's `types` hold it.
fn plain_type(data_type: &DataType) -> DataType {
    let plain = |field: &FieldRef| -> FieldRef {
        Arc::new(Field::new(
            field.name(),
            plain_type(field.data_type()),
            true,
        ))
    };
    match data_type {
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(plain).collect()),
        DataType::List(item) => DataType::List(plain(item)),
        DataType::LargeList(item) => DataType::LargeList(plain(item)),
        DataType::FixedSizeList(item, size) => DataType::FixedSizeList(plain(item), *size),
        other => other.clone(),
    }
}

/// An [`Expression`] of `expr`, each column resolved by `resolve` to the
/// column it reads and that column's Arrow type.
///
/// # Errors
///
/// A column does not resolve, or DataFusion cannot encode the
/// expression.
pub fn expression(
    expr: &Expr,
    mut resolve: impl FnMut(&Column) -> Result<(ColumnRef, DataType), CompileError>,
) -> Result<Expression, CompileError> {
    let mut columns: Vec<ColumnRef> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut failed: Option<CompileError> = None;
    let positional = expr
        .clone()
        .transform_up(|node| {
            let (Expr::Column(column) | Expr::OuterReferenceColumn(_, column)) = &node else {
                return Ok(Transformed::no(node));
            };
            let (reference, data_type) = match resolve(column) {
                Ok(found) => found,
                Err(error) => {
                    failed.get_or_insert(error);
                    return Ok(Transformed::no(node));
                }
            };
            let index = match columns.iter().position(|found| *found == reference) {
                Some(index) => index,
                None => {
                    columns.push(reference);
                    types.push(plain_type(&data_type).to_string());
                    columns.len() - 1
                }
            };
            Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                position_name(index),
            ))))
        })
        .map_err(|error| CompileError::new(error.to_string()))?
        .data
        .unalias_nested()
        .data;
    if let Some(error) = failed {
        return Err(error);
    }
    let bytes = positional.to_bytes().map_err(|_| {
        CompileError::unsupported(
            "DataFusion must evaluate this expression outside the model region.",
        )
    })?;
    Ok(Expression {
        encoded: base64::engine::general_purpose::STANDARD.encode(bytes),
        columns,
        types,
        text: template(&positional),
    })
}

/// An [`Expression`] of SQL text over the columns of `schema`, each a
/// column of table `alias` from `provider`, such as `stars > 3`.
///
/// # Errors
///
/// The text is not a SQL expression over the schema's columns.
pub fn parse(
    sql: &str,
    schema: &Schema,
    alias: &str,
    provider: &str,
) -> Result<Expression, CompileError> {
    let columns =
        DFSchema::try_from(schema.clone()).map_err(|error| CompileError::new(error.to_string()))?;
    let expr = session()
        .parse_sql_expr(sql, &columns)
        .map_err(|error| CompileError::new(error.to_string()))?;
    expression(&expr, |column| {
        let field = schema
            .field_with_name(&column.name)
            .map_err(|error| CompileError::new(error.to_string()))?;
        Ok((
            ColumnRef {
                alias: alias.to_string(),
                provider: provider.to_string(),
                column: column.name.clone(),
            },
            field.data_type().clone(),
        ))
    })
}

/// The DataFusion expression an [`Expression`] encodes, reading column
/// `#i` for its `i`th column.
pub(crate) fn decode(expression: &Expression) -> Result<Expr, CompileError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&expression.encoded)
        .map_err(|error| CompileError::new(error.to_string()))?;
    Expr::from_bytes_with_ctx(&bytes, &session().task_ctx())
        .map_err(|error| CompileError::new(error.to_string()))
}

// ---------------------------------------------------------------- evaluation

/// `array` with each struct's nulls on its fields too, so a field of a
/// null struct reads as null, as SQL reads it.
fn with_field_nulls(array: ArrayRef) -> ArrayRef {
    let Some(structs) = array.as_struct_opt() else {
        return array;
    };
    let (fields, columns, nulls) = structs.clone().into_parts();
    let columns = columns
        .into_iter()
        .map(|column| {
            let column = with_field_nulls(column);
            match NullBuffer::union(nulls.as_ref(), column.logical_nulls().as_ref()) {
                Some(merged) if column.null_count() != merged.null_count() => {
                    // the field keeps its values; only its validity widens
                    match column.to_data().into_builder().nulls(Some(merged)).build() {
                        Ok(data) => make_array(data),
                        Err(_) => column,
                    }
                }
                _ => column,
            }
        })
        .collect();
    Arc::new(StructArray::new(fields, columns, nulls))
}

/// The rows' column `name`; a dotted name the rows do not hold reads a
/// field of a struct column, as result rows hold an extraction's
/// answers. A dictionary column reads as its values, as result rows
/// hold labels.
fn column_of(batch: &RecordBatch, name: &str) -> Result<ArrayRef, CompileError> {
    if let Some(found) = batch.column_by_name(name) {
        if let DataType::Dictionary(_, values) = found.data_type() {
            return cast(found, values).map_err(|error| CompileError::new(error.to_string()));
        }
        return Ok(with_field_nulls(Arc::clone(found)));
    }
    if let Some((owner, field)) = name.rsplit_once('.')
        && let Ok(owner) = column_of(batch, owner)
        && let Some(found) = owner.as_struct_opt().and_then(|s| s.column_by_name(field))
    {
        return Ok(Arc::clone(found));
    }
    Err(CompileError::new(format!(
        "the rows have no column {}",
        repr_str(name)
    )))
}

/// An [`Expression`] built for evaluation: DataFusion's physical
/// expression over the types of the columns it reads, with DataFusion's
/// type coercion applied once, when it is built.
#[derive(Debug, Clone)]
pub struct PhysicalExpression {
    expression: Expression,
    schema: SchemaRef,
    physical: Arc<dyn PhysicalExpr>,
}

impl PhysicalExpression {
    /// Decode `expression` and build it over the types it names.
    ///
    /// # Errors
    ///
    /// The encoded bytes do not decode, a type does not read, or
    /// DataFusion cannot build the expression over those types.
    pub fn new(expression: &Expression) -> Result<Self, CompileError> {
        let failed = |error: String| {
            CompileError::new(format!(
                "{} cannot be built: {error}",
                expression.text_with(true)
            ))
        };
        // an expression saved before types were written is built over the
        // types of the rows it is evaluated on
        if expression.types.is_empty() && !expression.columns.is_empty() {
            return Ok(Self {
                expression: expression.clone(),
                schema: Arc::new(Schema::empty()),
                physical: datafusion::physical_expr::expressions::lit(true),
            });
        }
        if expression.types.len() != expression.columns.len() {
            return Err(failed(format!(
                "it reads {} columns but names {} types",
                expression.columns.len(),
                expression.types.len()
            )));
        }
        let fields = expression
            .types
            .iter()
            .enumerate()
            .map(|(index, written)| {
                DataType::from_str(written)
                    .map(|data_type| Field::new(position_name(index), data_type, true))
                    .map_err(|error| failed(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let schema = Arc::new(Schema::new(fields));
        let columns = DFSchema::try_from(schema.as_ref().clone())
            .map_err(|error| failed(error.to_string()))?;
        let physical = session()
            .create_physical_expr(decode(expression)?, &columns)
            .map_err(|error| failed(error.to_string()))?;
        Ok(Self {
            expression: expression.clone(),
            schema,
            physical,
        })
    }

    /// Evaluate the expression over `batch`, one result per row.
    ///
    /// A column is read by its name, with its table alias when
    /// `qualified`, as result rows name their columns.
    ///
    /// # Errors
    ///
    /// The rows lack a column the expression reads, a column's type is
    /// not the one the expression was built for, or DataFusion cannot
    /// evaluate it.
    pub fn evaluate(&self, batch: &RecordBatch, qualified: bool) -> Result<ArrayRef, CompileError> {
        let text = || self.expression.text_with(qualified);
        if self.schema.fields().len() != self.expression.columns.len() {
            let mut typed = self.expression.clone();
            typed.types = Vec::new();
            for column in &self.expression.columns {
                let values = column_of(batch, &column.name(qualified))?;
                typed.types.push(plain_type(values.data_type()).to_string());
            }
            return Self::new(&typed)?.evaluate(batch, qualified);
        }
        let mut arrays = Vec::new();
        for (column, field) in self.expression.columns.iter().zip(self.schema.fields()) {
            let name = column.name(qualified);
            let values = column_of(batch, &name)?;
            if plain_type(values.data_type()) != *field.data_type() {
                return Err(CompileError::new(format!(
                    "{} reads {} as {}, but the rows hold {}",
                    text(),
                    repr_str(&name),
                    field.data_type(),
                    values.data_type()
                )));
            }
            arrays.push(values);
        }
        let failed = |error: datafusion::error::DataFusionError| {
            CompileError::new(format!("{} cannot be evaluated: {error}", text()))
        };
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        let bound = RecordBatch::try_new_with_options(Arc::clone(&self.schema), arrays, &options)
            .map_err(|error| failed(error.into()))?;
        self.physical
            .evaluate(&bound)
            .and_then(|values| values.into_array(bound.num_rows()))
            .map_err(failed)
    }
}

/// A value or condition built for evaluation: a column, a field of a
/// value, or a [`PhysicalExpression`].
#[derive(Debug, Clone)]
pub enum PhysicalValue {
    Column(ColumnRef),
    Field(Box<PhysicalValue>, String),
    Expression(PhysicalExpression),
}

impl PhysicalValue {
    /// Build `value` for evaluation, an alias as the value it names.
    ///
    /// # Errors
    ///
    /// The value is a model call, which the model answers, or an
    /// expression DataFusion cannot build.
    pub fn new(value: &ValueExpression) -> Result<Self, CompileError> {
        Ok(match value {
            ValueExpression::ColumnRef(column) => Self::Column(column.clone()),
            ValueExpression::Alias(alias) => Self::new(&alias.expression)?,
            ValueExpression::GetField(field) => {
                Self::Field(Box::new(Self::new(&field.input)?), field.field.clone())
            }
            ValueExpression::Value(expression) => {
                Self::Expression(PhysicalExpression::new(expression)?)
            }
            ValueExpression::ModelCall(_) => {
                return Err(CompileError::new(
                    "a model call is the model's to answer, not an expression to evaluate",
                ));
            }
        })
    }

    /// The columns the value reads.
    pub fn columns(&self) -> Vec<&ColumnRef> {
        match self {
            Self::Column(column) => vec![column],
            Self::Field(input, _) => input.columns(),
            Self::Expression(built) => built.expression.columns.iter().collect(),
        }
    }

    /// Evaluate the value over `batch`, one result per row.
    ///
    /// A column is read by its name, with its table alias when
    /// `qualified`, as result rows name their columns. A condition
    /// gives a boolean array, null where SQL's three-valued logic leaves
    /// it unknown.
    ///
    /// # Errors
    ///
    /// The rows lack a column the value reads, hold it with another
    /// type than the value was built for, or DataFusion cannot evaluate
    /// it.
    pub fn evaluate(&self, batch: &RecordBatch, qualified: bool) -> Result<ArrayRef, CompileError> {
        match self {
            Self::Column(column) => column_of(batch, &column.name(qualified)),
            Self::Field(input, field) => {
                let input = input.evaluate(batch, qualified)?;
                let Some(structs) = input.as_struct_opt() else {
                    return Err(CompileError::new(format!(
                        "field {} reads a value of type {}, which has no fields",
                        repr_str(field),
                        input.data_type()
                    )));
                };
                structs.column_by_name(field).cloned().ok_or_else(|| {
                    CompileError::new(format!("the struct has no field {}", repr_str(field)))
                })
            }
            Self::Expression(built) => built.evaluate(batch, qualified),
        }
    }
}

// ---------------------------------------------------------------- text

/// The text of an expression whose columns are positional, with `{i}`
/// for column `i` and doubled braces elsewhere.
fn template(expr: &Expr) -> String {
    let written = text(expr);
    let mut out = String::new();
    let mut chars = written.chars();
    while let Some(c) = chars.next() {
        match c {
            '{' => out.push_str("{{"),
            '}' => out.push_str("}}"),
            OPEN => {
                out.push('{');
                for digit in chars.by_ref() {
                    if digit == CLOSE {
                        break;
                    }
                    out.push(digit);
                }
                out.push('}');
            }
            other => out.push(other),
        }
    }
    out
}

/// A literal as Python writes it: strings quoted, `True`, `None`.
fn literal_text(value: &ScalarValue) -> String {
    if value.is_null() {
        return "None".to_string();
    }
    match value {
        ScalarValue::Boolean(Some(flag)) => if *flag { "True" } else { "False" }.to_string(),
        ScalarValue::Float32(Some(number)) => repr_float(f64::from(*number)),
        ScalarValue::Float64(Some(number)) => repr_float(*number),
        other => match other.try_as_str().flatten() {
            Some(text) => repr_str(text),
            None => other.to_string(),
        },
    }
}

/// An operand of a binary operator, in parentheses when it is one too.
fn operand(expr: &Expr) -> String {
    match expr {
        Expr::BinaryExpr(_) => format!("({})", text(expr)),
        other => text(other),
    }
}

/// The operands of a chain of `op`, such as `a AND b AND c`.
fn chain(expr: &Expr, op: Operator, found: &mut Vec<String>) {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == op => {
            chain(&binary.left, op, found);
            chain(&binary.right, op, found);
        }
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::And | Operator::Or,
            ..
        }) => found.push(format!("({})", text(expr))),
        other => found.push(text(other)),
    }
}

/// The text `explain()` shows for an expression whose columns are
/// positional: a column as its position between [`OPEN`] and
/// [`CLOSE`], a comparison as `a > 3`, a test as `a IS NULL`, and a call
/// as `name(arguments)`.
fn text(expr: &Expr) -> String {
    match expr {
        Expr::Column(column) => format!("{OPEN}{}{CLOSE}", column.name.trim_start_matches('#')),
        Expr::Alias(alias) => text(&alias.expr),
        Expr::Literal(value, _) => literal_text(value),
        Expr::BinaryExpr(BinaryExpr { op, .. }) if matches!(op, Operator::And | Operator::Or) => {
            let mut operands = Vec::new();
            chain(expr, *op, &mut operands);
            operands.join(&format!(" {op} "))
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            format!("{} {op} {}", operand(left), operand(right))
        }
        Expr::Not(inner) => format!("NOT ({})", text(inner)),
        Expr::IsNull(inner) => format!("{} IS NULL", operand(inner)),
        Expr::IsNotNull(inner) => format!("{} IS NOT NULL", operand(inner)),
        Expr::Negative(inner) => format!("-{}", operand(inner)),
        Expr::InList(list) => {
            let items: Vec<String> = list.list.iter().map(text).collect();
            let found = format!("{} IN ({})", operand(&list.expr), items.join(", "));
            if list.negated {
                format!("NOT ({found})")
            } else {
                found
            }
        }
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            case_insensitive,
            ..
        }) => {
            let name = if *case_insensitive { "ilike" } else { "like" };
            let found = format!("{} {name} {}", operand(expr), text(pattern));
            if *negated {
                format!("NOT ({found})")
            } else {
                found
            }
        }
        Expr::Cast(Cast { expr, field }) | Expr::TryCast(TryCast { expr, field }) => {
            format!("CAST({} AS {})", text(expr), field.data_type())
        }
        Expr::ScalarFunction(call) => {
            let args = &call.args;
            let test = match call.name() {
                "starts_with" => Some("starts with"),
                "ends_with" => Some("ends with"),
                "contains" => Some("contains"),
                "regexp_like" => Some("regex"),
                _ => None,
            };
            match (call.name(), test) {
                (_, Some(test)) if args.len() == 2 => {
                    format!("{} {test} {}", operand(&args[0]), text(&args[1]))
                }
                ("get_field", _) if !args.is_empty() => {
                    let mut found = operand(&args[0]);
                    for field in &args[1..] {
                        match field {
                            Expr::Literal(value, _) => match value.try_as_str().flatten() {
                                Some(name) => found = format!("{found}.{name}"),
                                None => found = format!("{found}[{}]", text(field)),
                            },
                            other => found = format!("{found}[{}]", text(other)),
                        }
                    }
                    found
                }
                (name, _) => {
                    let arguments: Vec<String> = args.iter().map(text).collect();
                    format!("{name}({})", arguments.join(", "))
                }
            }
        }
        other => {
            // DataFusion's own text, each column at its position
            let marked = other
                .clone()
                .transform_up(|node| match &node {
                    Expr::Column(column) => {
                        Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                            format!("{OPEN}{}{CLOSE}", column.name.trim_start_matches('#')),
                        ))))
                    }
                    _ => Ok(Transformed::no(node)),
                })
                .map(|done| done.data)
                .unwrap_or_else(|_| other.clone());
            marked.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{BooleanArray, Int64Array, StringArray};
    use datafusion::prelude::{col, lit};

    use super::*;

    fn reference(alias: &str, column: &str) -> ColumnRef {
        ColumnRef {
            alias: alias.into(),
            provider: "reviews".into(),
            column: column.into(),
        }
    }

    fn bound(expr: &Expr) -> Expression {
        expression(expr, |column| {
            let data_type = if column.name == "stars" {
                DataType::Int64
            } else {
                DataType::Utf8
            };
            Ok((
                reference(
                    column.relation.as_ref().map_or("", |found| found.table()),
                    &column.name,
                ),
                data_type,
            ))
        })
        .unwrap()
    }

    #[test]
    fn an_expression_reads_its_columns_by_position() {
        let found = bound(
            &col("r.stars")
                .gt(lit(3))
                .and(col("r.title").like(lit("a%")).or(col("r.stars").is_null())),
        );
        assert_eq!(
            found.columns,
            vec![reference("r", "stars"), reference("r", "title")]
        );
        assert_eq!(found.types, ["Int64", "Utf8"]);
        assert_eq!(
            found.text_with(true),
            "r.stars > 3 AND (r.title like 'a%' OR r.stars IS NULL)"
        );
    }

    #[test]
    fn a_condition_evaluates_over_rows_named_without_their_alias() {
        let found = bound(
            &col("r.stars")
                .gt(lit(3_i64))
                .and(col("r.title").not_eq(lit("x"))),
        );
        let batch = RecordBatch::try_from_iter([
            (
                "stars",
                Arc::new(Int64Array::from(vec![Some(5), Some(1), None])) as ArrayRef,
            ),
            (
                "title",
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ),
        ])
        .unwrap();
        let built = PhysicalValue::new(&ValueExpression::Value(found.clone())).unwrap();
        let mask = built.evaluate(&batch, false).unwrap();
        assert_eq!(
            mask.as_boolean(),
            &BooleanArray::from(vec![Some(true), Some(false), None])
        );
        // a plan saved before types were written reads them from the rows
        let saved = Expression {
            types: Vec::new(),
            ..found
        };
        let built = PhysicalValue::new(&ValueExpression::Value(saved)).unwrap();
        assert_eq!(
            built.evaluate(&batch, false).unwrap().as_boolean(),
            mask.as_boolean()
        );
    }
}
