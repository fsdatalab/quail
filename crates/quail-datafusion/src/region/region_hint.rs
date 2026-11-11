//! A region's hint: the tests above it that it may drop rows by.
//!
//! An operator above a region, such as a top-k sort (`SortExec` with a
//! fetch) or a hash join, can hand the region a dynamic filter: a test
//! of the region's columns that it tightens as it runs. A row where the
//! test is not TRUE leaves the query above anyway, so the region may
//! drop it before asking the model. Each test reads the region's
//! columns by position.
//!
//! A region column the region passes on unchanged from one input column,
//! such as an id, is known for each input row before the model runs. An
//! answer column is not, though its values may be bounded. [`RegionHint`]
//! decides each test per input row over what is known: TRUE, FALSE,
//! NULL, or undecided. A row is dropped only when every value its
//! answers may take leaves the test FALSE or NULL.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column as LogicalColumn, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, Operator};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, Column, IsNotNullExpr, IsNullExpr, NotExpr,
};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr_common::physical_expr::{
    snapshot_generation, snapshot_physical_expr,
};

use crate::ai::AiNode;
use crate::derive::partial::{Completions, LISTED, can_be_true};
use crate::region::ModelRegion;

/// The tests above a region that it may drop rows by, over the region's
/// columns. They may tighten while the query runs.
#[derive(Debug, Clone, Default)]
pub struct RegionHint {
    /// Each test, over the region's columns by position.
    pub tests: Vec<Arc<dyn PhysicalExpr>>,
}

impl RegionHint {
    /// Whether the region has no test to drop rows by.
    pub fn is_empty(&self) -> bool {
        self.tests.is_empty()
    }

    /// A number that changes whenever a test tightens.
    pub fn generation(&self) -> u64 {
        self.tests.iter().fold(0u64, |sum, test| {
            sum.wrapping_add(snapshot_generation(test))
        })
    }

    /// The tables whose passed-on columns a test reads, in the order
    /// the tests first read them.
    pub fn tables(&self, columns: &RegionColumns) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for test in &self.tests {
            for column in collect_columns(test) {
                if let Some(RegionColumn::Source(table, _)) = columns.0.get(column.index())
                    && !found.contains(table)
                {
                    found.push(table.clone());
                }
            }
        }
        found
    }

    /// Whether each row of `rows`, rows of the table `input` the region
    /// reads, may still pass every test as the tests are now.
    ///
    /// # Errors
    ///
    /// A test does not evaluate over the rows.
    pub fn keep(
        &self,
        columns: &RegionColumns,
        input: &str,
        rows: &RecordBatch,
    ) -> Result<Vec<bool>> {
        let mut keep = vec![true; rows.num_rows()];
        for test in &self.tests {
            let test = snapshot_physical_expr(Arc::clone(test))?;
            for (kept, truth) in keep.iter_mut().zip(decide(&test, columns, input, rows)?) {
                *kept &= !matches!(truth, Truth::False | Truth::Null);
            }
        }
        Ok(keep)
    }

    /// Whether a test reads the answer in column `index` of `schema`,
    /// or its field `field` when the column is an extraction's struct,
    /// and no other column.
    pub fn reads_answer(&self, index: usize, field: Option<&str>) -> bool {
        self.tests
            .iter()
            .any(|test| answer_test(test, index, field).is_some())
    }

    /// Whether some test, as the tests are now, is TRUE for no complete
    /// answer still possible in column `index` of `schema`, or in its
    /// field `field`: the row leaves the query above whatever the rest
    /// of the answer says.
    ///
    /// A test that also reads another column, or that DataFusion does
    /// not decide on a partial answer, rejects nothing.
    ///
    /// # Errors
    ///
    /// A test does not give its current form.
    pub fn rejects_answer(
        &self,
        schema: &Schema,
        index: usize,
        field: Option<&str>,
        answers: &Completions,
    ) -> Result<bool> {
        let data_type = match (schema.field(index).data_type(), field) {
            (DataType::Struct(fields), Some(field)) => {
                match fields.iter().find(|found| found.name() == field) {
                    Some(found) => found.data_type().clone(),
                    None => return Ok(false),
                }
            }
            (found, None) => found.clone(),
            _ => return Ok(false),
        };
        for test in &self.tests {
            let test = snapshot_physical_expr(Arc::clone(test))?;
            if let Some(test) = answer_test(&test, index, field)
                && can_be_true(&test, &data_type, answers) == Some(false)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// `test` reading the answer as column 0, named [`LISTED`], when the
/// only column it reads is the answer: column `index`, or `get_field`
/// of it and `field`. None when it reads no answer or another column.
fn answer_test(
    test: &Arc<dyn PhysicalExpr>,
    index: usize,
    field: Option<&str>,
) -> Option<Arc<dyn PhysicalExpr>> {
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion::physical_expr::expressions::Literal;
    let answer: Arc<dyn PhysicalExpr> = Arc::new(Column::new(LISTED, 0));
    let is_column = |expr: &Arc<dyn PhysicalExpr>| {
        expr.downcast_ref::<Column>()
            .is_some_and(|column| column.index() == index)
    };
    let mut found = false;
    let mut other = false;
    let read = Arc::clone(test)
        .transform_down(|node| {
            if let Some(call) = node.downcast_ref::<ScalarFunctionExpr>()
                && call.name() == "get_field"
                && let [input, name] = call.args()
                && is_column(input)
            {
                let named = name
                    .downcast_ref::<Literal>()
                    .and_then(|literal| literal.value().try_as_str().flatten());
                if field.is_some() && named == field {
                    found = true;
                    return Ok(Transformed::new(
                        Arc::clone(&answer),
                        true,
                        TreeNodeRecursion::Jump,
                    ));
                }
                other = true;
                return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
            }
            if node.downcast_ref::<Column>().is_some() {
                if field.is_none() && is_column(&node) {
                    found = true;
                    return Ok(Transformed::yes(Arc::clone(&answer)));
                }
                other = true;
            }
            Ok(Transformed::no(node))
        })
        .ok()?
        .data;
    (found && !other).then_some(read)
}

/// What a region knows about each of its columns before the model runs.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionColumn {
    /// The column of this position in this table, passed on unchanged:
    /// an input by the name the region reads it by, or a table the
    /// region reads by name.
    Source(String, usize),
    /// An answer that is never NULL and lies between the two bounds.
    Bounded(f64, f64),
    /// An answer of any value.
    Answer,
}

/// What a region knows about each of its columns, by position.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionColumns(pub Vec<RegionColumn>);

impl RegionColumns {
    /// Read a region's columns from its plan: each column passed on
    /// from a table, each AI.SCORE answer between 0 and 1, and every
    /// other answer.
    ///
    /// A name two of the region's columns share, such as `id` of two
    /// joined tables, is an answer of any value: DataFusion's filter
    /// pushdown binds a test's column by name, so the test may read
    /// either column.
    pub fn of(region: &ModelRegion) -> Self {
        let schema = region.plan.schema();
        let scores = score_columns(&region.plan);
        let shared = |name: &String| {
            schema
                .fields()
                .iter()
                .filter(|field| field.name() == name)
                .count()
                > 1
        };
        RegionColumns(
            schema
                .iter()
                .map(|(qualifier, field)| {
                    if shared(field.name()) {
                        return RegionColumn::Answer;
                    }
                    let column = LogicalColumn::new(qualifier.cloned(), field.name());
                    if let Some((table, position)) = passed_on(region, &region.plan, &column) {
                        return RegionColumn::Source(table, position);
                    }
                    if qualifier.is_none() && scores.contains(field.name()) {
                        return RegionColumn::Bounded(0.0, 1.0);
                    }
                    RegionColumn::Answer
                })
                .collect(),
        )
    }

    /// The table whose columns a test reads, when it reads only that
    /// table's passed-on columns.
    pub fn input_of(&self, test: &Arc<dyn PhysicalExpr>) -> Option<&str> {
        let mut input: Option<&str> = None;
        for column in collect_columns(test) {
            match self.0.get(column.index()) {
                Some(RegionColumn::Source(found, _))
                    if input.is_none_or(|table| table == found) =>
                {
                    input = Some(found);
                }
                _ => return None,
            }
        }
        input
    }
}

/// The names of the region's AI.SCORE answers. A score is the model's
/// probability of its yes answer, or the mean of several, from one
/// readout per row: it is never NULL and lies between 0 and 1, and no
/// partial answer bounds it more tightly before the readout.
fn score_columns(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::Extension(extension) => {
                if let Some(ai) = extension.node.as_any().downcast_ref::<AiNode>()
                    && ai.function() == "ai_score"
                {
                    found.extend(ai.fields.iter().map(|field| field.name().clone()));
                }
            }
            LogicalPlan::Projection(projection) => {
                for (expr, field) in projection.expr.iter().zip(projection.schema.fields()) {
                    let mut inner = expr;
                    while let Expr::Alias(alias) = inner {
                        inner = &alias.expr;
                    }
                    if let Expr::ScalarFunction(call) = inner
                        && call.name() == "ai_score"
                    {
                        found.push(field.name().clone());
                    }
                }
            }
            _ => {}
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    found
}

/// The table and column position a plan's column passes on unchanged,
/// through projections, aliases, and the operators that keep their
/// input's columns.
fn passed_on(
    region: &ModelRegion,
    plan: &LogicalPlan,
    column: &LogicalColumn,
) -> Option<(String, usize)> {
    let index = plan.schema().maybe_index_of_column(column)?;
    match plan {
        LogicalPlan::Projection(projection) => {
            let mut expr = &projection.expr[index];
            while let datafusion::logical_expr::Expr::Alias(alias) = expr {
                expr = &alias.expr;
            }
            match expr {
                datafusion::logical_expr::Expr::Column(inner) => {
                    passed_on(region, &projection.input, inner)
                }
                _ => None,
            }
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let (qualifier, field) = alias.input.schema().qualified_field(index);
            passed_on(
                region,
                &alias.input,
                &LogicalColumn::new(qualifier.cloned(), field.name()),
            )
        }
        LogicalPlan::TableScan(scan) => {
            let table = scan.table_name.table();
            let name = plan.schema().field(index).name();
            let found = match region.input_names.iter().position(|found| found == table) {
                Some(input) => region.inputs[input].schema().as_arrow().index_of(name),
                None => scan.source.schema().index_of(name),
            };
            Some((table.to_string(), found.ok()?))
        }
        _ => {
            let inputs = plan.inputs();
            let holder = inputs
                .iter()
                .find(|input| input.schema().maybe_index_of_column(column).is_some())?;
            passed_on(region, holder, column)
        }
    }
}

/// What a test is for one row, before the region's answers are known.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Truth {
    True,
    False,
    Null,
    Undecided,
}

impl Truth {
    fn of(value: Option<bool>) -> Self {
        match value {
            Some(true) => Truth::True,
            Some(false) => Truth::False,
            None => Truth::Null,
        }
    }

    fn and(self, other: Truth) -> Truth {
        use Truth::*;
        match (self, other) {
            (False, _) | (_, False) => False,
            (Undecided, _) | (_, Undecided) => Undecided,
            (Null, _) | (_, Null) => Null,
            _ => True,
        }
    }

    fn or(self, other: Truth) -> Truth {
        use Truth::*;
        match (self, other) {
            (True, _) | (_, True) => True,
            (Undecided, _) | (_, Undecided) => Undecided,
            (Null, _) | (_, Null) => Null,
            _ => False,
        }
    }

    fn not(self) -> Truth {
        match self {
            Truth::True => Truth::False,
            Truth::False => Truth::True,
            other => other,
        }
    }
}

/// A test's truth for each input row.
fn decide(
    test: &Arc<dyn PhysicalExpr>,
    columns: &RegionColumns,
    input: &str,
    rows: &RecordBatch,
) -> Result<Vec<Truth>> {
    let num_rows = rows.num_rows();
    if let Some(values) = input_values(test, columns, input, rows)? {
        let values = cast(&values, &DataType::Boolean)?;
        let values = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("a boolean array");
        return Ok((0..num_rows)
            .map(|row| Truth::of((!values.is_null(row)).then(|| values.value(row))))
            .collect());
    }
    if let Some(not) = test.downcast_ref::<NotExpr>() {
        return Ok(decide(not.arg(), columns, input, rows)?
            .into_iter()
            .map(Truth::not)
            .collect());
    }
    let bounded = |expr: &Arc<dyn PhysicalExpr>| {
        expr.downcast_ref::<Column>()
            .and_then(|column| match columns.0.get(column.index()) {
                Some(RegionColumn::Bounded(low, high)) => Some((*low, *high)),
                _ => None,
            })
    };
    for (expr, null) in [
        (test.downcast_ref::<IsNullExpr>().map(IsNullExpr::arg), true),
        (
            test.downcast_ref::<IsNotNullExpr>().map(IsNotNullExpr::arg),
            false,
        ),
    ] {
        if expr.is_some_and(|expr| bounded(expr).is_some()) {
            // a bounded answer is never NULL
            let truth = if null { Truth::False } else { Truth::True };
            return Ok(vec![truth; num_rows]);
        }
    }
    let Some(binary) = test.downcast_ref::<BinaryExpr>() else {
        return Ok(vec![Truth::Undecided; num_rows]);
    };
    let (left, right) = (binary.left(), binary.right());
    match binary.op() {
        Operator::And | Operator::Or => {
            let left = decide(left, columns, input, rows)?;
            let right = decide(right, columns, input, rows)?;
            let and = matches!(binary.op(), Operator::And);
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(l, r)| if and { l.and(r) } else { l.or(r) })
                .collect())
        }
        op => {
            let (bounds, other, op) = match (bounded(left), bounded(right)) {
                (Some(bounds), None) => (bounds, right, *op),
                (None, Some(bounds)) => match op.swap() {
                    Some(swapped) => (bounds, left, swapped),
                    None => return Ok(vec![Truth::Undecided; num_rows]),
                },
                _ => return Ok(vec![Truth::Undecided; num_rows]),
            };
            let Some(values) = input_values(other, columns, input, rows)? else {
                return Ok(vec![Truth::Undecided; num_rows]);
            };
            let values = cast(&values, &DataType::Float64)?;
            let values = values
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("a float array");
            Ok((0..num_rows)
                .map(|row| {
                    if values.is_null(row) {
                        return Truth::Null;
                    }
                    compare(bounds, op, values.value(row))
                })
                .collect())
        }
    }
}

/// `x op value` for every x between the bounds: TRUE or FALSE when it
/// is the same for all of them.
fn compare((low, high): (f64, f64), op: Operator, value: f64) -> Truth {
    let (always, never) = match op {
        Operator::Gt => (low > value, high <= value),
        Operator::GtEq => (low >= value, high < value),
        Operator::Lt => (high < value, low >= value),
        Operator::LtEq => (high <= value, low > value),
        Operator::Eq => (low == value && high == value, value < low || value > high),
        Operator::NotEq => (value < low || value > high, low == value && high == value),
        _ => (false, false),
    };
    if always {
        Truth::True
    } else if never {
        Truth::False
    } else {
        Truth::Undecided
    }
}

/// An expression's value for each input row, when it reads only
/// constants and the input's passed-on columns.
fn input_values(
    expr: &Arc<dyn PhysicalExpr>,
    columns: &RegionColumns,
    input: &str,
    rows: &RecordBatch,
) -> Result<Option<ArrayRef>> {
    let reads_columns = !collect_columns(expr).is_empty();
    if reads_columns && columns.input_of(expr) != Some(input) {
        return Ok(None);
    }
    let rewritten = Arc::clone(expr)
        .transform(|node| {
            let Some(column) = node.downcast_ref::<Column>() else {
                return Ok(Transformed::no(node));
            };
            let Some(RegionColumn::Source(_, position)) = columns.0.get(column.index()) else {
                return Ok(Transformed::no(node));
            };
            let name = rows.schema().field(*position).name().clone();
            Ok(Transformed::yes(
                Arc::new(Column::new(&name, *position)) as Arc<dyn PhysicalExpr>
            ))
        })?
        .data;
    Ok(Some(rewritten.evaluate(rows)?.into_array(rows.num_rows())?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_score_between_0_and_1_decides_a_threshold_of_1() {
        let bounds = (0.0, 1.0);
        assert_eq!(compare(bounds, Operator::Gt, 1.0), Truth::False);
        assert_eq!(compare(bounds, Operator::Gt, 0.9), Truth::Undecided);
        assert_eq!(compare(bounds, Operator::Eq, 1.0), Truth::Undecided);
        assert_eq!(compare(bounds, Operator::GtEq, -0.5), Truth::True);
    }

    /// `expr`, over the columns of `schema`, as the one test of a hint.
    fn hint_of(schema: &Schema, expr: Expr) -> RegionHint {
        use datafusion::common::DFSchema;
        let fields = DFSchema::try_from(schema.clone()).unwrap();
        let test = datafusion::prelude::SessionContext::new()
            .create_physical_expr(expr, &fields)
            .unwrap();
        RegionHint { tests: vec![test] }
    }

    #[test]
    fn a_top_k_threshold_rejects_a_map_answer_that_cannot_sort_before_it() {
        use datafusion::arrow::datatypes::Field;
        use datafusion::logical_expr::{col, lit};
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vendor", DataType::Utf8, true),
        ]);
        // the sort keeps the vendors that sort before Globex
        let hint = hint_of(&schema, col("vendor").lt(lit("Globex")));
        assert!(hint.reads_answer(1, None));
        assert!(!hint.reads_answer(0, None));
        let rejects = |prefix: &str| {
            hint.rejects_answer(&schema, 1, None, &Completions::text(prefix))
                .unwrap()
        };
        assert!(rejects("Initech"));
        assert!(rejects("H"));
        assert!(!rejects("Acme"));
        // an answer that starts as the threshold does may still sort first
        assert!(!rejects("G"));
        // a test that also reads the id is not the answer's alone
        let both = hint_of(&schema, col("vendor").lt(col("id")));
        assert!(!both.reads_answer(1, None));
    }

    #[test]
    fn a_join_key_set_rejects_an_extracted_field_that_can_match_no_key() {
        use datafusion::arrow::datatypes::{Field, Fields};
        use datafusion::functions::core::expr_fn::get_field;
        use datafusion::logical_expr::{col, lit};
        let answer = DataType::Struct(Fields::from(vec![
            Field::new("law", DataType::Utf8, true),
            Field::new("party", DataType::Utf8, true),
        ]));
        let schema = Schema::new(vec![Field::new("contract", answer, true)]);
        let keys = vec![lit("Delaware"), lit("New York")];
        let hint = hint_of(
            &schema,
            get_field(col("contract"), "law").in_list(keys, false),
        );
        assert!(hint.reads_answer(0, Some("law")));
        assert!(!hint.reads_answer(0, Some("party")));
        // the span copies the document, which goes on as one of two texts
        let copied = |prefix: &str, tails: &[&str]| Completions {
            prefix: prefix.to_string(),
            complete: false,
            may_be_null: false,
            tails: Some(tails.iter().map(|tail| (*tail).to_string()).collect()),
            max_chars: None,
        };
        let rejects = |answers: &Completions| {
            hint.rejects_answer(&schema, 0, Some("law"), answers)
                .unwrap()
        };
        assert!(rejects(&copied("Texas", &[" law", " courts"])));
        assert!(!rejects(&copied("New", &[" York law", " Jersey"])));
        assert!(rejects(&copied("New", &[" Jersey", " Mexico"])));
    }

    #[test]
    fn false_and_undecided_is_false_and_true_or_undecided_is_true() {
        assert_eq!(Truth::False.and(Truth::Undecided), Truth::False);
        assert_eq!(Truth::True.or(Truth::Undecided), Truth::True);
        assert_eq!(Truth::Null.or(Truth::False), Truth::Null);
    }
}
