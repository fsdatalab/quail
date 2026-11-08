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
use datafusion::arrow::datatypes::DataType;
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
    pub fn of(region: &ModelRegion) -> Self {
        let schema = region.plan.schema();
        let scores = score_columns(&region.plan);
        RegionColumns(
            schema
                .iter()
                .map(|(qualifier, field)| {
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

    #[test]
    fn false_and_undecided_is_false_and_true_or_undecided_is_true() {
        assert_eq!(Truth::False.and(Truth::Undecided), Truth::False);
        assert_eq!(Truth::True.or(Truth::Undecided), Truth::True);
        assert_eq!(Truth::Null.or(Truth::False), Truth::Null);
    }
}
