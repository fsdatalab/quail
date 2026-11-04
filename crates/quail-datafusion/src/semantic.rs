//! Quail's logical operators as DataFusion plan nodes.
//!
//! Each operator asks the model about rows of its input and is a
//! DataFusion extension node, so Quail's operators live in DataFusion's
//! own plan and DataFusion's rules run around them:
//!
//! - [`SemanticFilter`] keeps the rows of one table that pass its
//!   `AI.IF` predicates (or `AI.SCORE` comparisons), in written order.
//! - [`SemanticJoin`] keeps the pairs of its input join that pass one
//!   `AI.IF` over a row of each table.
//! - [`SemanticColumn`] adds the answer of one `AI.CLASSIFY`,
//!   `AI.EXTRACT`, `AI.MAP`, or `AI.SAE_FEATURES` call to each row.
//!
//! The columns a call reads are DataFusion expressions, which
//! `expressions()` returns; the rest of the call (its template, kind,
//! and options) is fixed. A prompt is not rendered for a model here:
//! physical planning renders it for the model it plans for.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::{DFSchema, DFSchemaRef, Result, plan_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// The rows of one table that pass every predicate, as `SemanticFilter`.
///
/// Each predicate is the query's own term: an `AI.IF` call, or a
/// comparison of an `AI.SCORE` call, either inside a `HINT`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticFilter {
    pub input: Arc<LogicalPlan>,
    pub predicates: Vec<Expr>,
    /// The columns a DISTINCT keys the table's rows by: the executor
    /// stops a key at its first survivor.
    pub stop_key: Vec<String>,
    /// Where each predicate was written among the terms of its WHERE.
    pub written: Written,
}

/// Where a node's terms were written among the terms of one WHERE or
/// ON: each term's position, and how many terms it has.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Written {
    pub positions: Vec<usize>,
    pub num_terms: usize,
}

/// The pairs of the input join that pass one predicate, as `SemanticJoin`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticJoin {
    pub input: Arc<LogicalPlan>,
    /// The query's own term: an `AI.IF` call over a row of each table.
    pub predicate: Expr,
    /// "full", or a gate: "semi" or "anti".
    pub semantics: String,
    /// Whether the query wrote the predicate in the join's ON rather
    /// than in WHERE.
    pub in_on: bool,
    /// Where the predicate was written among the terms of its WHERE or ON.
    pub written: Written,
}

/// Each row of the input with the answer of one call, as `SemanticColumn`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticColumn {
    pub input: Arc<LogicalPlan>,
    /// The query's own call: `AI.CLASSIFY`, `AI.EXTRACT`, `AI.MAP`, or
    /// `AI.SAE_FEATURES`.
    pub call: Expr,
    /// The answer column's name.
    pub name: String,
    /// The columns the call adds, the answer first.
    pub fields: Vec<Field>,
    schema: DFSchemaRef,
}

impl SemanticColumn {
    /// A column operator adding `fields` to its input's rows.
    ///
    /// # Errors
    ///
    /// A field's name is already a column of the input.
    pub fn try_new(
        input: Arc<LogicalPlan>,
        call: Expr,
        name: String,
        fields: Vec<Field>,
    ) -> Result<Self> {
        let added = DFSchema::from_unqualified_fields(fields.clone().into(), Default::default())?;
        let mut schema = input.schema().as_ref().clone();
        schema.merge(&added);
        if schema.fields().len() != input.schema().fields().len() + fields.len() {
            return plan_err!("{name} is already a column of the input");
        }
        Ok(Self {
            input,
            call,
            name,
            fields,
            schema: Arc::new(schema),
        })
    }
}

/// The positions in `input` of the output columns asked for and of the
/// columns the terms read: what a node needs of its input.
fn needed_columns(
    input: &LogicalPlan,
    output_columns: &[usize],
    terms: &[&Expr],
) -> Option<Vec<usize>> {
    let width = input.schema().fields().len();
    let mut needed: Vec<usize> = output_columns
        .iter()
        .copied()
        .filter(|index| *index < width)
        .collect();
    for term in terms {
        for column in term.column_refs() {
            needed.push(input.schema().index_of_column(column).ok()?);
        }
    }
    needed.sort_unstable();
    needed.dedup();
    Some(needed)
}

fn joined(terms: &[Expr]) -> String {
    terms
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

macro_rules! ordered_by_name {
    ($($node:ty),*) => {$(
        impl PartialOrd for $node {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // DataFusion orders nodes only to compare plans; the
                // printed node decides
                format!("{self:?}").partial_cmp(&format!("{other:?}"))
            }
        }
    )*};
}

ordered_by_name!(SemanticFilter, SemanticJoin, SemanticColumn);

impl UserDefinedLogicalNodeCore for SemanticFilter {
    fn name(&self) -> &str {
        "SemanticFilter"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        self.predicates.clone()
    }

    // a test of the table's own columns runs before the model's
    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let terms: Vec<&Expr> = self.predicates.iter().collect();
        Some(vec![needed_columns(&self.input, output_columns, &terms)?])
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticFilter: {}", joined(&self.predicates))?;
        if !self.stop_key.is_empty() {
            write!(f, " stop key [{}]", self.stop_key.join(", "))?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        Ok(Self {
            input: Arc::new(inputs.remove(0)),
            predicates: exprs,
            stop_key: self.stop_key.clone(),
            written: self.written.clone(),
        })
    }
}

impl UserDefinedLogicalNodeCore for SemanticJoin {
    fn name(&self) -> &str {
        "SemanticJoin"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.predicate.clone()]
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![needed_columns(
            &self.input,
            output_columns,
            &[&self.predicate],
        )?])
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticJoin: {} {}", self.semantics, self.predicate)
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        Ok(Self {
            input: Arc::new(inputs.remove(0)),
            predicate: exprs.into_iter().next().expect("the join's predicate"),
            semantics: self.semantics.clone(),
            in_on: self.in_on,
            written: self.written.clone(),
        })
    }
}

impl UserDefinedLogicalNodeCore for SemanticColumn {
    fn name(&self) -> &str {
        "SemanticColumn"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.call.clone()]
    }

    // a test of the answer waits for it; any other runs first
    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        self.fields
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![needed_columns(
            &self.input,
            output_columns,
            &[&self.call],
        )?])
    }

    // the answer is one per row, so a limit can read fewer rows first
    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticColumn: {} = {}", self.name, self.call)
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        Self::try_new(
            Arc::new(inputs.remove(0)),
            exprs.into_iter().next().expect("the column's call"),
            self.name.clone(),
            self.fields.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::{Extension, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    fn reviews() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
            Field::new("stars", DataType::Int64, true),
        ]));
        LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .alias("r")
        .unwrap()
        .build()
        .unwrap()
    }

    fn ask(template: &str) -> Expr {
        Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|function| function.name() == "ai_filter")
                .unwrap(),
            vec![lit(template), col("r.body")],
        ))
    }

    #[test]
    fn a_semantic_filter_keeps_its_input_schema_and_reads_its_prompt_columns() {
        let node = SemanticFilter {
            input: Arc::new(reviews()),
            predicates: vec![ask("Is {0} kind?")],
            stop_key: Vec::new(),
            written: Written::default(),
        };
        assert_eq!(node.schema(), reviews().schema());
        assert_eq!(node.expressions(), vec![ask("Is {0} kind?")]);
        let rebuilt = node
            .with_exprs_and_inputs(node.expressions(), vec![reviews()])
            .unwrap();
        assert_eq!(rebuilt, node);
    }

    #[test]
    fn a_semantic_column_adds_its_answer_after_the_input_columns() {
        let call = ask("Tone of {0}?");
        let node = SemanticColumn::try_new(
            Arc::new(reviews()),
            call,
            "tone".into(),
            vec![Field::new("tone", DataType::Utf8, true)],
        )
        .unwrap();
        let names: Vec<String> = node
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["id", "body", "stars", "tone"]);
    }

    #[test]
    fn datafusion_optimizes_a_plan_around_a_semantic_filter() {
        let filter = LogicalPlan::Extension(Extension {
            node: Arc::new(SemanticFilter {
                input: Arc::new(reviews()),
                predicates: vec![ask("Is {0} kind?")],
                stop_key: Vec::new(),
                written: Written::default(),
            }),
        });
        let plan = LogicalPlanBuilder::from(filter)
            .filter(col("r.stars").gt(lit(3)))
            .unwrap()
            .project(vec![col("r.id")])
            .unwrap()
            .build()
            .unwrap();
        let optimized = Optimizer::new()
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap();
        let text = optimized.display_indent().to_string();
        // the table's own test runs below the model's, and the scan reads
        // only the columns above it and the prompt's
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let semantic = lines
            .iter()
            .position(|line| line.starts_with("SemanticFilter"))
            .unwrap();
        let test = lines
            .iter()
            .position(|line| line.contains("stars > Int32(3)"))
            .unwrap();
        assert!(test > semantic, "{text}");
        assert!(
            text.contains("Projection: reviews.id, reviews.body"),
            "{text}"
        );
    }
}
