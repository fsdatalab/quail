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
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::{DFSchema, DFSchemaRef, Result, plan_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// A selectivity hint, compared and hashed by its bits.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Selectivity(pub f64);

impl Eq for Selectivity {}

impl Hash for Selectivity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// One model call: what it asks of each row, and the columns it reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct AiCall {
    /// "boolean" for `AI.IF`, or "score", "label", "extract", "map", "features".
    pub kind: String,
    /// The prompt text, with `{0}`, `{1}`, ... where the arguments go.
    pub template: String,
    /// The columns the prompt reads, in placeholder order.
    pub args: Vec<Expr>,
    /// The call's other options as JSON: labels, descriptions,
    /// probabilities, the question, pooling, and the answer token limit.
    pub options: String,
}

impl AiCall {
    fn fmt_call(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(ToString::to_string).collect();
        write!(
            f,
            "{}({:?}, [{}])",
            self.kind,
            self.template,
            args.join(", ")
        )
    }
}

/// One predicate of a [`SemanticFilter`] or a [`SemanticJoin`]: a call,
/// and for a score, the comparison it passes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct Asked {
    pub call: AiCall,
    /// For `AI.SCORE`, the comparison and the constant, such as (">", 0.5).
    pub comparison: Option<(String, String)>,
    /// The fraction of rows the query says pass, if it says.
    pub selectivity: Option<Selectivity>,
}

impl Asked {
    fn fmt_asked(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.call.fmt_call(f)?;
        if let Some((comparison, value)) = &self.comparison {
            write!(f, " {comparison} {value}")?;
        }
        if let Some(selectivity) = self.selectivity {
            write!(f, " (selectivity {})", selectivity.0)?;
        }
        Ok(())
    }
}

/// The rows of one table that pass every predicate, as `SemanticFilter`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticFilter {
    pub input: Arc<LogicalPlan>,
    pub predicates: Vec<Asked>,
    /// The columns a DISTINCT keys the table's rows by: the executor
    /// stops a key at its first survivor.
    pub stop_key: Vec<String>,
}

/// The pairs of the input join that pass one predicate, as `SemanticJoin`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticJoin {
    pub input: Arc<LogicalPlan>,
    pub predicate: Asked,
    /// "full", or a gate: "semi" or "anti".
    pub semantics: String,
    /// The table a query asked to anchor the join on, if any.
    pub anchor: Option<String>,
}

/// Each row of the input with the answer of one call, as `SemanticColumn`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticColumn {
    pub input: Arc<LogicalPlan>,
    pub call: AiCall,
    /// The answer column's name.
    pub name: String,
    /// The columns the call adds, the answer first: a classification's
    /// probabilities, an extraction's spans.
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
        call: AiCall,
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
/// columns the calls read: what a node needs of its input.
fn needed_columns(
    input: &LogicalPlan,
    output_columns: &[usize],
    calls: &[&AiCall],
) -> Option<Vec<usize>> {
    let width = input.schema().fields().len();
    let mut needed: Vec<usize> = output_columns
        .iter()
        .copied()
        .filter(|index| *index < width)
        .collect();
    for call in calls {
        for arg in &call.args {
            for column in arg.column_refs() {
                needed.push(input.schema().index_of_column(column).ok()?);
            }
        }
    }
    needed.sort_unstable();
    needed.dedup();
    Some(needed)
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
        self.predicates
            .iter()
            .flat_map(|predicate| predicate.call.args.iter().cloned())
            .collect()
    }

    // a test of the table's own columns runs before the model's
    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let calls: Vec<&AiCall> = self
            .predicates
            .iter()
            .map(|predicate| &predicate.call)
            .collect();
        Some(vec![needed_columns(&self.input, output_columns, &calls)?])
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticFilter: ")?;
        for (index, predicate) in self.predicates.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            predicate.fmt_asked(f)?;
        }
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
        let mut exprs = exprs.into_iter();
        let mut predicates = self.predicates.clone();
        for predicate in &mut predicates {
            for arg in &mut predicate.call.args {
                *arg = exprs.next().expect("an expression per argument");
            }
        }
        Ok(Self {
            input: Arc::new(inputs.remove(0)),
            predicates,
            stop_key: self.stop_key.clone(),
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
        self.predicate.call.args.clone()
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![needed_columns(
            &self.input,
            output_columns,
            &[&self.predicate.call],
        )?])
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticJoin: {} ", self.semantics)?;
        self.predicate.fmt_asked(f)?;
        if let Some(anchor) = &self.anchor {
            write!(f, " anchor {anchor}")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let mut predicate = self.predicate.clone();
        predicate.call.args = exprs;
        Ok(Self {
            input: Arc::new(inputs.remove(0)),
            predicate,
            semantics: self.semantics.clone(),
            anchor: self.anchor.clone(),
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
        self.call.args.clone()
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
        write!(f, "SemanticColumn: {} = ", self.name)?;
        self.call.fmt_call(f)
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let mut call = self.call.clone();
        call.args = exprs;
        Self::try_new(
            Arc::new(inputs.remove(0)),
            call,
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

    fn ask(template: &str) -> Asked {
        Asked {
            call: AiCall {
                kind: "boolean".into(),
                template: template.into(),
                args: vec![col("r.body")],
                options: "{}".into(),
            },
            comparison: None,
            selectivity: Some(Selectivity(0.4)),
        }
    }

    #[test]
    fn a_semantic_filter_keeps_its_input_schema_and_reads_its_prompt_columns() {
        let node = SemanticFilter {
            input: Arc::new(reviews()),
            predicates: vec![ask("Is {0} kind?")],
            stop_key: Vec::new(),
        };
        assert_eq!(node.schema(), reviews().schema());
        assert_eq!(node.expressions(), vec![col("r.body")]);
        let rebuilt = node
            .with_exprs_and_inputs(node.expressions(), vec![reviews()])
            .unwrap();
        assert_eq!(rebuilt, node);
    }

    #[test]
    fn a_semantic_column_adds_its_answer_after_the_input_columns() {
        let call = AiCall {
            kind: "label".into(),
            template: "Tone of {0}?".into(),
            args: vec![col("r.body")],
            options: r#"{"labels": ["good", "bad"]}"#.into(),
        };
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
