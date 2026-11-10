//! Quail's AI operator as a DataFusion plan node.
//!
//! [`AiNode`] asks the model one AI call `f` about each row of its input
//! `R`, adding the call's answer as the column `a`:
//!
//! ```text
//! AI(R; g, p, h, P, n) = π_P(Limit_n(σ_p(π_{*, f_g → a}(R))))
//! f_g = CASE WHEN g THEN f END
//! ```
//!
//! - `g`, the guard, says which rows the model is asked about.
//! - `p`, the filter, is an exact test of the input's columns and `a`.
//! - `h`, the hint, is the query's inexact `HINT` options, and the hint
//!   test: the node may drop rows where it is not TRUE. A node with a
//!   hint test has no fetch, and the filter that holds the test stays
//!   above it.
//! - `P`, the projection, names the added columns the node returns.
//! - `n`, the fetch, keeps the first rows: of all rows, or of each key.
//!
//! An `AI.IF` filter is the node with `p = a` and none of the added
//! columns returned; a comparison of an `AI.SCORE` is the node with `p`
//! the comparison. An `AI.IF` join is the node over a join of its inputs.
//! An `EXISTS` gate is the node over a join, returning the left input's
//! rows that have a pair passing `p` (semi) or none (anti).
//!
//! The call's function names the node's kind: `ai_filter` (`AI.IF`),
//! `ai_score`, `ai_classify`, `ai_extract`, `ai_map`, or
//! `ai_sae_features`. The columns the call, guard, and filter read are
//! DataFusion expressions, which `expressions()` returns; a prompt is not
//! rendered for a model here: physical planning renders it for the model
//! it plans for.
//!
//! DataFusion's optimizer rules move filters and column lists around the
//! node as its `UserDefinedLogicalNodeCore` methods allow; each method
//! says which rewrites keep the node's meaning. A LIMIT or DISTINCT above
//! the node becomes its fetch (`PushDownLimitIntoAiNodes`,
//! `PushDownDistinct`).

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::{DFSchema, DFSchemaRef, Result, internal_err, plan_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Which rows an [`AiNode`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Semantics {
    /// Each row that passes the filter.
    Rows,
    /// Each row of the input join's left input with a pair that passes.
    Semi,
    /// Each row of the input join's left input with no pair that passes.
    Anti,
}

/// How many rows an [`AiNode`] returns after its filter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Fetch {
    /// The first `n` rows.
    Global(usize),
    /// The first `n` rows of each value of the `key` columns: a DISTINCT
    /// on `key` stops each key at its first row.
    PerKey { key: Vec<String>, n: usize },
}

/// Where the query wrote an AI predicate's term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Written {
    /// The term's position among the terms of its WHERE or ON.
    pub position: usize,
    /// How many terms that WHERE or ON has.
    pub num_terms: usize,
    /// Which WHERE or ON it is, numbered as the lowering meets them.
    pub clause: usize,
    /// The predicate's number among the region's AI predicates, in
    /// written order.
    pub index: usize,
    /// The alias of the table whose JOIN's ON holds the term, or None
    /// for a WHERE.
    pub on: Option<String>,
}

/// A test of answer columns, as the query wrote it in a WHERE.
///
/// The optimizer rules move the test with the columns it reads, which
/// loses its place among the WHERE's terms; the `AiNode`s whose answers
/// it reads keep it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WrittenTest {
    /// The test as the plan holds it once DataFusion's
    /// `SimplifyExpressions` has rewritten it, such as
    /// `sentiment = 'negative' OR sentiment = 'mixed'`.
    pub test: Expr,
    /// The test with each AI call as the query wrote it, such as
    /// `ai_classify(...) IN ('negative')`.
    pub written: Expr,
    /// The term's position among the terms of its WHERE.
    pub position: usize,
    /// Which WHERE it is, numbered as the lowering meets them.
    pub clause: usize,
}

fn filters_on_or(node: &AiNode) -> bool {
    crate::rules::or_guards::or_terms(node).is_some()
}

/// Whether an `AiNode` in `plan` filters on an OR.
pub(crate) fn any_filters_on_or(plan: &LogicalPlan) -> bool {
    datafusion::common::tree_node::TreeNode::exists(plan, |below| {
        Ok(crate::extension::as_node::<AiNode>(below).is_some_and(filters_on_or))
    })
    .unwrap_or(false)
}

/// One AI call over the rows of its input, as `AiNode`; the module
/// explains each part.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AiNode {
    pub input: Arc<LogicalPlan>,
    /// The query's own call, such as `ai_classify(...)`, without its `HINT`.
    pub call: Expr,
    /// The answer column's name.
    pub answer: String,
    /// The columns the call adds, the answer first.
    pub fields: Vec<Field>,
    pub guard: Option<Expr>,
    pub filter: Option<Expr>,
    /// The options of the `HINT` the query wraps the call's term in,
    /// such as `{'selectivity': 0.4}`.
    pub hint: Option<Expr>,
    /// A test of the input's columns and the added columns: the node may
    /// drop rows where it is not TRUE, as an executor that stops an
    /// answer once the test is decided.
    pub hint_test: Option<Expr>,
    /// The positions in `fields` of the added columns the node returns;
    /// None returns every one.
    pub projection: Option<Vec<usize>>,
    pub fetch: Option<Fetch>,
    pub semantics: Semantics,
    /// Where the query wrote the call's term, if in a WHERE or ON.
    pub written: Option<Written>,
    /// The tests of the node's answers the query wrote in a WHERE.
    pub tests: Vec<WrittenTest>,
    schema: DFSchemaRef,
}

impl AiNode {
    /// A node adding `fields`, the answer of `call` first, to each row of
    /// `input`, and returning every row with every added column.
    ///
    /// # Errors
    ///
    /// A field's name is already a column of the input.
    pub fn try_new(
        input: Arc<LogicalPlan>,
        call: Expr,
        answer: String,
        fields: Vec<Field>,
    ) -> Result<Self> {
        Self {
            input,
            call,
            answer,
            fields,
            guard: None,
            filter: None,
            hint: None,
            hint_test: None,
            projection: None,
            fetch: None,
            semantics: Semantics::Rows,
            written: None,
            tests: Vec::new(),
            schema: Arc::new(DFSchema::empty()),
        }
        .checked()
    }

    /// This node with its schema computed from its parts.
    ///
    /// # Errors
    ///
    /// A field's name is already a column of the input, a projection
    /// position is not a field's, or a gate's input is not a join.
    pub fn checked(mut self) -> Result<Self> {
        let added =
            DFSchema::from_unqualified_fields(self.fields.clone().into(), Default::default())?;
        let mut all = self.input.schema().as_ref().clone();
        all.merge(&added);
        if all.fields().len() != self.input.schema().fields().len() + self.fields.len() {
            return plan_err!("{} is already a column of the input", self.answer);
        }
        if let Some(projection) = &self.projection
            && projection.iter().any(|index| *index >= self.fields.len())
        {
            return plan_err!("{} returns a column it does not add", self.answer);
        }
        self.schema = match self.semantics {
            Semantics::Rows => {
                let kept = DFSchema::from_unqualified_fields(
                    self.returned_fields().into(),
                    Default::default(),
                )?;
                let mut schema = self.input.schema().as_ref().clone();
                schema.merge(&kept);
                Arc::new(schema)
            }
            Semantics::Semi | Semantics::Anti => match self.input.as_ref() {
                LogicalPlan::Join(join) => Arc::clone(join.left.schema()),
                _ => return plan_err!("a {} gate reads a join", self.semantics_name()),
            },
        };
        Ok(self)
    }

    /// Whether the node can take a fetch, a number of passing rows after
    /// which the executor stops asking the model.
    ///
    /// A node with a `HINT` cannot: the hint's options estimate the rows
    /// of its test. A node that filters on an OR cannot, nor can a node
    /// above one: the executor counts the rows the model passes, not the
    /// rows the OR's other terms pass.
    pub fn accepts_fetch(&self) -> bool {
        self.hint.is_none() && !filters_on_or(self) && !any_filters_on_or(&self.input)
    }

    /// Whether `term` is the node's answer column or its `NOT`, and which:
    /// true for the `NOT`. None for any other term.
    pub fn answer_term(&self, term: &Expr) -> Option<bool> {
        let (read, negated) = match term {
            Expr::Not(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        matches!(read, Expr::Column(column)
            if column.relation.is_none() && column.name == self.answer)
        .then_some(negated)
    }

    /// This node over `input` instead of its own.
    ///
    /// # Errors
    ///
    /// As [`AiNode::checked`].
    pub fn with_input(&self, input: LogicalPlan) -> Result<Self> {
        Self {
            input: Arc::new(input),
            ..self.clone()
        }
        .checked()
    }

    /// The name of the call's function, such as `ai_classify`.
    pub fn function(&self) -> &str {
        match &self.call {
            Expr::ScalarFunction(call) => call.name(),
            _ => "",
        }
    }

    /// Whether the call uses a predicate operator: every `AI.IF` (including
    /// one that keeps all rows and returns its answer), or a tested `AI.SCORE`.
    pub fn is_predicate(&self) -> bool {
        self.function() == "ai_filter" || (self.function() == "ai_score" && self.filter.is_some())
    }

    /// Whether the query wrote the call's term in a JOIN's ON.
    pub fn in_on(&self) -> bool {
        self.written
            .as_ref()
            .is_some_and(|written| written.on.is_some())
    }

    /// The added columns the node returns.
    pub fn returned_fields(&self) -> Vec<Field> {
        match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|index| self.fields[*index].clone())
                .collect(),
            None => self.fields.clone(),
        }
    }

    fn semantics_name(&self) -> &'static str {
        match self.semantics {
            Semantics::Rows => "rows",
            Semantics::Semi => "semi",
            Semantics::Anti => "anti",
        }
    }

    /// This node with `rewrite` applied to its call, guard, filter, hint
    /// test, and each test as the plan holds it and as written.
    ///
    /// # Errors
    ///
    /// `rewrite` fails.
    pub(crate) fn map_expressions(&self, rewrite: impl Fn(Expr) -> Result<Expr>) -> Result<Self> {
        let mut node = self.clone();
        node.call = rewrite(node.call)?;
        node.guard = node.guard.map(&rewrite).transpose()?;
        node.filter = node.filter.map(&rewrite).transpose()?;
        node.hint_test = node.hint_test.map(&rewrite).transpose()?;
        for test in &mut node.tests {
            test.test = rewrite(test.test.clone())?;
            test.written = rewrite(test.written.clone())?;
        }
        Ok(node)
    }

    /// Whether `name` is a column the node adds.
    pub(crate) fn is_field(&self, name: &str) -> bool {
        self.fields.iter().any(|field| field.name() == name)
    }

    /// Write the node's line in a plan's text, before line breaks are
    /// escaped.
    fn write_explain(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "AiNode: ")?;
        if self.semantics != Semantics::Rows {
            write!(f, "{} ", self.semantics_name())?;
        }
        write!(f, "{} = {}", self.answer, self.call)?;
        if let Some(guard) = &self.guard {
            write!(f, ", guard: {guard}")?;
        }
        if let Some(filter) = &self.filter {
            write!(f, ", filter: {filter}")?;
        }
        if let Some(hint) = &self.hint {
            write!(f, ", hint: {hint}")?;
        }
        if let Some(test) = &self.hint_test {
            write!(f, ", hint test: {test}")?;
        }
        match &self.fetch {
            Some(Fetch::Global(n)) => write!(f, ", fetch: {n}")?,
            Some(Fetch::PerKey { key, n }) => write!(f, ", fetch: {n} per [{}]", key.join(", "))?,
            None => {}
        }
        if self.projection.is_some() {
            let names: Vec<String> = self
                .returned_fields()
                .iter()
                .map(|field| field.name().clone())
                .collect();
            write!(f, ", returns: [{}]", names.join(", "))?;
        }
        Ok(())
    }
}

impl PartialOrd for AiNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // nodes have no meaningful order; this agrees with `Eq`
        (self == other).then_some(Ordering::Equal)
    }
}

impl UserDefinedLogicalNodeCore for AiNode {
    fn name(&self) -> &str {
        "AiNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    /// The call and the guard: the expressions over the input's columns.
    ///
    /// The filter and tests also read the answer, which is not a column
    /// of the input; DataFusion's rules resolve a node's expressions
    /// against its input, so they do not rewrite these.
    fn expressions(&self) -> Vec<Expr> {
        let mut exprs = vec![self.call.clone()];
        exprs.extend(self.guard.clone());
        exprs
    }

    /// The columns a filter above the node must not read to move below it.
    ///
    /// A test of the answer waits for it. A test of the input's columns
    /// moves below a node with no fetch, `σ_q(AI(R)) = AI(σ_q(R))`. Above
    /// a fetch it stays: limiting rows before the test is not limiting
    /// them after. Only a test of a per-key fetch's key columns moves,
    /// as it keeps or drops each key's rows whole.
    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut prevent: HashSet<String> = self
            .fields
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let key: &[String] = match &self.fetch {
            None => return prevent,
            Some(Fetch::Global(_)) => &[],
            Some(Fetch::PerKey { key, .. }) => key,
        };
        // DataFusion names the columns by name alone, so a key name that
        // two input columns share stays blocked
        let names: Vec<&String> = self
            .input
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect();
        for name in &names {
            let unique_key =
                key.contains(name) && names.iter().filter(|found| *found == name).count() == 1;
            if !unique_key {
                prevent.insert((*name).clone());
            }
        }
        prevent
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let input = self.input.schema();
        let width = input.fields().len();
        // the output starts with the input's columns, or a gate's left input's
        let mut needed: Vec<usize> = output_columns
            .iter()
            .copied()
            .filter(|index| *index < width)
            .collect();
        let read = self
            .expressions()
            .into_iter()
            .chain(self.filter.clone())
            .chain(self.hint_test.clone())
            .chain(self.tests.iter().map(|test| test.test.clone()));
        for expr in read {
            for column in expr.column_refs() {
                match input.index_of_column(column) {
                    Ok(index) => needed.push(index),
                    Err(_) if column.relation.is_none() && self.is_field(&column.name) => {}
                    Err(_) => return None,
                }
            }
        }
        // a per-key fetch reads its key columns, unless the key is the answer
        if let Some(Fetch::PerKey { key, .. }) = &self.fetch {
            for name in key.iter().filter(|name| !self.is_field(name)) {
                let found: Vec<usize> = input
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| field.name() == name)
                    .map(|(index, _)| index)
                    .collect();
                let [index] = found.as_slice() else {
                    return None;
                };
                needed.push(*index);
            }
        }
        needed.sort_unstable();
        needed.dedup();
        Some(vec![needed])
    }

    /// False: a limit above the node becomes its fetch
    /// ([`crate::PushDownLimitIntoAiNodes`]) rather than a copy below it,
    /// so the executor counts the rows that pass the node.
    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // a prompt literal's line break would break the plan's indentation
        let mut text = String::new();
        self.write_explain(&mut text)?;
        write!(f, "{}", text.replace('\r', "\\r").replace('\n', "\\n"))
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        let Ok([input]) = <[LogicalPlan; 1]>::try_from(inputs) else {
            return internal_err!("AiNode reads one input");
        };
        // a rule that rewrites an expression may alias it to keep its name;
        // the node's schema does not name its expressions
        let mut exprs = exprs.into_iter().map(|expr| match expr {
            Expr::Alias(alias) => *alias.expr,
            other => other,
        });
        let (Some(call), guard, None) = (exprs.next(), exprs.next(), exprs.next()) else {
            return internal_err!("AiNode has its call and an optional guard");
        };
        if guard.is_some() != self.guard.is_some() {
            return internal_err!("AiNode's guard was added or removed");
        }
        let mut node = self.clone();
        node.input = Arc::new(input);
        node.call = call;
        node.guard = guard;
        node.checked()
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

    fn ask(function: &str, template: &str) -> Expr {
        Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == function)
                .unwrap(),
            vec![lit(template), col("r.body")],
        ))
    }

    fn ai_filter(template: &str) -> AiNode {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_filter", template),
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        node.filter = Some(col("kind"));
        node.projection = Some(Vec::new());
        node.checked().unwrap()
    }

    #[test]
    fn an_ai_filter_keeps_its_input_schema_and_reads_its_prompt_columns() {
        let node = ai_filter("Is {0} kind?");
        assert_eq!(node.schema(), reviews().schema());
        assert_eq!(node.expressions(), vec![ask("ai_filter", "Is {0} kind?")]);
        let rebuilt = node
            .with_exprs_and_inputs(node.expressions(), vec![reviews()])
            .unwrap();
        assert_eq!(rebuilt, node);
        assert_eq!(node.necessary_children_exprs(&[0]), Some(vec![vec![0, 1]]));
    }

    #[test]
    fn scores_are_predicates_only_when_their_answer_is_tested() {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_score", "Relevance of {0}?"),
            "score".into(),
            vec![Field::new("score", DataType::Float64, true)],
        )
        .unwrap();
        assert!(!node.is_predicate());
        node.filter = Some(col("score").gt(lit(0.5)));
        assert!(node.is_predicate());

        let mut boolean = ai_filter("Is {0} kind?");
        boolean.filter = None;
        assert!(boolean.is_predicate());
    }

    #[test]
    fn an_ai_column_adds_its_answer_after_the_input_columns() {
        let node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_classify", "Tone of {0}?"),
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
    fn datafusion_optimizes_a_plan_around_an_ai_filter() {
        let filter = LogicalPlan::Extension(Extension {
            node: Arc::new(ai_filter("Is {0} kind?")),
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
        let asked = lines
            .iter()
            .position(|line| line.starts_with("AiNode"))
            .unwrap();
        let test = lines
            .iter()
            .position(|line| line.contains("stars > Int32(3)"))
            .unwrap();
        assert!(test > asked, "{text}");
        assert!(
            text.contains("Projection: reviews.id, reviews.body"),
            "{text}"
        );
    }

    fn tone(fetch: Option<Fetch>, hint: Option<Expr>) -> LogicalPlan {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_classify", "Tone of {0}?"),
            "tone".into(),
            vec![Field::new("tone", DataType::Utf8, true)],
        )
        .unwrap();
        node.fetch = fetch;
        node.hint = hint;
        crate::extension::extension_node(node.checked().unwrap())
    }

    /// The plan after the default optimizer rules, one line per operator.
    fn optimized(plan: LogicalPlan) -> Vec<String> {
        Optimizer::with_rules(crate::rules::default_rules())
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
            .lines()
            .map(|line| line.trim().to_string())
            .collect()
    }

    fn position(lines: &[String], start: &str) -> usize {
        lines
            .iter()
            .position(|line| line.starts_with(start))
            .unwrap_or_else(|| panic!("no {start} in {lines:?}"))
    }

    #[test]
    fn a_filter_of_the_input_stays_above_a_node_with_a_fetch() {
        let per_key = Fetch::PerKey {
            key: vec!["id".into()],
            n: 1,
        };
        for fetch in [Fetch::Global(2), per_key] {
            let plan = LogicalPlanBuilder::from(tone(Some(fetch), None))
                .filter(col("r.stars").gt(lit(3)))
                .unwrap()
                .build()
                .unwrap();
            let lines = optimized(plan);
            // limiting rows before the test is not limiting them after
            assert!(
                position(&lines, "Filter") < position(&lines, "AiNode"),
                "{lines:?}"
            );
        }
    }

    #[test]
    fn a_filter_of_a_per_key_fetchs_key_moves_below_the_node() {
        let fetch = Fetch::PerKey {
            key: vec!["id".into()],
            n: 1,
        };
        let plan = LogicalPlanBuilder::from(tone(Some(fetch), None))
            .filter(col("r.id").gt(lit(3)))
            .unwrap()
            .build()
            .unwrap();
        let lines = optimized(plan);
        // the test keeps or drops each key's rows whole
        assert!(
            position(&lines, "AiNode") < position(&lines, "Filter"),
            "{lines:?}"
        );
    }

    #[test]
    fn a_test_of_the_answer_stays_above_a_node_with_a_fetch() {
        let plan = LogicalPlanBuilder::from(tone(Some(Fetch::Global(2)), None))
            .filter(col("tone").eq(lit("kind")))
            .unwrap()
            .build()
            .unwrap();
        let lines = optimized(plan);
        let node = position(&lines, "AiNode");
        assert!(position(&lines, "Filter") < node, "{lines:?}");
        assert!(!lines[node].contains("filter:"), "{lines:?}");
    }

    #[test]
    fn a_limit_becomes_the_fetch_of_a_node_without_a_hint() {
        let limited = |hint: Option<Expr>| {
            let plan = LogicalPlanBuilder::from(tone(None, hint))
                .limit(0, Some(2))
                .unwrap()
                .build()
                .unwrap();
            optimized(plan)
        };
        let lines = limited(None);
        assert!(
            lines[position(&lines, "AiNode")].contains("fetch: 2"),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("Limit")),
            "{lines:?}"
        );
        let lines = limited(Some(lit(0.5)));
        let node = position(&lines, "AiNode");
        assert!(position(&lines, "Limit") < node, "{lines:?}");
        assert!(
            !lines[node..].iter().any(|line| line.starts_with("Limit")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_where_filter_under_a_distinct_moves_below_its_per_key_fetch() {
        let filter = LogicalPlan::Extension(Extension {
            node: Arc::new(ai_filter("Is {0} kind?")),
        });
        let plan = LogicalPlanBuilder::from(filter)
            .filter(col("r.stars").gt(lit(3)))
            .unwrap()
            .project(vec![col("r.id")])
            .unwrap()
            .distinct()
            .unwrap()
            .build()
            .unwrap();
        let distinct =
            Optimizer::with_rules(vec![Arc::new(crate::rules::distinct::PushDownDistinct)])
                .optimize(plan, &OptimizerContext::new(), |_, _| {})
                .unwrap()
                .display_indent()
                .to_string();
        let lines: Vec<String> = distinct
            .lines()
            .map(|line| line.trim().to_string())
            .collect();
        // the WHERE runs before the DISTINCT stops each key at one row
        let node = position(&lines, "AiNode");
        assert!(lines[node].contains("fetch: 1 per [id]"), "{lines:?}");
        assert!(node < position(&lines, "Filter"), "{lines:?}");
    }
}
