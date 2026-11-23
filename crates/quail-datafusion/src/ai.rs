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
//! the node becomes its fetch (`Derive`,
//! `Derive`).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DFSchemaRef, Result, TableReference, internal_err, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::utils::{conjunction, disjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Filter, LogicalPlan, LogicalPlanBuilder, Operator,
    UserDefinedLogicalNodeCore,
};

use crate::functions::{column_call_kind, is_ai_call, quail_function};
use crate::plan::{as_node, extension_node};

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

/// The parts of the OR an `AI.IF` node filters on
/// ([`AiNode::or_parts`]).
pub(crate) struct OrParts {
    /// The terms that ask no model, as one OR; None when it has none.
    pub(crate) plain: Option<Expr>,
    /// The answer of each other `AI.IF` of the OR, and whether the OR
    /// reads it as `NOT`.
    pub(crate) others: Vec<(String, bool)>,
    /// Whether the OR reads the node's own answer as `NOT`.
    pub(crate) negated: bool,
}

/// One AI call over the rows of its input, as `AiNode`; the module
/// explains each part.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AiNode {
    pub(crate) input: Arc<LogicalPlan>,
    /// The query's own call, such as `ai_classify(...)`, without its `HINT`.
    pub(crate) call: Expr,
    /// The answer column's name.
    pub(crate) answer: String,
    /// The columns the call adds, the answer first.
    pub(crate) fields: Vec<Field>,
    pub(crate) guard: Option<Expr>,
    pub(crate) filter: Option<Expr>,
    /// The options of the `HINT` the query wraps the call's term in,
    /// such as `{'selectivity': 0.4}`.
    pub(crate) hint: Option<Expr>,
    /// The positions in `fields` of the added columns the node returns;
    /// None returns every one.
    pub(crate) projection: Option<Vec<usize>>,
    pub(crate) fetch: Option<Fetch>,
    pub(crate) semantics: Semantics,
    /// Where the query wrote the call's term, if in a WHERE or ON.
    pub(crate) written: Option<Written>,
    /// The tests of the node's answers the query wrote in a WHERE.
    pub(crate) tests: Vec<WrittenTest>,
    pub(crate) schema: DFSchemaRef,
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
            projection: None,
            fetch: None,
            semantics: Semantics::Rows,
            written: None,
            tests: Vec::new(),
            schema: Arc::new(DFSchema::empty()),
        }
        .checked()
    }

    /// A node over `input` asking `call`, an `AI.IF` or `AI.SCORE`, and
    /// returning its answer as the column `answer`; the query wrote the
    /// call's term at `written`.
    ///
    /// # Errors
    ///
    /// The call's type does not read, or `answer` is already a column of
    /// the input.
    pub(crate) fn answering(
        input: LogicalPlan,
        call: Expr,
        answer: String,
        written: Written,
    ) -> Result<Self> {
        let (_, field) = call.to_field(input.schema())?;
        let field = field.as_ref().clone().with_name(answer.clone());
        Self {
            written: Some(written),
            ..Self::try_new(Arc::new(input), call, answer, vec![field])?
        }
        .checked()
    }

    /// This node with the options of the `HINT` the query wraps its
    /// term in.
    pub(crate) fn hinted(self, hint: Option<Expr>) -> Self {
        Self { hint, ..self }
    }

    /// This node keeping the rows `filter`, a test of its answer, passes,
    /// as `semantics` says, and returning none of the columns it adds.
    ///
    /// # Errors
    ///
    /// As [`AiNode::checked`].
    pub(crate) fn tested(self, filter: Expr, semantics: Semantics) -> Result<Self> {
        Self {
            filter: Some(filter),
            projection: Some(Vec::new()),
            semantics,
            ..self
        }
        .checked()
    }

    /// This node keeping `tests`, the tests of its answers the query
    /// wrote in a WHERE, beside its own.
    pub(crate) fn with_tests(mut self, tests: &[WrittenTest]) -> Self {
        self.tests.extend(tests.iter().cloned());
        self
    }

    /// Whether the node asks `call`, as `other` asks its own, with no
    /// options, adding columns of the types `other` adds: its answers
    /// are `other`'s.
    pub(crate) fn repeats(&self, call: &Expr, other: &AiNode) -> bool {
        let same_types = self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(&other.fields)
                .all(|(found, field)| found.data_type() == field.data_type());
        self.call == *call && same_types && !self.has_options()
    }

    /// Whether the node adds `call`'s answer as a column every row it
    /// returns has: a column node, or an `AI.IF` or `AI.SCORE` that
    /// filters no row.
    pub(crate) fn answers_every_row(&self, call: &Expr) -> bool {
        (!self.is_predicate() || self.filter.is_none()) && self.call == *call
    }

    /// This extraction with its answer and span columns typed by its
    /// call, once a rewrite of the call has changed the fields it asks.
    ///
    /// # Errors
    ///
    /// The call's type does not read.
    pub(crate) fn retyped_extraction(mut self) -> Result<Self> {
        let Expr::ScalarFunction(call) = &self.call else {
            return Ok(self);
        };
        let spans = Expr::ScalarFunction(ScalarFunction::new_udf(
            quail_function("ai_extract_span"),
            call.args.clone(),
        ));
        let schema = self.input.schema();
        for (index, typed) in [&self.call, &spans].into_iter().enumerate() {
            if let Some(field) = self.fields.get(index) {
                let (_, found) = typed.to_field(schema)?;
                self.fields[index] = found.as_ref().clone().with_name(field.name());
            }
        }
        Ok(self)
    }

    /// This node with its schema computed from its parts.
    ///
    /// # Errors
    ///
    /// A field's name is already a column of the input, a projection
    /// position is not a field's, a gate's input is not a join, or the
    /// node reads a column its input does not have.
    pub fn checked(mut self) -> Result<Self> {
        if let Some(missing) = self
            .reads()
            .into_iter()
            .find(|column| !self.input.schema().has_column(column))
        {
            return plan_err!("{} reads {missing}, which its input lacks", self.answer);
        }
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

    /// Every column the node reads from its input: the columns of its
    /// call, guard, and filter that it does not add itself. A rule may
    /// put the node only over an input that has them all.
    pub fn reads(&self) -> Vec<datafusion::common::Column> {
        let mut reads = Vec::new();
        for expr in [Some(&self.call), self.guard.as_ref(), self.filter.as_ref()]
            .into_iter()
            .flatten()
        {
            for column in expr.column_refs() {
                let own = column.relation.is_none() && self.is_field(&column.name);
                if !own && !reads.contains(column) {
                    reads.push(column.clone());
                }
            }
        }
        reads
    }

    /// Whether the node can take `term`, a test that reads its answer,
    /// into its filter. A test it does not accept stays a `Filter` above
    /// the node, which DataFusion runs on the answers.
    ///
    /// An `AI.IF` takes one test: its answer `b`; for a call that reads
    /// one table, also `NOT b` or an OR of `b` with other `AI.IF`
    /// answers and tests of that table. Any other call takes a test of
    /// its own answers and its input's columns, not one that reads
    /// another call's answer.
    pub fn accepts_test(&self, term: &Expr) -> bool {
        // DataFusion decorrelates a subquery it finds in a `Filter`, not
        // one a node holds
        let subquery = |expr: &Expr| {
            Ok(matches!(
                expr,
                Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
            ))
        };
        if term.exists(subquery).unwrap_or(true) {
            return false;
        }
        let one_table = self
            .call
            .column_refs()
            .iter()
            .filter_map(|column| column.relation.as_ref())
            .collect::<HashSet<_>>()
            .len()
            == 1;
        if self.function() != "ai_filter" {
            let own = |column: &&datafusion::common::Column| {
                column.relation.is_some() || self.is_field(&column.name)
            };
            return term.column_refs().iter().all(own);
        }
        if self.filter.is_some() {
            return false;
        }
        match self.answer_term(term) {
            Some(negated) => one_table || !negated,
            None => {
                let mut tested = self.clone();
                tested.filter = Some(term.clone());
                one_table
                    && tested.or_parts().is_some_and(|parts| {
                        parts
                            .plain
                            .is_none_or(|plain| self.reads_call_table(&plain))
                    })
            }
        }
    }

    /// The terms of an `AI.IF` node's filter OR other than its answer
    /// `b`, when the filter is `a ∨ b`, and whether the OR reads the
    /// answer as `NOT b`.
    ///
    /// The lowering writes `WHERE r.flagged OR AI.IF(...)` as such a
    /// node: its filter is the OR, with the answer in the call's place.
    /// No other term of the filter may read `b`, and the OR reads `b`
    /// one way: `b OR NOT b` is no test of it.
    pub(crate) fn or_terms(&self) -> Option<(Expr, bool)> {
        let filter = self.filter.as_ref()?;
        let reads_answer = |term: &Expr| {
            term.column_refs()
                .iter()
                .any(|column| column.relation.is_none() && self.is_field(&column.name))
        };
        let conjuncts = split_conjunction(filter);
        let mut found = None;
        for conjunct in &conjuncts {
            let terms = split_binary(conjunct, Operator::Or);
            let (answers, others): (Vec<&Expr>, Vec<&Expr>) = terms
                .into_iter()
                .partition(|term| self.answer_term(term).is_some());
            if answers.is_empty() || others.is_empty() {
                if reads_answer(conjunct) {
                    return None;
                }
                continue;
            }
            if found.is_some() || others.iter().any(|term| reads_answer(term)) {
                return None;
            }
            let [answer] = answers.as_slice() else {
                return None;
            };
            let negated = self.answer_term(answer) == Some(true);
            found = disjunction(others.into_iter().cloned()).map(|rest| (rest, negated));
        }
        found
    }

    /// The parts of the OR the node filters on, when its filter is one
    /// ([`Self::or_terms`]): its terms that ask no model, and the answers
    /// of the other `AI.IF`s of the OR, which nodes below it return.
    ///
    /// An OR of several `AI.IF`s, `a ∨ f_1 ∨ … ∨ f_k`, lowers to one
    /// node per call: each `f_j` but the last returns its answer `b_j`
    /// and drops no row, in written order, below the last, whose filter
    /// is `a ∨ b_1 ∨ … ∨ b_k`.
    pub(crate) fn or_parts(&self) -> Option<OrParts> {
        let (rest, negated) = self.or_terms()?;
        let answers = Self::answers_below(&self.input);
        let mut plain = Vec::new();
        let mut others = Vec::new();
        for term in split_binary(&rest, Operator::Or) {
            let (read, reversed) = match term {
                Expr::Not(inner) => (inner.as_ref(), true),
                other => (other, false),
            };
            match read.try_as_col() {
                Some(column) if column.relation.is_none() && answers.contains(&column.name) => {
                    if others.iter().any(|(name, _)| *name == column.name) {
                        return None;
                    }
                    others.push((column.name.clone(), reversed));
                }
                _ => plain.push(term.clone()),
            }
        }
        Some(OrParts {
            plain: disjunction(plain),
            others,
            negated,
        })
    }

    /// The answers of the `AI.IF` nodes in `plan` that return their
    /// answer and drop no row: the other `AI.IF`s of an OR.
    fn answers_below(plan: &LogicalPlan) -> Vec<String> {
        let mut found = Vec::new();
        let _ = plan.apply(|below| {
            if let Some(asked) = as_node::<AiNode>(below)
                && asked.semantics == Semantics::Rows
                && asked.function() == "ai_filter"
                && asked.filter.is_none()
            {
                found.push(asked.answer.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        found
    }

    /// Whether `test` reads columns of the node's input, of the one table
    /// its call reads, and calls no AI function: a test the executor can
    /// evaluate on the rows the node reads, as a guard.
    pub(crate) fn reads_call_table(&self, test: &Expr) -> bool {
        let tables = |expr: &Expr| -> Vec<String> {
            let mut found: Vec<String> = Vec::new();
            for column in expr.column_refs() {
                let table = column
                    .relation
                    .as_ref()
                    .map(|table| table.table().to_string());
                if let Some(table) = table
                    && !found.contains(&table)
                {
                    found.push(table);
                }
            }
            found
        };
        let call = tables(&self.call);
        let read = tables(test);
        let input = self.input.schema();
        call.len() == 1
            && read == call
            // an unqualified column is another node's answer, not the table's
            && test
                .column_refs()
                .iter()
                .all(|column| column.relation.is_some() && input.has_column(column))
            && !test.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(true)
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

    /// Whether a rule gave the node a filter, guard, fetch, or projection.
    pub fn has_options(&self) -> bool {
        self.filter.is_some()
            || self.guard.is_some()
            || self.fetch.is_some()
            || self.projection.is_some()
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

    /// The answer column's name.
    pub fn answer(&self) -> &str {
        &self.answer
    }

    /// Whether the node asks about the pairs of its input join and
    /// returns the join's left rows: a semi or anti join, not a node of
    /// rows.
    pub fn is_join(&self) -> bool {
        self.semantics != Semantics::Rows
    }

    /// Whether the node adds its call's answers as columns to each row of
    /// its input: neither an `AI.IF` or tested score, nor a join.
    pub fn is_column(&self) -> bool {
        !self.is_predicate() && !self.is_join()
    }

    /// Whether the node returns the left rows with no pair that passes.
    pub fn is_anti(&self) -> bool {
        self.semantics == Semantics::Anti
    }

    /// The query's own call, such as `ai_classify(...)`, without its
    /// `HINT`.
    pub fn call(&self) -> Expr {
        self.call.clone()
    }

    /// The model call kind of a column node: `label`, `extract`, `map`,
    /// or `features`; None for an `AI.IF` or `AI.SCORE`.
    pub(crate) fn column_kind(&self) -> Option<&'static str> {
        column_call_kind(&self.call)
    }

    /// Which rows the model is asked about, as the binder reads it.
    pub(crate) fn guard(&self) -> Option<Expr> {
        self.guard.clone()
    }

    /// The exact test of the input's columns and the answer, as the
    /// binder reads it.
    pub(crate) fn filter(&self) -> Option<Expr> {
        self.filter.clone()
    }

    /// Where the query wrote the call's term, if in a WHERE or ON.
    pub(crate) fn written(&self) -> Option<Written> {
        self.written.clone()
    }

    /// The tests of the node's answers the query wrote in a WHERE.
    pub(crate) fn tests(&self) -> Vec<WrittenTest> {
        self.tests.clone()
    }

    /// The number of rows a global fetch keeps.
    pub(crate) fn global_fetch(&self) -> Option<usize> {
        match self.fetch {
            Some(Fetch::Global(rows)) => Some(rows),
            _ => None,
        }
    }

    /// The key columns and the number of rows of each key that a per-key
    /// fetch keeps.
    pub(crate) fn key_fetch(&self) -> Option<(Vec<String>, usize)> {
        match &self.fetch {
            Some(Fetch::PerKey { key, n }) => Some((key.clone(), *n)),
            _ => None,
        }
    }

    /// The tables whose columns the call reads.
    pub(crate) fn call_tables(&self) -> BTreeSet<TableReference> {
        self.call
            .column_refs()
            .into_iter()
            .filter_map(|column| column.relation.clone())
            .collect()
    }

    /// The columns of the rows the node reads.
    pub(crate) fn input_schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    /// The names of the columns the node adds, the answer first.
    pub(crate) fn added_names(&self) -> Vec<String> {
        self.fields
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    /// The run of `AiNode`s at `plan`'s root, top first, and the plan
    /// below the run.
    pub(crate) fn run(plan: &LogicalPlan) -> (Vec<&AiNode>, &LogicalPlan) {
        let mut nodes = Vec::new();
        let mut bottom = plan;
        while let Some(node) = as_node::<AiNode>(bottom) {
            nodes.push(node);
            bottom = node.input.as_ref();
        }
        (nodes, bottom)
    }

    /// This node without the tests of its answers the query wrote; None
    /// when it keeps none.
    pub(crate) fn without_tests(&self) -> Option<Self> {
        if self.tests.is_empty() {
            return None;
        }
        Some(Self {
            tests: Vec::new(),
            ..self.clone()
        })
    }

    /// Whether the node has no option at all: no filter, guard, hint,
    /// fetch, projection, or test.
    pub(crate) fn is_bare(&self) -> bool {
        !self.has_options() && self.hint.is_none() && self.tests.is_empty()
    }

    /// The node's filter and each test of its answers as the plan holds
    /// it: what the node reads beside its expressions, the answer
    /// included.
    pub(crate) fn filter_expressions(&self) -> Vec<Expr> {
        self.filter
            .iter()
            .cloned()
            .chain(self.tests.iter().map(|test| test.test.clone()))
            .collect()
    }

    /// Whether the node drops no row and returns each row of its input
    /// with the columns it adds: no filter, written test, or fetch.
    pub(crate) fn only_adds_columns(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.filter.is_none()
            && self.tests.is_empty()
            && self.fetch.is_none()
    }

    /// This node keeping only the rows `fetch` keeps after its filter.
    pub(crate) fn with_fetch(self, fetch: Fetch) -> Self {
        Self {
            fetch: Some(fetch),
            ..self
        }
    }

    /// Return only the added columns the plan above reads, by `read`, the
    /// answer always. An `AI.IF` that keeps only the rows whose answer is
    /// `b`, or only those whose answer is `NOT b`, returns no column:
    /// where every read reached the node through operators that return
    /// their input's rows as they are (`carried`), a read of the answer
    /// above reads the value every row it returns has, which this
    /// returns with the answer's name.
    pub(crate) fn project(
        &mut self,
        read: impl Fn(&str) -> bool,
        carried: bool,
    ) -> Option<(String, bool)> {
        if self.semantics != Semantics::Rows {
            return None;
        }
        if !self.is_predicate() {
            let returned =
                (self.projection.clone()).unwrap_or_else(|| (0..self.fields.len()).collect());
            let kept: Vec<usize> = (returned.iter().copied())
                .filter(|index| *index == 0 || read(self.fields[*index].name()))
                .collect();
            if kept.len() < returned.len() {
                self.projection = Some(kept);
            }
            return None;
        }
        let returns_answer =
            (self.returned_fields().iter()).any(|field| *field.name() == self.answer);
        // a score only its own test reads returns nothing
        if self.function() == "ai_score" && returns_answer && !read(&self.answer) {
            self.projection = Some(Vec::new());
        }
        if self.function() != "ai_filter" || !returns_answer {
            return None;
        }
        let filter = self.filter.as_ref()?;
        if !read(&self.answer) {
            self.projection = Some(Vec::new());
            return None;
        }
        let negated = self.answer_term(filter)?;
        if !carried {
            return None;
        }
        self.projection = Some(Vec::new());
        Some((self.answer.clone(), !negated))
    }

    /// Whether the node keeps only its first rows, in all or per key: a
    /// rule that moves an operator past it would limit other rows.
    pub(crate) fn has_fetch(&self) -> bool {
        self.fetch.is_some()
    }

    /// Whether the node asks the model about every row of its input and
    /// returns each: no filter, guard, or fetch.
    pub(crate) fn asks_every_row(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.filter.is_none()
            && self.fetch.is_none()
            && self.guard.is_none()
    }

    /// Whether the node returns other rows than the ones of its input
    /// that pass its filter: a fetch cuts them off, and a gate returns
    /// its join's left rows.
    pub(crate) fn limits_rows(&self) -> bool {
        self.fetch.is_some() || self.semantics != Semantics::Rows
    }

    /// Take out of a column node's filter the terms `split` moves, the
    /// first part of what it returns, and the written tests that hold
    /// them; return the moved terms. A node with a fetch keeps its
    /// filter whole: it counts the rows that pass all of it.
    pub(crate) fn lift_filter(
        &mut self,
        split: impl FnOnce(&Expr) -> (Vec<Expr>, Vec<Expr>),
    ) -> Vec<Expr> {
        if self.semantics != Semantics::Rows || self.is_predicate() {
            return Vec::new();
        }
        let Some(filter) = self.filter.take() else {
            return Vec::new();
        };
        let (moved, kept) = split(&filter);
        if !moved.is_empty() && self.fetch.is_some() {
            self.filter = Some(filter);
            return Vec::new();
        }
        self.tests.retain(|test| {
            !split_conjunction(&test.test)
                .iter()
                .any(|term| moved.contains(term))
        });
        self.filter = conjunction(kept);
        moved
    }

    /// This node without its options, under a `Filter` of its filter and
    /// a `Limit` of its global fetch: the same rows, as its guard, hint,
    /// and tests only save work.
    ///
    /// # Errors
    ///
    /// The `Filter` or `Limit` does not fit the node's columns.
    pub(crate) fn options_above(&self) -> Result<LogicalPlan> {
        let bare = Self {
            filter: None,
            guard: None,
            hint: None,
            fetch: None,
            projection: None,
            tests: Vec::new(),
            ..self.clone()
        };
        let mut above = extension_node(bare.checked()?);
        if let Some(filter) = &self.filter {
            above = LogicalPlan::Filter(Filter::try_new(filter.clone(), Arc::new(above))?);
        }
        if let Some(Fetch::Global(rows)) = &self.fetch {
            above = LogicalPlanBuilder::from(above)
                .limit(0, Some(*rows))?
                .build()?;
        }
        Ok(above)
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
            .chain(self.filter_expressions());
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
    /// ([`crate::Derive`]) rather than a copy below it,
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
        let mut exprs = exprs.into_iter().map(Expr::unalias);
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
        crate::plan::extension_node(node.checked().unwrap())
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
            Optimizer::with_rules(vec![Arc::new(crate::rules::derive::Derive::default())])
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
