//! The `Derive` optimizer rule: what the operators above an `AiNode`
//! need of it, as the node's own options.
//!
//! An `AiNode` with no options asks the model about every row of its
//! input and returns every answer. The operators above it usually need
//! less: a `Filter` keeps only some answers, a `LIMIT` only some rows.
//! The rule gives each node what it can use of that, so the executor
//! skips the rest:
//!
//! - **filters.** A term `q` of a `Filter` that reads a node's answer
//!   becomes the node's filter when the node can decide it
//!   ([`AiNode::accepts_test`]), `σ_q(AI(R; p)) ⇒ AI(R; p ∧ q)`, and
//!   leaves the `Filter`. A term the node cannot decide stays, and
//!   DataFusion runs it on the answers.
//! - **fetch.** A `LIMIT` of `k` rows after an `OFFSET` of `m` becomes
//!   the fetch `k + m` of the node whose rows it reads,
//!   `Limit_k(AI(R; n)) ⇒ AI(R; n = min(k, n))`, and stays when it has
//!   an offset.
//! - **projection.** An `AI.IF` node whose answer only its own filter
//!   reads returns no column, and a read of the answer above `σ_b` is
//!   the constant the filter keeps.
//!
//! A term or a limit reaches its node through the operators between
//! them that return the same rows: a SELECT list, a derived table's
//! alias, another `Filter`, another `AiNode` of rows without a fetch,
//! and, for a term, an inner join's side or an outer join's preserved
//! side. A node with a fetch takes no new filter and lets none pass
//! below it: limiting rows before a test is not limiting them after.
//! A limit stops at any operator that can change how many rows there
//! are, at a node that does not accept a fetch
//! ([`AiNode::accepts_fetch`]), and at a node with a per-key fetch.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, FetchType, Filter, JoinType, Limit, LogicalPlan, SkipType, lit,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use super::repeated::{through_alias, through_projection};
use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::{as_node, extension_node};

/// One kind of option [`Derive`] sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeriveOption {
    /// The rows a node drops: tests of its answer, and a document test
    /// an extraction's answer implies.
    Filters,
    /// How many rows are enough: a `LIMIT`, or one or `k` rows for each
    /// key of a `DISTINCT`, a semi-join, or a count threshold.
    Fetch,
    /// The added columns and extraction fields a node returns.
    Projection,
    /// The rows a node need not ask about: those a `CASE` does not read
    /// its answer for, or another term of an OR already passes.
    Guard,
}

impl DeriveOption {
    const ALL: [Self; 4] = [Self::Filters, Self::Fetch, Self::Projection, Self::Guard];

    /// The name of the rule that derives this option alone.
    pub const fn rule_name(self) -> &'static str {
        match self {
            Self::Filters => "derive.filters",
            Self::Fetch => "derive.fetch",
            Self::Projection => "derive.projection",
            Self::Guard => "derive.guard",
        }
    }
}

/// Give each `AiNode` what the operators above it need of it, as the
/// module explains: every option, or `only` one.
#[derive(Debug, Default)]
pub struct Derive {
    only: Option<DeriveOption>,
}

impl Derive {
    /// The rule that derives `option` and no other. A session lists such
    /// rules in place of `derive` to leave an option out.
    pub fn only(option: DeriveOption) -> Self {
        Self { only: Some(option) }
    }

    /// The rule named `name`: `derive`, or one option's own.
    pub fn named(name: &str) -> Option<Self> {
        if name == "derive" {
            return Some(Self::default());
        }
        DeriveOption::ALL
            .into_iter()
            .find(|option| option.rule_name() == name)
            .map(Self::only)
    }
}

/// A step of the rule: the option it derives and its rewrite.
type Step = (
    DeriveOption,
    fn(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
);

/// The steps in the order they run. A later step reads what an earlier
/// one set: a fetch follows the filters, and a guard follows both.
const STEPS: [Step; 10] = [
    // a subquery's filters reach its nodes where the query wrote them,
    // before DataFusion makes the subquery a join
    (DeriveOption::Filters, |plan| {
        plan.transform_up_with_subqueries(filter_down)
    }),
    (DeriveOption::Projection, drop_unread_answers),
    (DeriveOption::Fetch, super::distinct::distinct_fetches),
    (DeriveOption::Fetch, |plan| plan.transform_down(fetch_down)),
    (DeriveOption::Projection, super::projection::projections),
    (DeriveOption::Guard, super::guards::case_guards),
    (DeriveOption::Guard, super::or_guards::or_guards),
    (DeriveOption::Filters, super::substring::substring_prechecks),
    (DeriveOption::Fetch, super::keyed::semi_join_fetches),
    (DeriveOption::Fetch, super::keyed::count_thresholds),
];

impl OptimizerRule for Derive {
    fn name(&self) -> &str {
        match self.only {
            Some(option) => option.rule_name(),
            None => "derive",
        }
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut plan = plan;
        let mut changed = false;
        for (option, step) in STEPS {
            if self.only.is_none_or(|only| only == option) {
                let done = step(plan)?;
                changed |= done.transformed;
                plan = done.data;
            }
        }
        Ok(Transformed::new_transformed(plan, changed))
    }
}

/// Offer each term of a `Filter` to the `AiNode`s below it.
fn filter_down(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let mut input = filter.input.as_ref().clone();
    let mut kept = Vec::new();
    for term in split_conjunction(&filter.predicate) {
        match offer(&input, term)? {
            Some(below) => input = below,
            None => kept.push(term.clone()),
        }
    }
    if input == *filter.input {
        return Ok(Transformed::no(plan));
    }
    Ok(Transformed::yes(match conjunction(kept) {
        Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?),
        None => input,
    }))
}

/// `plan` with `term`, a test of its rows, as the filter of the `AiNode`
/// whose answer the term reads; None when no node takes it.
fn offer(plan: &LogicalPlan, term: &Expr) -> Result<Option<LogicalPlan>> {
    let below = |input: &LogicalPlan, term: &Expr| -> Result<Option<LogicalPlan>> {
        match offer(input, term)? {
            Some(input) => Ok(Some(plan.with_new_exprs(plan.expressions(), vec![input])?)),
            None => Ok(None),
        }
    };
    match plan {
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            if node.fetch.is_some() || node.semantics != Semantics::Rows {
                return Ok(None);
            }
            let columns = term.column_refs();
            if !columns
                .iter()
                .any(|column| column.relation.is_none() && node.is_field(&column.name))
            {
                return below(&node.input, term);
            }
            let returned: BTreeSet<String> = node
                .returned_fields()
                .iter()
                .map(|field| field.name().clone())
                .collect();
            let reads_returned = columns
                .iter()
                .any(|column| column.relation.is_none() && returned.contains(&column.name));
            if !(reads_returned && node.accepts_test(term)) {
                return Ok(None);
            }
            let mut taking = node.clone();
            taking.filter = conjunction(taking.filter.take().into_iter().chain([term.clone()]));
            Ok(Some(extension_node(taking.checked()?)))
        }
        LogicalPlan::Projection(projection) => match through_projection(term, projection) {
            Some(term) => below(&projection.input, &term),
            None => Ok(None),
        },
        LogicalPlan::SubqueryAlias(alias) => match through_alias(term, alias) {
            Some(term) => below(&alias.input, &term),
            None => Ok(None),
        },
        LogicalPlan::Filter(filter) => below(&filter.input, term),
        LogicalPlan::Join(join) => {
            let (keeps_left, keeps_right) = match join.join_type {
                JoinType::Inner => (true, true),
                JoinType::Left => (true, false),
                JoinType::Right => (false, true),
                _ => (false, false),
            };
            let columns = term.column_refs();
            let within = |side: &LogicalPlan| {
                columns
                    .iter()
                    .all(|column| side.schema().has_column(column))
            };
            let (left, right) = (join.left.as_ref(), join.right.as_ref());
            let inputs = if keeps_left && within(left) {
                offer(left, term)?.map(|left| vec![left, right.clone()])
            } else if keeps_right && within(right) {
                offer(right, term)?.map(|right| vec![left.clone(), right])
            } else {
                None
            };
            match inputs {
                Some(inputs) => Ok(Some(plan.with_new_exprs(plan.expressions(), inputs)?)),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Give a `LIMIT` to the `AiNode` whose rows it reads, as its fetch.
fn fetch_down(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Limit(limit) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let (Ok(SkipType::Literal(skip)), Ok(FetchType::Literal(Some(fetch)))) =
        (limit.get_skip_type(), limit.get_fetch_type())
    else {
        return Ok(Transformed::no(plan));
    };
    // DataFusion's `eliminate_limit` replaces a LIMIT 0 with no rows
    if fetch == 0 {
        return Ok(Transformed::no(plan));
    }
    let Some((input, changed)) = with_fetch(&limit.input, skip + fetch)? else {
        return Ok(Transformed::no(plan));
    };
    if skip == 0 {
        return Ok(Transformed::yes(input));
    }
    if !changed {
        return Ok(Transformed::no(plan));
    }
    Ok(Transformed::yes(LogicalPlan::Limit(Limit {
        skip: limit.skip.clone(),
        fetch: limit.fetch.clone(),
        input: Arc::new(input),
    })))
}

/// `plan` with its `AiNode` returning at most `rows` rows, and whether
/// that changed the node; None when the node is out of reach or cannot
/// take the fetch, as the module explains.
fn with_fetch(plan: &LogicalPlan, rows: usize) -> Result<Option<(LogicalPlan, bool)>> {
    match plan {
        LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => {
            let Some((input, changed)) = with_fetch(plan.inputs()[0], rows)? else {
                return Ok(None);
            };
            let rebuilt = plan.with_new_exprs(plan.expressions(), vec![input])?;
            Ok(Some((rebuilt, changed)))
        }
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            let rows = match &node.fetch {
                _ if !node.accepts_fetch() => return Ok(None),
                Some(Fetch::PerKey { .. }) => return Ok(None),
                Some(Fetch::Global(found)) if *found <= rows => {
                    return Ok(Some((plan.clone(), false)));
                }
                Some(Fetch::Global(_)) | None => rows,
            };
            let mut fetched: AiNode = node.clone();
            fetched.fetch = Some(Fetch::Global(rows));
            Ok(Some((extension_node(fetched.checked()?), true)))
        }
        _ => Ok(None),
    }
}

/// `plan` with each filtering `AI.IF` node returning no column when no
/// operator above it reads its answer; a subquery returns its own columns.
fn drop_unread_answers(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let plan = plan.transform_up(|below| below.map_subqueries(drop_unread_answers))?;
    let (read, _) = read_kept_answers(plan.data.clone())?;
    let mut dropped = drop_in(read.data)?;
    dropped.transformed |= plan.transformed || read.transformed;
    Ok(dropped)
}

/// An `AI.IF` answer and the value every row its node passes has.
type Kept = Vec<(String, bool)>;

/// `plan` with each read of an answer above the `AI.IF` node that filters
/// on it as the value the filter keeps: TRUE above `σ_b`, FALSE above
/// `σ_{NOT b}`. For example, `SELECT AI.IF(q) AS k ... WHERE AI.IF(q)`
/// returns `k` as TRUE without asking again. Also the answers that hold
/// such a value in the rows `plan` returns.
///
/// Only operators that return rows of their input unchanged carry a
/// value up: an outer join's unmatched rows hold NULL instead.
fn read_kept_answers(plan: LogicalPlan) -> Result<(Transformed<LogicalPlan>, Kept)> {
    let mut kept = Kept::new();
    let mut changed = false;
    let mut inputs = Vec::new();
    for input in plan.inputs() {
        let (read, below) = read_kept_answers(input.clone())?;
        changed |= read.transformed;
        kept.extend(below);
        inputs.push(read.data);
    }
    let own = as_node::<AiNode>(&plan).and_then(|node| {
        let filters = node.function() == "ai_filter"
            && node.semantics == Semantics::Rows
            && !node.returned_fields().is_empty();
        let negated = node.answer_term(node.filter.as_ref().filter(|_| filters)?)?;
        Some((node.answer.clone(), !negated))
    });
    let carries = match &plan {
        LogicalPlan::Join(join) => join.join_type == datafusion::logical_expr::JoinType::Inner,
        LogicalPlan::Filter(_)
        | LogicalPlan::Projection(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_) => true,
        other => as_node::<AiNode>(other).is_some(),
    };
    let preserver = NamePreserver::new(&plan);
    let mut exprs = Vec::new();
    for expr in plan.expressions() {
        let saved = preserver.save(&expr);
        let read = expr.transform_up(|found| {
            let value = match &found {
                Expr::Column(column) if column.relation.is_none() => kept
                    .iter()
                    .find(|(name, _)| *name == column.name)
                    .map(|(_, value)| *value),
                _ => None,
            };
            Ok(match value {
                Some(value) => Transformed::yes(lit(value)),
                None => Transformed::no(found),
            })
        })?;
        changed |= read.transformed;
        exprs.push(saved.restore(read.data));
    }
    let plan = if changed {
        plan.with_new_exprs(exprs, inputs)?
    } else {
        plan
    };
    if !carries {
        kept.clear();
    }
    kept.extend(own);
    Ok((Transformed::new_transformed(plan, changed), kept))
}

fn drop_in(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let unread = |node: &AiNode, read: &BTreeSet<String>| {
        node.function() == "ai_filter"
            && node.semantics == Semantics::Rows
            && node.filter.is_some()
            && !node.returned_fields().is_empty()
            && !read.contains(&node.answer)
    };
    // the plan's own columns, then the columns each operator reads, top down
    let mut read: BTreeSet<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut found = false;
    plan.apply(|below| {
        let node = as_node::<AiNode>(below);
        found = found || node.is_some_and(|node| unread(node, &read));
        // a node's own filter and tests read its answer before it is dropped
        let own = node.map(|node| node.answer.as_str());
        for expr in below.expressions() {
            for column in expr.column_refs() {
                if Some(column.name.as_str()) != own {
                    read.insert(column.name.clone());
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if !found {
        return Ok(Transformed::no(plan));
    }
    plan.transform_up(|below| {
        let dropped = match as_node::<AiNode>(&below) {
            Some(node) if unread(node, &read) => {
                let mut node = node.clone();
                node.projection = Some(Vec::new());
                extension_node(node.checked()?)
            }
            _ => below.recompute_schema()?,
        };
        Ok(Transformed::yes(dropped))
    })
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{Expr, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    fn kind_reviews(hint: Option<Expr>) -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
        ]));
        let reviews = LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .build()
        .unwrap();
        let call = Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == "ai_filter")
                .unwrap(),
            vec![lit("Is {0} kind?"), col("reviews.body")],
        ));
        let mut node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        node.filter = Some(col("kind"));
        node.projection = Some(Vec::new());
        node.hint = hint;
        extension_node(node.checked().unwrap())
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(Derive::default())])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_limit_over_an_ai_filter_becomes_its_fetch() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .project(vec![col("reviews.id")])
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(!text.contains("Limit"), "{text}");
        assert!(text.contains("fetch: 3"), "{text}");
    }

    #[test]
    fn a_limit_with_an_offset_stays_and_fetches_its_offset_rows_too() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .limit(2, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("Limit: skip=2, fetch=3"), "{text}");
        assert!(text.contains("fetch: 5"), "{text}");
    }

    #[test]
    fn a_limit_stays_above_a_filter_or_a_hint() {
        let filtered = LogicalPlanBuilder::from(kind_reviews(None))
            .filter(col("reviews.id").gt(lit(3)))
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let hinted = LogicalPlanBuilder::from(kind_reviews(Some(lit(0.5))))
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        for plan in [filtered, hinted] {
            let text = pushed(plan);
            assert!(text.contains("Limit: skip=0, fetch=3"), "{text}");
            assert!(!text.contains("fetch: 3"), "{text}");
        }
    }
}
