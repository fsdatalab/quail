//! The demand pass of [`super::Derive`]: what the plan above each operator
//! needs of its rows, carried down the plan once, top to bottom.
//!
//! A demand on a plan's rows has three parts:
//!
//! - **tests**: terms the operators above apply to the rows. A row where a
//!   test is not TRUE leaves the query, so it is not needed.
//! - **reads**: for each column read above, the rows where its value is
//!   read. A column read nowhere is absent.
//! - **count**: how many of the rows that pass the tests are enough: all,
//!   the first `n`, or the first `n` of each key.
//!
//! Each operator has one transfer function: given the demand on its
//! rows, the demand on its input's rows. A `Filter` adds its terms to the
//! tests; a `LIMIT` sets the count and clears the tests, which it does not
//! pass; a SELECT list renames each part through its expressions, and
//! reads an input column only where its expression needs it (a `CASE`
//! reads a branch only where its condition holds); a join passes each
//! test and read to the side whose columns it reads, where that side's
//! rows reach the join's output. An operator with no transfer function
//! needs every row and every column of its inputs.
//!
//! An `AiNode` the demand reaches takes what it can of it:
//!
//! - a test of its answer that it can decide becomes its filter;
//! - the count becomes its fetch, when every test either became its filter
//!   or reads only its input, and then moves below it;
//! - where its answer is read becomes its guard, as far as the executor can
//!   evaluate the condition on the rows the node reads.
//!
//! A test, a count, or a guard that a node takes leaves the operator that
//! wrote it: a `Filter` drops the term, and a `LIMIT` with no offset
//! leaves the plan.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::utils::{conjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{
    BinaryExpr, Distinct, Expr, FetchType, Filter, JoinType, LogicalPlan, Operator, SkipType, lit,
};

use super::fetch::{at_least, distinct_key, input_key, is_count_star, stops_per_key};
use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::{as_node, extension_node, with_inputs};
use crate::rules::distinct::{is_distinct, is_unique};
use crate::rules::guards::reads_call_table;
use crate::rules::or_guards::or_terms;
use crate::rules::repeated::{through_alias, through_projection};
use crate::rules::substring::precheck;

/// Which options the pass derives.
#[derive(Debug, Clone, Copy)]
pub(super) struct Options {
    pub(super) filters: bool,
    pub(super) fetch: bool,
    pub(super) guard: bool,
}

/// How many of the rows that pass a demand's tests are enough.
#[derive(Debug, Clone, PartialEq)]
enum Count {
    All,
    /// The first `n` rows.
    Rows(usize),
    /// One row of each value of the keys, for a `DISTINCT`.
    Distinct(Vec<Column>),
    /// `n` rows of each value of the keys, where the executor stops a key.
    PerKey(Vec<Column>, usize),
}

/// One test of a demand; `id` names it to the operator that wrote it.
#[derive(Debug, Clone)]
struct Test {
    id: usize,
    expr: Expr,
}

/// What the operators above a plan need of its rows, as the module
/// explains. `reads` is None when every column is read in every row; it
/// leaves out what the tests read.
#[derive(Debug, Clone)]
struct Demand {
    tests: Vec<Test>,
    reads: Option<BTreeMap<Column, Expr>>,
    count: Count,
}

impl Demand {
    fn everything() -> Self {
        Demand {
            tests: Vec::new(),
            reads: None,
            count: Count::All,
        }
    }

    /// Where `column` is read, by the tests or otherwise: None when
    /// nowhere.
    fn read_where(&self, column: &Column) -> Option<Expr> {
        let Some(reads) = &self.reads else {
            return Some(lit(true));
        };
        (reads.get(column).cloned().into_iter())
            .chain((self.tests.iter()).filter_map(|test| read_condition(&test.expr, column)))
            .reduce(|one, two| or(&one, &two))
    }

    /// The reads with what the tests read, for an operator that does not
    /// pass the tests on.
    fn reads_with_tests(&self) -> Option<BTreeMap<Column, Expr>> {
        let mut reads = self.reads.clone()?;
        for test in &self.tests {
            for column in test.expr.column_refs() {
                if let Some(within) = read_condition(&test.expr, column) {
                    add_read(&mut reads, column, within);
                }
            }
        }
        Some(reads)
    }
}

/// What a plan's operators took of the demand on its rows.
#[derive(Debug, Default)]
struct Taken {
    /// The tests a node below took: as its filter, or below itself.
    tests: BTreeSet<usize>,
    /// Whether a node below took the count as its fetch.
    count: bool,
}

/// The state of one pass: the next test's id.
struct Pass {
    options: Options,
    next_test: usize,
}

/// Carry the demand of the whole query down `plan`, as the module explains.
pub(super) fn demand_pass(plan: LogicalPlan, options: Options) -> Result<Transformed<LogicalPlan>> {
    let before = plan.clone();
    let mut pass = Pass {
        options,
        next_test: 0,
    };
    let (after, _) = pass.down(plan, Demand::everything())?;
    let changed = after != before;
    Ok(Transformed::new_transformed(after, changed))
}

impl Pass {
    fn test(&mut self, expr: Expr) -> Test {
        self.next_test += 1;
        Test {
            id: self.next_test,
            expr,
        }
    }

    /// `plan` with every node below it given what `demand` lets it take.
    fn down(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        // a subquery an expression holds has its own demand
        let plan = plan
            .map_subqueries(|subquery| {
                let (found, _) = self.down(subquery, Demand::everything())?;
                Ok(Transformed::yes(found))
            })?
            .data;
        match plan {
            LogicalPlan::Projection(_) => self.projection(plan, demand),
            LogicalPlan::SubqueryAlias(_) => self.alias(plan, demand),
            LogicalPlan::Filter(_) => self.filter(plan, demand),
            LogicalPlan::Limit(_) => self.limit(plan, demand),
            LogicalPlan::Sort(_) => self.sort(plan, demand),
            LogicalPlan::Distinct(Distinct::All(_)) => self.distinct(plan, demand),
            LogicalPlan::Aggregate(_) => self.aggregate(plan, demand),
            LogicalPlan::Join(_) => self.join(plan, demand),
            LogicalPlan::Extension(_) if as_node::<AiNode>(&plan).is_some() => {
                self.node(plan, demand)
            }
            _ => self.barrier(plan),
        }
    }

    /// An operator with no transfer function: each input is needed whole.
    fn barrier(&mut self, plan: LogicalPlan) -> Result<(LogicalPlan, Taken)> {
        let inputs: Vec<LogicalPlan> = plan.inputs().into_iter().cloned().collect();
        if inputs.is_empty() {
            return Ok((plan, Taken::default()));
        }
        let mut found = Vec::new();
        for input in inputs {
            found.push(self.down(input, Demand::everything())?.0);
        }
        Ok((with_inputs(&plan, found)?, Taken::default()))
    }

    fn projection(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Projection(projection) = &plan else {
            unreachable!()
        };
        let below = |expr: &Expr| through_projection(expr, projection);
        let tests = (demand.tests.iter())
            .filter_map(|test| below(&test.expr).map(|expr| Test { id: test.id, expr }))
            .collect();
        // each input column, where the SELECT list's expressions read it
        let mut reads = BTreeMap::new();
        for (index, expr) in projection.expr.iter().enumerate() {
            let column = Column::from(projection.schema.qualified_field(index));
            let Some(read) = demand.read_where(&column) else {
                continue;
            };
            let read = weaken_through(&read, &below);
            let expr = expr.clone().unalias();
            for input in expr.column_refs() {
                if let Some(within) = read_condition(&expr, input) {
                    add_read(&mut reads, input, and(&read, &within));
                }
            }
        }
        let count = match &demand.count {
            Count::Distinct(keys) => keys_below(keys, &below).map_or(Count::All, Count::Distinct),
            Count::PerKey(keys, n) => {
                keys_below(keys, &below).map_or(Count::All, |keys| Count::PerKey(keys, *n))
            }
            other => other.clone(),
        };
        let input = projection.input.as_ref().clone();
        let (input, taken) = self.down(
            input,
            Demand {
                tests,
                reads: Some(reads),
                count,
            },
        )?;
        Ok((with_inputs(&plan, vec![input])?, taken))
    }

    fn alias(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::SubqueryAlias(alias) = &plan else {
            unreachable!()
        };
        let below = |expr: &Expr| through_alias(expr, alias);
        let column_below = |column: &Column| -> Option<Column> {
            let index = alias.schema.index_of_column(column).ok()?;
            Some(Column::from(alias.input.schema().qualified_field(index)))
        };
        let tests = (demand.tests.iter())
            .filter_map(|test| below(&test.expr).map(|expr| Test { id: test.id, expr }))
            .collect();
        let reads = demand.reads.as_ref().map(|reads| {
            (reads.iter())
                .filter_map(|(column, read)| {
                    Some((column_below(column)?, weaken_through(read, &below)))
                })
                .collect()
        });
        let keys = |keys: &[Column]| keys.iter().map(column_below).collect::<Option<Vec<_>>>();
        let count = match &demand.count {
            Count::Distinct(found) => keys(found).map_or(Count::All, Count::Distinct),
            Count::PerKey(found, n) => {
                keys(found).map_or(Count::All, |keys| Count::PerKey(keys, *n))
            }
            other => other.clone(),
        };
        let input = alias.input.as_ref().clone();
        let (input, taken) = self.down(
            input,
            Demand {
                tests,
                reads,
                count,
            },
        )?;
        Ok((with_inputs(&plan, vec![input])?, taken))
    }

    fn filter(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Filter(filter) = plan else {
            unreachable!()
        };
        let own: Vec<Test> = (split_conjunction(&filter.predicate).into_iter())
            .map(|term| self.test(term.clone()))
            .collect();
        let reads = demand.reads.clone();
        let mut tests = demand.tests.clone();
        tests.extend(own.iter().cloned());
        let input = Arc::unwrap_or_clone(filter.input);
        let (input, mut taken) = self.down(
            input,
            Demand {
                tests,
                reads,
                count: demand.count,
            },
        )?;
        let kept: Vec<Expr> = (own.iter())
            .filter(|test| !taken.tests.contains(&test.id))
            .map(|test| test.expr.clone())
            .collect();
        taken
            .tests
            .retain(|id| !own.iter().any(|test| test.id == *id));
        let plan = match conjunction(kept) {
            Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?),
            None => input,
        };
        Ok((plan, taken))
    }

    fn limit(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Limit(limit) = &plan else {
            unreachable!()
        };
        let literal = match (limit.get_skip_type(), limit.get_fetch_type()) {
            (Ok(SkipType::Literal(skip)), Ok(FetchType::Literal(Some(fetch)))) if fetch > 0 => {
                Some((skip, fetch))
            }
            _ => None,
        };
        let count = match (literal, &demand.count) {
            (Some((skip, fetch)), Count::Rows(above)) => Count::Rows(skip + fetch.min(*above)),
            (Some((skip, fetch)), _) => Count::Rows(skip + fetch),
            (None, _) => Count::All,
        };
        let count = if self.options.fetch {
            count
        } else {
            Count::All
        };
        let input = limit.input.as_ref().clone();
        let (input, taken) = self.down(
            input,
            Demand {
                tests: Vec::new(),
                reads: demand.reads_with_tests(),
                count,
            },
        )?;
        // a LIMIT with no offset that a node took leaves the plan
        if taken.count && literal.is_some_and(|(skip, _)| skip == 0) {
            return Ok((input, Taken::default()));
        }
        Ok((with_inputs(&plan, vec![input])?, Taken::default()))
    }

    fn sort(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Sort(sort) = &plan else {
            unreachable!()
        };
        if sort.fetch.is_some() {
            return self.barrier(plan);
        }
        let mut reads = demand.reads.clone();
        if let Some(reads) = &mut reads {
            for expr in &sort.expr {
                for column in expr.expr.column_refs() {
                    add_read(reads, column, lit(true));
                }
            }
        }
        // the sort decides which rows are first
        let input = sort.input.as_ref().clone();
        let (input, mut taken) = self.down(
            input,
            Demand {
                tests: demand.tests,
                reads,
                count: Count::All,
            },
        )?;
        // a count taken below is the sort's, not the demand's
        taken.count = false;
        Ok((with_inputs(&plan, vec![input])?, taken))
    }

    fn distinct(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let input = plan.inputs()[0].clone();
        let keys = input.schema().columns();
        // a DISTINCT that removes no row leaves, so a count reaches below
        if self.options.fetch && is_unique(&input, &keys.iter().cloned().collect()) {
            return self.down(input, demand);
        }
        let count = if self.options.fetch {
            Count::Distinct(keys)
        } else {
            Count::All
        };
        let (input, mut taken) = self.down(
            input,
            Demand {
                tests: demand.tests,
                reads: None,
                count,
            },
        )?;
        // a count taken below is the DISTINCT's, not the demand's
        taken.count = false;
        Ok((with_inputs(&plan, vec![input])?, taken))
    }

    fn aggregate(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Aggregate(aggregate) = &plan else {
            unreachable!()
        };
        if is_distinct(aggregate) {
            return self.distinct(plan, demand);
        }
        let groups: Option<Vec<Column>> = (aggregate.group_expr.iter())
            .map(|expr| match expr {
                Expr::Column(column) => Some(column.clone()),
                _ => None,
            })
            .collect();
        // HAVING COUNT(*) >= k needs k rows of each group, when nothing
        // else reads the count
        let count = match (aggregate.aggr_expr.as_slice(), &groups) {
            ([counted], Some(keys)) if is_count_star(counted) && !keys.is_empty() => {
                let column =
                    Column::from(aggregate.schema.qualified_field(aggregate.group_expr.len()));
                // the tests may read the count only to compare it with k
                let tests_read: Vec<&Test> = (demand.tests.iter())
                    .filter(|test| test.expr.column_refs().contains(&column))
                    .collect();
                let unread = demand
                    .reads
                    .as_ref()
                    .is_some_and(|reads| !reads.contains_key(&column))
                    && (tests_read.iter()).all(|test| at_least(&test.expr, &column).is_some());
                let threshold = (tests_read.iter())
                    .filter_map(|test| at_least(&test.expr, &column))
                    .max();
                match (unread, threshold) {
                    (true, Some(n)) if self.options.fetch => Count::PerKey(keys.clone(), n),
                    _ => Count::All,
                }
            }
            _ => Count::All,
        };
        let input = aggregate.input.as_ref().clone();
        let (input, _) = self.down(
            input,
            Demand {
                tests: Vec::new(),
                reads: None,
                count,
            },
        )?;
        Ok((with_inputs(&plan, vec![input])?, Taken::default()))
    }

    fn join(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let LogicalPlan::Join(join) = &plan else {
            unreachable!()
        };
        // which side's rows reach the output as they are
        let (keeps_left, keeps_right) = match join.join_type {
            JoinType::Inner => (true, true),
            JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => (true, false),
            JoinType::Right | JoinType::RightSemi | JoinType::RightAnti => (false, true),
            _ => (false, false),
        };
        let mut sides = Vec::new();
        for (input, keeps, filtered) in [
            (
                join.left.as_ref(),
                keeps_left,
                join.join_type == JoinType::RightSemi,
            ),
            (
                join.right.as_ref(),
                keeps_right,
                join.join_type == JoinType::LeftSemi,
            ),
        ] {
            let schema = input.schema();
            let within = |expr: &Expr| {
                expr.column_refs()
                    .iter()
                    .all(|column| schema.has_column(column))
            };
            let tests = if keeps {
                demand
                    .tests
                    .iter()
                    .filter(|test| within(&test.expr))
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            };
            let reads_with_tests = demand.reads_with_tests();
            let reads = reads_with_tests.as_ref().map(|reads| {
                let mut found: BTreeMap<Column, Expr> = (reads.iter())
                    .filter(|(column, _)| schema.has_column(column))
                    .map(|(column, read)| (column.clone(), weaken(read, &within)))
                    .collect();
                let on = join.on.iter().flat_map(|(left, right)| [left, right]);
                for expr in on.chain(join.filter.iter()) {
                    for column in expr.column_refs() {
                        if schema.has_column(column) {
                            found.insert(column.clone(), lit(true));
                        }
                    }
                }
                found
            });
            // a semi join needs one row of each key of its other side
            let count = match semi_join_keys(join, filtered) {
                Some(keys) if self.options.fetch => Count::PerKey(keys, 1),
                _ => Count::All,
            };
            sides.push((
                input.clone(),
                Demand {
                    tests,
                    reads,
                    count,
                },
            ));
        }
        let mut taken = Taken::default();
        let mut inputs = Vec::new();
        for (input, demand) in sides {
            let (input, below) = self.down(input, demand)?;
            taken.tests.extend(below.tests);
            inputs.push(input);
        }
        Ok((with_inputs(&plan, inputs)?, taken))
    }

    fn node(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let mut node = as_node::<AiNode>(&plan).expect("an AiNode").clone();
        let mut taken = Taken::default();
        let input_schema = Arc::clone(node.input.schema());
        let reads_input_only = |expr: &Expr| {
            expr.column_refs()
                .iter()
                .all(|column| input_schema.has_column(column))
        };
        // a test of the node's answer becomes its filter
        if self.options.filters && node.fetch.is_none() && node.semantics == Semantics::Rows {
            let returned: BTreeSet<String> = (node.returned_fields().iter())
                .map(|field| field.name().clone())
                .collect();
            for test in &demand.tests {
                let reads_returned = (test.expr.column_refs().iter())
                    .any(|column| column.relation.is_none() && returned.contains(&column.name));
                if reads_returned && node.accepts_test(&test.expr) {
                    node.filter =
                        conjunction(node.filter.take().into_iter().chain([test.expr.clone()]));
                    taken.tests.insert(test.id);
                }
            }
        }
        // the count becomes its fetch when every test is its filter or reads
        // only its input, which then moves below it
        let below_tests: Vec<&Test> = (demand.tests.iter())
            .filter(|test| !taken.tests.contains(&test.id) && reads_input_only(&test.expr))
            .collect();
        let decided = (demand.tests.iter())
            .all(|test| taken.tests.contains(&test.id) || reads_input_only(&test.expr));
        let fetch = match &demand.count {
            _ if !decided || !node.accepts_fetch() => None,
            Count::All => None,
            Count::Rows(rows) => match &node.fetch {
                Some(Fetch::PerKey { .. }) => None,
                Some(Fetch::Global(found)) if *found <= *rows => None,
                _ => Some(Fetch::Global(*rows)),
            },
            Count::Distinct(keys) => (node.semantics == Semantics::Rows && node.fetch.is_none())
                .then(|| distinct_key(&node, keys))
                .flatten()
                .map(|key| Fetch::PerKey { key, n: 1 }),
            Count::PerKey(keys, n) => stops_per_key(&node)
                .then(|| input_key(&node, keys))
                .flatten()
                .map(|key| Fetch::PerKey { key, n: *n }),
        };
        let mut input = node.input.as_ref().clone();
        let fetched = fetch.is_some() || node.fetch.is_some() && demand.count != Count::All;
        if let Some(fetch) = fetch {
            if let Some(predicate) = conjunction(below_tests.iter().map(|test| test.expr.clone())) {
                input = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?);
            }
            taken.tests.extend(below_tests.iter().map(|test| test.id));
            node.fetch = Some(fetch);
            taken.count = true;
        }
        // an extraction whose answer must contain a literal reads only the
        // documents that contain it
        if self.options.filters
            && node.fetch.is_none()
            && let Some(test) = precheck(&node, &input)
        {
            input = LogicalPlan::Filter(Filter::try_new(test, Arc::new(input))?);
        }
        // where the answer is read becomes the node's guard
        if self.options.guard && !fetched {
            // the tests the node took read its answer in its own filter
            let above = Demand {
                tests: (demand.tests.iter())
                    .filter(|test| !taken.tests.contains(&test.id))
                    .cloned()
                    .collect(),
                reads: demand.reads.clone(),
                count: Count::All,
            };
            guard(&mut node, &above);
        }
        // the node's input: the tests of its input, and the reads of its
        // input's columns and of its own expressions
        let tests = (demand.tests.iter())
            .filter(|test| !taken.tests.contains(&test.id) && reads_input_only(&test.expr))
            .cloned()
            .collect();
        let passed: BTreeSet<usize> = (demand.tests.iter())
            .filter(|test| !taken.tests.contains(&test.id) && reads_input_only(&test.expr))
            .map(|test| test.id)
            .collect();
        let held = Demand {
            tests: (demand.tests.iter())
                .filter(|test| !passed.contains(&test.id))
                .cloned()
                .collect(),
            reads: demand.reads.clone(),
            count: Count::All,
        };
        let reads = held.reads_with_tests().map(|reads| {
            let mut found: BTreeMap<Column, Expr> = (reads.iter())
                .filter(|(column, _)| input_schema.has_column(column))
                .map(|(column, read)| (column.clone(), weaken(read, &reads_input_only)))
                .collect();
            for expr in [Some(&node.call), node.guard.as_ref(), node.hint.as_ref()]
                .into_iter()
                .flatten()
            {
                for column in expr.column_refs() {
                    add_read(&mut found, column, lit(true));
                }
            }
            if let Some(filter) = &node.filter {
                for column in filter.column_refs() {
                    if input_schema.has_column(column) {
                        let within = read_condition(filter, column).unwrap_or(lit(true));
                        let within = weaken(&within, &reads_input_only);
                        add_read(&mut found, column, within);
                    }
                }
            }
            for test in &node.tests {
                for column in test.test.column_refs() {
                    add_read(&mut found, column, lit(true));
                }
            }
            found
        });
        // a node that returns each input row as it is passes the count on
        let count = match &demand.count {
            _ if taken.count || node.fetch.is_some() => Count::All,
            Count::Rows(_)
                if node.semantics == Semantics::Rows && node.filter.is_none() && decided =>
            {
                demand.count.clone()
            }
            _ => Count::All,
        };
        let (input, below) = self.down(
            input,
            Demand {
                tests,
                reads,
                count,
            },
        )?;
        taken.tests.extend(below.tests);
        taken.count |= below.count;
        let node = node.with_input(input)?;
        let rebuilt = extension_node(node.checked()?);
        Ok((rebuilt, taken))
    }
}

/// Guard a column node with where its answer is read, as far as the
/// executor can evaluate it: the terms that read only the one table the
/// call reads.
fn guard(node: &mut AiNode, demand: &Demand) {
    if node.function() == "ai_filter" {
        return or_guard(node, demand);
    }
    let plain = node.semantics == Semantics::Rows
        && !node.is_predicate()
        && node.filter.is_none()
        && node.tests.is_empty()
        && node.hint.is_none()
        && node.fetch.is_none();
    if !plain {
        return;
    }
    let added: Vec<Column> = (node.fields.iter())
        .map(|field| Column::from_name(field.name()))
        .collect();
    let Some(read) = added
        .iter()
        .filter_map(|column| demand.read_where(column))
        .reduce(|one, two| or(&one, &two))
    else {
        return;
    };
    let kept: Vec<Expr> = (split_conjunction(&read).into_iter())
        .filter(|term| !matches!(term, Expr::Literal(..)) && reads_call_table(term, node))
        .cloned()
        .collect();
    let guarded_already = |term: &Expr| {
        node.guard
            .as_ref()
            .is_some_and(|guard| split_conjunction(guard).contains(&term))
    };
    let new: Vec<Expr> = kept
        .into_iter()
        .filter(|term| !guarded_already(term))
        .collect();
    if let Some(found) = conjunction(new) {
        node.guard = Some(match node.guard.take() {
            Some(guard) => guard.and(found),
            None => found,
        });
    }
}

/// Guard an `AI.IF` of an OR with the OR's other terms: it is asked only
/// where none of them is TRUE, as far as the executor can tell before
/// asking. The OR's last `AI.IF` filters on the OR, so its own filter reads
/// its answer; each earlier one returns its answer, which the last one's
/// filter reads. A term the guard keeps reads only the one table the call
/// reads, or is an earlier `AI.IF`'s answer not TRUE.
fn or_guard(node: &mut AiNode, demand: &Demand) {
    if node.semantics != Semantics::Rows
        || node.guard.is_some()
        || node.hint.is_some()
        || node.fetch.is_some()
    {
        return;
    }
    let answer = Column::from_name(node.answer.clone());
    let read = match &node.filter {
        // nothing above reads the last one's answer, so it returns none
        Some(filter) if unread(node, demand) && or_terms(node).is_some() => {
            read_condition(filter, &answer)
        }
        Some(_) => None,
        None => demand.read_where(&answer),
    };
    let Some(read) = read else {
        return;
    };
    let earlier = earlier_answers(&node.input);
    let is_earlier = |expr: &Expr| {
        let read = match expr {
            Expr::Not(inner) => inner.as_ref(),
            other => other,
        };
        read.try_as_col()
            .is_some_and(|column| column.relation.is_none() && earlier.contains(&column.name))
    };
    let kept = (split_conjunction(&read).into_iter()).filter(|term| match term {
        Expr::IsNotTrue(inner) if is_earlier(inner) => true,
        Expr::Literal(..) => false,
        other => reads_call_table(other, node),
    });
    node.guard = conjunction(kept.cloned());
}

/// Whether no operator above reads the node's answer: it does not
/// return it, or the demand reads it nowhere.
fn unread(node: &AiNode, demand: &Demand) -> bool {
    let returned = (node.returned_fields().iter()).any(|field| *field.name() == node.answer);
    !returned
        || demand
            .read_where(&Column::from_name(node.answer.clone()))
            .is_none()
}

/// The answers of the `AI.IF` nodes below that return their answer and
/// drop no row, as each earlier `AI.IF` of an OR does, guarded or not.
fn earlier_answers(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|below| {
        if let Some(asked) = as_node::<AiNode>(below)
            && asked.semantics == Semantics::Rows
            && asked.function() == "ai_filter"
            && asked.filter.is_none()
            && asked.hint.is_none()
            && asked.fetch.is_none()
        {
            found.push(asked.answer.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The keys of a semi join's filtered side, when the join compares plain
/// columns and has no other test.
fn semi_join_keys(join: &datafusion::logical_expr::Join, filtered: bool) -> Option<Vec<Column>> {
    if !filtered || join.filter.is_some() || join.on.is_empty() {
        return None;
    }
    let right = join.join_type == JoinType::LeftSemi;
    (join.on.iter())
        .map(|(left, other)| match if right { other } else { left } {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect()
}

fn keys_below(keys: &[Column], below: &dyn Fn(&Expr) -> Option<Expr>) -> Option<Vec<Column>> {
    (keys.iter())
        .map(|key| match below(&Expr::Column(key.clone()))? {
            Expr::Column(column) => Some(column),
            _ => None,
        })
        .collect()
}

/// The conjunction of `cond`'s terms that `keep` accepts: a weaker
/// condition, TRUE when it accepts none.
fn weaken(cond: &Expr, keep: &dyn Fn(&Expr) -> bool) -> Expr {
    conjunction(
        split_conjunction(cond)
            .into_iter()
            .filter(|term| keep(term))
            .cloned(),
    )
    .unwrap_or(lit(true))
}

/// `cond` below an operator that renames columns: each term `below`
/// rewrites, TRUE when it rewrites none.
fn weaken_through(cond: &Expr, below: &dyn Fn(&Expr) -> Option<Expr>) -> Expr {
    conjunction(split_conjunction(cond).into_iter().filter_map(below)).unwrap_or(lit(true))
}

fn add_read(reads: &mut BTreeMap<Column, Expr>, column: &Column, read: Expr) {
    let read = match reads.remove(column) {
        Some(found) => or(&found, &read),
        None => read,
    };
    reads.insert(column.clone(), read);
}

fn is_true(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value, _) if value == &datafusion::scalar::ScalarValue::Boolean(Some(true)))
}

fn and(one: &Expr, two: &Expr) -> Expr {
    match (is_true(one), is_true(two)) {
        (true, _) => two.clone(),
        (_, true) => one.clone(),
        _ => one.clone().and(two.clone()),
    }
}

fn or(one: &Expr, two: &Expr) -> Expr {
    if is_true(one) || is_true(two) {
        lit(true)
    } else if one == two {
        one.clone()
    } else {
        one.clone().or(two.clone())
    }
}

/// Where `expr`'s value depends on `column`'s: None when nowhere, TRUE
/// when everywhere. A `CASE` reads a branch where its condition is TRUE;
/// `a AND b` reads `b` where `a` is not FALSE, and `a OR b` where `a` is
/// not TRUE.
pub(super) fn read_condition(expr: &Expr, column: &Column) -> Option<Expr> {
    let reads = |expr: &Expr| expr.column_refs().contains(column);
    if !reads(expr) {
        return None;
    }
    match expr {
        Expr::Column(found) if found == column => Some(lit(true)),
        Expr::Alias(alias) => read_condition(&alias.expr, column),
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::And | Operator::Or) =>
        {
            // `x OR y` is not TRUE where neither term is TRUE
            let gate = |other: &Expr| match op {
                Operator::And => other.clone().is_not_false(),
                _ => conjunction(
                    (split_binary(other, Operator::Or).into_iter())
                        .map(|term| term.clone().is_not_true()),
                )
                .unwrap_or(lit(true)),
            };
            match (reads(left), reads(right)) {
                (true, false) => Some(and(&gate(right), &read_condition(left, column)?)),
                (false, true) => Some(and(&gate(left), &read_condition(right, column)?)),
                _ => Some(lit(true)),
            }
        }
        Expr::Case(Case {
            expr: None,
            when_then_expr,
            else_expr,
        }) => {
            if when_then_expr.iter().any(|(when, _)| reads(when)) {
                return Some(lit(true));
            }
            let mut found: Option<Expr> = None;
            let mut earlier: Vec<Expr> = Vec::new();
            let not_earlier = |earlier: &[Expr]| {
                conjunction(earlier.iter().map(|when| when.clone().is_not_true()))
                    .unwrap_or(lit(true))
            };
            for (when, then) in when_then_expr {
                if let Some(within) = read_condition(then, column) {
                    let here = and(&and(&not_earlier(&earlier), when), &within);
                    found = Some(found.map_or(here.clone(), |found| or(&found, &here)));
                }
                earlier.push(when.as_ref().clone());
            }
            if let Some(otherwise) = else_expr
                && let Some(within) = read_condition(otherwise, column)
            {
                let here = and(&not_earlier(&earlier), &within);
                found = Some(found.map_or(here.clone(), |found| or(&found, &here)));
            }
            found
        }
        _ => Some(lit(true)),
    }
}
