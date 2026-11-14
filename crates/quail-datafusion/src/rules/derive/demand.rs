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
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::{conjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{
    BinaryExpr, Distinct, EmptyRelation, Expr, FetchType, Filter, JoinType, LogicalPlan, Operator,
    SkipType, lit,
};

use super::fetch::{at_least, distinct_key, input_key, is_count_star, stops_per_key};
use crate::ai::{AiNode, Fetch, Semantics};
use crate::derive::unfolded;
use crate::extension::{as_node, extension_node, with_inputs};
use crate::rules::distinct::{is_distinct, is_unique};
use crate::rules::guards::reads_call_table;
use crate::rules::or_guards::or_terms;
use crate::rules::projection::{question_fields, replace_call, with_fields};
use crate::rules::repeated::{through_alias, through_projection};
use crate::rules::substring::precheck;
use quail_planner::constants::SPAN_SUFFIX;

/// Which options the pass derives.
#[derive(Debug, Clone, Copy)]
pub(super) struct Options {
    pub(super) filters: bool,
    pub(super) fetch: bool,
    pub(super) guard: bool,
    pub(super) projection: bool,
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
    /// Whether every read reached here through operators that return
    /// their input's rows as they are: a value every row below has, every
    /// read above sees.
    carried: bool,
    /// The columns of `reads` read only as named fields of a struct, such
    /// as `a['who']`, with those fields.
    fields: BTreeMap<Column, BTreeSet<String>>,
}

impl Demand {
    fn everything() -> Self {
        Demand {
            tests: Vec::new(),
            reads: None,
            count: Count::All,
            carried: false,
            fields: BTreeMap::new(),
        }
    }

    /// Where `column` is read, by the tests or otherwise: None when
    /// nowhere.
    fn read_where(&self, column: &Column) -> Option<Expr> {
        let Some(reads) = &self.reads else {
            return Some(lit(true));
        };
        // a value is needed only in the rows that may pass the tests
        let passing =
            conjunction(self.tests.iter().map(|test| test.expr.clone())).unwrap_or(lit(true));
        (reads
            .get(column)
            .map(|read| and(read, &passing))
            .into_iter())
        .chain((self.tests.iter()).filter_map(|test| read_condition(&test.expr, column)))
        .reduce(|one, two| or(&one, &two))
    }

    /// The fields of `column` read, by the tests or otherwise: None when
    /// it is read whole, or may be.
    fn fields_read(&self, column: &Column) -> Option<BTreeSet<String>> {
        let reads = self.reads.as_ref()?;
        let mut found = BTreeSet::new();
        if reads.contains_key(column) {
            found.extend(self.fields.get(column)?.iter().cloned());
        }
        for test in &self.tests {
            if test.expr.column_refs().contains(column) {
                found.extend(read_fields(&test.expr, column)?);
            }
        }
        Some(found)
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
    /// Answers a node below no longer returns, with the value every row
    /// it returns has: each operator above that reads one reads the value.
    kept: Vec<(String, bool)>,
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
            // an output nothing reads is still computed, so its input
            // columns must exist, though no row reads their values
            let read = match demand.read_where(&column) {
                Some(read) => weaken_through(&read, &below),
                None => lit(false),
            };
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
        // the fields of each input column that the read outputs read
        let mut fields: BTreeMap<Column, BTreeSet<String>> = BTreeMap::new();
        let mut whole: BTreeSet<Column> = BTreeSet::new();
        for (index, expr) in projection.expr.iter().enumerate() {
            let column = Column::from(projection.schema.qualified_field(index));
            if demand.read_where(&column).is_none() {
                continue;
            }
            let expr = expr.clone().unalias();
            for input in expr.column_refs() {
                let found = match &expr {
                    Expr::Column(plain) if plain == input => demand.fields_read(&column),
                    _ => read_fields(&expr, input),
                };
                match found {
                    Some(found) if !whole.contains(input) => {
                        fields.entry(input.clone()).or_default().extend(found)
                    }
                    _ => {
                        whole.insert(input.clone());
                        fields.remove(input);
                    }
                }
            }
        }
        let input = projection.input.as_ref().clone();
        let (input, taken) = self.down(
            input,
            Demand {
                tests,
                reads: Some(reads),
                count,
                carried: demand.carried,
                fields,
            },
        )?;
        Ok((
            read_kept(with_inputs(&plan, vec![input])?, &taken.kept)?,
            taken,
        ))
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
        let fields = (demand.fields.iter())
            .filter_map(|(column, found)| Some((column_below(column)?, found.clone())))
            .collect();
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
                carried: demand.carried,
                fields,
            },
        )?;
        Ok((
            read_kept(with_inputs(&plan, vec![input])?, &taken.kept)?,
            taken,
        ))
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
                carried: demand.carried,
                fields: demand.fields.clone(),
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
        Ok((read_kept(plan, &taken.kept)?, taken))
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
                carried: demand.carried,
                fields: without_tests(&demand.fields, &demand.tests),
            },
        )?;
        // a LIMIT with no offset that a node took leaves the plan
        if taken.count && literal.is_some_and(|(skip, _)| skip == 0) {
            return Ok((
                input,
                Taken {
                    kept: taken.kept,
                    ..Taken::default()
                },
            ));
        }
        Ok((
            with_inputs(&plan, vec![input])?,
            Taken {
                kept: taken.kept,
                ..Taken::default()
            },
        ))
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
        let mut fields = demand.fields.clone();
        for expr in &sort.expr {
            for column in expr.expr.column_refs() {
                fields.remove(column);
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
                carried: demand.carried,
                fields,
            },
        )?;
        // a count taken below is the sort's, not the demand's
        taken.count = false;
        Ok((
            read_kept(with_inputs(&plan, vec![input])?, &taken.kept)?,
            taken,
        ))
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
                carried: false,
                fields: BTreeMap::new(),
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
                carried: false,
                fields: BTreeMap::new(),
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
        // an outer join's unmatched rows hold NULLs, not a kept value
        let carried = demand.carried && join.join_type == JoinType::Inner;
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
            let on = (join.on.iter().flat_map(|(left, right)| [left, right]))
                .chain(join.filter.iter())
                .flat_map(|expr| expr.column_refs())
                .collect::<BTreeSet<_>>();
            let fields = (without_tests(&demand.fields, &demand.tests).into_iter())
                .filter(|(column, _)| schema.has_column(column) && !on.contains(column))
                .collect();
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
                    carried,
                    fields,
                },
            ));
        }
        let mut taken = Taken::default();
        let mut inputs = Vec::new();
        for (input, demand) in sides {
            let (input, below) = self.down(input, demand)?;
            taken.tests.extend(below.tests);
            taken.kept.extend(below.kept);
            inputs.push(input);
        }
        Ok((read_kept(with_inputs(&plan, inputs)?, &taken.kept)?, taken))
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
                // a call written twice tests the one answer twice
                let held =
                    (node.filter.iter().flat_map(split_conjunction)).any(|term| *term == test.expr);
                if reads_returned && (held || node.accepts_test(&test.expr)) {
                    if !held {
                        node.filter =
                            conjunction(node.filter.take().into_iter().chain([test.expr.clone()]));
                    }
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
        // the tests the node took read its answer in its own filter
        let above = Demand {
            tests: (demand.tests.iter())
                .filter(|test| !taken.tests.contains(&test.id))
                .cloned()
                .collect(),
            reads: demand.reads.clone(),
            count: Count::All,
            carried: demand.carried,
            fields: demand.fields.clone(),
        };
        // where the answer is read becomes the node's guard
        if self.options.guard && !fetched {
            guard(&mut node, &above);
        }
        let mut kept = None;
        if self.options.projection {
            kept = project(&mut node, &above);
            ask_read_fields(&mut node, &above)?;
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
            carried: demand.carried,
            fields: demand.fields.clone(),
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
        // the node's own expressions read their columns whole
        let own: BTreeSet<Column> = ([Some(&node.call), node.guard.as_ref(), node.hint.as_ref()])
            .into_iter()
            .flatten()
            .chain(node.filter.iter())
            .chain(node.tests.iter().map(|test| &test.test))
            .flat_map(|expr| expr.column_refs().into_iter().cloned())
            .collect();
        let fields = (without_tests(&held.fields, &held.tests).into_iter())
            .filter(|(column, _)| input_schema.has_column(column) && !own.contains(column))
            .collect();
        let (input, below) = self.down(
            input,
            Demand {
                tests,
                reads,
                count,
                carried: demand.carried,
                fields,
            },
        )?;
        taken.tests.extend(below.tests);
        taken.count |= below.count;
        let node = node.with_input(input)?;
        let rebuilt = read_kept(extension_node(node.checked()?), &below.kept)?;
        taken.kept = below.kept;
        taken.kept.extend(kept);
        Ok((rebuilt, taken))
    }
}

/// Return only the added columns the plan above reads, the answer always.
/// An `AI.IF` that keeps only the rows whose answer is `b`, or only those
/// whose answer is `NOT b`, returns no column: a read of the answer above
/// reads the value every row it returns has, which this returns.
fn project(node: &mut AiNode, above: &Demand) -> Option<(String, bool)> {
    if node.semantics != Semantics::Rows || above.reads.is_none() {
        return None;
    }
    let read = |name: &str| above.read_where(&Column::from_name(name)).is_some();
    if !node.is_predicate() {
        let returned =
            (node.projection.clone()).unwrap_or_else(|| (0..node.fields.len()).collect());
        let kept: Vec<usize> = (returned.iter().copied())
            .filter(|index| *index == 0 || read(node.fields[*index].name()))
            .collect();
        if kept.len() < returned.len() {
            node.projection = Some(kept);
        }
        return None;
    }
    let returns_answer = (node.returned_fields().iter()).any(|field| *field.name() == node.answer);
    // a score only its own test reads returns nothing
    if node.function() == "ai_score" && returns_answer && !read(&node.answer) {
        node.projection = Some(Vec::new());
    }
    if node.function() != "ai_filter" || !returns_answer {
        return None;
    }
    let filter = node.filter.as_ref()?;
    if !read(&node.answer) {
        node.projection = Some(Vec::new());
        return None;
    }
    let negated = node.answer_term(filter)?;
    if !above.carried {
        return None;
    }
    node.projection = Some(Vec::new());
    Some((node.answer.clone(), !negated))
}

/// Ask an extraction only the fields the plan above and its own filter
/// and tests read, of its answer and its spans; each field is a question
/// of its own, so leaving one out changes no other's answer.
fn ask_read_fields(node: &mut AiNode, above: &Demand) -> Result<()> {
    if node.function() != "ai_extract" || node.semantics != Semantics::Rows || above.reads.is_none()
    {
        return Ok(());
    }
    let call = unfolded(node.call.clone());
    let Some(questions) = question_fields(&call) else {
        return Ok(());
    };
    let names: Vec<Column> = [node.answer.clone(), format!("{}{SPAN_SUFFIX}", node.answer)]
        .into_iter()
        .filter(|name| node.fields.iter().any(|field| field.name() == name))
        .map(Column::from_name)
        .collect();
    let mut read = BTreeSet::new();
    for name in &names {
        if above.read_where(name).is_some() {
            let Some(found) = above.fields_read(name) else {
                return Ok(());
            };
            read.extend(found);
        }
        for expr in node
            .filter
            .iter()
            .chain(node.tests.iter().map(|test| &test.test))
        {
            if expr.column_refs().contains(name) {
                let Some(found) = read_fields(expr, name) else {
                    return Ok(());
                };
                read.extend(found);
            }
        }
    }
    let kept: Vec<usize> = (questions.iter().enumerate())
        .filter(|(_, question)| read.contains(*question))
        .map(|(index, _)| index)
        .collect();
    if kept.is_empty() || kept.len() == questions.len() {
        return Ok(());
    }
    // the call changes in this node alone: a node below with the same call
    // asks the fields its own readers read
    let input = Arc::clone(&node.input);
    let alone = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema: Arc::clone(input.schema()),
    });
    let alone = extension_node(node.with_input(alone)?.checked()?);
    let asked = replace_call(alone, &call, &with_fields(&call, &kept))?;
    let found = as_node::<AiNode>(&asked).expect("an AiNode");
    *node = found.with_input(Arc::unwrap_or_clone(input))?;
    Ok(())
}

/// `fields` without the columns the tests read, for an operator that
/// reads them as it folds the tests into its reads.
fn without_tests(
    fields: &BTreeMap<Column, BTreeSet<String>>,
    tests: &[Test],
) -> BTreeMap<Column, BTreeSet<String>> {
    let tested: BTreeSet<&Column> = tests
        .iter()
        .flat_map(|test| test.expr.column_refs())
        .collect();
    (fields.iter())
        .filter(|(column, _)| !tested.contains(column))
        .map(|(column, found)| (column.clone(), found.clone()))
        .collect()
}

/// The fields of `column` that `expr` reads when it reads `column` only as
/// `get_field(column, 'name')`; None when it reads more of it.
fn read_fields(expr: &Expr, column: &Column) -> Option<BTreeSet<String>> {
    let mut fields = BTreeSet::new();
    let mut whole = false;
    let _ = expr.apply(|found| {
        if let Expr::ScalarFunction(call) = found
            && call.name() == "get_field"
            && matches!(call.args.first(), Some(Expr::Column(read)) if read == column)
        {
            let name = match call.args.get(1) {
                Some(Expr::Literal(value, _)) => value.try_as_str().flatten(),
                _ => None,
            };
            match name {
                Some(name) => {
                    fields.insert(name.to_string());
                    return Ok(TreeNodeRecursion::Jump);
                }
                None => whole = true,
            }
        }
        if matches!(found, Expr::Column(read) if read == column) {
            whole = true;
        }
        Ok(if whole {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    (!whole).then_some(fields)
}

/// `plan` with each read of a kept answer as its value, each expression
/// keeping its name.
fn read_kept(plan: LogicalPlan, kept: &[(String, bool)]) -> Result<LogicalPlan> {
    if kept.is_empty() {
        return Ok(plan);
    }
    let preserver = NamePreserver::new(&plan);
    let mut changed = false;
    let mut exprs = Vec::new();
    for expr in plan.expressions() {
        let saved = preserver.save(&expr);
        let read = expr.transform_up(|found| {
            let value = match &found {
                // an answer's name is Quail's own, whatever qualifies it
                Expr::Column(column) => (kept.iter())
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
    if !changed {
        return Ok(plan);
    }
    let inputs = plan.inputs().into_iter().cloned().collect();
    plan.with_new_exprs(exprs, inputs)
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
    // a part the node cannot evaluate may be anything
    let known = |atom: &Expr| reads_rows_values(atom, node);
    let read = bounds(&read, &known).may_be_true;
    let kept: Vec<Expr> = (split_conjunction(&read).into_iter())
        .filter(|term| !matches!(term, Expr::Literal(..)))
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

/// What a condition may and must be, where some of its parts are
/// unknown: each is a condition of the known parts alone, never NULL.
struct Bounds {
    may_be_true: Expr,
    may_be_false: Expr,
    is_true: Expr,
    is_false: Expr,
}

/// The bounds of `expr` when the parts `known` rejects may take any value.
/// A part is a term other than AND, OR, NOT, and the IS TRUE family.
fn bounds(expr: &Expr, known: &dyn Fn(&Expr) -> bool) -> Bounds {
    let not = |expr: Expr| match expr {
        Expr::Literal(..) if is_true(&expr) => lit(false),
        Expr::Literal(..) if is_false(&expr) => lit(true),
        other => Expr::Not(Box::new(other)),
    };
    match expr {
        Expr::Literal(..) if is_true(expr) => Bounds {
            may_be_true: lit(true),
            may_be_false: lit(false),
            is_true: lit(true),
            is_false: lit(false),
        },
        Expr::Literal(..) if is_false(expr) => Bounds {
            may_be_true: lit(false),
            may_be_false: lit(true),
            is_true: lit(false),
            is_false: lit(true),
        },
        Expr::Alias(alias) => bounds(&alias.expr, known),
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::And | Operator::Or) =>
        {
            let (one, two) = (bounds(left, known), bounds(right, known));
            if *op == Operator::And {
                Bounds {
                    may_be_true: and(&one.may_be_true, &two.may_be_true),
                    may_be_false: or(&one.may_be_false, &two.may_be_false),
                    is_true: and(&one.is_true, &two.is_true),
                    is_false: or(&one.is_false, &two.is_false),
                }
            } else {
                Bounds {
                    may_be_true: or(&one.may_be_true, &two.may_be_true),
                    may_be_false: and(&one.may_be_false, &two.may_be_false),
                    is_true: or(&one.is_true, &two.is_true),
                    is_false: and(&one.is_false, &two.is_false),
                }
            }
        }
        Expr::Not(inner) => {
            let found = bounds(inner, known);
            Bounds {
                may_be_true: found.may_be_false,
                may_be_false: found.may_be_true,
                is_true: found.is_false,
                is_false: found.is_true,
            }
        }
        // x IS NOT TRUE is TRUE where x is FALSE or NULL
        Expr::IsNotTrue(inner) => {
            let found = bounds(inner, known);
            Bounds {
                may_be_true: not(found.is_true.clone()),
                may_be_false: found.may_be_true.clone(),
                is_true: not(found.may_be_true),
                is_false: found.is_true,
            }
        }
        Expr::IsNotFalse(inner) => {
            let found = bounds(inner, known);
            Bounds {
                may_be_true: not(found.is_false.clone()),
                may_be_false: found.may_be_false.clone(),
                is_true: not(found.may_be_false),
                is_false: found.is_false,
            }
        }
        Expr::IsTrue(inner) => {
            let found = bounds(inner, known);
            Bounds {
                may_be_true: found.may_be_true.clone(),
                may_be_false: not(found.is_true.clone()),
                is_true: found.is_true,
                is_false: not(found.may_be_true),
            }
        }
        Expr::IsFalse(inner) => {
            let found = bounds(inner, known);
            Bounds {
                may_be_true: found.may_be_false.clone(),
                may_be_false: not(found.is_false.clone()),
                is_true: found.is_false,
                is_false: not(found.may_be_false),
            }
        }
        part if known(part) => Bounds {
            may_be_true: part.clone().is_true(),
            may_be_false: part.clone().is_false(),
            is_true: part.clone().is_true(),
            is_false: part.clone().is_false(),
        },
        _ => Bounds {
            may_be_true: lit(true),
            may_be_false: lit(true),
            is_true: lit(false),
            is_false: lit(false),
        },
    }
}

/// Whether the executor can evaluate `term` on the rows a node reads:
/// it calls no AI function and reads columns of the one table the
/// node's call reads, and the answers of AI operators below the node,
/// which the rows carry.
fn reads_rows_values(term: &Expr, node: &AiNode) -> bool {
    if reads_call_table(term, node) {
        return true;
    }
    let tables: BTreeSet<String> = (node.call.column_refs().into_iter())
        .filter_map(|column| {
            column
                .relation
                .as_ref()
                .map(|table| table.table().to_string())
        })
        .collect();
    let mut earlier = rows_answers(&node.input);
    // classifications of the same rows run as one operator, so one cannot
    // read another's label before it runs
    if node.function() == "ai_classify" {
        earlier.retain(|name| !classified(&node.input, name));
    }
    let input = node.input.schema();
    tables.len() == 1
        && !term
            .exists(|inner| Ok(crate::is_ai_call(inner)))
            .unwrap_or(true)
        && term.column_refs().iter().all(|column| {
            input.has_column(column)
                && match &column.relation {
                    Some(table) => tables.contains(table.table()),
                    None => earlier.contains(&column.name),
                }
        })
}

/// Whether a classification below adds the column `name`.
fn classified(plan: &LogicalPlan, name: &str) -> bool {
    plan.exists(|below| {
        Ok(as_node::<AiNode>(below).is_some_and(|asked| {
            asked.function() == "ai_classify"
                && asked.fields.iter().any(|field| field.name() == name)
        }))
    })
    .unwrap_or(true)
}

/// The columns the AI operators of rows below add, which each row they
/// return carries. A map's text and SAE features are not carried on the
/// rows, so a guard cannot read them.
fn rows_answers(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let _ = plan.apply(|below| {
        if let Some(asked) = as_node::<AiNode>(below)
            && asked.semantics == Semantics::Rows
            && !matches!(asked.function(), "ai_map" | "ai_sae_features")
        {
            found.extend(asked.fields.iter().map(|field| field.name().clone()));
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
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

fn is_false(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value, _) if value == &datafusion::scalar::ScalarValue::Boolean(Some(false)))
}

fn and(one: &Expr, two: &Expr) -> Expr {
    if is_false(one) || is_false(two) {
        return lit(false);
    }
    match (is_true(one), is_true(two)) {
        (true, _) => two.clone(),
        (_, true) => one.clone(),
        _ => one.clone().and(two.clone()),
    }
}

fn or(one: &Expr, two: &Expr) -> Expr {
    if is_true(one) || is_true(two) {
        lit(true)
    } else if is_false(two) {
        one.clone()
    } else if is_false(one) {
        two.clone()
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
