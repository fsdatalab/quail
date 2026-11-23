use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::{conjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{
    BinaryExpr, Distinct, EmptyRelation, Expr, FetchType, Filter, JoinType, LogicalPlan, Operator,
    SkipType, UserDefinedLogicalNodeCore, lit,
};

use super::fetch::{at_least, is_count_star};
use crate::ai::{AiNode, Semantics};
use crate::functions::options::unfolded;
use crate::plan::{as_node, extension_node, with_inputs};
use crate::rules::distinct::{is_distinct, is_unique};
use crate::rules::projection::{question_fields, replace_call, with_fields};
use crate::rules::repeated::{through_alias, through_projection};
use crate::rules::substring::precheck;
use quail_planner::constants::SPAN_SUFFIX;

#[derive(Debug, Clone, Copy)]
pub(super) struct Options {
    pub(super) filters: bool,
    pub(super) fetch: bool,
    pub(super) guard: bool,
    pub(super) projection: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum Count {
    All,
    Rows(usize),
    Distinct(Vec<Column>),
    PerKey(Vec<Column>, usize),
}

#[derive(Debug, Clone)]
struct Test {
    id: usize,
    expr: Expr,
}

#[derive(Debug, Clone)]
struct Demand {
    tests: Vec<Test>,
    reads: Option<BTreeMap<Column, Expr>>,
    count: Count,
    carried: bool,
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

    fn read_where(&self, column: &Column) -> Option<Expr> {
        let Some(reads) = &self.reads else {
            return Some(lit(true));
        };
        let passing =
            conjunction(self.tests.iter().map(|test| test.expr.clone())).unwrap_or(lit(true));
        (reads
            .get(column)
            .map(|read| and(read, &passing))
            .into_iter())
        .chain((self.tests.iter()).filter_map(|test| read_condition(&test.expr, column)))
        .reduce(|one, two| or(&one, &two))
    }

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

#[derive(Debug, Default)]
struct Taken {
    tests: BTreeSet<usize>,
    count: bool,
    kept: Vec<(String, bool)>,
}

struct Pass {
    options: Options,
    next_test: usize,
}

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

    fn down(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
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
        let mut reads = BTreeMap::new();
        for (index, expr) in projection.expr.iter().enumerate() {
            let column = Column::from(projection.schema.qualified_field(index));
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
        taken.count = false;
        Ok((
            read_kept(with_inputs(&plan, vec![input])?, &taken.kept)?,
            taken,
        ))
    }

    fn distinct(&mut self, plan: LogicalPlan, demand: Demand) -> Result<(LogicalPlan, Taken)> {
        let input = plan.inputs()[0].clone();
        let keys = input.schema().columns();
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
        let count = match (aggregate.aggr_expr.as_slice(), &groups) {
            ([counted], Some(keys)) if is_count_star(counted) && !keys.is_empty() => {
                let column =
                    Column::from(aggregate.schema.qualified_field(aggregate.group_expr.len()));
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
        let (keeps_left, keeps_right) = match join.join_type {
            JoinType::Inner => (true, true),
            JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => (true, false),
            JoinType::Right | JoinType::RightSemi | JoinType::RightAnti => (false, true),
            _ => (false, false),
        };
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
        let input_schema = Arc::clone(node.input_schema());
        let reads_input_only = |expr: &Expr| {
            expr.column_refs()
                .iter()
                .all(|column| input_schema.has_column(column))
        };
        if self.options.filters {
            for test in &demand.tests {
                if node.take_test(&test.expr) {
                    taken.tests.insert(test.id);
                }
            }
        }
        let below_tests: Vec<&Test> = (demand.tests.iter())
            .filter(|test| !taken.tests.contains(&test.id) && reads_input_only(&test.expr))
            .collect();
        let decided = (demand.tests.iter())
            .all(|test| taken.tests.contains(&test.id) || reads_input_only(&test.expr));
        let fetch = match &demand.count {
            _ if !decided || !node.accepts_fetch() => None,
            Count::All => None,
            Count::Rows(rows) => node.rows_fetch(*rows),
            Count::Distinct(keys) => node.distinct_fetch(keys),
            Count::PerKey(keys, n) => node.per_key_fetch(keys, *n),
        };
        let mut input = node.inputs()[0].clone();
        let fetched = fetch.is_some() || node.has_fetch() && demand.count != Count::All;
        if let Some(fetch) = fetch {
            if let Some(predicate) = conjunction(below_tests.iter().map(|test| test.expr.clone())) {
                input = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?);
            }
            taken.tests.extend(below_tests.iter().map(|test| test.id));
            node = node.with_fetch(fetch);
            taken.count = true;
        }
        if self.options.filters
            && !node.has_fetch()
            && let Some(test) = precheck(&node, &input)
        {
            input = LogicalPlan::Filter(Filter::try_new(test, Arc::new(input))?);
        }
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
        if self.options.guard && !fetched {
            guard(&mut node, &above);
        }
        let mut kept = None;
        if self.options.projection {
            kept = project(&mut node, &above);
            ask_read_fields(&mut node, &above)?;
        }
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
            for expr in node.asked_expressions() {
                for column in expr.column_refs() {
                    add_read(&mut found, column, lit(true));
                }
            }
            if let Some(filter) = node.filter() {
                for column in filter.column_refs() {
                    if input_schema.has_column(column) {
                        let within = read_condition(&filter, column).unwrap_or(lit(true));
                        let within = weaken(&within, &reads_input_only);
                        add_read(&mut found, column, within);
                    }
                }
            }
            for test in node.tests() {
                for column in test.test.column_refs() {
                    add_read(&mut found, column, lit(true));
                }
            }
            found
        });
        let count = match &demand.count {
            _ if taken.count || node.has_fetch() => Count::All,
            Count::Rows(_)
                if node.semantics() == Semantics::Rows && node.filter().is_none() && decided =>
            {
                demand.count.clone()
            }
            _ => Count::All,
        };
        let own: BTreeSet<Column> = (node.asked_expressions().into_iter())
            .chain(node.filter_expressions())
            .flat_map(|expr| expr.column_refs().into_iter().cloned().collect::<Vec<_>>())
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

fn project(node: &mut AiNode, above: &Demand) -> Option<(String, bool)> {
    above.reads.as_ref()?;
    node.project(
        |name| above.read_where(&Column::from_name(name)).is_some(),
        above.carried,
    )
}

fn ask_read_fields(node: &mut AiNode, above: &Demand) -> Result<()> {
    if node.function() != "ai_extract"
        || node.semantics() != Semantics::Rows
        || above.reads.is_none()
    {
        return Ok(());
    }
    let call = unfolded(node.call());
    let Some(questions) = question_fields(&call) else {
        return Ok(());
    };
    let names: Vec<Column> = [
        node.answer().to_string(),
        format!("{}{SPAN_SUFFIX}", node.answer()),
    ]
    .into_iter()
    .filter(|name| node.is_field(name))
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
        for expr in node.filter_expressions() {
            if expr.column_refs().contains(name) {
                let Some(found) = read_fields(&expr, name) else {
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
    let input = node.inputs()[0].clone();
    let alone = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema: Arc::clone(input.schema()),
    });
    let alone = extension_node(node.with_input(alone)?.checked()?);
    let asked = replace_call(alone, &call, &with_fields(&call, &kept))?;
    let found = as_node::<AiNode>(&asked).expect("an AiNode");
    *node = found.with_input(input)?;
    Ok(())
}

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

fn guard(node: &mut AiNode, demand: &Demand) {
    if node.function() == "ai_filter" {
        return or_guard(node, demand);
    }
    if !node.is_plain_column() {
        return;
    }
    let Some(read) = (node.added_names().into_iter())
        .filter_map(|name| demand.read_where(&Column::from_name(name)))
        .reduce(|one, two| or(&one, &two))
    else {
        return;
    };
    let known = |atom: &Expr| node.evaluates_on_rows(atom);
    let read = bounds(&read, &known).may_be_true;
    let kept: Vec<Expr> = (split_conjunction(&read).into_iter())
        .filter(|term| !matches!(term, Expr::Literal(..)))
        .cloned()
        .collect();
    node.add_guard(kept);
}

struct Bounds {
    may_be_true: Expr,
    may_be_false: Expr,
    is_true: Expr,
    is_false: Expr,
}

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

fn or_guard(node: &mut AiNode, demand: &Demand) {
    if !node.takes_or_guard() {
        return;
    }
    let answer = Column::from_name(node.answer());
    let read = match node.filter() {
        Some(filter) if unread(node, demand) && node.or_terms().is_some() => {
            read_condition(&filter, &answer)
        }
        Some(_) => None,
        None => demand.read_where(&answer),
    };
    if let Some(read) = read {
        node.guard_or(&read);
    }
}

fn unread(node: &AiNode, demand: &Demand) -> bool {
    let returned = (node.returned_fields().iter()).any(|field| field.name() == node.answer());
    !returned
        || demand
            .read_where(&Column::from_name(node.answer()))
            .is_none()
}

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

fn weaken(cond: &Expr, keep: &dyn Fn(&Expr) -> bool) -> Expr {
    conjunction(
        split_conjunction(cond)
            .into_iter()
            .filter(|term| keep(term))
            .cloned(),
    )
    .unwrap_or(lit(true))
}

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
