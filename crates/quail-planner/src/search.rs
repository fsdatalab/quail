//! Choose each table's step order, the join order, and the anchors, as
//! `quail/planner/search` does.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use quail_cost::budgets::parallel_copies;
use quail_cost::joins::{
    PricedJoin, longest_join_input, packed_join_work, stage_work as join_work,
};
use quail_cost::quantity::Quantity;
use quail_cost::sol::speed_of_light;
use quail_cost::statistics::AliasStats;
use quail_cost::work::{Work, ask, scan};
use quail_physical::Prompt;

use crate::classify::classify_seconds;
use crate::columns::{ColumnSpec, column_spec};
use crate::context::PlanningContext;
use crate::costs::{CostCurve, prepare_filter_costs, question_tokens};
use crate::expressions::asked_call;
use crate::expressions::{CompileError, call_aliases, repr_str};
use crate::operators::{Operators, SemanticColumn};
use crate::plan_statistics::{
    PlanStatistics, expected_rows_after_filters, filtered_before_joins, plan_statistics,
};
use crate::scores::score_join_work;
use crate::statistics::{
    DEFAULT_SELECTIVITY, cross_tuples, effective_selectivity, surviving_documents, thin,
};

pub const ASK: &str = "ask";

/// A step's key: an ask's written position, or a column operator's name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StepKey {
    Position(usize),
    Name(String),
}

/// How a step's seconds are found at a live count.
#[derive(Debug, Clone)]
pub enum StepCost {
    Curve(CostCurve),
    /// A column operator's planner prices it.
    Column {
        column: Box<SemanticColumn>,
        resident: bool,
    },
}

impl StepCost {
    fn seconds(&self, context: &PlanningContext, rows: f64) -> Result<f64, CompileError> {
        match self {
            StepCost::Curve(curve) => Ok(curve.seconds(rows)),
            StepCost::Column { column, resident } => {
                column_seconds(context, column, *resident, rows)
            }
        }
    }
}

/// A column operator's seconds, as its planner's `seconds`.
///
/// # Errors
///
/// The tokenizer fails, or the model cannot run the operator.
pub fn column_seconds(
    context: &PlanningContext,
    column: &SemanticColumn,
    resident: bool,
    rows: f64,
) -> Result<f64, CompileError> {
    if column.call.kind == "label" {
        return classify_seconds(
            context,
            &column.call,
            &column.name,
            &column.alias(),
            resident,
            rows,
        );
    }
    Ok(match column_spec(context, column, rows, resident)? {
        ColumnSpec::Extract(spec) => spec.estimated_seconds,
        ColumnSpec::Map(spec) => spec.estimated_seconds,
        ColumnSpec::SaeFeatures(spec) => spec.estimated_seconds,
    })
}

/// One step of a table's sequence and its costs, as `TableStep`.
#[derive(Debug, Clone)]
pub struct TableStep {
    pub kind: &'static str,
    pub key: StepKey,
    pub selectivity: f64,
    pub first: StepCost,
    pub later: StepCost,
    pub reuse_key: Option<Prompt>,
    pub document_head: Vec<i64>,
}

/// Whether a step reads the KV the step before it left.
pub fn resident_after(previous: &TableStep, step: &TableStep) -> bool {
    previous.document_head == step.document_head
}

pub fn column_step_kind(kind: &str) -> &'static str {
    match kind {
        "label" => "classify",
        "extract" => "extract",
        "features" => "features",
        _ => "map",
    }
}

/// A table's steps in written order, priced, as `table_steps`.
///
/// # Errors
///
/// The tokenizer fails, or the arena does not fit the device.
/// How a table's steps are priced: by the planner's operator planners,
/// or, for the speed of light estimate, each as one question over the
/// live documents under no prompt head.
pub enum StepPricing<'a> {
    Plan(&'a PlanStatistics),
    Estimate { mean_tokens: f64, chunk: i64 },
}

pub fn table_steps(
    found: &Operators,
    alias: &str,
    asks: &[usize],
    pricing: &StepPricing,
    context: &PlanningContext,
) -> Result<Vec<TableStep>, CompileError> {
    let mut steps = Vec::new();
    let predicates = found
        .filters
        .iter()
        .find(|(owner, _)| owner == alias)
        .map(|(_, predicates)| predicates.clone())
        .unwrap_or_default();
    let (mean_tokens, chunk) = match pricing {
        StepPricing::Plan(statistics) => (
            statistics.pre as f64 + statistics.stats[alias].mean_document_tokens(),
            statistics.chunk,
        ),
        StepPricing::Estimate { mean_tokens, chunk } => (*mean_tokens, *chunk),
    };
    let estimate = matches!(pricing, StepPricing::Estimate { .. });
    let head = |prompt: &Prompt| -> Result<Vec<i64>, CompileError> {
        if estimate {
            Ok(Vec::new())
        } else {
            Ok(context.lowered(prompt)?.preamble_token_ids)
        }
    };
    let chosen: Vec<_> = asks
        .iter()
        .map(|position| predicates[*position].clone())
        .collect();
    let costs = prepare_filter_costs(&chosen, mean_tokens, chunk, context)?;
    for (position, cost) in asks.iter().zip(costs) {
        let prompt = &asked_call(&predicates[*position].expression).prompt;
        steps.push(TableStep {
            kind: ASK,
            key: StepKey::Position(*position),
            selectivity: cost.selectivity,
            first: StepCost::Curve(cost.first),
            later: StepCost::Curve(cost.later),
            reuse_key: cost.reuse_key,
            document_head: head(prompt)?,
        });
    }
    let mut tests: Vec<(String, Vec<f64>)> = Vec::new();
    if let Some((_, filters)) = found
        .column_filters
        .iter()
        .find(|(owner, _)| owner == alias)
    {
        for test in filters {
            match tests.iter_mut().find(|(name, _)| *name == test.name) {
                Some((_, found)) => found.push(effective_selectivity(test.selectivity)),
                None => tests.push((
                    test.name.clone(),
                    vec![effective_selectivity(test.selectivity)],
                )),
            }
        }
    }
    let mut columns: Vec<&SemanticColumn> = Vec::new();
    for (call, owner) in label_calls(found) {
        if owner != alias {
            continue;
        }
        if let Some(column) = found
            .classifies
            .iter()
            .find(|column| call_aliases(&column.call).len() == 1 && column.call == call)
        {
            columns.push(column);
        }
    }
    for list in [&found.extracts, &found.sae_features, &found.maps] {
        columns.extend(list.iter().filter(|column| column.alias() == alias));
    }
    for column in columns {
        let call = &column.call;
        let survival = tests
            .iter()
            .find(|(name, _)| *name == column.name)
            .map(|(_, found)| found.iter().product::<f64>())
            .unwrap_or(1.0);
        let (first, later) = if estimate {
            let tail = context.lowered(&call.prompt)?.tail_token_ids.len() as i64;
            let cost = |fresh: bool| {
                StepCost::Curve(CostCurve::Linear(quail_cost::filters::filter_cost(
                    tail,
                    mean_tokens,
                    &context.model,
                    &context.device,
                    chunk,
                    fresh,
                )))
            };
            (cost(true), cost(false))
        } else {
            (
                StepCost::Column {
                    column: Box::new(column.clone()),
                    resident: false,
                },
                StepCost::Column {
                    column: Box::new(column.clone()),
                    resident: true,
                },
            )
        };
        steps.push(TableStep {
            kind: column_step_kind(&call.kind),
            key: StepKey::Name(column.name.clone()),
            selectivity: survival,
            first,
            later,
            reuse_key: None,
            document_head: head(&call.prompt)?,
        });
    }
    Ok(steps)
}

/// The classification calls a filter tests or SELECT returns, as `LabelWork.calls`.
pub fn label_calls(found: &Operators) -> Vec<(quail_physical::ModelCall, String)> {
    let mut calls: Vec<(quail_physical::ModelCall, String)> = Vec::new();
    for (alias, tests) in &found.column_filters {
        for test in tests {
            if test.call.kind == "label" && !calls.iter().any(|(call, _)| *call == test.call) {
                calls.push((test.call.clone(), alias.clone()));
            }
        }
    }
    for column in &found.projections {
        if let Some(call) = crate::expressions::aliased_call(column, &["label"])
            && !calls.iter().any(|(found, _)| found == call)
        {
            calls.push((call.clone(), call_aliases(call)[0].clone()));
        }
    }
    calls
}

/// Choose an order using costs at each expected survivor count, as
/// `order_filters_indexed` with the steps' own residency rule.
///
/// # Errors
///
/// A step's cost fails.
pub fn order_steps(
    steps: &[TableStep],
    rule: &str,
    count: f64,
    context: &PlanningContext,
) -> Result<Vec<usize>, CompileError> {
    let indices: Vec<usize> = (0..steps.len()).collect();
    if rule == "as_written" || indices.len() < 2 {
        return Ok(indices);
    }
    let mut cache: HashMap<(usize, u64, bool), f64> = HashMap::new();
    let mut cost = |index: usize, rows: f64, first: bool| -> Result<f64, CompileError> {
        let key = (index, rows.to_bits(), first);
        if let Some(found) = cache.get(&key) {
            return Ok(*found);
        }
        let step = &steps[index];
        let seconds = if first { &step.first } else { &step.later }.seconds(context, rows)?;
        cache.insert(key, seconds);
        Ok(seconds)
    };
    let starters: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|index| steps[*index].selectivity < 1.0)
        .collect();
    if starters.is_empty() {
        return Ok(indices);
    }
    let mut best: Option<(f64, usize, Vec<usize>)> = None;
    for first in starters {
        let mut ordered = vec![first];
        let mut total = cost(first, count, true)?;
        let mut expected = count * steps[first].selectivity;
        let mut scored: Vec<Prompt> = steps[first].reuse_key.iter().cloned().collect();
        let mut remaining: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|index| *index != first)
            .collect();
        let mut previous = first;
        while !remaining.is_empty() {
            let mut choice: Option<(f64, usize, f64)> = None;
            for &index in &remaining {
                let step = &steps[index];
                let reused = step
                    .reuse_key
                    .as_ref()
                    .is_some_and(|key| scored.contains(key));
                let fresh = !resident_after(&steps[previous], step);
                let seconds = if reused {
                    0.0
                } else {
                    cost(index, if expected > 0.0 { expected } else { 1.0 }, fresh)?
                };
                let rejected = 1.0 - step.selectivity;
                let rank = if rejected > 0.0 {
                    seconds / rejected
                } else if seconds == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                };
                let candidate = (rank, index, if expected > 0.0 { seconds } else { 0.0 });
                if choice
                    .as_ref()
                    .is_none_or(|found| (candidate.0, candidate.1) < (found.0, found.1))
                {
                    choice = Some(candidate);
                }
            }
            let (_, selected, seconds) = choice.expect("a remaining step");
            total += seconds;
            ordered.push(selected);
            previous = selected;
            expected *= steps[selected].selectivity;
            if let Some(key) = &steps[selected].reuse_key {
                scored.push(key.clone());
            }
            remaining.retain(|index| *index != selected);
        }
        if best
            .as_ref()
            .is_none_or(|found| (total, first) < (found.0, found.1))
        {
            best = Some((total, first, ordered));
        }
    }
    Ok(best.expect("a starter").2)
}

// ---------------------------------------------------------------- joins

/// The small physical state needed for guaranteed group reuse.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct KvState {
    pub pending_anchor: Option<String>,
    pub group_open: bool,
    pub used_anchors: BTreeSet<String>,
}

/// One join as the search priced it in its place in the sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinStep {
    pub written_position: usize,
    pub anchor: String,
    /// "filter", "kept", or "none".
    pub resident: &'static str,
    pub resident_documents: i64,
    pub anchor_documents: f64,
    pub tuples: f64,
    pub tokens: f64,
    pub work: Work,
}

impl JoinStep {
    /// One model copy's share when the anchors are spread evenly.
    pub fn copy_work(&self, gpus: i64) -> Work {
        &self.work * (1.0 / parallel_copies(gpus, self.anchor_documents))
    }
}

/// The cheapest join sequence the search found.
#[derive(Debug, Clone)]
pub struct JoinSearch {
    pub seq: Vec<(usize, String)>,
    pub steps: Vec<JoinStep>,
    pub work: Work,
}

/// Candidate anchors for one join, as `anchor_candidates`.
pub fn anchor_candidates(spec: &PricedJoin, honor_forced: bool) -> Vec<String> {
    if spec.semantics != "full" || (honor_forced && !spec.anchor_free) {
        return vec![spec.anchor.clone().unwrap_or_default()];
    }
    spec.aliases.clone()
}

/// Tokens of the longest input under one join orientation.
pub fn join_input_tokens(
    spec: &PricedJoin,
    anchor: &str,
    maximum: &BTreeMap<String, i64>,
    pre: i64,
) -> i64 {
    let partners: Vec<(i64, i64)> = spec
        .aliases
        .iter()
        .filter(|partner| *partner != anchor)
        .map(|partner| (spec.label_tokens[partner], maximum[partner]))
        .collect();
    longest_join_input(
        pre,
        maximum[anchor],
        spec.frame_tokens[anchor],
        spec.tail_tokens,
        &partners,
    )
}

fn anchor_fits(
    spec: &PricedJoin,
    anchor: &str,
    lengths: &BTreeMap<String, AliasStats>,
    pre: i64,
    chunk: i64,
) -> bool {
    if spec.aliases.iter().any(|alias| lengths[alias].count == 0) {
        return true;
    }
    let maximum: BTreeMap<String, i64> = spec
        .aliases
        .iter()
        .map(|alias| (alias.clone(), lengths[alias].maximum))
        .collect();
    join_input_tokens(spec, anchor, &maximum, pre) <= chunk
}

fn feasible_anchors(
    spec: &PricedJoin,
    honor_forced: bool,
    lengths: &BTreeMap<String, AliasStats>,
    pre: i64,
    chunk: i64,
) -> Vec<String> {
    let candidates = anchor_candidates(spec, honor_forced);
    let fits: Vec<String> = candidates
        .iter()
        .filter(|anchor| anchor_fits(spec, anchor, lengths, pre, chunk))
        .cloned()
        .collect();
    if fits.is_empty() { candidates } else { fits }
}

fn residency(
    anchor: &str,
    state: &KvState,
    resident: &BTreeSet<String>,
    same_group: bool,
) -> &'static str {
    if same_group || state.used_anchors.contains(anchor) {
        "kept"
    } else if resident.contains(anchor) {
        "filter"
    } else {
        "none"
    }
}

/// Expected anchors the join reads: those with a candidate partner.
pub fn anchor_documents(spec: &PricedJoin, anchor: &str, expected: &BTreeMap<String, f64>) -> f64 {
    let partners: f64 = spec
        .aliases
        .iter()
        .filter(|alias| *alias != anchor)
        .map(|alias| expected[alias])
        .product();
    surviving_documents(expected[anchor], partners, Some(spec.pair_fraction))
}

/// A prepared join's work at the current survivor counts, as `stage_work`.
///
/// # Errors
///
/// The anchor's length summary uses another window.
#[allow(clippy::too_many_arguments)]
pub fn stage_work(
    spec: &PricedJoin,
    anchor: &str,
    expected: &BTreeMap<String, f64>,
    lengths: &BTreeMap<String, AliasStats>,
    pre: i64,
    resident: bool,
    window: i64,
    set_size: i64,
) -> Result<Work, CompileError> {
    if let Some(cost) = &spec.score {
        return Ok(score_join_work(
            cost,
            &spec.aliases,
            spec.pair_fraction,
            expected,
        ));
    }
    let documents = anchor_documents(spec, anchor, expected);
    let tuples = cross_tuples(spec, expected);
    if set_size != 0 {
        return packed_join_work(
            spec, anchor, documents, tuples, lengths, pre, window, set_size,
        )
        .map_err(CompileError::new);
    }
    join_work(
        spec, anchor, documents, tuples, lengths, pre, resident, window,
    )
    .map_err(CompileError::new)
}

#[allow(clippy::too_many_arguments)]
fn join_step(
    spec: &PricedJoin,
    anchor: &str,
    expected: &BTreeMap<String, f64>,
    lengths: &BTreeMap<String, AliasStats>,
    resident: &BTreeSet<String>,
    pre: i64,
    state: &KvState,
    context: &PlanningContext,
) -> Result<(Work, JoinStep, KvState), CompileError> {
    let model = &context.model;
    let same_group = state.group_open
        && state.pending_anchor.as_deref() == Some(anchor)
        && spec.semantics == "full";
    let set_size = if model.attention_mask == "set" {
        model.set_size
    } else {
        0
    };
    let kind = if model.reuses_kv() {
        residency(anchor, state, resident, same_group)
    } else {
        "none"
    };
    let work = stage_work(
        spec,
        anchor,
        expected,
        lengths,
        pre,
        kind != "none",
        model.sliding_window,
        set_size,
    )?;
    let step = JoinStep {
        written_position: spec.written_position,
        anchor: anchor.to_string(),
        resident: kind,
        resident_documents: if kind != "none" {
            lengths[anchor].count
        } else {
            0
        },
        anchor_documents: anchor_documents(spec, anchor, expected),
        tuples: cross_tuples(spec, expected),
        tokens: work.tokens.total(),
        work: work.clone(),
    };
    let mut used = state.used_anchors.clone();
    used.insert(anchor.to_string());
    let next = KvState {
        pending_anchor: Some(anchor.to_string()),
        group_open: spec.semantics == "full",
        used_anchors: used,
    };
    Ok((work, step, next))
}

/// Cost one sequence of (join, anchor), as `walk`.
///
/// # Errors
///
/// A join cannot be priced.
pub fn walk(
    seq: &[(PricedJoin, String)],
    live: &BTreeMap<String, f64>,
    lengths: &BTreeMap<String, AliasStats>,
    resident: &BTreeSet<String>,
    pre: i64,
    context: &PlanningContext,
) -> Result<(Work, Vec<JoinStep>), CompileError> {
    let mut state = KvState::default();
    let mut total = Work::default();
    let mut records = Vec::new();
    let mut applied: Vec<&PricedJoin> = Vec::new();
    for (spec, anchor) in seq {
        let mut expected = live.clone();
        let mut done = applied.clone();
        done.sort_by_key(|found| found.written_position);
        for found in done {
            thin(&mut expected, found);
        }
        let (work, step, next) = join_step(
            spec, anchor, &expected, lengths, resident, pre, &state, context,
        )?;
        state = next;
        records.push(step);
        total = &total + &work;
        applied.push(spec);
    }
    Ok((total, records))
}

/// A search candidate's ranking: seconds, then the work's leading fields.
type Rank = (f64, f64, f64, f64, f64);

/// The best fixed-order assignment so far: its rank, work, steps, and sequence.
type FixedBest = (Rank, Work, Vec<JoinStep>, Vec<(PricedJoin, String)>);

fn rank(base: &Work, work: &Work, context: &PlanningContext, chunk: i64) -> Rank {
    let seconds = speed_of_light(
        &(base + work),
        &context.model,
        &context.device,
        chunk,
        0.0,
        0,
    )
    .seconds();
    (
        seconds,
        work.tokens.total(),
        work.pairs.total(),
        work.kv_written.total(),
        work.kv_read.total(),
    )
}

fn sum_work<'a>(works: impl IntoIterator<Item = &'a Work>) -> Work {
    let mut total = Work::default();
    for work in works {
        total = &total + work;
    }
    total
}

#[derive(Clone)]
struct Candidate {
    work: Work,
    relation_order: Vec<String>,
    steps: Vec<JoinStep>,
}

fn insert_nondominated(frontier: &mut Vec<Candidate>, candidate: Candidate) {
    if frontier
        .iter()
        .any(|existing| existing.work.dominates(&candidate.work))
    {
        return;
    }
    frontier.retain(|existing| !candidate.work.dominates(&existing.work));
    frontier.push(candidate);
}

type StateKey = (BTreeSet<String>, BTreeSet<usize>, KvState);

/// Search stage order and anchor choice, as `search_joins`; None when
/// the joins have no connected left-deep order.
///
/// # Errors
///
/// A join cannot be priced.
#[allow(clippy::too_many_arguments)]
pub fn search_joins(
    specs: &[PricedJoin],
    expected: &BTreeMap<String, f64>,
    lengths: &BTreeMap<String, AliasStats>,
    resident: &BTreeSet<String>,
    pre: i64,
    chunk: i64,
    context: &PlanningContext,
    base: &Work,
    fixed_order: bool,
) -> Result<Option<JoinSearch>, CompileError> {
    let gpus = context.gpus;
    if specs.is_empty() {
        return Ok(Some(JoinSearch {
            seq: Vec::new(),
            steps: Vec::new(),
            work: Work::default(),
        }));
    }
    let copy_work = |steps: &[JoinStep]| {
        let works: Vec<Work> = steps.iter().map(|step| step.copy_work(gpus)).collect();
        sum_work(&works)
    };
    if fixed_order || specs.len() == 1 {
        let choices: Vec<Vec<String>> = specs
            .iter()
            .map(|spec| feasible_anchors(spec, true, lengths, pre, chunk))
            .collect();
        let mut best: Option<FixedBest> = None;
        let mut assignment = vec![0usize; specs.len()];
        loop {
            let seq: Vec<(PricedJoin, String)> = specs
                .iter()
                .zip(&assignment)
                .zip(&choices)
                .map(|((spec, index), options)| (spec.clone(), options[*index].clone()))
                .collect();
            let (work, records) = walk(&seq, expected, lengths, resident, pre, context)?;
            let key = rank(base, &copy_work(&records), context, chunk);
            if best.as_ref().is_none_or(|found| key < found.0) {
                best = Some((key, work, records, seq));
            }
            // the next assignment, last join fastest, as itertools.product
            let mut position = specs.len();
            loop {
                if position == 0 {
                    let (_, work, steps, seq) = best.expect("an assignment");
                    return Ok(Some(JoinSearch {
                        seq: seq
                            .into_iter()
                            .map(|(spec, anchor)| (spec.written_position, anchor))
                            .collect(),
                        steps,
                        work,
                    }));
                }
                position -= 1;
                assignment[position] += 1;
                if assignment[position] < choices[position].len() {
                    break;
                }
                assignment[position] = 0;
            }
        }
    }
    let mut aliases: Vec<String> = Vec::new();
    for spec in specs {
        for alias in &spec.aliases {
            if !aliases.contains(alias) {
                aliases.push(alias.clone());
            }
        }
    }
    let edges: Vec<BTreeSet<String>> = specs
        .iter()
        .map(|spec| spec.aliases.iter().cloned().collect())
        .collect();
    let mut live_cache: HashMap<BTreeSet<usize>, BTreeMap<String, f64>> = HashMap::new();
    let mut states: Vec<(StateKey, Vec<Candidate>)> = Vec::new();
    let mut index_of: HashMap<StateKey, usize> = HashMap::new();
    for alias in &aliases {
        let relations: BTreeSet<String> = [alias.clone()].into_iter().collect();
        let key = (relations, BTreeSet::new(), KvState::default());
        index_of.insert(key.clone(), states.len());
        states.push((
            key,
            vec![Candidate {
                work: Work::default(),
                relation_order: vec![alias.clone()],
                steps: Vec::new(),
            }],
        ));
    }
    for num_applied in 0..edges.len() {
        let current: Vec<(StateKey, Vec<Candidate>)> = states
            .iter()
            .filter(|(key, _)| key.1.len() == num_applied)
            .cloned()
            .collect();
        for ((relations, applied, state), frontier) in current {
            let available: Vec<usize> = (0..edges.len())
                .filter(|index| !applied.contains(index))
                .filter(|index| {
                    let ends = &edges[*index];
                    ends.is_subset(&relations)
                        || (!ends.is_disjoint(&relations)
                            && ends.difference(&relations).count() == 1)
                })
                .collect();
            for index in available {
                let spec = &specs[index];
                let added: Vec<String> = edges[index].difference(&relations).cloned().collect();
                let next_relations: BTreeSet<String> =
                    relations.union(&edges[index]).cloned().collect();
                let mut next_applied = applied.clone();
                next_applied.insert(index);
                let now = live_cache
                    .entry(applied.clone())
                    .or_insert_with(|| {
                        let mut rows = expected.clone();
                        for done in &applied {
                            thin(&mut rows, &specs[*done]);
                        }
                        rows
                    })
                    .clone();
                for anchor in feasible_anchors(spec, true, lengths, pre, chunk) {
                    let (_, step, next_state) =
                        join_step(spec, &anchor, &now, lengths, resident, pre, &state, context)?;
                    let work = step.copy_work(gpus);
                    let key = (next_relations.clone(), next_applied.clone(), next_state);
                    let position = match index_of.get(&key) {
                        Some(position) => *position,
                        None => {
                            index_of.insert(key.clone(), states.len());
                            states.push((key, Vec::new()));
                            states.len() - 1
                        }
                    };
                    for old in &frontier {
                        let mut relation_order = old.relation_order.clone();
                        relation_order.extend(added.iter().cloned());
                        let mut steps = old.steps.clone();
                        steps.push(step.clone());
                        insert_nondominated(
                            &mut states[position].1,
                            Candidate {
                                work: &old.work + &work,
                                relation_order,
                                steps,
                            },
                        );
                    }
                }
            }
        }
    }
    let all_aliases: BTreeSet<String> = aliases.iter().cloned().collect();
    let finals: Vec<&Candidate> = states
        .iter()
        .filter(|((relations, applied, _), _)| {
            applied.len() == edges.len() && all_aliases.is_subset(relations)
        })
        .flat_map(|(_, frontier)| frontier.iter())
        .collect();
    let mut best: Option<(Rank, &Candidate)> = None;
    for candidate in finals {
        let key = rank(base, &candidate.work, context, chunk);
        let better = match &best {
            None => true,
            Some((found, other)) => match key.partial_cmp(found) {
                Some(std::cmp::Ordering::Less) => true,
                Some(std::cmp::Ordering::Equal) => {
                    let mine = (
                        &candidate.relation_order,
                        candidate
                            .steps
                            .iter()
                            .map(|s| (s.written_position, &s.anchor))
                            .collect::<Vec<_>>(),
                    );
                    let theirs = (
                        &other.relation_order,
                        other
                            .steps
                            .iter()
                            .map(|s| (s.written_position, &s.anchor))
                            .collect::<Vec<_>>(),
                    );
                    mine < theirs
                }
                _ => false,
            },
        };
        if better {
            best = Some((key, candidate));
        }
    }
    let Some((_, best)) = best else {
        return Ok(None);
    };
    let works: Vec<Work> = best.steps.iter().map(|step| step.work.clone()).collect();
    Ok(Some(JoinSearch {
        seq: best
            .steps
            .iter()
            .map(|step| (step.written_position, step.anchor.clone()))
            .collect(),
        steps: best.steps.clone(),
        work: sum_work(&works),
    }))
}

// ---------------------------------------------------------------- operator order

/// Operator order, statistics, and work estimates, as `OperatorOrder`.
#[derive(Debug, Clone)]
pub struct OperatorOrder {
    pub statistics: PlanStatistics,
    /// alias -> its asks' written positions in run order.
    pub filter_orders: Vec<(String, Vec<usize>)>,
    /// alias -> its steps in run order.
    pub step_orders: Vec<(String, Vec<TableStep>)>,
    pub join_sequence: Vec<(PricedJoin, String)>,
    pub join_steps: Vec<JoinStep>,
    pub expected_rows: BTreeMap<String, f64>,
    pub filter_work: Vec<(String, Work)>,
    pub order_rule: String,
    pub order_source: String,
}

pub use crate::expressions::Refusal;

/// The default stage order rule, as `default_order_rule`: (rule, source).
pub fn default_order_rule(found: &Operators) -> (String, String) {
    let missing = found
        .filters
        .iter()
        .any(|(_, predicates)| predicates.iter().any(|p| p.selectivity.is_none()))
        || found
            .column_filters
            .iter()
            .any(|(_, tests)| tests.iter().any(|t| t.selectivity.is_none()))
        || found.joins.iter().any(|join| join.selectivity.is_none());
    if missing {
        return (
            "by_cost".to_string(),
            format!(
                "default: by cost, with selectivity {DEFAULT_SELECTIVITY} for predicates without one"
            ),
        );
    }
    ("by_cost".to_string(), "default: by cost".to_string())
}

/// Each table's asks' work in run order, as `filter_works`.
///
/// # Errors
///
/// A prompt has no question tail.
pub fn filter_works(
    found: &Operators,
    statistics: &PlanStatistics,
    context: &PlanningContext,
    step_orders: &[(String, Vec<TableStep>)],
) -> Result<Vec<(String, Work)>, CompileError> {
    let model = &context.model;
    let mut works = Vec::new();
    for (alias, predicates) in &found.filters {
        let stats = statistics.stats[alias];
        let prefix = Quantity::Scalar(statistics.pre as f64 + stats.mean_document_tokens());
        let mut count = stats.num_documents as f64;
        let mut total = Work::default();
        let mut previous: Option<&TableStep> = None;
        let steps = step_orders
            .iter()
            .find(|(owner, _)| owner == alias)
            .map(|(_, steps)| steps.as_slice())
            .unwrap_or(&[]);
        for step in steps {
            if step.kind == ASK
                && let StepKey::Position(position) = step.key
                && statistics.asks_of(alias).contains(&position)
            {
                let question = question_tokens(
                    statistics.lowered(&asked_call(&predicates[position].expression).prompt),
                    model.canvas_tokens,
                )?;
                let question = Quantity::Scalar(question as f64);
                let resident = previous.is_some_and(|before| resident_after(before, step));
                let operation = if resident {
                    ask(&prefix, &question, model.sliding_window)
                } else {
                    scan(&prefix, &question, model.sliding_window)
                };
                total = &total + &(&operation * count);
            }
            count *= step.selectivity;
            previous = Some(step);
        }
        works.push((alias.clone(), total));
    }
    Ok(works)
}

/// Choose operator order and estimate work, as `choose_operator_order`.
///
/// # Errors
///
/// A cost cannot be computed; a join pair too long for a forward pass
/// is a Refusal instead.
pub fn choose_operator_order(
    found: &Operators,
    context: &PlanningContext,
) -> Result<Result<OperatorOrder, Refusal>, CompileError> {
    let statistics = plan_statistics(found, context)?;
    let (rule, source) = match &context.order {
        Some(order) => (order.clone(), format!("user: order={}", repr_str(order))),
        None => default_order_rule(found),
    };
    let mut step_orders: Vec<(String, Vec<TableStep>)> = Vec::new();
    for table in &found.scans {
        let count = found
            .filters
            .iter()
            .find(|(alias, _)| *alias == table.alias)
            .map(|(_, p)| p.len())
            .unwrap_or(0);
        let positions: Vec<usize> = (0..count).collect();
        let steps = table_steps(
            found,
            &table.alias,
            &positions,
            &StepPricing::Plan(&statistics),
            context,
        )?;
        if steps.is_empty() {
            continue;
        }
        let ordered = order_steps(
            &steps,
            &rule,
            statistics.stats[&table.alias].num_documents as f64,
            context,
        )?;
        step_orders.push((
            table.alias.clone(),
            ordered
                .into_iter()
                .map(|index| steps[index].clone())
                .collect(),
        ));
    }
    let filter_orders: Vec<(String, Vec<usize>)> = step_orders
        .iter()
        .map(|(alias, steps)| {
            (
                alias.clone(),
                steps
                    .iter()
                    .filter(|step| step.kind == ASK)
                    .filter_map(|step| match step.key {
                        StepKey::Position(position) => Some(position),
                        StepKey::Name(_) => None,
                    })
                    .collect(),
            )
        })
        .collect();
    let live = expected_rows_after_filters(found, &statistics);
    let works = filter_works(found, &statistics, context, &step_orders)?;
    let resident = filtered_before_joins(found);
    let mut sequence = Vec::new();
    if !found.joins.is_empty() {
        let mut base = Work::default();
        for (_, work) in &works {
            base = &base + work;
        }
        let base = &base * (1.0 / context.gpus as f64);
        let search = |fixed: bool| {
            search_joins(
                &statistics.specs,
                &live,
                &statistics.lengths,
                &resident,
                statistics.pre,
                statistics.chunk,
                context,
                &base,
                fixed,
            )
        };
        let result = match search(rule == "as_written")? {
            Some(found) => found,
            None => search(true)?.expect("a fixed order"),
        };
        sequence = result.seq;
    }
    let seq: Vec<(PricedJoin, String)> = sequence
        .iter()
        .map(|(position, anchor)| (statistics.specs[*position].clone(), anchor.clone()))
        .collect();
    let (_, join_steps) = walk(
        &seq,
        &live,
        &statistics.lengths,
        &resident,
        statistics.pre,
        context,
    )?;
    for spec in &statistics.specs {
        if spec.score.is_some() {
            continue;
        }
        let anchor = &sequence
            .iter()
            .rev()
            .find(|(position, _)| *position == spec.written_position)
            .expect("an anchor")
            .1;
        let maximum: BTreeMap<String, i64> = spec
            .aliases
            .iter()
            .map(|alias| (alias.clone(), statistics.stats[alias].max_document_tokens))
            .collect();
        let need = join_input_tokens(spec, anchor, &maximum, statistics.pre);
        if need > statistics.chunk {
            return Ok(Err(Refusal {
                reasons: vec![format!(
                    "one join pair anchored on {} needs {need} tokens, but the forward pass budget is {} tokens",
                    repr_str(anchor),
                    statistics.chunk
                )],
                constraint: "suffix_over_chunk".to_string(),
                needed: need as f64,
                available: statistics.chunk as f64,
                unit: "tokens".to_string(),
            }));
        }
    }
    Ok(Ok(OperatorOrder {
        statistics,
        filter_orders,
        step_orders,
        join_sequence: seq,
        join_steps,
        expected_rows: live,
        filter_work: works,
        order_rule: rule,
        order_source: source,
    }))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use quail_cost::specs::{read_devices, read_models};

    use super::*;

    fn spec(position: usize, aliases: [&str; 2], selectivity: f64) -> PricedJoin {
        PricedJoin {
            written_position: position,
            aliases: aliases.iter().map(|alias| alias.to_string()).collect(),
            anchor: Some(aliases[0].to_string()),
            anchor_free: true,
            semantics: "full".to_string(),
            selectivity: Some(selectivity),
            on: Vec::new(),
            pair_fraction: 1.0,
            score: None,
            frame_tokens: aliases.iter().map(|alias| (alias.to_string(), 8)).collect(),
            label_tokens: aliases.iter().map(|alias| (alias.to_string(), 4)).collect(),
            tail_tokens: 2,
        }
    }

    #[test]
    fn the_join_search_finds_the_cheapest_enumerated_sequence() {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../quail/specs/data");
        let model = read_models(&data.join("models")).unwrap()["qwen3-4b-fp8"].clone();
        let device = read_devices(&data.join("devices")).unwrap()["h100-sxm"].clone();
        let tokenizer = |_: &str| Ok(Vec::new());
        let context = PlanningContext {
            model: model.clone(),
            device: device.clone(),
            gpus: 1,
            canvas_draws: 4,
            document_tokens: BTreeMap::new(),
            tokenizer: &tokenizer,
            decode: None,
            attention: None,
            pair_fractions: BTreeMap::new(),
            scan_fractions: BTreeMap::new(),
            order: None,
            operators: None,
            first_documents: BTreeMap::new(),
            page_shared: BTreeMap::new(),
            reader_groups: BTreeMap::new(),
            last_tokens: BTreeMap::new(),
            backend: "quail".to_string(),
        };
        let specs = [spec(0, ["a", "b"], 0.01), spec(1, ["b", "c"], 0.1)];
        let lengths: BTreeMap<String, AliasStats> =
            [("a", 800, 20), ("b", 100, 30), ("c", 400, 10)]
                .into_iter()
                .map(|(alias, tokens, count)| {
                    (
                        alias.to_string(),
                        AliasStats::summarize(&vec![tokens; count], 0),
                    )
                })
                .collect();
        let resident: BTreeSet<String> = ["a", "c"].iter().map(|alias| alias.to_string()).collect();
        let expected: BTreeMap<String, f64> = [("a", 10.0), ("b", 6.0), ("c", 8.0)]
            .into_iter()
            .map(|(alias, rows)| (alias.to_string(), rows))
            .collect();
        let seconds = |work: &Work| speed_of_light(work, &model, &device, 8192, 0.0, 0).seconds();
        let found = search_joins(
            &specs,
            &expected,
            &lengths,
            &resident,
            5,
            8192,
            &context,
            &Work::default(),
            false,
        )
        .unwrap()
        .unwrap();
        let mut cheapest = f64::INFINITY;
        for order in [[0, 1], [1, 0]] {
            for first in &specs[order[0]].aliases {
                for second in &specs[order[1]].aliases {
                    let sequence = [
                        (specs[order[0]].clone(), first.clone()),
                        (specs[order[1]].clone(), second.clone()),
                    ];
                    let (work, steps) =
                        walk(&sequence, &expected, &lengths, &resident, 5, &context).unwrap();
                    // a table's filter KV is read once, by its first anchor use
                    let anchors: Vec<&String> = steps.iter().map(|step| &step.anchor).collect();
                    if anchors[0] == anchors[1] {
                        assert_ne!(steps[1].resident, "filter");
                    }
                    cheapest = cheapest.min(seconds(&work));
                }
            }
        }
        assert_eq!(seconds(&found.work), cheapest);
    }
}
