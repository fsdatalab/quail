//! What filter ordering and join search price: each AI.IF predicate's
//! cost and each join's token counts.

use std::collections::BTreeMap;

use crate::cost::budgets::parallel_copies;
use crate::cost::filters::filter_work;
use crate::cost::joins::PricedJoin;
use crate::cost::score::ScoreCost;
use crate::cost::work::Work;
use crate::physical::{Equality, FilterPredicate, Prompt};

use crate::context::PlanningContext;
use crate::error::{CompileError, compile_error};
use crate::expressions::asked_call;
use crate::expressions::{argument_alias, column_name};
use crate::lowering::LoweredPrompt;
use crate::operators::{SemanticColumn, SemanticJoin};
use crate::scores::{prompt_aliases, score_statistics};

/// A step's work as a function of the documents it reads.
#[derive(Debug, Clone, PartialEq)]
pub enum StepCost {
    /// Work per document.
    PerDocument(Box<Work>),
    /// A score's own estimate at each count.
    Score(Box<ScoreCost>),
    /// A column operator's planner prices it.
    Column {
        column: Box<SemanticColumn>,
        resident: bool,
    },
}

impl StepCost {
    /// One model copy's work over `rows` documents, or None when the
    /// model cannot run the step.
    ///
    /// # Errors
    ///
    /// The tokenizer fails, or the model cannot run a column operator.
    pub fn work(&self, context: &PlanningContext, rows: f64) -> Result<Option<Work>, CompileError> {
        let work = match self {
            StepCost::PerDocument(work) => return Ok(Some(&**work * rows)),
            StepCost::Score(cost) => cost.estimate(rows, None).0,
            StepCost::Column { column, resident } => {
                match crate::search::column_work(context, column, *resident, rows)? {
                    Some(work) => work,
                    None => return Ok(None),
                }
            }
        };
        Ok(Some(&work * (1.0 / parallel_copies(context.gpus, rows))))
    }
}

/// One predicate's costs, and the score result a later predicate may reuse.
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateCost {
    pub selectivity: f64,
    /// Its work when the predicate reads its documents first.
    pub first: StepCost,
    /// Its work when an earlier predicate already read them.
    pub later: StepCost,
    /// The score prompt whose result the predicate reuses, or None.
    pub reuse_key: Option<Prompt>,
}

/// Token count of a prompt's per-evaluation tail, as `question_tokens`.
///
/// # Errors
///
/// The prompt is a score's, which has no question tail.
pub fn question_tokens(lowered: &LoweredPrompt, canvas: i64) -> Result<i64, CompileError> {
    if lowered.prompt.kind == "score" {
        return compile_error("a score prompt has no question tail");
    }
    Ok(lowered.tail_token_ids.len() as i64 + canvas)
}

/// Each predicate's costs for ordering one table's filters.
///
/// # Errors
///
/// The tokenizer fails, or the arena does not fit the device.
pub fn prepare_filter_costs(
    predicates: &[FilterPredicate],
    mean_tokens: f64,
    chunk_tokens: i64,
    context: &PlanningContext,
) -> Result<Vec<PredicateCost>, CompileError> {
    let model = &context.model;
    let mut prepared = Vec::new();
    for predicate in predicates {
        let call = asked_call(&predicate.expression);
        let (first, later, reuse_key) = if call.kind == "score" {
            let (_, cost) =
                score_statistics(&call.prompt, context, chunk_tokens, Some(mean_tokens))?;
            let curve = StepCost::Score(Box::new(cost));
            (curve.clone(), curve, Some((*call.prompt).clone()))
        } else {
            let question = question_tokens(&context.lowered(&call.prompt)?, model.canvas_tokens)?;
            // a guarded filter asks the model about its guard's rows alone
            let asked = crate::statistics::predicate_asked_fraction(predicate);
            let cost = |fresh: bool| {
                StepCost::PerDocument(Box::new(
                    &filter_work(question, mean_tokens, model, fresh) * asked,
                ))
            };
            (cost(true), cost(false), None)
        };
        prepared.push(PredicateCost {
            selectivity: crate::statistics::row_selectivity(predicate),
            first,
            later,
            reuse_key,
        });
    }
    Ok(prepared)
}

fn priced_key(condition: &Equality) -> (String, String, String, String) {
    (
        argument_alias(&condition.left),
        column_name(&condition.left, false),
        argument_alias(&condition.right),
        column_name(&condition.right, false),
    )
}

/// The joins as the cost model prices them, in written order.
///
/// `pair_fractions` maps a written position to the fraction of the
/// cross product its equality conditions keep; `canvas` is the rows a
/// diffusion model appends to every pair.
///
/// # Errors
///
/// The tokenizer fails, or the arena does not fit the device.
pub fn join_specs(
    joins: &[SemanticJoin],
    pair_fractions: &BTreeMap<usize, f64>,
    canvas: i64,
    context: &PlanningContext,
    chunk: i64,
) -> Result<Vec<PricedJoin>, CompileError> {
    let mut out = Vec::new();
    for (position, join) in joins.iter().enumerate() {
        let conditions = &join.on;
        let fraction = if conditions.is_empty() {
            1.0
        } else {
            pair_fractions.get(&position).copied().unwrap_or(1.0)
        };
        let on: Vec<_> = conditions.iter().map(priced_key).collect();
        let call = asked_call(&join.predicate);
        if call.kind == "score" {
            let (parts, cost) = score_statistics(&call.prompt, context, chunk, None)?;
            let aliases = prompt_aliases(&call.prompt);
            let (left, right) = (aliases[0].clone(), aliases[1].clone());
            let (head, middle, tail) = (
                parts[0].len() as i64,
                parts[1].len() as i64,
                parts[2].len() as i64,
            );
            out.push(PricedJoin {
                written_position: position,
                aliases: vec![left.clone(), right.clone()],
                anchor: Some(left.clone()),
                anchor_free: false,
                semantics: join.semantics.clone(),
                selectivity: join.selectivity,
                frame_tokens: BTreeMap::from([(left.clone(), head + middle), (right.clone(), 0)]),
                label_tokens: BTreeMap::from([(left, 0), (right, 0)]),
                tail_tokens: tail + canvas,
                on,
                pair_fraction: fraction,
                score: Some(cost),
            });
            continue;
        }
        let lowered = context.lowered(&call.prompt)?;
        let mut frame_tokens = BTreeMap::new();
        let mut label_tokens = BTreeMap::new();
        for (alias, label, frame) in &lowered.label_token_ids {
            frame_tokens.insert(alias.clone(), frame.len() as i64);
            label_tokens.insert(alias.clone(), label.len() as i64);
        }
        out.push(PricedJoin {
            written_position: position,
            aliases: call.prompt.args.iter().map(argument_alias).collect(),
            anchor: join.anchor.clone(),
            anchor_free: join.anchor.is_none() && join.semantics == "full",
            semantics: join.semantics.clone(),
            selectivity: join.selectivity,
            frame_tokens,
            label_tokens,
            tail_tokens: question_tokens(&lowered, canvas)?,
            on,
            pair_fraction: fraction,
            score: None,
        });
    }
    Ok(out)
}
