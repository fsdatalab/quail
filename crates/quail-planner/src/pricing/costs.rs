use std::collections::BTreeMap;

use crate::budgets::parallel_copies;
use crate::cost::joins::PricedJoin;
use crate::cost::score::ScoreCost;
use crate::cost::work::Work;
use crate::cost::work::filter_work;
use crate::physical::{Equality, FilterPredicate, Prompt};

use crate::context::PlanningContext;
use crate::error::{CompileError, compile_error};
use crate::expressions::asked_call;
use crate::expressions::{argument_alias, column_name};
use crate::operators::{SemanticColumn, SemanticJoin};
use crate::pricing::classify::classify_work;
use crate::pricing::columns::{ColumnSpec, column_spec};
use crate::pricing::scores::{prompt_aliases, score_statistics};
use crate::prompts::lowering::LoweredPrompt;

#[derive(Debug, Clone, PartialEq)]
pub enum StepCost {
    PerDocument(Box<Work>),
    Score(Box<ScoreCost>),
    Column {
        column: Box<SemanticColumn>,
        resident: bool,
    },
}

impl StepCost {
    pub fn work(&self, context: &PlanningContext, rows: f64) -> Result<Option<Work>, CompileError> {
        let work = match self {
            StepCost::PerDocument(work) => return Ok(Some(&**work * rows)),
            StepCost::Score(cost) => cost.estimate(rows, None).0,
            StepCost::Column { column, resident } => {
                match column_work(context, column, *resident, rows)? {
                    Some(work) => work,
                    None => return Ok(None),
                }
            }
        };
        Ok(Some(&work * (1.0 / parallel_copies(context.gpus, rows))))
    }
}

fn column_work(
    context: &PlanningContext,
    column: &SemanticColumn,
    resident: bool,
    rows: f64,
) -> Result<Option<Work>, CompileError> {
    let rows = rows * column.asked_fraction();
    if column.call.kind == "label" {
        return classify_work(
            context,
            &column.call,
            &column.name,
            &column.alias(),
            resident,
            rows,
        );
    }
    let values = match column_spec(context, column, rows, resident)? {
        ColumnSpec::Extract(spec) => spec.estimated_work,
        ColumnSpec::Map(spec) => spec.estimated_work,
        ColumnSpec::SaeFeatures(spec) => spec.estimated_work,
    };
    Ok(Some(Work::from_values(&values)))
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredicateCost {
    pub selectivity: f64,
    pub first: StepCost,
    pub later: StepCost,
    pub reuse_key: Option<Prompt>,
}

pub fn question_tokens(lowered: &LoweredPrompt, canvas: i64) -> Result<i64, CompileError> {
    if lowered.prompt.kind == "score" {
        return compile_error("a score prompt has no question tail");
    }
    Ok(lowered.tail_token_ids.len() as i64 + canvas)
}

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
            let asked = crate::operators::predicate_asked_fraction(predicate);
            let cost = |fresh: bool| {
                StepCost::PerDocument(Box::new(
                    &filter_work(question, mean_tokens, model, fresh) * asked,
                ))
            };
            (cost(true), cost(false), None)
        };
        prepared.push(PredicateCost {
            selectivity: crate::pricing::statistics::row_selectivity(predicate),
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

pub fn join_specs(
    joins: &[SemanticJoin],
    pair_fractions: &BTreeMap<usize, f64>,
    canvas: i64,
    context: &PlanningContext,
    chunk: i64,
) -> Result<Vec<PricedJoin>, CompileError> {
    let mut out = Vec::new();
    for (position, join) in joins.iter().enumerate() {
        let fraction = pair_fractions.get(&position).copied().unwrap_or(1.0);
        let on: Vec<_> = join.on.iter().map(priced_key).collect();
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
