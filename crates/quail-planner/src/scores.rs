//! AI.SCORE's statistics, cost, and spec.

use std::collections::BTreeMap;

use crate::cost::quantity::python_sum;
use crate::cost::score::ScoreCost;
use crate::cost::statistics::AliasStats;
use crate::cost::work::Work;
use crate::physical::{Prompt, ScoreSpec};

use crate::context::PlanningContext;
use crate::error::CompileError;
use crate::expressions::{argument_alias, column_name};
use crate::lowering::{lower_prompt, score_query_template, score_token_parts};
use crate::prompts::{bind_prompt, prompt_layout};
use crate::statistics::{arena_capacity, pair_counts};

/// A prompt's table aliases in placeholder order.
pub fn prompt_aliases(prompt: &Prompt) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for argument in &prompt.args {
        let alias = argument_alias(argument);
        if !found.contains(&alias) {
            found.push(alias);
        }
    }
    found
}

/// A score's prompt tokens around the documents, and its cost inputs.
///
/// # Errors
///
/// The tokenizer fails, or the arena does not fit the device.
pub fn score_statistics(
    prompt: &Prompt,
    context: &PlanningContext,
    chunk: i64,
    mean_tokens: Option<f64>,
) -> Result<(Vec<Vec<i64>>, ScoreCost), CompileError> {
    let parts = score_token_parts(prompt, &context.prompt_model(), context.tokenizer)?;
    let aliases = prompt_aliases(prompt);
    let model = &context.model;
    let mut summaries = Vec::new();
    for alias in &aliases {
        summaries.push(AliasStats::summarize(
            context.lengths(alias)?,
            model.sliding_window,
        ));
    }
    let mean_tokens =
        mean_tokens.unwrap_or_else(|| python_sum(summaries.iter().map(AliasStats::mean)));
    let cost = ScoreCost {
        token_lengths: parts.iter().map(|part| part.len() as i64).collect(),
        summaries,
        mean_tokens,
        model: model.clone(),
        device: context.device.clone(),
        chunk_tokens: chunk,
        capacity: arena_capacity(context, chunk)?,
        gpus: context.gpus,
        draws: if model.canvas_tokens != 0 && aliases.len() == 1 {
            context.canvas_draws
        } else {
            1
        },
    };
    Ok((parts, cost))
}

/// A set-attention score's state tail length and later-question header,
/// as `set_score_parts`; (0, []) for every other score.
///
/// # Errors
///
/// The tokenizer fails.
pub fn set_score_parts(
    prompt: &Prompt,
    context: &PlanningContext,
) -> Result<(i64, Vec<i64>), CompileError> {
    let model = &context.model;
    if model.attention_mask != "set" || prompt_aliases(prompt).len() != 1 {
        return Ok((0, Vec::new()));
    }
    let layout = prompt_layout(&model.prompt_layout)?;
    let turn = (model.turn_prefix.clone(), model.turn_suffix.clone());
    let lowered = lower_prompt(
        &bind_prompt(&prompt.template, &prompt.args, &turn, &layout)?,
        context.tokenizer,
    )?;
    crate::lowering::set_question_parts(&lowered, &model.prompt_layout, context.tokenizer)
}

/// A score's spec and its work, as `score_spec`.
///
/// # Errors
///
/// The tokenizer fails, or the arena does not fit the device.
#[allow(clippy::too_many_arguments)]
pub fn score_spec(
    prompt: &Prompt,
    name: &str,
    expected_rows: f64,
    mean_tokens: Option<f64>,
    context: &PlanningContext,
    chunk: i64,
    pair_fraction: f64,
    prefix_groups: Option<f64>,
) -> Result<(ScoreSpec, Work), CompileError> {
    let (parts, cost) = score_statistics(prompt, context, chunk, mean_tokens)?;
    let (work, seconds) = cost.estimate(expected_rows, prefix_groups);
    let (state_tail_tokens, question_header_token_ids) = set_score_parts(prompt, context)?;
    let spec = ScoreSpec {
        name: name.to_string(),
        aliases: prompt_aliases(prompt),
        query_template: score_query_template(prompt)?,
        arguments: prompt
            .args
            .iter()
            .map(|argument| (argument_alias(argument), column_name(argument, false)))
            .collect(),
        expected_rows,
        estimated_seconds: seconds,
        estimated_work: work.values(),
        prompt_token_parts: parts,
        share_prefixes: false,
        pair_fraction,
        draws: cost.draws,
        state_tail_tokens,
        question_header_token_ids,
    };
    Ok((spec, work))
}

/// A score join's work at the given survivor counts, as `ScoreJoinCost.work`.
pub fn score_join_work(
    cost: &ScoreCost,
    aliases: &[String],
    pair_fraction: f64,
    expected_rows: &BTreeMap<String, f64>,
) -> Work {
    let (expected, groups) = pair_counts(expected_rows, aliases, pair_fraction);
    cost.estimate(expected, Some(groups)).0
}
