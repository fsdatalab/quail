//! Each column operator's priced spec.

use crate::constants;
use crate::constants::{EXTRACT_ANSWER_TOKENS, MAP_ANSWER_TOKENS};
use crate::cost::budgets::chunk_budget;
use crate::cost::extract::{self, ExtractInputs};
use crate::cost::quantity::Quantity;
use crate::cost::retention::coefficients;
use crate::cost::sae_features::SaeFeaturesCost;
use crate::cost::sol::unrounded_seconds;
use crate::cost::statistics::AliasStats;
use crate::cost::work::packed;
use crate::physical::{ExtractSpec, MapSpec, ModelCall, Prompt, SaeFeaturesSpec};
use serde::Serialize;

use crate::context::PlanningContext;
use crate::error::{CompileError, Refusal, compile_error};
use crate::expressions::{argument_alias, column_name, repr_str};
use crate::lowering::read_slot_offsets;
use crate::operators::SemanticColumn;
use crate::prompts::prompt_layout;

/// A column operator's spec.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ColumnSpec {
    Extract(ExtractSpec),
    Map(MapSpec),
    SaeFeatures(SaeFeaturesSpec),
}

/// What `bind_inference` reads off a prompt: aliases, template, arguments.
struct Bound {
    aliases: Vec<String>,
    query_template: String,
    arguments: Vec<(String, String)>,
}

fn bind(prompt: &Prompt) -> Bound {
    let mut aliases: Vec<String> = Vec::new();
    for argument in &prompt.args {
        let alias = argument_alias(argument);
        if !aliases.contains(&alias) {
            aliases.push(alias);
        }
    }
    Bound {
        aliases,
        query_template: prompt.template.clone(),
        arguments: prompt
            .args
            .iter()
            .map(|argument| (argument_alias(argument), column_name(argument, false)))
            .collect(),
    }
}

fn lengths_of(context: &PlanningContext, call: &ModelCall) -> Result<Vec<f64>, CompileError> {
    let alias = argument_alias(&call.prompt.args[0]);
    Ok(context
        .lengths(&alias)?
        .iter()
        .map(|length| *length as f64)
        .collect())
}

/// The spec of one column operator over `expected_rows` documents.
///
/// # Errors
///
/// The model cannot run the call, or the tokenizer fails.
pub fn column_spec(
    context: &PlanningContext,
    column: &SemanticColumn,
    expected_rows: f64,
    resident: bool,
) -> Result<ColumnSpec, CompileError> {
    match column.call.kind.as_str() {
        "extract" => {
            extract_spec(context, column, expected_rows, resident).map(ColumnSpec::Extract)
        }
        "map" => map_spec(context, column, expected_rows, resident).map(ColumnSpec::Map),
        "features" => {
            sae_features_spec(context, column, expected_rows, resident).map(ColumnSpec::SaeFeatures)
        }
        _ => compile_error("the node is not a column operator Rust plans yet"),
    }
}

fn extract_spec(
    context: &PlanningContext,
    column: &SemanticColumn,
    expected_rows: f64,
    resident: bool,
) -> Result<ExtractSpec, CompileError> {
    let call = &column.call;
    let prompt = context.lowered(&call.prompt)?;
    let bound = bind(&call.prompt);
    let (model, device) = (&context.model, &context.device);
    let chunk = chunk_budget(model, device);
    let mut spec = ExtractSpec {
        name: column.name.clone(),
        aliases: bound.aliases,
        query_template: bound.query_template,
        arguments: bound.arguments,
        expected_rows,
        estimated_seconds: 0.0,
        estimated_work: Vec::new(),
        prompt_token_parts: vec![
            prompt.preamble_token_ids.clone(),
            prompt.tail_token_ids.clone(),
        ],
        share_prefixes: false,
        question: call.question.clone(),
        max_answer_tokens: constants::EXTRACT_MAX_ANSWER_TOKENS,
        draft_tokens: constants::EXTRACT_DRAFT_TOKENS,
        draft_policy: constants::EXTRACT_DRAFT_POLICY.to_string(),
        expected_answer_tokens: EXTRACT_ANSWER_TOKENS,
        round_cost_per_token: 0.0,
        top_tokens: constants::EXTRACT_TOP_TOKENS,
        first_window_chars: constants::FIRST_WINDOW_CHARS,
        window_chars: constants::WINDOW_CHARS,
        state_tail_tokens: 0,
        question_header_token_ids: Vec::new(),
        read_slots: Vec::new(),
    };
    if model.attention_mask == "set" {
        // a set-attention model reads the document, then points at it
        let alias = argument_alias(&call.prompt.args[0]);
        let summary = AliasStats::summarize(context.lengths(&alias)?, model.sliding_window);
        let layout = prompt_layout(&model.prompt_layout)?;
        let (state_tail, question_header) =
            crate::lowering::set_question_parts(&prompt, &model.prompt_layout, context.tokenizer)?;
        let head = prompt.preamble_token_ids.len() as i64;
        let tail = prompt.tail_token_ids.len() as i64;
        let work = packed(
            &Quantity::Scalar(head as f64 + summary.mean() + state_tail as f64),
            &[(tail - state_tail) as f64],
            model.sliding_window,
        );
        let work = &work * expected_rows;
        spec.estimated_seconds = unrounded_seconds(&work, model, device, chunk);
        spec.estimated_work = work.values();
        spec.max_answer_tokens = model.read_span_tokens;
        spec.state_tail_tokens = state_tail;
        spec.question_header_token_ids = question_header;
        let names: Vec<String> = layout
            .read_slots
            .iter()
            .map(|name| name.to_string())
            .collect();
        spec.read_slots = read_slot_offsets(
            prompt.prompt.tail_segments.last().expect("a read segment"),
            &names,
            context.tokenizer,
        )?;
        return Ok(spec);
    }
    // one document token's KV read per round, in tokens of compute
    let token_seconds = coefficients(model, device, false).linear_seconds;
    spec.round_cost_per_token = model.kappa() / device.hbm_bw / token_seconds;
    let inputs = ExtractInputs {
        head_tokens: prompt.preamble_token_ids.len() as i64,
        tail_tokens: prompt.tail_token_ids.len() as i64,
        window: 0,
        resident,
        max_answer_tokens: spec.max_answer_tokens,
        draft_tokens: spec.draft_tokens,
        draft_policy: spec.draft_policy.clone(),
        round_cost_per_token: spec.round_cost_per_token,
        answer_tokens: EXTRACT_ANSWER_TOKENS,
        recurrent: false,
    };
    let cost = extract::estimate(
        expected_rows,
        &lengths_of(context, call)?,
        &[],
        &inputs,
        chunk,
        model,
        device,
        context.gpus,
    )
    .map_err(CompileError::new)?;
    spec.estimated_seconds = cost.seconds;
    spec.estimated_work = cost.work.values();
    Ok(spec)
}

/// A map's estimated cost over its documents.
///
/// # Errors
///
/// The planner has no token counts for the map's table.
pub fn map_cost(
    context: &PlanningContext,
    spec: &MapSpec,
    expected_rows: f64,
    resident: bool,
    shared: &[f64],
) -> Result<extract::ExtractCost, CompileError> {
    let lengths: Vec<f64> = context
        .lengths(&spec.aliases[0])?
        .iter()
        .map(|length| *length as f64)
        .collect();
    let inputs = ExtractInputs {
        head_tokens: spec.prompt_token_parts[0].len() as i64,
        tail_tokens: spec.prompt_token_parts[1].len() as i64,
        window: 0,
        resident,
        max_answer_tokens: spec.max_answer_tokens,
        draft_tokens: 0,
        draft_policy: "fixed".to_string(),
        round_cost_per_token: 0.0,
        answer_tokens: spec.expected_answer_tokens,
        recurrent: false,
    };
    let model = &context.model;
    extract::estimate(
        expected_rows,
        &lengths,
        shared,
        &inputs,
        chunk_budget(model, &context.device),
        model,
        &context.device,
        context.gpus,
    )
    .map_err(CompileError::new)
}

fn map_spec(
    context: &PlanningContext,
    column: &SemanticColumn,
    expected_rows: f64,
    resident: bool,
) -> Result<MapSpec, CompileError> {
    let call = &column.call;
    let prompt = context.lowered(&call.prompt)?;
    let bound = bind(&call.prompt);
    let mut spec = MapSpec {
        name: column.name.clone(),
        aliases: bound.aliases,
        query_template: bound.query_template,
        arguments: bound.arguments,
        expected_rows,
        estimated_seconds: 0.0,
        estimated_work: Vec::new(),
        prompt_token_parts: vec![prompt.preamble_token_ids, prompt.tail_token_ids],
        share_prefixes: false,
        max_answer_tokens: call.max_answer_tokens,
        expected_answer_tokens: MAP_ANSWER_TOKENS.min(call.max_answer_tokens),
    };
    let cost = map_cost(context, &spec, expected_rows, resident, &[])?;
    spec.estimated_seconds = cost.seconds;
    spec.estimated_work = cost.work.values();
    Ok(spec)
}

fn sae_features_spec(
    context: &PlanningContext,
    column: &SemanticColumn,
    expected_rows: f64,
    resident: bool,
) -> Result<SaeFeaturesSpec, CompileError> {
    let model = &context.model;
    if model.sae.is_none() {
        return Err(CompileError::refused(Refusal::new(
            format!(
                "AI.SAE_FEATURES needs a model loaded with an SAE; {} has none",
                repr_str(&model.name)
            ),
            "unsupported_sae_features_query",
            1.0,
            0.0,
            "queries",
        )));
    }
    let call = &column.call;
    let alias = column.alias();
    let max_pooling = context.operators.as_ref().is_some_and(|found| {
        found
            .sae_features
            .iter()
            .any(|other| other.alias() == alias && other.call.pooling == "max")
    });
    let prompt = context.lowered(&call.prompt)?;
    let summary = AliasStats::summarize(context.lengths(&alias)?, model.sliding_window);
    let cost = SaeFeaturesCost {
        token_lengths: vec![
            prompt.preamble_token_ids.len() as i64,
            prompt.tail_token_ids.len() as i64,
        ],
        summary,
        pooling: call.pooling.clone(),
        model: model.clone(),
        device: context.device.clone(),
        chunk_tokens: chunk_budget(model, &context.device),
        gpus: context.gpus,
        max_pooling,
    };
    let (work, seconds) = cost.estimate(expected_rows, resident);
    let bound = bind(&call.prompt);
    Ok(SaeFeaturesSpec {
        name: column.name.clone(),
        aliases: bound.aliases,
        query_template: bound.query_template,
        arguments: bound.arguments,
        expected_rows,
        estimated_seconds: seconds,
        estimated_work: work.values(),
        prompt_token_parts: vec![prompt.preamble_token_ids, prompt.tail_token_ids],
        share_prefixes: false,
        pooling: call.pooling.clone(),
    })
}
