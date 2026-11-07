//! Execution constraints checked before a graph is built, as
//! `quail/planner/validation.py` checks them.

use quail_cost::budgets::{
    POOL_FRACTION, arena_bytes, chunk_budget, chunk_memory_bytes, minimum_weight_gpus,
    sliding_kv_fits, transient_sliding_pages,
};
use quail_cost::specs::{DeviceSpec, ModelSpec};
use quail_physical::{Asked, Prompt, ValueExpression};

use crate::context::PlanningContext;
use crate::expressions::asked_call;
use crate::expressions::{CompileError, Refusal, aliased_call, call_aliases, repr_str};
use crate::operators::Operators;
use crate::prompts::prompt_layout;
use crate::scores::prompt_aliases;
use crate::search::label_calls;

fn refusal(reason: impl Into<String>, constraint: &str) -> Refusal {
    Refusal::new(reason, constraint, 1.0, 0.0, "queries")
}

/// Whether a predicate compares an `AI.SCORE` value.
pub fn is_score(asked: &Asked) -> bool {
    asked_call(asked).kind == "score"
}

/// Whether a plan computes an `AI.SCORE` value anywhere, as `has_score`.
pub fn has_score(found: &Operators) -> bool {
    found
        .filters
        .iter()
        .any(|(_, predicates)| predicates.iter().any(|p| is_score(&p.expression)))
        || found.joins.iter().any(|join| is_score(&join.predicate))
        || found
            .projections
            .iter()
            .any(|expression| aliased_call(expression, &["score"]).is_some())
}

/// The `AI.SCORE` columns SELECT returns, as `score_projections`.
pub fn score_projections(found: &Operators) -> Vec<(String, quail_physical::ModelCall)> {
    found
        .projection
        .columns
        .iter()
        .filter_map(|column| match column {
            quail_physical::Projected::Expression(expression @ ValueExpression::Alias(alias)) => {
                aliased_call(expression, &["score"]).map(|call| (alias.name.clone(), call.clone()))
            }
            _ => None,
        })
        .collect()
}

/// (alias, call kind) of each one-table column operator above a join,
/// as `columns_above_joins`.
pub fn columns_above_joins(found: &Operators) -> Vec<(String, String)> {
    let mut above: Vec<(String, String)> = Vec::new();
    for column in found.columns().filter(|column| column.above_joins) {
        let aliases = call_aliases(&column.call);
        if aliases.len() != 1 {
            continue;
        }
        let entry = (aliases[0].clone(), column.call.kind.clone());
        if !above.contains(&entry) {
            above.push(entry);
        }
    }
    above
}

fn sae_features_refusal(found: &Operators, model: &ModelSpec, scored: bool) -> Option<Refusal> {
    let constraint = "unsupported_sae_features_query";
    let features = !found.sae_features.is_empty();
    let others = !found.filters.is_empty()
        || !found.joins.is_empty()
        || scored
        || !label_calls(found).is_empty()
        || !found.extracts.is_empty();
    if features && model.sae.is_none() {
        return Some(refusal(
            format!(
                "AI.SAE_FEATURES needs a model loaded with an SAE; {} has none",
                repr_str(&model.name)
            ),
            constraint,
        ));
    }
    if features && others {
        return Some(refusal(
            "AI.SAE_FEATURES does not mix with other AI operators in one session: the model \
             stops at the SAE's layer, so its KV cannot serve a full-depth operator",
            constraint,
        ));
    }
    if !features && model.sae.is_some() {
        return Some(refusal(
            format!(
                "{} reads the residual stream for AI.SAE_FEATURES only; run other operators on \
                 a full-depth model",
                repr_str(&model.name)
            ),
            constraint,
        ));
    }
    None
}

fn score_refusal(found: &Operators) -> Option<Refusal> {
    let constraint = "unsupported_score_query";
    let projected = score_projections(found);
    let mut prompts: Vec<&Prompt> = Vec::new();
    for (_, predicates) in &found.filters {
        prompts.extend(
            predicates
                .iter()
                .map(|p| asked_call(&p.expression).prompt.as_ref()),
        );
    }
    prompts.extend(
        found
            .joins
            .iter()
            .map(|join| asked_call(&join.predicate).prompt.as_ref()),
    );
    prompts.extend(projected.iter().map(|(_, call)| call.prompt.as_ref()));
    if prompts
        .iter()
        .any(|prompt| prompt_aliases(prompt).len() > 2)
    {
        return Some(refusal(
            "AI.SCORE supports one document or one document pair",
            constraint,
        ));
    }
    let all_scores = found
        .filters
        .iter()
        .all(|(_, predicates)| predicates.iter().all(|p| is_score(&p.expression)))
        && found.joins.iter().all(|join| is_score(&join.predicate));
    if !all_scores {
        return Some(refusal(
            "AI.SCORE cannot be mixed with generative AI predicates",
            constraint,
        ));
    }
    if found.joins.iter().any(|join| join.semantics != "full") {
        return Some(refusal("AI.SCORE supports full joins only", constraint));
    }
    if label_calls(found)
        .iter()
        .any(|(call, _)| call_aliases(call).len() != 1)
    {
        return Some(refusal(
            "AI.CLASSIFY over a document pair needs its AI join pipeline on one GPU; AI.SCORE \
             plans cannot execute that classification",
            "joined_classify_pipeline",
        ));
    }
    if columns_above_joins(found)
        .iter()
        .any(|(_, kind)| kind == "label")
        && !found.joins.is_empty()
    {
        return Some(refusal(
            "classification after a score join is unsupported",
            constraint,
        ));
    }
    let mut pairs: Vec<&Prompt> = Vec::new();
    for prompt in found
        .joins
        .iter()
        .map(|join| asked_call(&join.predicate).prompt.as_ref())
        .chain(
            projected
                .iter()
                .filter(|(_, call)| call_aliases(call).len() == 2)
                .map(|(_, call)| call.prompt.as_ref()),
        )
    {
        if !pairs.contains(&prompt) {
            pairs.push(prompt);
        }
    }
    if pairs.len() > 1 {
        return Some(refusal(
            "AI.SCORE currently supports one document pair expression",
            constraint,
        ));
    }
    if found.joins.iter().any(|join| {
        join.anchor
            .as_ref()
            .is_some_and(|anchor| *anchor != prompt_aliases(&asked_call(&join.predicate).prompt)[0])
    }) {
        return Some(refusal(
            "AI.SCORE anchors on its first prompt document",
            constraint,
        ));
    }
    None
}

fn extraction_refusal(model: &ModelSpec, function: &str) -> Result<Option<Refusal>, CompileError> {
    let constraint = format!(
        "unsupported_{}_query",
        function.trim_start_matches("AI.").to_lowercase()
    );
    let reason = if model.role == "reranker" || model.role == "decision" || model.canvas_tokens != 0
    {
        format!(
            "{function} decodes the model's next tokens, which {} does not give",
            repr_str(&model.name)
        )
    } else if prompt_layout(&model.prompt_layout)?.read_end.is_some() {
        // the model answers by scoring options and document positions
        if function == "AI.EXTRACT" {
            return Ok(None);
        }
        format!(
            "{function} writes text, and {} answers only by scoring options",
            repr_str(&model.name)
        )
    } else {
        return Ok(None);
    };
    Ok(Some(refusal(reason, &constraint)))
}

fn attention_device_refusal(model: &ModelSpec, device: &DeviceSpec) -> Option<Refusal> {
    if device.name != "l40s" || !["diffusion_gemma", "qwen3_5"].contains(&model.arch.as_str()) {
        return None;
    }
    Some(Refusal::new(
        format!(
            "{} does not run on {}: FlashAttention 4 does not run on this GPU",
            model.name, device.name
        ),
        "flash_attention_4_unsupported",
        4.0,
        2.0,
        "version",
    ))
}

/// A refusal for unsupported inference or hardware, as `validate_query`.
///
/// # Errors
///
/// The model names an unknown prompt layout.
pub fn validate_query(
    found: &Operators,
    context: &PlanningContext,
) -> Result<Option<Refusal>, CompileError> {
    let (model, device) = (&context.model, &context.device);
    let scored = has_score(found);
    if model.attention_mask == "set" && found.joins.iter().any(|join| is_score(&join.predicate)) {
        return Ok(Some(refusal(
            format!(
                "{} answers a join's partners as questions, but Quail does not score them yet, \
                 so AI.SCORE reads one table",
                repr_str(&model.name)
            ),
            "unsupported_set_attention_query",
        )));
    }
    if (!found.sae_features.is_empty() || model.sae.is_some())
        && let Some(refused) = sae_features_refusal(found, model, scored)
    {
        return Ok(Some(refused));
    }
    if scored && let Some(refused) = score_refusal(found) {
        return Ok(Some(refused));
    }
    let labels = label_calls(found);
    if !labels.is_empty() {
        let constraint = "unsupported_classify_query";
        if labels
            .iter()
            .any(|(call, _)| !matches!(call_aliases(call).len(), 1 | 2))
        {
            return Ok(Some(refusal(
                "AI.CLASSIFY reads one document, or one from each side of a join",
                constraint,
            )));
        }
        if model.canvas_tokens != 0 && model.answer_canvas.is_none() {
            return Ok(Some(refusal(
                format!(
                    "{} names no answer canvas to read labels on",
                    repr_str(&model.name)
                ),
                constraint,
            )));
        }
    }
    for (function, present) in [
        ("AI.EXTRACT", !found.extracts.is_empty()),
        ("AI.MAP", !found.maps.is_empty()),
    ] {
        if present && let Some(refused) = extraction_refusal(model, function)? {
            return Ok(Some(refused));
        }
    }
    if let Some(refused) = attention_device_refusal(model, device) {
        return Ok(Some(refused));
    }
    let weight_gpus = minimum_weight_gpus(model, device);
    if weight_gpus > 1 {
        return Ok(Some(Refusal::new(
            format!(
                "the model needs {weight_gpus} GPUs of memory, but Quail loads one complete \
                 copy per GPU"
            ),
            "weights_need_more_cards",
            weight_gpus as f64,
            1.0,
            "cards",
        )));
    }
    let needed = chunk_memory_bytes(model, 1, device.page_tokens);
    let available = device.mem_bytes * POOL_FRACTION;
    if needed > available {
        return Ok(Some(Refusal::new(
            format!(
                "{} does not fit {}: even a one-token chunk exceeds the memory available for \
                 weights, activations, and checkpoint reservations",
                model.name, device.name
            ),
            "execution_needs_more_memory",
            needed,
            available,
            "bytes",
        )));
    }
    if !sliding_kv_fits(model, device, None) {
        let chunk = chunk_budget(model, device);
        let free = arena_bytes(model, device, chunk);
        let floor = transient_sliding_pages(chunk, model.sliding_window, device.page_tokens);
        let needed = floor as f64 * model.kappa_sliding() * device.page_tokens as f64;
        return Ok(Some(Refusal::new(
            format!(
                "{} does not fit {}: a {chunk}-token chunk's sliding KV exceeds leftover GPU \
                 memory",
                model.name, device.name
            ),
            "sliding_kv_needs_more_memory",
            needed,
            free.max(0.0),
            "bytes",
        )));
    }
    Ok(None)
}
