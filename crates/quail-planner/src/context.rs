//! What the planner reads while it plans one query, as `PlanningContext`.

use std::collections::BTreeMap;

use quail_cost::specs::{DeviceSpec, ModelSpec};

use crate::expressions::CompileError;
use crate::logical::LogicalPlan;
use crate::lowering::{Decoder, LoweredPrompt, PromptModel, Tokenizer, lower_prompt};

/// The model, device, table statistics, and tokenizer of one query's planning.
pub struct PlanningContext<'a> {
    pub model: ModelSpec,
    pub device: DeviceSpec,
    pub gpus: i64,
    /// The most diffusion draws per individual-document classification or score.
    pub canvas_draws: i64,
    /// alias -> each document's token count.
    pub document_tokens: BTreeMap<String, Vec<i64>>,
    pub tokenizer: &'a Tokenizer<'a>,
    /// The tokenizer's decode of one token, if it has one.
    pub decode: Option<&'a Decoder<'a>>,
    /// An attention path forced for every operator, "tree" or "unified".
    pub attention: Option<String>,
    /// join written position -> the fraction of the cross product its
    /// equality conditions keep.
    pub pair_fractions: BTreeMap<usize, f64>,
    /// alias -> the fraction of its documents the regular predicates keep.
    pub scan_fractions: BTreeMap<String, f64>,
    /// The stage order rule the query asked for, if any.
    pub order: Option<String>,
    /// The logical plan being planned, if any.
    pub logical_plan: Option<LogicalPlan>,
}

impl PlanningContext<'_> {
    /// The fields of the model that prompt lowering reads.
    pub fn prompt_model(&self) -> PromptModel {
        PromptModel {
            role: self.model.role.clone(),
            answer_spellings: self.model.answer_spellings.clone(),
            turn: (
                self.model.turn_prefix.clone(),
                self.model.turn_suffix.clone(),
            ),
            prompt_layout: self.model.prompt_layout.clone(),
        }
    }

    /// The prompt lowered with the context's tokenizer.
    ///
    /// # Errors
    ///
    /// The tokenizer fails.
    pub fn lowered(&self, prompt: &quail_physical::Prompt) -> Result<LoweredPrompt, CompileError> {
        lower_prompt(prompt, self.tokenizer)
    }

    /// Each document's token count for one table alias.
    ///
    /// # Errors
    ///
    /// The planner has no token counts for the alias.
    pub fn lengths(&self, alias: &str) -> Result<&[i64], CompileError> {
        self.document_tokens
            .get(alias)
            .map(Vec::as_slice)
            .ok_or_else(|| CompileError(format!("no document token counts for {alias:?}")))
    }
}
