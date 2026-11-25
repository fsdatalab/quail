use std::collections::BTreeMap;

use crate::budgets::arena_pages;
use crate::cost::statistics::AliasStats;
use crate::specs::{DeviceSpec, ModelSpec};

use crate::error::CompileError;
use crate::operators::Operators;
use crate::prompts::lowering::{Decoder, LoweredPrompt, PromptModel, Tokenizer, lower_prompt};

#[derive(Clone)]
pub struct PlanningContext<'a> {
    pub model: ModelSpec,
    pub device: DeviceSpec,
    pub gpus: i64,
    pub canvas_draws: i64,
    pub document_tokens: BTreeMap<String, Vec<i64>>,
    pub tokenizer: &'a Tokenizer<'a>,
    pub decode: Option<&'a Decoder<'a>>,
    pub attention: Option<String>,
    pub pair_fractions: BTreeMap<usize, f64>,
    pub scan_fractions: BTreeMap<String, f64>,
    pub order: Option<String>,
    pub operators: Option<Operators>,
    pub first_documents: BTreeMap<String, Vec<bool>>,
    pub page_shared: BTreeMap<String, Vec<i64>>,
    pub reader_groups: BTreeMap<String, Vec<(i64, i64, i64)>>,
    pub last_tokens: BTreeMap<String, Vec<i64>>,
    pub backend: String,
}

impl PlanningContext<'_> {
    pub fn arena_capacity(&self, chunk: i64) -> Result<i64, CompileError> {
        let window = self.model.sliding_window;
        let mut longest: Option<f64> = None;
        for lengths in self.document_tokens.values() {
            let mean = AliasStats::summarize(lengths, window).mean();
            if longest.is_none_or(|found| mean > found) {
                longest = Some(mean);
            }
        }
        let (full, _) = arena_pages(
            &self.model,
            &self.device,
            Some(chunk),
            longest,
            self.writes_answers(),
        )
        .map_err(CompileError::new)?;
        Ok(full * self.device.page_tokens)
    }

    pub fn writes_answers(&self) -> bool {
        self.operators
            .as_ref()
            .is_none_or(Operators::writes_answers)
    }

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

    pub fn lowered(&self, prompt: &crate::physical::Prompt) -> Result<LoweredPrompt, CompileError> {
        lower_prompt(prompt, self.tokenizer)
    }

    pub fn key_documents(&self, alias: &str) -> Option<&[bool]> {
        if self.model.canvas_tokens != 0 {
            return None;
        }
        self.first_documents
            .get(alias)
            .map(Vec::as_slice)
            .filter(|keys| !keys.iter().all(|key| *key))
    }

    pub fn lengths(&self, alias: &str) -> Result<&[i64], CompileError> {
        self.document_tokens
            .get(alias)
            .map(Vec::as_slice)
            .ok_or_else(|| CompileError::new(format!("no document token counts for {alias:?}")))
    }
}
