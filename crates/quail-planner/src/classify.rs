//! AI.CLASSIFY's statistics, scoring rule, cost, and spec.

use std::collections::BTreeSet;

use crate::cost::budgets::{chunk_budget, tree_attention_allowed};
use crate::cost::classify::{self, ClassifyCost, ClassifyInputs, suffix_lengths};
use crate::cost::quantity::Quantity;
use crate::cost::sol::unrounded_seconds;
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::statistics::{AliasStats, round_half_even};
use crate::cost::work::{Work, packed};
use crate::physical::{ClassifySpec, ModelCall, Prompt};
use serde::Deserialize;
use serde_json::Value;

use crate::context::PlanningContext;
use crate::error::{CompileError, Refusal};
use crate::expressions::{argument_alias, column_name, repr_str};
use crate::lowering::{
    Decoder, LoweredPrompt, Tokenizer, choice_token_parts, classify_prompt_tokens, lower_prompt,
    option_spellings,
};
use crate::prompts::prompt_layout;
use crate::statistics::arena_capacity;

pub const LETTERS_SCORING: &str = "letters";
pub const TREE_SCORING: &str = "trie_tree";
pub const DECODE_SCORING: &str = "trie_decode";
pub const DECISION_SCORING: &str = "decision_choice";
pub const SET_CHOICE_SCORING: &str = "set_choice";

/// Whether every label's token sequence ends at a trie leaf.
pub fn decodable(labels: &[Vec<i64>]) -> bool {
    let ids: BTreeSet<&[i64]> = labels.iter().map(Vec::as_slice).collect();
    !labels
        .iter()
        .any(|label| (1..label.len()).any(|depth| ids.contains(&label[..depth])))
}

/// Which tokens of each label begin a new word, as `word_starts`.
///
/// # Errors
///
/// The decoder fails.
pub fn word_starts(
    labels: &[Vec<i64>],
    decode: Option<&Decoder>,
) -> Result<Vec<Vec<bool>>, CompileError> {
    let Some(decode) = decode else {
        return Ok(labels.iter().map(|ids| vec![false; ids.len()]).collect());
    };
    let mut flags = Vec::new();
    for ids in labels {
        let pieces: Vec<String> = ids
            .iter()
            .map(|token| decode(std::slice::from_ref(token)))
            .collect::<Result<_, _>>()?;
        flags.push(
            (0..ids.len())
                .map(|depth| {
                    depth > 0
                        && pieces[depth]
                            .chars()
                            .next()
                            .is_some_and(char::is_whitespace)
                        && pieces[depth - 1].chars().any(char::is_alphanumeric)
                })
                .collect(),
        );
    }
    Ok(flags)
}

/// A classification no scoring rule can run, as `ClassifyRefusedError`.
fn refused_with<T>(
    reason: impl Into<String>,
    needed: i64,
    available: i64,
) -> Result<T, CompileError> {
    Err(CompileError::refused(Refusal::new(
        reason,
        "suffix_over_chunk",
        needed as f64,
        available as f64,
        "tokens",
    )))
}

fn refused<T>(reason: impl Into<String>) -> Result<T, CompileError> {
    refused_with(reason, 1, 0)
}

fn over_budget<T>(what: &str, need: i64, budget: i64) -> Result<T, CompileError> {
    refused_with(
        format!(
            "{what} needs {need} tokens with its classification prompt, but the forward pass \
             budget is {budget} tokens"
        ),
        need,
        budget,
    )
}

/// One table's lengths, model, and budgets for its classifications, as
/// `ClassifyStatistics`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClassifyTable {
    pub alias: String,
    pub mean: f64,
    pub longest: i64,
    pub budget: i64,
    pub chunk: i64,
    pub model: ModelSpec,
    pub device: DeviceSpec,
    pub capacity: i64,
    pub draws: i64,
    pub lengths: Vec<f64>,
    pub shared: Vec<f64>,
    pub tree: bool,
    pub gpus: i64,
}

/// The table's statistics, as `classify_statistics`.
///
/// # Errors
///
/// The planner has no token counts for the table, or the arena does
/// not fit the device.
pub fn classify_statistics(
    context: &PlanningContext,
    alias: &str,
    shared: &[f64],
) -> Result<ClassifyTable, CompileError> {
    let model = &context.model;
    let lengths = context.lengths(alias)?;
    let summary = AliasStats::summarize(lengths, model.sliding_window);
    let chunk = chunk_budget(model, &context.device);
    let capacity = arena_capacity(context, chunk)?;
    Ok(ClassifyTable {
        alias: alias.to_string(),
        mean: summary.mean(),
        longest: summary.maximum,
        budget: chunk.min(capacity),
        chunk,
        model: model.clone(),
        device: context.device.clone(),
        capacity,
        draws: if model.answer_canvas.is_some() {
            context.canvas_draws
        } else {
            1
        },
        lengths: lengths.iter().map(|length| *length as f64).collect(),
        shared: shared.to_vec(),
        tree: tree_attention_allowed(model) && context.attention.as_deref() != Some("unified"),
        gpus: context.gpus,
    })
}

fn tokens_value(ids: &[i64]) -> Vec<Value> {
    ids.iter().map(|id| Value::from(*id)).collect()
}

fn labels_of(call: &ModelCall) -> Vec<String> {
    match &call.labels {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// A ClassifySpec with `bind_inference`'s fields and the class's defaults.
fn base_spec(
    prompt: &Prompt,
    name: &str,
    aliases: Vec<String>,
    expected_rows: f64,
    call: &ModelCall,
) -> ClassifySpec {
    ClassifySpec {
        name: name.to_string(),
        aliases,
        query_template: prompt.template.clone(),
        arguments: prompt
            .args
            .iter()
            .map(|argument| (argument_alias(argument), column_name(argument, false)))
            .collect(),
        expected_rows,
        estimated_seconds: 0.0,
        estimated_work: Vec::new(),
        prompt_token_parts: Vec::new(),
        share_prefixes: false,
        pair_fraction: 1.0,
        draws: 1,
        labels: labels_of(call),
        label_token_ids: Vec::new(),
        label_word_starts: Vec::new(),
        scoring: LETTERS_SCORING.to_string(),
        probabilities: false,
        join_layout: None,
        frame_tokens: 0,
        state_tail_tokens: 0,
        question_header_token_ids: Vec::new(),
        separator_tokens: 0,
    }
}

fn label_parts(prompt: &LoweredPrompt, alias: &str) -> (Vec<i64>, Vec<i64>) {
    prompt
        .label_token_ids
        .iter()
        .rev()
        .find(|(found, _, _)| found == alias)
        .map(|(_, label, frame)| (label.clone(), frame.clone()))
        .expect("a label piece per alias")
}

impl ClassifyTable {
    fn inputs(
        &self,
        expected_rows: f64,
        head: i64,
        frame: i64,
        resident: bool,
    ) -> ClassifyInputs<'_> {
        ClassifyInputs {
            head_tokens: head,
            frame_tokens: frame,
            expected_rows,
            lengths: &self.lengths,
            shared: &self.shared,
            chunk: self.chunk,
            model: &self.model,
            device: &self.device,
            resident,
            gpus: self.gpus,
        }
    }

    /// One scoring rule's cost on this table.
    ///
    /// # Errors
    ///
    /// The scoring rule is unknown, or a label has no tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        scoring: &str,
        expected_rows: f64,
        head: i64,
        frame: i64,
        labels: &[Vec<i64>],
        resident: bool,
        starts: &[Vec<bool>],
    ) -> Result<ClassifyCost, CompileError> {
        classify::estimate(
            &self.inputs(expected_rows, head, frame, resident),
            scoring,
            labels,
            self.draws,
            starts,
        )
        .map_err(CompileError::new)
    }

    /// The cost of a spec's chosen scoring rule on this table.
    ///
    /// # Errors
    ///
    /// The spec's scoring rule is unknown.
    pub fn estimate_spec(
        &self,
        spec: &ClassifySpec,
        resident: bool,
    ) -> Result<ClassifyCost, CompileError> {
        let (head, tail) = (
            spec.prompt_token_parts[0].len() as i64,
            spec.prompt_token_parts[1].len() as i64,
        );
        if spec.scoring == DECISION_SCORING {
            return Ok(classify::estimate_suffixes(
                &self.inputs(spec.expected_rows, head, spec.frame_tokens, resident),
                &[tail - spec.frame_tokens],
                0,
                None,
                Some((spec.label_token_ids.len() + 1) as f64),
                0,
            ));
        }
        self.estimate(
            &spec.scoring,
            spec.expected_rows,
            head,
            tail - 1,
            &spec.label_token_ids,
            resident,
            &spec.label_word_starts,
        )
    }

    /// The supported scoring rule with the lowest estimated time.
    ///
    /// # Errors
    ///
    /// No supported scoring rule can run.
    #[allow(clippy::too_many_arguments)]
    pub fn choose(
        &self,
        expected_rows: f64,
        head: i64,
        frame: i64,
        labels: &[Vec<i64>],
        resident: bool,
        lettered: Option<&(i64, i64, Vec<Vec<i64>>)>,
        probabilities: bool,
        starts: &[Vec<bool>],
    ) -> Result<(String, ClassifyCost), CompileError> {
        let mut candidates = Vec::new();
        if lettered.is_some() {
            candidates.push(LETTERS_SCORING);
        }
        if self.tree {
            candidates.push(TREE_SCORING);
        }
        if self.model.canvas_tokens == 0 && !probabilities && decodable(labels) {
            candidates.push(DECODE_SCORING);
        }
        if candidates.is_empty() {
            return refused(
                "no label scoring rule can run: no one-token letter for every label, no tree \
                 attention for the packed trie, and no greedy decode (an answer canvas, or a \
                 label that is another's prefix)",
            );
        }
        let mut best: Option<((f64, f64), &str, ClassifyCost)> = None;
        for scoring in candidates {
            let estimated = match (scoring, lettered) {
                (LETTERS_SCORING, Some((head, frame, ids))) => {
                    self.estimate(scoring, expected_rows, *head, *frame, ids, resident, &[])?
                }
                _ => self.estimate(
                    scoring,
                    expected_rows,
                    head,
                    frame,
                    labels,
                    resident,
                    starts,
                )?,
            };
            let key = (estimated.seconds, estimated.suffix_tokens);
            if best.as_ref().is_none_or(|(found, _, _)| key < *found) {
                best = Some((key, scoring, estimated));
            }
        }
        let (_, scoring, estimated) = best.expect("a candidate");
        Ok((scoring.to_string(), estimated))
    }

    /// The prepared spec with its scoring rule chosen, as `choose_scoring`.
    ///
    /// # Errors
    ///
    /// No scoring rule can run, or the tokenizer fails.
    pub fn choose_scoring(
        &self,
        spec: &ClassifySpec,
        call: &ModelCall,
        resident: bool,
        tokenizer: &Tokenizer,
    ) -> Result<ClassifySpec, CompileError> {
        if spec.scoring == DECISION_SCORING {
            return Ok(spec.clone());
        }
        let (mut head, mut tail) = (
            spec.prompt_token_parts[0].clone(),
            spec.prompt_token_parts[1].clone(),
        );
        let (mut labels, mut starts) =
            (spec.label_token_ids.clone(), spec.label_word_starts.clone());
        let mut lettered = None;
        let mut letter_parts = None;
        if let Some(prompt) = lower_prompt(&call.prompt, tokenizer)?.lettered {
            let parts = classify_prompt_tokens(&prompt, &prompt.prompt.letters, tokenizer)?;
            lettered = Some((
                parts.0.len() as i64,
                parts.1.len() as i64 - 1,
                parts.2.clone(),
            ));
            letter_parts = Some(parts);
        }
        let (scoring, estimated) = self.choose(
            spec.expected_rows,
            head.len() as i64,
            tail.len() as i64 - 1,
            &labels,
            resident,
            lettered.as_ref(),
            spec.probabilities,
            &starts,
        )?;
        if scoring == LETTERS_SCORING {
            let (letter_head, letter_tail, letter_ids) = letter_parts.expect("a lettered prompt");
            (head, tail, labels) = (letter_head, letter_tail, letter_ids);
            starts = Vec::new();
        }
        if scoring == DECODE_SCORING && !decodable(&labels) {
            return refused(format!(
                "the {} rule cannot decode the labels of {}: one label is a proper prefix of \
                 another",
                repr_str(&scoring),
                repr_str(&spec.name)
            ));
        }
        let mut chosen = spec.clone();
        chosen.estimated_seconds = estimated.seconds;
        chosen.estimated_work = estimated.work.values();
        chosen.prompt_token_parts = vec![head, tail];
        chosen.label_token_ids = labels;
        chosen.label_word_starts = starts;
        chosen.draws = if scoring == LETTERS_SCORING {
            self.draws
        } else {
            1
        };
        chosen.scoring = scoring;
        Ok(chosen)
    }

    /// A classification's spec and work before its scoring rule is chosen,
    /// as `ClassifyBuilder.prepare`.
    ///
    /// # Errors
    ///
    /// The prompt does not fit a forward pass, or a callback fails.
    pub fn prepare(
        &self,
        call: &ModelCall,
        name: &str,
        expected_rows: f64,
        tokenizer: &Tokenizer,
        decode: Option<&Decoder>,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        let lowered = lower_prompt(&call.prompt, tokenizer)?;
        if self.model.attention_mask == "set" {
            return self.set_choice_spec(call, name, expected_rows, &lowered, tokenizer);
        }
        if self.model.role == "decision" {
            return self.choice_spec(
                call,
                name,
                expected_rows,
                vec![self.alias.clone()],
                lowered.preamble_token_ids.clone(),
                &lowered,
                tokenizer,
            );
        }
        let (head, tail, labels) = classify_prompt_tokens(&lowered, &labels_of(call), tokenizer)?;
        let mut spec = base_spec(
            &call.prompt,
            name,
            vec![self.alias.clone()],
            expected_rows,
            call,
        );
        spec.prompt_token_parts = vec![head, tail];
        spec.label_word_starts = word_starts(&labels, decode)?;
        spec.label_token_ids = labels;
        spec.scoring = String::new();
        spec.probabilities = call.probabilities;
        Ok((spec, Work::default()))
    }

    /// A classification of joined rows, as `ClassifyBuilder.classify_joined`.
    ///
    /// # Errors
    ///
    /// Probabilities are asked for, the prompt has no lettered form, or
    /// a row does not fit the budget.
    #[allow(clippy::too_many_arguments)]
    pub fn classify_joined(
        &self,
        call: &ModelCall,
        name: &str,
        partner: &str,
        expected_rows: f64,
        partner_tokens: f64,
        anchors: Option<f64>,
        tokenizer: &Tokenizer,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        if call.probabilities {
            return refused(format!(
                "a classification of joined rows returns its label only; {} asks for the \
                 labels' probabilities",
                repr_str(name)
            ));
        }
        if self.model.attention_mask == "set" {
            return self.joined_set_choice_spec(
                call,
                name,
                partner,
                expected_rows,
                partner_tokens,
                anchors,
                tokenizer,
            );
        }
        if self.model.role == "decision" {
            return self.joined_choice_spec(
                call,
                name,
                partner,
                expected_rows,
                partner_tokens,
                tokenizer,
            );
        }
        let Some(prompt) = lower_prompt(&call.prompt, tokenizer)?.lettered else {
            return refused(
                "the letters rule needs a one-token letter for every label, which the tokenizer \
                 does not have",
            );
        };
        let (_, tail, labels) = classify_prompt_tokens(&prompt, &prompt.prompt.letters, tokenizer)?;
        let head = prompt.preamble_token_ids.clone();
        let note = label_parts(&prompt, &self.alias).1;
        let partner_label = label_parts(&prompt, partner).0;
        let block = round_half_even(partner_tokens) + tail.len() as i64 - 1;
        let suffixes: Vec<i64> = suffix_lengths(LETTERS_SCORING, &labels, 0, 1, &[])
            .map_err(CompileError::new)?
            .into_iter()
            .map(|length| block + length)
            .collect();
        let distinct: BTreeSet<&Vec<i64>> = labels.iter().collect();
        let estimated = classify::estimate_suffixes(
            &self.inputs(
                expected_rows,
                head.len() as i64,
                (note.len() + partner_label.len()) as i64,
                true,
            ),
            &suffixes,
            self.model.canvas_tokens,
            None,
            None,
            distinct.len() as i64,
        );
        let need = head.len() as i64
            + self.longest
            + note.len() as i64
            + suffixes.iter().max().copied().unwrap_or(0);
        if need > self.budget {
            return over_budget(
                &format!("a row of {} x {}", repr_str(&self.alias), repr_str(partner)),
                need,
                self.budget,
            );
        }
        let mut spec = base_spec(
            &call.prompt,
            name,
            vec![self.alias.clone(), partner.to_string()],
            expected_rows,
            call,
        );
        spec.estimated_seconds = estimated.seconds;
        spec.estimated_work = estimated.work.values();
        spec.prompt_token_parts = vec![head, tail];
        spec.label_token_ids = labels;
        spec.scoring = LETTERS_SCORING.to_string();
        spec.join_layout = Some((tokens_value(&note), tokens_value(&partner_label)));
        Ok((spec, estimated.work))
    }

    fn set_choice_spec(
        &self,
        call: &ModelCall,
        name: &str,
        expected_rows: f64,
        lowered: &LoweredPrompt,
        tokenizer: &Tokenizer,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        let head = lowered.preamble_token_ids.clone();
        let tail = lowered.tail_token_ids.clone();
        let (state_tail, question_header) =
            crate::lowering::set_question_parts(lowered, &self.model.prompt_layout, tokenizer)?;
        let need = head.len() as i64 + self.longest + tail.len() as i64;
        if need > self.chunk {
            return over_budget(
                &format!("a row of {}", repr_str(&self.alias)),
                need,
                self.chunk,
            );
        }
        let state = head.len() as f64 + self.mean + state_tail as f64;
        let work = &packed(
            &Quantity::Scalar(state),
            &[(tail.len() as i64 - state_tail) as f64],
            self.model.sliding_window,
        ) * expected_rows;
        let mut spec = base_spec(
            &call.prompt,
            name,
            vec![self.alias.clone()],
            expected_rows,
            call,
        );
        spec.estimated_seconds = unrounded_seconds(&work, &self.model, &self.device, self.chunk);
        spec.estimated_work = work.values();
        spec.prompt_token_parts = vec![head, tail];
        spec.label_token_ids = option_spellings(&lowered.prompt.letters, tokenizer)?;
        spec.scoring = SET_CHOICE_SCORING.to_string();
        spec.probabilities = call.probabilities;
        spec.state_tail_tokens = state_tail;
        spec.question_header_token_ids = question_header;
        Ok((spec, work))
    }

    #[allow(clippy::too_many_arguments)]
    fn joined_set_choice_spec(
        &self,
        call: &ModelCall,
        name: &str,
        partner: &str,
        expected_rows: f64,
        partner_tokens: f64,
        anchors: Option<f64>,
        tokenizer: &Tokenizer,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        let prompt = lower_prompt(&call.prompt, tokenizer)?;
        let (anchor_label, frame) = label_parts(&prompt, &self.alias);
        let (partner_label, _) = label_parts(&prompt, partner);
        let mut head = prompt.preamble_token_ids.clone();
        head.extend(anchor_label);
        let tail = prompt.tail_token_ids.clone();
        let question =
            partner_label.len() as i64 + round_half_even(partner_tokens) + tail.len() as i64;
        let need = head.len() as i64 + self.longest + frame.len() as i64 + question;
        if need > self.chunk {
            return over_budget(
                &format!("a row of {} x {}", repr_str(&self.alias), repr_str(partner)),
                need,
                self.chunk,
            );
        }
        let anchors = 1.0_f64.max(anchors.filter(|found| *found != 0.0).unwrap_or(1.0));
        let per_anchor = 1.0_f64.max(expected_rows / anchors);
        let passes = (per_anchor / self.model.set_size as f64).ceil();
        let asked = 1_i64.max(round_half_even(per_anchor / passes)) as usize;
        let state = head.len() as f64 + self.mean + frame.len() as f64;
        let work = &packed(
            &Quantity::Scalar(state),
            &vec![question as f64; asked],
            self.model.sliding_window,
        ) * (passes * anchors);
        let layout = prompt_layout(&self.model.prompt_layout)?;
        let mut spec = base_spec(
            &call.prompt,
            name,
            vec![self.alias.clone(), partner.to_string()],
            expected_rows,
            call,
        );
        spec.estimated_seconds = unrounded_seconds(&work, &self.model, &self.device, self.chunk);
        spec.estimated_work = work.values();
        spec.prompt_token_parts = vec![head, tail];
        spec.label_token_ids = option_spellings(&prompt.prompt.letters, tokenizer)?;
        spec.scoring = SET_CHOICE_SCORING.to_string();
        spec.join_layout = Some((tokens_value(&frame), tokens_value(&partner_label)));
        spec.separator_tokens = tokenizer(&prompt.prompt.tail_segments[0])?.len() as i64;
        spec.question_header_token_ids = tokenizer(layout.question_label)?;
        Ok((spec, work))
    }

    #[allow(clippy::too_many_arguments)]
    fn choice_spec(
        &self,
        call: &ModelCall,
        name: &str,
        expected_rows: f64,
        aliases: Vec<String>,
        head: Vec<i64>,
        lowered: &LoweredPrompt,
        tokenizer: &Tokenizer,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        let (tail, frame, blocks) = choice_token_parts(lowered, tokenizer)?;
        let request = tail.len() as i64 - frame;
        let estimated = classify::estimate_suffixes(
            &self.inputs(expected_rows, head.len() as i64, frame, false),
            &[request],
            0,
            None,
            Some((blocks.len() + 1) as f64),
            0,
        );
        let mut spec = base_spec(&call.prompt, name, aliases, expected_rows, call);
        spec.estimated_seconds = estimated.seconds;
        spec.estimated_work = estimated.work.values();
        spec.prompt_token_parts = vec![head, tail];
        spec.label_token_ids = blocks;
        spec.scoring = DECISION_SCORING.to_string();
        spec.probabilities = call.probabilities;
        spec.frame_tokens = frame;
        Ok((spec, estimated.work))
    }

    fn joined_choice_spec(
        &self,
        call: &ModelCall,
        name: &str,
        partner: &str,
        expected_rows: f64,
        partner_tokens: f64,
        tokenizer: &Tokenizer,
    ) -> Result<(ClassifySpec, Work), CompileError> {
        let prompt = lower_prompt(&call.prompt, tokenizer)?;
        let head = prompt.preamble_token_ids.clone();
        let (tail, frame, blocks) = choice_token_parts(&prompt, tokenizer)?;
        let note = label_parts(&prompt, &self.alias).1;
        let partner_label = label_parts(&prompt, partner).0;
        let request = round_half_even(partner_tokens) + tail.len() as i64;
        let estimated = classify::estimate_suffixes(
            &self.inputs(
                expected_rows,
                head.len() as i64,
                (note.len() + partner_label.len()) as i64,
                true,
            ),
            &[request],
            0,
            None,
            Some((blocks.len() + 1) as f64),
            0,
        );
        let need = head.len() as i64 + self.longest + note.len() as i64 + request;
        if need > self.budget {
            return over_budget(
                &format!("a row of {} x {}", repr_str(&self.alias), repr_str(partner)),
                need,
                self.budget,
            );
        }
        let mut spec = base_spec(
            &call.prompt,
            name,
            vec![self.alias.clone(), partner.to_string()],
            expected_rows,
            call,
        );
        spec.estimated_seconds = estimated.seconds;
        spec.estimated_work = estimated.work.values();
        spec.prompt_token_parts = vec![head, tail];
        spec.label_token_ids = blocks;
        spec.scoring = DECISION_SCORING.to_string();
        spec.join_layout = Some((tokens_value(&note), tokens_value(&partner_label)));
        spec.frame_tokens = frame;
        Ok((spec, estimated.work))
    }
}

/// A one-table classification's spec, as `ClassifyPlanner.spec`.
///
/// # Errors
///
/// As `ClassifyTable::prepare`.
pub fn classify_spec(
    context: &PlanningContext,
    call: &ModelCall,
    name: &str,
    alias: &str,
    expected_rows: f64,
) -> Result<ClassifySpec, CompileError> {
    let table = classify_statistics(context, alias, &[])?;
    Ok(table
        .prepare(call, name, expected_rows, context.tokenizer, context.decode)?
        .0)
}

/// A one-table classification's work with its cheapest scoring rule,
/// or None when no rule can run.
///
/// # Errors
///
/// As `ClassifyTable::prepare`.
pub fn classify_work(
    context: &PlanningContext,
    call: &ModelCall,
    name: &str,
    alias: &str,
    resident: bool,
    expected_rows: f64,
) -> Result<Option<Work>, CompileError> {
    let spec = classify_spec(context, call, name, alias, expected_rows)?;
    if spec.estimated_seconds != 0.0 {
        return Ok(Some(Work::from_values(&spec.estimated_work)));
    }
    let table = classify_statistics(context, alias, &[])?;
    let mut spec = spec;
    spec.expected_rows = expected_rows;
    Ok(table
        .choose_scoring(&spec, call, resident, context.tokenizer)
        .ok()
        .map(|chosen| Work::from_values(&chosen.estimated_work)))
}
