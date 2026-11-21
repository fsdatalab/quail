//! Lower a prompt's text to the token ids one model reads.
//!
//! Rust does not tokenize: the planner passes a [`Tokenizer`], which in
//! Quail calls back into Python's gigatoken tokenizer.

use std::collections::BTreeSet;

use crate::physical::Prompt;
use serde::{Deserialize, Serialize};

use crate::error::{CompileError, compile_error};
use crate::expressions::{argument_alias, repr_str};
use crate::prompts::{
    Turn, bind_classify_prompt, bind_join_prompt, bind_prompt, code_candidates, prompt_layout,
};

/// Text -> token ids, without special tokens.
pub type Tokenizer<'a> = dyn Fn(&str) -> Result<Vec<i64>, CompileError> + 'a;

/// A token id -> its text, as a tokenizer's `decode` of one token.
pub type Decoder<'a> = dyn Fn(&[i64]) -> Result<String, CompileError> + 'a;

/// AI.CLASSIFY's most labels; Jev's limit on one choice's options.
pub use crate::prompts::MAX_LETTERS;

/// A set-attention question's state tail length and later-question header:
/// the tokens of the tail's first segment, which the state keeps, and the
/// ids of the layout's question label, which each later question starts with.
///
/// # Errors
///
/// The layout is unknown, or the tokenizer fails.
pub fn set_question_parts(
    lowered: &LoweredPrompt,
    layout: &str,
    tokenizer: &Tokenizer,
) -> Result<(i64, Vec<i64>), CompileError> {
    let layout = crate::prompts::prompt_layout(layout)?;
    Ok((
        tokenizer(&lowered.prompt.tail_segments[0])?.len() as i64,
        tokenizer(layout.question_label)?,
    ))
}

/// Per row a decision readout reads, its distance before the prompt's
/// last row, then 0.
///
/// # Errors
///
/// The model's layout is unknown, or the tokenizer fails.
pub fn answer_row_offsets(
    model: &PromptModel,
    tokenizer: &Tokenizer,
) -> Result<Vec<usize>, CompileError> {
    let layout = crate::prompts::prompt_layout(&model.prompt_layout)?;
    let mut segments = layout.answer_segments.clone();
    let Some(last) = segments.last_mut() else {
        return Ok(vec![0]);
    };
    last.push_str(&model.turn.1);
    let lengths = segments
        .iter()
        .map(|segment| tokenizer(segment).map(|ids| ids.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut offsets: Vec<usize> = layout
        .answer_rows
        .iter()
        .map(|row| lengths[row + 1..].iter().sum())
        .collect();
    offsets.push(0);
    Ok(offsets)
}

/// A classification's head, tail, and label token sequences.
pub type ClassifyTokens = (Vec<i64>, Vec<i64>, Vec<Vec<i64>>);

/// A decision classification's tail tokens, frame length, and option blocks.
pub type ChoiceTokens = (Vec<i64>, i64, Vec<Vec<i64>>);

/// A prompt's text with the token ids one tokenizer gives it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoweredPrompt {
    pub prompt: Prompt,
    pub preamble_token_ids: Vec<i64>,
    pub tail_token_ids: Vec<i64>,
    pub frame_tokens: i64,
    /// Per placeholder: alias, its label's ids, its frame's ids.
    pub label_token_ids: Vec<(String, Vec<i64>, Vec<i64>)>,
    pub lettered: Option<Box<LoweredPrompt>>,
}

/// The model fields lowering reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptModel {
    /// "generative", "reranker", or another role.
    pub role: String,
    /// (true spellings, false spellings), or empty.
    pub answer_spellings: Vec<Vec<String>>,
    pub turn: Turn,
    pub prompt_layout: String,
}

/// Text with every `{N}` placeholder removed.
fn without_placeholders(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(index) = rest.find('{') {
        out.push_str(&rest[..index]);
        let after = &rest[index + 1..];
        let digits = after.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 && after[digits..].starts_with('}') {
            rest = &after[digits + 1..];
        } else {
            out.push('{');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Distinct one-token letters to stand for the labels, as `choice_letters`.
///
/// # Errors
///
/// More than `MAX_LETTERS` labels, or the tokenizer fails.
pub fn choice_letters(
    count: usize,
    tokenizer: &Tokenizer,
    prefix: &str,
) -> Result<Vec<String>, CompileError> {
    if count > MAX_LETTERS {
        return compile_error(format!(
            "AI.CLASSIFY takes at most {MAX_LETTERS} labels, got {count}"
        ));
    }
    let (mut letters, mut seen) = (Vec::new(), BTreeSet::new());
    let upper: Vec<String> = ('A'..='Z').map(String::from).collect();
    let lower: Vec<String> = ('a'..='z').map(String::from).collect();
    let mut found = upper.clone();
    found.extend(lower);
    for first in &upper {
        for second in &upper {
            found.push(format!("{first}{second}"));
        }
    }
    for letter in found {
        let ids = tokenizer(&format!("{prefix}{letter}"))?;
        if ids.len() != 1 || !seen.insert(ids[0]) {
            continue;
        }
        letters.push(letter);
        if letters.len() == count {
            break;
        }
    }
    Ok(letters)
}

/// The codes a coded prompt lists its labels under.
///
/// Decision 3.0's rule (`d3_format.answer_codes`), shared by Base Labs'
/// models: candidates in `code_candidates` order, each kept when it is
/// one token of its own, a token no earlier code took, and still the
/// same single token right after the assistant turn opens.
///
/// # Errors
///
/// The tokenizer keeps fewer than `count` codes, or it fails.
pub fn code_letters(
    count: usize,
    tokenizer: &Tokenizer,
    opener: &str,
) -> Result<Vec<String>, CompileError> {
    let opened = tokenizer(opener)?;
    let (mut letters, mut taken) = (Vec::new(), BTreeSet::new());
    for code in code_candidates() {
        let ids = tokenizer(&code)?;
        if ids.len() != 1 || taken.contains(&ids[0]) {
            continue;
        }
        let mut expected = opened.clone();
        expected.push(ids[0]);
        if tokenizer(&format!("{opener}{code}"))? != expected {
            continue;
        }
        taken.insert(ids[0]);
        letters.push(code);
        if letters.len() == count {
            return Ok(letters);
        }
    }
    compile_error(format!(
        "the tokenizer keeps {} one-token codes; the prompt names {count} labels",
        letters.len()
    ))
}

/// Each option's code as d1 writes it, as `option_aliases`.
///
/// # Errors
///
/// No one-token code is left for an option, or the tokenizer fails.
pub fn option_aliases(
    codes: &[String],
    tokenizer: &Tokenizer,
) -> Result<Vec<String>, CompileError> {
    let (mut used, mut out) = (BTreeSet::new(), Vec::new());
    let candidates = code_candidates();
    let mut fallbacks = candidates[..26].to_vec();
    fallbacks.extend((0..100).map(|index| format!("{index:02}")));
    fallbacks.extend(('a'..='z').map(String::from));
    fallbacks.extend((0..200).map(|index| format!("#{index}")));
    fallbacks.extend_from_slice(&candidates[26..]);
    let take = |code: &str,
                used: &mut BTreeSet<i64>,
                out: &mut Vec<String>|
     -> Result<bool, CompileError> {
        let ids = tokenizer(code)?;
        if ids.len() != 1 || used.contains(&ids[0]) {
            return Ok(false);
        }
        used.insert(ids[0]);
        out.push(code.to_string());
        Ok(true)
    };
    for code in codes {
        if take(code, &mut used, &mut out)? {
            continue;
        }
        let mut taken = false;
        for other in &fallbacks {
            if take(other, &mut used, &mut out)? {
                taken = true;
                break;
            }
        }
        if !taken {
            return compile_error(format!(
                "no one-token option code left for {} options",
                codes.len()
            ));
        }
    }
    Ok(out)
}

/// Per option, the token ids d1 scores, as `option_spellings`.
///
/// # Errors
///
/// A code is not one token, or the tokenizer fails.
pub fn option_spellings(
    codes: &[String],
    tokenizer: &Tokenizer,
) -> Result<Vec<Vec<i64>>, CompileError> {
    let mut out = Vec::new();
    for code in codes {
        let ids = tokenizer(code)?;
        let [first] = ids.as_slice() else {
            return compile_error(format!(
                "not enough values to unpack (expected 1, got {})",
                ids.len()
            ));
        };
        let spaced = tokenizer(&format!(" {code}"))?;
        let mut spelling = vec![*first];
        if spaced.len() == 1 && spaced[0] != *first {
            spelling.push(spaced[0]);
        }
        out.push(spelling);
    }
    Ok(out)
}

/// Each read slot's row, counted back from the segment's last token.
///
/// # Errors
///
/// The segment does not end in its slot lines, or a colon's token
/// differs between the segment and the text through it.
pub fn read_slot_offsets(
    segment: &str,
    names: &[String],
    tokenizer: &Tokenizer,
) -> Result<Vec<i64>, CompileError> {
    let marks: Vec<String> = names.iter().map(|name| format!("{name}:")).collect();
    let marks = marks.join("\n");
    if !segment.ends_with(&marks) {
        return compile_error("the read segment ends in its slot lines");
    }
    let whole = tokenizer(segment)?;
    let characters: Vec<char> = segment.chars().collect();
    let mut end = characters.len() - marks.chars().count();
    let mut offsets = Vec::new();
    for name in names {
        end += name.chars().count() + 1;
        let through = tokenizer(&characters[..end].iter().collect::<String>())?;
        let prefix = through.len().saturating_sub(1);
        if whole.get(..prefix) != Some(&through[..prefix]) {
            return compile_error(format!(
                "the {} slot's colon is not one token apart from the text before it",
                repr_str(name)
            ));
        }
        offsets.push(through.len() as i64 - whole.len() as i64);
        end += 1;
    }
    Ok(offsets)
}

/// Tokenize a prompt's pieces, and letter a classification when possible.
///
/// # Errors
///
/// The tokenizer fails, or rebinding a classification with letters is refused.
pub fn lower_prompt(prompt: &Prompt, tokenizer: &Tokenizer) -> Result<LoweredPrompt, CompileError> {
    if prompt.kind == "score" {
        return Ok(LoweredPrompt {
            prompt: prompt.clone(),
            preamble_token_ids: Vec::new(),
            tail_token_ids: Vec::new(),
            frame_tokens: 0,
            label_token_ids: Vec::new(),
            lettered: None,
        });
    }
    let layout = prompt_layout(&prompt.layout)?;
    let rebind = |letters: &[String]| {
        bind_classify_prompt(
            &prompt.source,
            &prompt.args,
            &prompt.categories,
            &prompt.descriptions,
            &prompt.turn,
            &prompt.task_description,
            &layout,
            letters,
        )
    };
    if prompt.kind == "label" && layout.option_end.is_some() {
        let codes = option_aliases(&prompt.letters, tokenizer)?;
        if codes != prompt.letters {
            return lower_prompt(&rebind(&codes)?, tokenizer);
        }
    }
    if prompt.kind == "label" && prompt.letters.is_empty() && layout.options_label.is_some() {
        // the codes a coded classification lists depend on the tokenizer
        let codes = code_letters(prompt.categories.len(), tokenizer, &prompt.turn.1)?;
        return lower_prompt(&rebind(&codes)?, tokenizer);
    }
    let mut lettered = None;
    if prompt.kind == "label"
        && !prompt.categories.is_empty()
        && prompt.letters.is_empty()
        && layout.choice_label.is_none()
    {
        let letters = choice_letters(prompt.categories.len(), tokenizer, &prompt.label_prefix)?;
        if letters.len() == prompt.categories.len() {
            lettered = Some(Box::new(lower_prompt(&rebind(&letters)?, tokenizer)?));
        }
    }
    let mut label_token_ids = Vec::new();
    for (alias, label, frame) in &prompt.label_texts {
        label_token_ids.push((alias.clone(), tokenizer(label)?, tokenizer(frame)?));
    }
    let preamble_token_ids = tokenizer(&prompt.preamble)?;
    let segments = if prompt.tail_segments.is_empty() {
        vec![without_placeholders(&prompt.tail)]
    } else {
        prompt.tail_segments.clone()
    };
    let mut tail_token_ids = Vec::new();
    for segment in &segments {
        tail_token_ids.extend(tokenizer(segment)?);
    }
    Ok(LoweredPrompt {
        prompt: prompt.clone(),
        preamble_token_ids,
        tail_token_ids,
        frame_tokens: if prompt.frame_text.is_empty() {
            0
        } else {
            tokenizer(&prompt.frame_text)?.len() as i64
        },
        label_token_ids,
        lettered,
    })
}

fn prompt_aliases(prompt: &Prompt) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for argument in &prompt.args {
        let alias = argument_alias(argument);
        if !found.contains(&alias) {
            found.push(alias);
        }
    }
    found
}

fn without_document(template: &str, marker: &str) -> String {
    let mut text = template.trim();
    if let Some(rest) = text.strip_prefix(marker) {
        text = rest.trim_start();
    }
    if let Some(rest) = text.strip_suffix(marker) {
        text = rest.trim_end();
    }
    text.replace(marker, "this document")
}

/// The query text a reranker reads, as `score_query_template`.
///
/// # Errors
///
/// The prompt reads more than a pair.
pub fn score_query_template(prompt: &Prompt) -> Result<String, CompileError> {
    let aliases = prompt_aliases(prompt);
    if aliases.len() == 1 {
        return Ok(without_document(&prompt.template, "{0}"));
    }
    if aliases.len() == 2 && prompt.args.len() == 2 {
        return Ok(without_document(&prompt.template, "{1}"));
    }
    compile_error("AI.SCORE supports one document or one document pair")
}

const TEXT_ANALYSIS_SYSTEM_PROMPT: &str = "You are an expert at text analysis";
const QWEN3_RERANKER_INSTRUCTION: &str =
    "Judge whether the document meets the requirements in the query.";

/// Tokenize the fixed pieces of a score's rendered prompt around the
/// documents, as `score_token_parts`.
///
/// # Errors
///
/// The prompt reads more than a pair, or the tokenizer fails.
pub fn score_token_parts(
    prompt: &Prompt,
    model: &PromptModel,
    tokenizer: &Tokenizer,
) -> Result<Vec<Vec<i64>>, CompileError> {
    if model.role != "reranker" {
        return answer_token_parts(prompt, model, tokenizer);
    }
    let query = score_query_template(prompt)?;
    let system = format!(
        "{TEXT_ANALYSIS_SYSTEM_PROMPT}. Judge whether the Document meets the requirements \
         based on the Query and the Instruct provided. Note that the answer can only be \"yes\" \
         or \"no\"."
    );
    let rendered = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<Instruct>: \
         {QWEN3_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: \
         {{document}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let (before, after) = rendered.split_once("{document}").expect("a document field");
    let mut parts = vec![before.to_string(), after.to_string()];
    if prompt_aliases(prompt).len() == 2 {
        let (first, middle) = before.split_once("{0}").expect("a first document");
        parts = vec![first.to_string(), middle.to_string(), after.to_string()];
    }
    parts.iter().map(|part| tokenizer(part)).collect()
}

fn answer_token_parts(
    prompt: &Prompt,
    model: &PromptModel,
    tokenizer: &Tokenizer,
) -> Result<Vec<Vec<i64>>, CompileError> {
    let layout = prompt_layout(&model.prompt_layout)?;
    let aliases = prompt_aliases(prompt);
    if aliases.len() == 1 {
        let bound = lower_prompt(
            &bind_prompt(&prompt.template, &prompt.args, &model.turn, &layout)?,
            tokenizer,
        )?;
        return Ok(vec![bound.preamble_token_ids, bound.tail_token_ids]);
    }
    if prompt.args.len() != 2 {
        return compile_error("AI.SCORE supports one document or one document pair");
    }
    let bound = lower_prompt(
        &bind_join_prompt(&prompt.template, &prompt.args, &model.turn, &layout)?,
        tokenizer,
    )?;
    let piece = |alias: &str| {
        bound
            .label_token_ids
            .iter()
            .rev()
            .find(|(found, _, _)| found == alias)
            .map(|(_, label, frame)| (label.clone(), frame.clone()))
            .expect("a piece per alias")
    };
    let (left, right) = (piece(&aliases[0]), piece(&aliases[1]));
    let mut middle = left.1;
    middle.extend(right.0);
    Ok(vec![bound.preamble_token_ids, middle, bound.tail_token_ids])
}

/// The sorted ids of the spellings that are one token each.
fn spelling_token_ids(
    tokenizer: &Tokenizer,
    spellings: &[Vec<String>],
) -> Result<(Vec<i64>, Vec<i64>), CompileError> {
    let mut sides = Vec::new();
    for words in spellings {
        let mut ids = BTreeSet::new();
        for word in words {
            let tokens = tokenizer(word)?;
            if tokens.len() == 1 {
                ids.insert(tokens[0]);
            }
        }
        if ids.is_empty() {
            let listed: Vec<String> = words.iter().map(|word| repr_str(word)).collect();
            let tuple = if listed.len() == 1 {
                format!("({},)", listed[0])
            } else {
                format!("({})", listed.join(", "))
            };
            return compile_error(format!("no spelling among {tuple} is one token"));
        }
        sides.push(ids.into_iter().collect::<Vec<_>>());
    }
    let falses = sides.pop().unwrap_or_default();
    let trues = sides.pop().unwrap_or_default();
    Ok((trues, falses))
}

/// The token ids an answer compares: the model's true and false spellings.
///
/// # Errors
///
/// A side has no one-token spelling, or the tokenizer fails.
pub fn answer_ids(
    model: &PromptModel,
    tokenizer: &Tokenizer,
) -> Result<(Vec<i64>, Vec<i64>), CompileError> {
    if model.role == "reranker" {
        return Ok((tokenizer("yes")?, tokenizer("no")?));
    }
    if !model.answer_spellings.is_empty() {
        return spelling_token_ids(tokenizer, &model.answer_spellings);
    }
    let (mut trues, mut falses) = (BTreeSet::new(), BTreeSet::new());
    for (words, ids) in [
        (["TRUE", " TRUE", "True", " True"], &mut trues),
        (["FALSE", " FALSE", "False", " False"], &mut falses),
    ] {
        for word in words {
            if let Some(first) = tokenizer(word)?.first() {
                ids.insert(*first);
            }
        }
    }
    Ok((trues.into_iter().collect(), falses.into_iter().collect()))
}

/// A classification's head, tail, and label token sequences.
///
/// # Errors
///
/// The tokenizer fails.
pub fn classify_prompt_tokens(
    lowered: &LoweredPrompt,
    labels: &[String],
    tokenizer: &Tokenizer,
) -> Result<ClassifyTokens, CompileError> {
    let mut ids = Vec::new();
    for label in labels {
        ids.push(tokenizer(&format!(
            "{}{label}",
            lowered.prompt.label_prefix
        ))?);
    }
    Ok((
        lowered.preamble_token_ids.clone(),
        lowered.tail_token_ids.clone(),
        ids,
    ))
}

/// A decision classification's tail tokens, frame length, and option blocks.
///
/// A layout with coded options has no frame and no blocks: the whole
/// tail is one request, and every option is read at its last row.
///
/// # Errors
///
/// The tokenizer fails.
pub fn choice_token_parts(
    lowered: &LoweredPrompt,
    tokenizer: &Tokenizer,
) -> Result<ChoiceTokens, CompileError> {
    if prompt_layout(&lowered.prompt.layout)?
        .options_label
        .is_some()
    {
        return Ok((lowered.tail_token_ids.clone(), 0, Vec::new()));
    }
    let segments = &lowered.prompt.tail_segments;
    let frame = tokenizer(&segments[0])?.len() as i64;
    let mut blocks = Vec::new();
    for block in &segments[1..segments.len() - 1] {
        blocks.push(tokenizer(block)?);
    }
    Ok((lowered.tail_token_ids.clone(), frame, blocks))
}
