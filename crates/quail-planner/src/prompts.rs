//! Prompt text for each model call, as `quail/common/prompts.py` binds it.
//!
//! A front end binds each AI function call's template and column
//! arguments into a `Prompt`: the text before the document, the text
//! after it, and the pieces the planner tokenizes on their own.

use std::collections::BTreeMap;

use quail_physical::{Prompt, ValueExpression};
use serde_json::Value;

use crate::expressions::{
    CompileError, argument_alias, column_references, compile_error, repr_str,
};

pub const SHARED_PRE: &str = "DOCUMENT:\n";
const JOIN_QUESTION_SEP: &str = "\n\n";
const TASK_INSTRUCTION: &str = "Evaluate TRUE or FALSE for the following question: ";
const ANSWER_CUE: &str = "\nANSWER:";
pub const LABEL_PREFIX: &str = " ";
const CLASSIFY_INSTRUCTION: &str =
    "Answer with exactly one of the categories below for the following question: ";
const LETTERS_INSTRUCTION: &str =
    "Answer with the letter of exactly one of the categories below for the following question: ";
const CATEGORIES_HEADER: &str = "\n\nCategories:";
const CHOICE_QUESTION: &str = "Which option best fits the context?";
const EXTRACT_QUESTION_LABEL: &str = "\n\nQuestion: ";
const MAP_TEMPLATE_HEAD: &str = "{0}\n\n";
const EXTRACT_INSTRUCTION: &str = "Copy the words that answer the question from the document. \
    Copy them exactly. Use the fewest words that answer it. If the document does not answer \
    the question, leave the phrase empty.";
const EXTRACT_FORMAT: &str = "Answer in this format:";
const EXTRACT_PHRASE_FORMAT: &str = "Phrase: <copied words>";
const EXTRACT_PHRASE_CUE: &str = "Phrase:";

/// The fixed text of an AI.IF filter or join prompt, as `PromptLayout`.
#[derive(Debug, Clone)]
pub struct PromptLayout {
    pub name: &'static str,
    pub document_label: &'static str,
    pub question_label: &'static str,
    pub question_end: &'static str,
    pub answer_segments: Vec<String>,
    pub question_separator: Option<&'static str>,
    pub answer_rows: Vec<usize>,
    pub choice_label: Option<&'static str>,
    pub record_label: Option<&'static str>,
    pub label_segment: bool,
    pub option_end: Option<&'static str>,
    pub read_end: Option<&'static str>,
    pub read_slots: Vec<&'static str>,
}

impl PromptLayout {
    fn plain(name: &'static str) -> Self {
        Self {
            name,
            document_label: SHARED_PRE,
            question_label: TASK_INSTRUCTION,
            question_end: ANSWER_CUE,
            answer_segments: Vec::new(),
            question_separator: None,
            answer_rows: Vec::new(),
            choice_label: None,
            record_label: None,
            label_segment: false,
            option_end: None,
            read_end: None,
            read_slots: Vec::new(),
        }
    }

    /// The text after a document: one entry per tokenized segment.
    pub fn question_segments(&self, question: &str, separator: &str) -> Vec<String> {
        let separator = self.question_separator.unwrap_or(separator);
        let mut segments = if self.label_segment {
            vec![
                format!("{separator}{}", self.question_label),
                format!("{question}{}", self.question_end),
            ]
        } else {
            vec![format!(
                "{separator}{}{question}{}",
                self.question_label, self.question_end
            )]
        };
        segments.extend(self.answer_segments.iter().cloned());
        segments
    }

    fn record(&self, name: &str) -> String {
        self.record_label
            .expect("a record layout")
            .replacen("{}", name, 1)
    }
}

fn decision2_option(key: &str, description: Option<&str>) -> String {
    let mut fields = BTreeMap::new();
    fields.insert(
        "description",
        description.map(Value::from).unwrap_or(Value::Null),
    );
    fields.insert("key", Value::from(key));
    let option = serde_json::to_string(&fields).expect("JSON text");
    format!("\n<option>\n{option}\n</option>")
}

/// The layout a ModelSpec's `prompt_layout` names.
///
/// # Errors
///
/// No layout has the name.
pub fn prompt_layout(name: &str) -> Result<PromptLayout, CompileError> {
    match name {
        "ai-if" => Ok(PromptLayout::plain("ai-if")),
        "decision2-noul" => Ok(PromptLayout {
            document_label: "Context:\n",
            question_label: "Task type: noul\nQuestion:\n",
            question_end: "\nOptions:",
            answer_segments: vec![
                decision2_option("false", Some("No")),
                decision2_option("true", Some("Yes")),
                "\n\nSelect the single option best supported by the context and instructions.\
                 \nDecision:"
                    .to_string(),
            ],
            question_separator: Some("\n\n"),
            answer_rows: vec![0, 1],
            choice_label: Some("Task type: choice\nQuestion:\n"),
            ..PromptLayout::plain("decision2-noul")
        }),
        "d1" => Ok(PromptLayout {
            document_label: "",
            question_label: "QUESTION:\n",
            question_end: "\n\nReply with yes or no only.",
            question_separator: Some("\n\n\n"),
            record_label: Some("{} record:\n"),
            label_segment: true,
            option_end: Some("\n\nReply with the option code only."),
            read_end: Some("\n\nRead the text from the state, or nothing if it is absent."),
            read_slots: vec!["start", "end"],
            ..PromptLayout::plain("d1")
        }),
        other => compile_error(format!("unknown prompt layout {}", repr_str(other))),
    }
}

fn marker(placeholder: usize) -> String {
    format!("{{{placeholder}}}")
}

fn join_label(placeholder: usize) -> String {
    format!("\n\nDOCUMENT {}:\n", marker(placeholder))
}

fn join_anchor_note(placeholder: usize) -> String {
    format!(
        "\n\n(The document above is DOCUMENT {}.)",
        marker(placeholder)
    )
}

fn render_join_frame(template: &str, placeholder: usize, layout: &PromptLayout) -> String {
    format!(
        "{}{JOIN_QUESTION_SEP}{}{template}",
        join_anchor_note(placeholder),
        layout.question_label
    )
}

/// The length of a `{N}` placeholder at the start of the text.
fn leading_placeholder(text: &str) -> Option<usize> {
    let rest = text.strip_prefix('{')?;
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    (digits > 0 && rest[digits..].starts_with('}')).then_some(digits + 2)
}

fn placeholders(template: &str) -> Vec<u64> {
    let mut found = Vec::new();
    for (index, _) in template.match_indices('{') {
        if let Some(length) = leading_placeholder(&template[index..]) {
            found.push(
                template[index + 1..index + length - 1]
                    .parse()
                    .unwrap_or(u64::MAX),
            );
        }
    }
    found
}

fn check_placeholders(template: &str, num_args: usize) -> Result<(), CompileError> {
    let mut slots = placeholders(template);
    slots.sort_unstable();
    slots.dedup();
    let expected: Vec<u64> = (0..num_args as u64).collect();
    if slots != expected {
        let listed: Vec<String> = slots.iter().map(u64::to_string).collect();
        return compile_error(format!(
            "PROMPT placeholders [{}] do not match {num_args} argument(s): expected {{0}}..{{{}}} \
             each used at least once",
            listed.join(", "),
            num_args as i64 - 1
        ));
    }
    Ok(())
}

/// Split into (pre-placeholder text, placeholder-onward text).
fn split_template(template: &str) -> (String, String) {
    match template.find('{') {
        Some(index) => (template[..index].to_string(), template[index..].to_string()),
        None => (template.to_string(), String::new()),
    }
}

fn python_strip(text: &str) -> &str {
    text.trim()
}

/// Split into (frame, canonical template), moving text before the first
/// placeholder to after it.
fn split_frame(template: &str, document_label: &str) -> Result<(String, String), CompileError> {
    let (user_pre, tail) = split_template(template);
    if tail.is_empty() {
        return Ok((String::new(), template.to_string()));
    }
    let Some(length) = leading_placeholder(&tail) else {
        let head: String = tail.chars().take(40).collect();
        return compile_error(format!(
            "template text before the first placeholder must not contain a brace that is not a \
             placeholder: {}",
            repr_str(&head)
        ));
    };
    let frame = python_strip(&user_pre).to_string();
    let rest = &tail[length..];
    let framed = if frame.is_empty() {
        String::new()
    } else {
        format!("\n\n{frame}")
    };
    Ok((
        frame.clone(),
        format!("{document_label}{}{framed}{rest}", &tail[..length]),
    ))
}

fn frame_text(frame: &str) -> String {
    if frame.is_empty() {
        String::new()
    } else {
        format!("\n\n{frame}")
    }
}

fn prompt(template: String, args: &[ValueExpression], kind: &str) -> Prompt {
    Prompt {
        template,
        args: args.to_vec(),
        preamble: String::new(),
        tail: String::new(),
        frame: String::new(),
        frame_text: String::new(),
        kind: kind.to_string(),
        layout: "ai-if".to_string(),
        tail_segments: Vec::new(),
        label_texts: Vec::new(),
        label_prefix: LABEL_PREFIX.to_string(),
        categories: Vec::new(),
        descriptions: Vec::new(),
        task_description: String::new(),
        turn: (String::new(), String::new()),
        source: String::new(),
        letters: Vec::new(),
    }
}

/// The model's chat-turn text: before the preamble, and after the answer cue.
pub type Turn = (String, String);

fn filter_question_segments(tail: &str, layout: &PromptLayout) -> Vec<String> {
    let content = tail.trim_start_matches('\n');
    let separator = &tail[..tail.len() - content.len()];
    let separator = if separator.is_empty() {
        "\n\n"
    } else {
        separator
    };
    layout.question_segments(content, separator)
}

/// Build a filter Prompt from a template and column arguments.
pub fn bind_prompt(
    template: &str,
    args: &[ValueExpression],
    turn: &Turn,
    layout: &PromptLayout,
) -> Result<Prompt, CompileError> {
    check_placeholders(template, args.len())?;
    let (_, canonical) = split_frame(template, SHARED_PRE)?;
    let (frame, template) = split_frame(template, layout.document_label)?;
    let (preamble, tail) = split_template(&template);
    let mut segments = Vec::new();
    let tail = match leading_placeholder(&tail) {
        Some(length) => {
            segments = filter_question_segments(&tail[length..], layout);
            segments.last_mut().expect("a segment").push_str(&turn.1);
            format!("{}{}", &tail[..length], segments.concat())
        }
        None => format!("{tail}{}", turn.1),
    };
    let mut bound = prompt(canonical, args, "filter");
    bound.preamble = format!("{}{preamble}", turn.0);
    bound.tail = tail;
    bound.frame_text = frame_text(&frame);
    bound.frame = frame;
    bound.layout = layout.name.to_string();
    bound.tail_segments = segments;
    Ok(bound)
}

fn answer_prefix(turn: &Turn) -> &'static str {
    if turn.1.is_empty() { LABEL_PREFIX } else { "" }
}

fn render_categories(labels: &[String], descriptions: &[String], letters: &[String]) -> String {
    let empty = vec![String::new(); labels.len()];
    let descriptions = if descriptions.is_empty() {
        &empty
    } else {
        descriptions
    };
    let mut text = CATEGORIES_HEADER.to_string();
    if !letters.is_empty() {
        for ((letter, label), description) in letters.iter().zip(labels).zip(descriptions) {
            text.push_str(&format!("\n- {letter}: {label}"));
            if !description.is_empty() {
                text.push_str(&format!(" ({description})"));
            }
        }
        return text;
    }
    for (label, description) in labels.iter().zip(descriptions) {
        text.push_str(&format!("\n- {label}"));
        if !description.is_empty() {
            text.push_str(&format!(": {description}"));
        }
    }
    text
}

fn render_classify_question(
    question: &str,
    labels: &[String],
    descriptions: &[String],
    task_description: &str,
    letters: &[String],
) -> String {
    let mut content = question.trim_start_matches('\n').to_string();
    let separator = &question[..question.len() - content.len()];
    let separator = if separator.is_empty() {
        "\n\n"
    } else {
        separator
    };
    if !task_description.is_empty() {
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(task_description);
    }
    let instruction = if letters.is_empty() {
        CLASSIFY_INSTRUCTION
    } else {
        LETTERS_INSTRUCTION
    };
    format!(
        "{separator}{instruction}{content}{}{ANSWER_CUE}",
        render_categories(labels, descriptions, letters)
    )
}

/// d1's option codes: the labels when each is one letter, else A to Z, then AA, AB, ...
fn option_codes(labels: &[String]) -> Vec<String> {
    let stripped: Vec<&str> = labels.iter().map(|label| label.trim()).collect();
    if !stripped.is_empty()
        && stripped.iter().all(|label| {
            let mut chars = label.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if c.is_alphabetic())
        })
    {
        return stripped.iter().map(|label| label.to_string()).collect();
    }
    let letters: Vec<String> = ('A'..='Z').map(|c| c.to_string()).collect();
    let mut codes = letters.clone();
    for first in &letters {
        for second in &letters {
            codes.push(format!("{first}{second}"));
        }
    }
    codes.truncate(labels.len());
    codes
}

fn record_names(args: &[ValueExpression]) -> Vec<String> {
    let tables: Vec<String> = args
        .iter()
        .map(|argument| {
            let provider = column_references(argument)
                .into_iter()
                .next()
                .map(|found| found.provider)
                .unwrap_or_default();
            if provider.is_empty() {
                argument_alias(argument)
            } else {
                provider
            }
        })
        .collect();
    args.iter()
        .zip(&tables)
        .map(|(argument, table)| {
            if tables.iter().filter(|found| *found == table).count() > 1 {
                argument_alias(argument)
            } else {
                table.clone()
            }
        })
        .collect()
}

fn name_records(template: &str, names: &[String]) -> String {
    let mut template = template.to_string();
    for (index, name) in names.iter().enumerate() {
        let record = format!("the {name} record");
        template = template.replace(&format!("DOCUMENT {}", marker(index)), &record);
        template = template.replace(&marker(index), &record);
    }
    template
}

fn distinct_aliases(args: &[ValueExpression]) -> bool {
    let mut aliases: Vec<String> = args.iter().map(argument_alias).collect();
    let count = aliases.len();
    aliases.sort();
    aliases.dedup();
    aliases.len() == count
}

fn alias_list(args: &[ValueExpression]) -> String {
    let aliases: Vec<String> = args
        .iter()
        .map(|argument| repr_str(&argument_alias(argument)))
        .collect();
    format!("[{}]", aliases.join(", "))
}

/// The classification fields every classification prompt records.
struct Recipe {
    categories: Vec<String>,
    descriptions: Vec<String>,
    task_description: String,
    turn: Turn,
    source: String,
    letters: Vec<String>,
}

impl Recipe {
    fn apply(&self, prompt: &mut Prompt) {
        prompt.categories = self.categories.clone();
        prompt.descriptions = self.descriptions.clone();
        prompt.task_description = self.task_description.clone();
        prompt.turn = self.turn.clone();
        prompt.source = self.source.clone();
        prompt.letters = self.letters.clone();
    }
}

/// Bind a classification prompt to one or two document columns.
#[allow(clippy::too_many_arguments)]
pub fn bind_classify_prompt(
    template: &str,
    args: &[ValueExpression],
    labels: &[String],
    descriptions: &[String],
    turn: &Turn,
    task_description: &str,
    layout: &PromptLayout,
    letters: &[String],
) -> Result<Prompt, CompileError> {
    check_placeholders(template, args.len())?;
    let recipe = Recipe {
        categories: labels.to_vec(),
        descriptions: descriptions.to_vec(),
        task_description: task_description.to_string(),
        turn: turn.clone(),
        source: template.to_string(),
        letters: letters.to_vec(),
    };
    if layout.option_end.is_some() {
        let codes = if letters.is_empty() {
            option_codes(labels)
        } else {
            letters.to_vec()
        };
        return bind_option_classify_prompt(
            template,
            args,
            labels,
            descriptions,
            turn,
            task_description,
            layout,
            &codes,
            recipe,
        );
    }
    if layout.choice_label.is_some() {
        return bind_choice_classify_prompt(
            template,
            args,
            labels,
            descriptions,
            turn,
            task_description,
            layout,
            recipe,
        );
    }
    if args.len() == 2 {
        if !distinct_aliases(args) {
            return compile_error(
                "each placeholder of a classification of joined rows names a distinct table",
            );
        }
        let question =
            render_classify_question(template, labels, descriptions, task_description, letters);
        let frame = join_anchor_note(0);
        let mut bound = prompt(template.to_string(), args, "label");
        bound.preamble = format!("{}{SHARED_PRE}", turn.0);
        bound.tail = format!("{question}{}", turn.1);
        bound.tail_segments = vec![bound.tail.clone()];
        bound.frame = frame.clone();
        bound.frame_text = frame;
        bound.label_texts = args
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                (
                    argument_alias(argument),
                    join_label(index),
                    join_anchor_note(index),
                )
            })
            .collect();
        bound.label_prefix = answer_prefix(turn).to_string();
        recipe.apply(&mut bound);
        return Ok(bound);
    }
    if args.len() != 1 {
        return compile_error(
            "AI.CLASSIFY reads one document per row, or one from each side of a join",
        );
    }
    let (frame, template) = split_frame(template, SHARED_PRE)?;
    let (preamble, tail) = split_template(&template);
    let Some(length) = leading_placeholder(&tail) else {
        return compile_error(format!(
            "unexpected AI.CLASSIFY template: {}",
            repr_str(&template)
        ));
    };
    let question = render_classify_question(
        &tail[length..],
        labels,
        descriptions,
        task_description,
        letters,
    );
    let mut bound = prompt(template.clone(), args, "label");
    bound.preamble = format!("{}{preamble}", turn.0);
    bound.tail = format!("{}{question}{}", &tail[..length], turn.1);
    bound.frame_text = frame_text(&frame);
    bound.frame = frame;
    bound.label_prefix = answer_prefix(turn).to_string();
    recipe.apply(&mut bound);
    Ok(bound)
}

#[allow(clippy::too_many_arguments)]
fn bind_option_classify_prompt(
    template: &str,
    args: &[ValueExpression],
    labels: &[String],
    descriptions: &[String],
    turn: &Turn,
    task_description: &str,
    layout: &PromptLayout,
    codes: &[String],
    recipe: Recipe,
) -> Result<Prompt, CompileError> {
    let empty = vec![String::new(); labels.len()];
    let descriptions = if descriptions.is_empty() {
        &empty
    } else {
        descriptions
    };
    let lines: Vec<String> = codes
        .iter()
        .zip(labels)
        .zip(descriptions)
        .map(|((code, label), description)| {
            let text = if description.is_empty() {
                label.replace('_', " ")
            } else {
                description.clone()
            };
            format!("{code} {text}")
        })
        .collect();
    let lines = lines.join("\n");
    let option_end = layout.option_end.expect("an option layout");
    let separator = layout.question_separator.unwrap_or_default();
    let with_task = |question: String| {
        if task_description.is_empty() {
            question
        } else if question.is_empty() {
            task_description.to_string()
        } else {
            format!("{question}\n{task_description}")
        }
    };
    if args.len() == 2 && layout.record_label.is_some() {
        if !distinct_aliases(args) {
            return compile_error(
                "each placeholder of a classification of joined rows names a distinct table",
            );
        }
        let names = record_names(args);
        let question = with_task(python_strip(&name_records(template, &names)).to_string());
        let state_tail = format!("{separator}{}", layout.question_label);
        let segments = vec![
            JOIN_QUESTION_SEP.to_string(),
            format!("{question}\n\nOptions:\n{lines}{option_end}{}", turn.1),
        ];
        let mut bound = prompt(template.to_string(), args, "label");
        bound.preamble = format!("{}{}", turn.0, layout.document_label);
        bound.tail = segments.concat();
        bound.frame = state_tail.clone();
        bound.frame_text = state_tail.clone();
        bound.label_prefix = String::new();
        bound.label_texts = args
            .iter()
            .zip(&names)
            .map(|(argument, name)| {
                (
                    argument_alias(argument),
                    layout.record(name),
                    state_tail.clone(),
                )
            })
            .collect();
        bound.layout = layout.name.to_string();
        bound.tail_segments = segments;
        recipe.apply(&mut bound);
        bound.letters = codes.to_vec();
        return Ok(bound);
    }
    if args.len() != 1 {
        return compile_error("this model classifies one document per row, or one joined pair");
    }
    let (_, canonical) = split_frame(template, layout.document_label)?;
    let (_, rest) = split_template(&canonical);
    let Some(length) = leading_placeholder(&rest) else {
        return compile_error(format!(
            "unexpected AI.CLASSIFY template: {}",
            repr_str(template)
        ));
    };
    let question = with_task(python_strip(&rest[length..]).to_string());
    let segments = vec![
        format!("{separator}{}", layout.question_label),
        format!("{question}\n\nOptions:\n{lines}{option_end}{}", turn.1),
    ];
    let mut bound = prompt(canonical.clone(), args, "label");
    bound.preamble = format!("{}{}", turn.0, layout.document_label);
    bound.tail = format!("{}{}", &rest[..length], segments.concat());
    bound.label_prefix = String::new();
    bound.layout = layout.name.to_string();
    bound.tail_segments = segments;
    recipe.apply(&mut bound);
    bound.letters = codes.to_vec();
    Ok(bound)
}

fn choice_segments(
    question: &str,
    labels: &[String],
    descriptions: &[String],
    layout: &PromptLayout,
) -> Vec<String> {
    let question = if question.is_empty() {
        CHOICE_QUESTION
    } else {
        question
    };
    let mut segments = vec![format!(
        "{JOIN_QUESTION_SEP}{}{question}{}",
        layout.choice_label.expect("a choice layout"),
        layout.question_end
    )];
    for (index, label) in labels.iter().enumerate() {
        let description = descriptions
            .get(index)
            .filter(|description| !description.is_empty())
            .map(String::as_str);
        segments.push(decision2_option(label, description));
    }
    segments.push(layout.answer_segments.last().expect("a close").clone());
    segments
}

#[allow(clippy::too_many_arguments)]
fn bind_choice_classify_prompt(
    template: &str,
    args: &[ValueExpression],
    labels: &[String],
    descriptions: &[String],
    turn: &Turn,
    task_description: &str,
    layout: &PromptLayout,
    recipe: Recipe,
) -> Result<Prompt, CompileError> {
    if !(1..=2).contains(&args.len()) {
        return compile_error(
            "AI.CLASSIFY reads one document per row, or one from each side of a join",
        );
    }
    let preamble = format!("{}{}", turn.0, layout.document_label);
    let (stored, mut question, frame, head, label_texts);
    if args.len() == 2 {
        if !distinct_aliases(args) {
            return compile_error(
                "each placeholder of a classification of joined rows names a distinct table",
            );
        }
        stored = template.to_string();
        question = template.to_string();
        frame = join_anchor_note(0);
        head = String::new();
        label_texts = args
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                (
                    argument_alias(argument),
                    join_label(index),
                    join_anchor_note(index),
                )
            })
            .collect();
    } else {
        stored = split_frame(template, SHARED_PRE)?.1;
        let (found_frame, rewritten) = split_frame(template, layout.document_label)?;
        let (_, rest) = split_template(&rewritten);
        let Some(length) = leading_placeholder(&rest) else {
            return compile_error(format!(
                "unexpected AI.CLASSIFY template: {}",
                repr_str(&rewritten)
            ));
        };
        frame = found_frame;
        head = rest[..length].to_string();
        question = python_strip(&rest[length..]).to_string();
        label_texts = Vec::new();
    }
    if !task_description.is_empty() {
        question = if question.is_empty() {
            task_description.to_string()
        } else {
            format!("{question}\n{task_description}")
        };
    }
    let mut segments = choice_segments(&question, labels, descriptions, layout);
    segments.last_mut().expect("a close").push_str(&turn.1);
    let mut bound = prompt(stored, args, "label");
    bound.preamble = preamble;
    bound.tail = format!("{head}{}", segments.concat());
    bound.frame = frame.clone();
    bound.frame_text = frame;
    bound.label_texts = label_texts;
    bound.layout = layout.name.to_string();
    bound.tail_segments = segments;
    recipe.apply(&mut bound);
    Ok(bound)
}

/// Bind an AI.SAE_FEATURES prompt over one document column.
pub fn bind_sae_features_prompt(
    template: &str,
    args: &[ValueExpression],
    turn: &Turn,
) -> Result<Prompt, CompileError> {
    if args.len() != 1 {
        return compile_error("AI.SAE_FEATURES reads one document column");
    }
    check_placeholders(template, args.len())?;
    let (_, canonical) = split_frame(template, SHARED_PRE)?;
    let (frame, template) = split_frame(template, SHARED_PRE)?;
    let (preamble, tail) = split_template(&template);
    let mut bound = prompt(canonical, args, "features");
    bound.preamble = format!("{}{preamble}", turn.0);
    bound.tail = format!("{tail}{}", turn.1);
    bound.frame_text = frame_text(&frame);
    bound.frame = frame;
    Ok(bound)
}

/// Build an AI.SCORE Prompt that keeps the template as written.
pub fn bind_score_prompt(template: &str, args: &[ValueExpression]) -> Result<Prompt, CompileError> {
    check_placeholders(template, args.len())?;
    if !distinct_aliases(args) {
        return compile_error(format!(
            "each AI.SCORE placeholder must name a distinct table, got aliases {}",
            alias_list(args)
        ));
    }
    Ok(prompt(template.to_string(), args, "score"))
}

/// Bind an AI.EXTRACT prompt over one document column.
pub fn bind_extract_prompt(
    args: &[ValueExpression],
    question: &str,
    turn: &Turn,
    layout: &PromptLayout,
) -> Result<Prompt, CompileError> {
    if args.len() != 1 {
        return compile_error("AI.EXTRACT reads one document column");
    }
    if question.trim().is_empty() {
        return compile_error("AI.EXTRACT needs a question with at least one word");
    }
    let mut bound = prompt("{0}".to_string(), args, "extract");
    bound.source = question.to_string();
    if let Some(read_end) = layout.read_end {
        let slots: Vec<String> = layout
            .read_slots
            .iter()
            .map(|name| format!("{name}:"))
            .collect();
        let segments = vec![
            format!(
                "{}{}",
                layout.question_separator.unwrap_or_default(),
                layout.question_label
            ),
            format!("{question}{read_end}{}{}", turn.1, slots.join("\n")),
        ];
        bound.preamble = format!("{}{}", turn.0, layout.document_label);
        bound.tail = format!("{{0}}{}", segments.concat());
        bound.layout = layout.name.to_string();
        bound.tail_segments = segments;
        return Ok(bound);
    }
    let lines = [EXTRACT_INSTRUCTION, EXTRACT_FORMAT, EXTRACT_PHRASE_FORMAT].join("\n");
    let cue = if turn.1.is_empty() {
        "\n"
    } else {
        turn.1.as_str()
    };
    bound.preamble = format!("{}{SHARED_PRE}", turn.0);
    bound.tail =
        format!("{{0}}{EXTRACT_QUESTION_LABEL}{question}\n\n{lines}{cue}{EXTRACT_PHRASE_CUE}");
    Ok(bound)
}

/// Bind an AI.MAP prompt over one document column.
pub fn bind_map_prompt(
    args: &[ValueExpression],
    instruction: &str,
    turn: &Turn,
) -> Result<Prompt, CompileError> {
    if args.len() != 1 {
        return compile_error("AI.MAP reads one document column");
    }
    if instruction.trim().is_empty() || instruction.contains('{') || instruction.contains('}') {
        return compile_error("AI.MAP needs an instruction with at least one word and no braces");
    }
    let cue = if turn.1.is_empty() {
        "\n"
    } else {
        turn.1.as_str()
    };
    let mut bound = prompt(format!("{MAP_TEMPLATE_HEAD}{instruction}"), args, "map");
    bound.preamble = format!("{}{SHARED_PRE}", turn.0);
    bound.tail = format!(
        "{{0}}\n\n{instruction}{cue}{}",
        ANSWER_CUE.trim_start_matches('\n')
    );
    Ok(bound)
}

/// Build a join Prompt with one placeholder per table.
pub fn bind_join_prompt(
    template: &str,
    args: &[ValueExpression],
    turn: &Turn,
    layout: &PromptLayout,
) -> Result<Prompt, CompileError> {
    check_placeholders(template, args.len())?;
    if args.len() < 2 {
        return compile_error(
            "a join prompt needs at least two placeholders, one per joined table",
        );
    }
    if !distinct_aliases(args) {
        return compile_error(format!(
            "each join placeholder must name a distinct table (one document block per table), \
             got aliases {}; to mention a table's document again, use its marker in the question \
             text, not a second placeholder",
            alias_list(args)
        ));
    }
    let mut bound = prompt(template.to_string(), args, "join");
    bound.preamble = format!("{}{}", turn.0, layout.document_label);
    bound.layout = layout.name.to_string();
    if layout.record_label.is_some() {
        let names = record_names(args);
        let state_tail = format!(
            "{}{}",
            layout.question_separator.unwrap_or_default(),
            layout.question_label
        );
        let segments = vec![
            JOIN_QUESTION_SEP.to_string(),
            format!(
                "{}{}{}",
                name_records(template, &names),
                layout.question_end,
                turn.1
            ),
        ];
        bound.tail = segments.concat();
        bound.frame = state_tail.clone();
        bound.frame_text = state_tail.clone();
        bound.label_texts = args
            .iter()
            .zip(&names)
            .map(|(argument, name)| {
                (
                    argument_alias(argument),
                    layout.record(name),
                    state_tail.clone(),
                )
            })
            .collect();
        bound.tail_segments = segments;
        return Ok(bound);
    }
    let frame = render_join_frame(template, 0, layout)[join_anchor_note(0).len()..].to_string();
    let mut segments: Vec<String> = std::iter::once(layout.question_end.to_string())
        .chain(layout.answer_segments.iter().cloned())
        .collect();
    segments.last_mut().expect("a segment").push_str(&turn.1);
    bound.tail = segments.concat();
    bound.frame = frame.clone();
    bound.frame_text = frame;
    bound.label_texts = args
        .iter()
        .enumerate()
        .map(|(index, argument)| {
            (
                argument_alias(argument),
                join_label(index),
                render_join_frame(template, index, layout),
            )
        })
        .collect();
    bound.tail_segments = segments;
    Ok(bound)
}

/// Bind the prompt of one AI function call, as `PromptBinder.bind`.
///
/// # Errors
///
/// The template or arguments do not fit the call, or the kind is unknown.
#[allow(clippy::too_many_arguments)]
pub fn bind(
    kind: &str,
    template: &str,
    args: &[ValueExpression],
    labels: &[String],
    descriptions: &[String],
    task_description: &str,
    turn: &Turn,
    layout: &str,
) -> Result<Prompt, CompileError> {
    let layout = prompt_layout(layout)?;
    match kind {
        "extract" => bind_extract_prompt(args, template, turn, &layout),
        "map" => bind_map_prompt(args, template, turn),
        "features" => bind_sae_features_prompt(template, args, turn),
        "label" => bind_classify_prompt(
            template,
            args,
            labels,
            descriptions,
            turn,
            task_description,
            &layout,
            &[],
        ),
        "filter" => bind_prompt(template, args, turn, &layout),
        "join" => bind_join_prompt(template, args, turn, &layout),
        "score" => bind_score_prompt(template, args),
        other => compile_error(format!("unknown prompt kind {}", repr_str(other))),
    }
}
