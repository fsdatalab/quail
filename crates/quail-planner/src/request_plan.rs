//! The physical plan of a request backend: one `RequestExecution` node
//! that submits each document or pair to vLLM or SGLang, with the scans
//! before it and the result nodes after it, as
//! `quail/baselines/planning.py` builds it.
//!
//! The request node and its specs are written as the JSON their Python
//! codecs read, since only the request executor reads them.

use quail_physical::{Node, PhysicalNode, PortRef, ScanAttributes, input_ports, port};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::context::PlanningContext;
use crate::expressions::asked_call;
use crate::expressions::{CompileError, Refusal, aliased_call, argument_alias, call_aliases};
use crate::graph::has_label;
use crate::lowering::answer_ids;
use crate::operators::{Operators, SemanticColumn};
use crate::prompts::{LABEL_PREFIX, join_label, prompt_layout, render_join_frame};
use crate::request_order::request_order;
use crate::results::limit_node;
use crate::search::label_calls;
use crate::validation::columns_above_joins;

/// How a request backend submits its operators.
#[derive(Debug, Clone, Deserialize)]
pub struct RequestSettings {
    /// The backend the plan names, such as `stock_vllm`.
    pub backend_name: String,
    /// How filter stages become requests.
    pub filter_submission: String,
    /// How join tuples become requests.
    pub join_submission: String,
    /// How each document's extraction and map prompts become requests.
    pub answer_submission: String,
    /// Whether the engine decodes an answer as text, which AI.CLASSIFY
    /// and AI.EXTRACT need.
    pub scores_labels: bool,
}

const SPECS: &str = "quail.baselines.nodes";

/// Plan a plan's operators for a request backend.
///
/// Returns `{"graph": {"nodes", "root"}, "settings"}`, or
/// `{"refusal": ...}` when the backend cannot run the plan.
///
/// # Errors
///
/// A prompt cannot be lowered, or the prompts do not share a preamble.
pub fn request_plan(
    found: &Operators,
    context: &PlanningContext,
    settings: &RequestSettings,
) -> Result<Value, CompileError> {
    // a decision model scores one option block per label
    let decides = context.model.role == "decision";
    if let Some(refusal) = refusal(found, context, settings, decides)? {
        return Ok(json!({"refusal": {
            "reasons": refusal.reasons, "constraint": refusal.constraint,
            "needed": refusal.needed, "available": refusal.available, "unit": refusal.unit,
        }}));
    }
    let order = request_order(found, context)?;
    let mut nodes: Vec<Value> = Vec::new();
    let mut input_refs: Vec<PortRef> = Vec::new();
    let aliases: Vec<String> = found.scans.iter().map(|scan| scan.alias.clone()).collect();
    for scan in &found.scans {
        let table = &order.tables[&scan.alias];
        let id = format!("scan:{}", scan.alias);
        nodes.push(node_value(PhysicalNode::Scan(Node {
            id: id.clone(),
            inputs: Vec::new(),
            attributes: ScanAttributes {
                alias: scan.alias.clone(),
                input_id: scan.alias.clone(),
                num_documents: table.num_documents,
                total_tokens: table.total_tokens,
                predicates: scan.predicates.clone(),
                expected_rows: (!scan.predicates.is_empty()).then_some(table.kept_documents as f64),
            },
        }))?);
        input_refs.push(port(&id, format!("rows:{}", scan.alias)));
    }
    for hash_join in &order.hash_joins {
        if let PhysicalNode::HashJoin(join) = hash_join {
            input_refs.push(port(
                &join.id,
                format!("pairs:{}", join.attributes.written_position),
            ));
        }
        nodes.push(node_value(hash_join.clone())?);
    }

    let first_anchor = order
        .join_sequence
        .first()
        .map(|(_, anchor)| anchor.clone());
    let filters = filter_specs(context, &aliases, found, &order.filter_orders, first_anchor)?;
    let classifies = classify_specs(context, found, decides)?;
    let joins = join_specs(context, found, &order.join_sequence)?;
    let extracts = answer_specs(context, &found.extracts, "RequestExtractSpec")?;
    let maps = answer_specs(context, &found.maps, "RequestMapSpec")?;

    let mut prompts = Vec::new();
    for spec in &filters {
        let alias = spec["alias"].as_str().unwrap_or_default();
        if let Some((_, predicates)) = found.filters.iter().find(|(found, _)| found == alias) {
            prompts.extend(
                predicates
                    .iter()
                    .map(|predicate| asked_call(&predicate.expression).prompt.as_ref().clone()),
            );
        }
    }
    prompts.extend(
        label_calls(found)
            .into_iter()
            .map(|(call, _)| call.prompt.as_ref().clone()),
    );
    for (position, _) in &order.join_sequence {
        prompts.push(
            asked_call(&found.joins[*position].predicate)
                .prompt
                .as_ref()
                .clone(),
        );
    }
    let (preamble, preamble_text) = request_preamble(context, &prompts)?;

    let request_id = "request-model";
    nodes.push(json!({
        "type": "quail.request_execution",
        "id": request_id,
        "inputs": serde_json::to_value(input_ports(&input_refs)).map_err(json_error)?,
        "attributes": {
            "backend_name": settings.backend_name,
            "aliases": aliases,
            "preamble_token_ids": preamble,
            "preamble_text": preamble_text,
            "filters": filters, "joins": joins, "classifies": classifies,
            "extracts": extracts, "maps": maps,
        },
    }));
    let (results, root) = result_nodes(found, request_id, &aliases, &joins);
    for node in results {
        nodes.push(node_value(node)?);
    }

    let (true_ids, false_ids) = answer_ids(&context.prompt_model(), context.tokenizer)?;
    let mut plan_settings = json!({
        "filter_submission": settings.filter_submission,
        "join_submission": settings.join_submission,
        "answer_submission": settings.answer_submission,
        "true_ids": true_ids,
        "false_ids": false_ids,
        "order_rule": order.rule,
    });
    if decides {
        plan_settings["decision_offsets"] = json!(crate::lowering::answer_row_offsets(
            &context.prompt_model(),
            context.tokenizer
        )?);
    }
    Ok(json!({
        "graph": {"nodes": nodes, "root": serde_json::to_value(root).map_err(json_error)?},
        "settings": plan_settings,
    }))
}

fn json_error(error: serde_json::Error) -> CompileError {
    CompileError::new(error.to_string())
}

fn node_value(node: PhysicalNode) -> Result<Value, CompileError> {
    serde_json::to_value(node).map_err(json_error)
}

/// A spec of the request node, tagged with its Python dataclass.
fn spec(name: &str, mut fields: Value) -> Value {
    fields["type"] = Value::from(format!("{SPECS}.{name}"));
    fields
}

fn refuse(reason: String, constraint: String, unit: &str) -> Option<Refusal> {
    Some(Refusal {
        reasons: vec![reason],
        constraint,
        needed: 1.0,
        available: 0.0,
        unit: unit.into(),
    })
}

/// Why the model cannot write AI.EXTRACT or AI.MAP answers, or None.
fn extraction_refusal(
    context: &PlanningContext,
    function: &str,
) -> Result<Option<Refusal>, CompileError> {
    let model = &context.model;
    let reason =
        if matches!(model.role.as_str(), "reranker" | "decision") || model.canvas_tokens != 0 {
            format!(
                "{function} decodes the model's next tokens, which {} does not give",
                py_repr(&model.name)
            )
        } else if prompt_layout(&model.prompt_layout)?.read_end.is_some() {
            // the model answers by scoring options and document positions
            if function == "AI.EXTRACT" {
                return Ok(None);
            }
            format!(
                "{function} writes text, and {} answers only by scoring options",
                py_repr(&model.name)
            )
        } else {
            return Ok(None);
        };
    let kind = function.trim_start_matches("AI.").to_lowercase();
    Ok(refuse(
        reason,
        format!("unsupported_{kind}_query"),
        "queries",
    ))
}

/// Why the request backends cannot run the plan, or None.
fn refusal(
    found: &Operators,
    context: &PlanningContext,
    settings: &RequestSettings,
    decides: bool,
) -> Result<Option<Refusal>, CompileError> {
    let backend = &settings.backend_name;
    let tests = || found.column_filters.iter().flat_map(|(_, tests)| tests);
    if tests().any(|test| test.call.kind == "label" && test.labels.is_none()) {
        return Ok(refuse(
            "plain label tests other than IN run on the Quail backend only".into(),
            "label_test_needs_quail_backend".into(),
            "backends",
        ));
    }
    if found.columns().any(|column| column.guard.is_some()) {
        return Ok(refuse(
            format!("a guarded AI column asks only some rows, which {backend} does not"),
            "guard_needs_quail_backend".into(),
            "backends",
        ));
    }
    let ored = found.filters.iter().flat_map(|(_, predicates)| predicates);
    if ored.clone().any(|predicate| {
        predicate.guard.is_some() || predicate.alternative.is_some() || predicate.answer.is_some()
    }) {
        return Ok(refuse(
            format!("an AI filter in an OR runs on the Quail backend only, not {backend}"),
            "or_filter_needs_quail_backend".into(),
            "backends",
        ));
    }
    let has_extract = !found.extracts.is_empty();
    for (function, present) in [
        ("AI.EXTRACT", has_extract),
        ("AI.MAP", !found.maps.is_empty()),
    ] {
        if !present {
            continue;
        }
        let kind = function[3..].to_lowercase();
        let mut refusal = extraction_refusal(context, function)?;
        if refusal.is_none() && !(settings.scores_labels && context.model.canvas_tokens == 0) {
            refusal = refuse(
                format!(
                    "{function} on {backend} decodes each answer as text, which its engine \
                     does not return for {}",
                    py_repr(&context.model.name)
                ),
                format!("{kind}_needs_quail_backend"),
                "backends",
            );
        }
        if refusal.is_none() && tests().any(|test| test.call.kind == kind) {
            refusal = refuse(
                format!(
                    "a filter on an {function} answer runs on the Quail backend only, not \
                     {backend}"
                ),
                format!("{kind}_filter_needs_quail_backend"),
                "backends",
            );
        }
        if refusal.is_some() {
            return Ok(refusal);
        }
    }
    if found.projection.columns.iter().any(|column| match column {
        quail_physical::Projected::Expression(expression) => {
            aliased_call(expression, &[]).is_some_and(|call| call.probabilities)
        }
        _ => false,
    }) {
        return Ok(refuse(
            format!(
                "AI.CLASSIFY on {backend} decodes the label as text and has no probability for \
                 the others"
            ),
            "classify_probabilities_need_quail_backend".into(),
            "backends",
        ));
    }
    // a diffusion model's vLLM canvas is sized for a filter's one answer
    // token, not a label
    let scores_labels = settings.scores_labels && context.model.canvas_tokens == 0;
    if has_label(found) && !(scores_labels || decides) {
        return Ok(refuse(
            format!(
                "AI.CLASSIFY on {backend} decodes each answer as text, which its engine does not \
                 return for {}",
                py_repr(&context.model.name)
            ),
            "classify_needs_quail_backend".into(),
            "backends",
        ));
    }
    if columns_above_joins(found)
        .iter()
        .any(|(_, kind)| kind == "label")
    {
        return Ok(refuse(
            format!("{backend} runs single-document classification before joins"),
            "classify_placement_not_supported".into(),
            "placements",
        ));
    }
    Ok(None)
}

/// One filter spec per table with AI.IF filters, the first join's anchor
/// last.
fn filter_specs(
    context: &PlanningContext,
    aliases: &[String],
    found: &Operators,
    filter_orders: &std::collections::BTreeMap<String, Vec<usize>>,
    first_anchor: Option<String>,
) -> Result<Vec<Value>, CompileError> {
    let mut ordered: Vec<&String> = aliases.iter().collect();
    ordered.sort_by_key(|alias| Some((*alias).clone()) == first_anchor);
    let mut specs = Vec::new();
    for alias in ordered {
        let Some((_, predicates)) = found
            .filters
            .iter()
            .find(|(found, predicates)| found == alias && !predicates.is_empty())
        else {
            continue;
        };
        let positions = filter_orders.get(alias).cloned().unwrap_or_default();
        let mut questions = Vec::new();
        let mut texts = Vec::new();
        for position in &positions {
            let prompt = asked_call(&predicates[*position].expression)
                .prompt
                .as_ref();
            let question = context.lowered(prompt)?.tail_token_ids;
            if question.is_empty() {
                return Err(CompileError::new("filter prompts have no token ids"));
            }
            questions.push(question);
            if prompt.args.len() != 1 || !prompt.tail.starts_with("{0}") {
                return Err(CompileError::new(
                    "a filter prompt must start its tail with {0}",
                ));
            }
            texts.push(prompt.tail["{0}".len()..].to_string());
        }
        specs.push(spec(
            "RequestFilterSpec",
            json!({
                "alias": alias, "written_positions": positions,
                "question_token_ids": questions, "question_texts": texts,
            }),
        ));
    }
    Ok(specs)
}

/// One classify spec per AI.CLASSIFY call.
fn classify_specs(
    context: &PlanningContext,
    found: &Operators,
    decides: bool,
) -> Result<Vec<Value>, CompileError> {
    let mut specs = Vec::new();
    for (call, alias) in label_calls(found) {
        let call_aliases_found = call_aliases(&call);
        let lowered = context.lowered(&call.prompt)?;
        let mut pair = json!({"partner": null, "join_layout_token_ids": [],
            "join_written_position": null});
        if call_aliases_found.len() == 2 {
            // a classification of the rows a join keeps: decoded after that
            // join, one request per row
            let partner = call_aliases_found[1].clone();
            let part = |wanted: &str| {
                lowered
                    .label_token_ids
                    .iter()
                    .find(|(found, _, _)| found == wanted)
                    .map(|(_, label, frame)| (label.clone(), frame.clone()))
                    .unwrap_or_default()
            };
            let mut wanted = call_aliases_found.clone();
            wanted.sort();
            let position = found.joins.iter().position(|join| {
                let mut found = call_aliases(asked_call(&join.predicate));
                found.sort();
                found == wanted
            });
            let Some(position) = position else {
                return Err(CompileError::new(format!(
                    "the classification of {} x {} pairs needs a join of the two",
                    py_repr(&alias),
                    py_repr(&partner)
                )));
            };
            pair = json!({
                "partner": partner,
                "join_layout_token_ids": [part(&alias).1, part(&partner).0],
                "join_written_position": position,
            });
        }
        let name = found
            .classifies
            .iter()
            .find(|column| column.call == call)
            .map(|column| column.name.clone())
            .ok_or_else(|| CompileError::new("a classification has no SemanticClassify"))?;
        let labels: Vec<String> = call
            .labels
            .as_array()
            .map(|labels| {
                labels
                    .iter()
                    .map(|label| label.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let label_token_ids = labels
            .iter()
            .map(|label| (context.tokenizer)(&format!("{LABEL_PREFIX}{label}")))
            .collect::<Result<Vec<_>, _>>()?;
        let tests: Vec<Value> = found
            .column_filters
            .iter()
            .filter(|(found, _)| *found == alias)
            .flat_map(|(_, tests)| tests)
            .filter(|test| test.call == call)
            .map(|test| json!([test.position, test.labels]))
            .collect();
        let option_offsets = if decides {
            let lengths = call
                .prompt
                .tail_segments
                .iter()
                .map(|segment| (context.tokenizer)(segment).map(|ids| ids.len()))
                .collect::<Result<Vec<_>, _>>()?;
            let mut offsets: Vec<usize> = (1..lengths.len().saturating_sub(1))
                .map(|index| lengths[index + 1..].iter().sum())
                .collect();
            offsets.push(0);
            offsets
        } else {
            Vec::new()
        };
        let mut fields = json!({
            "alias": alias, "output": name, "tail_token_ids": lowered.tail_token_ids,
            "labels": labels, "label_token_ids": label_token_ids, "tests": tests,
            "option_offsets": option_offsets,
        });
        for (key, value) in pair.as_object().cloned().unwrap_or_default() {
            fields[key] = value;
        }
        specs.push(spec("RequestClassifySpec", fields));
    }
    Ok(specs)
}

/// One join spec per join, in the searched sequence.
fn join_specs(
    context: &PlanningContext,
    found: &Operators,
    sequence: &[(usize, String)],
) -> Result<Vec<Value>, CompileError> {
    let mut specs = Vec::new();
    for (position, anchor) in sequence {
        let join = &found.joins[*position];
        let lowered = context.lowered(&asked_call(&join.predicate).prompt)?;
        let prompt = &lowered.prompt;
        let aliases: Vec<String> = prompt.args.iter().map(argument_alias).collect();
        if lowered.label_token_ids.len() != aliases.len() || lowered.tail_token_ids.is_empty() {
            return Err(CompileError::new("join prompts have no token ids"));
        }
        let layout = prompt_layout("ai-if")?;
        let labels: Vec<Value> = lowered
            .label_token_ids
            .iter()
            .map(|(alias, label, _)| json!([alias, label]))
            .collect();
        let frames: Vec<Value> = lowered
            .label_token_ids
            .iter()
            .map(|(alias, _, frame)| json!([alias, frame]))
            .collect();
        let label_texts: Vec<Value> = aliases
            .iter()
            .enumerate()
            .map(|(index, alias)| json!([alias, join_label(index)]))
            .collect();
        let frame_texts: Vec<Value> = aliases
            .iter()
            .enumerate()
            .map(|(index, alias)| {
                json!([alias, render_join_frame(&prompt.template, index, &layout)])
            })
            .collect();
        specs.push(spec(
            "RequestJoinSpec",
            json!({
                "written_position": position, "aliases": aliases,
                "outer_aliases": join.outer_aliases, "anchor": anchor,
                "semantics": join.semantics, "selectivity": join.selectivity,
                "label_token_ids": labels, "frame_token_ids": frames,
                "tail_token_ids": lowered.tail_token_ids, "label_texts": label_texts,
                "frame_texts": frame_texts, "tail_text": prompt.tail,
            }),
        ));
    }
    Ok(specs)
}

/// One answer spec per AI.EXTRACT field or AI.MAP column, each decoded
/// once per surviving document after the joins; its prompt carries its
/// own preamble, the model's turn, which stays out of the shared one.
fn answer_specs(
    context: &PlanningContext,
    columns: &[SemanticColumn],
    name: &str,
) -> Result<Vec<Value>, CompileError> {
    let mut specs = Vec::new();
    for column in columns {
        let call = &column.call;
        let lowered = context.lowered(&call.prompt)?;
        let mut fields = json!({
            "alias": call_aliases(call)[0], "output": column.name,
            "tail_token_ids": lowered.tail_token_ids,
            "tail_text": strip_placeholders(&call.prompt.tail),
            "preamble_token_ids": lowered.preamble_token_ids,
            // vLLM adds the model's beginning-of-text token itself
            "preamble_text": without_bos(&call.prompt.preamble, &context.model.bos_text),
            "max_answer_tokens": 64,
        });
        if name == "RequestExtractSpec" {
            fields["question"] = Value::from(call.question.clone());
        } else {
            fields["max_answer_tokens"] = Value::from(call.max_answer_tokens);
        }
        specs.push(spec(name, fields));
    }
    Ok(specs)
}

/// The text with every `{N}` placeholder removed.
fn strip_placeholders(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
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

fn without_bos(text: &str, bos_text: &str) -> String {
    if !bos_text.is_empty() && text.starts_with(bos_text) {
        text[bos_text.len()..].to_string()
    } else {
        text.to_string()
    }
}

/// The one preamble the prompts share, as token ids and text.
fn request_preamble(
    context: &PlanningContext,
    prompts: &[quail_physical::Prompt],
) -> Result<(Vec<i64>, String), CompileError> {
    let mut preambles: Vec<Vec<i64>> = Vec::new();
    for prompt in prompts {
        let ids = context.lowered(prompt)?.preamble_token_ids;
        if !ids.is_empty() && !preambles.contains(&ids) {
            preambles.push(ids);
        }
    }
    if preambles.len() > 1 {
        return Err(CompileError::new(
            "request prompts have different preambles",
        ));
    }
    let mut texts: Vec<&String> = Vec::new();
    for prompt in prompts {
        if !texts.contains(&&prompt.preamble) {
            texts.push(&prompt.preamble);
        }
    }
    if texts.len() > 1 {
        return Err(CompileError::new(
            "request prompts have different preamble text",
        ));
    }
    let shared = format!(
        "{}{}",
        context.model.turn_prefix,
        prompt_layout("ai-if")?.document_label
    );
    let mut preamble = preambles.into_iter().next().unwrap_or_default();
    if preamble.is_empty() && !prompts.is_empty() {
        preamble = (context.tokenizer)(&shared)?;
    }
    let text = texts
        .first()
        .filter(|text| !text.is_empty())
        .map_or(shared.clone(), |text| (*text).clone());
    // vLLM adds the model's beginning-of-text token to a text prompt
    Ok((preamble, without_bos(&text, &context.model.bos_text)))
}

/// The Limit of the region's fetch, reading the request node's result
/// rows, and the port that ends the plan.
fn result_nodes(
    found: &Operators,
    request: &str,
    aliases: &[String],
    joins: &[Value],
) -> (Vec<PhysicalNode>, PortRef) {
    let full = joins.iter().find(|spec| spec["semantics"] == "full");
    let rows = match full {
        Some(spec) => port(
            request,
            format!(
                "join_answers:{}",
                spec["written_position"].as_i64().unwrap_or_default()
            ),
        ),
        None => port(request, format!("rows:{}", aliases[0])),
    };
    match limit_node(found, &rows) {
        Some(limit) => (vec![limit], port("limit", "rows")),
        None => (Vec::new(), rows),
    }
}

/// A string as Python's repr writes it.
fn py_repr(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}
