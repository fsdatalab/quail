//! Rules that change executable graphs, as `quail/planner/rules/physical.py`
//! defines them, and the loop that applies them to a plan, as
//! `apply_plan_rules`.

use std::collections::{BTreeMap, BTreeSet};

use quail_cost::budgets::{cheaper_attention_path, chunk_budget, tree_attention_allowed};
use quail_cost::packing::choose_pass_groups;
use quail_cost::retention::coefficients;
use quail_cost::statistics::AliasStats;
use quail_physical::{PhysicalNode, PortRef, build_pipelines, topological_nodes};
use serde_json::{Map, Value};

use crate::classify::classify_statistics;
use crate::context::PlanningContext;
use crate::expressions::CompileError;
use crate::graph::{PhysicalPlan, Planned};
use crate::plan_statistics::{expected_rows_after_filters, plan_statistics};
use crate::pricing::{expected_documents, head_tokens, pass_question_tokens, set_pass_tokens};
use crate::retention::schedule;

/// The built-in physical rules, in the order the registry applies them.
pub const PHYSICAL_RULES: [&str; 8] = [
    "should_push_down_limit",
    "choose_classify_implementation",
    "pack_set_passes",
    "join_document_ends",
    "should_retain_kv",
    "should_share_prefixes",
    "choose_attention_path",
    "should_deduplicate",
];

/// An executable graph: its nodes and the port that returns the result.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub nodes: Vec<PhysicalNode>,
    pub root: PortRef,
}

impl Graph {
    fn with_nodes(&self, nodes: Vec<PhysicalNode>) -> Graph {
        Graph {
            nodes,
            root: self.root.clone(),
        }
    }
}

type Rewrite = Result<Option<Graph>, CompileError>;

fn should_push_down_limit(graph: &Graph, settings: &mut Map<String, Value>) -> Rewrite {
    let limit = graph.nodes.iter().find_map(|node| match node {
        PhysicalNode::Limit(limit) => Some(limit.attributes.count),
        _ => None,
    });
    let pipelines = build_pipelines(&graph.nodes, &[]);
    let blocked = pipelines.members.len() > 1
        || graph
            .nodes
            .iter()
            .any(|node| matches!(node, PhysicalNode::Sort(_) | PhysicalNode::AiJoin(_)));
    match limit {
        Some(count) if !blocked => {
            settings.insert("filter_limit".into(), Value::from(count));
        }
        _ => {
            settings.remove("filter_limit");
        }
    }
    Ok(None)
}

fn choose_classify_implementation(
    graph: &Graph,
    context: &PlanningContext,
    settings: &mut Map<String, Value>,
) -> Rewrite {
    let pending: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| match node {
            PhysicalNode::AiClassify(classify) => classify
                .attributes
                .spec
                .as_ref()
                .is_some_and(|spec| spec.scoring.is_empty()),
            _ => false,
        })
        .map(|(index, _)| index)
        .collect();
    if pending.is_empty() {
        return Ok(None);
    }
    let Some(found) = &context.operators else {
        return Err(CompileError::new(
            "choose_classify_implementation needs the plan's operators on the planning context \
             to score a classification without a scoring rule",
        ));
    };
    let paths = build_pipelines(&graph.nodes, &[]);
    let mut nodes = graph.nodes.clone();
    for index in pending {
        let PhysicalNode::AiClassify(classify) = &graph.nodes[index] else {
            unreachable!()
        };
        let spec = classify.attributes.spec.as_ref().expect("a spec");
        let call = found
            .classifies
            .iter()
            .rev()
            .find(|column| column.name == spec.name)
            .map(|column| &column.call)
            .expect("the classification's call");
        let table = classify_statistics(context, &spec.aliases[0], &[])?;
        let resident = paths.get(&classify.id).expect("a pipeline")[0] != index;
        let chosen = table.choose_scoring(spec, call, resident, context.tokenizer)?;
        if settings.contains_key("estimated_fresh_tokens") {
            let work = table.estimate_spec(&chosen, resident)?.work;
            for (key, value) in [
                ("estimated_fresh_tokens", work.tokens.total()),
                ("estimated_attention_pairs", work.pairs.total()),
            ] {
                let before = settings[key].as_f64().unwrap_or(0.0);
                settings.insert(key.into(), Value::from(before + value));
            }
        }
        let mut replaced = classify.clone();
        replaced.attributes.spec = Some(chosen);
        nodes[index] = PhysicalNode::AiClassify(replaced);
    }
    Ok(Some(graph.with_nodes(nodes)))
}

/// The kind of question a node asks in a set-attention pass, if any.
fn packed_kind(node: &PhysicalNode) -> Option<&'static str> {
    match node {
        PhysicalNode::AiFilter(_) => Some("filter"),
        PhysicalNode::AiClassify(_) if node.spec_aliases().len() != 2 => Some("label"),
        PhysicalNode::AiExtract(_) => Some("extract"),
        PhysicalNode::AiScore(_) if !node.document_aliases().is_empty() => Some("score"),
        _ => None,
    }
}

/// The pipeline's runs of packable operators, in order, as `chains`.
fn chains(nodes: &[PhysicalNode], pipeline: &[usize]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    for member in pipeline {
        let node = &nodes[*member];
        if packed_kind(node).is_some() {
            current.push(*member);
        } else if !matches!(node, PhysicalNode::Filter(filter) if filter.attributes.aliases.len() == 1)
            && !current.is_empty()
        {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Each packable operator's pass group, numbered across the plan, as
/// `set_pass_groups`.
fn set_pass_groups(
    nodes: &[PhysicalNode],
    context: &PlanningContext,
) -> Result<BTreeMap<String, i64>, CompileError> {
    let model = &context.model;
    let (state_tail, header) = set_pass_tokens(context)?;
    let chunk = chunk_budget(model, &context.device);
    let pipelines = build_pipelines(nodes, &[]);
    let mut groups = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut first = 0;
    for index in topological_nodes(nodes) {
        let Some(&pipeline) = pipelines.of.get(nodes[index].id()) else {
            continue;
        };
        if !seen.insert(pipeline) {
            continue;
        }
        for chain in chains(nodes, &pipelines.members[pipeline]) {
            let alias = nodes[chain[0]].document_aliases()[0].clone();
            let summary = AliasStats::summarize(context.lengths(&alias)?, model.sliding_window);
            let state = (head_tokens(&nodes[chain[0]]) + state_tail) as f64;
            let rows: Vec<f64> = chain
                .iter()
                .map(|member| expected_documents(&nodes[*member]))
                .collect();
            let mut kept: Vec<f64> = rows
                .windows(2)
                .map(|pair| {
                    if pair[0] != 0.0 {
                        pair[1] / pair[0]
                    } else {
                        1.0
                    }
                })
                .collect();
            kept.push(1.0);
            let questions: Vec<i64> = chain
                .iter()
                .map(|member| pass_question_tokens(&nodes[*member], state_tail))
                .collect();
            let chosen = choose_pass_groups(
                &questions,
                &kept,
                rows[0],
                state + summary.mean(),
                state as i64 + summary.maximum,
                model,
                &context.device,
                chunk,
                model.set_size as usize,
                header,
            )
            .map_err(CompileError::new)?;
            for (member, group) in chain.iter().zip(&chosen) {
                groups.insert(nodes[*member].id().to_string(), first + *group as i64);
            }
            first += *chosen.iter().max().expect("a group") as i64 + 1;
        }
    }
    Ok(groups)
}

fn set_pass_group(node: &mut PhysicalNode, group: i64) {
    match node {
        PhysicalNode::AiFilter(node) => node.attributes.pass_group = group,
        PhysicalNode::AiScore(node) => node.attributes.pass_group = group,
        PhysicalNode::AiClassify(node) => node.attributes.pass_group = group,
        PhysicalNode::AiExtract(node) => node.attributes.pass_group = group,
        PhysicalNode::AiMap(node) => node.attributes.pass_group = group,
        PhysicalNode::AiSaeFeatures(node) => node.attributes.pass_group = group,
        _ => {}
    }
}

fn pack_set_passes(graph: &Graph, context: &PlanningContext) -> Rewrite {
    if context.model.attention_mask != "set"
        || !graph
            .nodes
            .iter()
            .any(|node| packed_kind(node).is_some() && node.pass_group() < 0)
    {
        return Ok(None);
    }
    let groups = set_pass_groups(&graph.nodes, context)?;
    let mut nodes = graph.nodes.clone();
    for node in &mut nodes {
        if let Some(group) = groups.get(node.id()) {
            set_pass_group(node, *group);
        }
    }
    Ok(Some(graph.with_nodes(nodes)))
}

fn tokens_of(values: &[Value]) -> Vec<i64> {
    values.iter().filter_map(Value::as_i64).collect()
}

/// The tables a set-attention node reads and the tokens after their
/// documents, as `document_tails`: (alias, head, tail) per table.
fn document_tails(node: &PhysicalNode) -> Vec<(String, Vec<i64>, Vec<i64>)> {
    match node {
        PhysicalNode::AiFilter(filter) => {
            let attributes = &filter.attributes;
            let cut =
                (attributes.state_tail_tokens as usize).min(attributes.question_token_ids.len());
            vec![(
                attributes.alias.clone(),
                attributes.document_head.clone(),
                attributes.question_token_ids[..cut].to_vec(),
            )]
        }
        PhysicalNode::AiJoin(join) => {
            let attributes = &join.attributes;
            let cut = (attributes.separator_tokens as usize).min(attributes.tail_token_ids.len());
            let separator = attributes.tail_token_ids[..cut].to_vec();
            let mut out = vec![(
                attributes.anchor.clone(),
                attributes.document_head.clone(),
                attributes.frame_token_ids.clone(),
            )];
            out.extend(
                attributes
                    .partners
                    .iter()
                    .map(|partner| (partner.clone(), Vec::new(), separator.clone())),
            );
            out
        }
        PhysicalNode::AiClassify(classify) if node.spec_aliases().len() == 2 => {
            let spec = classify.attributes.spec.as_ref().expect("a spec");
            let (head, tail) = (&spec.prompt_token_parts[0], &spec.prompt_token_parts[1]);
            let cut = (spec.separator_tokens as usize).min(tail.len());
            let frame = spec
                .join_layout
                .as_ref()
                .map(|layout| tokens_of(&layout.0))
                .unwrap_or_default();
            vec![
                (spec.aliases[0].clone(), head.clone(), frame),
                (spec.aliases[1].clone(), Vec::new(), tail[..cut].to_vec()),
            ]
        }
        _ if packed_kind(node).is_some() => {
            let parts = node.prompt_token_parts();
            let state_tail = match node {
                PhysicalNode::AiScore(score) => score
                    .attributes
                    .spec
                    .as_ref()
                    .map_or(0, |s| s.state_tail_tokens),
                PhysicalNode::AiClassify(classify) => classify
                    .attributes
                    .spec
                    .as_ref()
                    .map_or(0, |s| s.state_tail_tokens),
                PhysicalNode::AiExtract(extract) => extract
                    .attributes
                    .spec
                    .as_ref()
                    .map_or(0, |s| s.state_tail_tokens),
                _ => 0,
            };
            let cut = (state_tail as usize).min(parts[1].len());
            vec![(
                node.spec_aliases()[0].clone(),
                parts[0].clone(),
                parts[1][..cut].to_vec(),
            )]
        }
        _ => Vec::new(),
    }
}

fn document_ends_of(node: &PhysicalNode) -> Option<&Value> {
    match node {
        PhysicalNode::AiFilter(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiJoin(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiScore(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiClassify(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiExtract(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiMap(node) => Some(&node.attributes.document_ends),
        PhysicalNode::AiSaeFeatures(node) => Some(&node.attributes.document_ends),
        _ => None,
    }
}

fn set_document_ends(node: &mut PhysicalNode, ends: Value) {
    match node {
        PhysicalNode::AiFilter(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiJoin(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiScore(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiClassify(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiExtract(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiMap(node) => node.attributes.document_ends = ends,
        PhysicalNode::AiSaeFeatures(node) => node.attributes.document_ends = ends,
        _ => {}
    }
}

/// Each last token's tokens when the tokenizer reads it with the tail,
/// as `document_ends`.
fn document_ends(
    last: &[i64],
    tail: &[i64],
    context: &PlanningContext,
) -> Result<Value, CompileError> {
    let decode = context.decode.expect("a decoding tokenizer");
    let mut out = Vec::new();
    for token in last {
        let mut ids = vec![*token];
        ids.extend_from_slice(tail);
        let text = decode(&ids)?;
        let mut joined = (context.tokenizer)(&text)?;
        if decode(&joined)? != text {
            joined = ids;
        }
        out.push(Value::Array(vec![Value::from(*token), Value::from(joined)]));
    }
    Ok(Value::Array(out))
}

/// A table a node reads, its prompt head, and the tokens after its documents.
type Tail = (String, Vec<i64>, Vec<i64>);

fn join_document_ends(graph: &Graph, context: &PlanningContext) -> Rewrite {
    if context.model.attention_mask != "set" || context.decode.is_none() {
        return Ok(None);
    }
    let mut joins: Vec<(Tail, Option<Value>)> = Vec::new();
    let mut nodes = Vec::new();
    let mut changed = false;
    for node in &graph.nodes {
        let tails = document_tails(node);
        let annotated = document_ends_of(node)
            .is_some_and(|ends| ends.as_array().is_some_and(|items| !items.is_empty()));
        if tails.is_empty() || annotated {
            nodes.push(node.clone());
            continue;
        }
        let mut entries = Vec::new();
        for key in tails {
            if !joins.iter().any(|(found, _)| *found == key) {
                let (alias, head, tail) = &key;
                let joined = match context.last_tokens.get(alias) {
                    None => None,
                    Some(last) => {
                        let mut last: Vec<i64> = last.clone();
                        last.extend(head.last());
                        last.sort();
                        last.dedup();
                        Some(document_ends(&last, tail, context)?)
                    }
                };
                joins.push((key.clone(), joined));
            }
            let joined = &joins
                .iter()
                .find(|(found, _)| *found == key)
                .expect("joined")
                .1;
            if let Some(joined) = joined {
                entries.push(Value::Array(vec![
                    Value::from(key.0.clone()),
                    Value::from(key.2.clone()),
                    joined.clone(),
                ]));
            }
        }
        let mut node = node.clone();
        if !entries.is_empty() {
            set_document_ends(&mut node, Value::Array(entries));
            changed = true;
        }
        nodes.push(node);
    }
    Ok(changed.then(|| graph.with_nodes(nodes)))
}

/// Set operators' KV retention from the next compatible document consumer,
/// as `plan_kv_retention`.
pub fn plan_kv_retention(graph: &Graph) -> Graph {
    let nodes = &graph.nodes;
    let paths = build_pipelines(nodes, &[]);
    let starts: Vec<usize> = topological_nodes(nodes)
        .into_iter()
        .filter(|index| {
            paths
                .get(nodes[*index].id())
                .is_some_and(|members| members[0] == *index)
        })
        .collect();
    let mut replaced = nodes.clone();
    for (position, first) in starts.iter().enumerate() {
        let members = paths.get(nodes[*first].id()).expect("a pipeline");
        let alias = nodes[*first].document_aliases();
        let following = starts[position + 1..]
            .iter()
            .find(|later| nodes[**later].document_aliases() == alias);
        let keep = following
            .is_some_and(|later| nodes[*first].document_head() == nodes[*later].document_head());
        let model_nodes: Vec<usize> = members
            .iter()
            .copied()
            .filter(|member| nodes[*member].backend().is_some())
            .collect();
        // joins settle anchors after any classification of their surviving pairs
        let settling = model_nodes
            .iter()
            .rev()
            .find(|member| matches!(nodes[**member], PhysicalNode::AiJoin(_)))
            .copied()
            .unwrap_or(*model_nodes.last().expect("a model node"));
        for (place, member) in model_nodes.iter().enumerate() {
            let mut retain = keep && *member == settling;
            if matches!(nodes[*member], PhysicalNode::AiJoin(_)) {
                retain |= model_nodes[place + 1..]
                    .iter()
                    .any(|later| matches!(nodes[*later], PhysicalNode::AiJoin(_)));
            }
            match &mut replaced[*member] {
                PhysicalNode::AiFilter(filter) => {
                    filter.attributes.keep_kv = retain;
                    filter.attributes.arena_writes =
                        filter.attributes.arena_writes || retain || place < model_nodes.len() - 1;
                }
                PhysicalNode::AiJoin(node) => node.attributes.keep_kv = retain,
                PhysicalNode::AiScore(node) => node.attributes.keep_kv = retain,
                PhysicalNode::AiClassify(node) => node.attributes.keep_kv = retain,
                PhysicalNode::AiExtract(node) => node.attributes.keep_kv = retain,
                PhysicalNode::AiMap(node) => node.attributes.keep_kv = retain,
                PhysicalNode::AiSaeFeatures(node) => node.attributes.keep_kv = retain,
                _ => {}
            }
        }
    }
    graph.with_nodes(replaced)
}

fn should_retain_kv(
    graph: &Graph,
    context: &PlanningContext,
    settings: &mut Map<String, Value>,
) -> Rewrite {
    let Some(found) = &context.operators else {
        return Ok(None);
    };
    if settings.contains_key("retention") || !context.model.reuses_kv() {
        return Ok(None);
    }
    let statistics = plan_statistics(found, context)?;
    let joins: Vec<&quail_physical::Node<quail_physical::AiJoinAttributes>> = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::AiJoin(join) => Some(join),
            _ => None,
        })
        .collect();
    let mut grouped: Vec<Vec<usize>> = Vec::new();
    for (index, join) in joins.iter().enumerate() {
        let opens = match grouped.last() {
            None => true,
            Some(group) => {
                let last = &joins[*group.last().expect("a member")].attributes;
                last.anchor != join.attributes.anchor
                    || last.semantics != "full"
                    || join.attributes.semantics != "full"
            }
        };
        if opens {
            grouped.push(Vec::new());
        }
        grouped.last_mut().expect("a group").push(index);
    }
    let sequence: Vec<(quail_cost::joins::PricedJoin, String)> = grouped
        .iter()
        .flatten()
        .map(|index| {
            let attributes = &joins[*index].attributes;
            (
                statistics.specs[attributes.written_position as usize].clone(),
                attributes.anchor.clone(),
            )
        })
        .collect();
    let ids: Vec<String> = grouped
        .iter()
        .map(|group| joins[group[0]].id.clone())
        .collect();
    let mut planned = schedule(
        &sequence,
        &expected_rows_after_filters(found, &statistics),
        Some(&ids),
    );
    let before = planned["before"].clone();
    let after = planned["after"].clone();
    let mut node_before = Map::new();
    let mut node_after = Map::new();
    for group in &grouped {
        let first = &joins[group[0]].id;
        for (place, index) in group.iter().enumerate() {
            let id = joins[*index].id.clone();
            node_before.insert(id.clone(), before[first].clone());
            node_after.insert(
                id,
                if place == group.len() - 1 {
                    after[first].clone()
                } else {
                    before[first].clone()
                },
            );
        }
    }
    planned["before"] = Value::Object(node_before);
    planned["after"] = Value::Object(node_after);
    let paths = build_pipelines(&graph.nodes, &[]);
    let streamed: BTreeSet<String> = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::AiFilter(filter) => {
                let joined = paths.get(&filter.id).is_some_and(|members| {
                    members
                        .iter()
                        .any(|member| matches!(graph.nodes[*member], PhysicalNode::AiJoin(_)))
                });
                joined.then(|| filter.attributes.alias.clone())
            }
            _ => None,
        })
        .collect();
    if let Value::Object(initial) = &mut planned["initial"] {
        initial.retain(|alias, _| !streamed.contains(alias));
    }
    let found = coefficients(&context.model, &context.device, false);
    let object = planned.as_object_mut().expect("a schedule");
    object.insert("linear_seconds".into(), Value::from(found.linear_seconds));
    object.insert("pair_seconds".into(), Value::from(found.pair_seconds));
    object.insert(
        "sliding_pair_seconds".into(),
        Value::from(found.sliding_pair_seconds),
    );
    object.insert("sliding_window".into(), Value::from(found.sliding_window));
    if let Some(pages) = found.checkpoint_pages {
        object.insert("checkpoint_pages".into(), Value::from(pages));
    }
    object.insert("cap_pages".into(), Value::from(statistics.cap_pages));
    settings.insert("retention".into(), planned);
    Ok(Some(plan_kv_retention(graph)))
}

/// Whether a node borrows its documents' shared prefix KV.
fn shares_prefixes(node: &PhysicalNode) -> bool {
    match node {
        PhysicalNode::AiFilter(node) => node.attributes.share_prefixes,
        PhysicalNode::AiJoin(node) => node.attributes.share_prefixes,
        PhysicalNode::AiScore(node) => node
            .attributes
            .spec
            .as_ref()
            .is_some_and(|s| s.share_prefixes),
        PhysicalNode::AiClassify(node) => node
            .attributes
            .spec
            .as_ref()
            .is_some_and(|s| s.share_prefixes),
        PhysicalNode::AiExtract(node) => node
            .attributes
            .spec
            .as_ref()
            .is_some_and(|s| s.share_prefixes),
        PhysicalNode::AiMap(node) => node
            .attributes
            .spec
            .as_ref()
            .is_some_and(|s| s.share_prefixes),
        PhysicalNode::AiSaeFeatures(node) => node
            .attributes
            .spec
            .as_ref()
            .is_some_and(|s| s.share_prefixes),
        _ => false,
    }
}

fn set_shares_prefixes(node: &mut PhysicalNode) {
    match node {
        PhysicalNode::AiFilter(node) => {
            node.attributes.share_prefixes = true;
            node.attributes.arena_writes = true;
        }
        PhysicalNode::AiJoin(node) => node.attributes.share_prefixes = true,
        PhysicalNode::AiScore(node) => {
            node.attributes
                .spec
                .as_mut()
                .expect("a spec")
                .share_prefixes = true
        }
        PhysicalNode::AiClassify(node) => {
            node.attributes
                .spec
                .as_mut()
                .expect("a spec")
                .share_prefixes = true
        }
        PhysicalNode::AiExtract(node) => {
            node.attributes
                .spec
                .as_mut()
                .expect("a spec")
                .share_prefixes = true
        }
        PhysicalNode::AiMap(node) => {
            node.attributes
                .spec
                .as_mut()
                .expect("a spec")
                .share_prefixes = true
        }
        PhysicalNode::AiSaeFeatures(node) => {
            node.attributes
                .spec
                .as_mut()
                .expect("a spec")
                .share_prefixes = true
        }
        _ => {}
    }
}

fn should_share_prefixes(graph: &Graph, context: &PlanningContext) -> Rewrite {
    let model = &context.model;
    if !model.reuses_kv() {
        return Ok(None);
    }
    let max_pooling: BTreeSet<String> = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::AiSaeFeatures(features)
                if features
                    .attributes
                    .spec
                    .as_ref()
                    .is_some_and(|s| s.pooling == "max") =>
            {
                node.document_aliases().into_iter().next()
            }
            _ => None,
        })
        .collect();
    let linear_seconds = coefficients(model, &context.device, false).linear_seconds;
    let mut nodes = Vec::new();
    let mut changed = false;
    for node in &graph.nodes {
        let model_node = matches!(
            node,
            PhysicalNode::AiFilter(_)
                | PhysicalNode::AiJoin(_)
                | PhysicalNode::AiClassify(_)
                | PhysicalNode::AiExtract(_)
                | PhysicalNode::AiMap(_)
                | PhysicalNode::AiSaeFeatures(_)
                | PhysicalNode::AiScore(_)
        );
        let inference = !matches!(node, PhysicalNode::AiFilter(_) | PhysicalNode::AiJoin(_));
        let is_score = matches!(node, PhysicalNode::AiScore(_));
        if !model_node
            || (inference && !has_spec(node))
            || shares_prefixes(node)
            || (is_score && node.spec_aliases().len() != 1)
        {
            nodes.push(node.clone());
            continue;
        }
        let alias = if is_score {
            node.spec_aliases()[0].clone()
        } else {
            node.document_aliases()[0].clone()
        };
        // borrowed KV has no residual rows from which to recover SAE maxima
        let Some(shared) = context
            .page_shared
            .get(&alias)
            .filter(|_| !max_pooling.contains(&alias))
        else {
            nodes.push(node.clone());
            continue;
        };
        let shared: i64 = shared.iter().sum();
        let writes_pages = match node {
            PhysicalNode::AiFilter(filter) => filter.attributes.arena_writes,
            _ => !is_score,
        };
        let total = AliasStats::summarize(context.lengths(&alias)?, model.sliding_window).total;
        let pays = shared > 0 && {
            let saved = shared as f64 * linear_seconds;
            let cost = if writes_pages {
                0.0
            } else {
                total as f64 * model.kappa() / context.device.hbm_bw
            };
            saved > cost
        };
        if !pays {
            nodes.push(node.clone());
            continue;
        }
        let mut node = node.clone();
        set_shares_prefixes(&mut node);
        nodes.push(node);
        changed = true;
    }
    Ok(changed.then(|| graph.with_nodes(nodes)))
}

fn has_spec(node: &PhysicalNode) -> bool {
    match node {
        PhysicalNode::AiScore(node) => node.attributes.spec.is_some(),
        PhysicalNode::AiClassify(node) => node.attributes.spec.is_some(),
        PhysicalNode::AiExtract(node) => node.attributes.spec.is_some(),
        PhysicalNode::AiMap(node) => node.attributes.spec.is_some(),
        PhysicalNode::AiSaeFeatures(node) => node.attributes.spec.is_some(),
        _ => true,
    }
}

fn mean_length(context: &PlanningContext, alias: &str) -> f64 {
    match context.document_tokens.get(alias) {
        Some(lengths) if !lengths.is_empty() => {
            lengths.iter().sum::<i64>() as f64 / lengths.len() as f64
        }
        _ => 0.0,
    }
}

/// The attention path for each table's answer branches, as `_answer_attention`.
fn answer_attention(graph: &Graph, context: &PlanningContext) -> BTreeMap<String, &'static str> {
    let mut branches: Vec<(String, Vec<(i64, i64)>)> = Vec::new();
    for node in &graph.nodes {
        let found = match node {
            PhysicalNode::AiMap(map) => map
                .attributes
                .spec
                .as_ref()
                .map(|spec| (spec.aliases[0].clone(), (spec.max_answer_tokens, 0))),
            PhysicalNode::AiExtract(extract) => extract.attributes.spec.as_ref().map(|spec| {
                (
                    spec.aliases[0].clone(),
                    (spec.max_answer_tokens, spec.draft_tokens),
                )
            }),
            _ => None,
        };
        if let Some((alias, limits)) = found {
            match branches.iter_mut().find(|(found, _)| *found == alias) {
                Some((_, specs)) => specs.push(limits),
                None => branches.push((alias, vec![limits])),
            }
        }
    }
    branches
        .into_iter()
        .map(|(alias, specs)| {
            let sum: i64 = specs.iter().map(|(limit, _)| limit).sum();
            let longest = specs
                .iter()
                .map(|(limit, _)| *limit)
                .max()
                .expect("a branch");
            let rows: i64 = specs.iter().map(|(_, draft)| 1 + draft).sum();
            let path = cheaper_attention_path(
                &context.model,
                &context.device,
                sum as f64 / longest as f64,
                rows as f64 / specs.len() as f64,
                mean_length(context, &alias),
            );
            (alias, path)
        })
        .collect()
}

fn filter_attention(context: &PlanningContext, alias: &str, question_tokens: i64) -> &'static str {
    let (mut tree, mut unified) = (0i64, 0i64);
    for (shared, readers, rows) in context
        .reader_groups
        .get(alias)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let path = cheaper_attention_path(
            &context.model,
            &context.device,
            *readers as f64,
            *rows as f64 / *readers as f64 + question_tokens as f64,
            *shared as f64,
        );
        if path == "tree" {
            tree += readers * shared;
        } else {
            unified += readers * shared;
        }
    }
    if tree > unified { "tree" } else { "unified" }
}

fn attention_of(node: &mut PhysicalNode) -> Option<&mut String> {
    match node {
        PhysicalNode::AiFilter(node) => Some(&mut node.attributes.attention),
        PhysicalNode::AiJoin(node) => Some(&mut node.attributes.attention),
        PhysicalNode::AiExtract(node) => Some(&mut node.attributes.attention),
        PhysicalNode::AiMap(node) => Some(&mut node.attributes.attention),
        _ => None,
    }
}

fn choose_attention_path(graph: &Graph, context: &PlanningContext) -> Rewrite {
    let model = &context.model;
    let forced = context.attention.as_deref();
    if let Some(forced) = forced
        && forced != "tree"
        && forced != "unified"
    {
        return Err(CompileError::new(format!(
            "attention must be 'tree' or 'unified', got {}",
            crate::expressions::repr_str(forced)
        )));
    }
    let tree_available = tree_attention_allowed(model);
    let answer_paths = if tree_available {
        answer_attention(graph, context)
    } else {
        BTreeMap::new()
    };
    let mut nodes = Vec::new();
    let mut changed = false;
    for node in &graph.nodes {
        let mut path: Option<&'static str> = match node {
            PhysicalNode::AiMap(map) => map.attributes.spec.as_ref().map(|spec| {
                answer_paths
                    .get(&spec.aliases[0])
                    .copied()
                    .unwrap_or("unified")
            }),
            PhysicalNode::AiExtract(extract) => extract.attributes.spec.as_ref().map(|spec| {
                answer_paths
                    .get(&spec.aliases[0])
                    .copied()
                    .unwrap_or("unified")
            }),
            PhysicalNode::AiFilter(filter) => {
                let attributes = &filter.attributes;
                Some(
                    if tree_available
                        && attributes.share_prefixes
                        && context.page_shared.contains_key(&attributes.alias)
                    {
                        filter_attention(context, &attributes.alias, attributes.question_tokens)
                    } else {
                        "unified"
                    },
                )
            }
            PhysicalNode::AiJoin(join) => {
                let stage = &join.attributes;
                Some(if tree_available {
                    // the partners each live anchor reads, on average
                    let readers = if stage.anchor_documents != 0.0 {
                        stage.expected_rows / stage.anchor_documents
                    } else {
                        1.0
                    };
                    // a partner's rows: its document and the question tail
                    let rows = stage.pair_tail_tokens as f64
                        + quail_cost::quantity::python_sum(
                            stage
                                .partners
                                .iter()
                                .map(|partner| mean_length(context, partner)),
                        );
                    cheaper_attention_path(
                        model,
                        &context.device,
                        readers,
                        rows,
                        mean_length(context, &stage.anchor),
                    )
                } else {
                    "unified"
                })
            }
            _ => None,
        };
        if path.is_some()
            && let Some(forced) = forced
        {
            path = Some(if forced == "tree" { "tree" } else { "unified" });
        }
        let mut node = node.clone();
        if let Some(path) = path
            && let Some(attention) = attention_of(&mut node)
            && attention != path
        {
            *attention = path.to_string();
            changed = true;
        }
        nodes.push(node);
    }
    Ok(changed.then(|| graph.with_nodes(nodes)))
}

fn should_deduplicate(
    graph: &Graph,
    context: &PlanningContext,
    settings: &mut Map<String, Value>,
) -> Rewrite {
    settings.remove("deduplicate");
    if context.backend != "quail" {
        return Ok(None);
    }
    let mut stop_keys: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut aliases: BTreeSet<String> = BTreeSet::new();
    for node in &graph.nodes {
        match node {
            PhysicalNode::AiFilter(filter) => {
                stop_keys.insert(
                    filter.attributes.alias.clone(),
                    filter.attributes.stop_key.clone(),
                );
                aliases.insert(filter.attributes.alias.clone());
            }
            PhysicalNode::AiJoin(join) => {
                aliases.insert(join.attributes.anchor.clone());
                aliases.extend(join.attributes.partners.iter().cloned());
            }
            _ => aliases.extend(node.spec_aliases()),
        }
    }
    let chosen: Map<String, Value> = aliases
        .into_iter()
        .filter(|alias| context.key_documents(alias).is_some())
        .map(|alias| {
            let key = stop_keys.get(&alias).cloned().unwrap_or_default();
            (alias, Value::from(key))
        })
        .collect();
    if !chosen.is_empty() {
        settings.insert("deduplicate".into(), Value::Object(chosen));
    }
    Ok(None)
}

/// Apply one rule by name.
///
/// # Errors
///
/// The rule is unknown, a classification no rule can score refuses the
/// plan, or a callback fails.
pub fn apply_rule(
    name: &str,
    graph: &Graph,
    context: &PlanningContext,
    settings: &mut Map<String, Value>,
) -> Rewrite {
    match name {
        "should_push_down_limit" => should_push_down_limit(graph, settings),
        "choose_classify_implementation" => {
            choose_classify_implementation(graph, context, settings)
        }
        "pack_set_passes" => pack_set_passes(graph, context),
        "join_document_ends" => join_document_ends(graph, context),
        "should_retain_kv" => should_retain_kv(graph, context, settings),
        "should_share_prefixes" => should_share_prefixes(graph, context),
        "choose_attention_path" => choose_attention_path(graph, context),
        "should_deduplicate" => should_deduplicate(graph, context, settings),
        other => Err(CompileError::new(format!(
            "unknown physical rule {other:?}"
        ))),
    }
}

/// Refine a plan by applying the rules in order and pricing it again,
/// as `apply_plan_rules`.
///
/// # Errors
///
/// A rule fails other than by refusing the plan.
pub fn apply_plan_rules(
    plan: PhysicalPlan,
    rules: &[String],
    context: &PlanningContext,
) -> Result<Planned, CompileError> {
    let mut settings = plan.settings.clone();
    let mut graph = Graph {
        nodes: plan.nodes.clone(),
        root: plan.root.clone(),
    };
    let mut changed = false;
    for rule in rules {
        match apply_rule(rule, &graph, context, &mut settings) {
            Ok(Some(replacement)) if replacement != graph => {
                graph = replacement;
                changed = true;
            }
            Ok(_) => {}
            Err(CompileError(_, Some(refusal))) => return Ok(Err(*refusal)),
            Err(error) => return Err(error),
        }
    }
    if !changed && settings == plan.settings {
        return Ok(Ok(plan));
    }
    let (seconds, estimates) = plan.pricer.price(context, &graph.nodes)?;
    Ok(Ok(PhysicalPlan {
        nodes: graph.nodes,
        root: graph.root,
        estimated_seconds: seconds,
        settings,
        estimates,
        ..plan
    }))
}
