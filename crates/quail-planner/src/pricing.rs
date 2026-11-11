//! Price a physical plan: each node on its own, and the plan as a whole.
//!
//! The plan's seconds pack every node's work into forward passes
//! together and add the seconds-priced nodes, so node prices do not add
//! up to the plan's.

use std::collections::BTreeMap;

use crate::cost::budgets::{floor_divide, pages_of, parallel_copies};
use crate::cost::extract::{self, ExtractInputs};
use crate::cost::quantity::{Quantity, numpy_sum};
use crate::cost::score::score_fixed_tokens;
use crate::cost::sol::speed_of_light;
use crate::cost::statistics::{AliasStats, round_half_even};
use crate::cost::work::{Work, ask, packed, scan};
use crate::physical::{AiFilterAttributes, PhysicalNode, build_pipelines, resident_inputs};
use serde_json::{Map, Value};

use crate::classify::classify_statistics;
use crate::columns::map_seconds;
use crate::context::PlanningContext;
use crate::error::CompileError;
use crate::plan_statistics::CorpusStats;
use crate::prompts::prompt_layout;

/// One node's price: its packed work, or its own estimated seconds.
#[derive(Debug, Clone, Default)]
struct NodePrice {
    work: Option<Work>,
    seconds: f64,
}

/// What prices the nodes of a plan's graph, as `PlanPricer`.
#[derive(Debug, Clone)]
pub struct PlanPricer {
    pub chunk: i64,
    pub cap_pages: i64,
    pub stats: BTreeMap<String, CorpusStats>,
    /// Join written position -> the work its stage does.
    pub stage_works: BTreeMap<usize, Work>,
    /// alias -> documents expected after its filters.
    pub expected_rows: BTreeMap<String, f64>,
}

type Priced = Vec<(String, NodePrice, Map<String, Value>)>;

fn set(priced: &mut Priced, node_id: &str, price: NodePrice, entry: Map<String, Value>) {
    match priced.iter_mut().find(|(found, _, _)| found == node_id) {
        Some(slot) => {
            slot.1 = price;
            slot.2 = entry;
        }
        None => priced.push((node_id.to_string(), price, entry)),
    }
}

fn spec_expected_rows(node: &PhysicalNode) -> f64 {
    match node {
        PhysicalNode::AiFilter(node) => node.attributes.expected_rows,
        PhysicalNode::AiScore(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0.0, |s| s.expected_rows),
        PhysicalNode::AiClassify(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0.0, |s| s.expected_rows),
        PhysicalNode::AiExtract(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0.0, |s| s.expected_rows),
        PhysicalNode::AiMap(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0.0, |s| s.expected_rows),
        PhysicalNode::AiSaeFeatures(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0.0, |s| s.expected_rows),
        _ => 0.0,
    }
}

fn spec_seconds(node: &PhysicalNode) -> Option<f64> {
    match node {
        PhysicalNode::AiScore(node) => node.attributes.spec.as_ref().map(|s| s.estimated_seconds),
        PhysicalNode::AiClassify(node) => {
            node.attributes.spec.as_ref().map(|s| s.estimated_seconds)
        }
        PhysicalNode::AiExtract(node) => node.attributes.spec.as_ref().map(|s| s.estimated_seconds),
        PhysicalNode::AiMap(node) => node.attributes.spec.as_ref().map(|s| s.estimated_seconds),
        PhysicalNode::AiSaeFeatures(node) => {
            node.attributes.spec.as_ref().map(|s| s.estimated_seconds)
        }
        _ => None,
    }
}

fn spec_state_tail(node: &PhysicalNode) -> i64 {
    match node {
        PhysicalNode::AiScore(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0, |s| s.state_tail_tokens),
        PhysicalNode::AiClassify(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0, |s| s.state_tail_tokens),
        PhysicalNode::AiExtract(node) => node
            .attributes
            .spec
            .as_ref()
            .map_or(0, |s| s.state_tail_tokens),
        _ => 0,
    }
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

/// The prompt tokens before each document, as `passes.head_tokens`.
pub fn head_tokens(node: &PhysicalNode) -> i64 {
    match node {
        PhysicalNode::AiFilter(node) => node.attributes.preamble_tokens,
        _ => node.prompt_token_parts()[0].len() as i64,
    }
}

/// An operator's question tokens after the state tail, as
/// `passes.question_tokens`.
pub fn pass_question_tokens(node: &PhysicalNode, state_tail: i64) -> i64 {
    match node {
        PhysicalNode::AiFilter(node) => (node.attributes.question_tokens - state_tail).max(1),
        _ => node.prompt_token_parts()[1].len() as i64 - spec_state_tail(node),
    }
}

/// The documents an operator is expected to be asked about.
pub fn expected_documents(node: &PhysicalNode) -> f64 {
    spec_expected_rows(node)
}

/// The positions of each pass group's operators, in execution order.
pub fn pass_members(nodes: &[PhysicalNode]) -> Vec<Vec<usize>> {
    let mut groups: Vec<(i64, Vec<usize>)> = Vec::new();
    for index in crate::physical::topological_nodes(nodes) {
        let group = nodes[index].pass_group();
        if group < 0 {
            continue;
        }
        match groups.iter_mut().find(|(found, _)| *found == group) {
            Some((_, members)) => members.push(index),
            None => groups.push((group, vec![index])),
        }
    }
    groups.into_iter().map(|(_, members)| members).collect()
}

/// A set-attention pass's state tail and later-question header lengths.
///
/// # Errors
///
/// The layout is unknown, or the tokenizer fails.
pub fn set_pass_tokens(context: &PlanningContext) -> Result<(i64, i64), CompileError> {
    let layout = prompt_layout(&context.model.prompt_layout)?;
    let label = layout.question_label;
    let separator = layout.question_separator.unwrap_or_default();
    Ok((
        (context.tokenizer)(&format!("{separator}{label}"))?.len() as i64,
        (context.tokenizer)(label)?.len() as i64,
    ))
}

impl PlanPricer {
    /// The seconds of packed work.
    pub fn seconds_of(&self, context: &PlanningContext, work: &Work) -> f64 {
        speed_of_light(work, &context.model, &context.device, self.chunk, 0).seconds()
    }

    /// Price a graph: its seconds, and node id -> its explain entry.
    ///
    /// # Errors
    ///
    /// A spec's estimator fails.
    pub fn price(
        &self,
        context: &PlanningContext,
        nodes: &[PhysicalNode],
    ) -> Result<(f64, Map<String, Value>), CompileError> {
        // Graphs may arrive from Python rather than the planner. Costing reads
        // each inference spec's document alias and head/tail token parts.
        for node in nodes {
            if matches!(
                node,
                PhysicalNode::AiScore(_)
                    | PhysicalNode::AiClassify(_)
                    | PhysicalNode::AiExtract(_)
                    | PhysicalNode::AiMap(_)
                    | PhysicalNode::AiSaeFeatures(_)
            ) && (node.spec_aliases().is_empty() || node.prompt_token_parts().len() < 2)
            {
                return Err(CompileError::new(format!(
                    "physical node {:?} needs a spec with aliases and head/tail prompt tokens",
                    node.id()
                )));
            }
        }
        let mut work = Work::default();
        let mut seconds = 0.0;
        let mut entries = Map::new();
        for (node_id, price, entry) in self.priced(context, nodes)? {
            entries.insert(node_id, Value::Object(entry));
            match price.work {
                Some(found) => work = &work + &found,
                None => seconds += price.seconds,
            }
        }
        Ok((self.seconds_of(context, &work) + seconds, entries))
    }

    fn priced(
        &self,
        context: &PlanningContext,
        nodes: &[PhysicalNode],
    ) -> Result<Priced, CompileError> {
        let mut out = self.each_priced(context, nodes)?;
        if context.model.attention_mask == "set" {
            for (node_id, price, entry) in self.set_passes(context, nodes)? {
                set(&mut out, &node_id, price, entry);
            }
            for node in nodes {
                if let PhysicalNode::AiJoin(_) = node {
                    let slot = out
                        .iter_mut()
                        .find(|(found, _, _)| found == node.id())
                        .expect("a priced join");
                    slot.2
                        .insert("pass_questions".into(), Value::from(context.model.set_size));
                }
            }
        }
        Ok(out)
    }

    /// A set-attention plan's passes: each group's first operator carries
    /// the work of its passes, the others none.
    fn set_passes(
        &self,
        context: &PlanningContext,
        nodes: &[PhysicalNode],
    ) -> Result<Priced, CompileError> {
        let model = &context.model;
        let (state_tail, header) = set_pass_tokens(context)?;
        let mut out = Priced::new();
        for members in pass_members(nodes) {
            let first = &nodes[members[0]];
            let alias = first.document_aliases()[0].clone();
            let summary = AliasStats::summarize(context.lengths(&alias)?, model.sliding_window);
            let questions: Vec<f64> = members
                .iter()
                .enumerate()
                .map(|(index, member)| {
                    (pass_question_tokens(&nodes[*member], state_tail)
                        + if index > 0 { header } else { 0 }) as f64
                })
                .collect();
            let rows = expected_documents(first) * self.distinct(context, &alias);
            let state =
                Quantity::Scalar(head_tokens(first) as f64 + summary.mean() + state_tail as f64);
            let pass = &packed(&state, &questions, model.sliding_window)
                + &Work {
                    pairs: Quantity::Scalar(summary.variance()),
                    ..Work::default()
                };
            let work = &pass * rows;
            let ids: Vec<Value> = members
                .iter()
                .map(|member| Value::from(nodes[*member].id()))
                .collect();
            for (index, member) in members.iter().enumerate() {
                let mut entry = Map::new();
                entry.insert("pass_members".into(), Value::Array(ids.clone()));
                let price = if index == 0 {
                    entry.insert(
                        "seconds".into(),
                        Value::from(self.seconds_of(context, &work)),
                    );
                    NodePrice {
                        work: Some(work.clone()),
                        seconds: 0.0,
                    }
                } else {
                    NodePrice {
                        work: Some(Work::default()),
                        seconds: 0.0,
                    }
                };
                out.push((nodes[*member].id().to_string(), price, entry));
            }
        }
        Ok(out)
    }

    fn each_priced(
        &self,
        context: &PlanningContext,
        nodes: &[PhysicalNode],
    ) -> Result<Priced, CompileError> {
        let pipelines = build_pipelines(nodes, &[]);
        let anchored: Vec<&str> = nodes
            .iter()
            .filter_map(|node| match node {
                PhysicalNode::AiJoin(join) => Some(join.attributes.anchor.as_str()),
                _ => None,
            })
            .collect();
        let mut last_filters: BTreeMap<&str, &str> = BTreeMap::new();
        for node in nodes {
            if let PhysicalNode::AiFilter(filter) = node {
                last_filters.insert(filter.attributes.alias.as_str(), node.id());
            }
        }
        let residents = resident_inputs(nodes, &pipelines);
        let mut out = Priced::new();
        for index in crate::physical::topological_nodes(nodes) {
            let node = &nodes[index];
            let mut entry = Map::new();
            let resident = residents.contains(node.id());
            let price = match node {
                PhysicalNode::AiFilter(filter) => {
                    let attributes = &filter.attributes;
                    let work = &self.filter_work(context, attributes, node, resident)?
                        * (1.0 / parallel_copies(context.gpus, attributes.expected_rows));
                    entry.insert(
                        "seconds".into(),
                        Value::from(self.seconds_of(context, &work)),
                    );
                    if anchored.contains(&attributes.alias.as_str())
                        && last_filters[attributes.alias.as_str()] == node.id()
                    {
                        self.release_recompute(context, attributes, &mut entry);
                    }
                    NodePrice {
                        work: Some(work),
                        seconds: 0.0,
                    }
                }
                PhysicalNode::AiJoin(join) => {
                    let attributes = &join.attributes;
                    let mut work = self
                        .stage_works
                        .get(&(attributes.written_position as usize))
                        .cloned()
                        .unwrap_or_default();
                    if attributes.pairs_from.is_empty() {
                        // over every pair, the model is asked once per pair of
                        // distinct texts; pairs from a hash join are priced in full
                        for alias in std::iter::once(&attributes.anchor).chain(&attributes.partners)
                        {
                            work = &work * self.distinct(context, alias);
                        }
                    }
                    entry.insert(
                        "seconds".into(),
                        Value::from(self.seconds_of(context, &work)),
                    );
                    NodePrice {
                        work: Some(work),
                        seconds: 0.0,
                    }
                }
                _ if spec_seconds(node).is_some() => {
                    let seconds = self.inference_seconds(context, node, resident)?;
                    entry.insert("seconds".into(), Value::from(seconds));
                    NodePrice {
                        work: None,
                        seconds,
                    }
                }
                _ => NodePrice::default(),
            };
            out.push((node.id().to_string(), price, entry));
        }
        Ok(out)
    }

    /// The share of an alias's documents the model is asked about.
    fn distinct(&self, context: &PlanningContext, alias: &str) -> f64 {
        match context.key_documents(alias) {
            None => 1.0,
            Some(keys) => keys.iter().filter(|key| **key).count() as f64 / keys.len() as f64,
        }
    }

    fn mean_prefix(&self, attributes: &AiFilterAttributes) -> f64 {
        attributes.preamble_tokens as f64 + self.stats[&attributes.alias].mean_document_tokens()
    }

    /// Per document, the prefix tokens a node borrows, or None without sharing.
    fn shared<'a>(
        &self,
        context: &'a PlanningContext,
        node: &PhysicalNode,
        alias: &str,
    ) -> Option<&'a [i64]> {
        if !shares_prefixes(node) {
            return None;
        }
        context.page_shared.get(alias).map(Vec::as_slice)
    }

    /// A filter's work: its question over resident KV, else a scan of each.
    fn filter_work(
        &self,
        context: &PlanningContext,
        attributes: &AiFilterAttributes,
        node: &PhysicalNode,
        resident: bool,
    ) -> Result<Work, CompileError> {
        let window = context.model.sliding_window;
        let question = Quantity::Scalar(attributes.question_tokens as f64);
        let mean_prefix = Quantity::Scalar(self.mean_prefix(attributes));
        let keys = context.key_documents(&attributes.alias);
        let asked = attributes.expected_rows * self.distinct(context, &attributes.alias);
        if resident {
            return Ok(&ask(&mean_prefix, &question, window) * asked);
        }
        let Some(shared) = self.shared(context, node, &attributes.alias) else {
            return Ok(&scan(&mean_prefix, &question, window) * asked);
        };
        let lengths = context.lengths(&attributes.alias)?;
        if lengths.is_empty() {
            return Ok(Work::default());
        }
        let mut work = Work::default();
        for (index, (length, shared)) in lengths.iter().zip(shared).enumerate() {
            if keys.is_some_and(|keys| !keys[index]) {
                continue;
            }
            let prefix = attributes.preamble_tokens + length;
            // a borrowed prefix includes the prompt head the documents share
            let borrowed = if *shared != 0 {
                shared + attributes.preamble_tokens
            } else {
                0
            };
            let question = attributes.question_tokens;
            let step = if *shared != 0 {
                ask(
                    &Quantity::Scalar(borrowed as f64),
                    &Quantity::Scalar((prefix + question - borrowed) as f64),
                    window,
                )
            } else {
                scan(
                    &Quantity::Scalar(prefix as f64),
                    &Quantity::Scalar(question as f64),
                    window,
                )
            };
            work = &work + &step;
        }
        Ok(&work * (attributes.expected_rows / lengths.len() as f64))
    }

    /// What a later join pays if the sequence's KV is released, not pinned.
    fn release_recompute(
        &self,
        context: &PlanningContext,
        attributes: &AiFilterAttributes,
        entry: &mut Map<String, Value>,
    ) {
        let prefix = self.mean_prefix(attributes);
        let pages = pages_of(prefix, context.device.page_tokens);
        let fits = floor_divide(self.cap_pages as f64, pages.max(1.0));
        let excess = (self
            .expected_rows
            .get(&attributes.alias)
            .copied()
            .unwrap_or(0.0)
            - fits)
            .max(0.0);
        entry.insert(
            "release_recompute_tokens".into(),
            Value::from(round_half_even(excess * prefix)),
        );
        let work = &scan(
            &Quantity::Scalar(prefix),
            &Quantity::Scalar(0.0),
            context.model.sliding_window,
        ) * excess;
        entry.insert(
            "release_recompute_seconds".into(),
            Value::from(self.seconds_of(context, &work)),
        );
    }

    /// An inference node's seconds: its spec's, scaled by the share it computes.
    fn inference_seconds(
        &self,
        context: &PlanningContext,
        node: &PhysicalNode,
        resident: bool,
    ) -> Result<f64, CompileError> {
        let aliases = node.spec_aliases();
        let alias = (aliases.len() == 1).then(|| aliases[0].clone());
        let shared = alias
            .as_deref()
            .and_then(|alias| self.shared(context, node, alias));
        let mut seconds = spec_seconds(node).unwrap_or(0.0);
        if let (Some(shared), Some(alias)) = (shared, &alias)
            && seconds != 0.0
        {
            seconds *= self.fresh_fraction(context, node, alias, shared, resident)?;
        }
        let keys = alias
            .as_deref()
            .and_then(|alias| context.key_documents(alias));
        if let (Some(keys), Some(alias)) = (keys, &alias)
            && seconds != 0.0
        {
            seconds *= self.asked_fraction(context, node, alias, keys, shared)?;
        }
        Ok(seconds)
    }

    /// The share of a node's fresh tokens that its key documents compute.
    fn asked_fraction(
        &self,
        context: &PlanningContext,
        node: &PhysicalNode,
        alias: &str,
        keys: &[bool],
        shared: Option<&[i64]>,
    ) -> Result<f64, CompileError> {
        let fixed: usize = node.prompt_token_parts().iter().map(Vec::len).sum();
        let mut fresh: Vec<f64> = context
            .lengths(alias)?
            .iter()
            .map(|length| *length as f64 + fixed as f64)
            .collect();
        if let Some(shared) = shared {
            for (value, borrowed) in fresh.iter_mut().zip(shared) {
                *value -= *borrowed as f64;
            }
        }
        let total = numpy_sum(&fresh);
        if total == 0.0 {
            return Ok(1.0);
        }
        let kept: Vec<f64> = fresh
            .iter()
            .zip(keys)
            .filter(|(_, key)| **key)
            .map(|(value, _)| *value)
            .collect();
        Ok(numpy_sum(&kept) / total)
    }

    fn fresh_fraction(
        &self,
        context: &PlanningContext,
        node: &PhysicalNode,
        alias: &str,
        shared: &[i64],
        resident: bool,
    ) -> Result<f64, CompileError> {
        let shared_f64: Vec<f64> = shared.iter().map(|value| *value as f64).collect();
        let ratio = |part: f64, whole: f64| if whole != 0.0 { part / whole } else { 1.0 };
        match node {
            PhysicalNode::AiExtract(extract) => {
                let spec = extract.attributes.spec.as_ref().expect("a spec");
                let lengths: Vec<f64> = context.lengths(alias)?.iter().map(|l| *l as f64).collect();
                let inputs = ExtractInputs {
                    head_tokens: spec.prompt_token_parts[0].len() as i64,
                    tail_tokens: spec.prompt_token_parts[1].len() as i64,
                    window: 0,
                    resident,
                    max_answer_tokens: spec.max_answer_tokens,
                    draft_tokens: spec.draft_tokens,
                    draft_policy: spec.draft_policy.clone(),
                    round_cost_per_token: spec.round_cost_per_token,
                    answer_tokens: crate::columns::EXTRACT_ANSWER_TOKENS,
                    recurrent: false,
                };
                let seconds = |shared: &[f64]| {
                    extract::estimate(
                        spec.expected_rows,
                        &lengths,
                        shared,
                        &inputs,
                        self.chunk,
                        &context.model,
                        &context.device,
                        context.gpus,
                    )
                    .map(|cost| cost.seconds)
                    .map_err(CompileError::new)
                };
                Ok(ratio(seconds(&shared_f64)?, seconds(&[])?))
            }
            PhysicalNode::AiMap(map) => {
                let spec = map.attributes.spec.as_ref().expect("a spec");
                let whole = map_seconds(context, spec, spec.expected_rows, resident, &[])?;
                let part = map_seconds(context, spec, spec.expected_rows, resident, &shared_f64)?;
                Ok(ratio(part, whole))
            }
            PhysicalNode::AiClassify(classify) => {
                let spec = classify.attributes.spec.as_ref().expect("a spec");
                let whole = classify_statistics(context, alias, &[])?
                    .estimate_spec(spec, resident)?
                    .seconds;
                let part = classify_statistics(context, alias, &shared_f64)?
                    .estimate_spec(spec, resident)?
                    .seconds;
                Ok(ratio(part, whole))
            }
            _ => {
                // a score's or a feature call's time is taken as proportional
                // to the tokens it computes
                let parts: Vec<i64> = node
                    .prompt_token_parts()
                    .iter()
                    .map(|part| part.len() as i64)
                    .collect();
                let fixed = match node {
                    PhysicalNode::AiScore(score) => score_fixed_tokens(
                        &parts,
                        context.model.canvas_tokens,
                        score.attributes.spec.as_ref().map_or(1, |s| s.draws),
                    ),
                    _ => parts.iter().sum(),
                };
                let summary =
                    AliasStats::summarize(context.lengths(alias)?, context.model.sliding_window);
                let total = summary.total + summary.count * fixed;
                if total == 0 {
                    return Ok(1.0);
                }
                let borrowed: i64 = shared.iter().sum();
                Ok((1.0 - borrowed as f64 / total as f64).max(0.0))
            }
        }
    }
}
