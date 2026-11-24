use std::collections::{BTreeMap, BTreeSet};

use crate::budgets::chunk_budget;
use crate::cost::statistics::AliasStats;
use crate::cost::work::Work;
use crate::physical::{PhysicalNode, PortRef};
use crate::topology::port;
use serde::Serialize;

use crate::context::PlanningContext;
use crate::error::CompileError;
use crate::expressions::asked_call;
use crate::expressions::repr_str;
use crate::operators::Operators;
use crate::operators::effective_selectivity;
use crate::plan::graph::hash_join_nodes;
use crate::pricing::costs::{join_specs, prepare_filter_costs};
use crate::pricing::plan_statistics::CorpusStats;
use crate::pricing::search::{StepKey, TableStep, default_order_rule, order_steps, search_joins};

#[derive(Debug, Clone, Serialize)]
pub struct TableSize {
    pub num_documents: i64,
    pub total_tokens: i64,
    pub kept_documents: i64,
    pub kept_tokens: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestOrder {
    pub rule: String,
    pub filter_orders: BTreeMap<String, Vec<usize>>,
    pub join_sequence: Vec<(usize, String)>,
    pub hash_joins: Vec<PhysicalNode>,
    pub tables: BTreeMap<String, TableSize>,
}

pub fn request_order(
    found: &Operators,
    context: &PlanningContext,
) -> Result<RequestOrder, CompileError> {
    let mut context = context.clone();
    context.gpus = 1;
    let context = &context;
    let (model, device) = (&context.model, &context.device);
    let rule = context
        .order
        .clone()
        .unwrap_or_else(|| default_order_rule(found).0);
    let mut summaries: BTreeMap<String, AliasStats> = BTreeMap::new();
    let mut tables = BTreeMap::new();
    let mut stats: BTreeMap<String, CorpusStats> = BTreeMap::new();
    for (alias, lengths) in &context.document_tokens {
        let summary = AliasStats::summarize(lengths, model.sliding_window);
        let kept = summary.scale(context.scan_fractions.get(alias).copied().unwrap_or(1.0));
        tables.insert(
            alias.clone(),
            TableSize {
                num_documents: summary.count,
                total_tokens: summary.total,
                kept_documents: kept.count,
                kept_tokens: kept.total,
            },
        );
        stats.insert(
            alias.clone(),
            CorpusStats {
                num_documents: kept.count,
                total_tokens: kept.total,
                max_document_tokens: summary.maximum,
            },
        );
        summaries.insert(alias.clone(), summary);
    }
    for scan in &found.scans {
        if !stats.contains_key(&scan.alias) {
            return Err(CompileError::new(format!(
                "no document tokens for alias {}",
                repr_str(&scan.alias)
            )));
        }
    }
    let chunk = chunk_budget(model, device);
    let asked = found
        .filters
        .iter()
        .filter(|(_, predicates)| !predicates.is_empty());
    let mut pre = 0;
    let prompts = asked
        .clone()
        .flat_map(|(_, predicates)| predicates.iter().map(|p| asked_call(&p.expression)))
        .chain(found.joins.iter().map(|join| asked_call(&join.predicate)));
    for call in prompts {
        if call.prompt.kind != "score" {
            pre = context.lowered(&call.prompt)?.preamble_token_ids.len() as i64;
            break;
        }
    }
    let mut filter_orders = BTreeMap::new();
    for (alias, predicates) in asked {
        let costs = prepare_filter_costs(
            predicates,
            pre as f64 + stats[alias].mean_document_tokens(),
            chunk,
            context,
        )?;
        let steps: Vec<TableStep> = costs
            .into_iter()
            .enumerate()
            .map(|(position, cost)| TableStep {
                kind: "filter",
                key: StepKey::Position(position),
                selectivity: cost.selectivity,
                or_term: None,
                first: cost.first,
                later: cost.later,
                reuse_key: cost.reuse_key,
                document_head: Vec::new(),
                reads: Vec::new(),
            })
            .collect();
        filter_orders.insert(
            alias.clone(),
            order_steps(&steps, &rule, 1.0, chunk, context)?,
        );
    }
    let mut expected: BTreeMap<String, f64> = stats
        .iter()
        .map(|(alias, table)| (alias.clone(), table.num_documents as f64))
        .collect();
    for (alias, predicates) in &found.filters {
        for predicate in predicates {
            *expected.entry(alias.clone()).or_default() *=
                effective_selectivity(predicate.selectivity);
        }
    }
    for (alias, tests) in &found.column_filters {
        for test in tests {
            *expected.entry(alias.clone()).or_default() *= effective_selectivity(test.selectivity);
        }
    }
    let specs = join_specs(&found.joins, &context.pair_fractions, 0, context, 0)?;
    let searched = search_joins(
        &specs,
        &expected,
        &summaries,
        &BTreeSet::new(),
        pre,
        chunk,
        context,
        &Work::default(),
        rule == "as_written",
    )?;
    let join_sequence = match searched {
        Some(search) => search.seq,
        None => specs
            .iter()
            .enumerate()
            .map(|(position, spec)| {
                (
                    position,
                    spec.anchor
                        .clone()
                        .unwrap_or_else(|| spec.aliases[0].clone()),
                )
            })
            .collect(),
    };
    let scan_ports: BTreeMap<String, PortRef> = found
        .scans
        .iter()
        .map(|scan| {
            (
                scan.alias.clone(),
                port(
                    &format!("scan:{}", scan.alias),
                    format!("rows:{}", scan.alias),
                ),
            )
        })
        .collect();
    Ok(RequestOrder {
        rule,
        filter_orders,
        join_sequence,
        hash_joins: hash_join_nodes(found, &context.pair_fractions, &scan_ports),
        tables,
    })
}
