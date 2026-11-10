//! The operator order a request backend runs: each table's AI.IF filters
//! and the join sequence, priced on one model copy without KV reuse
//! across operators.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::budgets::chunk_budget;
use crate::cost::statistics::AliasStats;
use crate::cost::work::Work;
use crate::physical::{PhysicalNode, PortRef, port};
use serde::Serialize;

use crate::context::PlanningContext;
use crate::costs::{join_specs, prepare_filter_costs};
use crate::error::CompileError;
use crate::expressions::asked_call;
use crate::expressions::repr_str;
use crate::graph::hash_join_nodes;
use crate::operators::Operators;
use crate::plan_statistics::CorpusStats;
use crate::search::{StepCost, StepKey, TableStep, default_order_rule, order_steps, search_joins};
use crate::statistics::effective_selectivity;

/// One table's documents and tokens, before and after its scan's tests.
#[derive(Debug, Clone, Serialize)]
pub struct TableSize {
    pub num_documents: i64,
    pub total_tokens: i64,
    /// The documents and tokens the regular predicates are expected to keep.
    pub kept_documents: i64,
    pub kept_tokens: i64,
}

/// The order a request backend runs a plan's operators in.
#[derive(Debug, Clone, Serialize)]
pub struct RequestOrder {
    /// "by_cost" or "as_written".
    pub rule: String,
    /// alias -> the written positions of its AI.IF filters, in run order.
    pub filter_orders: BTreeMap<String, Vec<usize>>,
    /// (written position, anchor) of each join, in run order.
    pub join_sequence: Vec<(usize, String)>,
    /// The hash joins that pair each join's tables on its equality conditions.
    pub hash_joins: Vec<PhysicalNode>,
    pub tables: BTreeMap<String, TableSize>,
}

/// Order a plan's filters and joins for a request backend.
///
/// The filters of a table are ordered at one document, as each runs
/// once per document; the joins are searched as on the Quail backend
/// but with no table's KV resident and on one model copy.
///
/// # Errors
///
/// A prompt cannot be lowered, or a scanned table has no document tokens.
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
                first: StepCost::Curve(cost.first),
                later: StepCost::Curve(cost.later),
                reuse_key: cost.reuse_key,
                document_head: Vec::new(),
            })
            .collect();
        filter_orders.insert(alias.clone(), order_steps(&steps, &rule, 1.0, context)?);
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
