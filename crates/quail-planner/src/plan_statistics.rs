//! What the cost model knows about a plan before any decision.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::budgets::{arena_pages, chunk_budget};
use crate::cost::joins::PricedJoin;
use crate::cost::retention::retention_pages;
use crate::cost::statistics::{AliasStats, int_true_divide};
use crate::physical::Prompt;

use crate::context::PlanningContext;
use crate::costs::join_specs;
use crate::expressions::asked_call;
use crate::expressions::{CompileError, compile_error, repr_str};
use crate::lowering::LoweredPrompt;
use crate::operators::Operators;
use crate::statistics::effective_selectivity;

/// One table's document count and token sums, as `CorpusStats`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorpusStats {
    pub num_documents: i64,
    pub total_tokens: i64,
    pub max_document_tokens: i64,
}

impl CorpusStats {
    /// The mean document length in tokens.
    pub fn mean_document_tokens(&self) -> f64 {
        int_true_divide(self.total_tokens as i128, self.num_documents.max(1) as i128)
    }
}

/// What the cost model knows about one plan before any decision.
#[derive(Debug, Clone)]
pub struct PlanStatistics {
    pub stats: BTreeMap<String, CorpusStats>,
    pub lengths: BTreeMap<String, AliasStats>,
    pub chunk: i64,
    pub arena_pages: (i64, i64),
    pub admission: i64,
    pub cap_pages: i64,
    pub pre: i64,
    pub specs: Vec<PricedJoin>,
    /// alias -> written positions of its AI.IF predicates, in filter order.
    pub asks: Vec<(String, Vec<usize>)>,
    /// Every prompt of the plan, lowered.
    pub lowered: Vec<(Prompt, LoweredPrompt)>,
}

impl PlanStatistics {
    /// A prompt's lowered form.
    ///
    /// # Panics
    ///
    /// The prompt is not one of the plan's.
    pub fn lowered(&self, prompt: &Prompt) -> &LoweredPrompt {
        self.lowered
            .iter()
            .find(|(found, _)| found == prompt)
            .map(|(_, lowered)| lowered)
            .expect("a prompt of the plan")
    }

    /// The written positions of an alias's AI.IF predicates.
    pub fn asks_of(&self, alias: &str) -> &[usize] {
        self.asks
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, positions)| positions.as_slice())
            .unwrap_or(&[])
    }
}

/// The prompts of a plan's filters, label tests, joins, and remaining
/// result columns, as `Operators.prompts`.
pub fn plan_prompts(found: &Operators) -> Vec<Prompt> {
    let mut calls: Vec<&crate::physical::ModelCall> = Vec::new();
    for (_, predicates) in &found.filters {
        calls.extend(
            predicates
                .iter()
                .map(|predicate| asked_call(&predicate.expression)),
        );
    }
    for (_, tests) in &found.column_filters {
        calls.extend(tests.iter().map(|test| &test.call));
    }
    calls.extend(found.joins.iter().map(|join| asked_call(&join.predicate)));
    let mut prompts: Vec<Prompt> = calls.iter().map(|call| (*call.prompt).clone()).collect();
    for column in &found.projections {
        if let Some(call) = crate::expressions::aliased_call(column, &[])
            && !calls.contains(&call)
        {
            prompts.push((*call.prompt).clone());
        }
    }
    prompts
}

/// Gather the statistics of one plan's operators, as `plan_statistics`.
///
/// # Errors
///
/// A scanned alias has no token counts, the arena does not fit the
/// device, or the tokenizer fails.
pub fn plan_statistics(
    found: &Operators,
    context: &PlanningContext,
) -> Result<PlanStatistics, CompileError> {
    let model = &context.model;
    let asks: Vec<(String, Vec<usize>)> = found
        .filters
        .iter()
        .map(|(alias, predicates)| {
            (
                alias.clone(),
                predicates
                    .iter()
                    .enumerate()
                    .filter(|(_, predicate)| asked_call(&predicate.expression).kind != "score")
                    .map(|(position, _)| position)
                    .collect(),
            )
        })
        .collect();
    let mut lowered: Vec<(Prompt, LoweredPrompt)> = Vec::new();
    let mut prompts = plan_prompts(found);
    prompts.extend(
        found
            .extracts
            .iter()
            .map(|column| (*column.call.prompt).clone()),
    );
    for prompt in prompts {
        if !lowered.iter().any(|(seen, _)| *seen == prompt) {
            let lowered_prompt = context.lowered(&prompt)?;
            lowered.push((prompt, lowered_prompt));
        }
    }
    let mut lengths: BTreeMap<String, AliasStats> = BTreeMap::new();
    for (alias, tokens) in &context.document_tokens {
        lengths.insert(
            alias.clone(),
            AliasStats::summarize(tokens, model.sliding_window),
        );
    }
    let chunk = chunk_budget(model, &context.device);
    let mut longest: Option<f64> = None;
    for summary in lengths.values() {
        if longest.is_none_or(|found| summary.mean() > found) {
            longest = Some(summary.mean());
        }
    }
    let split =
        arena_pages(model, &context.device, Some(chunk), longest).map_err(CompileError::new)?;
    for (alias, fraction) in &context.scan_fractions {
        if let Some(summary) = lengths.get_mut(alias) {
            *summary = summary.scale(*fraction);
        }
    }
    let stats: BTreeMap<String, CorpusStats> = lengths
        .iter()
        .map(|(alias, summary)| {
            (
                alias.clone(),
                CorpusStats {
                    num_documents: summary.count,
                    total_tokens: summary.total,
                    max_document_tokens: summary.maximum,
                },
            )
        })
        .collect();
    for scan in &found.scans {
        if !stats.contains_key(&scan.alias) {
            return compile_error(format!(
                "no document_tokens for alias {}",
                repr_str(&scan.alias)
            ));
        }
    }
    let admission = split.0 * context.device.page_tokens;
    let mut pre = 0;
    let preamble_prompts = asks
        .iter()
        .filter(|(_, positions)| !positions.is_empty())
        .flat_map(|(alias, positions)| {
            let predicates = &found
                .filters
                .iter()
                .find(|(found, _)| found == alias)
                .expect("filters")
                .1;
            positions.iter().map(move |position| {
                (*asked_call(&predicates[*position].expression).prompt).clone()
            })
        })
        .chain(
            found
                .joins
                .iter()
                .map(|join| (*asked_call(&join.predicate).prompt).clone()),
        );
    for prompt in preamble_prompts {
        let lowered_prompt = &lowered
            .iter()
            .find(|(seen, _)| *seen == prompt)
            .expect("lowered")
            .1;
        if lowered_prompt.prompt.kind != "score" {
            pre = lowered_prompt.preamble_token_ids.len() as i64;
            break;
        }
    }
    let specs = join_specs(
        &found.joins,
        &context.pair_fractions,
        model.canvas_tokens,
        context,
        chunk,
    )?;
    Ok(PlanStatistics {
        stats,
        lengths,
        chunk,
        arena_pages: split,
        admission,
        cap_pages: retention_pages(admission, chunk, context.device.page_tokens),
        pre,
        specs,
        asks,
        lowered,
    })
}

/// The aliases whose documents a filter before the joins computed.
pub fn filtered_before_joins(found: &Operators) -> BTreeSet<String> {
    let mut aliases = BTreeSet::new();
    for (alias, flags) in &found.filters_before_joins {
        if flags.iter().any(|before_joins| *before_joins) {
            aliases.insert(alias.clone());
        }
    }
    for (alias, tests) in &found.column_filters {
        if tests.iter().any(|test| test.before_joins) {
            aliases.insert(alias.clone());
        }
    }
    aliases
}

/// The expected live documents per alias after its filters run.
pub fn expected_rows_after_filters(
    found: &Operators,
    statistics: &PlanStatistics,
) -> BTreeMap<String, f64> {
    let mut expected: BTreeMap<String, f64> = statistics
        .stats
        .iter()
        .map(|(alias, stats)| (alias.clone(), stats.num_documents as f64))
        .collect();
    for ((alias, predicates), (_, flags)) in found.filters.iter().zip(&found.filters_before_joins) {
        for (predicate, before_joins) in predicates.iter().zip(flags) {
            if *before_joins {
                *expected.get_mut(alias).expect("a table") *=
                    crate::statistics::row_selectivity(predicate);
            }
        }
    }
    for (alias, tests) in &found.column_filters {
        for test in tests.iter().filter(|test| test.before_joins) {
            *expected.get_mut(alias).expect("a table") *= effective_selectivity(test.selectivity);
        }
    }
    expected
}
