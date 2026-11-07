//! The speed of light estimate of one query, as `quail/planner/sol`
//! computes it: every distinct document prefix computed once and kept
//! without limit, passes packed across the query, and exact survivors
//! from the caller's answer and label oracles.
//!
//! The data the estimate reads from the tables (token counts, the pairs
//! each join's equality conditions keep, stop key values) and the
//! oracles come from the caller, as do the tests of reference answers,
//! which Arrow evaluates.

use std::collections::{BTreeMap, BTreeSet};

use quail_cost::decode::DecodeCost;
use quail_cost::joins::{longest_join_input, partner_stream_work};
use quail_cost::quantity::Quantity;
use quail_cost::sae_features::sae_encoded_tokens;
use quail_cost::sol::speed_of_light;
use quail_cost::statistics::int_true_divide;
use quail_cost::work::{Work, ask, scan};
use quail_physical::{Condition, ModelCall, Prompt};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::context::PlanningContext;
use crate::expressions::{
    CompileError, SPAN_SUFFIX, aliased_call, call_aliases, condition_columns, condition_text,
    repr_str,
};
use crate::lowering::LoweredPrompt;
use crate::nodes::asked_call;
use crate::operators::{Operators, SemanticColumn};
use crate::plan_statistics::plan_prompts;
use crate::search::{
    ASK, StepKey, StepPricing, default_order_rule, label_calls, order_steps, resident_after,
    table_steps,
};

const FILTER: &str = "AI.IF";
const CLASSIFY: &str = "AI.CLASSIFY";
const EXTRACT: &str = "AI.EXTRACT";
const LABEL_IN: &str = "IN";
const ANSWER_TEST: &str = "ANSWER TEST";
const FEATURES: &str = "AI.SAE_FEATURES";
const MAP: &str = "AI.MAP";

/// A row assignment: each alias in a prompt and one of its row indexes.
pub type Assignment = [(String, i64)];

/// The reference label of a classification for one assignment.
pub type LabelOracle<'a> = dyn Fn(&Prompt, &Assignment) -> Result<String, CompileError> + 'a;
/// Which answers a one-column condition accepts.
pub type Accepts<'a> = dyn Fn(&Condition, &[Value]) -> Result<Vec<bool>, CompileError> + 'a;
/// Per document of an alias, the shared prefix tokens when only some rows compute.
pub type SubsetShared<'a> = dyn Fn(&str, &[i64]) -> Result<Vec<i64>, CompileError> + 'a;
/// alias -> (row, the partner rows its equality conditions keep).
pub type AllowedPairs = BTreeMap<String, Vec<(i64, Vec<i64>)>>;
/// row -> its partner rows.
type Partners = BTreeMap<i64, BTreeSet<i64>>;
/// Each alias's (partner label, anchor frame) tokens.
type LabelCounts = BTreeMap<String, (i64, i64)>;

/// The oracles and table callbacks the estimate calls.
pub struct Oracles<'a> {
    /// The exact answer of a prompt for one assignment: a filter's or
    /// join's bool, an extraction's text, a list of spans, or null, a
    /// map's text.
    pub answer: &'a dyn Fn(&Prompt, &Assignment) -> Result<Value, CompileError>,
    /// The reference label of a classification for one assignment.
    pub label: Option<&'a LabelOracle<'a>>,
    /// Which answers a one-column condition accepts.
    pub accepts: &'a Accepts<'a>,
    /// Per document of an alias, the shared prefix tokens when only the
    /// given rows compute.
    pub subset_shared: &'a SubsetShared<'a>,
}

/// One scanned alias's column and documents.
#[derive(Debug, Clone, Deserialize)]
pub struct AliasData {
    pub alias: String,
    pub column: String,
    pub tokens: Vec<i64>,
    pub shared: Vec<i64>,
}

/// What the estimate reads from the query's tables.
#[derive(Debug, Clone, Deserialize)]
pub struct EstimateInputs {
    pub aliases: Vec<AliasData>,
    /// alias -> each row's stop key value, for aliases whose filters stop
    /// a key at its first survivor.
    pub stop_groups: BTreeMap<String, Vec<Value>>,
    /// join written position -> alias -> (row, the partner rows its
    /// equality conditions keep).
    pub allowed_pairs: BTreeMap<usize, AllowedPairs>,
    pub order: Option<String>,
    pub credit_shared: bool,
    pub chunk_tokens: i64,
}

fn not_priced<T>(message: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError::new(format!(
        "NotImplementedError: {}",
        message.into()
    )))
}

fn round_to(value: f64, digits: usize) -> f64 {
    format!("{value:.digits$}").parse().expect("a float")
}

/// Refuse any operator the estimate cannot price, as `_refuse_unpriced`.
fn refuse_unpriced(found: &Operators) -> Result<(), CompileError> {
    let result = &found.result;
    if result.limit.is_some()
        && !(!result.order.is_empty() || result.distinct || result.offset != 0)
    {
        return not_priced("the speed of light estimate does not price LIMIT");
    }
    for (alias, predicates) in &found.filters {
        for predicate in predicates {
            if !matches!(&predicate.expression, quail_physical::Asked::Call(call) if call.kind == "boolean")
            {
                let shown = match &predicate.expression {
                    quail_physical::Asked::Call(call) => crate::expressions::explain_call(call),
                    quail_physical::Asked::Compare(test) => crate::expressions::test_text(test),
                };
                return not_priced(format!(
                    "the speed of light estimate does not price the filter {shown} on {}; it \
                     prices AI.IF and label filters",
                    repr_str(alias)
                ));
            }
        }
    }
    for (_, tests) in &found.column_filters {
        for test in tests {
            if test.call.kind == "features" {
                return not_priced(
                    "the speed of light estimate has no oracle for SAE feature tests",
                );
            }
            if test.call.kind == "map" {
                return not_priced(
                    "the speed of light estimate does not price a test of a map answer",
                );
            }
            let columns = condition_columns(&test.condition);
            let text = condition_text(&test.condition);
            if columns.len() > 1 {
                return not_priced(format!(
                    "the speed of light estimate does not price the test {text}, which reads \
                     more than one column"
                ));
            }
            if columns.iter().any(|column| column.ends_with(SPAN_SUFFIX)) {
                return not_priced(format!(
                    "the speed of light estimate does not price the test {text} of an \
                     answer's span; the answer oracle gives the answer's text only"
                ));
            }
        }
    }
    if !found.maps.is_empty() && !found.joins.is_empty() {
        return not_priced(
            "the speed of light estimate prices AI.MAP only in queries without joins",
        );
    }
    for join in &found.joins {
        let quail_physical::Asked::Call(call) = &join.predicate else {
            return not_priced(
                "the speed of light estimate does not price a join on RegularPredicate; it \
                 prices AI.IF joins",
            );
        };
        if call.kind != "boolean" {
            return not_priced(
                "the speed of light estimate does not price a join on ModelCall; it prices \
                 AI.IF joins",
            );
        }
        if join.semantics != "full" {
            return not_priced(format!(
                "the speed of light estimate does not price {} joins; it prices full joins",
                repr_str(&join.semantics)
            ));
        }
        if call_aliases(call).iter().collect::<BTreeSet<_>>().len() != 2 {
            return not_priced("the speed of light estimate prices binary join predicates");
        }
    }
    for column in &found.projections {
        let call = aliased_call(column, &[]).expect("a model call");
        if !["label", "extract", "features", "map"].contains(&call.kind.as_str()) {
            let name = match column {
                quail_physical::ValueExpression::Alias(alias) => alias.name.clone(),
                _ => String::new(),
            };
            return not_priced(format!(
                "the speed of light estimate does not price the {} column {}; it prices \
                 AI.CLASSIFY, AI.EXTRACT, AI.SAE_FEATURES, and AI.MAP columns",
                repr_str(&call.kind),
                repr_str(&name)
            ));
        }
    }
    let joined: Vec<BTreeSet<String>> = found
        .joins
        .iter()
        .map(|join| {
            call_aliases(asked_call(&join.predicate))
                .into_iter()
                .collect()
        })
        .collect();
    for (call, _) in label_calls(found) {
        let aliases = call_aliases(&call);
        if aliases.len() == 1 {
            continue;
        }
        let name = label_name(found, &call);
        let set: BTreeSet<String> = aliases.iter().cloned().collect();
        if aliases.len() != 2 || !joined.contains(&set) {
            let listed: Vec<String> = aliases.iter().map(|alias| repr_str(alias)).collect();
            return not_priced(format!(
                "the classification {} reads [{}]; the speed of light estimate prices a \
                 classification of one document or of one join's rows",
                repr_str(&name),
                listed.join(", ")
            ));
        }
        let tested = found
            .column_filters
            .iter()
            .any(|(_, tests)| tests.iter().any(|test| test.call == call));
        if tested {
            return not_priced(format!(
                "the speed of light estimate does not price a label filter on {}, a \
                 classification of joined rows",
                repr_str(&name)
            ));
        }
    }
    Ok(())
}

fn label_name(found: &Operators, call: &ModelCall) -> String {
    found
        .classifies
        .iter()
        .rev()
        .find(|column| column.call == *call)
        .map(|column| column.name.clone())
        .unwrap_or_default()
}

/// One work record of the join search and the choices that produced it.
#[derive(Debug, Clone)]
struct Candidate {
    work: Work,
    cached: BTreeSet<String>,
    relation_order: Vec<String>,
    steps: Vec<Value>,
}

/// The exact passing row pairs of one binary join predicate.
#[derive(Debug, Clone)]
struct PairRelation {
    left: String,
    right: String,
    pairs: BTreeSet<(i64, i64)>,
}

fn work_with(state_written: f64, state_read: f64) -> Work {
    Work {
        state_written: Quantity::Scalar(state_written),
        state_read: Quantity::Scalar(state_read),
        ..Work::default()
    }
}

fn scalar(value: f64) -> Quantity {
    Quantity::Scalar(value)
}

/// Each alias's partner label and anchor frame tokens, and the tail.
fn prompt_token_counts(lowered: &LoweredPrompt) -> Result<(LabelCounts, i64), CompileError> {
    let mut labels = BTreeMap::new();
    for (alias, label, frame) in &lowered.label_token_ids {
        labels.insert(alias.clone(), (label.len() as i64, frame.len() as i64));
    }
    if labels.is_empty() || lowered.prompt.kind == "score" {
        return Err(CompileError::new(
            "ValueError: a join prompt is needed to count its pieces",
        ));
    }
    Ok((labels, lowered.tail_token_ids.len() as i64))
}

/// Rows that occur in at least one exact satisfying assignment, as
/// `exact_survivors`.
fn exact_survivors(
    base_rows: &[(String, Vec<i64>)],
    relations: &[&PairRelation],
) -> BTreeMap<String, Vec<i64>> {
    // combine parallel relations, oriented by sorted alias pair
    let mut oriented: BTreeMap<(String, String), BTreeSet<(i64, i64)>> = BTreeMap::new();
    for relation in relations {
        let key = if relation.left <= relation.right {
            (relation.left.clone(), relation.right.clone())
        } else {
            (relation.right.clone(), relation.left.clone())
        };
        let pairs: BTreeSet<(i64, i64)> = if (relation.left.clone(), relation.right.clone()) == key
        {
            relation.pairs.clone()
        } else {
            relation.pairs.iter().map(|(l, r)| (*r, *l)).collect()
        };
        match oriented.get_mut(&key) {
            Some(found) => found.retain(|pair| pairs.contains(pair)),
            None => {
                oriented.insert(key, pairs);
            }
        }
    }
    let combined: Vec<PairRelation> = oriented
        .into_iter()
        .map(|((left, right), pairs)| PairRelation { left, right, pairs })
        .collect();
    let mut neighbors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for relation in &combined {
        neighbors
            .entry(relation.left.clone())
            .or_default()
            .insert(relation.right.clone());
        neighbors
            .entry(relation.right.clone())
            .or_default()
            .insert(relation.left.clone());
    }
    let base: BTreeMap<&String, &Vec<i64>> = base_rows.iter().map(|(a, r)| (a, r)).collect();
    let mut unseen: BTreeSet<String> = base_rows.iter().map(|(alias, _)| alias.clone()).collect();
    let mut output = BTreeMap::new();
    while let Some(root) = unseen.iter().next().cloned() {
        let mut component = BTreeSet::from([root.clone()]);
        let mut stack = vec![root.clone()];
        unseen.remove(&root);
        while let Some(alias) = stack.pop() {
            for other in neighbors.get(&alias).cloned().unwrap_or_default() {
                if unseen.remove(&other) {
                    component.insert(other.clone());
                    stack.push(other);
                }
            }
        }
        let inside: Vec<&PairRelation> = combined
            .iter()
            .filter(|relation| {
                component.contains(&relation.left) && component.contains(&relation.right)
            })
            .collect();
        let live = if inside.len() < component.len() {
            forest_live_rows(&base, &inside, &component)
        } else {
            cyclic_live_rows(&base, &inside, &component)
        };
        for (alias, rows) in live {
            output.insert(alias, rows.into_iter().collect());
        }
    }
    output
}

fn forest_live_rows(
    base: &BTreeMap<&String, &Vec<i64>>,
    relations: &[&PairRelation],
    aliases: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<i64>> {
    let mut live: BTreeMap<String, BTreeSet<i64>> = aliases
        .iter()
        .map(|alias| (alias.clone(), base[alias].iter().copied().collect()))
        .collect();
    let adjacency: Vec<(Partners, Partners)> = relations
        .iter()
        .map(|relation| {
            let (mut forward, mut backward) = (BTreeMap::new(), BTreeMap::new());
            for (left, right) in &relation.pairs {
                forward
                    .entry(*left)
                    .or_insert_with(BTreeSet::new)
                    .insert(*right);
                backward
                    .entry(*right)
                    .or_insert_with(BTreeSet::new)
                    .insert(*left);
            }
            (forward, backward)
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (relation, (forward, backward)) in relations.iter().zip(&adjacency) {
            let keep = |rows: &BTreeSet<i64>,
                        map: &BTreeMap<i64, BTreeSet<i64>>,
                        other: &BTreeSet<i64>| {
                rows.iter()
                    .copied()
                    .filter(|row| map.get(row).is_some_and(|found| !found.is_disjoint(other)))
                    .collect::<BTreeSet<i64>>()
            };
            let left = keep(&live[&relation.left], forward, &live[&relation.right]);
            let right = keep(&live[&relation.right], backward, &live[&relation.left]);
            if left != live[&relation.left] {
                live.insert(relation.left.clone(), left);
                changed = true;
            }
            if right != live[&relation.right] {
                live.insert(relation.right.clone(), right);
                changed = true;
            }
        }
    }
    live
}

fn cyclic_live_rows(
    base: &BTreeMap<&String, &Vec<i64>>,
    relations: &[&PairRelation],
    aliases: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<i64>> {
    let mut pair_sets: BTreeMap<(String, String), BTreeSet<(i64, i64)>> = BTreeMap::new();
    let mut neighbors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for relation in relations {
        pair_sets.insert(
            (relation.left.clone(), relation.right.clone()),
            relation.pairs.clone(),
        );
        pair_sets.insert(
            (relation.right.clone(), relation.left.clone()),
            relation.pairs.iter().map(|(l, r)| (*r, *l)).collect(),
        );
        neighbors
            .entry(relation.left.clone())
            .or_default()
            .insert(relation.right.clone());
        neighbors
            .entry(relation.right.clone())
            .or_default()
            .insert(relation.left.clone());
    }
    let mut found: BTreeMap<String, BTreeSet<i64>> = aliases
        .iter()
        .map(|alias| (alias.clone(), BTreeSet::new()))
        .collect();
    #[allow(clippy::too_many_arguments)]
    fn visit(
        assignment: &mut BTreeMap<String, i64>,
        aliases: &BTreeSet<String>,
        base: &BTreeMap<&String, &Vec<i64>>,
        neighbors: &BTreeMap<String, BTreeSet<String>>,
        pair_sets: &BTreeMap<(String, String), BTreeSet<(i64, i64)>>,
        found: &mut BTreeMap<String, BTreeSet<i64>>,
    ) {
        if assignment.len() == aliases.len() {
            for (alias, row) in assignment.iter() {
                found.get_mut(alias).expect("an alias").insert(*row);
            }
            return;
        }
        let empty = BTreeSet::new();
        let alias = aliases
            .iter()
            .filter(|name| !assignment.contains_key(*name))
            .min_by_key(|name| {
                let near = neighbors
                    .get(*name)
                    .unwrap_or(&empty)
                    .iter()
                    .filter(|other| assignment.contains_key(*other))
                    .count();
                (std::cmp::Reverse(near), base[name].len(), (*name).clone())
            })
            .expect("an unassigned alias")
            .clone();
        for row in base[&alias].iter() {
            let fits = neighbors.get(&alias).unwrap_or(&empty).iter().all(|other| {
                assignment.get(other).is_none_or(|bound| {
                    pair_sets[&(alias.clone(), other.clone())].contains(&(*row, *bound))
                })
            });
            if fits {
                assignment.insert(alias.clone(), *row);
                visit(assignment, aliases, base, neighbors, pair_sets, found);
                assignment.remove(&alias);
            }
        }
    }
    visit(
        &mut BTreeMap::new(),
        aliases,
        base,
        &neighbors,
        &pair_sets,
        &mut found,
    );
    found
}

struct Estimate<'a, 'c> {
    context: &'a PlanningContext<'c>,
    oracles: &'a Oracles<'a>,
    found: &'a Operators,
    inputs: EstimateInputs,
    aliases: BTreeMap<String, AliasData>,
    decode_cost: DecodeCost,
    pre: i64,
    canvas: i64,
    window: i64,
    chunk: i64,
    computed_rows_by_column: BTreeMap<String, BTreeSet<i64>>,
    answer_rows: f64,
    decode_passes: i64,
    map_passes: BTreeMap<(String, i64), i64>,
    max_pooling: BTreeSet<String>,
}

#[allow(clippy::too_many_arguments)]
fn stage(
    operator: &str,
    alias: &str,
    template: &str,
    written_position: Option<usize>,
    question_tokens: i64,
    selectivity: Option<f64>,
    rows: &[i64],
    passed: &[i64],
    work: &Work,
) -> Map<String, Value> {
    let mut record = Map::new();
    record.insert("operator".into(), Value::from(operator));
    record.insert("alias".into(), Value::from(alias));
    record.insert("template".into(), Value::from(template));
    record.insert("written_position".into(), json!(written_position));
    record.insert("question_tokens".into(), Value::from(question_tokens));
    record.insert("provided_selectivity".into(), json!(selectivity));
    record.insert("evaluated".into(), Value::from(rows.len()));
    record.insert("passed".into(), Value::from(passed.len()));
    record.insert(
        "selectivity".into(),
        Value::from(if rows.is_empty() {
            0.0
        } else {
            round_to(passed.len() as f64 / rows.len() as f64, 6)
        }),
    );
    record.insert("fresh_tokens".into(), Value::from(work.tokens.total()));
    record
}

impl<'a, 'c> Estimate<'a, 'c> {
    fn lower(&self, prompt: &Prompt) -> Result<LoweredPrompt, CompileError> {
        self.context.lowered(prompt)
    }

    fn assignment(pairs: &[(&str, i64)]) -> Vec<(String, i64)> {
        pairs
            .iter()
            .map(|(alias, row)| (alias.to_string(), *row))
            .collect()
    }

    fn passes(&self, prompt: &Prompt, assignment: &[(&str, i64)]) -> Result<bool, CompileError> {
        let answer = (self.oracles.answer)(prompt, &Self::assignment(assignment))?;
        Ok(match answer {
            Value::Bool(found) => found,
            Value::Null => false,
            Value::String(text) => !text.is_empty(),
            Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
            Value::Array(items) => !items.is_empty(),
            Value::Object(items) => !items.is_empty(),
        })
    }

    /// Compute one document's prefix for the first time in the query.
    fn first_use(&self, alias: &str, row: i64, suffix: i64) -> Work {
        let data = &self.aliases[alias];
        let prefix = self.pre + data.tokens[row as usize];
        let shared = data.shared[row as usize];
        if shared == 0 {
            return scan(&scalar(prefix as f64), &scalar(suffix as f64), self.window);
        }
        let resident = self.pre + shared;
        // the document saves its own state after reading the parent's
        &ask(
            &scalar(resident as f64),
            &scalar((prefix - resident + suffix) as f64),
            self.window,
        ) + &work_with(1.0, 0.0)
    }

    /// One question asked of every live document.
    fn question_work(&mut self, alias: &str, rows: &[i64], tail: i64, first: bool) -> Work {
        let column = self.aliases[alias].column.clone();
        let computed = self
            .computed_rows_by_column
            .entry(column.clone())
            .or_default()
            .clone();
        let suffix = tail + self.canvas;
        let mut work = Work::default();
        for row in rows {
            let step = if first && !(self.inputs.credit_shared && computed.contains(row)) {
                self.first_use(alias, *row, suffix)
            } else {
                let prefix = self.pre + self.aliases[alias].tokens[*row as usize];
                ask(&scalar(prefix as f64), &scalar(suffix as f64), self.window)
            };
            work = &work + &step;
        }
        if first {
            self.computed_rows_by_column
                .get_mut(&column)
                .expect("a column")
                .extend(rows.iter().copied());
        }
        work
    }

    #[allow(clippy::too_many_arguments)]
    fn join_stage_work(
        &self,
        anchor: &str,
        partners: &[String],
        survivors: &BTreeMap<String, Vec<i64>>,
        prompt: &LoweredPrompt,
        resident_rows: &[i64],
        cross_resident_rows: &BTreeSet<i64>,
        allowed: Option<&BTreeMap<i64, BTreeSet<i64>>>,
    ) -> Result<Work, CompileError> {
        let (labels, tail) = prompt_token_counts(prompt)?;
        let mut partner_rows: Vec<Vec<i64>> = vec![Vec::new()];
        for alias in partners {
            let mut next = Vec::new();
            for prefix in &partner_rows {
                for row in &survivors[alias] {
                    let mut extended = prefix.clone();
                    extended.push(*row);
                    next.push(extended);
                }
            }
            partner_rows = next;
        }
        let suffixes: Vec<i64> = partner_rows
            .iter()
            .map(|partner_row| {
                tail + self.canvas
                    + partners
                        .iter()
                        .zip(partner_row)
                        .enumerate()
                        .map(|(index, (alias, row))| {
                            self.aliases[alias].tokens[*row as usize]
                                + if index > 0 { labels[alias].0 } else { 0 }
                        })
                        .sum::<i64>()
            })
            .collect();
        let frame = labels[anchor].1 + partners.first().map_or(0, |partner| labels[partner].0);
        let mut resident: BTreeSet<i64> = resident_rows.iter().copied().collect();
        if self.inputs.credit_shared {
            resident.extend(cross_resident_rows.iter().copied());
        }
        let mut work = Work::default();
        for row in &survivors[anchor] {
            let streamed: Vec<f64> = match allowed {
                None => suffixes.iter().map(|suffix| *suffix as f64).collect(),
                Some(allowed) => {
                    let empty = BTreeSet::new();
                    let mine = allowed.get(row).unwrap_or(&empty);
                    suffixes
                        .iter()
                        .zip(&partner_rows)
                        .filter(|(_, partner_row)| mine.contains(&partner_row[0]))
                        .map(|(suffix, _)| *suffix as f64)
                        .collect()
                }
            };
            let prefix = self.pre + self.aliases[anchor].tokens[*row as usize];
            if resident.contains(row) {
                // the frame's state is saved for every partner to start from
                work = &(&work + &ask(&scalar(prefix as f64), &scalar(frame as f64), self.window))
                    + &work_with(1.0, 0.0);
            } else {
                work = &work + &self.first_use(anchor, *row, frame);
            }
            // every partner starts from the frame's saved state
            work = &(&work + &partner_stream_work((prefix + frame) as f64, &streamed, self.window))
                + &work_with(0.0, streamed.len() as f64);
        }
        Ok(work)
    }

    /// Per stage, the rows read when each key stops at its first survivor.
    fn stop_key_rows(
        &self,
        alias: &str,
        rows: &[i64],
        order: &[usize],
        predicates: &[quail_physical::FilterPredicate],
    ) -> Result<Vec<Vec<i64>>, CompileError> {
        let tokens = &self.aliases[alias].tokens;
        let keys = &self.inputs.stop_groups[alias];
        let mut groups: Vec<(&Value, Vec<i64>)> = Vec::new();
        for row in rows {
            let key = &keys[*row as usize];
            match groups.iter_mut().find(|(found, _)| *found == key) {
                Some((_, members)) => members.push(*row),
                None => groups.push((key, vec![*row])),
            }
        }
        let mut reached: Vec<Vec<i64>> = vec![Vec::new(); order.len()];
        for (_, members) in groups {
            let mut survivors = Vec::new();
            for row in &members {
                let mut all = true;
                for position in order {
                    if !self.passes(
                        &asked_call(&predicates[*position].expression).prompt,
                        &[(alias, *row)],
                    )? {
                        all = false;
                        break;
                    }
                }
                if all {
                    survivors.push(*row);
                }
            }
            if let Some(shortest) = survivors
                .iter()
                .copied()
                .min_by_key(|row| tokens[*row as usize])
            {
                for stage in &mut reached {
                    stage.push(shortest);
                }
                continue;
            }
            for row in &members {
                for (index, position) in order.iter().enumerate() {
                    reached[index].push(*row);
                    if !self.passes(
                        &asked_call(&predicates[*position].expression).prompt,
                        &[(alias, *row)],
                    )? {
                        break;
                    }
                }
            }
        }
        for stage in &mut reached {
            stage.sort();
        }
        Ok(reached)
    }

    fn answer_survivors(
        &self,
        condition: &Condition,
        rows: &[i64],
        answers: &BTreeMap<i64, Value>,
    ) -> Result<Vec<i64>, CompileError> {
        let values: Vec<Value> = rows.iter().map(|row| answers[row].clone()).collect();
        let kept = (self.oracles.accepts)(condition, &values)?;
        Ok(rows
            .iter()
            .zip(kept)
            .filter(|(_, keep)| *keep)
            .map(|(row, _)| *row)
            .collect())
    }

    /// Apply each alias's filters and column operators with exact answers.
    #[allow(clippy::type_complexity)]
    fn run_filters(
        &mut self,
    ) -> Result<(Vec<(String, Vec<i64>)>, Work, Vec<Map<String, Value>>), CompileError> {
        let rule = match &self.inputs.order {
            Some(order) => order.clone(),
            None => default_order_rule(self.found).0,
        };
        let mut survivors: Vec<(String, Vec<i64>)> = self
            .inputs
            .aliases
            .iter()
            .map(|data| (data.alias.clone(), (0..data.tokens.len() as i64).collect()))
            .collect();
        let mut work = Work::default();
        let mut stages = Vec::new();
        let scans: Vec<String> = self
            .found
            .scans
            .iter()
            .map(|scan| scan.alias.clone())
            .collect();
        for alias in scans {
            let predicates = self
                .found
                .filters
                .iter()
                .find(|(owner, _)| *owner == alias)
                .map(|(_, predicates)| predicates.clone())
                .unwrap_or_default();
            let asks: Vec<usize> = (0..predicates.len()).collect();
            let mut columns: Vec<(String, SemanticColumn)> = Vec::new();
            for column in self.found.columns() {
                if column.alias() != alias || call_aliases(&column.call).len() != 1 {
                    continue;
                }
                match columns.iter_mut().find(|(name, _)| *name == column.name) {
                    Some(slot) => slot.1 = column.clone(),
                    None => columns.push((column.name.clone(), column.clone())),
                }
            }
            if asks.is_empty() && columns.is_empty() {
                continue;
            }
            let tokens = &self.aliases[&alias].tokens;
            let mean = if tokens.is_empty() {
                0.0
            } else {
                int_true_divide(tokens.iter().sum::<i64>() as i128, tokens.len() as i128)
            };
            let steps = table_steps(
                self.found,
                &alias,
                &asks,
                &StepPricing::Estimate {
                    mean_tokens: self.pre as f64 + mean,
                    chunk: self.chunk,
                },
                self.context,
            )?;
            let live = survivors
                .iter()
                .find(|(owner, _)| *owner == alias)
                .expect("rows")
                .1
                .clone();
            let ordered: Vec<usize> = order_steps(&steps, &rule, live.len() as f64, self.context)?;
            let mut rows = live;
            let mut reached: Option<Vec<Vec<i64>>> = None;
            if self.inputs.stop_groups.contains_key(&alias)
                && ordered.iter().all(|index| steps[*index].kind == ASK)
            {
                let order: Vec<usize> = ordered
                    .iter()
                    .map(|index| match steps[*index].key {
                        StepKey::Position(position) => position,
                        StepKey::Name(_) => unreachable!(),
                    })
                    .collect();
                let found = self.stop_key_rows(&alias, &rows, &order, &predicates)?;
                if self.inputs.credit_shared {
                    // a skipped document computes no prefix for another
                    let shared = (self.oracles.subset_shared)(&alias, &found[0])?;
                    self.aliases.get_mut(&alias).expect("an alias").shared = shared;
                }
                reached = Some(found);
            }
            let mut previous: Option<usize> = None;
            for (place, index) in ordered.iter().enumerate() {
                if let Some(reached) = &reached {
                    rows = reached[place].clone();
                }
                let step = &steps[*index];
                let first = previous.is_none_or(|previous| !resident_after(&steps[previous], step));
                previous = Some(*index);
                let (passed, stage_work) = match &step.key {
                    StepKey::Position(position) => self.ask_stage(
                        &alias,
                        &predicates[*position],
                        *position,
                        &rows,
                        first,
                        &mut stages,
                    )?,
                    StepKey::Name(name) => {
                        let column = columns
                            .iter()
                            .find(|(found, _)| found == name)
                            .expect("a column")
                            .1
                            .clone();
                        match column.call.kind.as_str() {
                            "label" => {
                                self.classify_stage(&alias, &column, &rows, first, &mut stages)?
                            }
                            "extract" => {
                                self.extract_stage(&alias, &column, &rows, first, &mut stages)?
                            }
                            "features" => {
                                self.features_stage(&alias, &column, &rows, first, &mut stages)?
                            }
                            _ => self.map_stage(&alias, &column, &rows, first, &mut stages)?,
                        }
                    }
                };
                rows = passed;
                work = &work + &stage_work;
            }
            survivors
                .iter_mut()
                .find(|(owner, _)| *owner == alias)
                .expect("rows")
                .1 = rows;
        }
        Ok((survivors, work, stages))
    }

    fn ask_stage(
        &mut self,
        alias: &str,
        predicate: &quail_physical::FilterPredicate,
        position: usize,
        rows: &[i64],
        first: bool,
        stages: &mut Vec<Map<String, Value>>,
    ) -> Result<(Vec<i64>, Work), CompileError> {
        let prompt = &asked_call(&predicate.expression).prompt;
        let tail = self.lower(prompt)?.tail_token_ids.len() as i64;
        let work = self.question_work(alias, rows, tail, first);
        let mut passed = Vec::new();
        for row in rows {
            if self.passes(prompt, &[(alias, *row)])? {
                passed.push(*row);
            }
        }
        stages.push(stage(
            FILTER,
            alias,
            &prompt.template,
            Some(position),
            tail,
            predicate.selectivity,
            rows,
            &passed,
            &work,
        ));
        Ok((passed, work))
    }

    fn classify_stage(
        &mut self,
        alias: &str,
        column: &SemanticColumn,
        rows: &[i64],
        first: bool,
        stages: &mut Vec<Map<String, Value>>,
    ) -> Result<(Vec<i64>, Work), CompileError> {
        let call = &column.call;
        let tail = self.lower(&call.prompt)?.tail_token_ids.len() as i64;
        let work = self.question_work(alias, rows, tail, first);
        self.answer_rows += rows.len() as f64;
        let name = label_name(self.found, call);
        let mut record = stage(
            CLASSIFY,
            alias,
            &call.prompt.template,
            None,
            tail,
            None,
            rows,
            rows,
            &work,
        );
        record.insert("name".into(), Value::from(name.clone()));
        record.insert("answer_rows".into(), Value::from(rows.len()));
        stages.push(record);
        let mut rows = rows.to_vec();
        let tests = self
            .found
            .column_filters
            .iter()
            .find(|(owner, _)| owner == alias)
            .map(|(_, tests)| tests.clone())
            .unwrap_or_default();
        for test in tests {
            if test.call != *call {
                continue;
            }
            let label = self.oracles.label.expect("a label oracle");
            let mut answers = BTreeMap::new();
            for row in &rows {
                answers.insert(
                    *row,
                    Value::from(label(&call.prompt, &[(alias.to_string(), *row)])?),
                );
            }
            let passed = self.answer_survivors(&test.condition, &rows, &answers)?;
            let membership = matches!(&test.condition, Condition::Compare(compare) if compare.comparison == "in");
            let mut record = stage(
                if membership { LABEL_IN } else { ANSWER_TEST },
                alias,
                &call.prompt.template,
                Some(test.position),
                0,
                test.selectivity,
                &rows,
                &passed,
                &Work::default(),
            );
            record.insert("name".into(), Value::from(name.clone()));
            match &test.condition {
                Condition::Compare(compare) if membership => {
                    record.insert("accepted".into(), compare.value.clone());
                }
                _ => {
                    record.insert(
                        "condition".into(),
                        Value::from(condition_text(&test.condition)),
                    );
                }
            }
            stages.push(record);
            rows = passed;
        }
        Ok((rows, work))
    }

    fn features_stage(
        &mut self,
        alias: &str,
        column: &SemanticColumn,
        rows: &[i64],
        first: bool,
        stages: &mut Vec<Map<String, Value>>,
    ) -> Result<(Vec<i64>, Work), CompileError> {
        let call = &column.call;
        let tail = self.lower(&call.prompt)?.tail_token_ids.len() as i64;
        let mut work = self.question_work(alias, rows, tail, first);
        let tokens = &self.aliases[alias].tokens;
        let documents = if first {
            rows.iter().map(|row| tokens[*row as usize]).sum::<i64>() as f64
        } else {
            0.0
        };
        let encoded = sae_encoded_tokens(
            &call.pooling,
            rows.len() as f64,
            documents + rows.len() as f64 * (tail + if first { self.pre } else { 0 }) as f64,
            self.max_pooling.contains(alias),
        );
        work = &work
            + &Work {
                sae_encoded: scalar(encoded),
                ..Work::default()
            };
        let mut record = stage(
            FEATURES,
            alias,
            &call.prompt.template,
            None,
            tail,
            None,
            rows,
            rows,
            &work,
        );
        record.insert("name".into(), Value::from(column.name.clone()));
        stages.push(record);
        Ok((rows.to_vec(), work))
    }

    fn token_count(&self, text: &str) -> Result<i64, CompileError> {
        Ok((self.context.tokenizer)(text)?.len() as i64)
    }

    fn extract_stage(
        &mut self,
        alias: &str,
        column: &SemanticColumn,
        rows: &[i64],
        first: bool,
        stages: &mut Vec<Map<String, Value>>,
    ) -> Result<(Vec<i64>, Work), CompileError> {
        let call = &column.call;
        let tail = self.lower(&call.prompt)?.tail_token_ids.len() as i64;
        let mut work = self.question_work(alias, rows, tail, first);
        let mut answers = BTreeMap::new();
        for row in rows {
            answers.insert(
                *row,
                (self.oracles.answer)(&call.prompt, &[(alias.to_string(), *row)])?,
            );
        }
        let mut answer_rows = 0;
        for row in rows {
            let answer = &answers[row];
            let spans: Vec<String> = match answer {
                Value::String(text) => vec![text.clone()],
                Value::Array(items) => items
                    .iter()
                    .map(|item| item.as_str().unwrap_or_default().to_string())
                    .collect(),
                _ => Vec::new(),
            };
            let mut tokens = 0;
            for span in &spans {
                tokens += self.token_count(span)? + 1;
            }
            if tokens == 0 {
                tokens = 1;
            }
            let context = self.pre + self.aliases[alias].tokens[*row as usize] + tail;
            work = &work + &ask(&scalar(context as f64), &scalar(tokens as f64), self.window);
            answer_rows += tokens;
        }
        self.answer_rows += answer_rows as f64;
        let mut record = stage(
            EXTRACT,
            alias,
            &call.prompt.template,
            None,
            tail,
            None,
            rows,
            rows,
            &work,
        );
        record.insert("name".into(), Value::from(column.name.clone()));
        record.insert("answer_rows".into(), Value::from(answer_rows));
        stages.push(record);
        let mut rows = rows.to_vec();
        let tests = self
            .found
            .column_filters
            .iter()
            .find(|(owner, _)| owner == alias)
            .map(|(_, tests)| tests.clone())
            .unwrap_or_default();
        for test in tests {
            if test.name != column.name {
                continue;
            }
            let passed = self.answer_survivors(&test.condition, &rows, &answers)?;
            let mut record = stage(
                ANSWER_TEST,
                alias,
                &call.prompt.template,
                Some(test.position),
                0,
                test.selectivity,
                &rows,
                &passed,
                &Work::default(),
            );
            record.insert("name".into(), Value::from(column.name.clone()));
            record.insert(
                "condition".into(),
                Value::from(condition_text(&test.condition)),
            );
            stages.push(record);
            rows = passed;
        }
        Ok((rows, work))
    }

    fn map_stage(
        &mut self,
        alias: &str,
        column: &SemanticColumn,
        rows: &[i64],
        first: bool,
        stages: &mut Vec<Map<String, Value>>,
    ) -> Result<(Vec<i64>, Work), CompileError> {
        if self.window != 0 {
            return not_priced(
                "the speed of light estimate does not price AI.MAP on a model with sliding \
                 attention",
            );
        }
        let call = &column.call;
        let tail = self.lower(&call.prompt)?.tail_token_ids.len() as i64;
        let mut work = self.question_work(alias, rows, tail, first);
        let mut answer_rows = 0;
        for row in rows {
            let answer = (self.oracles.answer)(&call.prompt, &[(alias.to_string(), *row)])?;
            let Value::String(answer) = answer else {
                return Err(CompileError::new(format!(
                    "ValueError: the answer oracle gave no answer for map {} on {} row {row}",
                    repr_str(&column.name),
                    repr_str(alias)
                )));
            };
            // text can tokenize longer than the tokens the model wrote
            let count = self.token_count(&answer)?;
            let written = (if count == 0 { 1 } else { count }).min(call.max_answer_tokens);
            let fed = written - 1;
            let document = self.pre + self.aliases[alias].tokens[*row as usize];
            let key = (alias.to_string(), *row);
            let read_before = self.map_passes.get(&key).copied().unwrap_or(0);
            let document_reads = if self.inputs.credit_shared {
                (fed - read_before).max(0)
            } else {
                fed
            };
            self.map_passes.insert(key, read_before.max(fed));
            let (fed_f, document_f, tail_f) = (fed as f64, document as f64, tail as f64);
            work = &work
                + &Work {
                    tokens: scalar(fed_f),
                    pairs: scalar(fed_f * (document_f + tail_f) + fed_f * (fed_f + 1.0) / 2.0),
                    kv_written: scalar(fed_f),
                    kv_read: scalar(
                        document_reads as f64 * document_f
                            + fed_f * tail_f
                            + fed_f * (fed_f - 1.0) / 2.0,
                    ),
                    ..Work::default()
                };
            self.decode_passes = self.decode_passes.max(fed);
            answer_rows += written;
        }
        self.answer_rows += answer_rows as f64;
        let mut record = stage(
            MAP,
            alias,
            &call.prompt.template,
            None,
            tail,
            None,
            rows,
            rows,
            &work,
        );
        record.insert("name".into(), Value::from(column.name.clone()));
        record.insert("answer_rows".into(), Value::from(answer_rows));
        stages.push(record);
        Ok((rows.to_vec(), work))
    }

    fn longest(&self, alias: &str, survivors: &BTreeMap<String, Vec<i64>>) -> i64 {
        survivors[alias]
            .iter()
            .map(|row| self.aliases[alias].tokens[*row as usize])
            .max()
            .unwrap_or(0)
    }

    fn anchor_fits(
        &self,
        prompt: &LoweredPrompt,
        anchor: &str,
        stage_aliases: &[String],
        survivors: &BTreeMap<String, Vec<i64>>,
    ) -> Result<bool, CompileError> {
        let (labels, tail) = prompt_token_counts(prompt)?;
        let partners: Vec<&String> = stage_aliases
            .iter()
            .filter(|alias| *alias != anchor)
            .collect();
        let frame = labels[anchor].1;
        if survivors[anchor].is_empty()
            || partners
                .iter()
                .any(|partner| survivors[*partner].is_empty())
        {
            return Ok(self.pre + self.longest(anchor, survivors) + frame <= self.chunk);
        }
        let pieces: Vec<(i64, i64)> = partners
            .iter()
            .map(|partner| (labels[*partner].0, self.longest(partner, survivors)))
            .collect();
        Ok(longest_join_input(
            self.pre,
            self.longest(anchor, survivors),
            frame,
            tail,
            &pieces,
        ) <= self.chunk)
    }

    fn cross_resident(
        &self,
        anchor: &str,
        survivors: &BTreeMap<String, Vec<i64>>,
        cached: &BTreeSet<String>,
    ) -> BTreeSet<i64> {
        let column = &self.aliases[anchor].column;
        let mut rows: BTreeSet<i64> = self
            .computed_rows_by_column
            .get(column)
            .cloned()
            .unwrap_or_default();
        for other in cached {
            if other != anchor && self.aliases[other].column == *column {
                rows.extend(survivors[other].iter().copied());
            }
        }
        rows
    }

    fn joined_classification_work(
        &self,
        call: &ModelCall,
        anchor: &str,
        partner: &str,
        survivors: &BTreeMap<String, Vec<i64>>,
        relation: &PairRelation,
    ) -> Result<(Work, Value), CompileError> {
        let mut kept: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for (left, right) in &relation.pairs {
            let (anchor_row, partner_row) = if relation.left == anchor {
                (*left, *right)
            } else {
                (*right, *left)
            };
            kept.entry(anchor_row).or_default().insert(partner_row);
        }
        let partner_survivors: BTreeSet<i64> = survivors[partner].iter().copied().collect();
        let anchor_rows: Vec<i64> = survivors[anchor]
            .iter()
            .copied()
            .filter(|row| {
                kept.get(row)
                    .is_some_and(|found| !found.is_disjoint(&partner_survivors))
            })
            .collect();
        let local: BTreeMap<String, Vec<i64>> = BTreeMap::from([
            (anchor.to_string(), anchor_rows.clone()),
            (partner.to_string(), survivors[partner].clone()),
        ]);
        let work = self.join_stage_work(
            anchor,
            &[partner.to_string()],
            &local,
            &self.lower(&call.prompt)?,
            &anchor_rows,
            &BTreeSet::new(),
            Some(&kept),
        )?;
        let pairs: usize = anchor_rows
            .iter()
            .map(|row| kept[row].intersection(&partner_survivors).count())
            .sum();
        Ok((
            work.clone(),
            json!({
                "name": label_name(self.found, call),
                "template": call.prompt.template,
                "pairs": pairs,
                "fresh_tokens": work.tokens.total(),
                "answer_rows": pairs,
            }),
        ))
    }

    fn latency_rows(&self, candidate: &Candidate) -> f64 {
        self.answer_rows
            + candidate
                .steps
                .iter()
                .flat_map(|step| {
                    step["classifications"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                })
                .map(|record| record["answer_rows"].as_f64().unwrap_or(0.0))
                .sum::<f64>()
    }

    fn latency(&self, candidate: &Candidate) -> quail_cost::sol::SpeedOfLight {
        speed_of_light(
            &candidate.work,
            &self.context.model,
            &self.context.device,
            self.chunk,
            self.latency_rows(candidate),
            self.decode_passes,
        )
    }

    fn seconds(&self, candidate: &Candidate) -> f64 {
        self.decode_cost
            .lower_bound_seconds(&candidate.work, self.latency(candidate).passes as f64)
    }
}

fn allowed_map(rows: &[(i64, Vec<i64>)]) -> BTreeMap<i64, BTreeSet<i64>> {
    rows.iter()
        .map(|(row, partners)| (*row, partners.iter().copied().collect()))
        .collect()
}

/// The ideal time of one query on one device, as `speed_of_light_estimate`:
/// its record, as `SpeedOfLightEstimate.as_dict` writes it.
///
/// # Errors
///
/// The plan holds an operator the estimate does not price (the message
/// starts with "NotImplementedError: "), or a callback fails.
pub fn speed_of_light_estimate(
    found: &Operators,
    context: &PlanningContext,
    inputs: EstimateInputs,
    oracles: &Oracles,
) -> Result<Value, CompileError> {
    refuse_unpriced(found)?;
    if !label_calls(found).is_empty() && oracles.label.is_none() {
        return Err(CompileError::new(
            "ValueError: the query classifies documents; pass label=(prompt, assignment) -> str \
             giving each row's reference label",
        ));
    }
    let mut pre = 0;
    for prompt in plan_prompts(found) {
        if prompt.kind != "score" {
            pre = context.lowered(&prompt)?.preamble_token_ids.len() as i64;
            break;
        }
    }
    let max_pooling = found
        .sae_features
        .iter()
        .filter(|column| column.call.pooling == "max")
        .map(SemanticColumn::alias)
        .collect();
    let aliases = inputs
        .aliases
        .iter()
        .map(|data| (data.alias.clone(), data.clone()))
        .collect();
    let mut estimate = Estimate {
        context,
        oracles,
        decode_cost: DecodeCost::from_specs(&context.model, &context.device),
        pre,
        canvas: context.model.canvas_tokens,
        window: context.model.sliding_window,
        chunk: inputs.chunk_tokens,
        found,
        aliases,
        inputs,
        computed_rows_by_column: BTreeMap::new(),
        answer_rows: 0.0,
        decode_passes: 0,
        map_passes: BTreeMap::new(),
        max_pooling,
    };
    let (survivors, filter_work, filter_stages) = estimate.run_filters()?;
    let resident: BTreeSet<String> = filter_stages
        .iter()
        .filter(|stage| {
            let operator = stage["operator"].as_str().unwrap_or_default();
            operator != LABEL_IN && operator != ANSWER_TEST
        })
        .map(|stage| stage["alias"].as_str().unwrap_or_default().to_string())
        .collect();
    let (best, num_states, generated, finals) =
        search_joins(&estimate, &survivors, resident, filter_work)?;
    let latency = estimate.latency(&best);
    let seconds = estimate.seconds(&best);
    let work = &best.work;
    let device = &context.device;
    Ok(json!({
        "model": context.model.name,
        "device": device.name,
        "chunk_tokens": estimate.chunk,
        "credit_shared_prefixes": estimate.inputs.credit_shared,
        "sol_s": seconds,
        "t_compute": latency.compute(),
        "t_memory": latency.memory(),
        "passes": latency.passes,
        "bytes_moved": latency.bytes_moved(),
        "components": latency.components.iter().map(|component| {
            let (compute, memory) = (
                component.compute_seconds.total(),
                component.memory_seconds.total(),
            );
            json!({
                "name": component.component.name,
                "precision": component.component.precision,
                "flops": component.component.flops.total(),
                "bytes_moved": component.component.bytes_moved.total(),
                "t_compute": compute,
                "t_memory": memory,
                "seconds": component.seconds(),
                "bound_by": if compute >= memory { "compute" } else { "memory" },
            })
        }).collect::<Vec<_>>(),
        "tokens": work.tokens.total(),
        "pairs": work.pairs.total(),
        "sliding_pairs": work.sliding_pairs.total(),
        "sliding_kv_read": work.sliding_kv_read.total(),
        "kv_written": work.kv_written.total(),
        "kv_read": work.kv_read.total(),
        "work": {
            "tokens": work.tokens.total(),
            "pairs": work.pairs.total(),
            "kv_written": work.kv_written.total(),
            "kv_read": work.kv_read.total(),
            "sliding_pairs": work.sliding_pairs.total(),
            "sliding_kv_read": work.sliding_kv_read.total(),
            "state_written": work.state_written.total(),
            "state_read": work.state_read.total(),
            "sae_encoded": work.sae_encoded.total(),
        },
        "answer_rows": estimate.latency_rows(&best),
        "decode_passes": estimate.decode_passes,
        "usd_per_hour": device.usd_per_hour,
        "usd_per_query": seconds * device.usd_per_hour / 3600.0,
        "alias_columns": estimate.inputs.aliases.iter().map(|data| (data.alias.clone(), Value::from(data.column.clone()))).collect::<Map<String, Value>>(),
        "documents_by_alias": estimate.inputs.aliases.iter().map(|data| (data.alias.clone(), Value::from(data.tokens.len()))).collect::<Map<String, Value>>(),
        "post_filter_counts": survivors.iter().map(|(alias, rows)| (alias.clone(), Value::from(rows.len()))).collect::<Map<String, Value>>(),
        "filter_stages": filter_stages,
        "join_stages": best.steps,
        "relation_order": best.relation_order,
        "cached_prefixes": best.cached.iter().cloned().collect::<Vec<String>>(),
        "search": {"dp_states": num_states, "dp_records_generated": generated, "dp_final_records": finals},
    }))
}

type StateKey = (BTreeSet<String>, BTreeSet<usize>, BTreeSet<String>);

/// Search every feasible left deep plan over the exact filtered rows, as
/// `search_joins_sol` with `search_left_deep`.
fn search_joins(
    estimate: &Estimate,
    base_rows: &[(String, Vec<i64>)],
    resident: BTreeSet<String>,
    base_work: Work,
) -> Result<(Candidate, usize, usize, usize), CompileError> {
    let joins = &estimate.found.joins;
    let mut relations: Vec<PairRelation> = Vec::new();
    let mut edges: Vec<BTreeSet<String>> = Vec::new();
    let base: BTreeMap<&String, &Vec<i64>> = base_rows.iter().map(|(a, r)| (a, r)).collect();
    for join in joins {
        let call = asked_call(&join.predicate);
        let mut sides: Vec<String> = Vec::new();
        for alias in call_aliases(call) {
            if !sides.contains(&alias) {
                sides.push(alias);
            }
        }
        let (left, right) = (sides[0].clone(), sides[1].clone());
        let allowed = estimate
            .inputs
            .allowed_pairs
            .get(&relations.len())
            .and_then(|by_alias| by_alias.get(&left))
            .map(|rows| allowed_map(rows));
        let mut pairs = BTreeSet::new();
        for left_row in base[&left] {
            for right_row in base[&right] {
                let permitted = allowed.as_ref().is_none_or(|allowed| {
                    allowed
                        .get(left_row)
                        .is_some_and(|found| found.contains(right_row))
                });
                if permitted
                    && estimate.passes(&call.prompt, &[(&left, *left_row), (&right, *right_row)])?
                {
                    pairs.insert((*left_row, *right_row));
                }
            }
        }
        edges.push([left.clone(), right.clone()].into_iter().collect());
        relations.push(PairRelation { left, right, pairs });
    }
    let mut joined_classifications: Vec<(BTreeSet<String>, Vec<ModelCall>)> = Vec::new();
    for (call, _) in label_calls(estimate.found) {
        let aliases: BTreeSet<String> = call_aliases(&call).into_iter().collect();
        if aliases.len() != 2 {
            continue;
        }
        match joined_classifications
            .iter_mut()
            .find(|(found, _)| *found == aliases)
        {
            Some((_, calls)) => calls.push(call),
            None => joined_classifications.push((aliases, vec![call])),
        }
    }
    let mut survivors_cache: BTreeMap<BTreeSet<usize>, BTreeMap<String, Vec<i64>>> =
        BTreeMap::new();
    let mut survivors_for = |applied: &BTreeSet<usize>| -> BTreeMap<String, Vec<i64>> {
        survivors_cache
            .entry(applied.clone())
            .or_insert_with(|| {
                let chosen: Vec<&PairRelation> =
                    applied.iter().map(|index| &relations[*index]).collect();
                exact_survivors(base_rows, &chosen)
            })
            .clone()
    };
    let aliases: Vec<String> = estimate
        .found
        .scans
        .iter()
        .map(|scan| scan.alias.clone())
        .collect();
    let all_aliases: BTreeSet<String> = aliases.iter().cloned().collect();
    let mut states: Vec<(StateKey, Vec<Candidate>)> = Vec::new();
    for alias in &aliases {
        let relations_now: BTreeSet<String> = [alias.clone()].into_iter().collect();
        states.push((
            (relations_now, BTreeSet::new(), resident.clone()),
            vec![Candidate {
                work: base_work.clone(),
                cached: resident.clone(),
                relation_order: vec![alias.clone()],
                steps: Vec::new(),
            }],
        ));
    }
    let mut generated = aliases.len();
    for applied_count in 0..edges.len() {
        let current: Vec<(StateKey, Vec<Candidate>)> = states
            .iter()
            .filter(|(key, _)| key.1.len() == applied_count)
            .cloned()
            .collect();
        for ((joined, applied, cached), frontier) in current {
            let available: Vec<usize> = edges
                .iter()
                .enumerate()
                .filter(|(index, ends)| {
                    !applied.contains(index)
                        && (ends.is_subset(&joined)
                            || (!ends.is_disjoint(&joined)
                                && ends.difference(&joined).count() == 1))
                })
                .map(|(index, _)| index)
                .collect();
            for index in available {
                let added: Vec<String> = edges[index].difference(&joined).cloned().collect();
                let next_joined: BTreeSet<String> = joined.union(&edges[index]).cloned().collect();
                let mut next_applied = applied.clone();
                next_applied.insert(index);
                let survivors = survivors_for(&applied);
                let relation = &relations[index];
                let stage_aliases = vec![relation.left.clone(), relation.right.clone()];
                let join = &joins[index];
                let join_prompt = &asked_call(&join.predicate).prompt;
                let classifications: Vec<ModelCall> = joined_classifications
                    .iter()
                    .find(|(found, _)| {
                        *found == stage_aliases.iter().cloned().collect::<BTreeSet<_>>()
                    })
                    .map(|(_, calls)| calls.clone())
                    .unwrap_or_default();
                for anchor in &stage_aliases {
                    let mut fits = true;
                    let mut prompts = vec![join_prompt.as_ref().clone()];
                    prompts.extend(
                        classifications
                            .iter()
                            .map(|call| call.prompt.as_ref().clone()),
                    );
                    for prompt in &prompts {
                        if !estimate.anchor_fits(
                            &estimate.lower(prompt)?,
                            anchor,
                            &stage_aliases,
                            &survivors,
                        )? {
                            fits = false;
                            break;
                        }
                    }
                    if !fits {
                        continue;
                    }
                    let partners: Vec<String> = stage_aliases
                        .iter()
                        .filter(|alias| *alias != anchor)
                        .cloned()
                        .collect();
                    let allowed = estimate
                        .inputs
                        .allowed_pairs
                        .get(&index)
                        .and_then(|by_alias| by_alias.get(anchor))
                        .map(|rows| allowed_map(rows));
                    let mut stage_work = estimate.join_stage_work(
                        anchor,
                        &partners,
                        &survivors,
                        &estimate.lower(join_prompt)?,
                        if cached.contains(anchor) {
                            &survivors[anchor]
                        } else {
                            &[]
                        },
                        &estimate.cross_resident(anchor, &survivors, &cached),
                        allowed.as_ref(),
                    )?;
                    let mut classified = Vec::new();
                    for call in &classifications {
                        let (call_work, record) = estimate.joined_classification_work(
                            call,
                            anchor,
                            &partners[0],
                            &survivors,
                            relation,
                        )?;
                        stage_work = &stage_work + &call_work;
                        classified.push(record);
                    }
                    let mut next_cached = cached.clone();
                    next_cached.insert(anchor.clone());
                    let left_rows: BTreeSet<i64> =
                        survivors[&relation.left].iter().copied().collect();
                    let right_rows: BTreeSet<i64> =
                        survivors[&relation.right].iter().copied().collect();
                    let passing = relation
                        .pairs
                        .iter()
                        .filter(|(left, right)| {
                            left_rows.contains(left) && right_rows.contains(right)
                        })
                        .count();
                    let evaluated: usize = match &allowed {
                        None => stage_aliases
                            .iter()
                            .map(|alias| survivors[alias].len())
                            .product(),
                        Some(allowed) => survivors[anchor]
                            .iter()
                            .map(|row| {
                                survivors[&partners[0]]
                                    .iter()
                                    .filter(|partner| {
                                        allowed
                                            .get(row)
                                            .is_some_and(|found| found.contains(partner))
                                    })
                                    .count()
                            })
                            .sum(),
                    };
                    let step = json!({
                        "template": join_prompt.template,
                        "written_position": index,
                        "added_alias": added.first(),
                        "aliases": stage_aliases,
                        "anchor": anchor,
                        "partners": partners,
                        "evaluated_pairs": evaluated,
                        "passing_pairs": passing,
                        "fresh_tokens": stage_work.tokens.total(),
                        "classifications": classified,
                        "cached_prefixes_after": next_cached.iter().cloned().collect::<Vec<_>>(),
                    });
                    let key = (
                        next_joined.clone(),
                        next_applied.clone(),
                        next_cached.clone(),
                    );
                    let position = match states.iter().position(|(found, _)| *found == key) {
                        Some(position) => position,
                        None => {
                            states.push((key, Vec::new()));
                            states.len() - 1
                        }
                    };
                    for old in &frontier {
                        generated += 1;
                        let candidate = Candidate {
                            work: &old.work + &stage_work,
                            cached: next_cached.clone(),
                            relation_order: old
                                .relation_order
                                .iter()
                                .cloned()
                                .chain(added.iter().cloned())
                                .collect(),
                            steps: old
                                .steps
                                .iter()
                                .cloned()
                                .chain(std::iter::once(step.clone()))
                                .collect(),
                        };
                        let target = &mut states[position].1;
                        if target
                            .iter()
                            .any(|existing| existing.work.dominates(&candidate.work))
                        {
                            continue;
                        }
                        target.retain(|existing| !candidate.work.dominates(&existing.work));
                        target.push(candidate);
                    }
                }
            }
        }
    }
    let finals: Vec<Candidate> = states
        .iter()
        .filter(|(key, _)| key.1.len() == edges.len() && all_aliases.is_subset(&key.0))
        .flat_map(|(_, frontier)| frontier.iter().cloned())
        .collect();
    if finals.is_empty() {
        return Err(CompileError::new(
            "ValueError: query join graph has no connected left deep plan",
        ));
    }
    let rank = |candidate: &Candidate| {
        (
            estimate.seconds(candidate),
            candidate.work.tokens.total(),
            candidate.work.pairs.total(),
            candidate.work.kv_written.total(),
            candidate.work.kv_read.total(),
            candidate.relation_order.clone(),
        )
    };
    let mut best = 0;
    for index in 1..finals.len() {
        if rank(&finals[index]).partial_cmp(&rank(&finals[best])) == Some(std::cmp::Ordering::Less)
        {
            best = index;
        }
    }
    let count = finals.len();
    Ok((finals[best].clone(), states.len(), generated, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation(left: &str, right: &str, pairs: &[(i64, i64)]) -> PairRelation {
        PairRelation {
            left: left.to_string(),
            right: right.to_string(),
            pairs: pairs.iter().copied().collect(),
        }
    }

    #[test]
    fn survivors_of_a_tree_and_of_a_cycle_are_the_rows_some_assignment_keeps() {
        let base: Vec<(String, Vec<i64>)> = ["a", "b", "c"]
            .iter()
            .map(|alias| (alias.to_string(), vec![0, 1]))
            .collect();
        let tree = [
            relation("a", "b", &[(0, 0), (1, 1)]),
            relation("b", "c", &[(0, 0)]),
        ];
        let cycle = [
            relation("a", "b", &[(0, 0), (1, 1)]),
            relation("b", "c", &[(0, 0), (1, 1)]),
            relation("a", "c", &[(0, 1), (1, 0)]),
        ];
        for (relations, expected) in [(&tree[..], vec![0]), (&cycle[..], vec![])] {
            let found = exact_survivors(&base, &relations.iter().collect::<Vec<_>>());
            for alias in ["a", "b", "c"] {
                assert_eq!(found[alias], expected, "{alias}");
            }
        }
    }
}
