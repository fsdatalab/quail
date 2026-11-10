//! Build operators in the order of the plan.
//! Set each scan's columns from the operators that read them.

use super::binding::Binder;
use super::placement::PlacedConditions;
use super::relation::JoinSpec;
use super::walk::{Kind, Walk};
use super::{ByAlias, lookup, note, push};
use crate::ai::{AiNode, Fetch, Semantics};
use crate::functions::column_call_kind;
use quail_planner::expressions::{
    asked_call, asked_refs, call_aliases, call_refs, column_references, equality_refs, value_refs,
};
use quail_planner::operators::{Operators, Scan, SemanticColumn, SemanticJoin, source_columns};
use quail_planner::physical::{
    ColumnRef, Equality, FilterPredicate, ModelCall, Projected, ValueExpression,
};
use std::collections::{BTreeMap, BTreeSet};

/// One operator on a table's stack, lowest first.
#[derive(Debug, Clone)]
pub(super) enum StackItem {
    /// The table's AI filters.
    Filters,
    /// A column operator: its kind and position in its list.
    Column(&'static str, usize),
    /// A test of answers: its position in the table's tests.
    Test(usize),
}

/// The region's operators: the bound terms, named, and placed where the
/// plan holds them.
pub(super) struct Assembly<'w, 'a> {
    pub(super) walk: &'w Walk<'a>,
    pub(super) binder: &'w mut Binder,
    /// alias -> its AI filters, in written order.
    pub(super) filters: ByAlias<FilterPredicate>,
    /// Each AI join and gate, by its node.
    pub(super) joins: BTreeMap<usize, JoinSpec>,
    pub(super) conditions: PlacedConditions,
    pub(super) found: Operators,
    /// Each table's stack, by alias, in scan order.
    pub(super) stacks: ByAlias<StackItem>,
    /// The aliases whose AI filters or tests are placed, in order.
    pub(super) noted: Vec<String>,
    /// The ON equalities placed on each join's id.
    pub(super) join_on: BTreeMap<usize, Vec<Equality>>,
    /// The column operators placed at each AI column's id: (kind, position).
    pub(super) columns_at: BTreeMap<usize, Vec<(&'static str, usize)>>,
}

impl<'w, 'a> Assembly<'w, 'a> {
    /// Whether `id` is an AI filter of one table.
    fn is_table_filter(&self, id: usize) -> bool {
        matches!(&self.walk.get(id).kind, Kind::Ai(node)
            if node.is_predicate()
                && node.semantics == Semantics::Rows
                && self.walk.call_tables(&node.call).len() < 2)
    }

    /// Whether the executor's stop key of `alias`'s AI.IF filters keeps
    /// a per-key fetch's rows: the executor stops a key once a document
    /// passes the table's filters, so the region asks only those filters
    /// and joins nothing.
    fn stops_per_key(&self, alias: &str) -> bool {
        self.walk.ids().all(|id| match &self.walk.get(id).kind {
            Kind::Ai(node) => {
                node.function() == "ai_filter"
                    && self.is_table_filter(id)
                    && self.call_tables(id).iter().eq([alias])
            }
            Kind::Join(_) => false,
            _ => true,
        })
    }

    fn call_tables(&self, id: usize) -> BTreeSet<String> {
        match &self.walk.get(id).kind {
            Kind::Ai(node) => self.walk.call_tables(&node.call),
            _ => BTreeSet::new(),
        }
    }

    /// The `index`th input of `id`, passing through projections.
    fn input(&self, id: usize, index: usize) -> Option<usize> {
        let mut below = *self.walk.get(id).inputs.get(index)?;
        loop {
            let visit = self.walk.get(below);
            match &visit.kind {
                Kind::PassThrough if !visit.inputs.is_empty() => below = visit.inputs[0],
                _ => return Some(below),
            }
        }
    }

    /// Place the region's operators in the plan's order; None when the
    /// plan holds an operator Quail's operators do not read here.
    pub(super) fn place(&mut self) -> Option<()> {
        for id in self.walk.ids() {
            // a column node's and an AI.IF filter's executors read a guard
            if let Kind::Ai(node) = &self.walk.get(id).kind
                && node.guard.is_some()
                && (node.function() == "ai_score" || node.semantics != Semantics::Rows)
            {
                return None;
            }
            match &self.walk.get(id).kind {
                Kind::Scan(table) => {
                    let alias = &self.walk.tables[*table].alias;
                    self.found.scans.push(Scan {
                        predicates: lookup(&self.conditions.scans, alias).to_vec(),
                        ..self.binder.scan(alias)
                    });
                    self.stacks.push((alias.clone(), Vec::new()));
                }
                Kind::Ai(node) if node.is_predicate() || node.semantics != Semantics::Rows => {
                    if !self.is_table_filter(id) {
                        self.place_join(id)?;
                    } else if !self.walk.get(id).parent.is_some_and(|parent| {
                        self.is_table_filter(parent)
                            && self.call_tables(parent) == self.call_tables(id)
                    }) {
                        self.place_filters(id)?;
                    }
                }
                // Scores returned as values use the existing projected-score
                // operator; this node keeps their SQL position during optimization.
                Kind::Ai(node) if node.function() == "ai_score" => {}
                Kind::Ai(node) => self.place_column(id, node)?,
                _ => {}
            }
        }
        if self.found.joins.len() != self.joins.len() {
            return None;
        }
        self.place_tests()
    }

    /// Place a run of AI filters of one table, `top` the last.
    fn place_filters(&mut self, top: usize) -> Option<()> {
        let read = self.call_tables(top);
        let mut run = vec![top];
        let mut below = self.walk.get(top).inputs[0];
        while self.is_table_filter(below) && self.call_tables(below) == read {
            run.push(below);
            below = self.walk.get(below).inputs[0];
        }
        let [alias] = <[String; 1]>::try_from(read.into_iter().collect::<Vec<_>>()).ok()?;
        let predicates = lookup(&self.filters, &alias).to_vec();
        if predicates.len() != run.len() {
            return None;
        }
        let before_joins = !self.walk.get(below).ai_join;
        let stop_key = run.iter().find_map(|id| match &self.walk.get(*id).kind {
            Kind::Ai(AiNode {
                fetch: Some(Fetch::PerKey { key, n }),
                ..
            }) if self.stops_per_key(&alias) => Some((key.clone(), *n)),
            _ => None,
        });
        // the executor counts a key's survivors among the rows it asks
        if stop_key.is_some()
            && predicates
                .iter()
                .any(|predicate| predicate.alternative.is_some())
        {
            return None;
        }
        for predicate in predicates {
            push(&mut self.found.filters, &alias, predicate);
            push(&mut self.found.filters_before_joins, &alias, before_joins);
        }
        if let Some((key, n)) = stop_key {
            self.found.stop_keys.insert(alias.clone(), key);
            if n > 1 {
                self.found.num_key_survivors.insert(alias.clone(), n as i64);
            }
        }
        note(&mut self.noted, &alias);
        push(&mut self.stacks, &alias, StackItem::Filters);
        Some(())
    }

    fn place_join(&mut self, id: usize) -> Option<()> {
        let spec = self.joins.get(&id)?;
        let read: BTreeSet<String> = call_aliases(asked_call(&spec.predicate))
            .into_iter()
            .collect();
        if read != self.call_tables(id) {
            return None;
        }
        let input = self.input(id, 0)?;
        let on = match &self.walk.get(input).kind {
            Kind::Join(_) => spec.on.clone(),
            _ => Vec::new(),
        };
        self.join_on.insert(input, on.clone());
        let mut outer = input;
        while let Kind::Join(_) = &self.walk.get(outer).kind {
            outer = self.input(outer, 0)?;
        }
        let mut outer_aliases: Vec<String> = Vec::new();
        for alias in &self.walk.get(outer).tables {
            note(&mut outer_aliases, alias);
        }
        self.found.joins.push(SemanticJoin {
            predicate: spec.predicate.clone(),
            semantics: spec.semantics.to_string(),
            selectivity: spec.selectivity,
            anchor: spec.anchor.clone(),
            on,
            outer_aliases,
        });
        Some(())
    }

    /// The column operators an AI column node runs: its call, or an
    /// extraction's call per field, each with its name.
    fn bound_columns(&mut self, node: &AiNode) -> Option<(&'static str, Vec<(ModelCall, String)>)> {
        let kind = column_call_kind(&node.call)?;
        let found = self.binder.answer(&node.answer)?;
        Some((kind, found.calls.clone()))
    }

    /// A column node's guard as a condition; the outer None when it does
    /// not bind.
    fn bound_guard(&mut self, node: &AiNode) -> Option<Option<ValueExpression>> {
        match &node.guard {
            None => Some(None),
            Some(guard) => {
                let read = self.walk.through(guard);
                self.binder.bind_condition(&read).ok().map(Some)
            }
        }
    }

    fn place_column(&mut self, id: usize, node: &AiNode) -> Option<()> {
        let read: Vec<String> = self.call_tables(id).into_iter().collect();
        let is_column = |kind: &Kind| matches!(kind, Kind::Ai(found) if !found.is_predicate() && found.semantics == Semantics::Rows);
        if read.len() == 2 {
            // a run of classifications of the same joined rows is placed
            // from its top, once
            let continues = |id: usize| {
                self.walk.get(id).parent.is_some_and(|parent| {
                    is_column(&self.walk.get(parent).kind)
                        && self.call_tables(parent) == self.call_tables(id)
                })
            };
            if continues(id) {
                return Some(());
            }
            let mut run = vec![(id, node)];
            let mut below = self.walk.get(id).inputs[0];
            while let Kind::Ai(next) = &self.walk.get(below).kind
                && is_column(&self.walk.get(below).kind)
                && self.call_tables(below) == self.call_tables(id)
            {
                run.push((below, *next));
                below = self.walk.get(below).inputs[0];
            }
            let above_joins = self.walk.get(below).ai_join;
            for (at, found) in run {
                let guard = self.bound_guard(found)?;
                let (_, columns) = self.bound_columns(found)?;
                for (call, name) in columns {
                    self.found.classifies.push(SemanticColumn {
                        call,
                        name,
                        above_joins,
                        guard: guard.clone(),
                        guard_selectivity: found
                            .guard
                            .as_ref()
                            .map(|found| crate::rules::guards::guard_selectivity(found, &[])),
                    });
                    let position = self.found.classifies.len() - 1;
                    self.columns_at
                        .entry(at)
                        .or_default()
                        .push(("label", position));
                }
            }
            return Some(());
        }
        let [alias] = <[String; 1]>::try_from(read).ok()?;
        let above_joins = self.walk.get(self.walk.get(id).inputs[0]).ai_join;
        let guard = self.bound_guard(node)?;
        let (kind, columns) = self.bound_columns(node)?;
        for (call, name) in columns {
            let list = self.list(kind);
            list.push(SemanticColumn {
                call,
                name,
                above_joins,
                guard: guard.clone(),
                guard_selectivity: node
                    .guard
                    .as_ref()
                    .map(|found| crate::rules::guards::guard_selectivity(found, &[])),
            });
            let position = list.len() - 1;
            self.columns_at
                .entry(id)
                .or_default()
                .push((kind, position));
            push(&mut self.stacks, &alias, StackItem::Column(kind, position));
        }
        Some(())
    }

    fn list(&mut self, kind: &str) -> &mut Vec<SemanticColumn> {
        match kind {
            "label" => &mut self.found.classifies,
            "extract" => &mut self.found.extracts,
            "map" => &mut self.found.maps,
            _ => &mut self.found.sae_features,
        }
    }

    pub(super) fn column(&self, kind: &str, position: usize) -> &SemanticColumn {
        let list = match kind {
            "label" => &self.found.classifies,
            "extract" => &self.found.extracts,
            "map" => &self.found.maps,
            _ => &self.found.sae_features,
        };
        &list[position]
    }
}

/// The columns each table's operators read, by alias, in first use
/// order, as Quail's scans load them.
#[derive(Default)]
struct Needed {
    by_alias: BTreeMap<String, Vec<String>>,
}

impl Needed {
    fn add(&mut self, references: Vec<ColumnRef>) {
        for reference in references {
            let columns = self.by_alias.entry(reference.alias).or_default();
            if !columns.contains(&reference.column) {
                columns.push(reference.column);
            }
        }
    }

    fn remove(&mut self, alias: &str, added: &[String]) {
        if let Some(columns) = self.by_alias.get_mut(alias) {
            columns.retain(|name| !added.contains(name));
        }
    }
}

/// Each scan's columns: the column values the operators above it read,
/// nearest the result first.
pub(super) fn push_down_columns(
    found: &mut Operators,
    walk: &Walk<'_>,
    assembly: &Assembly<'_, '_>,
) {
    // a guard may read an earlier AI filter's answer, which no table holds
    let unread = found.clone();
    let answers = unread.filter_answers();
    let mut valued: BTreeSet<(String, String)> = BTreeSet::new();
    let mut top = Needed::default();
    for column in &found.projection.columns {
        if let Projected::Expression(expression) = column {
            top.add(value_refs(expression));
        }
    }
    let mut value = |references: Vec<ColumnRef>| {
        for reference in references {
            valued.insert((reference.alias, reference.column));
        }
    };
    // the columns the region returns or computes
    for column in &found.projection.columns {
        if let Projected::Expression(expression) = column {
            value(value_refs(expression));
        }
    }
    for on in assembly.join_on.values() {
        for equality in on {
            value(equality_refs(equality));
        }
    }
    for scan in &found.scans {
        for predicate in &scan.predicates {
            value(column_references(predicate));
        }
    }
    for column in found.extracts.iter().chain(&found.maps) {
        value(call_refs(&column.call));
    }
    let guards: Vec<ColumnRef> = found
        .columns()
        .filter_map(|column| column.guard.as_ref())
        .chain(found.filters.iter().flat_map(|(_, predicates)| {
            predicates
                .iter()
                .flat_map(|predicate| predicate.guard.iter().chain(&predicate.alternative))
        }))
        .flat_map(column_references)
        .filter(|reference| {
            !answers.contains(&(reference.alias.as_str(), reference.column.as_str()))
        })
        .collect();
    value(guards.clone());
    let scan_ids: Vec<usize> = walk
        .ids()
        .filter(|id| matches!(walk.get(*id).kind, Kind::Scan(_)))
        .collect();
    for (index, scan_id) in scan_ids.iter().enumerate() {
        let alias = found.scans[index].alias.clone();
        let stack = &assembly.stacks[index].1;
        let mut needed = Needed {
            by_alias: top.by_alias.clone(),
        };
        let ancestors = &walk.get(*scan_id).ancestors;
        let mut joined_run: Vec<SemanticColumn> = Vec::new();
        let flush = |needed: &mut Needed, run: &mut Vec<SemanticColumn>| {
            for column in run.drain(..).rev() {
                needed.add(call_refs(&column.call));
                let owner = call_aliases(&column.call)
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                needed.remove(&owner, &column.added_columns());
            }
        };
        for id in ancestors {
            if *id == *scan_id {
                continue;
            }
            let visit = walk.get(*id);
            let on_stack = match &visit.kind {
                Kind::Ai(node) => {
                    let tables = walk.call_tables(&node.call);
                    tables.len() < 2 && tables.contains(&alias) && node.semantics == Semantics::Rows
                }
                _ => false,
            };
            if on_stack {
                continue;
            }
            if let Some(columns) = assembly.columns_at.get(id)
                && walk
                    .call_tables(match &visit.kind {
                        Kind::Ai(node) => &node.call,
                        _ => unreachable!("a column at an AI node"),
                    })
                    .len()
                    == 2
            {
                for (kind, position) in columns {
                    joined_run.push(assembly.column(kind, *position).clone());
                }
                continue;
            }
            flush(&mut needed, &mut joined_run);
            if let Some(spec) = assembly.joins.get(id) {
                needed.add(asked_refs(&spec.predicate));
            }
            if let Some(on) = assembly.join_on.get(id) {
                for equality in on {
                    needed.add(equality_refs(equality));
                }
            }
            if let Some(columns) = assembly.columns_at.get(id) {
                for (kind, position) in columns.iter().rev() {
                    let column = assembly.column(kind, *position).clone();
                    needed.add(call_refs(&column.call));
                    let owner = call_aliases(&column.call)
                        .into_iter()
                        .next()
                        .unwrap_or_default();
                    needed.remove(&owner, &column.added_columns());
                }
            }
        }
        flush(&mut needed, &mut joined_run);
        let computed = lookup(&assembly.conditions.computed, &alias);
        for item in stack.iter().rev() {
            match item {
                StackItem::Filters => {
                    if let Some((_, predicates)) =
                        found.filters.iter().find(|(found, _)| *found == alias)
                    {
                        for predicate in predicates {
                            needed.add(asked_refs(&predicate.expression));
                        }
                    }
                }
                StackItem::Column(kind, position) => {
                    let column = assembly.column(kind, *position).clone();
                    needed.add(call_refs(&column.call));
                    needed.remove(&alias, &column.added_columns());
                }
                StackItem::Test(index) => {
                    needed.add(column_references(&computed[*index].condition));
                }
            }
        }
        let scan = &found.scans[index];
        for predicate in &scan.predicates {
            needed.add(column_references(predicate));
        }
        // a per-key fetch groups the documents by its key's values
        if let Some(key) = found.stop_keys.get(&alias) {
            let columns = needed.by_alias.entry(alias.clone()).or_default();
            for column in key {
                if !columns.contains(column) {
                    columns.push(column.clone());
                }
            }
        }
        // a guard reads its columns' values where its operator runs
        for reference in guards.iter().filter(|reference| reference.alias == alias) {
            let columns = needed.by_alias.entry(alias.clone()).or_default();
            if !columns.contains(&reference.column) {
                columns.push(reference.column.clone());
            }
        }
        let sources = source_columns(&scan.column);
        let columns: Vec<String> = needed
            .by_alias
            .get(&alias)
            .map(|columns| {
                columns
                    .iter()
                    .filter(|column| {
                        !answers.contains(&(alias.as_str(), column.as_str()))
                            && (!sources.contains(column)
                                || valued.contains(&(alias.clone(), (*column).clone())))
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        found.scans[index].columns = columns;
    }
}
