//! Build operators in the order of the plan.
//! Set each scan's columns from the operators that read them.

use super::binding::{Binder, check_labels, unsupported};
use super::placement::{PlacedConditions, Term};
use super::relation::{JoinSpec, Relation, check_calls, check_joins};
use super::walk::{Kind, Walk};
use super::{ByAlias, lookup, note, push};
use crate::ai::AiNode;
use crate::rules::guards::guard_selectivity;
use quail_planner::error::CompileError;
use quail_planner::expressions::{
    aliased_call, asked_call, asked_refs, call_aliases, call_refs, column_name, column_references,
    condition_text, equality_refs, value_refs,
};
use quail_planner::operators::{Operators, Scan, SemanticColumn, SemanticJoin, source_columns};
use quail_planner::physical::{ColumnRef, FilterPredicate, ModelCall, Projected, ValueExpression};
use std::collections::{BTreeMap, BTreeSet};

/// The region's operators: the bound terms, named, and placed where the
/// plan holds them.
pub(super) struct Assembly<'w, 'a> {
    walk: &'w Walk<'a>,
    binder: &'w mut Binder,
    /// alias -> its AI filters, in written order.
    filters: ByAlias<FilterPredicate>,
    /// Each AI join and gate, by its node.
    joins: BTreeMap<usize, JoinSpec>,
    conditions: PlacedConditions,
    pub(super) found: Operators,
    /// alias -> the column operators of that table, as (kind, position
    /// in its list), in the plan's order.
    placed: ByAlias<(&'static str, usize)>,
    /// The aliases whose AI filters or tests are placed, in order.
    noted: Vec<String>,
}

impl<'w, 'a> Assembly<'w, 'a> {
    /// Name each column operator and place the relation's conditions.
    ///
    /// `columns` is the SELECT list.
    ///
    /// # Errors
    ///
    /// A term does not fit the query's tables, or the query has no AI
    /// predicate where it needs one.
    pub(super) fn new(
        walk: &'w Walk<'a>,
        binder: &'w mut Binder,
        relation: Relation,
        columns: &[ValueExpression],
    ) -> Result<Self, CompileError> {
        let Relation {
            filters,
            joins,
            label_tests,
            mut conditions,
            equalities,
            join_filters,
            tables,
            ..
        } = relation;
        let projected: Vec<(&ModelCall, String)> = columns
            .iter()
            .filter_map(|column| Some((aliased_call(column, &[])?, column_name(column, true))))
            .collect();
        let specs: Vec<&JoinSpec> = joins.values().collect();
        check_joins(&equalities, &join_filters, &tables, &specs, &projected)?;
        check_calls(&filters, &specs, &projected)?;
        let (names, computed) = binder.column_operators();
        let mut terms: Vec<Term> = conditions
            .drain(..)
            .map(|(condition, selectivity)| Term::new(condition, selectivity, None))
            .collect();
        for (alias, found) in &label_tests {
            for (name, labels, selectivity) in found {
                let (call, _) = names
                    .iter()
                    .find(|(_, found)| found == name)
                    .expect("a named label");
                check_labels(name, labels, call)?;
                terms.push(Term::new(
                    binder.label_condition(alias, name, labels)?,
                    *selectivity,
                    Some(labels.clone()),
                ));
            }
        }
        let conditions = PlacedConditions::place(terms, &computed)?;
        Ok(Self {
            walk,
            binder,
            filters,
            joins,
            conditions,
            found: Operators::default(),
            placed: Vec::new(),
            noted: Vec::new(),
        })
    }

    /// Refuse a test of the result rows, which DataFusion runs above the
    /// region.
    ///
    /// # Errors
    ///
    /// A condition reads more than one table's columns or answers.
    pub(super) fn check_row_tests(&self) -> Result<(), CompileError> {
        match self.conditions.row_test() {
            Some(test) => unsupported(&format!(
                "DataFusion must evaluate the result-row test {}.",
                condition_text(test)
            )),
            None => Ok(()),
        }
    }

    /// Each table's tests of answers, as its operators' column filters:
    /// a test follows the last column operator whose answer it reads.
    fn place_tests(&mut self) -> Option<()> {
        for scan in &self.found.scans {
            if self.conditions.tests_answers(&scan.alias) {
                note(&mut self.noted, &scan.alias);
            }
        }
        for alias in &self.noted {
            let producers: Vec<&SemanticColumn> = lookup(&self.placed, alias)
                .iter()
                .map(|(kind, position)| self.column(kind, *position))
                .collect();
            let asks = lookup(&self.found.filters, alias).len();
            let tests = self.conditions.column_tests(alias, &producers, asks)?;
            if tests.is_empty() {
                continue;
            }
            self.found.column_filters.push((alias.clone(), tests));
        }
        Some(())
    }

    /// Whether `id` is an AI filter of one table.
    fn is_table_filter(&self, id: usize) -> bool {
        matches!(&self.walk.get(id).kind, Kind::Ai(node)
            if node.is_predicate()
                && !node.is_join()
                && self.walk.call_tables(&node.call()).len() < 2)
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
            Kind::Ai(node) => self.walk.call_tables(&node.call()),
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
                && node.guard().is_some()
                && (node.function() == "ai_score" || node.is_join())
            {
                return None;
            }
            match &self.walk.get(id).kind {
                Kind::Scan(table) => {
                    let alias = self.walk.scan_alias(*table);
                    self.found.scans.push(Scan {
                        predicates: self.conditions.at_scan(alias),
                        ..self.binder.scan(alias)
                    });
                }
                Kind::Ai(node) if !node.is_column() => {
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
            Kind::Ai(node) if self.stops_per_key(&alias) => node.key_fetch(),
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
        let (on, filter) = match &self.walk.get(input).kind {
            Kind::Join(_) => (spec.on.clone(), spec.filter.clone()),
            _ => (Vec::new(), None),
        };
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
            filter,
            outer_aliases,
        });
        Some(())
    }

    /// The column operators an AI column node runs: its call, or an
    /// extraction's call per field, each with its name.
    fn bound_columns(&mut self, node: &AiNode) -> Option<(&'static str, Vec<(ModelCall, String)>)> {
        let kind = node.column_kind()?;
        let found = self.binder.answer(node.answer())?;
        Some((kind, found.calls.clone()))
    }

    /// A column node's guard as a condition; the outer None when it does
    /// not bind.
    fn bound_guard(&mut self, node: &AiNode) -> Option<Option<ValueExpression>> {
        match node.guard() {
            None => Some(None),
            Some(guard) => {
                let read = self.walk.through(&guard);
                self.binder.bind_condition(&read).ok().map(Some)
            }
        }
    }

    fn place_column(&mut self, id: usize, node: &AiNode) -> Option<()> {
        let read: Vec<String> = self.call_tables(id).into_iter().collect();
        let is_column = |kind: &Kind| matches!(kind, Kind::Ai(found) if found.is_column());
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
            for (_, found) in run {
                let guard = self.bound_guard(found)?;
                let (_, columns) = self.bound_columns(found)?;
                for (call, name) in columns {
                    self.found.classifies.push(SemanticColumn {
                        call,
                        name,
                        above_joins,
                        guard: guard.clone(),
                        guard_selectivity: found
                            .guard()
                            .map(|found| guard_selectivity(&found, &[])),
                    });
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
                guard_selectivity: node.guard().map(|found| guard_selectivity(&found, &[])),
            });
            let position = list.len() - 1;
            push(&mut self.placed, &alias, (kind, position));
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

    fn column(&self, kind: &str, position: usize) -> &SemanticColumn {
        let list = match kind {
            "label" => &self.found.classifies,
            "extract" => &self.found.extracts,
            "map" => &self.found.maps,
            _ => &self.found.sae_features,
        };
        &list[position]
    }
}

/// Set each scan's columns: the values of its table that the region's
/// operators read, the SELECT list's first, then each join's, column
/// operator's, test's, and filter's.
///
/// A scan loads its document column as a value only when an operator
/// reads the value, not only its tokens.
pub(super) fn set_scan_columns(found: &mut Operators) {
    // an AI filter's answer and a column operator's columns are computed,
    // not loaded
    let read = found.clone();
    let answers = read.filter_answers();
    let added: BTreeSet<String> = read
        .columns()
        .flat_map(|column| column.added_columns())
        .collect();
    let guards: Vec<ColumnRef> = read
        .columns()
        .filter_map(|column| column.guard.as_ref())
        .chain(read.filters.iter().flat_map(|(_, predicates)| {
            predicates
                .iter()
                .flat_map(|predicate| predicate.guard.iter().chain(&predicate.alternative))
        }))
        .flat_map(column_references)
        .collect();
    let selected: Vec<ColumnRef> = read
        .projection
        .columns
        .iter()
        .filter_map(|column| match column {
            Projected::Expression(expression) => Some(value_refs(expression)),
            Projected::Name(_) => None,
        })
        .flatten()
        .collect();
    // the references that read a column's value
    let mut valued: Vec<ColumnRef> = selected.clone();
    valued.extend(
        read.joins
            .iter()
            .flat_map(|join| &join.on)
            .flat_map(equality_refs),
    );
    valued.extend(
        read.joins
            .iter()
            .filter_map(|join| join.filter.as_ref())
            .flat_map(column_references),
    );
    valued.extend(
        read.scans
            .iter()
            .flat_map(|scan| &scan.predicates)
            .flat_map(column_references),
    );
    valued.extend(
        read.extracts
            .iter()
            .chain(&read.maps)
            .flat_map(|column| call_refs(&column.call)),
    );
    valued.extend(guards.iter().cloned());
    let mut needed: Vec<ColumnRef> = selected;
    for join in read.joins.iter().rev() {
        needed.extend(asked_refs(&join.predicate));
        needed.extend(join.on.iter().flat_map(equality_refs));
        needed.extend(join.filter.iter().flat_map(column_references));
    }
    let columns: Vec<&SemanticColumn> = read.columns().collect();
    needed.extend(
        read.column_filters
            .iter()
            .flat_map(|(_, tests)| tests)
            .flat_map(|test| column_references(&test.condition)),
    );
    needed.extend(
        columns
            .iter()
            .rev()
            .flat_map(|column| call_refs(&column.call)),
    );
    for (_, predicates) in &read.filters {
        for predicate in predicates {
            needed.extend(asked_refs(&predicate.expression));
        }
    }
    for scan in &mut found.scans {
        let alias = scan.alias.clone();
        let sources = source_columns(&scan.column);
        let mut names: Vec<String> = Vec::new();
        let references = needed
            .iter()
            .chain(
                scan.predicates
                    .iter()
                    .flat_map(column_references)
                    .collect::<Vec<_>>()
                    .iter(),
            )
            .filter(|reference| reference.alias == alias)
            .map(|reference| reference.column.clone())
            .collect::<Vec<_>>();
        let keys = read.stop_keys.get(&alias).cloned().unwrap_or_default();
        let guarded = guards
            .iter()
            .filter(|reference| reference.alias == alias)
            .map(|reference| reference.column.clone());
        for name in references.into_iter().chain(keys).chain(guarded) {
            let computed =
                added.contains(&name) || answers.contains(&(alias.as_str(), name.as_str()));
            let loaded = !sources.contains(&name)
                || valued
                    .iter()
                    .any(|reference| reference.alias == alias && reference.column == name);
            if !computed && loaded && !names.contains(&name) {
                names.push(name);
            }
        }
        scan.columns = names;
    }
}
