use crate::functions::is_ai_call;
use crate::plan::as_node;
use datafusion::common::Column;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, LogicalPlan};
use std::collections::BTreeSet;

use crate::ai::{AiNode, RowLimit, Semantics};

impl AiNode {
    pub fn accepts_row_limit(&self) -> bool {
        self.hint.is_none() && !self.filters_on_or() && !Self::any_filters_on_or(&self.input)
    }

    pub(crate) fn filters_on_or(&self) -> bool {
        self.or_terms().is_some()
    }

    pub(crate) fn any_filters_on_or(plan: &LogicalPlan) -> bool {
        plan.exists(|below| Ok(as_node::<AiNode>(below).is_some_and(AiNode::filters_on_or)))
            .unwrap_or(false)
    }

    pub(crate) fn take_predicate(&mut self, predicate: &Expr) -> bool {
        if self.row_limit.is_some() || self.semantics != Semantics::Rows {
            return false;
        }
        let returned: BTreeSet<String> = (self.returned_fields().iter())
            .map(|field| field.name().clone())
            .collect();
        let reads_returned = (predicate.column_refs().iter())
            .any(|column| column.relation.is_none() && returned.contains(&column.name));
        let held = (self.inline_predicate.iter().flat_map(split_conjunction))
            .any(|term| term == predicate);
        if !reads_returned || !(held || self.accepts_predicate(predicate)) {
            return false;
        }
        if !held {
            self.inline_predicate = conjunction(
                self.inline_predicate
                    .take()
                    .into_iter()
                    .chain([predicate.clone()]),
            );
        }
        true
    }

    pub(crate) fn row_limit_for(&self, rows: usize) -> Option<RowLimit> {
        match &self.row_limit {
            Some(RowLimit::PerKey { .. }) => None,
            Some(RowLimit::Global(found)) if *found <= rows => None,
            _ => Some(RowLimit::Global(rows)),
        }
    }

    pub(crate) fn distinct_row_limit(&self, keys: &[Column]) -> Option<RowLimit> {
        if self.semantics != Semantics::Rows || self.row_limit.is_some() {
            return None;
        }
        let returns_answer = self.function() != "ai_filter"
            && self
                .returned_fields()
                .iter()
                .any(|field| field.name() == &self.answer);
        let key = match keys {
            [column]
                if column.relation.is_none() && column.name == self.answer && returns_answer =>
            {
                vec![self.answer.clone()]
            }
            _ => self.input_key(keys)?,
        };
        Some(RowLimit::PerKey { key, n: 1 })
    }

    pub(crate) fn per_key_row_limit(&self, keys: &[Column], n: usize) -> Option<RowLimit> {
        if !self.stops_per_key() {
            return None;
        }
        Some(RowLimit::PerKey {
            key: self.input_key(keys)?,
            n,
        })
    }

    pub(crate) fn with_row_limit(self, row_limit: RowLimit) -> Self {
        Self {
            row_limit: Some(row_limit),
            ..self
        }
    }

    pub(crate) fn input_key(&self, keys: &[Column]) -> Option<Vec<String>> {
        let input = self.input.schema();
        let mut names: Vec<String> = Vec::new();
        for key in keys {
            let named = input
                .fields()
                .iter()
                .filter(|field| field.name() == &key.name)
                .count();
            if !input.has_column(key) || named != 1 {
                return None;
            }
            if !names.contains(&key.name) {
                names.push(key.name.clone());
            }
        }
        (!names.is_empty()).then_some(names)
    }

    pub(crate) fn stops_per_key(&self) -> bool {
        let tables = |node: &AiNode| -> Vec<String> {
            let mut found: Vec<String> = Vec::new();
            for column in node.call.column_refs() {
                if let Some(table) = &column.relation
                    && !found.contains(&table.table().to_string())
                {
                    found.push(table.table().to_string());
                }
            }
            found
        };
        let table = tables(self);
        let filters = |found: &AiNode| {
            found.function() == "ai_filter"
                && found.semantics == Semantics::Rows
                && found.row_limit.is_none()
                && found.hint.is_none()
                && found.precondition.is_none()
                && tables(found) == table
        };
        if table.len() != 1 || !filters(self) {
            return false;
        }
        let mut plain = true;
        let _ = self.input.apply(|below| {
            let joins = matches!(below, LogicalPlan::Join(_));
            let other_call = as_node::<AiNode>(below).is_some_and(|found| !filters(found));
            let unknown =
                matches!(below, LogicalPlan::Extension(_)) && as_node::<AiNode>(below).is_none();
            if joins || other_call || unknown {
                plain = false;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        plain
    }

    pub(crate) fn asked_expressions(&self) -> Vec<Expr> {
        [
            Some(&self.call),
            self.precondition.as_ref(),
            self.hint.as_ref(),
        ]
        .into_iter()
        .flatten()
        .cloned()
        .collect()
    }

    pub(crate) fn prune_fields(
        &mut self,
        read: impl Fn(&str) -> bool,
        carried: bool,
    ) -> Option<(String, bool)> {
        if self.semantics != Semantics::Rows {
            return None;
        }
        if !self.is_predicate() {
            let returned =
                (self.pruned_fields.clone()).unwrap_or_else(|| (0..self.fields.len()).collect());
            let kept: Vec<usize> = (returned.iter().copied())
                .filter(|index| *index == 0 || read(self.fields[*index].name()))
                .collect();
            if kept.len() < returned.len() {
                self.pruned_fields = Some(kept);
            }
            return None;
        }
        let returns_answer =
            (self.returned_fields().iter()).any(|field| *field.name() == self.answer);
        if self.function() == "ai_score" && returns_answer && !read(&self.answer) {
            self.pruned_fields = Some(Vec::new());
        }
        if self.function() != "ai_filter" || !returns_answer {
            return None;
        }
        let filter = self.inline_predicate.as_ref()?;
        if !read(&self.answer) {
            self.pruned_fields = Some(Vec::new());
            return None;
        }
        let negated = self.answer_term(filter)?;
        if !carried {
            return None;
        }
        self.pruned_fields = Some(Vec::new());
        Some((self.answer.clone(), !negated))
    }

    pub(crate) fn is_plain_column(&self) -> bool {
        self.semantics == Semantics::Rows
            && !self.is_predicate()
            && self.inline_predicate.is_none()
            && self.hint.is_none()
            && self.row_limit.is_none()
    }

    pub(crate) fn evaluates_on_rows(&self, term: &Expr) -> bool {
        if self.reads_call_table(term) {
            return true;
        }
        let tables: BTreeSet<String> = (self.call.column_refs().into_iter())
            .filter_map(|column| {
                column
                    .relation
                    .as_ref()
                    .map(|table| table.table().to_string())
            })
            .collect();
        let mut earlier = Self::rows_answers(&self.input);
        if self.function() == "ai_classify" {
            earlier.retain(|name| !Self::classified(&self.input, name));
        }
        let input = self.input.schema();
        tables.len() == 1
            && !term.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(true)
            && term.column_refs().iter().all(|column| {
                input.has_column(column)
                    && match &column.relation {
                        Some(table) => tables.contains(table.table()),
                        None => earlier.contains(&column.name),
                    }
            })
    }

    pub(crate) fn classified(plan: &LogicalPlan, name: &str) -> bool {
        plan.exists(|below| {
            Ok(as_node::<AiNode>(below)
                .is_some_and(|asked| asked.function() == "ai_classify" && asked.is_field(name)))
        })
        .unwrap_or(true)
    }

    pub(crate) fn rows_answers(plan: &LogicalPlan) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        let _ = plan.apply(|below| {
            if let Some(asked) = as_node::<AiNode>(below)
                && asked.semantics == Semantics::Rows
                && !matches!(asked.function(), "ai_map" | "ai_sae_features")
            {
                found.extend(asked.added_names());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        found
    }

    pub(crate) fn add_precondition(&mut self, terms: Vec<Expr>) {
        let new: Vec<Expr> = terms
            .into_iter()
            .filter(|term| {
                !self
                    .precondition
                    .as_ref()
                    .is_some_and(|precondition| split_conjunction(precondition).contains(&term))
            })
            .collect();
        if let Some(found) = conjunction(new) {
            self.precondition = Some(match self.precondition.take() {
                Some(precondition) => precondition.and(found),
                None => found,
            });
        }
    }

    pub(crate) fn takes_or_precondition(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.precondition.is_none()
            && self.hint.is_none()
            && self.row_limit.is_none()
    }

    pub(crate) fn precondition_or(&mut self, read: &Expr) {
        let earlier = Self::earlier_answers(&self.input);
        let is_earlier = |expr: &Expr| {
            let read = match expr {
                Expr::Not(inner) => inner.as_ref(),
                other => other,
            };
            read.try_as_col()
                .is_some_and(|column| column.relation.is_none() && earlier.contains(&column.name))
        };
        let kept = (split_conjunction(read).into_iter()).filter(|term| match term {
            Expr::IsNotTrue(inner) if is_earlier(inner) => true,
            Expr::Literal(..) => false,
            other => self.reads_call_table(other),
        });
        self.precondition = conjunction(kept.cloned().collect::<Vec<_>>());
    }

    pub(crate) fn earlier_answers(plan: &LogicalPlan) -> Vec<String> {
        let mut found = Vec::new();
        let _ = plan.apply(|below| {
            if let Some(asked) = as_node::<AiNode>(below)
                && asked.semantics == Semantics::Rows
                && asked.function() == "ai_filter"
                && asked.inline_predicate.is_none()
                && asked.hint.is_none()
                && asked.row_limit.is_none()
            {
                found.push(asked.answer.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        found
    }
}
