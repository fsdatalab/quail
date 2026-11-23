//! The Derive rule's rewrites of one [`AiNode`]: which tests and fetches
//! it takes from the plan above it, and which guards it takes from the
//! answers read beside it.

use crate::functions::is_ai_call;
use crate::plan::as_node;
use datafusion::common::Column;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, LogicalPlan};
use std::collections::BTreeSet;

use crate::ai::{AiNode, Fetch, Semantics};

impl AiNode {
    /// Whether the node can take a fetch, a number of passing rows after
    /// which the executor stops asking the model.
    ///
    /// A node with a `HINT` cannot: the hint's options estimate the rows
    /// of its test. A node that filters on an OR cannot, nor can a node
    /// above one: the executor counts the rows the model passes, not the
    /// rows the OR's other terms pass.
    pub fn accepts_fetch(&self) -> bool {
        self.hint.is_none() && !self.filters_on_or() && !Self::any_filters_on_or(&self.input)
    }

    /// Whether the node filters on an OR of its answer ([`Self::or_terms`]).
    pub(crate) fn filters_on_or(&self) -> bool {
        self.or_terms().is_some()
    }

    /// Whether an `AiNode` in `plan` filters on an OR.
    pub(crate) fn any_filters_on_or(plan: &LogicalPlan) -> bool {
        plan.exists(|below| Ok(as_node::<AiNode>(below).is_some_and(AiNode::filters_on_or)))
            .unwrap_or(false)
    }

    /// Take `test`, a test the plan above applies to the node's rows,
    /// into its filter, `σ_q(AI(R; p)) ⇒ AI(R; p ∧ q)`, when it reads a
    /// column the node returns and the node can decide it
    /// ([`Self::accepts_test`]); true when the filter holds it. A node of
    /// rows with no fetch takes tests; a call written twice tests the one
    /// answer twice.
    pub(crate) fn take_test(&mut self, test: &Expr) -> bool {
        if self.fetch.is_some() || self.semantics != Semantics::Rows {
            return false;
        }
        let returned: BTreeSet<String> = (self.returned_fields().iter())
            .map(|field| field.name().clone())
            .collect();
        let reads_returned = (test.column_refs().iter())
            .any(|column| column.relation.is_none() && returned.contains(&column.name));
        let held = (self.filter.iter().flat_map(split_conjunction)).any(|term| term == test);
        if !reads_returned || !(held || self.accepts_test(test)) {
            return false;
        }
        if !held {
            self.filter = conjunction(self.filter.take().into_iter().chain([test.clone()]));
        }
        true
    }

    /// The fetch of the node's first `rows` rows, `Limit_k(AI(R; n)) ⇒
    /// AI(R; n = min(k, n))`; None when its own fetch keeps as few or is
    /// per key.
    pub(crate) fn rows_fetch(&self, rows: usize) -> Option<Fetch> {
        match &self.fetch {
            Some(Fetch::PerKey { .. }) => None,
            Some(Fetch::Global(found)) if *found <= rows => None,
            _ => Some(Fetch::Global(rows)),
        }
    }

    /// The fetch of one row of each value of `keys`, for a `DISTINCT` over
    /// them: keyed by the node's returned answer alone, or by columns of
    /// its input ([`Self::input_key`]). An `AI.IF` is keyed by its input
    /// only: the executor stops a filter per group of input rows, not per
    /// answer. None for a node with a fetch or of other rows.
    pub(crate) fn distinct_fetch(&self, keys: &[Column]) -> Option<Fetch> {
        if self.semantics != Semantics::Rows || self.fetch.is_some() {
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
        Some(Fetch::PerKey { key, n: 1 })
    }

    /// The fetch of `n` rows of each value of `keys`, columns of the
    /// node's input ([`Self::input_key`]), for a semi-join or a count
    /// threshold: only where the executor stops the node per key
    /// ([`Self::stops_per_key`]).
    pub(crate) fn per_key_fetch(&self, keys: &[Column], n: usize) -> Option<Fetch> {
        if !self.stops_per_key() {
            return None;
        }
        Some(Fetch::PerKey {
            key: self.input_key(keys)?,
            n,
        })
    }

    /// The names of `keys` when each is a column of the node's input whose
    /// name no other column there has: the executor groups documents by
    /// the key's column names.
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

    /// Whether the executor stops the node per key: it and every `AiNode`
    /// below it are AI.IF filters of rows over the same one table, with
    /// no fetch, `HINT`, or guard, and nothing below joins.
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
                && found.fetch.is_none()
                && found.hint.is_none()
                && found.guard.is_none()
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

    /// The call, guard, and `HINT` options: the expressions that read
    /// their columns whole, beside the filter and tests.
    pub(crate) fn asked_expressions(&self) -> Vec<Expr> {
        [Some(&self.call), self.guard.as_ref(), self.hint.as_ref()]
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }

    /// Whether the node is a column node of rows with no filter, `HINT`,
    /// or fetch: one a guard may limit to the rows where its answers are
    /// read.
    pub(crate) fn is_plain_column(&self) -> bool {
        self.semantics == Semantics::Rows
            && !self.is_predicate()
            && self.filter.is_none()
            && self.hint.is_none()
            && self.fetch.is_none()
    }

    /// Whether the executor can evaluate `term` on the rows the node
    /// reads: it calls no AI function and reads columns of the one table
    /// the node's call reads, and the answers of AI operators below the
    /// node, which the rows carry.
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
        // classifications of the same rows run as one operator, so one
        // cannot read another's label before it runs
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

    /// Whether a classification in `plan` adds the column `name`.
    pub(crate) fn classified(plan: &LogicalPlan, name: &str) -> bool {
        plan.exists(|below| {
            Ok(as_node::<AiNode>(below)
                .is_some_and(|asked| asked.function() == "ai_classify" && asked.is_field(name)))
        })
        .unwrap_or(true)
    }

    /// The columns the AI operators of rows in `plan` add, which each row
    /// they return carries. A map's text and SAE features are not carried
    /// on the rows, so a guard cannot read them.
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

    /// Guard the node with `terms` too, each it does not have already.
    pub(crate) fn add_guard(&mut self, terms: Vec<Expr>) {
        let new: Vec<Expr> = terms
            .into_iter()
            .filter(|term| {
                !self
                    .guard
                    .as_ref()
                    .is_some_and(|guard| split_conjunction(guard).contains(&term))
            })
            .collect();
        if let Some(found) = conjunction(new) {
            self.guard = Some(match self.guard.take() {
                Some(guard) => guard.and(found),
                None => found,
            });
        }
    }

    /// Whether the node may take an OR's guard: it is of rows, with no
    /// guard, `HINT`, or fetch.
    pub(crate) fn takes_or_guard(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.guard.is_none()
            && self.hint.is_none()
            && self.fetch.is_none()
    }

    /// Guard an `AI.IF` of an OR with the terms of `read`, where its
    /// answer is read, that it can evaluate before asking: a term that
    /// reads only the one table the call reads, or an earlier `AI.IF`'s
    /// answer not TRUE.
    pub(crate) fn guard_or(&mut self, read: &Expr) {
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
        self.guard = conjunction(kept.cloned().collect::<Vec<_>>());
    }

    /// The answers of the `AI.IF` nodes in `plan` that return their answer
    /// and drop no row, as each earlier `AI.IF` of an OR does, guarded or
    /// not.
    pub(crate) fn earlier_answers(plan: &LogicalPlan) -> Vec<String> {
        let mut found = Vec::new();
        let _ = plan.apply(|below| {
            if let Some(asked) = as_node::<AiNode>(below)
                && asked.semantics == Semantics::Rows
                && asked.function() == "ai_filter"
                && asked.filter.is_none()
                && asked.hint.is_none()
                && asked.fetch.is_none()
            {
                found.push(asked.answer.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        found
    }
}
