use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DFSchemaRef, Result, TableReference, internal_err, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::utils::{conjunction, disjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Filter, LogicalPlan, LogicalPlanBuilder, Operator,
    UserDefinedLogicalNodeCore,
};

use crate::functions::{column_call_kind, is_ai_call, quail_function};
use crate::plan::{as_node, extension_node};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Semantics {
    Rows,
    Semi,
    Anti,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RowLimit {
    Global(usize),
    PerKey { key: Vec<String>, n: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Written {
    pub position: usize,
    pub num_terms: usize,
    pub clause: usize,
    pub index: usize,
    pub on: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WrittenPredicate {
    pub predicate: Expr,
    pub written: Expr,
    pub position: usize,
    pub clause: usize,
}

pub(crate) struct OrParts {
    pub(crate) plain: Option<Expr>,
    pub(crate) others: Vec<(String, bool)>,
    pub(crate) negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AiNode {
    pub(crate) input: Arc<LogicalPlan>,
    pub(crate) call: Expr,
    pub(crate) answer: String,
    pub(crate) fields: Vec<Field>,
    pub(crate) precondition: Option<Expr>,
    pub(crate) inline_predicate: Option<Expr>,
    pub(crate) hint: Option<Expr>,
    pub(crate) pruned_fields: Option<Vec<usize>>,
    pub(crate) row_limit: Option<RowLimit>,
    pub(crate) semantics: Semantics,
    pub(crate) written: Option<Written>,
    pub(crate) written_predicates: Vec<WrittenPredicate>,
    pub(crate) schema: DFSchemaRef,
}

impl AiNode {
    pub fn try_new(
        input: Arc<LogicalPlan>,
        call: Expr,
        answer: String,
        fields: Vec<Field>,
    ) -> Result<Self> {
        Self {
            input,
            call,
            answer,
            fields,
            precondition: None,
            inline_predicate: None,
            hint: None,
            pruned_fields: None,
            row_limit: None,
            semantics: Semantics::Rows,
            written: None,
            written_predicates: Vec::new(),
            schema: Arc::new(DFSchema::empty()),
        }
        .checked()
    }

    pub(crate) fn answering(
        input: LogicalPlan,
        call: Expr,
        answer: String,
        written: Written,
    ) -> Result<Self> {
        let (_, field) = call.to_field(input.schema())?;
        let field = field.as_ref().clone().with_name(answer.clone());
        Self {
            written: Some(written),
            ..Self::try_new(Arc::new(input), call, answer, vec![field])?
        }
        .checked()
    }

    pub(crate) fn hinted(self, hint: Option<Expr>) -> Self {
        Self { hint, ..self }
    }

    pub(crate) fn tested(self, inline_predicate: Expr, semantics: Semantics) -> Result<Self> {
        Self {
            inline_predicate: Some(inline_predicate),
            pruned_fields: Some(Vec::new()),
            semantics,
            ..self
        }
        .checked()
    }

    pub(crate) fn with_written_predicates(mut self, predicates: &[WrittenPredicate]) -> Self {
        self.written_predicates.extend(predicates.iter().cloned());
        self
    }

    pub(crate) fn repeats(&self, call: &Expr, other: &AiNode) -> bool {
        let same_types = self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(&other.fields)
                .all(|(found, field)| found.data_type() == field.data_type());
        self.call == *call && same_types && !self.has_options()
    }

    pub(crate) fn answers_every_row(&self, call: &Expr) -> bool {
        (!self.is_predicate() || self.inline_predicate.is_none()) && self.call == *call
    }

    pub(crate) fn retyped_extraction(mut self) -> Result<Self> {
        let Expr::ScalarFunction(call) = &self.call else {
            return Ok(self);
        };
        let spans = Expr::ScalarFunction(ScalarFunction::new_udf(
            quail_function("ai_extract_span"),
            call.args.clone(),
        ));
        let schema = self.input.schema();
        for (index, typed) in [&self.call, &spans].into_iter().enumerate() {
            if let Some(field) = self.fields.get(index) {
                let (_, found) = typed.to_field(schema)?;
                self.fields[index] = found.as_ref().clone().with_name(field.name());
            }
        }
        Ok(self)
    }

    pub fn checked(mut self) -> Result<Self> {
        if let Some(missing) = self
            .reads()
            .into_iter()
            .find(|column| !self.input.schema().has_column(column))
        {
            return plan_err!("{} reads {missing}, which its input lacks", self.answer);
        }
        let added =
            DFSchema::from_unqualified_fields(self.fields.clone().into(), Default::default())?;
        let mut all = self.input.schema().as_ref().clone();
        all.merge(&added);
        if all.fields().len() != self.input.schema().fields().len() + self.fields.len() {
            return plan_err!("{} is already a column of the input", self.answer);
        }
        if let Some(pruned_fields) = &self.pruned_fields
            && pruned_fields
                .iter()
                .any(|index| *index >= self.fields.len())
        {
            return plan_err!("{} returns a column it does not add", self.answer);
        }
        self.schema = match self.semantics {
            Semantics::Rows => {
                let kept = DFSchema::from_unqualified_fields(
                    self.returned_fields().into(),
                    Default::default(),
                )?;
                let mut schema = self.input.schema().as_ref().clone();
                schema.merge(&kept);
                Arc::new(schema)
            }
            Semantics::Semi | Semantics::Anti => match self.input.as_ref() {
                LogicalPlan::Join(join) => Arc::clone(join.left.schema()),
                _ => return plan_err!("a {} gate reads a join", self.semantics_name()),
            },
        };
        Ok(self)
    }

    pub fn reads(&self) -> Vec<datafusion::common::Column> {
        let mut reads = Vec::new();
        for expr in [
            Some(&self.call),
            self.precondition.as_ref(),
            self.inline_predicate.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            for column in expr.column_refs() {
                let own = column.relation.is_none() && self.is_field(&column.name);
                if !own && !reads.contains(column) {
                    reads.push(column.clone());
                }
            }
        }
        reads
    }

    pub fn accepts_predicate(&self, term: &Expr) -> bool {
        let subquery = |expr: &Expr| {
            Ok(matches!(
                expr,
                Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
            ))
        };
        if term.exists(subquery).unwrap_or(true) {
            return false;
        }
        let one_table = self
            .call
            .column_refs()
            .iter()
            .filter_map(|column| column.relation.as_ref())
            .collect::<HashSet<_>>()
            .len()
            == 1;
        if self.function() != "ai_filter" {
            let own = |column: &&datafusion::common::Column| {
                column.relation.is_some() || self.is_field(&column.name)
            };
            return term.column_refs().iter().all(own);
        }
        if self.inline_predicate.is_some() {
            return false;
        }
        match self.answer_term(term) {
            Some(negated) => one_table || !negated,
            None => {
                let mut tested = self.clone();
                tested.inline_predicate = Some(term.clone());
                one_table
                    && tested.or_parts().is_some_and(|parts| {
                        parts
                            .plain
                            .is_none_or(|plain| self.reads_call_table(&plain))
                    })
            }
        }
    }

    pub(crate) fn or_terms(&self) -> Option<(Expr, bool)> {
        let filter = self.inline_predicate.as_ref()?;
        let reads_answer = |term: &Expr| {
            term.column_refs()
                .iter()
                .any(|column| column.relation.is_none() && self.is_field(&column.name))
        };
        let conjuncts = split_conjunction(filter);
        let mut found = None;
        for conjunct in &conjuncts {
            let terms = split_binary(conjunct, Operator::Or);
            let (answers, others): (Vec<&Expr>, Vec<&Expr>) = terms
                .into_iter()
                .partition(|term| self.answer_term(term).is_some());
            if answers.is_empty() || others.is_empty() {
                if reads_answer(conjunct) {
                    return None;
                }
                continue;
            }
            if found.is_some() || others.iter().any(|term| reads_answer(term)) {
                return None;
            }
            let [answer] = answers.as_slice() else {
                return None;
            };
            let negated = self.answer_term(answer) == Some(true);
            found = disjunction(others.into_iter().cloned()).map(|rest| (rest, negated));
        }
        found
    }

    pub(crate) fn or_parts(&self) -> Option<OrParts> {
        let (rest, negated) = self.or_terms()?;
        let answers = Self::answers_below(&self.input);
        let mut plain = Vec::new();
        let mut others = Vec::new();
        for term in split_binary(&rest, Operator::Or) {
            let (read, reversed) = match term {
                Expr::Not(inner) => (inner.as_ref(), true),
                other => (other, false),
            };
            match read.try_as_col() {
                Some(column) if column.relation.is_none() && answers.contains(&column.name) => {
                    if others.iter().any(|(name, _)| *name == column.name) {
                        return None;
                    }
                    others.push((column.name.clone(), reversed));
                }
                _ => plain.push(term.clone()),
            }
        }
        Some(OrParts {
            plain: disjunction(plain),
            others,
            negated,
        })
    }

    fn answers_below(plan: &LogicalPlan) -> Vec<String> {
        let mut found = Vec::new();
        let _ = plan.apply(|below| {
            if let Some(asked) = as_node::<AiNode>(below)
                && asked.semantics == Semantics::Rows
                && asked.function() == "ai_filter"
                && asked.inline_predicate.is_none()
            {
                found.push(asked.answer.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        found
    }

    pub(crate) fn reads_call_table(&self, test: &Expr) -> bool {
        let tables = |expr: &Expr| -> Vec<String> {
            let mut found: Vec<String> = Vec::new();
            for column in expr.column_refs() {
                let table = column
                    .relation
                    .as_ref()
                    .map(|table| table.table().to_string());
                if let Some(table) = table
                    && !found.contains(&table)
                {
                    found.push(table);
                }
            }
            found
        };
        let call = tables(&self.call);
        let read = tables(test);
        let input = self.input.schema();
        call.len() == 1
            && read == call
            && test
                .column_refs()
                .iter()
                .all(|column| column.relation.is_some() && input.has_column(column))
            && !test.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(true)
    }

    pub fn answer_term(&self, term: &Expr) -> Option<bool> {
        let (read, negated) = match term {
            Expr::Not(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        matches!(read, Expr::Column(column)
            if column.relation.is_none() && column.name == self.answer)
        .then_some(negated)
    }

    pub fn with_input(&self, input: LogicalPlan) -> Result<Self> {
        Self {
            input: Arc::new(input),
            ..self.clone()
        }
        .checked()
    }

    pub fn function(&self) -> &str {
        match &self.call {
            Expr::ScalarFunction(call) => call.name(),
            _ => "",
        }
    }

    pub fn is_predicate(&self) -> bool {
        self.function() == "ai_filter"
            || (self.function() == "ai_score" && self.inline_predicate.is_some())
    }

    pub fn in_on(&self) -> bool {
        self.written
            .as_ref()
            .is_some_and(|written| written.on.is_some())
    }

    pub fn has_options(&self) -> bool {
        self.inline_predicate.is_some()
            || self.precondition.is_some()
            || self.row_limit.is_some()
            || self.pruned_fields.is_some()
    }

    pub fn returned_fields(&self) -> Vec<Field> {
        match &self.pruned_fields {
            Some(pruned_fields) => pruned_fields
                .iter()
                .map(|index| self.fields[*index].clone())
                .collect(),
            None => self.fields.clone(),
        }
    }

    fn semantics_name(&self) -> &'static str {
        match self.semantics {
            Semantics::Rows => "rows",
            Semantics::Semi => "semi",
            Semantics::Anti => "anti",
        }
    }

    pub(crate) fn map_expressions(&self, rewrite: impl Fn(Expr) -> Result<Expr>) -> Result<Self> {
        let mut node = self.clone();
        node.call = rewrite(node.call)?;
        node.precondition = node.precondition.map(&rewrite).transpose()?;
        node.inline_predicate = node.inline_predicate.map(&rewrite).transpose()?;
        for pred in &mut node.written_predicates {
            pred.predicate = rewrite(pred.predicate.clone())?;
            pred.written = rewrite(pred.written.clone())?;
        }
        Ok(node)
    }

    pub(crate) fn is_field(&self, name: &str) -> bool {
        self.fields.iter().any(|field| field.name() == name)
    }

    pub fn answer(&self) -> &str {
        &self.answer
    }

    pub fn is_join(&self) -> bool {
        self.semantics != Semantics::Rows
    }

    pub fn is_anti(&self) -> bool {
        self.semantics == Semantics::Anti
    }

    pub fn is_column(&self) -> bool {
        !self.is_predicate() && !self.is_join()
    }

    pub fn call(&self) -> Expr {
        self.call.clone()
    }

    pub(crate) fn column_kind(&self) -> Option<&'static str> {
        column_call_kind(&self.call)
    }

    pub(crate) fn precondition(&self) -> Option<Expr> {
        self.precondition.clone()
    }

    pub(crate) fn inline_predicate(&self) -> Option<Expr> {
        self.inline_predicate.clone()
    }

    pub(crate) fn written(&self) -> Option<Written> {
        self.written.clone()
    }

    pub(crate) fn written_predicates(&self) -> Vec<WrittenPredicate> {
        self.written_predicates.clone()
    }

    pub(crate) fn global_row_limit(&self) -> Option<usize> {
        match self.row_limit {
            Some(RowLimit::Global(rows)) => Some(rows),
            _ => None,
        }
    }

    pub(crate) fn key_row_limit(&self) -> Option<(Vec<String>, usize)> {
        match &self.row_limit {
            Some(RowLimit::PerKey { key, n }) => Some((key.clone(), *n)),
            _ => None,
        }
    }

    pub(crate) fn call_tables(&self) -> BTreeSet<TableReference> {
        self.call
            .column_refs()
            .into_iter()
            .filter_map(|column| column.relation.clone())
            .collect()
    }

    pub(crate) fn input_schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    pub(crate) fn added_names(&self) -> Vec<String> {
        self.fields
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    pub(crate) fn run(plan: &LogicalPlan) -> (Vec<&AiNode>, &LogicalPlan) {
        let mut nodes = Vec::new();
        let mut bottom = plan;
        while let Some(node) = as_node::<AiNode>(bottom) {
            nodes.push(node);
            bottom = node.input.as_ref();
        }
        (nodes, bottom)
    }

    pub(crate) fn without_predicates(&self) -> Option<Self> {
        if self.written_predicates.is_empty() {
            return None;
        }
        Some(Self {
            written_predicates: Vec::new(),
            ..self.clone()
        })
    }

    pub(crate) fn is_bare(&self) -> bool {
        !self.has_options() && self.hint.is_none() && self.written_predicates.is_empty()
    }

    pub(crate) fn predicate_expressions(&self) -> Vec<Expr> {
        self.inline_predicate
            .iter()
            .cloned()
            .chain(
                self.written_predicates
                    .iter()
                    .map(|test| test.predicate.clone()),
            )
            .collect()
    }

    pub(crate) fn only_adds_columns(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.inline_predicate.is_none()
            && self.written_predicates.is_empty()
            && self.row_limit.is_none()
    }

    pub(crate) fn has_row_limit(&self) -> bool {
        self.row_limit.is_some()
    }

    pub(crate) fn asks_every_row(&self) -> bool {
        self.semantics == Semantics::Rows
            && self.inline_predicate.is_none()
            && self.row_limit.is_none()
            && self.precondition.is_none()
    }

    pub(crate) fn limits_rows(&self) -> bool {
        self.row_limit.is_some() || self.semantics != Semantics::Rows
    }

    pub(crate) fn lift_filter(
        &mut self,
        split: impl FnOnce(&Expr) -> (Vec<Expr>, Vec<Expr>),
    ) -> Vec<Expr> {
        if self.semantics != Semantics::Rows || self.is_predicate() {
            return Vec::new();
        }
        let Some(filter) = self.inline_predicate.take() else {
            return Vec::new();
        };
        let (moved, kept) = split(&filter);
        if !moved.is_empty() && self.row_limit.is_some() {
            self.inline_predicate = Some(filter);
            return Vec::new();
        }
        self.written_predicates.retain(|test| {
            !split_conjunction(&test.predicate)
                .iter()
                .any(|term| moved.contains(term))
        });
        self.inline_predicate = conjunction(kept);
        moved
    }

    pub(crate) fn options_above(&self) -> Result<LogicalPlan> {
        let bare = Self {
            inline_predicate: None,
            precondition: None,
            hint: None,
            row_limit: None,
            pruned_fields: None,
            written_predicates: Vec::new(),
            ..self.clone()
        };
        let mut above = extension_node(bare.checked()?);
        if let Some(filter) = &self.inline_predicate {
            above = LogicalPlan::Filter(Filter::try_new(filter.clone(), Arc::new(above))?);
        }
        if let Some(RowLimit::Global(rows)) = &self.row_limit {
            above = LogicalPlanBuilder::from(above)
                .limit(0, Some(*rows))?
                .build()?;
        }
        Ok(above)
    }

    fn write_explain(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "AiNode: ")?;
        if self.semantics != Semantics::Rows {
            write!(f, "{} ", self.semantics_name())?;
        }
        write!(f, "{} = {}", self.answer, self.call)?;
        if let Some(precondition) = &self.precondition {
            write!(f, ", precondition: {precondition}")?;
        }
        if let Some(inline_predicate) = &self.inline_predicate {
            write!(f, ", inline_predicate: {inline_predicate}")?;
        }
        if let Some(hint) = &self.hint {
            write!(f, ", hint: {hint}")?;
        }
        match &self.row_limit {
            Some(RowLimit::Global(n)) => write!(f, ", row_limit: {n}")?,
            Some(RowLimit::PerKey { key, n }) => {
                write!(f, ", row_limit: {n} per [{}]", key.join(", "))?
            }
            None => {}
        }
        if self.pruned_fields.is_some() {
            let names: Vec<String> = self
                .returned_fields()
                .iter()
                .map(|field| field.name().clone())
                .collect();
            write!(f, ", returns: [{}]", names.join(", "))?;
        }
        Ok(())
    }
}

impl PartialOrd for AiNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self == other).then_some(Ordering::Equal)
    }
}

impl UserDefinedLogicalNodeCore for AiNode {
    fn name(&self) -> &str {
        "AiNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        let mut exprs = vec![self.call.clone()];
        exprs.extend(self.precondition.clone());
        exprs
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut prevent: HashSet<String> = self
            .fields
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let key: &[String] = match &self.row_limit {
            None => return prevent,
            Some(RowLimit::Global(_)) => &[],
            Some(RowLimit::PerKey { key, .. }) => key,
        };
        let names: Vec<&String> = self
            .input
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect();
        for name in &names {
            let unique_key =
                key.contains(name) && names.iter().filter(|found| *found == name).count() == 1;
            if !unique_key {
                prevent.insert((*name).clone());
            }
        }
        prevent
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let input = self.input.schema();
        let width = input.fields().len();
        let mut needed: Vec<usize> = output_columns
            .iter()
            .copied()
            .filter(|index| *index < width)
            .collect();
        let read = self
            .expressions()
            .into_iter()
            .chain(self.predicate_expressions());
        for expr in read {
            for column in expr.column_refs() {
                match input.index_of_column(column) {
                    Ok(index) => needed.push(index),
                    Err(_) if column.relation.is_none() && self.is_field(&column.name) => {}
                    Err(_) => return None,
                }
            }
        }
        if let Some(RowLimit::PerKey { key, .. }) = &self.row_limit {
            for name in key.iter().filter(|name| !self.is_field(name)) {
                let found: Vec<usize> = input
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| field.name() == name)
                    .map(|(index, _)| index)
                    .collect();
                let [index] = found.as_slice() else {
                    return None;
                };
                needed.push(*index);
            }
        }
        needed.sort_unstable();
        needed.dedup();
        Some(vec![needed])
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut text = String::new();
        self.write_explain(&mut text)?;
        write!(f, "{}", text.replace('\r', "\\r").replace('\n', "\\n"))
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        let Ok([input]) = <[LogicalPlan; 1]>::try_from(inputs) else {
            return internal_err!("AiNode reads one input");
        };
        let mut exprs = exprs.into_iter().map(Expr::unalias);
        let (Some(call), precondition, None) = (exprs.next(), exprs.next(), exprs.next()) else {
            return internal_err!("AiNode has its call and an optional precondition");
        };
        if precondition.is_some() != self.precondition.is_some() {
            return internal_err!("AiNode's precondition was added or removed");
        }
        let mut node = self.clone();
        node.input = Arc::new(input);
        node.call = call;
        node.precondition = precondition;
        node.checked()
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::{Extension, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    fn reviews() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
            Field::new("stars", DataType::Int64, true),
        ]));
        LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .alias("r")
        .unwrap()
        .build()
        .unwrap()
    }

    fn ask(function: &str, template: &str) -> Expr {
        Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == function)
                .unwrap(),
            vec![lit(template), col("r.body")],
        ))
    }

    fn ai_filter(template: &str) -> AiNode {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_filter", template),
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        node.inline_predicate = Some(col("kind"));
        node.pruned_fields = Some(Vec::new());
        node.checked().unwrap()
    }

    #[test]
    fn an_ai_filter_keeps_its_input_schema_and_reads_its_prompt_columns() {
        let node = ai_filter("Is {0} kind?");
        assert_eq!(node.schema(), reviews().schema());
        assert_eq!(node.expressions(), vec![ask("ai_filter", "Is {0} kind?")]);
        let rebuilt = node
            .with_exprs_and_inputs(node.expressions(), vec![reviews()])
            .unwrap();
        assert_eq!(rebuilt, node);
        assert_eq!(node.necessary_children_exprs(&[0]), Some(vec![vec![0, 1]]));
    }

    #[test]
    fn scores_are_predicates_only_when_their_answer_is_tested() {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_score", "Relevance of {0}?"),
            "score".into(),
            vec![Field::new("score", DataType::Float64, true)],
        )
        .unwrap();
        assert!(!node.is_predicate());
        node.inline_predicate = Some(col("score").gt(lit(0.5)));
        assert!(node.is_predicate());

        let mut boolean = ai_filter("Is {0} kind?");
        boolean.inline_predicate = None;
        assert!(boolean.is_predicate());
    }

    #[test]
    fn an_ai_column_adds_its_answer_after_the_input_columns() {
        let node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_classify", "Tone of {0}?"),
            "tone".into(),
            vec![Field::new("tone", DataType::Utf8, true)],
        )
        .unwrap();
        let names: Vec<String> = node
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["id", "body", "stars", "tone"]);
    }

    #[test]
    fn datafusion_optimizes_a_plan_around_an_ai_filter() {
        let filter = LogicalPlan::Extension(Extension {
            node: Arc::new(ai_filter("Is {0} kind?")),
        });
        let plan = LogicalPlanBuilder::from(filter)
            .filter(col("r.stars").gt(lit(3)))
            .unwrap()
            .project(vec![col("r.id")])
            .unwrap()
            .build()
            .unwrap();
        let optimized = Optimizer::new()
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap();
        let text = optimized.display_indent().to_string();
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let asked = lines
            .iter()
            .position(|line| line.starts_with("AiNode"))
            .unwrap();
        let test = lines
            .iter()
            .position(|line| line.contains("stars > Int32(3)"))
            .unwrap();
        assert!(test > asked, "{text}");
        assert!(
            text.contains("Projection: reviews.id, reviews.body"),
            "{text}"
        );
    }

    fn tone(row_limit: Option<RowLimit>, hint: Option<Expr>) -> LogicalPlan {
        let mut node = AiNode::try_new(
            Arc::new(reviews()),
            ask("ai_classify", "Tone of {0}?"),
            "tone".into(),
            vec![Field::new("tone", DataType::Utf8, true)],
        )
        .unwrap();
        node.row_limit = row_limit;
        node.hint = hint;
        crate::plan::extension_node(node.checked().unwrap())
    }

    fn optimized(plan: LogicalPlan) -> Vec<String> {
        Optimizer::with_rules(crate::rules::default_rules())
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
            .lines()
            .map(|line| line.trim().to_string())
            .collect()
    }

    fn position(lines: &[String], start: &str) -> usize {
        lines
            .iter()
            .position(|line| line.starts_with(start))
            .unwrap_or_else(|| panic!("no {start} in {lines:?}"))
    }

    #[test]
    fn a_filter_of_the_input_stays_above_a_node_with_a_row_limit() {
        let per_key = RowLimit::PerKey {
            key: vec!["id".into()],
            n: 1,
        };
        for row_limit in [RowLimit::Global(2), per_key] {
            let plan = LogicalPlanBuilder::from(tone(Some(row_limit), None))
                .filter(col("r.stars").gt(lit(3)))
                .unwrap()
                .build()
                .unwrap();
            let lines = optimized(plan);
            assert!(
                position(&lines, "Filter") < position(&lines, "AiNode"),
                "{lines:?}"
            );
        }
    }

    #[test]
    fn a_filter_of_a_per_key_row_limits_key_moves_below_the_node() {
        let row_limit = RowLimit::PerKey {
            key: vec!["id".into()],
            n: 1,
        };
        let plan = LogicalPlanBuilder::from(tone(Some(row_limit), None))
            .filter(col("r.id").gt(lit(3)))
            .unwrap()
            .build()
            .unwrap();
        let lines = optimized(plan);
        assert!(
            position(&lines, "AiNode") < position(&lines, "Filter"),
            "{lines:?}"
        );
    }

    #[test]
    fn a_predicate_of_the_answer_stays_above_a_node_with_a_row_limit() {
        let plan = LogicalPlanBuilder::from(tone(Some(RowLimit::Global(2)), None))
            .filter(col("tone").eq(lit("kind")))
            .unwrap()
            .build()
            .unwrap();
        let lines = optimized(plan);
        let node = position(&lines, "AiNode");
        assert!(position(&lines, "Filter") < node, "{lines:?}");
        assert!(!lines[node].contains("inline_predicate:"), "{lines:?}");
    }

    #[test]
    fn a_limit_becomes_the_row_limit_of_a_node_without_a_hint() {
        let limited = |hint: Option<Expr>| {
            let plan = LogicalPlanBuilder::from(tone(None, hint))
                .limit(0, Some(2))
                .unwrap()
                .build()
                .unwrap();
            optimized(plan)
        };
        let lines = limited(None);
        assert!(
            lines[position(&lines, "AiNode")].contains("row_limit: 2"),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("Limit")),
            "{lines:?}"
        );
        let lines = limited(Some(lit(0.5)));
        let node = position(&lines, "AiNode");
        assert!(position(&lines, "Limit") < node, "{lines:?}");
        assert!(
            !lines[node..].iter().any(|line| line.starts_with("Limit")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_where_filter_under_a_distinct_moves_below_its_per_key_row_limit() {
        let filter = LogicalPlan::Extension(Extension {
            node: Arc::new(ai_filter("Is {0} kind?")),
        });
        let plan = LogicalPlanBuilder::from(filter)
            .filter(col("r.stars").gt(lit(3)))
            .unwrap()
            .project(vec![col("r.id")])
            .unwrap()
            .distinct()
            .unwrap()
            .build()
            .unwrap();
        let distinct =
            Optimizer::with_rules(vec![Arc::new(crate::rules::derive::Derive::default())])
                .optimize(plan, &OptimizerContext::new(), |_, _| {})
                .unwrap()
                .display_indent()
                .to_string();
        let lines: Vec<String> = distinct
            .lines()
            .map(|line| line.trim().to_string())
            .collect();
        let node = position(&lines, "AiNode");
        assert!(lines[node].contains("row_limit: 1 per [id]"), "{lines:?}");
        assert!(node < position(&lines, "Filter"), "{lines:?}");
    }
}
