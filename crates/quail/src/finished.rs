//! The rows a region returns, each built once the operators writing it
//! have reported it.
//!
//! A region's rows are the natural join, on document ids, of a few
//! relations: each full join's true pairs, each table's surviving
//! documents, and each answer column's rows. Operators report rows of
//! these relations as they finish them, each tagged with the node and
//! port that wrote it. [`FinishedRows`] keeps every reported row by its
//! ids and, for each new one, looks up the rows of the other relations
//! it joins: a region row is built when the last of its parts arrives,
//! so each row is built once, and the region's fetch keeps the first
//! rows built.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, DictionaryArray, Int32Array, RecordBatch,
    RecordBatchOptions, StringArray, StructArray, new_null_array,
};
use arrow::compute::{
    SortColumn, cast, concat_batches, interleave, is_not_null, lexsort_to_indices, take,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use quail_planner::expressions::{
    PROBABILITIES_SUFFIX, asked_call, call_aliases, column_references,
};
use quail_planner::operators::Operators;
use quail_planner::results::{answer_columns, region_columns, region_structs, source_columns};
use serde_json::Value;

use crate::results::{AnswerRequest, arrow_error, document_ids};

type Error = String;

/// Which reported rows of a relation count.
#[derive(Debug, Clone)]
enum Keep {
    /// Every row.
    All,
    /// The rows whose last column, a join's answer, is true.
    True,
    /// The rows whose named column, a classification's label, is not
    /// null: a document whose answer names no label has no row.
    Valid(String),
}

/// The rows one port reports, kept by their document ids.
#[derive(Debug)]
struct Relation {
    /// The (node, port) that reports the rows.
    step: (String, String),
    /// The row's aliases this relation holds, by index.
    aliases: Vec<usize>,
    /// The region columns this relation holds.
    columns: Vec<String>,
    keep: Keep,
    /// Whether its rows were all known before the run: every document
    /// of a table no operator drops.
    known: bool,
    /// Whether its rows are a table's surviving documents: the table's
    /// last rows port.
    survivors: bool,
    /// Every column a report of it held.
    seen: HashSet<String>,
    /// ids -> row number.
    rows: HashMap<Vec<i32>, usize>,
    /// Per row, its ids in `aliases` order.
    keys: Vec<Vec<i32>>,
    /// Per row, its (batch, row) in `batches`.
    places: Vec<(usize, usize)>,
    /// (alias, id) -> the rows with that id, for a relation of several
    /// aliases.
    by_alias: HashMap<(usize, i32), Vec<usize>>,
    /// The reported values of `columns`, one batch per report.
    batches: Vec<Vec<ArrayRef>>,
    /// Each column's type, once a report held it.
    types: Vec<Option<DataType>>,
}

impl Relation {
    fn new(step: (String, String), aliases: Vec<usize>, columns: Vec<String>) -> Self {
        Relation {
            step,
            aliases,
            types: vec![None; columns.len()],
            columns,
            keep: Keep::All,
            known: false,
            survivors: false,
            seen: HashSet::new(),
            rows: HashMap::new(),
            keys: Vec::new(),
            places: Vec::new(),
            by_alias: HashMap::new(),
            batches: Vec::new(),
        }
    }

    /// Add a row under its ids; its number, or None when a row with those
    /// ids is in.
    fn insert(&mut self, key: Vec<i32>, place: (usize, usize)) -> Option<usize> {
        if self.rows.contains_key(&key) {
            return None;
        }
        let row = self.keys.len();
        if self.aliases.len() > 1 {
            for (alias, id) in self.aliases.iter().zip(&key) {
                self.by_alias.entry((*alias, *id)).or_default().push(row);
            }
        }
        self.rows.insert(key.clone(), row);
        self.keys.push(key);
        self.places.push(place);
        Some(row)
    }
}

/// A region column and where its values come from.
#[derive(Debug)]
enum Column {
    /// A source expression's values, by document of an alias.
    Source(String, usize, ArrayRef),
    /// A relation's column; `labels` are a classification's labels,
    /// which make the column a dictionary over them unless a table's
    /// surviving documents carry the column; `answer` says whether it is
    /// a call's answer, which a run must return, not a label's
    /// probabilities or a span.
    Answer {
        name: String,
        relation: usize,
        labels: Option<Value>,
        answer: bool,
    },
}

/// A region row: its ids by alias and its row in each relation.
type Found = (Vec<i32>, Vec<usize>);

/// The step computing an answer column, and the labels that make it a
/// dictionary.
type Producer = (usize, Option<Value>);

/// A region's rows, built as operators report the rows they finish.
#[derive(Debug)]
pub struct FinishedRows {
    aliases: Vec<String>,
    relations: Vec<Relation>,
    /// Per relation, the order to join the others to a new row of it.
    orders: Vec<Vec<usize>>,
    columns: Vec<Column>,
    structs: Vec<(String, Vec<(String, String)>)>,
    fetch: Option<usize>,
    /// The rows' schema before struct packing, fixed by the first rows
    /// built.
    schema: Option<SchemaRef>,
    /// The rows built so far, before struct packing, in the order they
    /// were built.
    built: Vec<RecordBatch>,
    /// Each built row's ids, by alias.
    keys: Vec<Vec<i32>>,
    /// Rows complete before any report: those of known relations alone.
    pending: Vec<Found>,
}

/// A port the rows read, before the rows' aliases are known.
struct Step {
    step: (String, String),
    aliases: Vec<String>,
    columns: Vec<String>,
    keep: Keep,
}

/// The index of the port's step, added with the aliases when new.
fn step_of(steps: &mut Vec<Step>, step: (String, String), aliases: Vec<String>) -> usize {
    if let Some(index) = steps.iter().position(|known| known.step == step) {
        return index;
    }
    steps.push(Step {
        step,
        aliases,
        columns: Vec::new(),
        keep: Keep::All,
    });
    steps.len() - 1
}

impl FinishedRows {
    /// The rows of a region from its operators and every port its run
    /// returns.
    ///
    /// `sources` holds each column a source expression computes, by
    /// document position.
    ///
    /// # Errors
    ///
    /// A column has no source and no port computes it, or the operators
    /// do not parse.
    pub fn new(
        request: &AnswerRequest,
        sources: &HashMap<String, ArrayRef>,
    ) -> Result<Self, Error> {
        let found: Operators =
            serde_json::from_str(&request.operators).map_err(|error| error.to_string())?;
        let scans: Vec<String> = found.scans.iter().map(|scan| scan.alias.clone()).collect();
        let source_aliases: HashMap<String, String> = source_columns(&found)
            .into_iter()
            .filter_map(|(name, expression)| {
                let alias = column_references(&expression).into_iter().next()?.alias;
                Some((name, alias))
            })
            .collect();
        let answers = answer_columns(&found);
        let mut steps: Vec<Step> = Vec::new();
        for port in &request.ports {
            let join = port
                .port
                .strip_prefix("join_answers:")
                .and_then(|position| position.parse::<usize>().ok())
                .and_then(|position| found.joins.get(position));
            if let Some(join) =
                join.filter(|join| port.value_type == "join_answers" && join.semantics == "full")
            {
                let aliases = call_aliases(asked_call(&join.predicate));
                let index = step_of(&mut steps, (port.node.clone(), port.port.clone()), aliases);
                steps[index].keep = Keep::True;
            }
        }
        // each answer column's step and the labels that make it a
        // dictionary
        let mut wanted: Vec<(String, Option<Producer>)> = Vec::new();
        for name in region_columns(&found) {
            if source_aliases.contains_key(&name) {
                wanted.push((name, None));
                continue;
            }
            let producer = request.ports.iter().rev().find_map(|port| {
                let answer = port.answer.as_ref()?;
                let holds = answer
                    .columns
                    .iter()
                    .any(|column| *column == name && !scans.contains(column))
                    || (answer.kind == "classifies"
                        && name == format!("{}{PROBABILITIES_SUFFIX}", answer.name));
                holds.then_some((port, answer))
            });
            let entry = answers.iter().find(|column| column.0 == name);
            let Some((port, answer)) = producer else {
                if entry.is_some_and(|column| !column.4) {
                    // a label's probabilities or a span no port reports
                    continue;
                }
                return Err(format!("the run returned no column {name}"));
            };
            let aliases = (answer.columns.iter())
                .filter(|column| scans.contains(column))
                .cloned()
                .collect();
            let index = step_of(&mut steps, (port.node.clone(), port.port.clone()), aliases);
            if answer.kind == "classifies" {
                steps[index].keep = Keep::Valid(answer.name.clone());
            }
            steps[index].columns.push(name.clone());
            let labels = match entry {
                Some((_, _, kind, labels, true))
                    if kind == "label" && answer.kind == "classifies" =>
                {
                    Some(labels.clone())
                }
                _ => None,
            };
            wanted.push((name, Some((index, labels))));
        }
        // the aliases of the joins and pair answers first: the rows are
        // ordered by the first of them, then by the rest
        let mut aliases: Vec<String> = Vec::new();
        let (pairs, single): (Vec<_>, Vec<_>) =
            steps.iter().partition(|step| step.aliases.len() > 1);
        let sourced = wanted
            .iter()
            .filter_map(|(name, _)| source_aliases.get(name));
        for alias in (pairs.into_iter().chain(single))
            .flat_map(|step| step.aliases.iter())
            .chain(sourced)
        {
            if !aliases.contains(alias) {
                aliases.push(alias.clone());
            }
        }
        if aliases.is_empty() {
            aliases.extend(scans.first().cloned());
        }
        // each table's surviving documents: its last rows port, or every
        // document when no operator drops any
        let mut known: Vec<(usize, usize)> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for alias in &aliases {
            let rows_port = format!("rows:{alias}");
            match request
                .ports
                .iter()
                .rev()
                .find(|port| port.port == rows_port)
            {
                Some(port) => survivors.push(step_of(
                    &mut steps,
                    (port.node.clone(), port.port.clone()),
                    vec![alias.clone()],
                )),
                None => {
                    let count = request.num_documents.get(alias).copied().unwrap_or(0);
                    steps.push(Step {
                        step: (String::new(), rows_port),
                        aliases: vec![alias.clone()],
                        columns: Vec::new(),
                        keep: Keep::All,
                    });
                    known.push((steps.len() - 1, count.max(0) as usize));
                }
            }
        }
        let position = |alias: &String| {
            (aliases.iter().position(|found| found == alias))
                .expect("every relation's alias is a row alias")
        };
        let mut relations: Vec<Relation> = steps
            .into_iter()
            .map(|step| {
                let mut relation = Relation::new(
                    step.step,
                    step.aliases.iter().map(position).collect(),
                    step.columns,
                );
                relation.keep = step.keep;
                relation
            })
            .collect();
        for index in survivors {
            relations[index].survivors = true;
            // the survivors carry their labels as they are, a NULL one
            // too, such as a guard's for a row it did not ask
            relations[index].keep = Keep::All;
        }
        let mut columns = Vec::new();
        for (name, producer) in wanted {
            columns.push(match producer {
                None => {
                    let values = sources
                        .get(&name)
                        .ok_or_else(|| format!("the column {name} has no source"))?;
                    let alias = position(&source_aliases[&name]);
                    Column::Source(name, alias, Arc::clone(values))
                }
                Some((relation, labels)) => Column::Answer {
                    // a classification's own rows port carries its labels
                    // as they are
                    labels: labels.filter(|_| !relations[relation].survivors),
                    answer: (answers.iter().find(|column| column.0 == name))
                        .is_none_or(|column| column.4),
                    name,
                    relation,
                },
            });
        }
        let orders = join_orders(&relations);
        let mut rows = FinishedRows {
            aliases,
            relations,
            orders,
            columns,
            structs: region_structs(&found),
            fetch: found.fetch.map(|fetch| fetch.max(0) as usize),
            schema: None,
            built: Vec::new(),
            keys: Vec::new(),
            pending: Vec::new(),
        };
        for (relation, count) in known {
            rows.relations[relation].known = true;
            rows.relations[relation].batches.push(Vec::new());
            for document in 0..count {
                if let Some(row) = rows.relations[relation].insert(vec![document as i32], (0, 0)) {
                    let mut found = rows.joined(relation, row);
                    rows.pending.append(&mut found);
                }
            }
        }
        Ok(rows)
    }

    /// Whether the rows read the port's reports.
    pub fn wants(&self, node: &str, port: &str) -> bool {
        (0..self.relations.len()).any(|relation| self.wants_relation(relation, node, port))
    }

    /// Take a batch of rows a port finished; return the region rows they
    /// complete, or None when they complete none.
    ///
    /// A row whose ids the port reported before is skipped, so a port
    /// may report its whole output once it ends.
    ///
    /// # Errors
    ///
    /// The batch lacks an id column of the port's relation, or an id is
    /// null.
    pub fn report(
        &mut self,
        node: &str,
        port: &str,
        batch: &RecordBatch,
    ) -> Result<Option<RecordBatch>, Error> {
        let mut found = std::mem::take(&mut self.pending);
        for relation in 0..self.relations.len() {
            if self.wants_relation(relation, node, port) {
                found.extend(self.take_rows(relation, batch)?);
            }
        }
        self.build(found)
    }

    fn wants_relation(&self, relation: usize, node: &str, port: &str) -> bool {
        let relation = &self.relations[relation];
        !relation.known && relation.step.0 == node && relation.step.1 == port
    }

    /// The rows complete before any report, when every relation was known
    /// before the run.
    ///
    /// # Errors
    ///
    /// The rows' columns do not build.
    pub fn finish(&mut self) -> Result<Option<RecordBatch>, Error> {
        let found = std::mem::take(&mut self.pending);
        self.build(found)
    }

    /// Every row built, ordered by its ids in alias order.
    ///
    /// A classification's label is a dictionary over its labels unless a
    /// table's surviving documents carry it, as a later operator's rows
    /// carry its input's columns; then it comes as they carry it.
    ///
    /// # Errors
    ///
    /// The rows' columns do not build.
    pub fn rows(&mut self) -> Result<RecordBatch, Error> {
        let rows = match self.built.first() {
            None => self.columns_of(&[])?,
            Some(first) => concat_batches(&first.schema(), &self.built).map_err(arrow_error)?,
        };
        let order = match self.keys.is_empty() || self.aliases.is_empty() {
            true => None,
            false => {
                let sort: Vec<SortColumn> = (0..self.aliases.len())
                    .map(|alias| SortColumn {
                        values: Arc::new(
                            self.keys
                                .iter()
                                .map(|key| key[alias])
                                .collect::<Int32Array>(),
                        ) as ArrayRef,
                        options: None,
                    })
                    .collect();
                Some(lexsort_to_indices(&sort, None).map_err(arrow_error)?)
            }
        };
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        for (field, column) in rows.schema().fields().iter().zip(rows.columns()) {
            let mut column = match &order {
                Some(order) => take(column.as_ref(), order, None).map_err(arrow_error)?,
                None => Arc::clone(column),
            };
            if self.carried(field.name()) {
                column = cast(&column, &DataType::Utf8).map_err(arrow_error)?;
            }
            fields.push(Field::new(field.name(), column.data_type().clone(), true));
            arrays.push(column);
        }
        packed_batch(fields, arrays, rows.num_rows(), &self.structs)
    }

    /// Whether a table's surviving documents carry a label column whose
    /// classification's own rows port is not those survivors.
    fn carried(&self, name: &str) -> bool {
        let labeled = self.columns.iter().any(|column| {
            matches!(column, Column::Answer { name: found, labels: Some(_), .. } if found == name)
        });
        labeled
            && (self.relations.iter())
                .any(|relation| relation.survivors && relation.seen.contains(name))
    }

    /// Keep a batch's new rows in a relation and return the region rows
    /// each completes.
    fn take_rows(&mut self, relation: usize, batch: &RecordBatch) -> Result<Vec<Found>, Error> {
        let schema = batch.schema();
        let kept: Option<BooleanArray> = match &self.relations[relation].keep {
            Keep::All => None,
            Keep::True => Some(
                (batch.columns().last())
                    .and_then(|column| column.as_boolean_opt())
                    .ok_or("a join's rows have no answer column")?
                    .clone(),
            ),
            Keep::Valid(name) => (schema.index_of(name).ok())
                .map(|index| is_not_null(batch.column(index)))
                .transpose()
                .map_err(arrow_error)?,
        };
        let ids = (self.relations[relation].aliases.iter())
            .map(|alias| {
                let name = &self.aliases[*alias];
                let index = schema.index_of(name).map_err(|_| {
                    let (node, port) = &self.relations[relation].step;
                    format!("the rows of {node}:{port} have no ids of {name}")
                })?;
                document_ids(batch.column(index))
            })
            .collect::<Result<Vec<Int32Array>, Error>>()?;
        let target = &mut self.relations[relation];
        (target.seen).extend(schema.fields().iter().map(|field| field.name().clone()));
        let mut values = Vec::new();
        for (column, known) in target.columns.iter().zip(target.types.iter_mut()) {
            values.push(match schema.index_of(column) {
                Ok(index) => {
                    let value = Arc::clone(batch.column(index));
                    match known {
                        Some(data_type) if data_type != value.data_type() => {
                            cast(&value, data_type).map_err(arrow_error)?
                        }
                        _ => {
                            known.get_or_insert_with(|| value.data_type().clone());
                            value
                        }
                    }
                }
                Err(_) => {
                    new_null_array(known.as_ref().unwrap_or(&DataType::Null), batch.num_rows())
                }
            });
        }
        let place = target.batches.len();
        target.batches.push(values);
        let mut found = Vec::new();
        for row in 0..batch.num_rows() {
            if kept
                .as_ref()
                .is_some_and(|kept| !kept.is_valid(row) || !kept.value(row))
            {
                continue;
            }
            let key: Vec<i32> = ids.iter().map(|column| column.value(row)).collect();
            if let Some(inserted) = self.relations[relation].insert(key, (place, row)) {
                found.extend(self.joined(relation, inserted));
            }
        }
        Ok(found)
    }

    /// The region rows a relation's row completes with the rows the other
    /// relations hold.
    fn joined(&self, relation: usize, row: usize) -> Vec<Found> {
        let mut binding: Vec<Option<i32>> = vec![None; self.aliases.len()];
        let own = &self.relations[relation];
        for (alias, id) in own.aliases.iter().zip(&own.keys[row]) {
            binding[*alias] = Some(*id);
        }
        let mut rows = vec![usize::MAX; self.relations.len()];
        rows[relation] = row;
        let mut found = Vec::new();
        self.extend(&self.orders[relation], &mut binding, &mut rows, &mut found);
        found
    }

    /// Join the next relation of `order` to a partly bound row: on its
    /// ids when they are all bound, else through the rows with one bound
    /// id, else with every row.
    fn extend(
        &self,
        order: &[usize],
        binding: &mut Vec<Option<i32>>,
        rows: &mut Vec<usize>,
        found: &mut Vec<Found>,
    ) {
        let Some((&next, rest)) = order.split_first() else {
            let ids = binding.iter().map(|id| id.expect("every alias is bound"));
            found.push((ids.collect(), rows.clone()));
            return;
        };
        let relation = &self.relations[next];
        let bound: Vec<Option<i32>> = relation
            .aliases
            .iter()
            .map(|alias| binding[*alias])
            .collect();
        let matches: Vec<usize> = if bound.iter().all(Option::is_some) {
            let key: Vec<i32> = bound.iter().flatten().copied().collect();
            relation.rows.get(&key).into_iter().copied().collect()
        } else if let Some((alias, id)) =
            (relation.aliases.iter().zip(&bound)).find_map(|(alias, id)| id.map(|id| (*alias, id)))
        {
            (relation.by_alias.get(&(alias, id)).into_iter().flatten())
                .copied()
                .filter(|matched| {
                    (bound.iter().zip(&relation.keys[*matched]))
                        .all(|(id, key)| id.is_none_or(|id| id == *key))
                })
                .collect()
        } else {
            (0..relation.keys.len()).collect()
        };
        for matched in matches {
            for (alias, id) in relation.aliases.iter().zip(&relation.keys[matched]) {
                binding[*alias] = Some(*id);
            }
            rows[next] = matched;
            self.extend(rest, binding, rows, found);
            for (alias, id) in relation.aliases.iter().zip(&bound) {
                binding[*alias] = *id;
            }
        }
        rows[next] = usize::MAX;
    }

    /// Build the region rows found, up to the fetch, and keep them.
    fn build(&mut self, mut found: Vec<Found>) -> Result<Option<RecordBatch>, Error> {
        if let Some(fetch) = self.fetch {
            found.truncate(fetch.saturating_sub(self.keys.len()));
        }
        if found.is_empty() {
            return Ok(None);
        }
        let rows = self.columns_of(&found)?;
        self.keys.extend(found.into_iter().map(|(ids, _)| ids));
        self.built.push(rows.clone());
        let fields = (rows.schema().fields().iter())
            .map(|field| field.as_ref().clone())
            .collect();
        packed_batch(
            fields,
            rows.columns().to_vec(),
            rows.num_rows(),
            &self.structs,
        )
        .map(Some)
    }

    /// The region's columns for the rows found, before struct packing, in
    /// the rows' schema once it is fixed.
    fn columns_of(&mut self, found: &[Found]) -> Result<RecordBatch, Error> {
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for column in &self.columns {
            let (name, values) = match column {
                Column::Source(name, alias, values) => {
                    let ids: Int32Array = found.iter().map(|(ids, _)| ids[*alias]).collect();
                    (
                        name,
                        take(values.as_ref(), &ids, None).map_err(arrow_error)?,
                    )
                }
                Column::Answer {
                    name,
                    relation: read,
                    labels,
                    answer,
                } => {
                    let relation = &self.relations[*read];
                    let index = (relation.columns.iter().position(|known| known == name))
                        .expect("a relation holds its columns");
                    let Some(data_type) = relation.types[index].clone() else {
                        if *answer {
                            return Err(format!("the run returned no column {name}"));
                        }
                        // a label's probabilities or a span the backend
                        // does not report
                        continue;
                    };
                    let parts: Vec<&dyn Array> = relation
                        .batches
                        .iter()
                        .map(|batch| batch[index].as_ref())
                        .collect();
                    let places: Vec<(usize, usize)> = (found.iter())
                        .map(|(_, rows)| relation.places[rows[*read]])
                        .collect();
                    let values = match places.is_empty() {
                        true => new_null_array(&data_type, 0),
                        false => interleave(&parts, &places).map_err(arrow_error)?,
                    };
                    let values = match labels {
                        Some(labels) => label_dictionary(&values, labels)?,
                        None => values,
                    };
                    (name, values)
                }
            };
            fields.push(Field::new(name, values.data_type().clone(), true));
            arrays.push(values);
        }
        let schema = match &self.schema {
            Some(schema) => {
                for (array, field) in arrays.iter_mut().zip(schema.fields()) {
                    if array.data_type() != field.data_type() {
                        *array = cast(array, field.data_type()).map_err(arrow_error)?;
                    }
                }
                Arc::clone(schema)
            }
            None => {
                let schema = Arc::new(Schema::new(fields));
                if !found.is_empty() {
                    self.schema = Some(Arc::clone(&schema));
                }
                schema
            }
        };
        RecordBatch::try_new_with_options(
            schema,
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(found.len())),
        )
        .map_err(arrow_error)
    }
}

/// For each relation, the order to join the others to a new row of it:
/// next the first one sharing a known alias, or, when none shares one,
/// the first.
fn join_orders(relations: &[Relation]) -> Vec<Vec<usize>> {
    (0..relations.len())
        .map(|start| {
            let mut bound: Vec<usize> = relations[start].aliases.clone();
            let mut rest: Vec<usize> = (0..relations.len())
                .filter(|other| *other != start)
                .collect();
            let mut order = Vec::new();
            while !rest.is_empty() {
                let next = (rest.iter())
                    .position(|other| relations[*other].aliases.iter().any(|a| bound.contains(a)))
                    .unwrap_or(0);
                let other = rest.remove(next);
                bound.extend(&relations[other].aliases);
                order.push(other);
            }
            order
        })
        .collect()
}

/// A label column as a dictionary over the call's labels, in order.
fn label_dictionary(values: &ArrayRef, labels: &Value) -> Result<ArrayRef, Error> {
    let order: Vec<&str> = (labels.as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .collect();
    let values = cast(values, &DataType::Utf8).map_err(arrow_error)?;
    let keys: Int32Array = (values.as_string::<i32>().iter())
        .map(|value| value.and_then(|value| order.iter().position(|label| *label == value)))
        .map(|position| position.map(|position| position as i32))
        .collect();
    let dictionary =
        DictionaryArray::<Int32Type>::try_new(keys, Arc::new(StringArray::from(order)))
            .map_err(arrow_error)?;
    Ok(Arc::new(dictionary))
}

/// The rows with each struct column's members packed into it: the
/// struct takes its first member's place, and the members leave.
fn packed_batch(
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
    num_rows: usize,
    structs: &[(String, Vec<(String, String)>)],
) -> Result<RecordBatch, Error> {
    let (mut out_fields, mut out_arrays) = (Vec::new(), Vec::new());
    let mut done: Vec<&String> = Vec::new();
    for (field, array) in fields.iter().zip(&arrays) {
        let owner = (structs.iter())
            .find(|(_, members)| members.iter().any(|(_, member)| member == field.name()));
        let Some((name, members)) = owner else {
            out_fields.push(field.clone());
            out_arrays.push(Arc::clone(array));
            continue;
        };
        if done.contains(&name) {
            continue;
        }
        done.push(name);
        let parts: Vec<(Arc<Field>, ArrayRef)> = (members.iter())
            .filter_map(|(member_field, member)| {
                let position = fields.iter().position(|found| found.name() == member)?;
                let part = Arc::clone(&arrays[position]);
                let field = Field::new(member_field, part.data_type().clone(), true);
                Some((Arc::new(field), part))
            })
            .collect();
        let packed = StructArray::from(parts);
        out_fields.push(Field::new(name, packed.data_type().clone(), true));
        out_arrays.push(Arc::new(packed) as ArrayRef);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(out_fields)),
        out_arrays,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(arrow_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).expect("a batch")
    }

    fn ids(values: Vec<i32>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    #[test]
    fn a_row_is_built_once_every_port_writing_it_reported_it() {
        // a filter's survivors joined to a score's answers, with a fetch
        // of two
        let survivors = Relation::new(("filter".into(), "rows:r".into()), vec![0], vec![]);
        let scores = Relation::new(
            ("score".into(), "rows:r".into()),
            vec![0],
            vec!["fun".into()],
        );
        let relations = vec![survivors, scores];
        let mut rows = FinishedRows {
            aliases: vec!["r".into()],
            orders: join_orders(&relations),
            relations,
            columns: vec![Column::Answer {
                name: "fun".into(),
                relation: 1,
                labels: None,
                answer: true,
            }],
            structs: Vec::new(),
            fetch: Some(2),
            schema: None,
            built: Vec::new(),
            keys: Vec::new(),
            pending: Vec::new(),
        };
        let fun = Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3])) as ArrayRef;
        let scored = batch(vec![("r", ids(vec![0, 1, 2])), ("fun", fun)]);
        // no document has passed the filter yet
        assert!(
            rows.report("score", "rows:r", &scored)
                .expect("a report")
                .is_none()
        );
        let passed = rows
            .report("filter", "rows:r", &batch(vec![("r", ids(vec![2]))]))
            .expect("a report")
            .expect("a row");
        assert_eq!(passed.num_rows(), 1);
        // the same ids again build nothing; the fetch keeps two rows
        assert!(
            rows.report("score", "rows:r", &scored)
                .expect("a report")
                .is_none()
        );
        rows.report("filter", "rows:r", &batch(vec![("r", ids(vec![0, 1, 2]))]))
            .expect("a report");
        let all = rows.rows().expect("the rows");
        let fun = all
            .column(0)
            .as_primitive::<arrow::datatypes::Float64Type>();
        assert_eq!(fun.values().to_vec(), vec![0.1, 0.3]);
    }
}
