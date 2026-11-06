//! Plan a model region in Rust: tokenize the documents it reads, measure
//! them, and search its physical plans.
//!
//! [`Planner::plan`] takes a region's logical plan and the rows of each
//! table it scans. For each scan it writes a token store of the document
//! column and a column file of each value column the executor reads, in
//! the files Python's `TokenStore` and `ColumnStore` open. It computes
//! the statistics the planner reads from them, then runs the column
//! placements and the plan search with the model's tokenizer.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::row::{RowConverter, SortField};
use quail_inputs::{HfTokenizer, PrefixIndex, TokenStore, TokenStoreWriter, last_tokens};
use quail_physical::ValueExpression;
use quail_planner::LogicalPlan;
use quail_planner::context::PlanningContext;
use quail_planner::expressions::{CompileError, Refusal};
use quail_planner::logical::{Scan, ScanColumn};
use quail_planner::optimize::Selected;

/// Rows the planner samples to estimate how many pairs a join's
/// equalities keep.
const SAMPLE_ROWS: usize = 1024;

/// The configuration a region is planned for.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Settings {
    /// The model's spec, as Python's codec writes it without type tags.
    pub model: serde_json::Value,
    /// The device's spec, likewise.
    pub device: serde_json::Value,
    pub gpus: i64,
    pub canvas_draws: i64,
    #[serde(default)]
    pub attention: Option<String>,
    #[serde(default)]
    pub order: Option<String>,
    pub backend: String,
    /// The physical rules, in the order they apply.
    pub physical_rules: Vec<String>,
}

/// The rows of one table a region scans.
pub struct TableRows {
    /// What the rows are, such as a registered provider's content
    /// identity, when they are the same for every query; the token store
    /// of a table with an identity is kept for later queries.
    pub identity: Option<String>,
    pub schema: arrow::datatypes::SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// The message of a region whose document is an expression, which
/// [`Planner::plan`] does not tokenize.
pub const DOCUMENT_EXPRESSION: &str = "a document must be a column or a struct field";

/// The files the executor reads for one scan.
#[derive(Debug, Clone)]
pub struct ScanFiles {
    pub alias: String,
    /// The token store of the scan's document column.
    pub tokens: PathBuf,
    /// Each value column's name and column file, in the scan's order.
    pub columns: Vec<(String, PathBuf)>,
}

/// A region's selected physical plan, the logical plan it lowers, the
/// context it was planned in, and the files its scans read.
pub struct Planned {
    pub selected: Selected,
    pub logical: LogicalPlan,
    /// The planning context's fields, as Python's planner reads them, to
    /// price an edited plan the same way.
    pub fields: serde_json::Value,
    pub scans: Vec<ScanFiles>,
}

/// Plans regions for one session: holds the model's tokenizer, and the
/// token stores and column files its queries wrote.
pub struct Planner {
    tokenizer: HfTokenizer,
    directory: tempfile::TempDir,
    files: AtomicUsize,
    /// (identity, document column) -> the token store of a table read
    /// whole, kept for later queries.
    stores: Mutex<HashMap<(String, String), PathBuf>>,
}

fn arrow_error(error: impl std::fmt::Display) -> CompileError {
    CompileError::new(error.to_string())
}

impl Planner {
    /// A planner with the `tokenizer.json` at `tokenizer_file`.
    ///
    /// # Errors
    ///
    /// The tokenizer does not read, or no temporary directory is made.
    pub fn new(tokenizer_file: &Path) -> Result<Self, CompileError> {
        Ok(Self {
            tokenizer: HfTokenizer::from_file(tokenizer_file).map_err(arrow_error)?,
            directory: tempfile::Builder::new()
                .prefix("quail-tokens-")
                .tempdir()
                .map_err(arrow_error)?,
            files: AtomicUsize::new(0),
            stores: Mutex::new(HashMap::new()),
        })
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    fn file(&self, kind: &str) -> PathBuf {
        let number = self.files.fetch_add(1, Ordering::Relaxed);
        self.directory.path().join(format!("{kind}-{number}.arrow"))
    }

    /// Plan `logical` over `tables`, each scanned table's rows by
    /// provider name, for `settings`.
    ///
    /// Returns the plan, or the refusal of every alternative.
    ///
    /// # Errors
    ///
    /// A scanned table is missing, a document column is not text, or
    /// planning fails.
    pub fn plan(
        &self,
        logical: LogicalPlan,
        tables: &HashMap<String, TableRows>,
        settings: &Settings,
    ) -> Result<Result<Planned, Refusal>, CompileError> {
        let alternatives = quail_planner::rules::column_placements(logical.clone())?;
        let scans = scans_of(alternatives.iter().map(|alternative| &alternative.plan));
        if scans.iter().any(|scan| match &scan.column {
            ScanColumn::Unnamed(_) => false,
            ScanColumn::Expression(expression) => !is_column(expression),
        }) {
            return Err(CompileError::new(DOCUMENT_EXPRESSION));
        }
        let page_tokens = settings.device["page_tokens"].as_i64().unwrap_or(16);
        let mut files = Vec::new();
        let mut document_tokens = BTreeMap::new();
        let mut first_documents = BTreeMap::new();
        let mut page_shared = BTreeMap::new();
        let mut reader_groups = BTreeMap::new();
        let mut ends = BTreeMap::new();
        for scan in &scans {
            let rows = tables
                .get(&scan.provider)
                .ok_or_else(|| CompileError::new(format!("no rows for table {}", scan.provider)))?;
            let (store, path) = self.token_store(scan, rows)?;
            let mut columns = Vec::new();
            let mut names = scan.columns.clone();
            // the executor reads the document's text to locate answers in it
            if let Some(source) = source_column(&scan.column)
                && !names.contains(&source)
            {
                names.push(source);
            }
            for name in &names {
                columns.push((name.clone(), self.column_file(name, rows)?));
            }
            files.push(ScanFiles {
                alias: scan.alias.clone(),
                tokens: path,
                columns,
            });
            let documents = store.documents();
            document_tokens.insert(scan.alias.clone(), store.lengths().to_vec());
            let first = quail_inputs::first_documents(&documents, &[]).map_err(arrow_error)?;
            if first
                .iter()
                .enumerate()
                .any(|(index, found)| index != *found)
            {
                first_documents.insert(
                    scan.alias.clone(),
                    first
                        .iter()
                        .enumerate()
                        .map(|(index, found)| index == *found)
                        .collect(),
                );
            }
            let tree = PrefixIndex::new(&documents).tree(page_tokens);
            reader_groups.insert(
                scan.alias.clone(),
                tree.reader_groups(store.lengths())
                    .iter()
                    .map(|group| (group.shared, group.count, group.rows))
                    .collect(),
            );
            page_shared.insert(scan.alias.clone(), tree.shared);
            ends.insert(
                scan.alias.clone(),
                last_tokens(&documents).into_iter().map(i64::from).collect(),
            );
        }
        let pair_fractions = pair_fractions(&logical, &scans, tables)?;
        let encode = |text: &str| -> Result<Vec<i64>, CompileError> {
            Ok(self
                .tokenizer
                .encode(text)
                .map_err(arrow_error)?
                .into_iter()
                .map(i64::from)
                .collect())
        };
        let decode = |ids: &[i64]| -> Result<String, CompileError> {
            let ids: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
            self.tokenizer.decode(&ids).map_err(arrow_error)
        };
        let fields = serde_json::json!({
            "model": settings.model,
            "device": settings.device,
            "gpus": settings.gpus,
            "canvas_draws": settings.canvas_draws,
            "document_tokens": document_tokens,
            "plan": serde_json::Value::Null,
            "attention": settings.attention,
            "order": settings.order,
            "pair_fractions": pair_fractions
                .iter()
                .map(|(position, fraction)| (position.to_string(), *fraction))
                .collect::<BTreeMap<String, f64>>(),
            "scan_fractions": {},
            "backend": settings.backend,
            "first_documents": first_documents,
            "page_shared": page_shared,
            "reader_groups": reader_groups,
            "last_tokens": ends,
        });
        let context = PlanningContext {
            model: serde_json::from_value(settings.model.clone()).map_err(arrow_error)?,
            device: serde_json::from_value(settings.device.clone()).map_err(arrow_error)?,
            gpus: settings.gpus,
            canvas_draws: settings.canvas_draws,
            document_tokens,
            tokenizer: &encode,
            decode: Some(&decode),
            attention: settings.attention.clone(),
            pair_fractions,
            scan_fractions: BTreeMap::new(),
            order: settings.order.clone(),
            logical_plan: None,
            first_documents,
            page_shared,
            reader_groups,
            last_tokens: ends,
            backend: settings.backend.clone(),
        };
        let found = quail_planner::optimize::search_plan(
            &alternatives,
            &context,
            &settings.physical_rules,
        )?;
        Ok(found.map(|selected| Planned {
            logical: alternatives[selected.alternative].plan.clone(),
            selected,
            fields,
            scans: files,
        }))
    }

    /// The token store of a scan's document column: a table read whole
    /// keeps its store for later queries.
    fn token_store(
        &self,
        scan: &Scan,
        rows: &TableRows,
    ) -> Result<(TokenStore, PathBuf), CompileError> {
        let key = rows
            .identity
            .as_ref()
            .map(|identity| (identity.clone(), format!("{:?}", scan.column)));
        if let Some(key) = &key
            && let Some(path) = self.stores.lock().expect("the stores lock").get(key)
        {
            return Ok((TokenStore::open(path).map_err(arrow_error)?, path.clone()));
        }
        let path = self.file("tokens");
        let mut writer = TokenStoreWriter::create(&path).map_err(arrow_error)?;
        for batch in &rows.batches {
            let texts = document_texts(&scan.column, batch)?;
            writer
                .write_texts(&self.tokenizer, &texts)
                .map_err(arrow_error)?;
        }
        let store = writer.finish().map_err(arrow_error)?;
        if let Some(key) = key {
            self.stores
                .lock()
                .expect("the stores lock")
                .insert(key, path.clone());
        }
        Ok((store, path))
    }

    /// Write one value column of the rows to a column file.
    fn column_file(&self, name: &str, rows: &TableRows) -> Result<PathBuf, CompileError> {
        let path = self.file("column");
        let batches = &rows.batches;
        let field = rows
            .schema
            .field_with_name(name)
            .cloned()
            .map_err(|_| CompileError::new(format!("no column {name}")))?;
        let metadata = HashMap::from([("quail.kind".to_string(), "column_store".to_string())]);
        let schema = Arc::new(Schema::new(vec![field]).with_metadata(metadata));
        let file = File::create(&path).map_err(arrow_error)?;
        let mut writer = FileWriter::try_new(file, &schema).map_err(arrow_error)?;
        for batch in batches {
            let column = batch
                .column_by_name(name)
                .ok_or_else(|| CompileError::new(format!("no column {name}")))?;
            writer
                .write(
                    &RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(column)])
                        .map_err(arrow_error)?,
                )
                .map_err(arrow_error)?;
        }
        writer.finish().map_err(arrow_error)?;
        Ok(path)
    }
}

/// Each alias's scan, with the value columns any alternative reads.
fn scans_of<'a>(plans: impl Iterator<Item = &'a LogicalPlan>) -> Vec<Scan> {
    let mut found: Vec<Scan> = Vec::new();
    for plan in plans {
        for scan in quail_planner::operators::operators(plan).scans {
            match found.iter_mut().find(|known| known.alias == scan.alias) {
                Some(known) => {
                    for column in scan.columns {
                        if !known.columns.contains(&column) {
                            known.columns.push(column);
                        }
                    }
                }
                None => found.push(scan),
            }
        }
    }
    found
}

/// A column, or a field of a struct column, as an array of each row's value.
fn value(expression: &ValueExpression, batch: &RecordBatch) -> Result<ArrayRef, CompileError> {
    match expression {
        ValueExpression::ColumnRef(column) => batch
            .column_by_name(&column.column)
            .cloned()
            .ok_or_else(|| CompileError::new(format!("no column {}", column.column))),
        ValueExpression::GetField(field) => {
            let input = value(&field.input, batch)?;
            let structs = input
                .as_struct_opt()
                .ok_or_else(|| CompileError::new(format!("{} is not a struct", field.field)))?;
            structs
                .column_by_name(&field.field)
                .cloned()
                .ok_or_else(|| CompileError::new(format!("no field {}", field.field)))
        }
        other => Err(CompileError::new(format!(
            "{DOCUMENT_EXPRESSION}, not {other:?}"
        ))),
    }
}

/// The text of a scan's document column in one batch.
fn document_texts(column: &ScanColumn, batch: &RecordBatch) -> Result<StringArray, CompileError> {
    let values = match column {
        ScanColumn::Unnamed(name) => batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| CompileError::new(format!("no column {name}")))?,
        ScanColumn::Expression(expression) => value(expression, batch)?,
    };
    let text = cast(&values, &DataType::Utf8).map_err(arrow_error)?;
    Ok(text.as_string::<i32>().clone())
}

/// The fraction of each join's cross product its equalities keep,
/// estimated from the first rows of each side, by join position.
fn pair_fractions(
    logical: &LogicalPlan,
    scans: &[Scan],
    tables: &HashMap<String, TableRows>,
) -> Result<BTreeMap<usize, f64>, CompileError> {
    let mut fractions = BTreeMap::new();
    let operators = quail_planner::operators::operators(logical);
    for (position, join) in operators.joins.iter().enumerate() {
        let conditions = quail_planner::costs::join_conditions(join);
        let Some(first) = conditions.first() else {
            continue;
        };
        let left_alias = quail_planner::expressions::argument_alias(&first.left);
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for condition in &conditions {
            let (a, b) =
                if quail_planner::expressions::argument_alias(&condition.left) == left_alias {
                    (&condition.left, &condition.right)
                } else {
                    (&condition.right, &condition.left)
                };
            left.push(a.clone());
            right.push(b.clone());
        }
        let sample = |side: &[ValueExpression]| -> Result<Vec<ArrayRef>, CompileError> {
            let alias = quail_planner::expressions::argument_alias(&side[0]);
            let scan = scans
                .iter()
                .find(|scan| scan.alias == alias)
                .ok_or_else(|| CompileError::new(format!("no scan {alias}")))?;
            let rows = &tables[&scan.provider];
            let batch = first_rows(rows)?;
            side.iter().map(|key| value(key, &batch)).collect()
        };
        fractions.insert(
            position,
            estimate_pair_fraction(&sample(&left)?, &sample(&right)?)?,
        );
    }
    Ok(fractions)
}

fn first_rows(rows: &TableRows) -> Result<RecordBatch, CompileError> {
    let joined =
        arrow::compute::concat_batches(&rows.schema, &rows.batches).map_err(arrow_error)?;
    Ok(joined.slice(0, joined.num_rows().min(SAMPLE_ROWS)))
}

/// The pairs whose keys are equal over the cross product of two samples:
/// the sum over the keys both sides hold of the product of their counts,
/// over the product of the sample sizes; a null key matches nothing, and
/// samples that share no key still give one pair.
fn estimate_pair_fraction(left: &[ArrayRef], right: &[ArrayRef]) -> Result<f64, CompileError> {
    let rows = |keys: &[ArrayRef]| keys.first().map_or(0, |key| key.len());
    let (num_left, num_right) = (rows(left), rows(right));
    if num_left == 0 || num_right == 0 {
        return Ok(0.0);
    }
    // each side's keys take the left side's types, as the join compares them
    let types: Vec<DataType> = left.iter().map(|key| key.data_type().clone()).collect();
    let counts = |keys: &[ArrayRef]| -> Result<HashMap<Vec<u8>, f64>, CompileError> {
        let typed: Vec<ArrayRef> = keys
            .iter()
            .zip(&types)
            .map(|(key, kind)| cast(key, kind).map_err(arrow_error))
            .collect::<Result<_, CompileError>>()?;
        let converter = RowConverter::new(
            types
                .iter()
                .map(|kind| SortField::new(kind.clone()))
                .collect(),
        )
        .map_err(arrow_error)?;
        let converted = converter.convert_columns(&typed).map_err(arrow_error)?;
        let mut found = HashMap::new();
        for index in 0..rows(keys) {
            if typed.iter().any(|key| key.is_null(index)) {
                continue;
            }
            *found
                .entry(converted.row(index).as_ref().to_vec())
                .or_insert(0.0) += 1.0;
        }
        Ok(found)
    };
    let (left_counts, right_counts) = (counts(left)?, counts(right)?);
    let pairs: f64 = left_counts
        .iter()
        .filter_map(|(key, count)| right_counts.get(key).map(|other| count * other))
        .sum();
    Ok(pairs.max(1.0) / (num_left as f64 * num_right as f64))
}

/// Whether the region scans a document that is not a column or a struct
/// field, which this planner does not tokenize.
pub fn has_document_expression(logical: &LogicalPlan) -> bool {
    quail_planner::operators::operators(logical)
        .scans
        .iter()
        .any(|scan| match &scan.column {
            ScanColumn::Unnamed(_) => false,
            ScanColumn::Expression(expression) => !is_column(expression),
        })
}

/// The table column a document reads, or a struct column it reads a
/// field of.
fn source_column(column: &ScanColumn) -> Option<String> {
    fn root(expression: &ValueExpression) -> Option<String> {
        match expression {
            ValueExpression::ColumnRef(column) => Some(column.column.clone()),
            ValueExpression::GetField(field) => root(&field.input),
            _ => None,
        }
    }
    match column {
        ScanColumn::Unnamed(name) => Some(name.clone()),
        ScanColumn::Expression(expression) => root(expression),
    }
}

fn is_column(expression: &ValueExpression) -> bool {
    match expression {
        ValueExpression::ColumnRef(_) => true,
        ValueExpression::GetField(field) => is_column(&field.input),
        _ => false,
    }
}
