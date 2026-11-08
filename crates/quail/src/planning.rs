//! Plan a model region in Rust: tokenize the documents it reads, measure
//! them, and search its physical plans.
//!
//! [`Planner::plan`] takes a region's operators and the rows of each
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
use quail_datafusion::expressions::PhysicalValue;
use quail_inputs::{HfTokenizer, PrefixIndex, TokenStore, TokenStoreWriter, last_tokens};
use quail_physical::ValueExpression;
use quail_planner::context::PlanningContext;
use quail_planner::expressions::{CompileError, Refusal};
use quail_planner::operators::Operators;
use quail_planner::operators::{Scan, ScanColumn};
use quail_planner::optimize::Selected;
use quail_planner::placements::{Alternative, placements};

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
    /// How a request backend, such as stock vLLM, submits its operators;
    /// None plans for Quail's backend.
    #[serde(default)]
    pub request: Option<quail_planner::request_plan::RequestSettings>,
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

/// A region's selected physical plan, the operators it lowers, the
/// context it was planned in, and the files its scans read.
pub struct Planned {
    pub selected: Selected,
    pub operators: Operators,
    /// The planning context's fields, as Python's planner reads them, to
    /// price an edited plan the same way.
    pub fields: serde_json::Value,
    pub scans: Vec<ScanFiles>,
}

/// What [`Planner::plan`] returns for a region.
pub enum PlanOutcome {
    /// Quail's backend's plan, priced, which an edit prices again.
    Planned(Box<Planned>),
    /// The refusal of every alternative.
    Refused(Refused),
    /// A request backend's plan, which nothing prices, as JSON with its
    /// scans' files.
    Fixed(serde_json::Value),
}

/// A region's column placements, its scans' files, and the planning
/// context's fields, before a backend lowers it.
pub struct Prepared {
    pub alternatives: Vec<Alternative>,
    /// The planning context's fields, as Python's planner reads them.
    pub fields: serde_json::Value,
    pub scans: Vec<ScanFiles>,
    measured: Measured,
}

/// What the Quail backend's search reads of a region's scans.
struct Measured {
    document_tokens: BTreeMap<String, Vec<i64>>,
    first_documents: BTreeMap<String, Vec<bool>>,
    page_shared: BTreeMap<String, Vec<i64>>,
    reader_groups: BTreeMap<String, Vec<(i64, i64, i64)>>,
    ends: BTreeMap<String, Vec<i64>>,
    pair_fractions: BTreeMap<usize, f64>,
}

/// A region every alternative of which was refused, with the files of
/// its scans, which the speed-of-light estimate still reads.
#[derive(Debug)]
pub struct Refused {
    pub refusal: Refusal,
    pub scans: Vec<ScanFiles>,
}

/// Plans regions for one session: holds the model's tokenizer, and the
/// token stores and column files its queries wrote.
pub struct Planner {
    tokenizer: HfTokenizer,
    directory: tempfile::TempDir,
    files: AtomicUsize,
    /// (identity, document column, whether it holds character ranges)
    /// -> the token store of a table read whole, kept for later queries.
    stores: Mutex<HashMap<(String, String, bool), PathBuf>>,
}

fn arrow_error(error: impl std::fmt::Display) -> CompileError {
    CompileError::new(error.to_string())
}

impl std::fmt::Debug for Planner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Planner")
            .field("directory", &self.directory.path())
            .finish_non_exhaustive()
    }
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

    /// Place the column operators of `found`, write each scan's token
    /// store and column files from `tables`, and measure them: what every
    /// backend plans from.
    ///
    /// # Errors
    ///
    /// A scanned table is missing, or a document is an expression or not
    /// text.
    pub fn prepare(
        &self,
        found: &Operators,
        tables: &HashMap<String, TableRows>,
        settings: &Settings,
    ) -> Result<Prepared, CompileError> {
        let alternatives = placements(found);
        let scans = scans_of(
            alternatives
                .iter()
                .map(|alternative| &alternative.operators),
        );
        let extracted = extracted_aliases(
            alternatives
                .iter()
                .map(|alternative| &alternative.operators),
        );
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
            let (store, path) = self.token_store(scan, rows, extracted.contains(&scan.alias))?;
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
        let pair_fractions = pair_fractions(found, &scans, tables)?;
        let fields = serde_json::json!({
            "model": settings.model,
            "device": settings.device,
            "gpus": settings.gpus,
            "canvas_draws": settings.canvas_draws,
            "document_tokens": document_tokens,
            "operators": serde_json::Value::Null,
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
        Ok(Prepared {
            alternatives,
            fields,
            scans: files,
            measured: Measured {
                document_tokens,
                first_documents,
                page_shared,
                reader_groups,
                ends,
                pair_fractions,
            },
        })
    }

    /// Plan `found` over `tables`, each scanned table's rows by provider
    /// name, for `settings`.
    ///
    /// Returns the plan, or the refusal of every alternative.
    ///
    /// # Errors
    ///
    /// A scanned table is missing, a document column is not text, or
    /// planning fails.
    pub fn plan(
        &self,
        found: &Operators,
        tables: &HashMap<String, TableRows>,
        settings: &Settings,
    ) -> Result<PlanOutcome, CompileError> {
        let Prepared {
            alternatives,
            fields,
            scans: files,
            measured,
        } = self.prepare(found, tables, settings)?;
        let Measured {
            document_tokens,
            first_documents,
            page_shared,
            reader_groups,
            ends,
            pair_fractions,
        } = measured;
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
            operators: None,
            first_documents,
            page_shared,
            reader_groups,
            last_tokens: ends,
            backend: settings.backend.clone(),
        };
        if let Some(request) = &settings.request {
            return request_outcome(&alternatives, &context, request, &files);
        }
        let operators: Vec<(Operators, Vec<String>)> = alternatives
            .iter()
            .map(|alternative| (alternative.operators.clone(), alternative.rules.clone()))
            .collect();
        let searched =
            quail_planner::optimize::search_plan(&operators, &context, &settings.physical_rules)?;
        Ok(match searched {
            Ok(selected) => PlanOutcome::Planned(Box::new(Planned {
                operators: alternatives[selected.alternative].operators.clone(),
                selected,
                fields,
                scans: files,
            })),
            Err(refusal) => PlanOutcome::Refused(Refused {
                refusal,
                scans: files,
            }),
        })
    }

    /// The token store of a scan's document column: a table read whole
    /// keeps its store for later queries.
    fn token_store(
        &self,
        scan: &Scan,
        rows: &TableRows,
        with_offsets: bool,
    ) -> Result<(TokenStore, PathBuf), CompileError> {
        let key = rows
            .identity
            .as_ref()
            .map(|identity| (identity.clone(), format!("{:?}", scan.column), with_offsets));
        if let Some((identity, column, _)) = &key {
            let stores = self.stores.lock().expect("the stores lock");
            // a store with character ranges also serves a scan without
            let found = stores
                .get(&(identity.clone(), column.clone(), true))
                .or_else(|| {
                    (!with_offsets)
                        .then(|| stores.get(&(identity.clone(), column.clone(), false)))
                        .flatten()
                });
            if let Some(path) = found {
                return Ok((TokenStore::open(path).map_err(arrow_error)?, path.clone()));
            }
        }
        let path = self.file("tokens");
        let mut writer = if with_offsets {
            TokenStoreWriter::create_with_offsets(&path)
        } else {
            TokenStoreWriter::create(&path)
        }
        .map_err(arrow_error)?;
        let built = match &scan.column {
            ScanColumn::Expression(expression) => Some(PhysicalValue::new(expression)?),
            ScanColumn::Unnamed(_) => None,
        };
        for batch in &rows.batches {
            let texts = document_texts(&scan.column, built.as_ref(), batch)?;
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
fn scans_of<'a>(alternatives: impl Iterator<Item = &'a Operators>) -> Vec<Scan> {
    let mut found: Vec<Scan> = Vec::new();
    for operators in alternatives {
        for scan in operators.scans.iter().cloned() {
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

/// The aliases whose token stores hold each token's character range:
/// AI.EXTRACT places its answers in its documents' text by them.
pub(crate) fn extracted_aliases<'a>(
    alternatives: impl Iterator<Item = &'a Operators>,
) -> Vec<String> {
    alternatives
        .flat_map(|operators| &operators.extracts)
        .flat_map(|extract| quail_planner::expressions::call_aliases(&extract.call))
        .collect()
}

/// The text of a scan's document column in one batch, the column
/// built as `built` when it is an expression.
pub(crate) fn document_texts(
    column: &ScanColumn,
    built: Option<&PhysicalValue>,
    batch: &RecordBatch,
) -> Result<StringArray, CompileError> {
    let values = match (column, built) {
        (_, Some(built)) => built.evaluate(batch, false)?,
        (ScanColumn::Unnamed(name), None) => batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| CompileError::new(format!("no column {name}")))?,
        (ScanColumn::Expression(_), None) => {
            return Err(CompileError::new("a document expression is not built"));
        }
    };
    let text = cast(&values, &DataType::Utf8).map_err(arrow_error)?;
    Ok(text.as_string::<i32>().clone())
}

/// The fraction of each join's cross product its equalities keep,
/// estimated from the first rows of each side, by join position.
fn pair_fractions(
    found: &Operators,
    scans: &[Scan],
    tables: &HashMap<String, TableRows>,
) -> Result<BTreeMap<usize, f64>, CompileError> {
    let mut fractions = BTreeMap::new();
    for (position, join) in found.joins.iter().enumerate() {
        let conditions = &join.on;
        let Some(first) = conditions.first() else {
            continue;
        };
        let left_alias = quail_planner::expressions::argument_alias(&first.left);
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for condition in conditions {
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
            side.iter()
                .map(|key| PhysicalValue::new(key)?.evaluate(&batch, false))
                .collect()
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

/// The table column a document reads, or a struct column it reads a
/// field of.
pub(crate) fn source_column(column: &ScanColumn) -> Option<String> {
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

/// The field names of [`work_values`], in order.
pub const WORK_FIELDS: [&str; 9] = [
    "tokens",
    "pairs",
    "kv_written",
    "kv_read",
    "sliding_pairs",
    "sliding_kv_read",
    "state_written",
    "state_read",
    "sae_encoded",
];

/// A work's quantities in [`WORK_FIELDS`] order.
pub fn work_values(work: &quail_cost::work::Work) -> [&quail_cost::quantity::Quantity; 9] {
    [
        &work.tokens,
        &work.pairs,
        &work.kv_written,
        &work.kv_read,
        &work.sliding_pairs,
        &work.sliding_kv_read,
        &work.state_written,
        &work.state_read,
        &work.sae_encoded,
    ]
}

/// A plan's pricer as JSON, for Python to hand back when it reprices an
/// edited plan.
pub fn pricer_fields(pricer: &quail_planner::pricing::PlanPricer) -> serde_json::Value {
    let work = |work: &quail_cost::work::Work| {
        WORK_FIELDS
            .iter()
            .zip(work_values(work))
            .map(|(name, value)| (name.to_string(), serde_json::Value::from(value.total())))
            .collect::<serde_json::Map<String, serde_json::Value>>()
    };
    serde_json::json!({
        "chunk": pricer.chunk,
        "cap_pages": pricer.cap_pages,
        "stats": pricer.stats.iter().map(|(alias, stats)| (alias.clone(), serde_json::json!(
            [stats.num_documents, stats.total_tokens, stats.max_document_tokens]
        ))).collect::<serde_json::Map<String, serde_json::Value>>(),
        "stage_works": pricer.stage_works.iter().map(|(position, found)| (position.to_string(),
            serde_json::Value::Object(work(found)))).collect::<serde_json::Map<String, serde_json::Value>>(),
        "expected_rows": pricer.expected_rows,
    })
}

/// A request backend's plan of the first alternative it does not refuse,
/// as Python's backend search selected it: a request backend prices no
/// plan, so ties go to the earlier alternative. The plan's settings list
/// every alternative's outcome under `candidates`.
fn request_outcome(
    alternatives: &[Alternative],
    context: &PlanningContext,
    request: &quail_planner::request_plan::RequestSettings,
    files: &[ScanFiles],
) -> Result<PlanOutcome, CompileError> {
    let mut found = Vec::new();
    for alternative in alternatives {
        found.push(quail_planner::request_plan::request_plan(
            &alternative.operators,
            context,
            request,
        )?);
    }
    let Some(selected) = found.iter().position(|plan| plan.get("refusal").is_none()) else {
        let mut refused = found.swap_remove(0);
        refused["scans"] = scans_json(files);
        return Ok(PlanOutcome::Fixed(refused));
    };
    let candidates: Vec<serde_json::Value> = alternatives
        .iter()
        .zip(&found)
        .enumerate()
        .map(|(index, (alternative, plan))| {
            let refusal = plan.get("refusal").map(|refusal| refusal["constraint"].clone());
            serde_json::json!({
                "rewrites": alternative.rules,
                "estimated_seconds": if refusal.is_some() { serde_json::Value::Null } else { 0.0.into() },
                "refusal": refusal,
                "selected": index == selected,
            })
        })
        .collect();
    let plan = &found[selected];
    let mut plan_settings = plan["settings"].clone();
    plan_settings["candidates"] = candidates.into();
    let envelope = serde_json::json!({
        "backend": request.backend_name, "model": context.model.name,
        "device": context.device.name, "gpus": 1,
        "settings": plan_settings, "graph": plan["graph"],
    });
    Ok(PlanOutcome::Fixed(fixed_json(
        envelope,
        &alternatives[selected].operators,
        files,
    )?))
}

/// A plan nothing prices as JSON: its envelope, the operators it
/// lowers, and each scan's files.
fn fixed_json(
    envelope: serde_json::Value,
    found: &Operators,
    files: &[ScanFiles],
) -> Result<serde_json::Value, CompileError> {
    Ok(serde_json::json!({
        "envelope": envelope, "estimated_seconds": 0.0,
        "operators": operators_json(found)?,
        "scans": scans_json(files),
    }))
}

/// Operators as JSON text, as Python passes them back to Rust.
///
/// # Errors
///
/// The operators do not write.
pub fn operators_json(found: &Operators) -> Result<String, CompileError> {
    serde_json::to_string(found).map_err(arrow_error)
}

/// Each scan's files as JSON, by alias.
fn scans_json(scans: &[ScanFiles]) -> serde_json::Value {
    scans
        .iter()
        .map(|scan| {
            (
                scan.alias.clone(),
                serde_json::json!({"tokens": scan.tokens, "columns": scan.columns}),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into()
}

/// A prepared region as JSON: each column placement's operators and
/// rules, the context fields, and each scan's files.
///
/// # Errors
///
/// The operators do not write.
pub fn prepared_json(prepared: &Prepared) -> Result<serde_json::Value, CompileError> {
    let alternatives = prepared
        .alternatives
        .iter()
        .map(|alternative| {
            Ok(serde_json::json!([
                operators_json(&alternative.operators)?,
                alternative.rules,
            ]))
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    Ok(serde_json::json!({
        "alternatives": alternatives,
        "fields": prepared.fields,
        "scans": scans_json(&prepared.scans),
    }))
}

/// A refusal's fields as JSON.
pub fn refusal_json(refusal: &Refusal) -> serde_json::Value {
    serde_json::json!({
        "reasons": refusal.reasons, "constraint": refusal.constraint,
        "needed": refusal.needed, "available": refusal.available, "unit": refusal.unit,
    })
}

/// A refused region as JSON: the refusal and each scan's files.
pub fn refused_json(refused: &Refused) -> serde_json::Value {
    serde_json::json!({
        "refusal": refusal_json(&refused.refusal),
        "scans": scans_json(&refused.scans),
    })
}

/// A planned region as JSON: the plan's envelope, estimates, and pricer,
/// the chosen operators, the context fields, and each scan's files.
///
/// # Errors
///
/// The operators do not write.
pub fn planned_json(planned: &Planned) -> Result<serde_json::Value, CompileError> {
    let scans = scans_json(&planned.scans);
    let plan = &planned.selected.plan;
    Ok(serde_json::json!({
        "envelope": plan.envelope(),
        "estimated_seconds": plan.estimated_seconds,
        "estimates": plan.estimates,
        "alternative": planned.selected.alternative,
        "pricer": pricer_fields(&plan.pricer),
        "operators": operators_json(&planned.operators)?,
        "fields": planned.fields,
        "scans": scans,
    }))
}
