use quail_planner::constants::{NULL_DOCUMENT, SAMPLE_ROWS};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, SortField};
use quail_datafusion::derive::expressions::PhysicalValue;
use quail_planner::context::PlanningContext;
use quail_planner::error::{CompileError, Refusal};
use quail_planner::inputs::{
    HfTokenizer, PrefixIndex, PrefixTree, TokenStore, TokenStoreWriter, last_tokens,
};
use quail_planner::operators::Operators;
use quail_planner::operators::{Scan, ScanColumn};
use quail_planner::physical::ValueExpression;
use quail_planner::plan::optimize::Selected;
use quail_planner::plan::placements::{Alternative, placements};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Settings {
    pub model: serde_json::Value,
    pub device: serde_json::Value,
    pub gpus: i64,
    pub canvas_draws: i64,
    #[serde(default)]
    pub gpu_timing: bool,
    #[serde(default)]
    pub host_kv_bytes: u64,
    #[serde(default)]
    pub attention: Option<String>,
    #[serde(default)]
    pub order: Option<String>,
    pub backend: String,
    pub physical_rules: Vec<String>,
    #[serde(default)]
    pub request: Option<quail_planner::plan::request_plan::RequestSettings>,
}

pub struct TableRows {
    pub identity: Option<String>,
    pub schema: arrow::datatypes::SchemaRef,
    pub batches: Vec<RecordBatch>,
}

pub const DOCUMENT_EXPRESSION: &str = "a document must be a column or a struct field";

#[derive(Debug, Clone)]
pub struct ScanInputs {
    pub alias: String,
    pub tokens: Arc<TokenStore>,
    pub columns: crate::shared::Table,
}

pub type PlannedRegion = (String, Vec<ScanInputs>);

pub struct Planned {
    pub selected: Selected,
    pub operators: Operators,
    pub fields: serde_json::Value,
}

pub enum PlanOutcome {
    Planned(Box<Planned>),
    Refused(Refusal),
    Fixed(serde_json::Value),
}

pub struct Prepared {
    pub alternatives: Vec<Alternative>,
    pub fields: serde_json::Value,
    pub scans: Vec<ScanInputs>,
    measured: Measured,
}

struct Measured {
    document_tokens: BTreeMap<String, Vec<i64>>,
    first_documents: BTreeMap<String, Vec<bool>>,
    page_shared: BTreeMap<String, Vec<i64>>,
    reader_groups: BTreeMap<String, Vec<(i64, i64, i64)>>,
    ends: BTreeMap<String, Vec<i64>>,
    pair_fractions: BTreeMap<usize, f64>,
}

pub struct Planner {
    tokenizer: HfTokenizer,
    stores: Mutex<HashMap<(String, String, bool), Arc<TokenStore>>>,
    images: Mutex<crate::images::ImageIds>,
}

fn arrow_error(error: impl std::fmt::Display) -> CompileError {
    CompileError::new(error.to_string())
}

impl std::fmt::Debug for Planner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Planner").finish_non_exhaustive()
    }
}

impl Planner {
    pub fn new(tokenizer_file: &Path) -> Result<Self, CompileError> {
        Ok(Self {
            tokenizer: HfTokenizer::from_file(tokenizer_file).map_err(arrow_error)?,
            stores: Mutex::new(HashMap::new()),
            images: Mutex::new(crate::images::ImageIds::default()),
        })
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

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
        let extracted: Vec<String> = alternatives
            .iter()
            .flat_map(|alternative| &alternative.operators.extracts)
            .flat_map(|extract| quail_planner::expressions::call_aliases(&extract.call))
            .collect();
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
            let sizing = crate::images::ImageSizing::of_model(&settings.model);
            let store = self.token_store(scan, rows, extracted.contains(&scan.alias), sizing)?;
            let mut names = scan.columns.clone();
            if let Some(source) = source_column(&scan.column)
                && !names.contains(&source)
            {
                names.push(source);
            }
            let columns = value_columns(&names, rows)?;
            files.push(ScanInputs {
                alias: scan.alias.clone(),
                tokens: Arc::clone(&store),
                columns: with_null_documents(columns, &scan.column, rows)?,
            });
            let documents = store.documents();
            document_tokens.insert(scan.alias.clone(), store.lengths().to_vec());
            let first =
                quail_planner::inputs::first_documents(&documents, &[]).map_err(arrow_error)?;
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
            let tree = if settings.model["attention_mask"] == "document_isolated" {
                let count = store.lengths().len();
                PrefixTree {
                    order: (0..count).collect(),
                    parent: vec![None; count],
                    shared: vec![0; count],
                }
            } else {
                PrefixIndex::new(&documents).tree(page_tokens)
            };
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

    pub fn plan(
        &self,
        found: &Operators,
        tables: &HashMap<String, TableRows>,
        settings: &Settings,
    ) -> Result<(PlanOutcome, Vec<ScanInputs>), CompileError> {
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
            return Ok((request_outcome(&alternatives, &context, request)?, files));
        }
        let operators: Vec<(Operators, Vec<String>)> = alternatives
            .iter()
            .map(|alternative| (alternative.operators.clone(), alternative.rules.clone()))
            .collect();
        let searched = quail_planner::plan::optimize::search_plan(
            &operators,
            &context,
            &settings.physical_rules,
        )?;
        let outcome = match searched {
            Ok(selected) => PlanOutcome::Planned(Box::new(Planned {
                operators: alternatives[selected.alternative].operators.clone(),
                selected,
                fields,
            })),
            Err(refusal) => PlanOutcome::Refused(refusal),
        };
        Ok((outcome, files))
    }

    fn token_store(
        &self,
        scan: &Scan,
        rows: &TableRows,
        with_offsets: bool,
        sizing: Option<crate::images::ImageSizing>,
    ) -> Result<Arc<TokenStore>, CompileError> {
        let key = rows
            .identity
            .as_ref()
            .map(|identity| (identity.clone(), format!("{:?}", scan.column), with_offsets));
        if let Some((identity, column, _)) = &key {
            let stores = self.stores.lock().expect("the stores lock");
            let found = stores
                .get(&(identity.clone(), column.clone(), true))
                .or_else(|| {
                    (!with_offsets)
                        .then(|| stores.get(&(identity.clone(), column.clone(), false)))
                        .flatten()
                });
            if let Some(store) = found {
                return Ok(Arc::clone(store));
            }
        }
        let mut writer = if with_offsets {
            TokenStoreWriter::with_offsets()
        } else {
            TokenStoreWriter::new()
        };
        let built = match &scan.column {
            ScanColumn::Expression(expression) => Some(PhysicalValue::new(expression)?),
            ScanColumn::Unnamed(_) => None,
        };
        for batch in &rows.batches {
            let values = document_values(&scan.column, built.as_ref(), batch)?;
            if crate::images::holds_images(values.data_type()) {
                let Some(sizing) = sizing else {
                    return Err(CompileError::new(
                        "the model reads no images; an image document needs a model \
                         that loads its vision tower, such as qwen3.5-4b-fp8",
                    ));
                };
                if with_offsets {
                    return Err(CompileError::new(
                        "AI.EXTRACT copies its answers from a document's text, and an \
                         image has none",
                    ));
                }
                let mut ids = self.images.lock().expect("the image ids lock");
                let tokens =
                    crate::images::image_token_lists(&values, &self.tokenizer, &mut ids, sizing)?;
                writer.write(tokens).map_err(arrow_error)?;
            } else {
                let text = cast(&values, &DataType::Utf8).map_err(arrow_error)?;
                writer
                    .write_texts(&self.tokenizer, text.as_string::<i32>())
                    .map_err(arrow_error)?;
            }
        }
        let store = Arc::new(writer.finish());
        if let Some(key) = key {
            self.stores
                .lock()
                .expect("the stores lock")
                .insert(key, Arc::clone(&store));
        }
        Ok(store)
    }
}

fn value_columns(names: &[String], rows: &TableRows) -> Result<crate::shared::Table, CompileError> {
    let missing = |name: &String| CompileError::new(format!("no column {name}"));
    let indices = names
        .iter()
        .map(|name| rows.schema.index_of(name).map_err(|_| missing(name)))
        .collect::<Result<Vec<_>, _>>()?;
    let schema = Arc::new(rows.schema.project(&indices).map_err(arrow_error)?);
    let batches = rows
        .batches
        .iter()
        .map(|batch| {
            let columns = indices
                .iter()
                .map(|index| Arc::clone(batch.column(*index)))
                .collect();
            let options =
                arrow::array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            RecordBatch::try_new_with_options(Arc::clone(&schema), columns, &options)
                .map_err(arrow_error)
        })
        .collect::<Result<_, _>>()?;
    Ok((schema, batches))
}

fn with_null_documents(
    columns: crate::shared::Table,
    column: &ScanColumn,
    rows: &TableRows,
) -> Result<crate::shared::Table, CompileError> {
    let built = match column {
        ScanColumn::Expression(expression) => Some(PhysicalValue::new(expression)?),
        ScanColumn::Unnamed(_) => None,
    };
    let mut masks: Vec<ArrayRef> = Vec::new();
    let mut found = false;
    for batch in &rows.batches {
        let values = match (column, &built) {
            (_, Some(built)) => built.evaluate(batch, false)?,
            (ScanColumn::Unnamed(name), None) => batch
                .column_by_name(name)
                .cloned()
                .ok_or_else(|| CompileError::new(format!("no column {name}")))?,
            (ScanColumn::Expression(_), None) => unreachable!("an expression is built"),
        };
        found |= values.logical_null_count() > 0;
        masks.push(Arc::new(
            arrow::compute::is_null(&values).map_err(arrow_error)?,
        ));
    }
    if !found {
        return Ok(columns);
    }
    let (schema, batches) = columns;
    let mut fields: Vec<arrow::datatypes::FieldRef> = schema.fields().iter().cloned().collect();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        NULL_DOCUMENT,
        DataType::Boolean,
        false,
    )));
    let schema = Arc::new(arrow::datatypes::Schema::new(fields));
    let batches = batches
        .into_iter()
        .zip(masks)
        .map(|(batch, mask)| {
            let mut values = batch.columns().to_vec();
            values.push(mask);
            RecordBatch::try_new(Arc::clone(&schema), values).map_err(arrow_error)
        })
        .collect::<Result<_, _>>()?;
    Ok((schema, batches))
}

pub(crate) fn required_columns(operators: &Operators) -> HashMap<String, Vec<String>> {
    let alternatives = placements(operators);
    let scans = scans_of(
        alternatives
            .iter()
            .map(|alternative| &alternative.operators),
    );
    let mut required: HashMap<String, Vec<String>> = HashMap::new();
    for scan in scans {
        let names = required.entry(scan.provider).or_default();
        for name in scan.columns.into_iter().chain(source_column(&scan.column)) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    required
}

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

fn document_values(
    column: &ScanColumn,
    built: Option<&PhysicalValue>,
    batch: &RecordBatch,
) -> Result<ArrayRef, CompileError> {
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
    Ok(values)
}

fn pair_fractions(
    found: &Operators,
    scans: &[Scan],
    tables: &HashMap<String, TableRows>,
) -> Result<BTreeMap<usize, f64>, CompileError> {
    let mut fractions = BTreeMap::new();
    for (position, join) in found.joins.iter().enumerate() {
        let kept = match join.filter {
            Some(_) => quail_planner::constants::DEFAULT_SELECTIVITY,
            None => 1.0,
        };
        let conditions = &join.on;
        let Some(first) = conditions.first() else {
            if join.filter.is_some() {
                fractions.insert(position, kept);
            }
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
            estimate_pair_fraction(&sample(&left)?, &sample(&right)?)? * kept,
        );
    }
    Ok(fractions)
}

fn first_rows(rows: &TableRows) -> Result<RecordBatch, CompileError> {
    let joined =
        arrow::compute::concat_batches(&rows.schema, &rows.batches).map_err(arrow_error)?;
    Ok(joined.slice(0, joined.num_rows().min(SAMPLE_ROWS)))
}

fn estimate_pair_fraction(left: &[ArrayRef], right: &[ArrayRef]) -> Result<f64, CompileError> {
    let rows = |keys: &[ArrayRef]| keys.first().map_or(0, |key| key.len());
    let (num_left, num_right) = (rows(left), rows(right));
    if num_left == 0 || num_right == 0 {
        return Ok(0.0);
    }
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

fn request_outcome(
    alternatives: &[Alternative],
    context: &PlanningContext,
    request: &quail_planner::plan::request_plan::RequestSettings,
) -> Result<PlanOutcome, CompileError> {
    let mut found = Vec::new();
    for alternative in alternatives {
        found.push(quail_planner::plan::request_plan::request_plan(
            &alternative.operators,
            context,
            request,
        )?);
    }
    let Some(selected) = found.iter().position(|plan| plan.get("refusal").is_none()) else {
        return Ok(PlanOutcome::Fixed(found.swap_remove(0)));
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
    Ok(PlanOutcome::Fixed(serde_json::json!({
        "envelope": envelope, "estimated_seconds": 0.0,
        "operators": operators_json(&alternatives[selected].operators)?,
    })))
}

pub fn operators_json(found: &Operators) -> Result<String, CompileError> {
    serde_json::to_string(found).map_err(arrow_error)
}

pub fn refusal_json(refusal: &Refusal) -> serde_json::Value {
    serde_json::json!({
        "reasons": refusal.reasons, "constraint": refusal.constraint,
        "needed": refusal.needed, "available": refusal.available, "unit": refusal.unit,
    })
}

pub fn outcome_json(outcome: &PlanOutcome) -> Result<serde_json::Value, CompileError> {
    Ok(match outcome {
        PlanOutcome::Planned(planned) => planned_json(planned)?,
        PlanOutcome::Refused(refusal) => serde_json::json!({"refusal": refusal_json(refusal)}),
        PlanOutcome::Fixed(plan) => plan.clone(),
    })
}

pub fn planned_json(planned: &Planned) -> Result<serde_json::Value, CompileError> {
    let plan = &planned.selected.plan;
    Ok(serde_json::json!({
        "envelope": plan.envelope(),
        "estimated_seconds": plan.estimated_seconds,
        "estimates": plan.estimates,
        "alternative": planned.selected.alternative,
        "operators": operators_json(&planned.operators)?,
        "fields": planned.fields,
    }))
}
