//! The `_quail_native` Python module: Quail's Rust crates for Python.
//!
//! Tables and results cross between Python and Rust as Arrow IPC
//! stream bytes, so the module does not depend on pyarrow's version.

mod cost;

use std::io::Cursor;

use async_trait::async_trait;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::arrow::ipc::writer::StreamWriter;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{
    DataFusionError, Result, tree_node::TreeNode, tree_node::TreeNodeRecursion,
};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use quail_datafusion::{
    ModelRegion, RegionRunner, bind, quail_column_names, quail_schema, region_sql, session_context,
    sql,
};
use quail_planner::prompts::Turn;
use tokio::runtime::Runtime;

/// Runs each model region by calling a Python function with the
/// region's SQL; the function returns the rows as IPC stream bytes.
#[derive(Debug)]
struct PythonRunner {
    run_region: Py<PyAny>,
    /// The session model's chat-turn text and prompt layout, to bind
    /// regions in Rust; None sends each region's SQL instead.
    prompting: Option<(Turn, String)>,
}

impl PythonRunner {
    /// The request for one region: ("plan", its logical plan as JSON),
    /// or ("sql", its SQL) for a region the Rust binder does not read.
    fn request(&self, region: &ModelRegion) -> Result<(&'static str, String)> {
        let Some((turn, layout)) = &self.prompting else {
            return Ok(("sql", region_sql(region)?));
        };
        match bind::bind_region(&region.plan, turn, layout) {
            Ok(plan) => Ok((
                "plan",
                quail_planner::write_logical_plan(&plan)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            Err(error) if error.0.contains(bind::UNSUPPORTED) => Ok(("sql", region_sql(region)?)),
            Err(error) => Err(DataFusionError::Plan(format!("CompileError: {}", error.0))),
        }
    }
}

impl RegionRunner for PythonRunner {
    fn run(&self, region: &ModelRegion) -> Result<Vec<RecordBatch>> {
        let (kind, payload) = self.request(region)?;
        let bytes: Vec<u8> = Python::attach(|py| {
            self.run_region
                .call1(py, (kind, payload))
                .and_then(|answer| answer.extract::<Vec<u8>>(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let schema = region.plan.schema();
        let names = quail_column_names(schema);
        read_ipc(&bytes)?
            .into_iter()
            .map(|batch| conform(batch, &names, schema.inner()))
            .collect()
    }
}

/// Give a batch of Quail's rows the region's schema: each region
/// column is the Quail column of the same name, cast to its type.
fn conform(batch: RecordBatch, names: &[String], schema: &SchemaRef) -> Result<RecordBatch> {
    let columns = names
        .iter()
        .zip(schema.fields())
        .map(|(name, field)| {
            let column = batch.column_by_name(name).ok_or_else(|| {
                let returned: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                DataFusionError::Execution(format!(
                    "Quail returned columns {returned:?}, without the region's column {name}"
                ))
            })?;
            Ok(cast(column, field.data_type())?)
        })
        .collect::<Result<Vec<ArrayRef>>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// A table whose rows a Python function returns on each scan.
///
/// Model regions read their tables through Quail, so DataFusion scans
/// a table only for operators outside every region.
#[derive(Debug)]
struct PythonTable {
    schema: SchemaRef,
    scan: Py<PyAny>,
}

#[async_trait]
impl TableProvider for PythonTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let bytes: Vec<u8> = Python::attach(|py| {
            self.scan
                .call1(py, (columns,))
                .and_then(|answer| answer.extract::<Vec<u8>>(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let batches = read_ipc(&bytes)?;
        Ok(MemorySourceConfig::try_new_exec(&[batches], schema, None)?)
    }
}

fn read_ipc(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)?;
    Ok(reader.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn write_ipc(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new(&mut bytes, schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    drop(writer);
    Ok(bytes)
}

fn python_error(error: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A DataFusion session context with Quail registered.
#[pyclass(name = "NativeContext", module = "_quail_native")]
struct NativeContext {
    context: SessionContext,
    runtime: Runtime,
}

#[pymethods]
impl NativeContext {
    /// Create a context whose model regions run through `run_region`.
    ///
    /// `run_region(kind, payload)` takes ("plan", a region's Quail
    /// logical plan as JSON), bound in Rust with the model's chat-turn
    /// text `turn` and prompt `layout`, or ("sql", the region's SQL)
    /// for a region the Rust binder does not read, or for every region
    /// when no turn and layout are given.
    #[new]
    #[pyo3(signature = (run_region, turn=None, layout=None))]
    fn new(
        run_region: Py<PyAny>,
        turn: Option<(String, String)>,
        layout: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()?;
        let prompting = turn.zip(layout);
        let context = session_context(Arc::new(PythonRunner {
            run_region,
            prompting,
        }));
        Ok(Self { context, runtime })
    }

    /// Register a table by its schema, given as the IPC stream bytes of
    /// an empty table, and a function `scan(columns)` that returns
    /// those columns' rows as IPC stream bytes.
    fn register(&self, name: &str, schema: &[u8], scan: Py<PyAny>) -> PyResult<()> {
        let reader = StreamReader::try_new(Cursor::new(schema), None)
            .map_err(|error| python_error(error.into()))?;
        let table = PythonTable {
            schema: reader.schema(),
            scan,
        };
        self.context
            .register_table(name, Arc::new(table))
            .map_err(python_error)?;
        Ok(())
    }

    /// Run a query and return its rows as IPC stream bytes.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.runtime.block_on(async {
                    let frame = sql(&self.context, query).await?;
                    let schema = quail_schema(frame.schema());
                    let batches = frame
                        .collect()
                        .await?
                        .into_iter()
                        .map(|batch| {
                            RecordBatch::try_new(Arc::clone(&schema), batch.columns().to_vec())
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    write_ipc(&schema, &batches)
                })
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Return the SQL each model region of a query sends to Quail.
    fn region_sql(&self, py: Python<'_>, query: &str) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan = sql(&self.context, query).await?.into_optimized_plan()?;
                let mut texts = Vec::new();
                plan.apply(|node| {
                    if let LogicalPlan::Extension(extension) = node
                        && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                    {
                        texts.push(region_sql(region)?);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                Ok(texts)
            })
        })
        .map_err(python_error)
    }

    /// Bind each model region of a query into Quail's logical plan.
    ///
    /// Returns one (plan JSON, None) or (None, CompileError message)
    /// per region, and whether one region is the whole query.
    /// `turn` and `layout` are the model's chat-turn text and prompt
    /// layout.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query, turn, layout, dialect=None))]
    fn logical_plans(
        &self,
        py: Python<'_>,
        query: &str,
        turn: (String, String),
        layout: &str,
        dialect: Option<String>,
    ) -> PyResult<(Vec<(Option<String>, Option<String>)>, bool)> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan =
                    quail_datafusion::sql_in_dialect(&self.context, query, dialect.as_deref())
                        .await?
                        .into_optimized_plan()?;
                let mut bound = Vec::new();
                plan.apply(|node| {
                    if let LogicalPlan::Extension(extension) = node
                        && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                    {
                        bound.push(
                            match quail_datafusion::bind::bind_region(&region.plan, &turn, layout) {
                                Ok(found) => (
                                    Some(quail_planner::write_logical_plan(&found).map_err(
                                        |error| {
                                            datafusion::error::DataFusionError::External(Box::new(
                                                error,
                                            ))
                                        },
                                    )?),
                                    None,
                                ),
                                Err(error) => (None, Some(error.0)),
                            },
                        );
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                let whole = matches!(&plan, LogicalPlan::Extension(_));
                Ok((bound, whole))
            })
        })
        .map_err(python_error)
    }

    /// Return DataFusion's optimized logical plan for a query.
    fn explain(&self, py: Python<'_>, query: &str) -> PyResult<String> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan = sql(&self.context, query).await?.into_optimized_plan()?;
                Ok(plan.display_indent().to_string())
            })
        })
        .map_err(python_error)
    }
}

/// Read a physical plan envelope into the Rust plan types and write
/// it back as JSON.
///
/// Raises a ValueError when the plan does not fit the Rust types.
#[pyfunction]
fn read_and_write_plan(text: &str) -> PyResult<String> {
    let plan = quail_physical::read_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    quail_physical::write_plan(&plan)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Read a logical plan written by the Python codec into the Rust types
/// and write it back.
///
/// Raises a ValueError when the plan does not fit the Rust types.
#[pyfunction]
fn read_and_write_logical_plan(text: &str) -> PyResult<String> {
    let plan = quail_planner::read_logical_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    quail_planner::write_logical_plan(&plan)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Check a logical plan as `LogicalPlan.validate` does: None, or the
/// message of the CompileError it raises.
#[pyfunction]
fn validate_logical_plan(text: &str) -> PyResult<Option<String>> {
    let plan = quail_planner::read_logical_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(quail_planner::nodes::validate_plan(&plan)
        .err()
        .map(|error| error.0))
}

/// Place a bound SELECT and build its logical plan, as the front end's
/// `place_relation` and `LogicalPlanBuilder.project` do.
///
/// Returns (the plan as JSON, None), or (None, the CompileError's
/// message). Without project arguments it places the relation only and
/// returns (None, None) when that succeeds.
#[pyfunction]
#[pyo3(signature = (place, project=None))]
fn build_select(place: &str, project: Option<&str>) -> PyResult<(Option<String>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let place: quail_planner::builder::PlaceRelationArguments =
        serde_json::from_str(place).map_err(invalid)?;
    let placed = match quail_planner::builder::place_relation(place) {
        Ok(placed) => placed,
        Err(error) => return Ok((None, Some(error.0))),
    };
    let Some(project) = project else {
        return Ok((None, None));
    };
    let project: quail_planner::builder::ProjectArguments =
        serde_json::from_str(project).map_err(invalid)?;
    match placed.builder.project(&project, &placed.rows) {
        Ok(plan) => Ok((
            Some(quail_planner::write_logical_plan(&plan).map_err(invalid)?),
            None,
        )),
        Err(error) => Ok((None, Some(error.0))),
    }
}

/// Bind one AI function call's prompt, as `PromptBinder.bind` does.
///
/// Returns (the Prompt as JSON, None), or (None, the CompileError's message).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bind_prompt(
    kind: &str,
    template: &str,
    args: &str,
    labels: Vec<String>,
    descriptions: Vec<String>,
    task_description: &str,
    turn: (String, String),
    layout: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let args: Vec<quail_physical::ValueExpression> = serde_json::from_str(args).map_err(invalid)?;
    match quail_planner::prompts::bind(
        kind,
        template,
        &args,
        &labels,
        &descriptions,
        task_description,
        &turn,
        layout,
    ) {
        Ok(prompt) => Ok((Some(serde_json::to_string(&prompt).map_err(invalid)?), None)),
        Err(error) => Ok((None, Some(error.0))),
    }
}

/// Apply the built-in logical rules, as `logical_alternatives` does.
///
/// Returns ([(plan JSON, rule names), ...], None), or ([], the
/// CompileError's message). `id_columns` maps each provider to its id
/// column, or is None without a catalog.
#[pyfunction]
#[pyo3(signature = (plan, rules, id_columns=None))]
#[allow(clippy::type_complexity)]
fn logical_alternatives(
    plan: &str,
    rules: Vec<String>,
    id_columns: Option<std::collections::BTreeMap<String, String>>,
) -> PyResult<(Vec<(String, Vec<String>)>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let plan = quail_planner::read_logical_plan(plan).map_err(invalid)?;
    let context = quail_planner::rules::RuleContext { id_columns };
    match quail_planner::rules::logical_alternatives(plan, &rules, &context) {
        Ok(found) => Ok((
            found
                .into_iter()
                .map(|alternative| {
                    Ok((
                        quail_planner::write_logical_plan(&alternative.plan).map_err(invalid)?,
                        alternative.rules,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?,
            None,
        )),
        Err(error) => Ok((Vec::new(), Some(error.0))),
    }
}

/// A Rust tokenizer that calls a Python callable text -> token ids.
fn python_tokenizer<'py>(
    py: Python<'py>,
    tokenizer: &'py Bound<'py, PyAny>,
) -> impl Fn(&str) -> Result<Vec<i64>, quail_planner::CompileError> + 'py {
    move |text: &str| {
        tokenizer
            .call1((text,))
            .and_then(|ids| ids.extract::<Vec<i64>>())
            .map_err(|error| quail_planner::CompileError::new(error.value(py).to_string()))
    }
}

/// Serialize a lowering result: (JSON, None), or (None, the error's message).
fn lowering_result<T: serde::Serialize>(
    found: Result<T, quail_planner::CompileError>,
) -> PyResult<(Option<String>, Option<String>)> {
    match found {
        Ok(value) => Ok((
            Some(
                serde_json::to_string(&value)
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            ),
            None,
        )),
        Err(error) => Ok((None, Some(error.0))),
    }
}

fn from_json<T: serde::de::DeserializeOwned>(text: &str) -> PyResult<T> {
    serde_json::from_str(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Run one function of `quail/planner/prompt_lowering.py` in Rust.
///
/// `name` is the function; `arguments` its JSON-encoded arguments other
/// than the tokenizer. Returns (the result as JSON, None), or (None, the
/// error's message).
#[pyfunction]
fn prompt_lowering<'py>(
    py: Python<'py>,
    name: &str,
    arguments: &str,
    tokenizer: Bound<'py, PyAny>,
) -> PyResult<(Option<String>, Option<String>)> {
    use quail_planner::lowering as lower;
    let tokens = python_tokenizer(py, &tokenizer);
    let arguments: serde_json::Value = from_json(arguments)?;
    let get = |index: usize| arguments[index].to_string();
    match name {
        "lower_prompt" => lowering_result(lower::lower_prompt(&from_json(&get(0))?, &tokens)),
        "score_token_parts" => lowering_result(lower::score_token_parts(
            &from_json(&get(0))?,
            &from_json(&get(1))?,
            &tokens,
        )),
        "answer_ids" => lowering_result(lower::answer_ids(&from_json(&get(0))?, &tokens)),
        "option_spellings" => lowering_result(lower::option_spellings(
            &from_json::<Vec<String>>(&get(0))?,
            &tokens,
        )),
        "read_slot_offsets" => lowering_result(lower::read_slot_offsets(
            &from_json::<String>(&get(0))?,
            &from_json::<Vec<String>>(&get(1))?,
            &tokens,
        )),
        "classify_prompt_tokens" => lowering_result(lower::classify_prompt_tokens(
            &from_json(&get(0))?,
            &from_json::<Vec<String>>(&get(1))?,
            &tokens,
        )),
        "choice_token_parts" => {
            lowering_result(lower::choice_token_parts(&from_json(&get(0))?, &tokens))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "no prompt lowering function {other}"
        ))),
    }
}

/// The planning context's fields, as `tests/helpers/rust_builder.py`
/// writes them.
#[derive(serde::Deserialize)]
struct ContextFields {
    model: serde_json::Value,
    device: serde_json::Value,
    gpus: i64,
    canvas_draws: i64,
    document_tokens: std::collections::BTreeMap<String, Vec<i64>>,
    plan: Option<serde_json::Value>,
    attention: Option<String>,
    #[serde(default)]
    pair_fractions: std::collections::BTreeMap<String, f64>,
    #[serde(default)]
    scan_fractions: std::collections::BTreeMap<String, f64>,
    #[serde(default)]
    order: Option<String>,
    #[serde(default)]
    first_documents: std::collections::BTreeMap<String, Vec<bool>>,
    #[serde(default)]
    page_shared: std::collections::BTreeMap<String, Vec<i64>>,
    #[serde(default)]
    reader_groups: std::collections::BTreeMap<String, Vec<(i64, i64, i64)>>,
    #[serde(default)]
    last_tokens: std::collections::BTreeMap<String, Vec<i64>>,
    #[serde(default = "quail_backend")]
    backend: String,
}

fn quail_backend() -> String {
    "quail".to_string()
}

/// Run one operator planning function of `quail/planner/operators` in Rust.
///
/// `context` is the planning context's fields as JSON, and `arguments`
/// the function's other arguments. Returns (the result as JSON, None),
/// or (None, the refusal's message).
#[pyfunction]
#[pyo3(signature = (name, context, arguments, tokenizer, decode=None, callbacks=None))]
fn plan_operator<'py>(
    py: Python<'py>,
    name: &str,
    context: &str,
    arguments: &str,
    tokenizer: Bound<'py, PyAny>,
    decode: Option<Bound<'py, PyAny>>,
    callbacks: Option<Bound<'py, PyAny>>,
) -> PyResult<(Option<String>, Option<String>)> {
    use quail_planner::{classify, columns, costs, scores};
    let tokens = python_tokenizer(py, &tokenizer);
    let decoder = decode.as_ref().map(|decode| {
        move |ids: &[i64]| -> Result<String, quail_planner::CompileError> {
            decode
                .call1((ids.to_vec(),))
                .and_then(|text| text.extract::<String>())
                .map_err(|error| quail_planner::CompileError::new(error.value(py).to_string()))
        }
    });
    let decoder_ref: Option<&quail_planner::lowering::Decoder> = decoder
        .as_ref()
        .map(|found| found as &quail_planner::lowering::Decoder);
    let fields: ContextFields = from_json(context)?;
    let context = quail_planner::context::PlanningContext {
        model: json_value(fields.model)?,
        device: json_value(fields.device)?,
        gpus: fields.gpus,
        canvas_draws: fields.canvas_draws,
        document_tokens: fields.document_tokens,
        tokenizer: &tokens,
        decode: decoder_ref,
        attention: fields.attention,
        pair_fractions: fields
            .pair_fractions
            .into_iter()
            .map(|(key, fraction)| (key.parse().unwrap_or(usize::MAX), fraction))
            .collect(),
        scan_fractions: fields.scan_fractions,
        order: fields.order,
        logical_plan: fields.plan.map(json_value).transpose()?,
        first_documents: fields.first_documents,
        page_shared: fields.page_shared,
        reader_groups: fields.reader_groups,
        last_tokens: fields.last_tokens,
        backend: fields.backend,
    };
    let arguments: serde_json::Value = from_json(arguments)?;
    let get = |key: &str| arguments[key].clone();
    match name {
        "column_spec" => lowering_result(columns::column_spec(
            &context,
            &json_value(get("node"))?,
            get("expected_rows").as_f64().unwrap_or_default(),
            get("resident").as_bool().unwrap_or_default(),
        )),
        "score_spec" => lowering_result(
            scores::score_spec(
                &json_value(get("prompt"))?,
                get("name").as_str().unwrap_or_default(),
                get("expected_rows").as_f64().unwrap_or_default(),
                get("mean_tokens").as_f64(),
                &context,
                get("chunk").as_i64().unwrap_or_default(),
                get("pair_fraction").as_f64().unwrap_or(1.0),
                get("prefix_groups").as_f64(),
            )
            .map(|(spec, work)| (spec, work_fields(&work))),
        ),
        "filter_costs" => {
            let rows: Vec<f64> = json_value(get("rows"))?;
            lowering_result(costs::prepare_filter_costs(
                &json_value::<Vec<quail_physical::FilterPredicate>>(get("predicates"))?,
                get("mean_tokens").as_f64().unwrap_or_default(),
                get("chunk").as_i64().unwrap_or_default(),
                &context,
            ).map(|prepared| {
                prepared
                    .iter()
                    .map(|cost| {
                        serde_json::json!({
                            "selectivity": cost.selectivity,
                            "reuse_key": cost.reuse_key,
                            "first": rows.iter().map(|count| cost.first.seconds(*count)).collect::<Vec<_>>(),
                            "later": rows.iter().map(|count| cost.later.seconds(*count)).collect::<Vec<_>>(),
                        })
                    })
                    .collect::<Vec<_>>()
            }))
        }
        "join_specs" => {
            let fractions: std::collections::BTreeMap<String, f64> =
                json_value(get("pair_fractions"))?;
            let fractions = fractions
                .into_iter()
                .map(|(key, fraction)| (key.parse().unwrap_or(usize::MAX), fraction))
                .collect();
            let counts: std::collections::BTreeMap<String, f64> = json_value(get("counts"))?;
            lowering_result(costs::join_specs(
                &json_value::<Vec<quail_planner::logical::SemanticJoin>>(get("joins"))?,
                &fractions,
                get("canvas").as_i64().unwrap_or_default(),
                &context,
                get("chunk").as_i64().unwrap_or_default(),
            ).map(|specs| {
                specs
                    .iter()
                    .map(|spec| {
                        serde_json::json!({
                            "written_position": spec.written_position,
                            "aliases": spec.aliases,
                            "anchor": spec.anchor,
                            "anchor_free": spec.anchor_free,
                            "semantics": spec.semantics,
                            "selectivity": spec.selectivity,
                            "frame_tokens": spec.frame_tokens,
                            "label_tokens": spec.label_tokens,
                            "tail_tokens": spec.tail_tokens,
                            "on": spec.on,
                            "pair_fraction": spec.pair_fraction,
                            "work": spec.score.as_ref().map(|cost| work_fields(
                                &scores::score_join_work(cost, &spec.aliases, spec.pair_fraction, &counts),
                            )),
                        })
                    })
                    .collect::<Vec<_>>()
            }))
        }
        "classify_spec" | "classify_seconds" => {
            let node: quail_planner::LogicalNode = json_value(get("node"))?;
            let quail_planner::LogicalNode::SemanticClassify(column) = &node else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "not a SemanticClassify",
                ));
            };
            let call = quail_planner::nodes::column_call(column);
            let alias = quail_planner::nodes::column_alias(column);
            let rows = get("expected_rows").as_f64().unwrap_or_default();
            if name == "classify_spec" {
                lowering_result(classify::classify_spec(
                    &context,
                    call,
                    &column.name,
                    &alias,
                    rows,
                ))
            } else {
                lowering_result(classify::classify_seconds(
                    &context,
                    call,
                    &column.name,
                    &alias,
                    get("resident").as_bool().unwrap_or_default(),
                    rows,
                ))
            }
        }
        "choose_scoring" => {
            let table: classify::ClassifyTable = json_value(get("table"))?;
            let call: quail_physical::TaggedModelCall = json_value(get("call"))?;
            let quail_physical::TaggedModelCall::Call(call) = &call;
            lowering_result(table.choose_scoring(
                &json_value(get("spec"))?,
                call,
                get("resident").as_bool().unwrap_or_default(),
                &tokens,
            ))
        }
        "classify_joined" => {
            let table: classify::ClassifyTable = json_value(get("table"))?;
            let call: quail_physical::TaggedModelCall = json_value(get("call"))?;
            let quail_physical::TaggedModelCall::Call(call) = &call;
            lowering_result(
                table
                    .classify_joined(
                        call,
                        get("name").as_str().unwrap_or_default(),
                        get("partner").as_str().unwrap_or_default(),
                        get("expected_rows").as_f64().unwrap_or_default(),
                        get("partner_tokens").as_f64().unwrap_or_default(),
                        get("anchors").as_f64(),
                        &tokens,
                    )
                    .map(|(spec, work)| (spec, work_fields(&work))),
            )
        }
        "operator_order" => {
            let plan: quail_planner::LogicalPlan = json_value(get("plan"))?;
            lowering_result(
                quail_planner::search::choose_operator_order(&plan, &context)
                    .map(|found| operator_order_fields(&found)),
            )
        }
        "speed_of_light" => {
            let Some(callbacks) = callbacks else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the speed of light estimate needs its oracles",
                ));
            };
            let plan: quail_planner::LogicalPlan = json_value(get("plan"))?;
            let inputs: quail_planner::sol::EstimateInputs = json_value(get("inputs"))?;
            let failed =
                |error: PyErr| quail_planner::CompileError::new(error.value(py).to_string());
            let text = |method: &str, arguments: (String, String)| {
                callbacks
                    .call_method1(method, arguments)
                    .and_then(|found| found.extract::<String>())
                    .map_err(failed)
            };
            let parse = |text: String| {
                serde_json::from_str::<serde_json::Value>(&text)
                    .map_err(|error| quail_planner::CompileError::new(error.to_string()))
            };
            let encode = |value: &dyn erased_json::Json| value.json();
            let answer = |prompt: &quail_physical::Prompt,
                          assignment: &quail_planner::sol::Assignment| {
                parse(text(
                    "answer",
                    (encode(prompt), encode(&assignment.to_vec())),
                )?)
            };
            let label = |prompt: &quail_physical::Prompt,
                         assignment: &quail_planner::sol::Assignment| {
                text("label", (encode(prompt), encode(&assignment.to_vec())))
            };
            let accepts = |condition: &quail_physical::Condition, values: &[serde_json::Value]| {
                let found = parse(text(
                    "accepts",
                    (encode(condition), encode(&values.to_vec())),
                )?)?;
                serde_json::from_value::<Vec<bool>>(found)
                    .map_err(|error| quail_planner::CompileError::new(error.to_string()))
            };
            let subset_shared = |alias: &str, rows: &[i64]| {
                let found = parse(text(
                    "subset_shared",
                    (alias.to_string(), encode(&rows.to_vec())),
                )?)?;
                serde_json::from_value::<Vec<i64>>(found)
                    .map_err(|error| quail_planner::CompileError::new(error.to_string()))
            };
            let has_label = callbacks.getattr("has_label")?.extract::<bool>()?;
            let oracles = quail_planner::sol::Oracles {
                answer: &answer,
                label: if has_label { Some(&label) } else { None },
                accepts: &accepts,
                subset_shared: &subset_shared,
            };
            lowering_result(quail_planner::sol::speed_of_light_estimate(
                &plan, &context, inputs, &oracles,
            ))
        }
        "price" => {
            let pricer = read_pricer(&get("pricer"))?;
            let graph: quail_physical::PhysicalGraph = json_value(get("graph"))?;
            lowering_result(pricer.price(&context, &graph.nodes).map(|(seconds, estimates)| {
                serde_json::json!({"seconds": seconds, "estimates": estimates})
            }))
        }
        "request_order" => {
            let plan: quail_planner::LogicalPlan = json_value(get("plan"))?;
            lowering_result(quail_planner::request_order::request_order(&plan, &context))
        }
        "relational_plan" => {
            let plan: quail_planner::LogicalPlan = json_value(get("plan"))?;
            let num_documents = get("num_documents").as_i64().unwrap_or_default();
            lowering_result(Ok::<_, quail_planner::CompileError>(
                quail_planner::graph::relational_plan(&plan, num_documents, &context),
            ))
        }
        "refine_plan" => {
            let envelope: quail_physical::PlanEnvelope = json_value(get("envelope"))?;
            let rules: Vec<String> = json_value(get("rules"))?;
            let plan = quail_planner::graph::PhysicalPlan {
                model: envelope.model,
                device: envelope.device,
                gpus: envelope.gpus,
                backend: envelope.backend,
                estimated_seconds: get("estimated_seconds").as_f64().unwrap_or_default(),
                nodes: envelope.graph.nodes,
                settings: envelope.settings,
                root: envelope.graph.root,
                estimates: json_value(get("estimates"))?,
                pricer: read_pricer(&get("pricer"))?,
            };
            let pricer = pricer_fields(&plan.pricer);
            lowering_result(
                quail_planner::physical_rules::apply_plan_rules(plan, &rules, &context).map(
                    |found| match found {
                        Ok(plan) => serde_json::json!({
                            "envelope": plan.envelope(),
                            "estimated_seconds": plan.estimated_seconds,
                            "estimates": plan.estimates,
                            "pricer": pricer,
                        }),
                        Err(refusal) => refusal_fields(&refusal),
                    },
                ),
            )
        }
        "search_plan" => {
            let alternatives: Vec<(quail_planner::LogicalPlan, Vec<String>)> =
                json_value(get("alternatives"))?;
            let alternatives: Vec<quail_planner::rules::LogicalAlternative> = alternatives
                .into_iter()
                .map(|(plan, rules)| quail_planner::rules::LogicalAlternative { plan, rules })
                .collect();
            let rules: Vec<String> = json_value(get("rules"))?;
            lowering_result(
                quail_planner::optimize::search_plan(&alternatives, &context, &rules).map(
                    |found| match found {
                        Ok(selected) => serde_json::json!({
                            "envelope": selected.plan.envelope(),
                            "estimated_seconds": selected.plan.estimated_seconds,
                            "estimates": selected.plan.estimates,
                            "alternative": selected.alternative,
                            "pricer": pricer_fields(&selected.plan.pricer),
                        }),
                        Err(refusal) => refusal_fields(&refusal),
                    },
                ),
            )
        }
        "plan_region" => {
            let plan: quail_planner::LogicalPlan = json_value(get("plan"))?;
            lowering_result(
                quail_planner::graph::plan_region(&plan, &context).map(|found| match found {
                    Ok(physical) => serde_json::json!({
                        "envelope": physical.envelope(),
                        "estimated_seconds": physical.estimated_seconds,
                        "estimates": physical.estimates,
                    }),
                    Err(refusal) => refusal_fields(&refusal),
                }),
            )
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "no operator planning function {other}"
        ))),
    }
}

fn json_value<T: serde::de::DeserializeOwned>(found: serde_json::Value) -> PyResult<T> {
    serde_json::from_value(found)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// A refusal, as the test helpers compare it.
fn refusal_fields(refusal: &quail_planner::search::Refusal) -> serde_json::Value {
    serde_json::json!({"refusal": {
        "reasons": refusal.reasons, "constraint": refusal.constraint,
        "needed": refusal.needed, "available": refusal.available, "unit": refusal.unit,
    }})
}

/// An operator order, or its refusal, as the test helpers compare it.
fn operator_order_fields(
    found: &Result<quail_planner::search::OperatorOrder, quail_planner::search::Refusal>,
) -> serde_json::Value {
    use quail_planner::search::StepKey;
    let order = match found {
        Err(refusal) => return refusal_fields(refusal),
        Ok(order) => order,
    };
    let statistics = &order.statistics;
    let key = |key: &StepKey| match key {
        StepKey::Position(position) => serde_json::json!(position),
        StepKey::Name(name) => serde_json::json!(name),
    };
    serde_json::json!({
        "statistics": {
            "stats": statistics.stats.iter().map(|(alias, stats)| (alias.clone(), serde_json::json!([
                stats.num_documents, stats.total_tokens, stats.max_document_tokens]))).collect::<serde_json::Map<_, _>>(),
            "lengths": statistics.lengths.iter().map(|(alias, found)| (alias.clone(), serde_json::json!([
                found.count, found.total, found.squared.to_string(), found.maximum, found.window, found.short_counts]))).collect::<serde_json::Map<_, _>>(),
            "chunk": statistics.chunk, "arena_pages": [statistics.arena_pages.0, statistics.arena_pages.1],
            "admission": statistics.admission, "cap_pages": statistics.cap_pages, "pre": statistics.pre,
            "asks": statistics.asks.iter().map(|(alias, positions)| (alias.clone(), serde_json::json!(positions))).collect::<serde_json::Map<_, _>>(),
            "specs": statistics.specs.iter().map(|spec| spec.written_position).collect::<Vec<_>>(),
        },
        "filter_orders": order.filter_orders.iter().map(|(alias, positions)| (alias.clone(), serde_json::json!(positions))).collect::<serde_json::Map<_, _>>(),
        "step_orders": order.step_orders.iter().map(|(alias, steps)| (alias.clone(), serde_json::json!(steps.iter().map(|step| serde_json::json!({
            "kind": step.kind, "key": key(&step.key), "selectivity": step.selectivity,
            "reuse_key": step.reuse_key.is_some(), "document_head": step.document_head,
        })).collect::<Vec<_>>()))).collect::<serde_json::Map<_, _>>(),
        "join_sequence": order.join_sequence.iter().map(|(spec, anchor)| serde_json::json!([spec.written_position, anchor])).collect::<Vec<_>>(),
        "join_steps": order.join_steps.iter().map(|step| serde_json::json!({
            "written_position": step.written_position, "anchor": step.anchor, "resident": step.resident,
            "resident_documents": step.resident_documents, "anchor_documents": step.anchor_documents,
            "tuples": step.tuples, "tokens": step.tokens, "work": work_fields(&step.work),
        })).collect::<Vec<_>>(),
        "expected_rows": order.expected_rows,
        "filter_work": order.filter_work.iter().map(|(alias, work)| (alias.clone(), serde_json::json!(work_fields(work)))).collect::<serde_json::Map<_, _>>(),
        "order_rule": order.order_rule, "order_source": order.order_source,
    })
}

/// A Work record's fields, as a dict of floats.
fn work_fields(work: &quail_cost::work::Work) -> std::collections::BTreeMap<&'static str, f64> {
    std::collections::BTreeMap::from([
        ("tokens", work.tokens.total()),
        ("pairs", work.pairs.total()),
        ("kv_written", work.kv_written.total()),
        ("kv_read", work.kv_read.total()),
        ("sliding_pairs", work.sliding_pairs.total()),
        ("sliding_kv_read", work.sliding_kv_read.total()),
        ("state_written", work.state_written.total()),
        ("state_read", work.state_read.total()),
        ("sae_encoded", work.sae_encoded.total()),
    ])
}

#[pymodule]
fn _quail_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeContext>()?;
    module.add_class::<cost::CostModel>()?;
    module.add_function(wrap_pyfunction!(read_and_write_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(validate_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(build_select, module)?)?;
    module.add_function(wrap_pyfunction!(bind_prompt, module)?)?;
    module.add_function(wrap_pyfunction!(logical_alternatives, module)?)?;
    module.add_function(wrap_pyfunction!(prompt_lowering, module)?)?;
    module.add_function(wrap_pyfunction!(plan_operator, module)?)?;
    module.add_function(wrap_pyfunction!(read_and_write_plan, module)?)
}

/// Values the speed of light callbacks pass to Python as JSON text.
mod erased_json {
    /// A value written as JSON text.
    pub trait Json {
        fn json(&self) -> String;
    }

    impl<T: serde::Serialize> Json for T {
        fn json(&self) -> String {
            serde_json::to_string(self).expect("a JSON value")
        }
    }
}

const WORK_FIELDS: [&str; 9] = [
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

fn work_values(work: &quail_cost::work::Work) -> [&quail_cost::quantity::Quantity; 9] {
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
/// edited graph.
fn pricer_fields(pricer: &quail_planner::pricing::PlanPricer) -> serde_json::Value {
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

fn read_pricer(value: &serde_json::Value) -> PyResult<quail_planner::pricing::PlanPricer> {
    let bad = || pyo3::exceptions::PyValueError::new_err("a malformed plan pricer");
    let stats: std::collections::BTreeMap<String, (i64, i64, i64)> =
        json_value(value["stats"].clone())?;
    let works: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f64>> =
        json_value(value["stage_works"].clone())?;
    let mut stage_works = std::collections::BTreeMap::new();
    for (position, fields) in works {
        let field = |name: &str| quail_cost::quantity::Quantity::Scalar(fields[name]);
        stage_works.insert(
            position.parse().map_err(|_| bad())?,
            quail_cost::work::Work {
                tokens: field("tokens"),
                pairs: field("pairs"),
                kv_written: field("kv_written"),
                kv_read: field("kv_read"),
                sliding_pairs: field("sliding_pairs"),
                sliding_kv_read: field("sliding_kv_read"),
                state_written: field("state_written"),
                state_read: field("state_read"),
                sae_encoded: field("sae_encoded"),
            },
        );
    }
    Ok(quail_planner::pricing::PlanPricer {
        chunk: value["chunk"].as_i64().ok_or_else(bad)?,
        cap_pages: value["cap_pages"].as_i64().ok_or_else(bad)?,
        stats: stats
            .into_iter()
            .map(
                |(alias, (num_documents, total_tokens, max_document_tokens))| {
                    (
                        alias,
                        quail_planner::plan_statistics::CorpusStats {
                            num_documents,
                            total_tokens,
                            max_document_tokens,
                        },
                    )
                },
            )
            .collect(),
        stage_works,
        expected_rows: json_value(value["expected_rows"].clone())?,
    })
}
