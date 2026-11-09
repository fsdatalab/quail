//! The `_quail_native` Python module: Quail's Rust crates for Python.
//!
//! `NativeSession` is `crate::Session`; Python's `quail.Session` runs
//! queries through it, as a Rust program does.

mod budgets;
mod cost;
mod physical;

use std::io::Cursor;

use crate::planning::pricer_fields;
use crate::pyarrow::to_pyarrow;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use quail_datafusion::derive;

fn python_error(error: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Quail's Rust `Session`: DataFusion with Quail's AI functions, whose
/// model regions run on a Python engine.
#[pyclass(name = "NativeSession", module = "_quail_native")]
struct NativeSession {
    session: crate::Session,
}

#[pymethods]
impl NativeSession {
    /// Create a session whose model regions run on `engine`, planned by
    /// `planner` when given, as `crate::Session::with_planner` describes.
    #[new]
    #[pyo3(signature = (engine, planner=None))]
    fn new(engine: Py<PyAny>, planner: Option<PyRef<'_, NativePlanner>>) -> PyResult<Self> {
        let planner = planner.map(|found| std::sync::Arc::clone(&found.planner));
        Ok(Self {
            session: crate::Session::with_planner(engine, planner).map_err(python_error)?,
        })
    }

    /// Register a table by its schema, given as the IPC stream bytes of
    /// an empty table, and a function `scan(columns)` that returns
    /// those columns' rows as a pyarrow Table. `id_column` names the
    /// column whose values are unique, and `identity` the rows.
    #[pyo3(signature = (name, schema, scan, id_column=None, identity=""))]
    fn register(
        &self,
        name: &str,
        schema: &[u8],
        scan: Py<PyAny>,
        id_column: Option<&str>,
        identity: &str,
    ) -> PyResult<()> {
        let reader = StreamReader::try_new(Cursor::new(schema), None)
            .map_err(|error| python_error(error.into()))?;
        self.session
            .register_python_table(name, reader.schema(), scan, id_column, identity)
            .map_err(python_error)
    }

    /// Run a query, read in `dialect` ("Snowflake" by default), and
    /// return its rows as a pyarrow Table over DataFusion's buffers.
    #[pyo3(signature = (query, dialect=None))]
    fn sql<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        dialect: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (schema, batches) = py
            .detach(|| self.session.sql_in_dialect(query, dialect.as_deref()))
            .map_err(python_error)?;
        to_pyarrow(py, schema, batches)
    }

    /// Derive each model region of a query's operators from its plan.
    ///
    /// Returns one (the region as JSON, None, false) or (None, CompileError
    /// message, whether DataFusion runs that shape instead) per region,
    /// and whether one region is the whole query. The region's JSON holds
    /// `operators`, its operators as JSON text, `logical`, DataFusion's
    /// plan of the region, and `warnings`, such as two calls a typo apart. `turn` and
    /// `layout` are the model's chat-turn text and prompt layout.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query, turn, layout, dialect=None))]
    fn region_plans(
        &self,
        py: Python<'_>,
        query: &str,
        turn: (String, String),
        layout: &str,
        dialect: Option<String>,
    ) -> PyResult<(Vec<(Option<String>, Option<String>, bool)>, bool)> {
        py.detach(|| {
            let plan = self.session.optimized_plan(query, dialect.as_deref())?;
            let whole = matches!(&plan, LogicalPlan::Extension(_));
            let mut derived = Vec::new();
            for region in self.session.regions(query, dialect.as_deref())? {
                derived.push(match derive::derive_region(&region, &turn, layout) {
                    Ok((operators, plan)) => {
                        (Some(crate::region_json(&operators, &plan)?), None, false)
                    }
                    Err(error) => {
                        let unsupported = error.is_unsupported();
                        (None, Some(error.to_string()), unsupported)
                    }
                });
            }
            Ok::<_, DataFusionError>((derived, whole))
        })
        .map_err(python_error)
    }

    /// Plan each model region of a query without running it: per region,
    /// (the region as JSON, None when Python plans it or else (its
    /// planned region as JSON, its scans as `crate::scans_to_pyarrow`
    /// hands them over))).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query, dialect=None))]
    fn plans<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        dialect: Option<String>,
    ) -> PyResult<Vec<(String, Option<(String, Bound<'py, pyo3::types::PyDict>)>)>> {
        let found = py
            .detach(|| self.session.plans(query, dialect.as_deref()))
            .map_err(python_error)?;
        found
            .into_iter()
            .map(|(payload, planned)| {
                let planned = planned
                    .map(|(plan, scans)| {
                        Ok::<_, PyErr>((plan, crate::scans_to_pyarrow(py, &scans)?))
                    })
                    .transpose()?;
                Ok((payload, planned))
            })
            .collect()
    }

    /// Return the inputs of each model region of a query that call no
    /// model, as dicts of pyarrow Tables by input name.
    #[pyo3(signature = (query, dialect=None))]
    fn region_inputs<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        dialect: Option<String>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        let regions = py
            .detach(|| self.session.region_inputs(query, dialect.as_deref()))
            .map_err(python_error)?;
        regions
            .into_iter()
            .map(|inputs| {
                let found = pyo3::types::PyDict::new(py);
                for (name, schema, batches) in inputs {
                    found.set_item(name, to_pyarrow(py, schema, batches)?)?;
                }
                Ok(found)
            })
            .collect()
    }

    /// Return DataFusion's optimized logical plan for a query read in
    /// `dialect` ("Snowflake" by default).
    #[pyo3(signature = (query, dialect=None))]
    fn explain(&self, py: Python<'_>, query: &str, dialect: Option<String>) -> PyResult<String> {
        py.detach(|| self.session.explain(query, dialect.as_deref()))
            .map_err(python_error)
    }
}

/// Read a physical plan envelope into the Rust plan types and write
/// it back as JSON.
///
/// Raises a ValueError when the plan does not fit the Rust types.
#[pyfunction]
fn read_and_write_plan(text: &str) -> PyResult<String> {
    let plan = quail_planner::physical::read_plan(text)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    quail_planner::physical::write_plan(&plan)
        .map_err(|error| PyValueError::new_err(error.to_string()))
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
    let invalid = |error: serde_json::Error| PyValueError::new_err(error.to_string());
    let args: Vec<quail_planner::physical::ValueExpression> =
        serde_json::from_str(args).map_err(invalid)?;
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
        Err(error) => Ok((None, Some(error.to_string()))),
    }
}

/// The placements of a region's column operators that physical planning
/// prices, as `quail_planner::placements::placements` returns them.
///
/// Returns [(operators JSON, rewrite names), ...].
#[pyfunction]
fn column_placements(operators: &str) -> PyResult<Vec<(String, Vec<String>)>> {
    let invalid = |error: serde_json::Error| PyValueError::new_err(error.to_string());
    let found: quail_planner::operators::Operators =
        serde_json::from_str(operators).map_err(invalid)?;
    quail_planner::placements::placements(&found)
        .into_iter()
        .map(|alternative| {
            Ok((
                serde_json::to_string(&alternative.operators).map_err(invalid)?,
                alternative.rules,
            ))
        })
        .collect()
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
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
            None,
        )),
        Err(error) => Ok((None, Some(error.to_string()))),
    }
}

fn from_json<T: serde::de::DeserializeOwned>(text: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Run one function of the planner's prompt lowering (`quail_planner::lowering`).
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
        "answer_ids" => lowering_result(lower::answer_ids(&from_json(&get(0))?, &tokens)),
        "set_question_parts" => {
            let layout: String = from_json(&get(1))?;
            lowering_result(
                lower::lower_prompt(&from_json(&get(0))?, &tokens)
                    .and_then(|lowered| lower::set_question_parts(&lowered, &layout, &tokens)),
            )
        }
        "option_aliases" => lowering_result(lower::option_aliases(
            &from_json::<Vec<String>>(&get(0))?,
            &tokens,
        )),
        "option_spellings" => lowering_result(lower::option_spellings(
            &from_json::<Vec<String>>(&get(0))?,
            &tokens,
        )),
        "read_slot_offsets" => lowering_result(lower::read_slot_offsets(
            &from_json::<String>(&get(0))?,
            &from_json::<Vec<String>>(&get(1))?,
            &tokens,
        )),
        other => Err(PyValueError::new_err(format!(
            "no prompt lowering function {other}"
        ))),
    }
}

/// The token ids a model's answer readout reads, from its spec as JSON:
/// (TRUE ids, FALSE ids, the decision readout's row offsets).
///
/// The GPU worker calls this when it loads a model, before any plan.
#[pyfunction]
fn model_answers<'py>(
    py: Python<'py>,
    model: &str,
    tokenizer: Bound<'py, PyAny>,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<usize>)> {
    let spec: quail_planner::cost::specs::ModelSpec = from_json(model)?;
    let model = quail_planner::lowering::PromptModel {
        role: spec.role.clone(),
        answer_spellings: spec.answer_spellings.clone(),
        turn: (spec.turn_prefix.clone(), spec.turn_suffix.clone()),
        prompt_layout: spec.prompt_layout.clone(),
    };
    let tokens = python_tokenizer(py, &tokenizer);
    let failed = |error: quail_planner::CompileError| PyRuntimeError::new_err(error.to_string());
    let (true_ids, false_ids) =
        quail_planner::lowering::answer_ids(&model, &tokens).map_err(failed)?;
    let offsets = quail_planner::lowering::answer_row_offsets(&model, &tokens).map_err(failed)?;
    Ok((true_ids, false_ids, offsets))
}

/// Every model or device spec in a directory, by name, as JSON; each
/// model's `base` is already applied. `kind` is "models" or "devices".
#[pyfunction]
fn read_specs(kind: &str, directory: &str) -> PyResult<String> {
    let directory = std::path::Path::new(directory);
    let failed = |error: quail_planner::cost::specs::SpecError| PyValueError::new_err(error.0);
    let found = if kind == "models" {
        serde_json::to_value(quail_planner::cost::specs::read_models(directory).map_err(failed)?)
    } else {
        serde_json::to_value(quail_planner::cost::specs::read_devices(directory).map_err(failed)?)
    };
    found
        .map(|value| value.to_string())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// A prompt layout's fixed text, by layout name, as JSON.
#[pyfunction]
fn prompt_layout(name: &str) -> PyResult<String> {
    let layout = quail_planner::prompts::prompt_layout(name)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&layout).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// The planning context's fields, as `tests/helpers/planning.py` and
/// `NativePlanner.plan` write them.
#[derive(serde::Deserialize)]
struct ContextFields {
    model: serde_json::Value,
    device: serde_json::Value,
    gpus: i64,
    canvas_draws: i64,
    document_tokens: std::collections::BTreeMap<String, Vec<i64>>,
    operators: Option<serde_json::Value>,
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

/// Run one planning function of `quail_planner` by name.
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
        operators: fields
            .operators
            .map(json_value::<quail_planner::operators::Operators>)
            .transpose()?,
        first_documents: fields.first_documents,
        page_shared: fields.page_shared,
        reader_groups: fields.reader_groups,
        last_tokens: fields.last_tokens,
        backend: fields.backend,
    };
    let arguments: serde_json::Value = from_json(arguments)?;
    let get = |key: &str| arguments[key].clone();
    match name {
        "speed_of_light" => {
            let Some(callbacks) = callbacks else {
                return Err(PyValueError::new_err(
                    "the speed of light estimate needs its oracles",
                ));
            };
            let found: quail_planner::operators::Operators = json_value(get("operators"))?;
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
            let answer = |prompt: &quail_planner::physical::Prompt,
                          assignment: &quail_planner::sol::Assignment| {
                parse(text(
                    "answer",
                    (encode(prompt), encode(&assignment.to_vec())),
                )?)
            };
            let label = |prompt: &quail_planner::physical::Prompt,
                         assignment: &quail_planner::sol::Assignment| {
                text("label", (encode(prompt), encode(&assignment.to_vec())))
            };
            let accepts = |condition: &quail_planner::physical::ValueExpression,
                           values: &[serde_json::Value]| {
                accepted_answers(condition, values)
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
                &found, &context, inputs, &oracles,
            ))
        }
        "price" => {
            let pricer = read_pricer(&get("pricer"))?;
            let graph: quail_planner::physical::PhysicalGraph = json_value(get("graph"))?;
            quail_planner::physical::validate_graph(&graph.nodes, &graph.root)
                .map_err(PyValueError::new_err)?;
            lowering_result(pricer.price(&context, &graph.nodes).map(|(seconds, estimates)| {
                serde_json::json!({"seconds": seconds, "estimates": estimates})
            }))
        }
        "edit_plan" => {
            let pricer = read_pricer(&get("pricer"))?;
            let graph: quail_planner::physical::PhysicalGraph = json_value(get("graph"))?;
            let edit: quail_planner::edits::Edit = json_value(get("edit"))?;
            let retained: std::collections::BTreeSet<String> = json_value(get("retained"))?;
            let graph = quail_planner::physical::PhysicalGraph {
                nodes: graph.nodes,
                root: graph.root,
            };
            let edited = quail_planner::edits::edit_graph(&graph, &edit)
                .map_err(quail_planner::CompileError::new)
                .and_then(|edited| {
                    let (seconds, estimates) = pricer.price(&context, &edited.nodes)?;
                    let recompute =
                        quail_planner::edits::recompute_seconds(&edited, &estimates, &retained);
                    Ok(serde_json::json!({
                        "graph": {"nodes": edited.nodes, "root": edited.root},
                        "seconds": seconds + recompute, "estimates": estimates,
                    }))
                });
            lowering_result(edited)
        }
        "search_plan" => {
            let alternatives: Vec<(quail_planner::operators::Operators, Vec<String>)> =
                json_value(get("alternatives"))?;
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
        other => Err(PyValueError::new_err(format!(
            "no operator planning function {other}"
        ))),
    }
}

fn json_value<T: serde::de::DeserializeOwned>(found: serde_json::Value) -> PyResult<T> {
    serde_json::from_value(found).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A refusal, as the test helpers compare it.
fn refusal_fields(refusal: &quail_planner::error::Refusal) -> serde_json::Value {
    serde_json::json!({"refusal": crate::planning::refusal_json(refusal)})
}

/// The token lists of a pyarrow Table's first column, as int32 lists.
fn token_batches(
    table: &Bound<'_, PyAny>,
) -> PyResult<Vec<datafusion::arrow::array::LargeListArray>> {
    use datafusion::arrow::array::AsArray;
    use datafusion::arrow::datatypes::{DataType, Field};
    let lists = DataType::LargeList(std::sync::Arc::new(Field::new_list_field(
        DataType::Int32,
        true,
    )));
    crate::pyarrow::from_pyarrow(table)?
        .iter()
        .map(|batch| {
            let column = batch
                .columns()
                .first()
                .ok_or_else(|| PyValueError::new_err("a token table needs a column"))?;
            let column = datafusion::arrow::compute::cast(column, &lists).map_err(arrow_error)?;
            Ok(column.as_list::<i64>().clone())
        })
        .collect()
}

/// The documents of a token store or any token sequences sorted by their
/// tokens, as `quail_planner::inputs::PrefixIndex`; its trees at each page size
/// share the sort.
#[pyclass(name = "PrefixIndex", module = "_quail_native", frozen)]
struct PrefixIndex {
    index: quail_planner::inputs::PrefixIndex,
}

#[pymethods]
impl PrefixIndex {
    /// Sort the documents of `tokens`, a pyarrow Table whose first column
    /// holds each document's integer token ids.
    #[new]
    fn new(py: Python<'_>, tokens: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batches = token_batches(tokens)?;
        let index = py.detach(|| {
            quail_planner::inputs::PrefixIndex::new(&quail_planner::inputs::Documents::new(
                &batches,
            ))
        });
        Ok(Self { index })
    }

    /// The prefix tree whose shares round down to `page_tokens`: each
    /// document's position in sorted order, the position it borrows from
    /// or None, and the tokens it borrows.
    #[allow(clippy::type_complexity)]
    fn tree(&self, page_tokens: i64) -> PyResult<(Vec<usize>, Vec<Option<usize>>, Vec<i64>)> {
        if page_tokens < 1 {
            return Err(PyValueError::new_err("page_tokens must be positive"));
        }
        let tree = self.index.tree(page_tokens);
        Ok((tree.order, tree.parent, tree.shared))
    }
}

/// Each document's first document with the same tokens and the same
/// values in each column of `columns`, as `quail_planner::inputs::first_documents`:
/// an int32 pyarrow Array. `tokens` is a pyarrow Table whose first column
/// holds each document's token ids; `columns` a pyarrow Table or None.
#[pyfunction]
#[pyo3(signature = (tokens, columns=None))]
fn first_documents<'py>(
    py: Python<'py>,
    tokens: &Bound<'py, PyAny>,
    columns: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use datafusion::arrow::array::{ArrayRef, Int32Array, RecordBatchReader};
    let batches = token_batches(tokens)?;
    let values: Vec<ArrayRef> = match columns {
        None => Vec::new(),
        Some(table) => {
            let reader = crate::pyarrow::stream_from_pyarrow(table)?;
            let schema = reader.schema();
            let found = reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(arrow_error)?;
            datafusion::arrow::compute::concat_batches(&schema, &found)
                .map_err(arrow_error)?
                .columns()
                .to_vec()
        }
    };
    let first = py
        .detach(|| {
            quail_planner::inputs::first_documents(
                &quail_planner::inputs::Documents::new(&batches),
                &values,
            )
        })
        .map_err(arrow_error)?;
    let first: ArrayRef = std::sync::Arc::new(
        first
            .into_iter()
            .map(|position| position as i32)
            .collect::<Int32Array>(),
    );
    crate::pyarrow::array_to_pyarrow(py, &first)
}

/// A model's tokenizer, read from its `tokenizer.json`: the one the
/// planner tokenizes documents with and the GPU worker decodes with.
#[pyclass(name = "Tokenizer", module = "_quail_native", frozen)]
struct Tokenizer {
    inner: quail_planner::inputs::HfTokenizer,
    /// The ids that end an answer: the model's turn close and its end of
    /// text, which no document holds.
    #[pyo3(get)]
    stop_ids: Vec<u32>,
}

fn arrow_error(error: datafusion::arrow::error::ArrowError) -> PyErr {
    python_error(error.into())
}

#[pymethods]
impl Tokenizer {
    /// The tokenizer at `path`, a model directory or its
    /// `tokenizer.json`; `turn_suffix` is the model's chat-turn text after
    /// a prompt, whose first token closes a turn.
    #[new]
    #[pyo3(signature = (path, turn_suffix=""))]
    fn new(path: std::path::PathBuf, turn_suffix: &str) -> PyResult<Self> {
        let inner = quail_planner::inputs::HfTokenizer::from_file(&path).map_err(arrow_error)?;
        let mut stop_ids: Vec<u32> = inner.eos_token_id().into_iter().collect();
        if !turn_suffix.is_empty()
            && let Some(&close) = inner.encode(turn_suffix).map_err(arrow_error)?.first()
        {
            stop_ids.push(close);
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        Ok(Self { inner, stop_ids })
    }

    /// The token ids of a text, without special tokens.
    fn __call__(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode(text).map_err(arrow_error)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The end-of-text token the model's config names, or None.
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// Every text's token ids, encoded together, with each token's
    /// character range: three buffers of little-endian int64, the ids,
    /// the (start, end) pairs, and each text's first token with one past
    /// the last.
    fn encode_packed<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PyBytes>,
        Bound<'py, pyo3::types::PyBytes>,
        Bound<'py, pyo3::types::PyBytes>,
    )> {
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let found = self
            .inner
            .encode_with_offsets(&texts)
            .map_err(arrow_error)?;
        let ids: Vec<u8> = found
            .ids
            .iter()
            .flat_map(|id| i64::from(*id).to_le_bytes())
            .collect();
        let ranges: Vec<u8> = found
            .offsets
            .iter()
            .flat_map(|(start, end)| [start.to_le_bytes(), end.to_le_bytes()].concat())
            .collect();
        let boundaries: Vec<u8> = found
            .boundaries
            .iter()
            .flat_map(|boundary| boundary.to_le_bytes())
            .collect();
        Ok((
            pyo3::types::PyBytes::new(py, &ids),
            pyo3::types::PyBytes::new(py, &ranges),
            pyo3::types::PyBytes::new(py, &boundaries),
        ))
    }

    /// The bytes of each token id's text, or None for an id the
    /// vocabulary does not hold.
    fn token_bytes<'py>(&self, py: Python<'py>) -> Vec<Option<Bound<'py, pyo3::types::PyBytes>>> {
        self.inner
            .token_bytes()
            .iter()
            .map(|found| {
                found
                    .as_deref()
                    .map(|bytes| pyo3::types::PyBytes::new(py, bytes))
            })
            .collect()
    }

    /// Each token id's text in the tokenizer's own spelling.
    fn token_texts(&self) -> Vec<String> {
        self.inner.token_texts()
    }

    /// The bytes of token ids' texts, joined.
    fn decode<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.decode_bytes(&ids))
    }

    /// The text of each token id list, without special tokens.
    fn decode_batch(&self, lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        self.inner.decode_batch(&lists).map_err(arrow_error)
    }
}

/// The `explain()` text of a request, as JSON: see
/// `quail_planner::explain::ExplainRequest`.
#[pyfunction]
fn explain_plan(request: &str) -> PyResult<String> {
    let request: quail_planner::explain::ExplainRequest = from_json(request)?;
    quail_planner::explain::explain_text(&request).map_err(PyRuntimeError::new_err)
}

/// Collect a run's answers from its output relations, as
/// `crate::results::collect_answers` does.
///
/// `request` is the AnswerRequest as JSON; `outputs` maps (node id,
/// port) to a pyarrow Table. Returns (answer tables as (kind, key,
/// table), survivors as alias -> one-column table, true join tables by
/// written position, stages as JSON). A key is a name, an (alias,
/// written position) pair, or a written position.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn collect_answers<'py>(
    py: Python<'py>,
    request: &str,
    outputs: std::collections::HashMap<(String, String), Bound<'py, PyAny>>,
) -> PyResult<(
    Vec<(String, Py<PyAny>, Py<PyAny>)>,
    Vec<(String, Py<PyAny>)>,
    Vec<(i64, Py<PyAny>)>,
    String,
)> {
    use pyo3::IntoPyObjectExt;
    let request: crate::results::AnswerRequest = from_json(request)?;
    let relations = read_relations(outputs)?;
    let found =
        crate::results::collect_answers(&request, &relations).map_err(PyValueError::new_err)?;
    let table = |batch: datafusion::arrow::array::RecordBatch| -> PyResult<Py<PyAny>> {
        Ok(crate::pyarrow::to_pyarrow(py, batch.schema(), vec![batch])?.unbind())
    };
    let mut tables = Vec::new();
    for (kind, key, batch) in found.tables {
        let key = match key {
            crate::results::AnswerKey::Name(name) => name.into_py_any(py)?,
            crate::results::AnswerKey::Filter(alias, position) => {
                (alias, position).into_py_any(py)?
            }
            crate::results::AnswerKey::Join(position) => position.into_py_any(py)?,
        };
        tables.push((kind, key, table(batch)?));
    }
    let mut survivors = Vec::new();
    for (alias, ids) in found.survivors {
        let batch = datafusion::arrow::array::RecordBatch::try_from_iter([(
            alias.clone(),
            std::sync::Arc::new(ids) as datafusion::arrow::array::ArrayRef,
        )])
        .map_err(|error| python_error(error.into()))?;
        survivors.push((alias, table(batch)?));
    }
    let mut true_joins = Vec::new();
    for (position, batch) in found.true_joins {
        true_joins.push((position, table(batch)?));
    }
    Ok((
        tables,
        survivors,
        true_joins,
        serde_json::Value::from(found.stages).to_string(),
    ))
}

/// Each pyarrow Table of `outputs` as one record batch, by (node id, port).
fn read_relations(
    outputs: std::collections::HashMap<(String, String), Bound<'_, PyAny>>,
) -> PyResult<std::collections::HashMap<(String, String), datafusion::arrow::array::RecordBatch>> {
    let mut relations = std::collections::HashMap::new();
    for (key, table) in outputs {
        relations.insert(key, one_batch(&table)?);
    }
    Ok(relations)
}

/// A pyarrow Table as one record batch.
fn one_batch(table: &Bound<'_, PyAny>) -> PyResult<datafusion::arrow::array::RecordBatch> {
    let reader = crate::pyarrow::stream_from_pyarrow(table)?;
    let schema = datafusion::arrow::array::RecordBatchReader::schema(&reader);
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| python_error(error.into()))?;
    datafusion::arrow::compute::concat_batches(&schema, &batches)
        .map_err(|error| python_error(error.into()))
}

/// A region's rows, built as its operators report the rows they finish,
/// as `crate::finished::FinishedRows` builds them.
///
/// `request` is the AnswerRequest as JSON, listing the ports the run
/// returns; `evaluate(name, expression)` returns a one-column pyarrow
/// Table of the values a source column's expression, as JSON, computes
/// for each document; `on_rows`, when given, is handed each pyarrow
/// Table of rows as they are built.
#[pyclass(name = "FinishedRows", module = "_quail_native")]
struct NativeFinishedRows {
    rows: crate::finished::FinishedRows,
    on_rows: Option<Py<PyAny>>,
}

#[pymethods]
impl NativeFinishedRows {
    #[new]
    #[pyo3(signature = (request, evaluate, on_rows=None))]
    fn new(
        request: &str,
        evaluate: Bound<'_, PyAny>,
        on_rows: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let request: crate::results::AnswerRequest = from_json(request)?;
        let found: quail_planner::operators::Operators = from_json(&request.operators)?;
        let mut sources = std::collections::HashMap::new();
        for (name, expression) in quail_planner::results::source_columns(&found) {
            let expression = serde_json::to_string(&expression)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let values = one_batch(&evaluate.call1((&name, expression))?)?;
            let column = values.columns().first().ok_or_else(|| {
                PyValueError::new_err(format!("source {name} returned no column"))
            })?;
            sources.insert(name, std::sync::Arc::clone(column));
        }
        let rows = crate::finished::FinishedRows::new(&request, &sources)
            .map_err(PyValueError::new_err)?;
        Ok(NativeFinishedRows { rows, on_rows })
    }

    /// Whether the rows read the reports of a node's port.
    fn wants(&self, node: &str, port: &str) -> bool {
        self.rows.wants(node, port)
    }

    /// Take a pyarrow Table of rows a port finished, the node and port in
    /// its schema's `quail.node` and `quail.port` metadata; hand the
    /// region rows they complete to `on_rows`.
    fn report(&mut self, py: Python<'_>, table: &Bound<'_, PyAny>) -> PyResult<()> {
        let reader = crate::pyarrow::stream_from_pyarrow(table)?;
        let schema = datafusion::arrow::array::RecordBatchReader::schema(&reader);
        let tag = |key: &str| {
            schema.metadata().get(key).cloned().ok_or_else(|| {
                PyValueError::new_err("finished rows need quail.node and quail.port metadata")
            })
        };
        let (node, port) = (tag("quail.node")?, tag("quail.port")?);
        let mut batches = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| python_error(error.into()))?;
        if batches.is_empty() {
            // an empty table still gives its columns' types
            batches.push(datafusion::arrow::array::RecordBatch::new_empty(schema));
        }
        for batch in batches {
            let built = self.rows.report(&node, &port, &batch);
            self.hand_on(py, built)?;
        }
        Ok(())
    }

    /// Hand `on_rows` the rows complete before any report.
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        let built = self.rows.finish();
        self.hand_on(py, built)
    }

    /// Every row built so far, as a pyarrow Table ordered by document ids.
    fn rows<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rows = self.rows.rows().map_err(PyValueError::new_err)?;
        to_pyarrow(py, rows.schema(), vec![rows])
    }
}

impl NativeFinishedRows {
    fn hand_on(
        &self,
        py: Python<'_>,
        built: Result<Option<datafusion::arrow::array::RecordBatch>, String>,
    ) -> PyResult<()> {
        if let (Some(rows), Some(on_rows)) = (built.map_err(PyValueError::new_err)?, &self.on_rows)
        {
            on_rows.call1(py, (to_pyarrow(py, rows.schema(), vec![rows])?,))?;
        }
        Ok(())
    }
}

/// The report of one region's run, as `crate::results::execution_report`
/// builds it; `metrics` is the backend's measurements as JSON.
#[pyfunction]
#[pyo3(signature = (backend, estimated_seconds, order_rule, metrics, coordinator_wall))]
fn execution_report(
    backend: &str,
    estimated_seconds: f64,
    order_rule: Option<&str>,
    metrics: &str,
    coordinator_wall: f64,
) -> PyResult<String> {
    let metrics: serde_json::Map<String, serde_json::Value> = from_json(metrics)?;
    Ok(crate::results::execution_report(
        backend,
        estimated_seconds,
        order_rule,
        &metrics,
        coordinator_wall,
    )
    .to_string())
}

/// Apply an edit to a graph, set KV retention again, and validate it,
/// without pricing: for a plan the planner did not price.
///
/// Returns (the graph as JSON, None) or (None, why the edit fails).
#[pyfunction]
fn edit_graph(graph: &str, edit: &str) -> PyResult<(Option<String>, Option<String>)> {
    let graph: quail_planner::physical::PhysicalGraph = from_json(graph)?;
    let edit: quail_planner::edits::Edit = from_json(edit)?;
    let graph = quail_planner::physical::PhysicalGraph {
        nodes: graph.nodes,
        root: graph.root,
    };
    Ok(match quail_planner::edits::edit_graph(&graph, &edit) {
        Ok(edited) => (
            Some(serde_json::json!({"nodes": edited.nodes, "root": edited.root}).to_string()),
            None,
        ),
        Err(error) => (None, Some(error)),
    })
}

/// A region's facts as JSON, as `quail_planner::facts::region_facts`
/// reads them from its operators, given as JSON.
#[pyfunction]
fn region_facts(operators: &str) -> PyResult<String> {
    quail_planner::facts::region_facts(&read_operators(operators)?)
        .map(|facts| facts.to_string())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A region's operators, as JSON, that return at most `fetch` rows.
#[pyfunction]
fn limited_operators(operators: &str, fetch: i64) -> PyResult<String> {
    serde_json::to_string(&read_operators(operators)?.limited(fetch))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn read_operators(operators: &str) -> PyResult<quail_planner::operators::Operators> {
    serde_json::from_str(operators).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Plans model regions in Rust for one session, with the session's
/// tokenizer, keeping the token stores of tables read whole.
#[pyclass(name = "NativePlanner", module = "_quail_native")]
struct NativePlanner {
    planner: std::sync::Arc<crate::planning::Planner>,
}

#[pymethods]
impl NativePlanner {
    /// A planner with the `tokenizer.json` at `tokenizer_file`.
    #[new]
    fn new(tokenizer_file: &str) -> PyResult<Self> {
        Ok(Self {
            planner: std::sync::Arc::new(
                crate::planning::Planner::new(std::path::Path::new(tokenizer_file))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
        })
    }
}

/// An `Expression` of SQL text over the columns of `rows`, a pyarrow
/// Table or RecordBatch, each a column of table `alias` from `provider`:
/// (the Expression, built for evaluation, None), or (None, why it does
/// not parse).
#[pyfunction]
#[pyo3(signature = (sql, rows, alias="", provider=""))]
fn parse_expression<'py>(
    py: Python<'py>,
    sql: &str,
    rows: &Bound<'py, PyAny>,
    alias: &str,
    provider: &str,
) -> PyResult<(Option<Bound<'py, PyAny>>, Option<String>)> {
    use datafusion::arrow::array::RecordBatchReader;
    let schema = crate::pyarrow::stream_from_pyarrow(rows)?.schema();
    match quail_datafusion::derive::expressions::parse(sql, &schema, alias, provider) {
        Ok(found) => {
            let value =
                serde_json::to_value(quail_planner::physical::ValueExpression::Value(found))
                    .expect("an expression writes");
            Ok((Some(physical::to_python(py, &value)?), None))
        }
        Err(error) => Ok((None, Some(error.to_string()))),
    }
}

/// Cut a plan's graph, as JSON, into stages for several GPUs, as the
/// planner cuts it into the plan's `stages` setting; returns the stages
/// as JSON.
#[pyfunction]
fn region_stages(graph: &str) -> PyResult<String> {
    let graph: quail_planner::physical::PhysicalGraph = from_json(graph)?;
    let cut = quail_planner::stages::stages(&graph).map_err(PyValueError::new_err)?;
    serde_json::to_string(&cut).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Run a region's stages on GPU servers, as `crate::stages::run_stages`
/// runs them.
///
/// `graph` is the region's compute graph as JSON; `stages` is the plan's
/// `stages` setting as JSON; `groupings` gives, by alias, lists of each
/// document's group, for a table hashed by a key other than its ids;
/// `servers` names each GPU's Unix socket; `report(data)` takes each
/// finished table as Arrow IPC bytes.
#[pyfunction]
fn run_stages(
    py: Python<'_>,
    graph: &str,
    stages: &str,
    groupings: std::collections::HashMap<String, Vec<Vec<i64>>>,
    servers: Vec<std::path::PathBuf>,
    report: Py<PyAny>,
) -> PyResult<()> {
    let graph: quail_planner::physical::PhysicalGraph = from_json(graph)?;
    let cut: Vec<quail_planner::stages::Stage> = from_json(stages)?;
    let report: crate::stages::Report = Box::new(move |table: Vec<u8>| {
        Python::attach(|py| {
            report.call1(py, (pyo3::types::PyBytes::new(py, &table),))?;
            Ok::<_, PyErr>(())
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))
    });
    py.detach(|| crate::stages::run_stages(&graph, cut, groupings, &servers, report))
        .map_err(python_error)
}

/// One GPU's server end of a region's stages, as
/// `crate::stages::StageListener`: it listens at `path` once made.
#[pyclass(name = "StageServer", module = "_quail_native")]
struct StageServer {
    listener: crate::stages::StageListener,
}

#[pymethods]
impl StageServer {
    #[new]
    fn new(path: std::path::PathBuf) -> PyResult<Self> {
        let listener = crate::stages::StageListener::bind(&path).map_err(python_error)?;
        Ok(StageServer { listener })
    }

    /// The next stage and its inputs' rows as pyarrow Tables, or None
    /// once the region ended.
    #[allow(clippy::type_complexity)]
    fn next_stage<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, Vec<Bound<'py, PyAny>>)>> {
        let listener = &mut self.listener;
        let Some((stage, inputs)) = py.detach(|| listener.next_stage()).map_err(python_error)?
        else {
            return Ok(None);
        };
        let tables = inputs
            .into_iter()
            .map(|(schema, batches)| to_pyarrow(py, schema, batches))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some((stage, tables)))
    }

    /// Send a stage's output tables and finished tables.
    fn reply(
        &mut self,
        py: Python<'_>,
        outputs: Vec<Bound<'_, PyAny>>,
        finished: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let tables = |values: Vec<Bound<'_, PyAny>>| {
            values
                .iter()
                .map(|value| {
                    let reader = crate::pyarrow::stream_from_pyarrow(value)?;
                    let schema = datafusion::arrow::array::RecordBatchReader::schema(&reader);
                    let batches = reader
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(|error| python_error(error.into()))?;
                    Ok((schema, batches))
                })
                .collect::<PyResult<Vec<crate::stages::Table>>>()
        };
        let (outputs, finished) = (tables(outputs)?, tables(finished)?);
        let listener = &mut self.listener;
        py.detach(|| listener.reply(&outputs, &finished))
            .map_err(python_error)
    }

    /// Send why a stage failed.
    fn fail(&mut self, py: Python<'_>, error: &str) -> PyResult<()> {
        let listener = &mut self.listener;
        py.detach(|| listener.fail(error)).map_err(python_error)
    }
}

/// The `_quail_native` module. A Rust program that embeds Python adds it
/// with `pyo3::append_to_inittab!` before Python starts.
#[pymodule]
pub fn _quail_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    physical::register(module)?;
    // the rule lists Python's registry starts from
    module.add("LOGICAL_RULES", quail_datafusion::default_rule_names())?;
    module.add(
        "PHYSICAL_RULES",
        quail_planner::physical_rules::PHYSICAL_RULES.to_vec(),
    )?;
    module.add_class::<NativeSession>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<PrefixIndex>()?;
    module.add_class::<NativePlanner>()?;
    module.add_class::<NativeFinishedRows>()?;
    module.add_class::<StageServer>()?;
    module.add_class::<cost::CostModel>()?;
    budgets::register(module)?;
    module.add_function(wrap_pyfunction!(explain_plan, module)?)?;
    module.add_function(wrap_pyfunction!(parse_expression, module)?)?;
    module.add_function(wrap_pyfunction!(model_answers, module)?)?;
    module.add_function(wrap_pyfunction!(first_documents, module)?)?;
    module.add_function(wrap_pyfunction!(prompt_layout, module)?)?;
    module.add_function(wrap_pyfunction!(read_specs, module)?)?;
    module.add("SHARED_PRE", quail_planner::prompts::SHARED_PRE)?;
    module.add_function(wrap_pyfunction!(bind_prompt, module)?)?;
    module.add_function(wrap_pyfunction!(column_placements, module)?)?;
    module.add_function(wrap_pyfunction!(prompt_lowering, module)?)?;
    module.add_function(wrap_pyfunction!(plan_operator, module)?)?;
    module.add_function(wrap_pyfunction!(collect_answers, module)?)?;
    module.add_function(wrap_pyfunction!(edit_graph, module)?)?;
    module.add_function(wrap_pyfunction!(region_facts, module)?)?;
    module.add_function(wrap_pyfunction!(limited_operators, module)?)?;
    module.add_function(wrap_pyfunction!(execution_report, module)?)?;
    module.add_function(wrap_pyfunction!(region_stages, module)?)?;
    module.add_function(wrap_pyfunction!(run_stages, module)?)?;
    module.add_function(wrap_pyfunction!(read_and_write_plan, module)?)
}

/// Which answers a one-column condition accepts, each answer text or
/// null, with every null rejected.
fn accepted_answers(
    condition: &quail_planner::physical::ValueExpression,
    values: &[serde_json::Value],
) -> Result<Vec<bool>, quail_planner::CompileError> {
    use datafusion::arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StringArray};
    let built = quail_datafusion::derive::expressions::PhysicalValue::new(condition)?;
    let [column] = built.columns()[..] else {
        return Err(quail_planner::CompileError::new(
            "an answer test reads one column",
        ));
    };
    let texts = values
        .iter()
        .map(|value| match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(text) => Ok(Some(text.as_str())),
            other => Err(quail_planner::CompileError::new(format!(
                "an answer test reads text, not {other}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let answers: ArrayRef = std::sync::Arc::new(StringArray::from(texts));
    let batch = RecordBatch::try_from_iter([(column.name(false), answers)])
        .map_err(|error| quail_planner::CompileError::new(error.to_string()))?;
    let mask = built.evaluate(&batch, false)?;
    let mask = mask.as_boolean_opt().ok_or_else(|| {
        quail_planner::CompileError::new("an answer test gives a boolean per answer")
    })?;
    Ok((0..mask.len())
        .map(|index| mask.is_valid(index) && mask.value(index))
        .collect())
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

/// Read a plan's pricer from the JSON [`pricer_fields`] writes.
fn read_pricer(value: &serde_json::Value) -> PyResult<quail_planner::pricing::PlanPricer> {
    let bad = || PyValueError::new_err("a malformed plan pricer");
    let stats: std::collections::BTreeMap<String, (i64, i64, i64)> =
        json_value(value["stats"].clone())?;
    let works: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f64>> =
        json_value(value["stage_works"].clone())?;
    let mut stage_works = std::collections::BTreeMap::new();
    for (position, fields) in works {
        let field = |name: &str| quail_planner::cost::quantity::Quantity::Scalar(fields[name]);
        stage_works.insert(
            position.parse().map_err(|_| bad())?,
            quail_planner::cost::work::Work {
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
