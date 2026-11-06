//! The `_quail_native` Python module: Quail's Rust crates for Python.
//!
//! `NativeSession` is `quail::Session`; Python's `quail.Session` runs
//! queries through it, as a Rust program does.

mod cost;

use std::io::Cursor;

use datafusion::arrow::array::AsArray;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use quail::planning::pricer_fields;
use quail::pyarrow::to_pyarrow;
use quail_datafusion::{bind, region_sql};

fn python_error(error: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Quail's Rust `Session`: DataFusion with Quail's AI functions, whose
/// model regions run on a Python engine.
#[pyclass(name = "NativeSession", module = "_quail_native")]
struct NativeSession {
    session: quail::Session,
}

#[pymethods]
impl NativeSession {
    /// Create a session whose model regions run on `engine`, planned by
    /// `planner` when given, as `quail::Session::with_planner` describes.
    #[new]
    #[pyo3(signature = (engine, planner=None))]
    fn new(engine: Py<PyAny>, planner: Option<PyRef<'_, NativePlanner>>) -> PyResult<Self> {
        let planner = planner.map(|found| std::sync::Arc::clone(&found.planner));
        Ok(Self {
            session: quail::Session::with_planner(engine, planner).map_err(python_error)?,
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

    /// Return the SQL each model region of a query sends to Quail.
    fn region_sql(&self, py: Python<'_>, query: &str) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.session
                .regions(query, None)?
                .iter()
                .map(region_sql)
                .collect::<Result<Vec<_>>>()
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
            let plan = self.session.optimized_plan(query, dialect.as_deref())?;
            let whole = matches!(&plan, LogicalPlan::Extension(_));
            let mut bound = Vec::new();
            for region in self.session.regions(query, dialect.as_deref())? {
                bound.push(match bind::bind_model_region(&region, &turn, layout) {
                    Ok(found) => (
                        Some(
                            quail_planner::write_logical_plan(&found)
                                .map_err(|error| DataFusionError::External(Box::new(error)))?,
                        ),
                        None,
                    ),
                    Err(error) => (None, Some(error.0)),
                });
            }
            Ok::<_, DataFusionError>((bound, whole))
        })
        .map_err(python_error)
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

/// The placements of a plan's column operators that physical planning
/// prices, as `quail_planner::rules::column_placements` returns them.
///
/// Returns ([(plan JSON, rewrite names), ...], None), or ([], the
/// CompileError's message).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn column_placements(plan: &str) -> PyResult<(Vec<(String, Vec<String>)>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let plan = quail_planner::read_logical_plan(plan).map_err(invalid)?;
    match quail_planner::rules::column_placements(plan) {
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
    serde_json::json!({"refusal": quail::planning::refusal_json(refusal)})
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

/// Tokenize the texts of `table`'s first column into a token store at
/// `path`, with the `tokenizer.json` at `tokenizer_file`.
#[pyfunction]
fn write_token_store(path: &str, table: &Bound<'_, PyAny>, tokenizer_file: &str) -> PyResult<()> {
    let tokenizer = quail_inputs::HfTokenizer::from_file(std::path::Path::new(tokenizer_file))
        .map_err(|error| python_error(error.into()))?;
    let batches = quail::pyarrow::from_pyarrow(table)?;
    let py = table.py();
    py.detach(|| {
        let mut writer = quail_inputs::TokenStoreWriter::create(std::path::Path::new(path))?;
        for batch in &batches {
            writer.write_texts(&tokenizer, batch.column(0).as_string::<i32>())?;
        }
        writer.finish().map(|_| ())
    })
    .map_err(|error| python_error(error.into()))
}

/// Return the planner's statistics of the token store at `path`: the
/// prefix index order and parent and shared tokens of its tree at
/// `page_tokens`, the reader groups, each document's first equal
/// document (equal also in each column of `columns`, a pyarrow Table),
/// and the distinct last tokens.
#[pyfunction]
#[pyo3(signature = (path, page_tokens, columns=None))]
fn prefix_statistics<'py>(
    py: Python<'py>,
    path: &str,
    page_tokens: i64,
    columns: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let store = quail_inputs::TokenStore::open(std::path::Path::new(path))
        .map_err(|error| python_error(error.into()))?;
    let values: Vec<datafusion::arrow::array::ArrayRef> = match columns {
        None => Vec::new(),
        Some(table) => {
            let batches = quail::pyarrow::from_pyarrow(table)?;
            let schema = batches
                .first()
                .map(|batch| batch.schema())
                .ok_or_else(|| PyRuntimeError::new_err("columns has no batch"))?;
            let batch = datafusion::arrow::compute::concat_batches(&schema, &batches)
                .map_err(|error| python_error(error.into()))?;
            batch.columns().to_vec()
        }
    };
    let documents = store.documents();
    let index = quail_inputs::PrefixIndex::new(&documents);
    let tree = index.tree(page_tokens);
    let groups: Vec<(i64, i64, i64)> = tree
        .reader_groups(store.lengths())
        .iter()
        .map(|group| (group.shared, group.count, group.rows))
        .collect();
    let first = quail_inputs::first_documents(&documents, &values)
        .map_err(|error| python_error(error.into()))?;
    let found = pyo3::types::PyDict::new(py);
    found.set_item("order", &tree.order)?;
    found.set_item("parent", &tree.parent)?;
    found.set_item("shared", &tree.shared)?;
    found.set_item("reader_groups", groups)?;
    found.set_item("first_documents", first)?;
    found.set_item("last_tokens", quail_inputs::last_tokens(&documents))?;
    Ok(found)
}

/// A model's tokenizer, read from its `tokenizer.json`: the one the
/// planner writes token stores with and the GPU worker decodes with.
#[pyclass(name = "Tokenizer", module = "_quail_native", frozen)]
struct Tokenizer {
    inner: quail_inputs::HfTokenizer,
    /// The ids that end an answer: the model's turn close and its end of
    /// text, which no document holds.
    #[pyo3(get)]
    stop_ids: Vec<u32>,
}

/// Token ids and each token's character range.
type TokensAndRanges = (Vec<u32>, Vec<(i64, i64)>);

/// Several texts' token ids and character ranges, flat, then each text's
/// first token, with one past the last.
type Packed = (Vec<u32>, Vec<(i64, i64)>, Vec<i64>);

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
        let inner = quail_inputs::HfTokenizer::from_file(&path).map_err(arrow_error)?;
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

    /// A text's token ids and each token's character range.
    fn encode(&self, text: &str) -> PyResult<TokensAndRanges> {
        let found = self
            .inner
            .encode_with_offsets(&[text])
            .map_err(arrow_error)?;
        Ok((found.ids, found.offsets))
    }

    /// Each text's token ids and character ranges, encoded together.
    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<TokensAndRanges>> {
        let (ids, offsets, boundaries) = self.encode_packed(texts)?;
        Ok(boundaries
            .windows(2)
            .map(|range| {
                let (start, end) = (range[0] as usize, range[1] as usize);
                (ids[start..end].to_vec(), offsets[start..end].to_vec())
            })
            .collect())
    }

    /// Every text's token ids and character ranges, encoded together.
    fn encode_packed(&self, texts: Vec<String>) -> PyResult<Packed> {
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let found = self
            .inner
            .encode_with_offsets(&texts)
            .map_err(arrow_error)?;
        Ok((found.ids, found.offsets, found.boundaries))
    }

    /// The tokens of each text in `table`'s first column, as a pyarrow
    /// Table of one large list of int32 column, `tokens`.
    fn encode_arrow(&self, table: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        let py = table.py();
        let batches = quail::pyarrow::from_pyarrow(table)?;
        let lists = py
            .detach(|| {
                batches
                    .iter()
                    .map(|batch| {
                        let tokens: datafusion::arrow::array::ArrayRef = std::sync::Arc::new(
                            self.inner
                                .encode_batch(batch.column(0).as_string::<i32>())?,
                        );
                        Ok(tokens)
                    })
                    .collect::<Result<Vec<_>, datafusion::arrow::error::ArrowError>>()
            })
            .map_err(arrow_error)?;
        let item = Field::new_list_field(DataType::Int32, true);
        let field = Field::new(
            "tokens",
            DataType::LargeList(std::sync::Arc::new(item)),
            true,
        );
        let schema = std::sync::Arc::new(Schema::new(vec![field]));
        let batches = lists
            .into_iter()
            .map(|tokens| {
                datafusion::arrow::array::RecordBatch::try_new(
                    std::sync::Arc::clone(&schema),
                    vec![tokens],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(arrow_error)?;
        Ok(quail::pyarrow::to_pyarrow(py, schema, batches)?.unbind())
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

    /// The id of an added token, such as `<|im_end|>`, or None.
    fn added_token_id(&self, token: &str) -> Option<u32> {
        self.inner.added_token_id(token)
    }
}

/// A region's logical plan, its tables' rows, and the settings, read
/// from what Python passes the planner.
fn planning_inputs(
    logical: &str,
    tables: &Bound<'_, pyo3::types::PyDict>,
    settings: &str,
) -> PyResult<(
    quail_planner::LogicalPlan,
    std::collections::HashMap<String, quail::planning::TableRows>,
    quail::planning::Settings,
)> {
    let logical = quail_planner::read_logical_plan(logical)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let settings: quail::planning::Settings = from_json(settings)?;
    let mut rows = std::collections::HashMap::new();
    for (name, entry) in tables.iter() {
        let (identity, table): (Option<String>, Bound<'_, PyAny>) = entry.extract()?;
        let reader = quail::pyarrow::stream_from_pyarrow(&table)?;
        let schema = datafusion::arrow::array::RecordBatchReader::schema(&reader);
        let batches = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| python_error(error.into()))?;
        rows.insert(
            name.extract::<String>()?,
            quail::planning::TableRows {
                identity,
                schema,
                batches,
            },
        );
    }
    Ok((logical, rows, settings))
}

/// Plans model regions in Rust for one session, with the session's
/// tokenizer, keeping the token stores of tables read whole.
#[pyclass(name = "NativePlanner", module = "_quail_native")]
struct NativePlanner {
    planner: std::sync::Arc<quail::planning::Planner>,
}

#[pymethods]
impl NativePlanner {
    /// A planner with the `tokenizer.json` at `tokenizer_file`.
    #[new]
    fn new(tokenizer_file: &str) -> PyResult<Self> {
        Ok(Self {
            planner: std::sync::Arc::new(
                quail::planning::Planner::new(std::path::Path::new(tokenizer_file))
                    .map_err(|error| PyRuntimeError::new_err(error.0))?,
            ),
        })
    }

    /// Prepare a region for a backend other than Quail's: its column
    /// placements, its scans' files, and the planning context's fields,
    /// over `tables` as `plan` reads them.
    ///
    /// Returns (the prepared region as JSON, None) or (None, a
    /// CompileError's message).
    fn prepare(
        &self,
        py: Python<'_>,
        logical: &str,
        tables: &Bound<'_, pyo3::types::PyDict>,
        settings: &str,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let (logical, rows, settings) = planning_inputs(logical, tables, settings)?;
        let prepared = match py.detach(|| self.planner.prepare(logical, &rows, &settings)) {
            Ok(prepared) => prepared,
            Err(error) => return Ok((None, Some(error.0))),
        };
        let found = quail::planning::prepared_json(&prepared)
            .map_err(|error| PyRuntimeError::new_err(error.0))?;
        Ok((Some(found.to_string()), None))
    }

    /// Plan a region's logical plan, as JSON, over `tables`: by provider
    /// name, (content identity or None, the rows as a pyarrow Table).
    /// `settings` is the configuration as JSON.
    ///
    /// Returns (the plan as JSON, None) or (None, a CompileError's
    /// message). The plan holds the envelope, estimates, pricer, the
    /// chosen logical plan, the context fields, and each scan's files,
    /// or the refusal.
    fn plan(
        &self,
        py: Python<'_>,
        logical: &str,
        tables: &Bound<'_, pyo3::types::PyDict>,
        settings: &str,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let (logical, rows, settings) = planning_inputs(logical, tables, settings)?;
        let found = py.detach(|| self.planner.plan(logical, &rows, &settings));
        let planned = match found {
            Ok(Ok(planned)) => planned,
            Ok(Err(refused)) => {
                return Ok((
                    Some(quail::planning::refused_json(&refused).to_string()),
                    None,
                ));
            }
            Err(error) => return Ok((None, Some(error.0))),
        };
        let found = quail::planning::planned_json(&planned)
            .map_err(|error| PyRuntimeError::new_err(error.0))?;
        Ok((Some(found.to_string()), None))
    }
}

#[pymodule]
fn _quail_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSession>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<NativePlanner>()?;
    module.add_class::<cost::CostModel>()?;
    module.add_function(wrap_pyfunction!(read_and_write_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(validate_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(bind_prompt, module)?)?;
    module.add_function(wrap_pyfunction!(column_placements, module)?)?;
    module.add_function(wrap_pyfunction!(prompt_lowering, module)?)?;
    module.add_function(wrap_pyfunction!(plan_operator, module)?)?;
    module.add_function(wrap_pyfunction!(write_token_store, module)?)?;
    module.add_function(wrap_pyfunction!(prefix_statistics, module)?)?;
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

/// Read a plan's pricer from the JSON [`pricer_fields`] writes.
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
