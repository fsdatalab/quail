use quail_planner::constants as shared;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use quail_datafusion::derive::expressions::PhysicalValue;
use quail_planner::physical;
use quail_planner::topology;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

const EXPRESSIONS: &str = "quail.common.expressions";

type Field = (&'static str, Option<&'static str>);

struct Kind {
    class: &'static str,
    tag: &'static str,
    node: bool,
    fields: &'static [Field],
}

const NODE: [Field; 2] = [("node_id", None), ("inputs", Some("[]"))];
const MODEL_NODE: [Field; 6] = [
    ("backend_name", Some("\"\"")),
    ("model", Some("\"\"")),
    ("spec", Some("null")),
    ("keep_kv", Some("false")),
    ("pass_group", Some("-1")),
    ("document_ends", Some("[]")),
];
const SPEC: [Field; 9] = [
    ("name", None),
    ("aliases", None),
    ("query_template", None),
    ("arguments", None),
    ("expected_rows", None),
    ("estimated_seconds", None),
    ("prompt_token_parts", Some("[]")),
    ("share_prefixes", Some("false")),
    ("estimated_work", Some("[]")),
];
const ANSWER: [Field; 7] = [
    ("alias", None),
    ("output", None),
    ("tail_token_ids", None),
    ("tail_text", Some("\"\"")),
    ("preamble_token_ids", Some("[]")),
    ("preamble_text", Some("\"\"")),
    (
        "max_answer_tokens",
        Some(shared::EXTRACT_MAX_ANSWER_TOKENS_JSON),
    ),
];

macro_rules! fields {
    ($($prefix:expr),* ; $($name:literal = $default:expr),* $(,)?) => {{
        const FIELDS: &[Field] = &{
            const PREFIX: &[&[Field]] = &[$(&$prefix),*];
            const OWN: &[Field] = &[$(($name, $default)),*];
            const LENGTH: usize = {
                let mut total = OWN.len();
                let mut index = 0;
                while index < PREFIX.len() {
                    total += PREFIX[index].len();
                    index += 1;
                }
                total
            };
            let mut all: [Field; LENGTH] = [("", None); LENGTH];
            let mut at = 0;
            let mut index = 0;
            while index < PREFIX.len() {
                let mut inner = 0;
                while inner < PREFIX[index].len() {
                    all[at] = PREFIX[index][inner];
                    at += 1;
                    inner += 1;
                }
                index += 1;
            }
            let mut inner = 0;
            while inner < OWN.len() {
                all[at] = OWN[inner];
                at += 1;
                inner += 1;
            }
            all
        };
        FIELDS
    }};
}

const S: Option<&str> = Some("\"\"");
const ZERO: Option<&str> = Some("0");
const FLOAT: Option<&str> = Some("0.0");
const NO: Option<&str> = Some("false");
const EMPTY: Option<&str> = Some("[]");
const NONE: Option<&str> = Some("null");
const REQUIRED: Option<&str> = None;

static KINDS: &[Kind] = &[
    Kind {
        class: "Scan",
        tag: "quail.scan",
        node: true,
        fields: fields!(NODE; "alias" = S, "input_id" = S, "num_documents" = ZERO,
            "total_tokens" = ZERO, "predicates" = EMPTY, "expected_rows" = NONE),
    },
    Kind {
        class: "AiFilter",
        tag: "quail.ai_filter",
        node: true,
        fields: fields!(NODE; "alias" = S, "arena_writes" = NO, "keep_kv" = NO,
            "written_position" = ZERO, "question_tokens" = ZERO, "preamble_tokens" = ZERO,
            "selectivity" = NONE, "expected_rows" = FLOAT, "question_token_ids" = EMPTY,
            "document_head" = EMPTY, "share_prefixes" = NO, "attention" = S,
            "stop_key" = EMPTY, "pass_group" = Some("-1"), "state_tail_tokens" = ZERO,
            "question_header_token_ids" = EMPTY, "document_ends" = EMPTY,
            "num_key_survivors" = Some("1"), "precondition" = NONE, "alternative" = NONE,
            "answer" = NONE, "negated" = NO, "disjunction" = NONE,
            "short_circuit" = NO, "closes_disjunction" = NO),
    },
    Kind {
        class: "AiScore",
        tag: "quail.ai_score",
        node: true,
        fields: fields!(NODE, MODEL_NODE;),
    },
    Kind {
        class: "Filter",
        tag: "quail.filter",
        node: true,
        fields: fields!(NODE; "predicate" = NONE, "aliases" = EMPTY, "selectivity" = NONE,
            "written_position" = ZERO),
    },
    Kind {
        class: "AiClassify",
        tag: "quail.ai_classify",
        node: true,
        fields: fields!(NODE, MODEL_NODE; "precondition" = NONE),
    },
    Kind {
        class: "AiExtract",
        tag: "quail.ai_extract",
        node: true,
        fields: fields!(NODE, MODEL_NODE; "attention" = S, "precondition" = NONE,
            "stop_tests" = EMPTY, "stop_when_true" = NO),
    },
    Kind {
        class: "AiMap",
        tag: "quail.ai_map",
        node: true,
        fields: fields!(NODE, MODEL_NODE; "attention" = S, "precondition" = NONE,
            "stop_tests" = EMPTY, "stop_when_true" = NO),
    },
    Kind {
        class: "AiSaeFeatures",
        tag: "quail.ai_sae_features",
        node: true,
        fields: fields!(NODE, MODEL_NODE; "precondition" = NONE),
    },
    Kind {
        class: "Barrier",
        tag: "quail.barrier",
        node: true,
        fields: fields!(NODE; "next_anchor" = S, "aliases" = EMPTY),
    },
    Kind {
        class: "HashJoin",
        tag: "quail.hash_join",
        node: true,
        fields: fields!(NODE; "left" = S, "right" = S, "on" = EMPTY, "filter" = NONE,
            "written_position" = Some("-1"), "pair_fraction" = Some("1.0")),
    },
    Kind {
        class: "AiJoin",
        tag: "quail.ai_join",
        node: true,
        fields: fields!(NODE; "anchor" = S, "anchor_resident" = Some("\"none\""),
            "keep_kv" = NO, "written_position" = ZERO, "exec_index" = ZERO,
            "partners" = EMPTY, "semantics" = Some("\"full\""), "selectivity" = NONE,
            "expected_rows" = FLOAT, "anchor_documents" = FLOAT,
            "anchor_frame_tokens" = ZERO, "pair_tail_tokens" = ZERO, "tuple_tokens" = FLOAT,
            "pairs_from" = S, "frame_token_ids" = EMPTY, "label_token_ids" = EMPTY,
            "tail_token_ids" = EMPTY, "document_head" = EMPTY, "attention" = S,
            "share_prefixes" = NO, "separator_tokens" = ZERO,
            "question_header_token_ids" = EMPTY, "document_ends" = EMPTY),
    },
    Kind {
        class: "Limit",
        tag: "quail.limit",
        node: true,
        fields: fields!(NODE; "count" = ZERO),
    },
    Kind {
        class: "RequestExecution",
        tag: "quail.request_execution",
        node: true,
        fields: fields!(NODE; "backend_name" = S, "aliases" = EMPTY,
            "preamble_token_ids" = EMPTY, "preamble_text" = S, "filters" = EMPTY,
            "joins" = EMPTY, "classifies" = EMPTY, "extracts" = EMPTY, "maps" = EMPTY),
    },
    Kind {
        class: "PortRef",
        tag: "quail.physical.base.PortRef",
        node: false,
        fields: fields!(; "node_id" = REQUIRED, "port" = REQUIRED),
    },
    Kind {
        class: "InputPort",
        tag: "quail.physical.base.InputPort",
        node: false,
        fields: fields!(; "name" = REQUIRED, "value_type" = REQUIRED, "source" = REQUIRED,
            "schema" = EMPTY),
    },
    Kind {
        class: "OutputPort",
        tag: "quail.physical.base.OutputPort",
        node: false,
        fields: fields!(; "name" = REQUIRED, "value_type" = REQUIRED, "schema" = EMPTY,
            "answer" = NONE),
    },
    Kind {
        class: "AnswerOutput",
        tag: "quail.physical.base.AnswerOutput",
        node: false,
        fields: fields!(; "kind" = REQUIRED, "name" = REQUIRED, "columns" = REQUIRED),
    },
    Kind {
        class: "ScoreSpec",
        tag: "quail.physical.nodes.inference.ScoreSpec",
        node: false,
        fields: fields!(SPEC; "pair_fraction" = Some("1.0"), "draws" = Some("1"),
            "state_tail_tokens" = ZERO, "question_header_token_ids" = EMPTY),
    },
    Kind {
        class: "ClassifySpec",
        tag: "quail.physical.nodes.classify.ClassifySpec",
        node: false,
        fields: fields!(SPEC; "pair_fraction" = Some("1.0"), "draws" = Some("1"),
            "labels" = EMPTY, "label_token_ids" = EMPTY,
            "scoring" = Some(shared::LETTERS_SCORING_JSON), "probabilities" = NO, "join_layout" = NONE,
            "frame_tokens" = ZERO, "state_tail_tokens" = ZERO,
            "question_header_token_ids" = EMPTY, "separator_tokens" = ZERO),
    },
    Kind {
        class: "ExtractSpec",
        tag: "quail.physical.nodes.extract.ExtractSpec",
        node: false,
        fields: fields!(SPEC; "question" = S, "max_answer_tokens" = Some(shared::EXTRACT_MAX_ANSWER_TOKENS_JSON),
            "draft_tokens" = Some(shared::EXTRACT_DRAFT_TOKENS_JSON),
            "draft_policy" = Some(shared::EXTRACT_DRAFT_POLICY_JSON),
            "expected_answer_tokens" = Some(shared::EXTRACT_ANSWER_TOKENS_JSON),
            "round_cost_per_token" = FLOAT,
            "top_tokens" = Some(shared::EXTRACT_TOP_TOKENS_JSON),
            "first_window_chars" = Some(shared::FIRST_WINDOW_CHARS_JSON),
            "window_chars" = Some(shared::WINDOW_CHARS_JSON), "state_tail_tokens" = ZERO,
            "question_header_token_ids" = EMPTY, "read_slots" = EMPTY),
    },
    Kind {
        class: "MapSpec",
        tag: "quail.physical.nodes.map.MapSpec",
        node: false,
        fields: fields!(SPEC; "max_answer_tokens" = Some(shared::DEFAULT_MAP_ANSWER_TOKENS_JSON),
            "expected_answer_tokens" = Some(shared::MAP_ANSWER_TOKENS_JSON)),
    },
    Kind {
        class: "SaeFeaturesSpec",
        tag: "quail.physical.nodes.sae_features.SaeFeaturesSpec",
        node: false,
        fields: fields!(SPEC; "pooling" = Some("\"last\"")),
    },
    Kind {
        class: "RequestFilterSpec",
        tag: "quail.baselines.nodes.RequestFilterSpec",
        node: false,
        fields: fields!(; "alias" = REQUIRED, "written_positions" = REQUIRED,
            "question_token_ids" = REQUIRED, "question_texts" = EMPTY),
    },
    Kind {
        class: "RequestClassifySpec",
        tag: "quail.baselines.nodes.RequestClassifySpec",
        node: false,
        fields: fields!(; "alias" = REQUIRED, "output" = REQUIRED,
            "tail_token_ids" = REQUIRED, "labels" = REQUIRED, "label_token_ids" = REQUIRED,
            "tests" = EMPTY, "partner" = NONE, "join_layout_token_ids" = EMPTY,
            "join_written_position" = NONE, "option_offsets" = EMPTY),
    },
    Kind {
        class: "RequestJoinSpec",
        tag: "quail.baselines.nodes.RequestJoinSpec",
        node: false,
        fields: fields!(; "written_position" = REQUIRED, "aliases" = REQUIRED,
            "outer_aliases" = REQUIRED, "anchor" = REQUIRED, "semantics" = REQUIRED,
            "selectivity" = REQUIRED, "label_token_ids" = REQUIRED,
            "frame_token_ids" = REQUIRED, "tail_token_ids" = REQUIRED, "label_texts" = EMPTY,
            "frame_texts" = EMPTY, "tail_text" = S),
    },
    Kind {
        class: "RequestExtractSpec",
        tag: "quail.baselines.nodes.RequestExtractSpec",
        node: false,
        fields: fields!(ANSWER; "question" = S),
    },
    Kind {
        class: "RequestMapSpec",
        tag: "quail.baselines.nodes.RequestMapSpec",
        node: false,
        fields: fields!(ANSWER;),
    },
    Kind {
        class: "ColumnRef",
        tag: "quail.common.expressions.ColumnRef",
        node: false,
        fields: fields!(; "alias" = REQUIRED, "provider" = REQUIRED, "column" = REQUIRED),
    },
    Kind {
        class: "GetField",
        tag: "quail.common.expressions.GetField",
        node: false,
        fields: fields!(; "input" = REQUIRED, "field" = REQUIRED),
    },
    Kind {
        class: "Expression",
        tag: "quail.common.expressions.Expression",
        node: false,
        fields: fields!(; "encoded" = REQUIRED, "columns" = REQUIRED, "types" = REQUIRED,
            "text" = REQUIRED),
    },
    Kind {
        class: "Alias",
        tag: "quail.common.expressions.Alias",
        node: false,
        fields: fields!(; "expression" = REQUIRED, "name" = REQUIRED),
    },
    Kind {
        class: "Prompt",
        tag: "quail.common.expressions.Prompt",
        node: false,
        fields: fields!(; "template" = REQUIRED, "args" = REQUIRED, "preamble" = REQUIRED,
            "tail" = REQUIRED, "frame" = S, "frame_text" = S, "kind" = Some("\"filter\""),
            "layout" = Some("\"ai-if\""), "tail_segments" = EMPTY, "label_texts" = EMPTY,
            "label_prefix" = Some("\" \""), "categories" = EMPTY, "descriptions" = EMPTY,
            "task_description" = S, "turn" = Some("[\"\", \"\"]"), "source" = S,
            "letters" = EMPTY),
    },
    Kind {
        class: "ModelCall",
        tag: "quail.common.expressions.ModelCall",
        node: false,
        fields: fields!(; "prompt" = REQUIRED, "kind" = Some("\"boolean\""), "labels" = EMPTY,
            "descriptions" = EMPTY, "probabilities" = NO, "question" = S, "pooling" = S,
            "max_answer_tokens" = ZERO),
    },
    Kind {
        class: "ScoreTest",
        tag: "quail.common.expressions.ScoreTest",
        node: false,
        fields: fields!(; "call" = REQUIRED, "test" = REQUIRED),
    },
    Kind {
        class: "FilterPredicate",
        tag: "quail.common.expressions.FilterPredicate",
        node: false,
        fields: fields!(; "expression" = REQUIRED, "selectivity" = NONE, "precondition" = NONE,
            "alternative" = NONE, "answer" = NONE, "precondition_selectivity" = NONE,
            "negated" = NO, "disjunction" = NONE, "short_circuit" = NO),
    },
    Kind {
        class: "Equality",
        tag: "quail.common.expressions.Equality",
        node: false,
        fields: fields!(; "left" = REQUIRED, "right" = REQUIRED),
    },
    Kind {
        class: "ExtractStruct",
        tag: "quail.common.expressions.ExtractStruct",
        node: false,
        fields: fields!(; "name" = REQUIRED, "fields" = REQUIRED),
    },
];

fn kind_of_class(class: &str) -> &'static Kind {
    KINDS
        .iter()
        .find(|kind| kind.class == class)
        .expect("every class has a kind")
}

pub(super) fn is_known_tag(tag: &str) -> bool {
    kind_of_value(tag).is_some()
}

fn kind_of_value(tag: &str) -> Option<&'static Kind> {
    let moved = tag
        .strip_prefix("quail.logical.expressions.")
        .map(|name| format!("{EXPRESSIONS}.{name}"));
    let tag = moved.as_deref().unwrap_or(tag);
    KINDS.iter().find(|kind| kind.tag == tag)
}

enum Held {
    Node(Arc<physical::PhysicalNode>),
    Value(Value),
}

impl Held {
    fn json(&self) -> Value {
        match self {
            Held::Node(node) => serde_json::to_value(node.as_ref()).expect("a node writes"),
            Held::Value(value) => value.clone(),
        }
    }
}

fn round<T: Serialize + DeserializeOwned>(value: Value) -> Result<Value, String> {
    let typed: T = serde_json::from_value(value).map_err(|error| error.to_string())?;
    serde_json::to_value(typed).map_err(|error| error.to_string())
}

fn canonical(kind: &Kind, value: Value) -> Result<Value, String> {
    use physical::*;
    let name = kind.class;
    match name {
        "PortRef" => round::<PortRef>(value),
        "InputPort" => round::<InputPort>(value),
        "OutputPort" => round::<OutputPort>(value),
        "AnswerOutput" => round::<AnswerOutput>(value),
        "ScoreSpec" => round::<ScoreSpec>(value),
        "ClassifySpec" => round::<ClassifySpec>(value),
        "ExtractSpec" => round::<ExtractSpec>(value),
        "MapSpec" => round::<MapSpec>(value),
        "SaeFeaturesSpec" => round::<SaeFeaturesSpec>(value),
        "RequestFilterSpec" => round::<RequestFilterSpec>(value),
        "RequestClassifySpec" => round::<RequestClassifySpec>(value),
        "RequestJoinSpec" => round::<RequestJoinSpec>(value),
        "RequestExtractSpec" => round::<RequestExtractSpec>(value),
        "RequestMapSpec" => round::<RequestMapSpec>(value),
        "Prompt" => round::<Prompt>(value),
        "FilterPredicate" => round::<FilterPredicate>(value),
        "Equality" => round::<Equality>(value),
        "ExtractStruct" => round::<ExtractStruct>(value),
        "ScoreTest" => round::<Asked>(value),
        _ => round::<ValueExpression>(value),
    }
}

static VALUE_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static LOCATION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ENUM: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn python_enum<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    name: &str,
    members: &[(&str, &str)],
) -> PyResult<&'py Bound<'py, PyAny>> {
    let found = cell.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let module = py.import("enum")?;
        let keywords = PyDict::new(py);
        keywords.set_item("type", py.get_type::<PyString>())?;
        keywords.set_item("module", "quail.physical")?;
        Ok(module
            .getattr("Enum")?
            .call((name, members.to_vec()), Some(&keywords))?
            .unbind())
    })?;
    Ok(found.bind(py))
}

fn value_type(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    python_enum(
        py,
        &VALUE_TYPE,
        "ValueType",
        &[
            ("FILTER_ANSWERS", "filter_answers"),
            ("JOIN_ANSWERS", "join_answers"),
            ("LABEL_ANSWERS", "label_answers"),
            ("EXTRACT_ANSWERS", "extract_answers"),
            ("PAIRS", "pairs"),
            ("ROWS", "rows"),
            ("ANY", "any"),
        ],
    )
}

fn location(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    python_enum(
        py,
        &LOCATION,
        "ExecutionLocation",
        &[
            ("COORDINATOR", "coordinator"),
            ("GPU_EXECUTOR", "gpu_executor"),
        ],
    )
}

pub fn to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                integer.into_pyobject(py)?.into_any()
            } else if let Some(integer) = number.as_u64() {
                integer.into_pyobject(py)?.into_any()
            } else {
                number
                    .as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
            }
        }
        Value::String(text) => PyString::new(py, text).into_any(),
        Value::Array(items) => PyTuple::new(
            py,
            items
                .iter()
                .map(|item| to_python(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Value::Object(map) => {
            let tag = map.get("type").and_then(Value::as_str);
            match tag.and_then(kind_of_value) {
                Some(kind) if kind.node => {
                    let node: physical::PhysicalNode = serde_json::from_value(value.clone())
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    node_object(py, Arc::new(node))?
                }
                Some(kind) => {
                    let value = canonical(kind, value.clone()).map_err(PyValueError::new_err)?;
                    create(py, kind.class, Held::Value(value))?
                }
                None => {
                    let dict = PyDict::new(py);
                    for (key, item) in map {
                        dict.set_item(key, to_python(py, item)?)?;
                    }
                    dict.into_any()
                }
            }
        }
    })
}

pub fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = value.py();
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(tagged) = value.cast::<Tagged>() {
        return Ok(tagged.get().held.json());
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(integer) = value.extract::<i64>() {
            return Ok(Value::from(integer));
        }
        return Ok(Value::from(value.extract::<u64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        let number: f64 = value.extract()?;
        return serde_json::Number::from_f64(number)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err(format!("{number} is not a JSON number")));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Value::String(value.extract()?));
    }
    let enum_type = ENUM.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("enum")?.getattr("Enum")?.unbind())
    })?;
    if value.is_instance(enum_type.bind(py))? {
        return from_python(&value.getattr("value")?);
    }
    if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        return value
            .try_iter()?
            .map(|item| from_python(&item?))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, item) in dict.iter() {
            map.insert(key.extract::<String>()?, from_python(&item)?);
        }
        return Ok(Value::Object(map));
    }
    if value.hasattr("dtype")? && value.hasattr("item")? {
        return from_python(&value.call_method0("item")?);
    }
    Err(PyTypeError::new_err(format!(
        "cannot encode a {} value",
        value.get_type().name()?
    )))
}

#[pyclass(name = "Tagged", module = "quail.physical", subclass, dict, frozen)]
pub struct Tagged {
    held: Held,
    filled: AtomicBool,
    physical: Option<PhysicalValue>,
}

impl Tagged {
    fn new(held: Held) -> PyResult<Self> {
        let physical =
            match &held {
                Held::Value(value) => physical_value(value)?,
                Held::Node(node) => {
                    if let physical::PhysicalNode::AiScore(score) = node.as_ref()
                        && score.attributes.spec.as_ref().is_some_and(|spec| {
                            spec.state_tail_tokens != 0 && spec.aliases.is_empty()
                        })
                    {
                        return Err(PyValueError::new_err(
                            "an AiScore with state tail tokens requires an alias",
                        ));
                    }
                    None
                }
            };
        Ok(Self {
            held,
            filled: AtomicBool::new(false),
            physical,
        })
    }

    pub(super) fn held_node(object: &Bound<'_, PyAny>) -> PyResult<Arc<physical::PhysicalNode>> {
        match &object.cast::<Tagged>()?.get().held {
            Held::Node(node) => Ok(Arc::clone(node)),
            Held::Value(_) => Err(PyTypeError::new_err(
                "a physical graph holds physical nodes",
            )),
        }
    }

    pub(super) fn held_value<T: DeserializeOwned>(object: &Bound<'_, PyAny>) -> PyResult<T> {
        serde_json::from_value(object.cast::<Tagged>()?.get().held.json())
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn field_order(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        Ok(match kind_of_object(slf.as_any())? {
            Some(kind) => kind
                .fields
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
            None => constructor_fields(&slf.get().held)
                .keys()
                .cloned()
                .collect(),
        })
    }

    fn fill(slf: &Bound<'_, Self>) -> PyResult<()> {
        let tagged = slf.get();
        if tagged.filled.load(Ordering::Acquire) {
            return Ok(());
        }
        let py = slf.py();
        let dict = slf.getattr("__dict__")?.cast_into::<PyDict>()?;
        match &tagged.held {
            Held::Node(node) => Self::fill_node(py, &dict, node)?,
            Held::Value(value) => Self::fill_value(py, &dict, value)?,
        }
        tagged.filled.store(true, Ordering::Release);
        Ok(())
    }

    fn fill_node(
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        node: &physical::PhysicalNode,
    ) -> PyResult<()> {
        let Value::Object(mut map) = serde_json::to_value(node).expect("a node writes") else {
            unreachable!("a node writes an object")
        };
        dict.set_item("node_id", node.id())?;
        dict.set_item(
            "inputs",
            to_python(py, map.get("inputs").unwrap_or(&Value::Null))?,
        )?;
        if let Some(Value::Object(attributes)) = map.remove("attributes") {
            for (name, value) in &attributes {
                dict.set_item(name, to_python(py, value)?)?;
            }
        }
        let ports = node
            .output_ports()
            .iter()
            .map(|port| to_python(py, &serde_json::to_value(port).expect("a port writes")))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("outputs", PyTuple::new(py, ports)?)?;
        dict.set_item(
            "location",
            location(py)?.call1((match node.location() {
                physical::Location::Coordinator => "coordinator",
                physical::Location::GpuExecutor => "gpu_executor",
            },))?,
        )?;
        dict.set_item("backend", node.backend())?;
        dict.set_item("document_aliases", tuple_of(py, &node.document_aliases())?)?;
        dict.set_item("document_head", PyTuple::new(py, node.document_head())?)?;
        match node {
            physical::PhysicalNode::Filter(filter) => {
                dict.set_item("rows_port", node.rows_port())?;
                let columns = filter
                    .attributes
                    .predicate
                    .as_ref()
                    .map(quail_planner::expressions::condition_columns)
                    .unwrap_or_default();
                dict.set_item("tested_columns", tuple_of(py, &columns)?)?;
            }
            physical::PhysicalNode::AiScore(_)
            | physical::PhysicalNode::AiClassify(_)
            | physical::PhysicalNode::AiExtract(_)
            | physical::PhysicalNode::AiMap(_)
            | physical::PhysicalNode::AiSaeFeatures(_) => {
                dict.set_item("rows_port", node.rows_port())?;
                let shares = dict
                    .get_item("spec")?
                    .filter(|spec| !spec.is_none())
                    .map(|spec| spec.getattr("share_prefixes"))
                    .transpose()?;
                dict.set_item(
                    "share_prefixes",
                    shares.map_or(Ok(false), |found| found.extract())?,
                )?;
            }
            physical::PhysicalNode::Extension(extension) => {
                dict.set_item("type_name", &extension.type_name)?;
                dict.set_item("runtime_key", &extension.type_name)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn fill_value(py: Python<'_>, dict: &Bound<'_, PyDict>, value: &Value) -> PyResult<()> {
        let Value::Object(map) = value else {
            return Ok(());
        };
        let tag = map.get("type").and_then(Value::as_str).unwrap_or_default();
        for (name, item) in map {
            if name == "type" {
                continue;
            }
            let converted = if name == "value_type" {
                value_type(py)?.call1((to_python(py, item)?,))?
            } else {
                to_python(py, item)?
            };
            dict.set_item(name, converted)?;
        }
        let text = |name: &str| map.get(name).and_then(Value::as_str).unwrap_or_default();
        let aliases: Vec<String> = map
            .get("aliases")
            .and_then(Value::as_array)
            .map(|aliases| {
                aliases
                    .iter()
                    .filter_map(|alias| alias.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let class = kind_of_value(tag)
            .map(|kind| kind.class)
            .unwrap_or_default();
        if class.ends_with("Spec") && !class.starts_with("Request") {
            let mut columns = aliases.clone();
            columns.push(text("name").to_string());
            if class == "ExtractSpec" {
                let span = format!("{}_span", text("name"));
                dict.set_item("alias", aliases.first())?;
                dict.set_item("span_name", &span)?;
                columns.push(span);
            }
            if class == "ClassifySpec" {
                dict.set_item("anchor", aliases.first())?;
                dict.set_item("partner", (aliases.len() == 2).then(|| aliases[1].clone()))?;
            }
            dict.set_item("output_columns", tuple_of(py, &columns)?)?;
        }
        match class {
            "RequestExtractSpec" => {
                dict.set_item("span_output", format!("{}_span", text("output")))?;
            }
            "ExtractStruct" => {
                let members: Vec<String> = map
                    .get("fields")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|field| format!("{}.{}", text("name"), field.as_str().unwrap_or_default()))
                    .collect();
                dict.set_item("members", tuple_of(py, &members)?)?;
            }
            "FilterPredicate" => {
                let predicate: physical::FilterPredicate = serde_json::from_value(value.clone())
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                let call = quail_planner::expressions::asked_call(&predicate.expression);
                let prompt = serde_json::to_value(call.prompt.as_ref()).expect("a prompt writes");
                dict.set_item("prompt", to_python(py, &prompt)?)?;
            }
            "ColumnRef" | "GetField" | "Alias" | "Expression" => {
                let expression: physical::ValueExpression =
                    serde_json::from_value(value.clone())
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                let references: Vec<physical::ValueExpression> =
                    quail_planner::expressions::column_references(&expression)
                        .into_iter()
                        .map(physical::ValueExpression::ColumnRef)
                        .collect();
                let references = serde_json::to_value(references).expect("columns write");
                dict.set_item("references", to_python(py, &references)?)?;
            }
            _ => {}
        }
        Ok(())
    }
}

const EVALUATED: [&str; 4] = ["ColumnRef", "GetField", "Alias", "Expression"];

fn asks_model(value: &physical::ValueExpression) -> bool {
    match value {
        physical::ValueExpression::ModelCall(_) => true,
        physical::ValueExpression::Alias(alias) => asks_model(&alias.expression),
        physical::ValueExpression::GetField(field) => asks_model(&field.input),
        physical::ValueExpression::ColumnRef(_) | physical::ValueExpression::Value(_) => false,
    }
}

fn expression_name(expression: &physical::ValueExpression) -> PyResult<String> {
    match expression {
        physical::ValueExpression::ModelCall(_) => {
            Err(PyValueError::new_err("a model call has no column name"))
        }
        physical::ValueExpression::GetField(field) => Ok(format!(
            "{}.{}",
            expression_name(&field.input)?,
            field.field
        )),
        _ => Ok(quail_planner::expressions::column_name(expression, true)),
    }
}

fn physical_value(value: &Value) -> PyResult<Option<PhysicalValue>> {
    let class = value
        .get("type")
        .and_then(Value::as_str)
        .and_then(kind_of_value)
        .map(|kind| kind.class);
    if !class.is_some_and(|class| EVALUATED.contains(&class)) {
        return Ok(None);
    }
    let expression: physical::ValueExpression = serde_json::from_value(value.clone())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if asks_model(&expression) {
        return Ok(None);
    }
    PhysicalValue::new(&expression)
        .map(Some)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn compile_error(py: Python<'_>, message: String) -> PyErr {
    match py
        .import(EXPRESSIONS)
        .and_then(|module| module.getattr("CompileError"))
    {
        Ok(class) => match class.cast_into() {
            Ok(class) => PyErr::from_type(class, message),
            Err(error) => PyTypeError::new_err(format!("an exception type: {error}")),
        },
        Err(error) => error,
    }
}

fn evaluate_rows<'py>(
    object: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    qualified: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchReader};
    let py = object.py();
    let Some(built) = &object.cast::<Tagged>()?.get().physical else {
        return Err(compile_error(
            py,
            "a model call is the model's to answer, not an expression to evaluate".into(),
        ));
    };
    let rows = if rows.hasattr("__arrow_c_stream__")? {
        rows.clone()
    } else {
        let columns = PyDict::new(py);
        for column in built.columns() {
            let name = column.name(qualified);
            columns.set_item(&name, rows.call_method1("column", (&name,))?)?;
        }
        py.import("pyarrow")?.call_method1("table", (columns,))?
    };
    let reader = crate::pyarrow::stream_from_pyarrow(&rows)?;
    let schema = reader.schema();
    let mut batches = reader
        .collect::<Result<Vec<RecordBatch>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
    }
    let found = py.detach(|| -> Result<ArrayRef, quail_planner::error::CompileError> {
        let arrays = batches
            .iter()
            .map(|batch| built.evaluate(batch, qualified))
            .collect::<Result<Vec<_>, _>>()?;
        if arrays.len() == 1 {
            return Ok(arrays.into_iter().next().expect("one array"));
        }
        let parts: Vec<&dyn Array> = arrays.iter().map(AsRef::as_ref).collect();
        datafusion::arrow::compute::concat(&parts)
            .map_err(|error| quail_planner::error::CompileError::new(error.to_string()))
    });
    match found {
        Ok(array) => crate::pyarrow::array_to_pyarrow(py, &array),
        Err(error) => Err(compile_error(py, error.to_string())),
    }
}

fn kind_of_object(object: &Bound<'_, PyAny>) -> PyResult<Option<&'static Kind>> {
    let class: String = object.get_type().name()?.extract()?;
    Ok(KINDS.iter().find(|kind| kind.class == class))
}

fn constructor_fields(held: &Held) -> Map<String, Value> {
    let value = held.json();
    let Value::Object(mut map) = value else {
        return Map::new();
    };
    map.remove("type");
    if let Held::Node(_) = held {
        let mut fields = Map::new();
        fields.insert("node_id".into(), map.remove("id").unwrap_or_default());
        fields.insert("inputs".into(), map.remove("inputs").unwrap_or_default());
        if let Some(Value::Object(attributes)) = map.remove("attributes") {
            fields.extend(attributes);
        }
        fields.extend(map);
        return fields;
    }
    map
}

fn tuple_of<'py, T: AsRef<str>>(py: Python<'py>, items: &[T]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, items.iter().map(AsRef::as_ref))
}

fn given_fields(
    kind: &Kind,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Map<String, Value>> {
    let mut given = Map::new();
    if args.len() > kind.fields.len() {
        return Err(PyTypeError::new_err(format!(
            "{}() takes {} positional arguments but {} were given",
            kind.class,
            kind.fields.len(),
            args.len()
        )));
    }
    for ((name, _), value) in kind.fields.iter().zip(args.iter()) {
        given.insert((*name).to_string(), from_python(&value)?);
    }
    for (name, value) in kwargs.into_iter().flat_map(|kwargs| kwargs.iter()) {
        let name: String = name.extract()?;
        if given.contains_key(&name) {
            return Err(PyTypeError::new_err(format!(
                "{}() got multiple values for argument {name:?}",
                kind.class
            )));
        }
        given.insert(name, from_python(&value)?);
    }
    Ok(given)
}

fn build(kind: &Kind, mut given: Map<String, Value>) -> PyResult<Held> {
    let mut fields = Map::new();
    for (name, default) in kind.fields {
        let value = match (given.remove(*name), default) {
            (Some(value), _) => value,
            (None, Some(default)) => serde_json::from_str(default).expect("a default reads"),
            (None, None) => {
                return Err(PyTypeError::new_err(format!(
                    "{}() missing required argument {name:?}",
                    kind.class
                )));
            }
        };
        fields.insert((*name).to_string(), value);
    }
    if let Some(name) = given.keys().next() {
        return Err(PyTypeError::new_err(format!(
            "{}() got an unexpected keyword argument {name:?}",
            kind.class
        )));
    }
    if !kind.node {
        let mut value = Map::new();
        value.insert("type".into(), Value::from(kind.tag));
        value.extend(fields);
        return canonical(kind, Value::Object(value))
            .map(Held::Value)
            .map_err(|error| PyValueError::new_err(format!("{}: {error}", kind.class)));
    }
    let mut node = Map::new();
    node.insert("type".into(), Value::from(kind.tag));
    node.insert("id".into(), fields.remove("node_id").unwrap_or_default());
    node.insert("inputs".into(), fields.remove("inputs").unwrap_or_default());
    node.insert("attributes".into(), Value::Object(fields));
    serde_json::from_value(Value::Object(node))
        .map(|node| Held::Node(Arc::new(node)))
        .map_err(|error| PyValueError::new_err(format!("{}: {error}", kind.class)))
}

fn build_extension(mut given: Map<String, Value>) -> PyResult<Held> {
    let mut take = |name: &str, default: Value| given.remove(name).unwrap_or(default);
    let type_name = take("type_name", Value::Null);
    let id = take("node_id", Value::Null);
    let inputs = take("inputs", Value::Array(Vec::new()));
    let outputs = take("outputs", Value::Array(Vec::new()));
    let backend = take("backend", Value::Null);
    let place = take("location", Value::from("coordinator"));
    if let Some(tag) = type_name.as_str()
        && physical::NODE_TYPES.contains(&tag)
    {
        return Err(PyValueError::new_err(format!(
            "{tag:?} is one of Quail's node types, not an extension's"
        )));
    }
    let node = serde_json::json!({
        "type": type_name, "id": id, "inputs": inputs, "attributes": given,
        "outputs": outputs, "backend": backend, "location": place,
    });
    serde_json::from_value(node)
        .map(|node| Held::Node(Arc::new(node)))
        .map_err(|error| PyValueError::new_err(format!("ExtensionNode: {error}")))
}

#[pymethods]
impl Tagged {
    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let filled = slf.get().filled.load(Ordering::Acquire);
        if !filled && !name.starts_with("__") {
            Self::fill(slf)?;
            if let Some(found) = slf
                .getattr("__dict__")?
                .cast_into::<PyDict>()?
                .get_item(name)?
            {
                return Ok(found.unbind());
            }
        }
        Err(PyAttributeError::new_err(format!(
            "{:?} object has no attribute {name:?}",
            slf.get_type().name()?.to_string()
        )))
    }

    fn __setattr__(&self, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "cannot assign to field {name:?}: the value is frozen"
        )))
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "cannot delete field {name:?}: the value is frozen"
        )))
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let Ok(other) = other.cast::<Tagged>() else {
            return Ok(py.NotImplemented());
        };
        if !slf.get_type().is(other.get_type()) {
            return Ok(false.into_pyobject(py)?.to_owned().into_any().unbind());
        }
        let same = match (&slf.get().held, &other.get().held) {
            (Held::Node(one), Held::Node(two)) => one == two,
            (one, two) => one.json() == two.json(),
        };
        Ok(same.into_pyobject(py)?.to_owned().into_any().unbind())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.held.json().to_string().hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::fill(slf)?;
        let dict = slf.getattr("__dict__")?.cast_into::<PyDict>()?;
        let mut parts = Vec::new();
        for name in Self::field_order(slf)? {
            if let Some(value) = dict.get_item(&name)? {
                parts.push(format!("{name}={}", value.repr()?));
            }
        }
        Ok(format!("{}({})", slf.get_type().name()?, parts.join(", ")))
    }

    #[pyo3(signature = (**changes))]
    fn replace<'py>(
        slf: &Bound<'py, Self>,
        changes: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let mut fields = constructor_fields(&slf.get().held);
        for (name, value) in changes.into_iter().flat_map(|changes| changes.iter()) {
            fields.insert(name.extract()?, from_python(&value)?);
        }
        match kind_of_object(slf.as_any())? {
            Some(kind) => {
                let held = build(kind, fields)?;
                match held {
                    Held::Node(node) => node_object(py, node),
                    held => create(py, kind.class, held),
                }
            }
            None => {
                let Held::Node(node) = &slf.get().held else {
                    unreachable!("only an extension node has no kind")
                };
                let physical::PhysicalNode::Extension(extension) = node.as_ref() else {
                    unreachable!("only an extension node has no kind")
                };
                fields.insert("type_name".into(), Value::from(extension.type_name.clone()));
                if let Some(Value::Array(ports)) = fields.get("outputs") {
                    let ports = ports.clone();
                    fields.insert("outputs".into(), Value::Array(ports));
                }
                match build_extension(fields)? {
                    Held::Node(node) => node_object(py, node),
                    Held::Value(_) => unreachable!("an extension is a node"),
                }
            }
        }
    }

    fn to_json(&self) -> String {
        self.held.json().to_string()
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let value = slf.get().held.json();
        let class = kind_of_object(slf.as_any())?
            .map(|kind| kind.class)
            .unwrap_or_default();
        let text = match class {
            "ColumnRef" | "GetField" | "Expression" => serde_json::from_value(value)
                .ok()
                .map(|expression| expression_name(&expression))
                .transpose()?,
            "Equality" => serde_json::from_value::<physical::Equality>(value)
                .ok()
                .map(|pair| {
                    Ok::<_, PyErr>(format!(
                        "{} = {}",
                        expression_name(&pair.left)?,
                        expression_name(&pair.right)?
                    ))
                })
                .transpose()?,
            _ => None,
        };
        match text {
            Some(text) => Ok(text),
            None => Self::__repr__(slf),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let decode = py.import("_quail_native")?.getattr("decode_plan_value")?;
        let text = slf.get().held.json().to_string();
        PyTuple::new(py, [decode, PyTuple::new(py, [text])?.into_any()])
    }
}

#[pyclass(name = "PhysicalNode", module = "quail.physical", extends = Tagged, subclass, frozen)]
pub struct NodeBase;

#[pyclass(name = "AiInference", module = "quail.physical", extends = NodeBase, subclass, frozen)]
pub struct AiInference;

#[pyclass(name = "InferenceSpec", module = "quail.physical", extends = Tagged, subclass, frozen)]
pub struct InferenceSpec;

#[pyclass(name = "RequestAnswerSpec", module = "quail.physical", extends = Tagged, subclass, frozen)]
pub struct RequestAnswerSpec;

fn args_constructor(
    class: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Held> {
    let kind = kind_of_class(class);
    build(kind, given_fields(kind, args, kwargs)?)
}

macro_rules! leaf_classes {
    ($($name:ident = $class:literal : $base:ident [$($parent:ident),*] { $($methods:tt)* }),* $(,)?) => {
        $(
            #[pyclass(name = $class, module = "quail.physical", extends = $base, frozen)]
            pub struct $name;

            #[pymethods]
            impl $name {
                #[new]
                #[pyo3(signature = (*args, **kwargs))]
                fn new(
                    args: &Bound<'_, PyTuple>,
                    kwargs: Option<&Bound<'_, PyDict>>,
                ) -> PyResult<PyClassInitializer<Self>> {
                    Self::initializer(args_constructor($class, args, kwargs)?)
                }

                #[classattr]
                fn type_name() -> &'static str {
                    kind_of_class($class).tag
                }

                #[classattr]
                fn runtime_key() -> &'static str {
                    kind_of_class($class).tag
                }

                $($methods)*
            }

            impl $name {
                fn initializer(held: Held) -> PyResult<PyClassInitializer<Self>> {
                    Ok(PyClassInitializer::from(Tagged::new(held)?)
                        $(.add_subclass($parent))*
                        .add_subclass($name))
                }
            }
        )*

        fn create_class<'py>(py: Python<'py>, class: &str, held: Held) -> PyResult<Bound<'py, PyAny>> {
            match class {
                $($class => Ok(Bound::new(py, $name::initializer(held)?)?.into_any()),)*
                _ => Err(PyTypeError::new_err(format!("no class {class:?}"))),
            }
        }

        fn add_leaf_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_class::<$name>()?;)*
            Ok(())
        }
    };
}

leaf_classes! {
    Scan = "Scan": NodeBase [NodeBase] {},
    AiFilter = "AiFilter": NodeBase [NodeBase] {},
    Filter = "Filter": NodeBase [NodeBase] {},
    Barrier = "Barrier": NodeBase [NodeBase] {},
    HashJoin = "HashJoin": NodeBase [NodeBase] {},
    AiJoin = "AiJoin": NodeBase [NodeBase] {
    fn keeps<'py>(slf: &Bound<'py, Self>, matched: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let node = Tagged::held_node(slf.as_any())?;
        let physical::PhysicalNode::AiJoin(join) = node.as_ref() else {
            unreachable!("an AiJoin holds a join")
        };
        matched.rich_compare(
            join.attributes.semantics == "anti",
            pyo3::basic::CompareOp::Ne,
        )
    }
    },
    Limit = "Limit": NodeBase [NodeBase] {},
    RequestExecution = "RequestExecution": NodeBase [NodeBase] {},
    AiScore = "AiScore": AiInference [NodeBase, AiInference] {},
    AiClassify = "AiClassify": AiInference [NodeBase, AiInference] {
    fn classifies_join(slf: &Bound<'_, Self>, join: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(Tagged::held_node(slf.as_any())?.classifies_join(Tagged::held_node(join)?.as_ref()))
    }
    },
    AiExtract = "AiExtract": AiInference [NodeBase, AiInference] {},
    AiMap = "AiMap": AiInference [NodeBase, AiInference] {},
    AiSaeFeatures = "AiSaeFeatures": AiInference [NodeBase, AiInference] {},
    ScoreSpec = "ScoreSpec": InferenceSpec [InferenceSpec] {},
    ClassifySpec = "ClassifySpec": InferenceSpec [InferenceSpec] {},
    ExtractSpec = "ExtractSpec": InferenceSpec [InferenceSpec] {
    fn same_decoding(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let one: physical::ExtractSpec = Tagged::held_value(slf.as_any())?;
        let two: physical::ExtractSpec = Tagged::held_value(other)?;
        Ok(topology::same_decoding(&one, &two))
    }
    },
    MapSpec = "MapSpec": InferenceSpec [InferenceSpec] {},
    SaeFeaturesSpec = "SaeFeaturesSpec": InferenceSpec [InferenceSpec] {},
    RequestFilterSpec = "RequestFilterSpec": Tagged [] {},
    RequestClassifySpec = "RequestClassifySpec": Tagged [] {},
    RequestJoinSpec = "RequestJoinSpec": Tagged [] {},
    RequestExtractSpec = "RequestExtractSpec": RequestAnswerSpec [RequestAnswerSpec] {},
    RequestMapSpec = "RequestMapSpec": RequestAnswerSpec [RequestAnswerSpec] {},
    PortRef = "PortRef": Tagged [] {},
    InputPort = "InputPort": Tagged [] {},
    OutputPort = "OutputPort": Tagged [] {},
    AnswerOutput = "AnswerOutput": Tagged [] {},
    ColumnRef = "ColumnRef": Tagged [] {
    #[pyo3(signature = (rows, qualified=false))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        rows: &Bound<'py, PyAny>,
        qualified: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        evaluate_rows(slf.as_any(), rows, qualified)
    }
    },
    GetField = "GetField": Tagged [] {
    #[pyo3(signature = (rows, qualified=false))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        rows: &Bound<'py, PyAny>,
        qualified: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        evaluate_rows(slf.as_any(), rows, qualified)
    }
    },
    Expression = "Expression": Tagged [] {
    #[pyo3(signature = (rows, qualified=false))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        rows: &Bound<'py, PyAny>,
        qualified: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        evaluate_rows(slf.as_any(), rows, qualified)
    }

    #[pyo3(signature = (answer, prefix, complete=false, may_be_null=false, tails=None, max_chars=None))]
    #[allow(clippy::too_many_arguments)]
    fn decide(
        slf: &Bound<'_, Self>,
        answer: &str,
        prefix: &str,
        complete: bool,
        may_be_null: bool,
        tails: Option<Vec<String>>,
        max_chars: Option<usize>,
    ) -> PyResult<&'static str> {
        use quail_datafusion::derive::partial::{Completions, Decision, decide};
        let physical::ValueExpression::Value(expression) = Tagged::held_value(slf.as_any())? else {
            return Err(compile_error(slf.py(), "an Expression holds an expression".into()));
        };
        let answers = Completions {
            prefix: prefix.to_string(),
            complete,
            may_be_null,
            tails,
            max_chars,
        };
        let decided = decide(&expression, answer, &answers)
            .map_err(|error| compile_error(slf.py(), error.to_string()))?;
        Ok(match decided {
            Decision::True => "true",
            Decision::NotTrue => "not true",
            Decision::Undecided => "undecided",
        })
    }
    },
    Alias = "Alias": Tagged [] {
    #[pyo3(signature = (rows, qualified=false))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        rows: &Bound<'py, PyAny>,
        qualified: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        evaluate_rows(slf.as_any(), rows, qualified)
    }
    },
    Prompt = "Prompt": Tagged [] {},
    ModelCall = "ModelCall": Tagged [] {
    fn aliases<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let call: physical::ValueExpression = Tagged::held_value(slf.as_any())?;
        let physical::ValueExpression::ModelCall(call) = call else {
            unreachable!("a ModelCall holds a model call")
        };
        let mut found: Vec<String> = Vec::new();
        for argument in &call.prompt.args {
            if let physical::ValueExpression::ColumnRef(column) = argument
                && !found.contains(&column.alias)
            {
                found.push(column.alias.clone());
            }
        }
        PyTuple::new(slf.py(), found)
    }
    },
    ScoreTest = "ScoreTest": Tagged [] {},
    FilterPredicate = "FilterPredicate": Tagged [] {},
    Equality = "Equality": Tagged [] {},
    ExtractStruct = "ExtractStruct": Tagged [] {},
}

#[pyclass(name = "ExtensionNode", module = "quail.physical", extends = NodeBase, frozen)]
pub struct ExtensionNode;

#[pymethods]
impl ExtensionNode {
    #[new]
    #[pyo3(signature = (type_name, node_id, inputs=None, outputs=None, backend=None, location=None, **attributes))]
    fn new(
        type_name: &str,
        node_id: &str,
        inputs: Option<&Bound<'_, PyAny>>,
        outputs: Option<&Bound<'_, PyAny>>,
        backend: Option<&str>,
        location: Option<&Bound<'_, PyAny>>,
        attributes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut fields = Map::new();
        for (name, value) in attributes
            .into_iter()
            .flat_map(|attributes| attributes.iter())
        {
            fields.insert(name.extract()?, from_python(&value)?);
        }
        fields.insert("type_name".into(), Value::from(type_name));
        fields.insert("node_id".into(), Value::from(node_id));
        if let Some(inputs) = inputs {
            fields.insert("inputs".into(), from_python(inputs)?);
        }
        if let Some(outputs) = outputs {
            fields.insert("outputs".into(), from_python(outputs)?);
        }
        fields.insert("backend".into(), backend.map_or(Value::Null, Value::from));
        if let Some(location) = location {
            fields.insert("location".into(), from_python(location)?);
        }
        Self::initializer(build_extension(fields)?)
    }
}

impl ExtensionNode {
    fn initializer(held: Held) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(Tagged::new(held)?)
            .add_subclass(NodeBase)
            .add_subclass(ExtensionNode))
    }
}

fn create<'py>(py: Python<'py>, class: &str, held: Held) -> PyResult<Bound<'py, PyAny>> {
    create_class(py, class, held)
}

pub(super) fn node_object(
    py: Python<'_>,
    node: Arc<physical::PhysicalNode>,
) -> PyResult<Bound<'_, PyAny>> {
    if let physical::PhysicalNode::Extension(_) = node.as_ref() {
        return Ok(Bound::new(py, ExtensionNode::initializer(Held::Node(node))?)?.into_any());
    }
    let tag = node.type_name().to_string();
    let kind = kind_of_value(&tag).expect("a known node has a kind");
    create(py, kind.class, Held::Node(node))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Tagged>()?;
    module.add_class::<NodeBase>()?;
    module.add_class::<AiInference>()?;
    module.add_class::<InferenceSpec>()?;
    module.add_class::<RequestAnswerSpec>()?;
    module.add_class::<ExtensionNode>()?;
    add_leaf_classes(module)?;
    module.add("ValueType", value_type(py)?)?;
    module.add("ExecutionLocation", location(py)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_tagged_values_return_errors() {
        let port = kind_of_class("PortRef");
        assert!(canonical(port, serde_json::json!({"type": port.tag})).is_err());
        assert!(
            canonical(
                port,
                serde_json::json!({
                    "type": port.tag, "node_id": [], "port": "rows:r"
                })
            )
            .is_err()
        );
        let expression = kind_of_class("Expression");
        assert!(
            canonical(
                expression,
                serde_json::json!({
                    "type": expression.tag, "encoded": 42, "columns": [], "types": [], "text": ""
                })
            )
            .is_err()
        );
    }
}
