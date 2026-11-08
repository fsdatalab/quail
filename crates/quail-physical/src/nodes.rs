//! The built-in physical nodes.
//!
//! A node in a plan's graph is written `{"type": <type name>, "id",
//! "inputs", "attributes"}`; an extension's node also writes its
//! `outputs`, `backend`, and `location`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::expressions::ValueExpression;
use crate::graph::{InputPort, OutputPort};
use crate::specs::{
    ClassifySpec, ExtractSpec, MapSpec, RequestClassifySpec, RequestExtractSpec, RequestFilterSpec,
    RequestJoinSpec, RequestMapSpec, SaeFeaturesSpec, ScoreSpec, TokenIds,
};

/// A node's id, inputs, and type-specific attributes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node<A> {
    pub id: String,
    pub inputs: Vec<InputPort>,
    pub attributes: A,
}

macro_rules! physical_nodes {
    ($(
        $(#[$doc:meta])*
        $variant:ident($attributes:ident) = $type_name:literal {
            $($(#[$field_attr:meta])* $field:ident: $field_type:ty),* $(,)?
        }
    )*) => {
        $(
            $(#[$doc])*
            #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct $attributes {
                $($(#[$field_attr])* pub $field: $field_type,)*
            }
        )*

        /// A node of a plan's graph, tagged with its type name: one of
        /// Quail's, or an extension's.
        #[derive(Debug, Clone, PartialEq)]
        pub enum PhysicalNode {
            $($variant(Node<$attributes>),)*
            /// A node of a type an extension registers.
            Extension(ExtensionNode),
        }

        /// Quail's own node types, read and written with their tag.
        #[derive(Serialize)]
        #[serde(tag = "type")]
        enum KnownRef<'a> {
            $(#[serde(rename = $type_name)] $variant(&'a Node<$attributes>),)*
        }

        #[derive(Deserialize)]
        #[serde(tag = "type")]
        enum Known {
            $(#[serde(rename = $type_name)] $variant(Node<$attributes>),)*
        }

        /// The type names of Quail's own nodes.
        pub const NODE_TYPES: &[&str] = &[$($type_name,)*];

        impl Serialize for PhysicalNode {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    $(PhysicalNode::$variant(node) => KnownRef::$variant(node).serialize(serializer),)*
                    PhysicalNode::Extension(node) => node.serialize(serializer),
                }
            }
        }

        impl<'de> Deserialize<'de> for PhysicalNode {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                use serde::de::Error;
                let value = Value::deserialize(deserializer)?;
                let known = value
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|tag| NODE_TYPES.contains(&tag));
                if !known {
                    return serde_json::from_value(value)
                        .map(PhysicalNode::Extension)
                        .map_err(D::Error::custom);
                }
                Ok(match serde_json::from_value(value).map_err(D::Error::custom)? {
                    $(Known::$variant(node) => PhysicalNode::$variant(node),)*
                })
            }
        }

        impl PhysicalNode {
            /// The node's id.
            pub fn id(&self) -> &str {
                match self {
                    $(PhysicalNode::$variant(node) => &node.id,)*
                    PhysicalNode::Extension(node) => &node.id,
                }
            }

            /// The node's input ports.
            pub fn inputs(&self) -> &[InputPort] {
                match self {
                    $(PhysicalNode::$variant(node) => &node.inputs,)*
                    PhysicalNode::Extension(node) => &node.inputs,
                }
            }

            /// The node's input ports, to rewire.
            pub fn inputs_mut(&mut self) -> &mut Vec<InputPort> {
                match self {
                    $(PhysicalNode::$variant(node) => &mut node.inputs,)*
                    PhysicalNode::Extension(node) => &mut node.inputs,
                }
            }

            /// The node's registered type name.
            pub fn type_name(&self) -> &str {
                match self {
                    $(PhysicalNode::$variant(_) => $type_name,)*
                    PhysicalNode::Extension(node) => &node.type_name,
                }
            }
        }
    };
}

/// Where a node runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    Coordinator,
    GpuExecutor,
}

fn coordinator() -> Location {
    Location::Coordinator
}

fn one() -> i64 {
    1
}

/// A node of a type Quail does not define: an extension's.
///
/// It carries what Quail cannot derive from its attributes: its outputs,
/// the backend of a model node, and where it runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtensionNode {
    #[serde(rename = "type")]
    pub type_name: String,
    pub id: String,
    pub inputs: Vec<InputPort>,
    pub attributes: Map<String, Value>,
    pub outputs: Vec<OutputPort>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default = "coordinator")]
    pub location: Location,
}

physical_nodes! {
    /// Reads one table's documents, after its regular predicates.
    Scan(ScanAttributes) = "quail.scan" {
        alias: String,
        input_id: String,
        num_documents: i64,
        total_tokens: i64,
        shard_ranges: Vec<(i64, i64)>,
        shard_token_loads: Vec<i64>,
        predicates: Vec<ValueExpression>,
        expected_rows: Option<f64>,
    }
    /// An `AI.IF` over one table.
    AiFilter(AiFilterAttributes) = "quail.ai_filter" {
        alias: String,
        arena_writes: bool,
        keep_kv: bool,
        written_position: i64,
        question_tokens: i64,
        preamble_tokens: i64,
        selectivity: Option<f64>,
        expected_rows: f64,
        question_token_ids: TokenIds,
        document_head: TokenIds,
        share_prefixes: bool,
        attention: String,
        stop_key: Vec<String>,
        pass_group: i64,
        state_tail_tokens: i64,
        question_header_token_ids: TokenIds,
        document_ends: Value,
        // plans saved before these fields read with the behavior they had
        #[serde(default = "one")]
        num_key_survivors: i64,
        #[serde(default)]
        guard: Option<ValueExpression>,
        #[serde(default)]
        alternative: Option<ValueExpression>,
        #[serde(default)]
        answer: Option<String>,
    }
    /// An `AI.SCORE` call.
    AiScore(AiScoreAttributes) = "quail.ai_score" {
        backend_name: String,
        model: String,
        spec: Option<ScoreSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
    }
    /// A condition that asks no model, on one or more tables' documents.
    Filter(FilterAttributes) = "quail.filter" {
        predicate: Option<ValueExpression>,
        aliases: Vec<String>,
        selectivity: Option<f64>,
        written_position: i64,
    }
    /// An `AI.CLASSIFY` call.
    AiClassify(AiClassifyAttributes) = "quail.ai_classify" {
        backend_name: String,
        model: String,
        spec: Option<ClassifySpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        #[serde(default)]
        guard: Option<ValueExpression>,
    }
    /// One `AI.EXTRACT` question.
    ///
    /// The executor stops an answer once a stop test or a hint test is
    /// decided not TRUE on the tokens decoded so far, or, with
    /// `stop_when_true`, once every stop test is decided TRUE. Stop tests
    /// are the region's exact tests that read the answer alone; hint
    /// tests are tests the node may drop rows by.
    AiExtract(AiExtractAttributes) = "quail.ai_extract" {
        backend_name: String,
        model: String,
        spec: Option<ExtractSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        attention: String,
        #[serde(default)]
        guard: Option<ValueExpression>,
        #[serde(default)]
        stop_tests: Vec<ValueExpression>,
        #[serde(default)]
        stop_when_true: bool,
        #[serde(default)]
        hint_tests: Vec<ValueExpression>,
    }
    /// An `AI.MAP` call.
    ///
    /// The executor stops an answer once a stop test or a hint test is
    /// decided not TRUE on the tokens decoded so far, or, with
    /// `stop_when_true`, once every stop test is decided TRUE. Stop tests
    /// are the region's exact tests that read the answer alone; hint
    /// tests are tests the node may drop rows by.
    AiMap(AiMapAttributes) = "quail.ai_map" {
        backend_name: String,
        model: String,
        spec: Option<MapSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        attention: String,
        #[serde(default)]
        guard: Option<ValueExpression>,
        #[serde(default)]
        stop_tests: Vec<ValueExpression>,
        #[serde(default)]
        stop_when_true: bool,
        #[serde(default)]
        hint_tests: Vec<ValueExpression>,
    }
    /// An `AI.SAE_FEATURES` call.
    AiSaeFeatures(AiSaeFeaturesAttributes) = "quail.ai_sae_features" {
        backend_name: String,
        model: String,
        spec: Option<SaeFeaturesSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        #[serde(default)]
        guard: Option<ValueExpression>,
    }
    /// Prunes each table's survivors with the join answers so far.
    Barrier(BarrierAttributes) = "quail.barrier" {
        next_anchor: String,
        aliases: Vec<String>,
    }
    /// An equality join that asks no model.
    HashJoin(HashJoinAttributes) = "quail.hash_join" {
        left: String,
        right: String,
        on: Vec<(ValueExpression, ValueExpression)>,
        written_position: i64,
        pair_fraction: f64,
    }
    /// An `AI.IF` over the pairs of an anchor table and its partners.
    AiJoin(AiJoinAttributes) = "quail.ai_join" {
        anchor: String,
        anchor_resident: String,
        keep_kv: bool,
        written_position: i64,
        exec_index: i64,
        partners: Vec<String>,
        semantics: String,
        selectivity: Option<f64>,
        expected_rows: f64,
        anchor_documents: f64,
        anchor_frame_tokens: i64,
        pair_tail_tokens: i64,
        tuple_tokens: f64,
        pairs_from: String,
        frame_token_ids: TokenIds,
        label_token_ids: Vec<(String, TokenIds)>,
        tail_token_ids: TokenIds,
        document_head: TokenIds,
        attention: String,
        share_prefixes: bool,
        separator_tokens: i64,
        question_header_token_ids: TokenIds,
        document_ends: Value,
    }
    /// The first rows a region returns: its fetch.
    Limit(LimitAttributes) = "quail.limit" {
        count: i64,
    }
    /// A request backend's whole model work: each document or pair
    /// submitted to an engine such as vLLM.
    RequestExecution(RequestExecutionAttributes) = "quail.request_execution" {
        backend_name: String,
        aliases: Vec<String>,
        preamble_token_ids: TokenIds,
        preamble_text: String,
        filters: Vec<RequestFilterSpec>,
        joins: Vec<RequestJoinSpec>,
        classifies: Vec<RequestClassifySpec>,
        extracts: Vec<RequestExtractSpec>,
        maps: Vec<RequestMapSpec>,
    }
}
