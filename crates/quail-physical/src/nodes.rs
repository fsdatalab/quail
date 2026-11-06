//! The built-in physical nodes.
//!
//! A node in a plan's graph is written `{"type": <type name>, "id",
//! "inputs", "attributes"}`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::expressions::{Condition, Projected, ValueExpression};
use crate::graph::InputPort;
use crate::specs::{ClassifySpec, ExtractSpec, MapSpec, SaeFeaturesSpec, ScoreSpec, TokenIds};

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
            $($field:ident: $field_type:ty),* $(,)?
        }
    )*) => {
        $(
            $(#[$doc])*
            #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct $attributes {
                $(pub $field: $field_type,)*
            }
        )*

        /// A node of a plan's graph, tagged with its type name.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "type")]
        pub enum PhysicalNode {
            $(#[serde(rename = $type_name)] $variant(Node<$attributes>),)*
        }

        impl PhysicalNode {
            /// The node's id.
            pub fn id(&self) -> &str {
                match self {
                    $(PhysicalNode::$variant(node) => &node.id,)*
                }
            }

            /// The node's input ports.
            pub fn inputs(&self) -> &[InputPort] {
                match self {
                    $(PhysicalNode::$variant(node) => &node.inputs,)*
                }
            }

            /// The node's input ports, to rewire.
            pub fn inputs_mut(&mut self) -> &mut Vec<InputPort> {
                match self {
                    $(PhysicalNode::$variant(node) => &mut node.inputs,)*
                }
            }

            /// The node's registered type name.
            pub fn type_name(&self) -> &'static str {
                match self {
                    $(PhysicalNode::$variant(_) => $type_name,)*
                }
            }
        }
    };
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
        predicates: Vec<Condition>,
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
    /// A condition that asks no model, on one or more tables' rows.
    Filter(FilterAttributes) = "quail.filter" {
        predicate: Option<Condition>,
        aliases: Vec<String>,
        selectivity: Option<f64>,
        written_position: i64,
        columns: Vec<String>,
    }
    /// An `AI.CLASSIFY` call.
    AiClassify(AiClassifyAttributes) = "quail.ai_classify" {
        backend_name: String,
        model: String,
        spec: Option<ClassifySpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
    }
    /// One `AI.EXTRACT` question.
    AiExtract(AiExtractAttributes) = "quail.ai_extract" {
        backend_name: String,
        model: String,
        spec: Option<ExtractSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        attention: String,
    }
    /// An `AI.MAP` call.
    AiMap(AiMapAttributes) = "quail.ai_map" {
        backend_name: String,
        model: String,
        spec: Option<MapSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
        attention: String,
    }
    /// An `AI.SAE_FEATURES` call.
    AiSaeFeatures(AiSaeFeaturesAttributes) = "quail.ai_sae_features" {
        backend_name: String,
        model: String,
        spec: Option<SaeFeaturesSpec>,
        keep_kv: bool,
        pass_group: i64,
        document_ends: Value,
    }
    /// Prunes each table's survivors with the join answers so far.
    Barrier(BarrierAttributes) = "quail.barrier" {
        next_anchor: String,
        aliases: Vec<String>,
    }
    /// Routes a join's anchor ids to the GPUs that hold their KV.
    Exchange(ExchangeAttributes) = "quail.exchange" {
        anchor: String,
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
    /// Joins each table's surviving rows into result rows.
    Recombine(RecombineAttributes) = "quail.recombine" {
        alias_order: Vec<String>,
    }
    /// The columns a query returns.
    Project(ProjectAttributes) = "quail.project" {
        columns: Vec<String>,
        structs: Value,
        expressions: Option<Vec<Projected>>,
    }
    /// An `ORDER BY`, with `DISTINCT`, `OFFSET`, and `FETCH`.
    Sort(SortAttributes) = "quail.sort" {
        keys: Vec<(String, bool, bool)>,
        columns: Vec<String>,
        distinct: bool,
        offset: i64,
        fetch: Option<i64>,
    }
    /// A `LIMIT`.
    Limit(LimitAttributes) = "quail.limit" {
        count: i64,
    }
    /// One row per item of each listed list column.
    Unnest(UnnestAttributes) = "quail.unnest" {
        lists: Vec<String>,
        outer: bool,
        columns: Vec<String>,
    }
}
