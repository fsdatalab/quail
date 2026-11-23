//! Quail's physical plan types, which the GPU executor runs.
//!
//! A plan crosses from the planner to the GPU executor as a JSON
//! envelope. These types read and write that envelope, so the Rust
//! planner can produce plans the Python executor runs.
//!
//! The types are the built-in nodes, each written `{"type": <type name>,
//! "id", "inputs", "attributes"}`; the ports that join them; the specs an
//! AI node carries, its prompt as token ids and its plan estimates; and
//! the expressions nodes carry, from `quail.common.expressions`. A value
//! or condition the executor computes is an [`Expression`], which
//! DataFusion evaluates. Each struct is written with a `"type"` key naming
//! its Python class, and also accepts the tag the class had before it
//! moved from `quail.logical.expressions`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Define a struct that reads and writes `{"type": <tag>, ...fields}`.
///
/// serde's own struct tag cannot be combined with
/// `deny_unknown_fields`, which then refuses the tag key. The struct
/// instead goes through a one-variant tagged enum: reading refuses
/// another tag, a missing field, and an unknown field. Further literals after the tag are older tags that are
/// also accepted.
#[macro_export]
macro_rules! tagged_struct {
    (
        $(#[$doc:meta])*
        $name:ident($body:ident, $wrapper:ident) = $tag:literal $(| $old:literal)* {
            $($(#[$field_doc:meta])* $field:ident: $field_type:ty $(= $absent:ident)?),* $(,)?
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            $($(#[$field_doc])* pub $field: $field_type,)*
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct $body {
            $($(#[serde($absent)])? $field: $field_type,)*
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(tag = "type")]
        enum $wrapper {
            #[serde(rename = $tag $(, alias = $old)*)]
            Value($body),
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&$wrapper::Value($body {
                    $($field: self.$field.clone(),)*
                }), serializer)

            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let $wrapper::Value(body) = <$wrapper as serde::Deserialize>::deserialize(deserializer)?;
                Ok($name {
                    $($field: body.$field,)*
                })
            }
        }
    };
}

/// A value computed from one row's columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ValueExpression {
    #[serde(
        rename = "quail.common.expressions.ColumnRef",
        alias = "quail.logical.expressions.ColumnRef"
    )]
    ColumnRef(ColumnRef),
    #[serde(
        rename = "quail.common.expressions.GetField",
        alias = "quail.logical.expressions.GetField"
    )]
    GetField(GetField),
    #[serde(rename = "quail.common.expressions.Expression")]
    Value(Expression),
    #[serde(
        rename = "quail.common.expressions.Alias",
        alias = "quail.logical.expressions.Alias"
    )]
    Alias(Alias),
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    ModelCall(ModelCall),
}

/// One column of one table in the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnRef {
    pub alias: String,
    pub provider: String,
    pub column: String,
}

impl ColumnRef {
    /// The column's name, with its table alias when `qualified` and it
    /// has one.
    pub fn name(&self, qualified: bool) -> String {
        if qualified && !self.alias.is_empty() {
            format!("{}.{}", self.alias, self.column)
        } else {
            self.column.clone()
        }
    }
}

/// One field of a struct value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GetField {
    pub input: Box<ValueExpression>,
    pub field: String,
}

/// A value or condition DataFusion computes from one row's columns.
///
/// `encoded` is DataFusion's logical expression as datafusion-proto
/// bytes in base64. It reads the column named `#i` where the rows hold
/// `columns[i]`, so a planner can rename the columns it reads without
/// DataFusion. `types` holds the Arrow type of each column as Arrow
/// writes it, such as `Int64` or `Utf8`. `text` is how `explain()`
/// writes the expression, with `{i}` for the name of `columns[i]` and
/// `{{` and `}}` for braces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expression {
    pub encoded: String,
    pub columns: Vec<ColumnRef>,
    /// Empty in an expression saved before types were written.
    #[serde(default)]
    pub types: Vec<String>,
    pub text: String,
}

impl Expression {
    /// The expression's text, each column named with its table alias
    /// when `qualified`.
    pub fn text_with(&self, qualified: bool) -> String {
        let mut out = String::new();
        let mut chars = self.text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '{' if chars.peek() == Some(&'{') => {
                    chars.next();
                    out.push('{');
                }
                '}' if chars.peek() == Some(&'}') => {
                    chars.next();
                    out.push('}');
                }
                '{' => {
                    let mut digits = String::new();
                    for digit in chars.by_ref() {
                        if digit == '}' {
                            break;
                        }
                        digits.push(digit);
                    }
                    match digits
                        .parse::<usize>()
                        .ok()
                        .and_then(|at| self.columns.get(at))
                    {
                        Some(column) => out.push_str(&column.name(qualified)),
                        None => out.push_str(&format!("{{{digits}}}")),
                    }
                }
                other => out.push(other),
            }
        }
        out
    }
}

/// A value under the name a query gives it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alias {
    pub expression: Box<ValueExpression>,
    pub name: String,
}

tagged_struct! {
    /// The text of one model call, laid out for a model's prompt format.
    Prompt(PromptFields, PromptTagged) = "quail.common.expressions.Prompt" | "quail.logical.expressions.Prompt" {
        template: String,
        /// Column references in placeholder order.
        args: Vec<ValueExpression>,
        preamble: String,
        tail: String,
        frame: String,
        frame_text: String,
        kind: String,
        layout: String,
        tail_segments: Vec<String>,
        /// Per placeholder: alias, its label text, its frame text.
        label_texts: Vec<(String, String, String)>,
        label_prefix: String,
        categories: Vec<String>,
        descriptions: Vec<String>,
        task_description: String,
        turn: (String, String),
        source: String,
        letters: Vec<String>,
    }
}

/// One prompt asked of every row; the model answers with a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCall {
    pub prompt: Box<Prompt>,
    pub kind: String,
    pub labels: Value,
    pub descriptions: Value,
    pub probabilities: bool,
    pub question: String,
    pub pooling: String,
    pub max_answer_tokens: i64,
}

/// An `AI.SCORE` call compared with a constant. `test` is an
/// `Expression` that reads the score as its one column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreTest {
    pub call: ModelCall,
    pub test: ValueExpression,
}

/// A value expression or a column name, as a projection lists them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Projected {
    Name(String),
    Expression(ValueExpression),
}

/// A model call, written with its class tag where a field holds one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TaggedModelCall {
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    Call(ModelCall),
}

/// What a semantic filter or join asks: a yes-or-no model call, or a
/// comparison of an `AI.SCORE` call with a constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Asked {
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    Call(ModelCall),
    #[serde(rename = "quail.common.expressions.ScoreTest")]
    Compare(ScoreTest),
}

tagged_struct! {
    /// One `AI.IF` predicate of a semantic filter.
    FilterPredicate(FilterPredicateFields, FilterPredicateTagged) = "quail.common.expressions.FilterPredicate" | "quail.logical.expressions.FilterPredicate" {
        expression: Asked,
        /// The fraction of documents that pass, if the query gives one.
        selectivity: Option<f64>,
        /// The condition that says which documents the model is asked
        /// about; None asks about every document.
        guard: Option<ValueExpression>,
        /// The terms of an OR the predicate is one term of that ask no
        /// model: a document where it is TRUE passes whatever the model
        /// answers, and the model is not asked about it. None for a
        /// predicate that is not in an OR, or whose OR has no such term.
        alternative: Option<ValueExpression>,
        /// The column the predicate adds with its answer, when it keeps
        /// every row it reads instead of dropping the rows it rejects:
        /// TRUE or FALSE for a document it asks about, NULL for one its
        /// guard leaves out. None for a predicate that drops rows.
        answer: Option<String>,
        /// The estimated fraction of documents where the guard is TRUE;
        /// None without a guard, or for the default selectivity.
        guard_selectivity: Option<f64>,
        /// Whether the predicate is `NOT AI.IF(...)`: a document passes
        /// when the model answers no, and `selectivity` is the fraction
        /// of documents that do.
        negated: bool = default,
        /// The name of the OR the predicate is one term of, when the OR
        /// has other `AI.IF` terms: a document passes the OR when any
        /// term passes it, in whatever order the terms run.
        disjunction: Option<String>,
        /// Whether the predicate asks the model only about the documents
        /// no earlier term of its OR passed.
        short_circuit: bool = default,
    }
}

tagged_struct! {
    /// An equality between a value of one table and a value of another.
    Equality(EqualityFields, EqualityTagged) = "quail.common.expressions.Equality" | "quail.logical.expressions.Equality" {
        left: ValueExpression,
        right: ValueExpression,
    }
}

tagged_struct! {
    /// The struct column of an `AI.EXTRACT` call and its fields.
    ExtractStruct(ExtractStructFields, ExtractStructTagged) = "quail.common.expressions.ExtractStruct" | "quail.logical.expressions.ExtractStruct" {
        name: String,
        fields: Vec<String>,
    }
}

/// What a port carries between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    FilterAnswers,
    JoinAnswers,
    LabelAnswers,
    ExtractAnswers,
    Pairs,
    Rows,
    Any,
}

tagged_struct! {
    /// One output port of another node.
    PortRef(PortRefFields, PortRefTagged) = "quail.physical.base.PortRef" {
        node_id: String,
        port: String,
    }
}

tagged_struct! {
    /// One input of a node: its name, what it carries, and where from.
    InputPort(InputPortFields, InputPortTagged) = "quail.physical.base.InputPort" {
        name: String,
        value_type: ValueType,
        source: PortRef,
        schema: Vec<String>,
    }
}

tagged_struct! {
    /// A model call's answer columns, as a node's output reports them
    /// before later row operations.
    AnswerOutput(AnswerOutputFields, AnswerOutputTagged) = "quail.physical.base.AnswerOutput" {
        kind: String,
        name: String,
        columns: Vec<String>,
    }
}

tagged_struct! {
    /// One output of a node: its name, what it carries, its columns, and
    /// the model answer it holds, if any.
    OutputPort(OutputPortFields, OutputPortTagged) = "quail.physical.base.OutputPort" {
        name: String,
        value_type: ValueType,
        schema: Vec<String>,
        answer: Option<AnswerOutput>,
    }
}

/// Token ids, as the tokenizer returns them.
pub type TokenIds = Vec<i64>;

tagged_struct! {
    /// An `AI.SCORE` call.
    ScoreSpec(ScoreSpecFields, ScoreSpecTagged) = "quail.physical.nodes.inference.ScoreSpec" {
        name: String,
        aliases: Vec<String>,
        query_template: String,
        arguments: Vec<(String, String)>,
        expected_rows: f64,
        estimated_seconds: f64,
        /// The work the seconds price, as `Work::values` lists it.
        estimated_work: Vec<f64> = default,
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        pair_fraction: f64,
        draws: i64,
        state_tail_tokens: i64,
        question_header_token_ids: TokenIds,
    }
}

tagged_struct! {
    /// An `AI.CLASSIFY` call.
    ClassifySpec(ClassifySpecFields, ClassifySpecTagged) = "quail.physical.nodes.classify.ClassifySpec" {
        name: String,
        aliases: Vec<String>,
        query_template: String,
        arguments: Vec<(String, String)>,
        expected_rows: f64,
        estimated_seconds: f64,
        /// The work the seconds price, as `Work::values` lists it.
        estimated_work: Vec<f64> = default,
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        pair_fraction: f64,
        draws: i64,
        labels: Vec<String>,
        label_token_ids: Vec<TokenIds>,
        scoring: String,
        probabilities: bool,
        /// A joined classification's anchor and partner layouts.
        join_layout: Option<(Vec<Value>, Vec<Value>)>,
        frame_tokens: i64,
        state_tail_tokens: i64,
        question_header_token_ids: TokenIds,
        separator_tokens: i64,
    }
}

tagged_struct! {
    /// One `AI.EXTRACT` question.
    ExtractSpec(ExtractSpecFields, ExtractSpecTagged) = "quail.physical.nodes.extract.ExtractSpec" {
        name: String,
        aliases: Vec<String>,
        query_template: String,
        arguments: Vec<(String, String)>,
        expected_rows: f64,
        estimated_seconds: f64,
        /// The work the seconds price, as `Work::values` lists it.
        estimated_work: Vec<f64> = default,
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        question: String,
        max_answer_tokens: i64,
        draft_tokens: i64,
        draft_policy: String,
        expected_answer_tokens: i64,
        round_cost_per_token: f64,
        top_tokens: i64,
        first_window_chars: i64,
        window_chars: i64,
        state_tail_tokens: i64,
        question_header_token_ids: TokenIds,
        read_slots: Vec<i64>,
    }
}

tagged_struct! {
    /// An `AI.MAP` call.
    MapSpec(MapSpecFields, MapSpecTagged) = "quail.physical.nodes.map.MapSpec" {
        name: String,
        aliases: Vec<String>,
        query_template: String,
        arguments: Vec<(String, String)>,
        expected_rows: f64,
        estimated_seconds: f64,
        /// The work the seconds price, as `Work::values` lists it.
        estimated_work: Vec<f64> = default,
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        max_answer_tokens: i64,
        expected_answer_tokens: i64,
    }
}

tagged_struct! {
    /// An `AI.SAE_FEATURES` call.
    SaeFeaturesSpec(SaeFeaturesSpecFields, SaeFeaturesSpecTagged) = "quail.physical.nodes.sae_features.SaeFeaturesSpec" {
        name: String,
        aliases: Vec<String>,
        query_template: String,
        arguments: Vec<(String, String)>,
        expected_rows: f64,
        estimated_seconds: f64,
        /// The work the seconds price, as `Work::values` lists it.
        estimated_work: Vec<f64> = default,
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        pooling: String,
    }
}

tagged_struct! {
    /// One filter sequence a request backend submits as model requests.
    RequestFilterSpec(RequestFilterSpecFields, RequestFilterSpecTagged) = "quail.baselines.nodes.RequestFilterSpec" {
        alias: String,
        written_positions: Vec<i64>,
        question_token_ids: Vec<TokenIds>,
        question_texts: Vec<String>,
    }
}

tagged_struct! {
    /// One decoded classification answer per document or joined row.
    ///
    /// The tail follows the document: the question, the label list, and
    /// the answer cue. `tests` are (written position, accepted labels)
    /// pairs applied in order after classification. A classification of
    /// joined rows names its partner, the anchor note and partner label
    /// token ids, and the join whose rows it labels. `option_offsets`
    /// are a decision model's scored options: distances before the
    /// prompt's last row of each option block's end, then 0.
    RequestClassifySpec(RequestClassifySpecFields, RequestClassifySpecTagged) = "quail.baselines.nodes.RequestClassifySpec" {
        alias: String,
        output: String,
        tail_token_ids: TokenIds,
        labels: Vec<String>,
        label_token_ids: Vec<TokenIds>,
        tests: Vec<(i64, Value)>,
        partner: Option<String>,
        join_layout_token_ids: Vec<TokenIds>,
        join_written_position: Option<i64>,
        option_offsets: Vec<i64>,
    }
}

tagged_struct! {
    /// One join predicate submitted over its complete cross product.
    RequestJoinSpec(RequestJoinSpecFields, RequestJoinSpecTagged) = "quail.baselines.nodes.RequestJoinSpec" {
        written_position: i64,
        aliases: Vec<String>,
        outer_aliases: Vec<String>,
        anchor: Option<String>,
        semantics: String,
        selectivity: Option<f64>,
        label_token_ids: Vec<(String, TokenIds)>,
        frame_token_ids: Vec<(String, TokenIds)>,
        tail_token_ids: TokenIds,
        label_texts: Vec<(String, String)>,
        frame_texts: Vec<(String, String)>,
        tail_text: String,
    }
}

tagged_struct! {
    /// An answer copied from the document, located in its text.
    ///
    /// The preamble precedes the document: the model's turn and the
    /// document label. The tail holds the question, the instruction, and
    /// the phrase cue.
    RequestExtractSpec(RequestExtractSpecFields, RequestExtractSpecTagged) = "quail.baselines.nodes.RequestExtractSpec" {
        alias: String,
        output: String,
        tail_token_ids: TokenIds,
        tail_text: String,
        preamble_token_ids: TokenIds,
        preamble_text: String,
        max_answer_tokens: i64,
        question: String,
    }
}

tagged_struct! {
    /// An answer the model writes for an instruction, up to its turn end.
    RequestMapSpec(RequestMapSpecFields, RequestMapSpecTagged) = "quail.baselines.nodes.RequestMapSpec" {
        alias: String,
        output: String,
        tail_token_ids: TokenIds,
        tail_text: String,
        preamble_token_ids: TokenIds,
        preamble_text: String,
        max_answer_tokens: i64,
    }
}

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
        /// Whether a document passes when the model answers no.
        #[serde(default)]
        negated: bool,
        /// The column that holds whether a term of the filter's OR has
        /// passed each row, which the filter writes.
        #[serde(default)]
        disjunction: Option<String>,
        /// Whether the filter asks the model only about the rows where
        /// that column is not TRUE.
        #[serde(default)]
        short_circuit: bool,
        /// Whether the filter is the last term of its OR to run: it
        /// returns the rows a term passed, without the column.
        #[serde(default)]
        closes_disjunction: bool,
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
    /// The executor stops an answer once a stop test is decided not TRUE
    /// on the tokens decoded so far, or, with `stop_when_true`, once
    /// every stop test is decided TRUE. Stop tests are the region's exact
    /// tests that read the answer alone.
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
    }
    /// An `AI.MAP` call.
    ///
    /// The executor stops an answer once a stop test is decided not TRUE
    /// on the tokens decoded so far, or, with `stop_when_true`, once
    /// every stop test is decided TRUE. Stop tests are the region's exact
    /// tests that read the answer alone.
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
    /// The pairs of two tables an AI join asks about: those whose `on`
    /// keys are equal, or every pair without keys, where `filter` is
    /// TRUE. It asks no model.
    HashJoin(HashJoinAttributes) = "quail.hash_join" {
        left: String,
        right: String,
        on: Vec<(ValueExpression, ValueExpression)>,
        #[serde(default)]
        filter: Option<ValueExpression>,
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

/// A physical graph: its nodes, and the port that returns the result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalGraph {
    pub nodes: Vec<PhysicalNode>,
    pub root: PortRef,
}

/// The physical plan as it crosses a process boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanEnvelope {
    pub backend: String,
    pub model: String,
    pub device: String,
    pub gpus: i64,
    /// Settings the planner chose for the run, keyed by name.
    pub settings: Map<String, Value>,
    pub graph: PhysicalGraph,
}

/// Read a plan envelope from JSON.
pub fn read_plan(text: &str) -> serde_json::Result<PlanEnvelope> {
    serde_json::from_str(text)
}

/// Write a plan envelope as JSON.
pub fn write_plan(plan: &PlanEnvelope) -> serde_json::Result<String> {
    serde_json::to_string(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_schema_payloads_return_errors() {
        use crate::operators::Schema;
        for (ipc, message) in [
            ("not base64!", "invalid schema base64"),
            ("", "invalid IPC schema"),
            ("AAAAAA==", "invalid IPC schema"),
        ] {
            let schema = Schema { ipc: ipc.into() };
            assert!(schema.names().unwrap_err().message().starts_with(message));
        }
    }

    #[test]
    fn a_tagged_struct_reads_and_writes_its_tag() {
        let text = r#"{"type":"quail.physical.base.PortRef","node_id":"scan:r","port":"rows:r"}"#;
        let port: PortRef = serde_json::from_str(text).unwrap();
        assert_eq!(port.node_id, "scan:r");
        assert_eq!(serde_json::to_string(&port).unwrap(), text);
    }

    #[test]
    fn a_struct_with_another_class_tag_is_refused() {
        let text = r#"{"type":"quail.physical.base.InputPort","node_id":"a","port":"b"}"#;
        assert!(serde_json::from_str::<PortRef>(text).is_err());
    }

    #[test]
    fn an_expression_saved_before_the_move_reads() {
        let text = r#"{"type":"quail.logical.expressions.ColumnRef","alias":"r","provider":"reviews","column":"id"}"#;
        let expression: ValueExpression = serde_json::from_str(text).unwrap();
        assert!(matches!(expression, ValueExpression::ColumnRef(_)));
    }

    #[test]
    fn a_plan_saved_before_guards_and_key_survivors_reads_with_their_defaults() {
        let filter = PhysicalNode::AiFilter(Node {
            id: "ai_filter:r".into(),
            inputs: Vec::new(),
            attributes: AiFilterAttributes {
                alias: "r".into(),
                arena_writes: false,
                keep_kv: false,
                written_position: 0,
                question_tokens: 4,
                preamble_tokens: 2,
                selectivity: None,
                expected_rows: 6.0,
                question_token_ids: Vec::new(),
                document_head: Vec::new(),
                share_prefixes: false,
                attention: String::new(),
                stop_key: Vec::new(),
                pass_group: -1,
                state_tail_tokens: 0,
                question_header_token_ids: Vec::new(),
                document_ends: serde_json::Value::Array(Vec::new()),
                num_key_survivors: 1,
                guard: None,
                alternative: None,
                answer: None,
                negated: false,
                disjunction: None,
                short_circuit: false,
                closes_disjunction: false,
            },
        });
        let map = PhysicalNode::AiMap(Node {
            id: "ai-map:0".into(),
            inputs: Vec::new(),
            attributes: AiMapAttributes {
                backend_name: "quail".into(),
                model: String::new(),
                spec: None,
                keep_kv: false,
                pass_group: -1,
                document_ends: serde_json::Value::Array(Vec::new()),
                attention: String::new(),
                guard: None,
                stop_tests: Vec::new(),
                stop_when_true: false,
            },
        });
        let added = [
            "num_key_survivors",
            "guard",
            "alternative",
            "answer",
            "negated",
            "disjunction",
            "short_circuit",
            "closes_disjunction",
            "stop_tests",
            "stop_when_true",
        ];
        for node in [filter, map] {
            let mut saved = serde_json::to_value(&node).unwrap();
            let attributes = saved["attributes"].as_object_mut().unwrap();
            for name in added {
                attributes.remove(name);
            }
            let read: PhysicalNode = serde_json::from_value(saved).unwrap();
            assert_eq!(read, node);
        }
        let predicate = r#"{"type":"quail.common.expressions.FilterPredicate","expression":
            {"type":"quail.common.expressions.ModelCall","prompt":{"type":
            "quail.common.expressions.Prompt","template":"t","args":[],"preamble":"",
            "tail":"","frame":"","frame_text":"","kind":"boolean","layout":"",
            "tail_segments":[],"label_texts":[],"label_prefix":"","categories":[],
            "descriptions":[],"task_description":"","turn":["",""],"source":"",
            "letters":[]},"kind":"boolean","labels":null,"descriptions":null,
            "probabilities":false,"question":"","pooling":"","max_answer_tokens":0},
            "selectivity":null}"#;
        let predicate: FilterPredicate = serde_json::from_str(predicate).unwrap();
        assert_eq!(
            (predicate.guard, predicate.alternative, predicate.answer),
            (None, None, None)
        );
        assert_eq!(predicate.guard_selectivity, None);
        let text = r#"{"type":"quail.common.expressions.Expression","encoded":"",
            "columns":[],"text":"TRUE"}"#;
        let ValueExpression::Value(expression) = serde_json::from_str(text).unwrap() else {
            panic!("an expression");
        };
        assert!(expression.types.is_empty());
    }

    #[test]
    fn an_expression_names_its_columns_in_its_text() {
        let column = |alias: &str, name: &str| ColumnRef {
            alias: alias.into(),
            provider: String::new(),
            column: name.into(),
        };
        let expression = Expression {
            encoded: String::new(),
            columns: vec![column("r", "stars"), column("", "label")],
            types: vec!["Int64".into(), "Utf8".into()],
            text: "{0} > 3 AND {1} = '{{x}}'".into(),
        };
        assert_eq!(expression.text_with(true), "r.stars > 3 AND label = '{x}'");
        assert_eq!(expression.text_with(false), "stars > 3 AND label = '{x}'");
    }
}
