//! The specs an AI node carries: its prompt as token ids, and its plan
//! estimates.

use serde_json::Value;

use crate::tagged_struct;

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
