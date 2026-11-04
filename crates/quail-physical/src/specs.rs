//! The specs an AI node carries: its prompt as token ids, and its plan
//! estimates.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tagged::tagged_struct;

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
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        pair_fraction: f64,
        draws: i64,
        labels: Vec<String>,
        label_token_ids: Vec<TokenIds>,
        label_word_starts: Vec<Vec<bool>>,
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
        prompt_token_parts: Vec<TokenIds>,
        share_prefixes: bool,
        pooling: String,
    }
}
