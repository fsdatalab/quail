//! The names and numbers the planner, the DataFusion extension, and the
//! Python executor share. Each is defined here and nowhere else; Python
//! reads them from `_quail_native`.

/// A number, and its text as the JSON default of a Python-built spec.
macro_rules! number {
    ($(#[$doc:meta])* $name:ident, $json:ident: $kind:ty = $value:literal) => {
        $(#[$doc])*
        pub const $name: $kind = $value;
        pub const $json: &str = stringify!($value);
    };
}

/// A name, and its text as the JSON default of a Python-built spec.
macro_rules! name {
    ($(#[$doc:meta])* $name:ident, $json:ident = $value:literal) => {
        $(#[$doc])*
        pub const $name: &str = $value;
        pub const $json: &str = concat!("\"", $value, "\"");
    };
}

// ---- column names

/// The suffix of the column that holds an extraction's answer positions.
pub const SPAN_SUFFIX: &str = "_span";
/// The suffix of the column that holds a classification's probabilities.
pub const PROBABILITIES_SUFFIX: &str = "_probabilities";
/// The value column that is TRUE for each NULL document of a scan.
pub const NULL_DOCUMENT: &str = "__null_document";

// ---- classification

/// The text before each label of a classification prompt.
pub const LABEL_PREFIX: &str = " ";
name!(LETTERS_SCORING, LETTERS_SCORING_JSON = "letters");
pub const TREE_SCORING: &str = "trie_tree";
pub const DECODE_SCORING: &str = "trie_decode";

// ---- estimates

/// The planner's selectivity when the query gives none.
pub const DEFAULT_SELECTIVITY: f64 = 0.2;
number!(
    /// The answer length `ExtractAssumptions` takes by default.
    EXTRACT_ANSWER_TOKENS, EXTRACT_ANSWER_TOKENS_JSON: i64 = 16
);
number!(
    /// AI.MAP's guess at an answer's mean length; an answer limit below caps it.
    MAP_ANSWER_TOKENS, MAP_ANSWER_TOKENS_JSON: i64 = 64
);
/// The rows the planner samples for an estimate: the documents it
/// prices a larger table from, and the rows it joins to estimate how
/// many pairs a join's equalities keep.
pub const SAMPLE_ROWS: usize = 1024;

// ---- extraction

number!(
    /// The most answer tokens an extraction writes.
    EXTRACT_MAX_ANSWER_TOKENS, EXTRACT_MAX_ANSWER_TOKENS_JSON: i64 = 64
);
number!(
    /// The tokens an extraction drafts in one round.
    EXTRACT_DRAFT_TOKENS, EXTRACT_DRAFT_TOKENS_JSON: i64 = 63
);
name!(
    /// How an extraction chooses its draft length.
    EXTRACT_DRAFT_POLICY, EXTRACT_DRAFT_POLICY_JSON = "adaptive"
);
number!(
    /// The most likely tokens an extraction keeps at each position.
    EXTRACT_TOP_TOKENS, EXTRACT_TOP_TOKENS_JSON: i64 = 1024
);
number!(
    /// The characters of a document an extraction's first window reads.
    FIRST_WINDOW_CHARS, FIRST_WINDOW_CHARS_JSON: i64 = 64
);
number!(
    /// The characters of a document each later window reads.
    WINDOW_CHARS, WINDOW_CHARS_JSON: i64 = 512
);

// ---- limits

number!(
    /// The answer limit of an AI.MAP call written without one.
    DEFAULT_MAP_ANSWER_TOKENS, DEFAULT_MAP_ANSWER_TOKENS_JSON: i64 = 256
);
/// The most units `order_steps` tries every order of. Past it, the
/// search takes each next unit by rank.
pub const MAX_EXACT_UNITS: usize = 10;
/// The most complete answers DataFusion evaluates a condition on.
pub const MAX_LISTED: usize = 1024;

// ---- work

/// The field names of a work's serialized quantities, in order.
pub const WORK_FIELDS: [&str; 9] = [
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
