//! The names and numbers the planner, the DataFusion extension, and the
//! Python executor share. Each is defined here and nowhere else; Python
//! reads them from `_quail_native`.

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
pub const LETTERS_SCORING: &str = "letters";
pub const TREE_SCORING: &str = "trie_tree";
pub const DECODE_SCORING: &str = "trie_decode";

// ---- estimates

/// The planner's selectivity when the query gives none.
pub const DEFAULT_SELECTIVITY: f64 = 0.2;
/// The answer length `ExtractAssumptions` takes by default.
pub const EXTRACT_ANSWER_TOKENS: i64 = 16;
/// AI.MAP's guess at an answer's mean length; an answer limit below caps it.
pub const MAP_ANSWER_TOKENS: i64 = 64;
/// A larger table is priced from this many evenly spaced documents.
pub const SAMPLE_DOCUMENTS: usize = 1024;
/// Rows the planner samples to estimate how many pairs a join's
/// equalities keep.
pub const SAMPLE_ROWS: usize = 1024;

// ---- limits

/// The answer limit of an AI.MAP call written without one.
pub const DEFAULT_MAP_ANSWER_TOKENS: i64 = 256;
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
