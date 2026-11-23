macro_rules! number {
    ($(#[$doc:meta])* $name:ident, $json:ident: $kind:ty = $value:literal) => {
        $(#[$doc])*
        pub const $name: $kind = $value;
        pub const $json: &str = stringify!($value);
    };
}

macro_rules! name {
    ($(#[$doc:meta])* $name:ident, $json:ident = $value:literal) => {
        $(#[$doc])*
        pub const $name: &str = $value;
        pub const $json: &str = concat!("\"", $value, "\"");
    };
}

pub const SPAN_SUFFIX: &str = "_span";
pub const PROBABILITIES_SUFFIX: &str = "_probabilities";
pub const NULL_DOCUMENT: &str = "__null_document";

pub const IMAGE_TOKEN_BASE: i32 = 1 << 30;

pub const LABEL_PREFIX: &str = " ";
name!(LETTERS_SCORING, LETTERS_SCORING_JSON = "letters");
pub const TREE_SCORING: &str = "trie_tree";
pub const DECODE_SCORING: &str = "trie_decode";

pub const DEFAULT_SELECTIVITY: f64 = 0.2;
number!(
    EXTRACT_ANSWER_TOKENS, EXTRACT_ANSWER_TOKENS_JSON: i64 = 16
);
number!(
    MAP_ANSWER_TOKENS, MAP_ANSWER_TOKENS_JSON: i64 = 64
);
pub const SAMPLE_ROWS: usize = 1024;

number!(
    EXTRACT_MAX_ANSWER_TOKENS, EXTRACT_MAX_ANSWER_TOKENS_JSON: i64 = 64
);
number!(
    EXTRACT_DRAFT_TOKENS, EXTRACT_DRAFT_TOKENS_JSON: i64 = 63
);
name!(EXTRACT_DRAFT_POLICY, EXTRACT_DRAFT_POLICY_JSON = "adaptive");
number!(
    EXTRACT_TOP_TOKENS, EXTRACT_TOP_TOKENS_JSON: i64 = 1024
);
number!(
    FIRST_WINDOW_CHARS, FIRST_WINDOW_CHARS_JSON: i64 = 64
);
number!(
    WINDOW_CHARS, WINDOW_CHARS_JSON: i64 = 512
);

number!(
    DEFAULT_MAP_ANSWER_TOKENS, DEFAULT_MAP_ANSWER_TOKENS_JSON: i64 = 256
);
pub const MAX_MAP_ANSWER_TOKENS: i64 = 32_768;
pub const MAX_EXACT_UNITS: usize = 10;
pub const MAX_LISTED: usize = 1024;

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

pub const MAX_LETTERS: usize = 255;
