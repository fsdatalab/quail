//! Quail's tokenized document columns.
//!
//! A query's documents are a table column's texts. Planning reads their
//! token counts and the prefixes they share; the executor reads their
//! tokens. This crate tokenizes a column into a token store file, which
//! Python's executor opens by path, and computes those statistics.

pub mod prefix;
pub mod store;
pub mod tokenizer;

pub use prefix::{PrefixIndex, PrefixTree, ReaderGroup, first_documents, last_tokens};
pub use store::{COUNT_COLUMN, Documents, TOKEN_COLUMN, TokenStore, TokenStoreWriter};
pub use tokenizer::{HfTokenizer, Tokenizer, token_lists};
