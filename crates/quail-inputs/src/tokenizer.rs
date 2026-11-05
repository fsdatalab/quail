//! Tokenizers: text to token lists, as Arrow arrays.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Builder, LargeListArray, LargeListBuilder, StringArray};
use arrow::error::ArrowError;

/// Turns document texts into token lists.
///
/// A model's tokenizer returns integer token ids. A test tokenizer may
/// return words, which the token store keeps as strings.
pub trait Tokenizer: Send + Sync {
    /// The tokens of each text, as a large list array of int32 ids or of
    /// strings; a null text has no tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text.
    fn encode_batch(&self, texts: &StringArray) -> Result<LargeListArray, ArrowError>;

    /// The tokenizer's name, for the notes a query reports.
    fn name(&self) -> &str;
}

/// A Hugging Face tokenizer read from a model's `tokenizer.json`.
pub struct HfTokenizer {
    inner: tokenizers::pipeline::PipelineTokenizer,
    options: tokenizers::pipeline::EncodeOptions,
}

fn external(error: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(error.to_string().into())
}

impl HfTokenizer {
    /// Read the tokenizer in `path`, a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// The file does not read as a tokenizer.
    pub fn from_file(path: &std::path::Path) -> Result<Self, ArrowError> {
        let text = std::fs::read_to_string(path)?;
        // a model's file is in the 0.x format, which 1.0 reads once converted
        let converted = tokenizers::canonicalize_str(&text).map_err(external)?;
        Ok(Self {
            inner: tokenizers::from_json(&converted).map_err(external)?,
            options: tokenizers::pipeline::EncodeOptions::no_specials(),
        })
    }

    /// The token ids of one text, without special tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on the text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ArrowError> {
        let encodings = self
            .inner
            .encode(text, &self.options)
            .wait()
            .map_err(external)?;
        Ok(encodings
            .first()
            .map(|found| tokenizers::pipeline::PipelineToken::cast_slice(found.ids()).to_vec())
            .unwrap_or_default())
    }

    /// The text of token ids.
    ///
    /// # Errors
    ///
    /// An id is outside the vocabulary.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ArrowError> {
        self.inner.decode(ids, false).map_err(external)
    }
}

impl Tokenizer for HfTokenizer {
    fn encode_batch(&self, texts: &StringArray) -> Result<LargeListArray, ArrowError> {
        let inputs: Vec<&str> = texts.iter().map(|text| text.unwrap_or("")).collect();
        let encodings = self
            .inner
            .encode(inputs, &self.options)
            .wait()
            .map_err(external)?;
        let total = encodings.iter().map(|found| found.len()).sum();
        let mut builder =
            LargeListBuilder::with_capacity(Int32Builder::with_capacity(total), encodings.len());
        for encoding in &encodings {
            for &id in tokenizers::pipeline::PipelineToken::cast_slice(encoding.ids()) {
                builder.values().append_value(id as i32);
            }
            builder.append(true);
        }
        Ok(builder.finish())
    }

    fn name(&self) -> &str {
        "Hugging Face tokenizers"
    }
}

/// Each document's tokens as one array, for tests and callers that
/// build token lists themselves.
pub fn token_lists(lists: Vec<ArrayRef>) -> Result<LargeListArray, ArrowError> {
    let field = Arc::new(arrow::datatypes::Field::new_list_field(
        lists
            .first()
            .map_or(arrow::datatypes::DataType::Int32, |list| {
                list.data_type().clone()
            }),
        true,
    ));
    let lengths: Vec<usize> = lists.iter().map(|list| list.len()).collect();
    let offsets = arrow::buffer::OffsetBuffer::<i64>::from_lengths(lengths);
    let parts: Vec<&dyn Array> = lists.iter().map(|list| list.as_ref()).collect();
    let values = if parts.is_empty() {
        arrow::array::new_empty_array(field.data_type())
    } else {
        arrow::compute::concat(&parts)?
    };
    LargeListArray::try_new(field, offsets, values, None)
}
