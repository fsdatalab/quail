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

    /// The id of an added token, such as `<|im_end|>`, or None.
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.inner.get_added_vocabulary().token_to_id(token)
    }

    /// Each token id's text in the tokenizer's own spelling, such as
    /// `Ġthe` in byte-level BPE, empty for an id the vocabulary does not
    /// hold.
    pub fn token_texts(&self) -> Vec<String> {
        let size = self.token_bytes().len();
        (0..size as u32)
            .map(|id| {
                self.inner
                    .decode_tokens(&[id], false)
                    .pop()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// The text of each token id list, without special tokens.
    ///
    /// # Errors
    ///
    /// An id is outside the vocabulary.
    pub fn decode_batch(&self, lists: &[Vec<u32>]) -> Result<Vec<String>, ArrowError> {
        lists
            .iter()
            .map(|ids| self.inner.decode(ids, true).map_err(external))
            .collect()
    }

    /// The bytes of each token id's text, None for an id the vocabulary
    /// does not hold. An added token is its text; a model token has the
    /// pre-tokenizer's spelling undone: byte-level BPE's characters for
    /// bytes, or SentencePiece's `▁` for a space and `<0xHH>` for a byte.
    pub fn token_bytes(&self) -> Vec<Option<Vec<u8>>> {
        // ids past this many consecutive misses are past the vocabulary
        const MISSES: u32 = 4096;
        let added = self.inner.get_added_vocabulary();
        let model = self.inner.get_model();
        let mut texts: Vec<Option<(bool, String)>> = Vec::new();
        let (mut id, mut misses) = (0u32, 0u32);
        while misses < MISSES {
            let found = match added.simple_id_to_token(id) {
                Some(text) => Some((true, text)),
                None => model.id_to_token(id).map(|text| (false, text)),
            };
            misses = if found.is_some() { 0 } else { misses + 1 };
            texts.push(found);
            id += 1;
        }
        let size = texts
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last + 1);
        texts.truncate(size);
        let byte_level = texts
            .iter()
            .flatten()
            .any(|(is_added, text)| !is_added && text.starts_with('\u{120}'));
        let bytes = byte_level_bytes();
        texts
            .into_iter()
            .map(|found| {
                found.map(|(is_added, text)| {
                    if is_added {
                        text.into_bytes()
                    } else if byte_level {
                        text.chars()
                            .flat_map(|c| match bytes.get(&c) {
                                Some(&byte) => vec![byte],
                                None => c.to_string().into_bytes(),
                            })
                            .collect()
                    } else {
                        sentencepiece_bytes(&text)
                    }
                })
            })
            .collect()
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

/// Byte-level BPE's character for each byte, reversed: printable
/// Latin-1 bytes stand for themselves, and the rest map to code points
/// from 256 in byte order.
fn byte_level_bytes() -> std::collections::HashMap<char, u8> {
    let printable =
        |byte: u8| (b'!'..=b'~').contains(&byte) || (0xA1..=0xAC).contains(&byte) || byte >= 0xAE;
    let mut next = 256u32;
    (0..=255u8)
        .map(|byte| {
            let c = if printable(byte) {
                char::from(byte)
            } else {
                next += 1;
                char::from_u32(next - 1).expect("a code point below 512")
            };
            (c, byte)
        })
        .collect()
}

/// A SentencePiece token's bytes: `<0xHH>` is the byte, and `▁` a space.
fn sentencepiece_bytes(text: &str) -> Vec<u8> {
    if let Some(hex) = text
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))
        && let Ok(byte) = u8::from_str_radix(hex, 16)
    {
        return vec![byte];
    }
    text.replace('\u{2581}', " ").into_bytes()
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

#[cfg(test)]
mod byte_tests {
    use super::{byte_level_bytes, sentencepiece_bytes};

    #[test]
    fn byte_level_characters_map_back_to_bytes() {
        let bytes = byte_level_bytes();
        assert_eq!(bytes.len(), 256);
        assert_eq!(bytes[&'\u{120}'], b' ');
        assert_eq!(bytes[&'\u{10a}'], b'\n');
        assert_eq!(bytes[&'a'], b'a');
    }

    #[test]
    fn sentencepiece_spaces_and_byte_tokens_map_to_bytes() {
        assert_eq!(sentencepiece_bytes("\u{2581}the"), b" the");
        assert_eq!(sentencepiece_bytes("<0x0A>"), b"\n");
    }
}
