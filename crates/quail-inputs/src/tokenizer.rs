//! The model's tokenizer: text to token ids and back, with each token's
//! bytes and character range.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Builder, LargeListArray, LargeListBuilder, StringArray};
use arrow::error::ArrowError;

/// A model's Hugging Face tokenizer, read from its `tokenizer.json`.
pub struct HfTokenizer {
    inner: tokenizers::pipeline::PipelineTokenizer,
    options: tokenizers::pipeline::EncodeOptions,
    /// Each token id's bytes, None for an id the vocabulary does not hold.
    token_bytes: Vec<Option<Vec<u8>>>,
    /// The end-of-text token the model's `tokenizer_config.json` names.
    eos_token_id: Option<u32>,
}

/// Token ids of several texts, flat, with each token's character range
/// in its text and each text's first token.
#[derive(Debug, Default, PartialEq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub offsets: Vec<(i64, i64)>,
    /// `boundaries[i]..boundaries[i + 1]` are text `i`'s tokens.
    pub boundaries: Vec<i64>,
}

fn external(error: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(error.to_string().into())
}

impl HfTokenizer {
    /// Read the tokenizer at `path`: a model directory holding
    /// `tokenizer.json`, or the file itself.
    ///
    /// # Errors
    ///
    /// The file does not read as a tokenizer.
    pub fn from_file(path: &Path) -> Result<Self, ArrowError> {
        let (file, directory) = if path.is_dir() {
            (path.join("tokenizer.json"), Some(path))
        } else {
            (path.to_path_buf(), path.parent())
        };
        let text = std::fs::read_to_string(&file)?;
        // a model's file is in the 0.x format, which 1.0 reads once converted
        let converted = tokenizers::canonicalize_str(&text).map_err(external)?;
        let mut tokenizer = Self {
            inner: tokenizers::from_json(&converted).map_err(external)?,
            options: tokenizers::pipeline::EncodeOptions::no_specials(),
            token_bytes: Vec::new(),
            eos_token_id: None,
        };
        tokenizer.token_bytes = tokenizer.read_token_bytes();
        tokenizer.eos_token_id = directory
            .and_then(eos_token)
            .and_then(|token| tokenizer.added_token_id(&token));
        Ok(tokenizer)
    }

    /// The number of token ids.
    pub fn len(&self) -> usize {
        self.token_bytes.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.token_bytes.is_empty()
    }

    /// The end-of-text token the model's config names, or None.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
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

    /// The tokens of each text, as a large list array of int32 ids; a
    /// null text has no tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text.
    pub fn encode_batch(&self, texts: &StringArray) -> Result<LargeListArray, ArrowError> {
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

    /// The tokens of each text with each token's character range: a token
    /// starts at the character holding its first byte and ends after the
    /// character holding its last.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text, or a text's tokens' bytes do not
    /// spell it, so no range could be read.
    pub fn encode_with_offsets(&self, texts: &[&str]) -> Result<Encoded, ArrowError> {
        let encodings = self
            .inner
            .encode(texts.to_vec(), &self.options)
            .wait()
            .map_err(external)?;
        let mut found = Encoded {
            boundaries: vec![0],
            ..Encoded::default()
        };
        for (text, encoding) in texts.iter().zip(&encodings) {
            let bytes = text.as_bytes();
            // the characters that start before each byte position
            let mut starts_before = Vec::with_capacity(bytes.len() + 1);
            let mut count = 0i64;
            starts_before.push(0);
            for &byte in bytes {
                count += i64::from(byte & 0xC0 != 0x80);
                starts_before.push(count);
            }
            let mut at = 0usize;
            for &id in tokenizers::pipeline::PipelineToken::cast_slice(encoding.ids()) {
                let piece = self
                    .token_bytes
                    .get(id as usize)
                    .and_then(Option::as_deref)
                    .unwrap_or_default();
                let end = at + piece.len();
                if bytes.get(at..end) != Some(piece) {
                    return Err(external(
                        "AI.EXTRACT requires token bytes that preserve the source text",
                    ));
                }
                let first = if piece.is_empty() {
                    starts_before[at]
                } else {
                    starts_before[at + 1] - 1
                };
                found.ids.push(id);
                found.offsets.push((first, starts_before[end]));
                at = end;
            }
            if at != bytes.len() {
                return Err(external(
                    "AI.EXTRACT requires token bytes that preserve the source text",
                ));
            }
            found.boundaries.push(found.ids.len() as i64);
        }
        Ok(found)
    }

    /// The id of an added token, such as `<|im_end|>`, or None.
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.inner.get_added_vocabulary().token_to_id(token)
    }

    /// Each token id's text in the tokenizer's own spelling, such as
    /// `Ġthe` in byte-level BPE, empty for an id the vocabulary does not
    /// hold.
    pub fn token_texts(&self) -> Vec<String> {
        (0..self.len() as u32)
            .map(|id| {
                self.inner
                    .decode_tokens(&[id], false)
                    .pop()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// The bytes of each token id's text, None for an id the vocabulary
    /// does not hold.
    pub fn token_bytes(&self) -> &[Option<Vec<u8>>] {
        &self.token_bytes
    }

    /// The bytes of token ids' texts, joined.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|id| {
                self.token_bytes
                    .get(*id as usize)
                    .and_then(Option::as_deref)
            })
            .flatten()
            .copied()
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

    /// An added token is its text; a model token has the pre-tokenizer's
    /// spelling undone: byte-level BPE's characters for bytes, or
    /// SentencePiece's `▁` for a space and `<0xHH>` for a byte.
    fn read_token_bytes(&self) -> Vec<Option<Vec<u8>>> {
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
}

/// The end-of-text token's text the `tokenizer_config.json` in a model's
/// directory names.
fn eos_token(directory: &Path) -> Option<String> {
    let text = std::fs::read_to_string(directory.join("tokenizer_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    match config.get("eos_token")? {
        serde_json::Value::String(token) => Some(token.clone()),
        other => other.get("content")?.as_str().map(str::to_string),
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

#[cfg(test)]
mod offset_tests {
    use super::HfTokenizer;

    fn fixture() -> HfTokenizer {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tokenizer");
        HfTokenizer::from_file(&path).expect("the fixture tokenizer")
    }

    #[test]
    fn each_token_covers_the_characters_its_bytes_spell() {
        let tokenizer = fixture();
        let texts = ["a café", "", "yes no"];
        let found = tokenizer.encode_with_offsets(&texts).expect("encoded");
        assert_eq!(found.boundaries.len(), texts.len() + 1);
        for (index, text) in texts.iter().enumerate() {
            let range = found.boundaries[index] as usize..found.boundaries[index + 1] as usize;
            let ids = &found.ids[range.clone()];
            assert_eq!(tokenizer.decode_bytes(ids), text.as_bytes());
            let chars: Vec<char> = text.chars().collect();
            let mut covered = String::new();
            let mut end = 0;
            for &(start, stop) in &found.offsets[range] {
                assert!(start >= 0 && start <= stop && stop as usize <= chars.len());
                if start >= end {
                    covered.extend(&chars[start as usize..stop as usize]);
                }
                end = end.max(stop);
            }
            assert_eq!(covered, *text);
        }
        // the two tokens of é's two bytes both cover é
        let cafe = &found.offsets[found.boundaries[0] as usize..found.boundaries[1] as usize];
        assert_eq!(cafe.last(), Some(&(5, 6)));
    }
}
